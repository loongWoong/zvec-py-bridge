"""本体定期维护调度器（Phase 4 / W4.2）。

定期扫描知识库（近期变更文档 + 查询日志盲区），借助 LLM 提议新的本体概念，
写入为 status=proposed（进入 W4.5 人工审核流），并生成本体版本快照与维护报告。
可手动触发（API）或由后台调度器（APScheduler，可选依赖）定时运行。

设计意图（AI推理引擎.md 风险对策 / Phase 4）：
  - LLM 定期扫描新文档提议本体更新（人工 review 合入）
  - 查询日志分析发现本体盲区
  - 本体版本化（回溯）
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ontology
import wiki_runtime as wr

_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ontology_maintenance_report.json"
)

# 调度器状态（模块级，进程内共享）
_SCHED = None
_SCHED_STATUS: dict = {
    "running": False,
    "interval_hours": 0,
    "last_run": None,
    "next_run": None,
    "last_summary": None,
    "note": "",
}


# ====================================================================== #
#  文档收集
# ====================================================================== #
def _collect_recent_docs(scan_days: int = 30, limit: int = 200) -> list[dict]:
    """收集近期变更的文档（title + content 样本）。"""
    try:
        docs = wr.list_documents(limit=limit)
    except Exception:
        return []
    if scan_days <= 0:
        return docs
    cutoff = time.time() - scan_days * 86400
    recent: list[dict] = []
    for d in docs:
        up = d.get("updated") or d.get("updated_at")
        ts = None
        if up:
            try:
                ts = datetime.fromisoformat(
                    str(up).replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                ts = None
        if ts is None or ts >= cutoff:
            recent.append(d)
    return recent


# ====================================================================== #
#  概念提议（LLM 优先，盲区兜底）
# ====================================================================== #
def _propose_from_llm(docs: list[dict], existing_names: set[str],
                      top_n: int = 10) -> list[dict]:
    """借助 LLM 从文档样本中提议新概念（best-effort）。"""
    try:
        import llm
    except Exception:
        return []
    try:
        return llm.propose_new_concepts(docs, existing_names, top_n=top_n)
    except Exception:
        return []


def _propose_from_blind_spots(blind_spots: list[dict], existing_names: set[str]) -> list[dict]:
    """无 LLM 时的兜底：从查询日志盲区问题中抽取候选概念名（核心名词短语）。"""
    proposed: list[dict] = []
    seen: set[str] = set()
    stop = {"如何", "怎么", "什么", "为什么", "怎样", "是否", "怎么", "the", "a",
            "an", "how", "what", "why", "does", "is", "are", "do", "?"}
    for b in blind_spots:
        q = (b.get("question") or "").strip()
        if not q:
            continue
        words = [w.strip("?.,。！!") for w in q.replace("？", " ").split()]
        cand = None
        for w in words:
            if len(w) >= 2 and w.lower() not in stop and not w.isdigit():
                cand = w
                break
        if cand and cand not in existing_names and cand not in seen:
            seen.add(cand)
            proposed.append({
                "name": cand,
                "type": "concept",
                "description": f"由查询盲区自动提议（待 review）：{q}",
            })
    return proposed


# ====================================================================== #
#  主入口：执行一次维护
# ====================================================================== #
def run_maintenance(repo_path: str | None = None, scan_days: int = 30,
                    top_n: int = 10, use_llm: bool = True) -> dict:
    """执行一次本体维护：扫描文档 + 查询盲区 → 提议新概念(proposed) → 快照。

    Returns: {scanned, proposed_new, skipped_existing, blind_spots,
              snapshot_path, report_path, timestamp}
    """
    docs = _collect_recent_docs(scan_days=scan_days)
    scanned = len(docs)

    # 现有概念名（approved + proposed 都算，避免重复提议）
    existing: set[str] = set()
    try:
        for c in ontology.list_concepts():
            if c.get("name"):
                existing.add(c["name"])
    except Exception:
        pass

    # 盲区分析
    blind: list[dict] = []
    try:
        blind = wr.query_log_analysis(limit=200, top_n=top_n).get("blind_spots", [])
    except Exception:
        blind = []

    # 提议新概念
    proposed: list[dict] = []
    if use_llm:
        proposed = _propose_from_llm(docs, existing, top_n=top_n)
    if not proposed:
        proposed = _propose_from_blind_spots(blind, existing)

    # 落库为 proposed（进入 W4.5 人工审核流）
    proposed_new: list[str] = []
    skipped = 0
    for p in proposed:
        name = (p.get("name") or "").strip()
        if not name or name in existing:
            skipped += 1
            continue
        try:
            ontology.create_concept(
                name=name,
                concept_type=p.get("type", "concept"),
                description=p.get("description", ""),
                status="proposed",
            )
            proposed_new.append(name)
            existing.add(name)
        except Exception:
            skipped += 1

    # 快照本体版本（可追溯）
    snapshot_path = ""
    try:
        snapshot_path = ontology.snapshot_concepts_versions()
    except Exception:
        pass

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scanned": scanned,
        "proposed_new": proposed_new,
        "proposed_new_count": len(proposed_new),
        "skipped_existing": skipped,
        "blind_spots": blind,
        "snapshot_path": snapshot_path,
    }

    # 持久化报告
    report_path = ""
    try:
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        report_path = _REPORT_PATH
    except Exception:
        pass

    _SCHED_STATUS["last_run"] = summary["timestamp"]
    _SCHED_STATUS["last_summary"] = {
        "scanned": scanned,
        "proposed_new_count": len(proposed_new),
        "blind_spots_count": len(blind),
        "snapshot_path": snapshot_path,
    }
    return {**summary, "report_path": report_path}


# ====================================================================== #
#  报告读取 + 后台调度器
# ====================================================================== #
def get_last_report() -> dict | None:
    """读取最近一次维护报告（供前端展示）。"""
    try:
        if os.path.exists(_REPORT_PATH):
            with open(_REPORT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def scheduler_status() -> dict:
    """返回调度器状态。"""
    return dict(_SCHED_STATUS)


def start_scheduler(interval_hours: int = 24, repo_path: str | None = None,
                    scan_days: int = 30, use_llm: bool = True) -> dict:
    """启动后台定时维护调度器（APScheduler，可选依赖）。"""
    global _SCHED
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        _SCHED_STATUS["note"] = "APScheduler 未安装，无法后台调度；请手动调用 /api/maintenance/run"
        return {"status": "unavailable", "reason": "apscheduler 未安装",
                "hint": "pip install apscheduler"}

    if _SCHED is not None:
        return {"status": "already_running", **scheduler_status()}

    _SCHED = BackgroundScheduler()

    def _job():
        try:
            run_maintenance(repo_path=repo_path, scan_days=scan_days, use_llm=use_llm)
        except Exception:
            pass

    _SCHED.add_job(_job, "interval", hours=interval_hours, id="ontology_maintenance")
    _SCHED.start()
    _SCHED_STATUS["running"] = True
    _SCHED_STATUS["interval_hours"] = interval_hours
    _SCHED_STATUS["next_run"] = datetime.now().isoformat(timespec="seconds")
    _SCHED_STATUS["note"] = ""
    return {"status": "started", **scheduler_status()}


def stop_scheduler() -> dict:
    """停止后台调度器。"""
    global _SCHED
    if _SCHED is not None:
        try:
            _SCHED.shutdown(wait=False)
        except Exception:
            pass
        _SCHED = None
    _SCHED_STATUS["running"] = False
    _SCHED_STATUS["interval_hours"] = 0
    _SCHED_STATUS["next_run"] = None
    return {"status": "stopped", **scheduler_status()}
