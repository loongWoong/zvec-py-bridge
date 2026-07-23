"""
多因子 Re-Rank 模块 — 替代简单 RRF 融合的高级排序。

对候选 chunk 做多维打分：
  score = α × 语义相似度 (向量分)
        + β × 概念距离分 (本体图上与目标概念的距离)
        + γ × 结构相关性 (同一文件/模块/调用链)
        + δ × 新鲜度 (最近修改的优先)

提供：
  rerank(candidates, query_info) → 排序后的精选结果
  score_batch(candidates, query_info) → 逐项打分

使用场景：
  1. search_documents 返回 RRF 粗排结果后，调用 rerank 做精排
  2. 闭环 Agent 评估时，对补充检索结果打分过滤
"""

from __future__ import annotations

import math
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ontology


# ====================================================================== #
#  可配置权重
# ====================================================================== #

# 默认权重（可调）
DEFAULT_WEIGHTS = {
    "alpha": 0.35,   # 语义相似度
    "beta": 0.30,    # 概念距离
    "gamma": 0.20,   # 结构相关性
    "delta": 0.15,   # 新鲜度
}


# ====================================================================== #
#  候选结果数据模型
# ====================================================================== #

class Candidate:
    """一个待重排的检索候选。"""

    def __init__(
        self,
        doc_id: str = "",
        title: str = "",
        excerpt: str = "",
        score: float = 0.0,
        vector_similarity: float = 0.0,
        sources: list[str] | None = None,
        file_path: str = "",
        function_name: str = "",
        updated_at: str = "",
        concept_ids: list[str] | None = None,
    ):
        self.doc_id = doc_id
        self.title = title
        self.excerpt = excerpt
        self.score = score           # 原始 RRF 分
        self.vector_similarity = vector_similarity
        self.sources = sources or []
        self.file_path = file_path
        self.function_name = function_name
        self.updated_at = updated_at
        self.concept_ids = concept_ids or []

        # 计算后的多维分数
        self.semantic_score: float = 0.0
        self.concept_distance_score: float = 0.0
        self.structural_score: float = 0.0
        self.freshness_score: float = 0.0
        self.final_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "excerpt": self.excerpt[:200],
            "original_score": self.score,
            "final_score": round(self.final_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "concept_score": round(self.concept_distance_score, 4),
            "structural_score": round(self.structural_score, 4),
            "freshness_score": round(self.freshness_score, 4),
            "concept_ids": self.concept_ids,
            "function_name": self.function_name,
            "updated_at": self.updated_at,
            "sources": self.sources,
            "file_path": self.file_path,
        }


def from_search_result(r: dict) -> Candidate:
    """从 search_documents 返回结果构造 Candidate。"""
    return Candidate(
        doc_id=r.get("doc_id", ""),
        title=r.get("title", ""),
        excerpt=r.get("excerpt", ""),
        score=r.get("score", 0.0),
        vector_similarity=r.get("vector_similarity", r.get("score", 0.0)),
        sources=r.get("sources", []),
    )


# ====================================================================== #
#  打分函数
# ====================================================================== #

def _semantic_score(candidate: Candidate, query: str) -> float:
    """语义相似度分：归一化向量分数到 [0, 1]。

    优先用 vector_similarity，fallback 到原始 RRF score。
    """
    if candidate.vector_similarity > 0:
        # zvec 返回的 score 通常是内积，大于 1 时做 sigmoid 归一化
        raw = candidate.vector_similarity
        if raw > 1.0:
            return 1.0 / (1.0 + math.exp(-(raw - 2.0)))
        return min(raw, 1.0)
    # fallback: RRF 分归一化
    return min(candidate.score * 10, 1.0)  # RRF 分通常在 0.01-0.05 范围


def _concept_distance_score(
    candidate: Candidate,
    target_concept_ids: list[str],
    max_distance: int = 4,
) -> float:
    """概念距离分：候选 chunk 所属概念与目标概念在本体图上的距离。

    距离越近分越高。0 距离（同一概念）= 1.0，max_distance = 0.0。

    实现：
      若候选有 concept_ids → 计算到 target 的最短距离
      若无概念标注 → 中性分 0.3
    """
    if not target_concept_ids:
        return 0.0

    candidate_concepts = candidate.concept_ids
    if not candidate_concepts:
        # 无概念标注：中性分
        return 0.3

    # 先查直接匹配
    for cc in candidate_concepts:
        if cc in target_concept_ids:
            return 1.0

    # 查子概念（候选是否是目标的后代）
    for tc in target_concept_ids:
        try:
            descendants = ontology.get_descendants(tc, max_depth=2)
            descendant_ids = {d["id"] for d in descendants}
            for cc in candidate_concepts:
                if cc in descendant_ids:
                    return 0.7
        except Exception:
            pass

    # 查共同绑定文档
    for tc in target_concept_ids:
        try:
            target_bindings = ontology.get_bindings(tc)
            target_docs = {b.get("doc_id", "") for b in target_bindings}
            if candidate.doc_id in target_docs:
                return 0.5
        except Exception:
            pass

    return 0.1  # 远距离


def _structural_score(candidate: Candidate, reference_files: list[str] = None,
                      reference_functions: list[str] = None) -> float:
    """结构相关性分：候选是否在同一个文件/模块/调用链内。

    与参考文件/函数越接近分越高。
    """
    if not reference_files and not reference_functions:
        return 0.0

    score = 0.0

    # 文件匹配
    if reference_files and candidate.file_path:
        for ref_file in reference_files:
            if candidate.file_path == ref_file:
                score += 0.4
                break
            # 同目录
            if os.path.dirname(candidate.file_path) == os.path.dirname(ref_file):
                score += 0.2
                break

    # 函数匹配（同名函数）
    if reference_functions and candidate.function_name:
        if candidate.function_name in reference_functions:
            score += 0.3

    return min(score, 1.0)


def _freshness_score(candidate: Candidate, max_age_days: int = 90) -> float:
    """新鲜度分：最近修改的文档优先级更高。

    今天修改 = 1.0，max_age_days 以前 = 0.0。
    """
    if not candidate.updated_at:
        return 0.5  # 无时间信息，中性分

    try:
        # 尝试解析 ISO 格式时间
        updated = datetime.fromisoformat(candidate.updated_at.replace("Z", "+00:00"))
        age = (datetime.now(updated.tzinfo) - updated).days if updated.tzinfo else (datetime.now() - updated.replace(tzinfo=None)).days
        if age < 0:
            age = 0
        return max(0.0, 1.0 - age / max_age_days)
    except (ValueError, TypeError):
        return 0.5


# ====================================================================== #
#  主排序入口
# ====================================================================== #

def rerank(
    candidates: list[dict],
    query: str = "",
    target_concept_ids: list[str] | None = None,
    reference_files: list[str] | None = None,
    reference_functions: list[str] | None = None,
    weights: dict | None = None,
    topk: int = 10,
    min_score: float = 0.05,
) -> list[dict]:
    """对候选结果做多因子精排。

    Args:
        candidates: search_documents 返回的 results 列表
        query: 原始用户查询（用于语义分数归一化）
        target_concept_ids: 目标概念 ID 列表（用于概念距离分）
        reference_files: 参考文件路径（用于结构相关性分）
        reference_functions: 参考函数名（用于结构相关性分）
        weights: 权重字典 {"alpha": 0.35, ...}
        topk: 返回前 N 个
        min_score: 最低分阈值，低于此分的丢弃

    Returns:
        排序后的精选结果列表
    """
    w = weights or DEFAULT_WEIGHTS
    alpha = w.get("alpha", 0.35)
    beta = w.get("beta", 0.30)
    gamma = w.get("gamma", 0.20)
    delta = w.get("delta", 0.15)

    scored: list[Candidate] = []

    for r in candidates:
        c = from_search_result(r)
        c.concept_ids = r.get("concept_ids", [])
        c.file_path = r.get("file_path", "")
        c.function_name = r.get("function_name", "")
        c.updated_at = r.get("updated_at", "")

        # 多维打分
        c.semantic_score = _semantic_score(c, query)
        c.concept_distance_score = _concept_distance_score(c, target_concept_ids or [])
        c.structural_score = _structural_score(c, reference_files, reference_functions)
        c.freshness_score = _freshness_score(c)

        # 加权融合
        c.final_score = (
            alpha * c.semantic_score
            + beta * c.concept_distance_score
            + gamma * c.structural_score
            + delta * c.freshness_score
        )

        scored.append(c)

    # 排序 + 过滤
    scored.sort(key=lambda x: -x.final_score)
    filtered = [c for c in scored if c.final_score >= min_score]

    # 去重：同一概念的多个 chunk 只保留最高分
    seen_concepts: set[str] = set()
    deduped: list[Candidate] = []
    for c in filtered:
        key = c.doc_id  # 按文档去重
        if key not in seen_concepts:
            seen_concepts.add(key)
            deduped.append(c)

    return [c.to_dict() for c in deduped[:topk]]


# ====================================================================== #
#  批打分（用于闭环 Agent 评估）
# ====================================================================== #

def score_batch(
    candidates: list[dict],
    query: str = "",
    target_concept_ids: list[str] | None = None,
    weights: dict | None = None,
) -> list[dict]:
    """逐项打分（不排序，用于 Agent 评估每项质量）。"""
    w = weights or DEFAULT_WEIGHTS
    alpha = w.get("alpha", 0.35)
    beta = w.get("beta", 0.30)
    gamma = w.get("gamma", 0.20)
    delta = w.get("delta", 0.15)

    results: list[dict] = []
    for r in candidates:
        c = from_search_result(r)
        c.semantic_score = _semantic_score(c, query)
        c.concept_distance_score = _concept_distance_score(c, target_concept_ids or [])
        c.structural_score = _structural_score(c)
        c.freshness_score = _freshness_score(c)

        c.final_score = (
            alpha * c.semantic_score
            + beta * c.concept_distance_score
            + gamma * c.structural_score
            + delta * c.freshness_score
        )

        results.append(c.to_dict())

    return results


# ====================================================================== #
#  过滤规则
# ====================================================================== #

def apply_filters(
    candidates: list[dict],
    max_concept_distance: float = 0.2,
    min_semantic: float = 0.05,
    dedup_by_doc: bool = True,
) -> list[dict]:
    """对候选结果应用硬过滤规则。

    Args:
        candidates: 已打分的候选结果（含 concept_score, semantic_score）
        max_concept_distance: 概念距离分低于此值的丢弃
        min_semantic: 语义分低于此值的丢弃
        dedup_by_doc: 是否按文档去重

    Returns:
        过滤后的结果
    """
    filtered = []
    for c in candidates:
        concept_score = c.get("concept_score", 0.5)
        semantic_score = c.get("semantic_score", 0)

        if concept_score < max_concept_distance:
            continue
        if semantic_score < min_semantic:
            continue
        filtered.append(c)

    if dedup_by_doc:
        seen = set()
        deduped = []
        for c in filtered:
            if c["doc_id"] not in seen:
                seen.add(c["doc_id"])
                deduped.append(c)
        return deduped

    return filtered
