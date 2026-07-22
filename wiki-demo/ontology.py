"""
本体层 — 概念层级 + 关系 + 文档绑定。

提供：
  Concept CRUD — 创建/查询/更新/删除概念节点
  Relation CRUD — 概念间关系边（depends/triggers/contains/contrasts）
  Binding CRUD  — 概念→文档绑定
  YAML 导入/导出 — 人工编辑 + Git 版本管理
  图遍历辅助 — 子概念查询、父链追溯、N 跳展开

存储层：SurrealDB concept 表 + concept_related 边 + concept_binding 边

关系类型定义：
  is_a       — 层级关系（父概念→子概念），通过 parent_id 字段实现
  depends    — 依赖关系（A 依赖 B）
  triggers   — 触发关系（A 触发 B）
  contains   — 包含关系（A 包含 B）
  contrasts  — 对比关系（A 与 B 对比/互补）
  related    — 一般关联
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wiki_runtime as wr
from db import get_db


# ====================================================================== #
#  概念类型枚举
# ====================================================================== #
CONCEPT_TYPES = [
    "mechanism",     # 机制/算法（如 Attention、HNSW）
    "component",     # 组件/模块（如 Encoder、Decoder）
    "process",       # 流程/方法（如 Fine-tuning、RAG Pipeline）
    "problem",       # 问题/故障模式（如 Hallucination、403 Error）
    "configuration", # 配置/参数（如 nginx.conf、M 参数）
    "concept",       # 通用概念
    "metric",        # 度量/指标
    "infrastructure",# 基础设施
    "pattern",       # 设计模式
    "tool",          # 工具/产品
]

# 概念间关系类型
RELATION_TYPES = [
    "depends",
    "triggers",
    "contains",
    "contrasts",
    "related",
    "implements",
    "supersedes",
]


# ====================================================================== #
#  辅助
# ====================================================================== #
def _concept_rid(key: str) -> str:
    """构造 concept record ID。"""
    safe = wr._safe_key(key.lower())
    return f"concept:{safe}"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _concept_name(rid: str) -> str:
    """轻量获取概念名（避免 get_concept 的递归展开）。"""
    row = _q_one(f"SELECT name FROM {rid}")
    if isinstance(row, dict):
        return row.get("name", "")
    return ""


def _q(sql: str, params: dict | None = None) -> list:
    db = get_db()
    res = db.query(sql, params or {})
    return res if isinstance(res, list) else [res]


def _q_one(sql: str, params: dict | None = None) -> dict | None:
    rows = _q(sql, params)
    if not rows:
        return None
    row = rows[0]
    if isinstance(row, list):
        return row[0] if row else None
    return row


def _flatten(rows) -> list:
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, list):
        return first
    if isinstance(first, dict):
        return rows if isinstance(rows, list) else [rows]
    return []


# ====================================================================== #
#  Concept CRUD
# ====================================================================== #

def create_concept(
    name: str,
    concept_type: str = "concept",
    description: str = "",
    parent_id: str | None = None,
    key: str | None = None,
) -> dict:
    """创建概念节点。

    Args:
        name: 概念名称（如 "Self-Attention"）
        concept_type: 类型（mechanism/component/process/problem/configuration/concept）
        description: 描述
        parent_id: 父概念 ID（is-a 层级）
        key: 自定义 key（默认用 name 生成）

    Returns:
        {id, name, type, description, parent_id, created}
    """
    if key is None:
        key = name
    rid = _concept_rid(key)

    db = get_db()
    db.query(f"""
        UPSERT {rid} SET
            name = $name,
            type = $type,
            description = $description,
            parent_id = $parent_id,
            created = time::now(),
            updated = time::now()
    """, {
        "name": name,
        "type": concept_type,
        "description": description,
        "parent_id": parent_id,
    })

    # 如果有 parent，建 concept_related 边（is-a 关系）
    if parent_id:
        parent_rid = parent_id if ":" in parent_id else _concept_rid(parent_id)
        try:
            db.query(f"RELATE {rid}->concept_related->{parent_rid} SET relation_type = 'is_a'")
        except Exception:
            pass

    return get_concept(rid) or {"id": rid, "name": name}


def get_concept(concept_id: str) -> dict | None:
    """获取概念节点（含子概念、父概念、绑定文档）。"""
    rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    concept = _q_one(f"SELECT * FROM {rid}")
    if not concept or not isinstance(concept, dict):
        return None
    rid = wr._extract_id(concept.get("id", ""))
    concept["id"] = rid

    # 子概念
    concept["children"] = get_children(rid)

    # 父概念
    parent_id = concept.get("parent_id")
    if parent_id:
        parent = get_concept(parent_id)
        concept["parent"] = {"id": parent_id, "name": parent.get("name", "")} if parent else None

    # 绑定文档
    concept["bound_documents"] = get_bindings(rid)

    # 关系
    concept["relations"] = get_relations(rid)

    return concept


def list_concepts(concept_type: str | None = None) -> list[dict]:
    """列出所有概念（可按类型过滤）。"""
    if concept_type:
        rows = _q("SELECT * FROM concept WHERE type = $type ORDER BY name", {"type": concept_type})
    else:
        rows = _q("SELECT * FROM concept ORDER BY name")
    results = []
    for c in _flatten(rows):
        if not isinstance(c, dict):
            continue
        rid = wr._extract_id(c.get("id", ""))
        results.append({
            "id": rid,
            "name": c.get("name", ""),
            "type": c.get("type", ""),
            "description": (c.get("description", "") or "")[:120],
            "parent_id": c.get("parent_id", ""),
            "children_count": len(get_children(rid)),
            "binding_count": len(get_bindings(rid)),
        })
    return results


def update_concept(concept_id: str, **fields) -> dict | None:
    """更新概念字段。"""
    rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    allowed = {"name", "type", "description", "parent_id"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_concept(rid)

    set_parts = [f"{k} = ${k}" for k in updates]
    set_parts.append("updated = time::now()")
    params = {k: v for k, v in updates.items()}
    _q(f"UPDATE {rid} SET {', '.join(set_parts)}", params)

    # 若 parent_id 变了，重建 is-a 边
    if "parent_id" in updates and updates["parent_id"]:
        parent_rid = updates["parent_id"] if ":" in updates["parent_id"] else _concept_rid(updates["parent_id"])
        try:
            _q(f"DELETE FROM concept_related WHERE in = {rid} AND relation_type = 'is_a'")
            _q(f"RELATE {rid}->concept_related->{parent_rid} SET relation_type = 'is_a'")
        except Exception:
            pass

    return get_concept(rid)


def delete_concept(concept_id: str) -> dict:
    """删除概念及其关联边、绑定。"""
    rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    concept = get_concept(rid)
    name = concept.get("name", "") if concept else ""

    # 删除关系边
    try:
        _q(f"DELETE FROM concept_related WHERE in = {rid} OR out = {rid}")
    except Exception:
        pass

    # 删除绑定
    try:
        _q(f"DELETE FROM concept_binding WHERE in = {rid} OR out = {rid}")
    except Exception:
        pass

    # 清除子概念的 parent_id
    try:
        _q(f"UPDATE concept SET parent_id = NONE WHERE parent_id = '{rid}'")
    except Exception:
        pass

    # 删除节点
    try:
        _q(f"DELETE FROM {rid}")
    except Exception as e:
        return {"deleted": False, "id": rid, "error": str(e)}

    return {"deleted": True, "id": rid, "name": name}


# ====================================================================== #
#  Concept Relation（概念间关系边）
# ====================================================================== #

def add_relation(
    source_id: str,
    target_id: str,
    relation_type: str = "related",
) -> dict:
    """在两个概念之间建立关系边。"""
    src_rid = source_id if ":" in source_id else _concept_rid(source_id)
    tgt_rid = target_id if ":" in target_id else _concept_rid(target_id)

    if relation_type not in RELATION_TYPES:
        return {"error": f"不支持的关系类型: {relation_type}"}

    try:
        _q(f"""
            RELATE {src_rid}->concept_related->{tgt_rid}
            SET relation_type = $type
        """, {"type": relation_type})
    except Exception as e:
        return {"error": str(e)}

    return {"source": src_rid, "target": tgt_rid, "type": relation_type, "status": "created"}


def get_relations(concept_id: str) -> list[dict]:
    """获取概念的所有关系边（出边+入边）。"""
    rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    results: list[dict] = []

    # 出边（排除 is_a：层级关系由 parent_id/children 专门处理，且会形成环）
    try:
        rows = _q(
            f"SELECT out AS target, relation_type FROM {rid}->concept_related "
            f"WHERE relation_type != 'is_a'"
        )
        for r in _flatten(rows):
            if isinstance(r, dict) and r.get("target"):
                target_rid = wr._extract_id(r["target"])
                results.append({
                    "direction": "out",
                    "source": rid,
                    "target": target_rid,
                    "target_name": _concept_name(target_rid),
                    "type": r.get("relation_type", ""),
                })
    except Exception:
        pass

    # 入边（排除 is_a）
    try:
        rows = _q(
            f"SELECT in AS source, relation_type FROM {rid}<-concept_related "
            f"WHERE relation_type != 'is_a'"
        )
        for r in _flatten(rows):
            if isinstance(r, dict) and r.get("source"):
                source_rid = wr._extract_id(r["source"])
                results.append({
                    "direction": "in",
                    "source": source_rid,
                    "source_name": _concept_name(source_rid),
                    "target": rid,
                    "type": r.get("relation_type", ""),
                })
    except Exception:
        pass

    return results


def remove_relation(source_id: str, target_id: str, relation_type: str | None = None) -> dict:
    """删除概念间关系边。"""
    src_rid = source_id if ":" in source_id else _concept_rid(source_id)
    tgt_rid = target_id if ":" in target_id else _concept_rid(target_id)

    if relation_type:
        _q(f"""
            DELETE FROM concept_related
            WHERE in = {src_rid} AND out = {tgt_rid} AND relation_type = $type
        """, {"type": relation_type})
    else:
        _q(f"DELETE FROM concept_related WHERE in = {src_rid} AND out = {tgt_rid}")

    return {"source": src_rid, "target": tgt_rid, "deleted": True}


# ====================================================================== #
#  Concept Binding（概念→文档绑定）
# ====================================================================== #

def bind_concept(
    concept_id: str,
    document_id: str,
    binding_type: str = "primary",
    file_path: str = "",
    function_name: str = "",
) -> dict:
    """将概念绑定到文档。

    Args:
        concept_id: 概念 ID
        document_id: 文档 ID（SurrealDB record ID）
        binding_type: primary（核心文档）| secondary（关联文档）| inferred（推断）
        file_path: 绑定到的具体文件路径
        function_name: 绑定到的具体函数/类名
    """
    c_rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    d_rid = document_id if ":" in document_id else f"document:{document_id}"

    try:
        _q(f"""
            RELATE {c_rid}->concept_binding->{d_rid}
            SET binding_type = $type, file_path = $fp, function_name = $fn
        """, {"type": binding_type, "fp": file_path, "fn": function_name})
    except Exception as e:
        return {"error": str(e)}

    return {"concept": c_rid, "document": d_rid, "type": binding_type, "status": "bound"}


def get_bindings(concept_id: str) -> list[dict]:
    """获取概念绑定的所有文档。"""
    rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    results: list[dict] = []

    try:
        rows = _q(f"SELECT out AS doc, binding_type, file_path, function_name FROM {rid}->concept_binding")
        for r in _flatten(rows):
            if isinstance(r, dict) and r.get("doc"):
                doc_id = wr._extract_id(r["doc"])
                doc = wr.get_document(doc_id)
                results.append({
                    "doc_id": doc_id,
                    "title": doc.get("title", "") if doc else "",
                    "binding_type": r.get("binding_type", ""),
                    "file_path": r.get("file_path", ""),
                    "function_name": r.get("function_name", ""),
                })
    except Exception:
        pass

    return results


def unbind_concept(concept_id: str, document_id: str) -> dict:
    """解除概念与文档的绑定。"""
    c_rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    d_rid = document_id if ":" in document_id else f"document:{document_id}"

    try:
        _q(f"DELETE FROM concept_binding WHERE in = {c_rid} AND out = {d_rid}")
    except Exception as e:
        return {"error": str(e)}

    return {"concept": c_rid, "document": d_rid, "status": "unbound"}


# ====================================================================== #
#  层级遍历
# ====================================================================== #

def get_children(concept_id: str) -> list[dict]:
    """获取子概念列表（is-a 层级）。"""
    rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    rows = _q("SELECT * FROM concept WHERE parent_id = $pid ORDER BY name", {"pid": rid})
    return [
        {
            "id": wr._extract_id(c.get("id", "")),
            "name": c.get("name", ""),
            "type": c.get("type", ""),
            "description": (c.get("description", "") or "")[:80],
        }
        for c in _flatten(rows) if isinstance(c, dict)
    ]


def get_ancestors(concept_id: str) -> list[dict]:
    """获取从根到当前概念的祖先链。"""
    chain: list[dict] = []
    current = concept_id if ":" in concept_id else _concept_rid(concept_id)
    visited: set[str] = set()

    while current and current not in visited:
        visited.add(current)
        concept = _q_one(f"SELECT id, name, type, parent_id FROM {current}")
        if not concept or not isinstance(concept, dict):
            break
        rid = wr._extract_id(concept.get("id", ""))
        chain.append({
            "id": rid,
            "name": concept.get("name", ""),
            "type": concept.get("type", ""),
        })
        current = concept.get("parent_id", "")

    return chain


def get_descendants(concept_id: str, max_depth: int = 5) -> list[dict]:
    """获取所有后代概念（BFS）。"""
    rid = concept_id if ":" in concept_id else _concept_rid(concept_id)
    results: list[dict] = []
    visited: set[str] = {rid}
    frontier: list[str] = [rid]
    depth = 0

    while frontier and depth < max_depth:
        depth += 1
        next_frontier: list[str] = []
        for current in frontier:
            children = get_children(current)
            for child in children:
                if child["id"] not in visited:
                    visited.add(child["id"])
                    child["depth"] = depth
                    results.append(child)
                    next_frontier.append(child["id"])
        frontier = next_frontier

    return results


def expand_concept(concept_id: str, depth: int = 2) -> dict:
    """以概念为中心展开本体子图（子概念 + 关系边 + 绑定文档）。

    Args:
        concept_id: 概念 ID
        depth: 展开深度

    Returns:
        {concept, children, relations, bindings, related_concepts}
    """
    concept = get_concept(concept_id)
    if not concept:
        return {"error": f"概念 {concept_id} 不存在"}

    result = {
        "concept": {
            "id": concept["id"],
            "name": concept.get("name", ""),
            "type": concept.get("type", ""),
            "description": concept.get("description", ""),
        },
        "ancestors": get_ancestors(concept_id),
        "children": get_descendants(concept_id, depth),
        "relations": get_relations(concept_id),
        "bindings": get_bindings(concept_id),
    }

    # 沿关系边展开关联概念
    related_concepts: list[dict] = []
    visited: set[str] = {concept["id"]}

    def _collect(current_id: str, remaining_depth: int):
        if remaining_depth <= 0:
            return
        rels = get_relations(current_id)
        for rel in rels:
            other = rel.get("target") if rel.get("direction") == "out" else rel.get("source")
            other_name = rel.get("target_name") or rel.get("source_name") or ""
            if other and other not in visited:
                visited.add(other)
                related_concepts.append({
                    "id": other,
                    "name": other_name,
                    "relation": rel.get("type", ""),
                    "direction": rel.get("direction", ""),
                    "depth": depth - remaining_depth + 1,
                })
                _collect(other, remaining_depth - 1)

    _collect(concept["id"], depth)
    result["related_concepts"] = related_concepts

    return result


# ====================================================================== #
#  YAML 导入/导出
# ====================================================================== #

def export_to_yaml(file_path: str | None = None) -> str:
    """将所有概念导出为 YAML 格式。

    Args:
        file_path: 若提供则写入文件，否则返回字符串

    Returns:
        YAML 字符串
    """
    concepts_data: list[dict] = []
    all_concepts = list_concepts()

    for c in all_concepts:
        concept = get_concept(c["id"])
        if not concept:
            continue
        entry = {
            "name": concept.get("name", ""),
            "type": concept.get("type", "concept"),
            "description": concept.get("description", ""),
            "key": c["id"].split(":")[-1] if ":" in c["id"] else c["id"],
        }
        if concept.get("parent_id"):
            parent_name = ""
            if concept.get("parent"):
                parent_name = concept["parent"].get("name", "")
            entry["parent"] = parent_name

        # 关系
        rels = concept.get("relations", [])
        if rels:
            entry["relations"] = [
                {
                    "target": r.get("target_name") or r.get("source_name") or "",
                    "type": r.get("type", ""),
                    "direction": r.get("direction", ""),
                }
                for r in rels
            ]

        # 绑定
        bindings = concept.get("bound_documents", [])
        if bindings:
            entry["bindings"] = [
                {
                    "document": b.get("title") or b.get("doc_id", ""),
                    "type": b.get("binding_type", ""),
                    "file_path": b.get("file_path", ""),
                    "function_name": b.get("function_name", ""),
                }
                for b in bindings
            ]

        concepts_data.append(entry)

    yaml_str = yaml.dump(
        {"concepts": concepts_data},
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )

    if file_path:
        Path(file_path).write_text(yaml_str, encoding="utf-8")

    return yaml_str


def import_from_yaml(file_path: str, clear_existing: bool = False) -> dict:
    """从 YAML 文件导入概念。

    YAML 格式：
    ```yaml
    concepts:
      - name: Self-Attention
        type: mechanism
        description: 自注意力机制
        key: self_attention
        parent: Attention
        relations:
          - target: Multi-Head Attention
            type: contains
            direction: out
        bindings:
          - document: Transformer
            type: primary
    ```

    Args:
        file_path: YAML 文件路径
        clear_existing: 是否清除已有概念再导入

    Returns:
        {imported: int, updated: int, errors: [...]}
    """
    text = Path(file_path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    if not data or "concepts" not in data:
        return {"error": "YAML 格式错误：缺少 concepts 字段"}

    if clear_existing:
        existing = list_concepts()
        for c in existing:
            delete_concept(c["id"])

    imported = 0
    updated = 0
    errors: list[str] = []

    # 第一遍：创建所有概念节点（不建立关系）
    name_to_id: dict[str, str] = {}
    for entry in data["concepts"]:
        name = entry.get("name", "")
        if not name:
            errors.append(f"跳过无名称条目: {entry}")
            continue

        key = entry.get("key") or name
        concept_type = entry.get("type", "concept")
        description = entry.get("description", "")
        parent_name = entry.get("parent", "")

        # 检查是否已存在
        rid = _concept_rid(key)
        existing = _q_one(f"SELECT id FROM {rid}")
        if existing:
            update_concept(rid, name=name, type=concept_type, description=description)
            updated += 1
        else:
            create_concept(
                name=name,
                concept_type=concept_type,
                description=description,
                key=key,
            )
            imported += 1

        name_to_id[name] = rid

    # 第二遍：建立 parent 关系
    for entry in data["concepts"]:
        name = entry.get("name", "")
        parent_name = entry.get("parent", "")
        if parent_name and name in name_to_id:
            parent_id = name_to_id.get(parent_name)
            if parent_id:
                update_concept(name_to_id[name], parent_id=parent_id)

    # 第三遍：建立关系和绑定
    for entry in data["concepts"]:
        name = entry.get("name", "")
        if name not in name_to_id:
            continue
        cid = name_to_id[name]

        # 关系
        for rel in entry.get("relations", []):
            target_name = rel.get("target", "")
            if target_name and target_name in name_to_id:
                rel_type = rel.get("type", "related")
                direction = rel.get("direction", "out")
                if direction == "out":
                    add_relation(cid, name_to_id[target_name], rel_type)
                else:
                    add_relation(name_to_id[target_name], cid, rel_type)

        # 绑定
        for bind in entry.get("bindings", []):
            doc_title = bind.get("document", "")
            if doc_title:
                docs = wr.find_documents_by_title(doc_title)
                if docs:
                    bind_concept(
                        cid, docs[0]["id"],
                        binding_type=bind.get("type", "primary"),
                        file_path=bind.get("file_path", ""),
                        function_name=bind.get("function_name", ""),
                    )

    return {
        "imported": imported,
        "updated": updated,
        "errors": errors,
    }


# ====================================================================== #
#  完整图（供前端 D3 渲染）
# ====================================================================== #

def get_full_graph() -> dict:
    """完整本体图：所有概念节点 + is-a 层级边 + concept_related 关系边。

    Returns:
        {nodes: [{id, label, type, description, parent_id, children_count, binding_count}],
         edges: [{source, target, type}]}
    """
    concepts = list_concepts()
    nodes = [
        {
            "id": c["id"],
            "label": c["name"],
            "type": c.get("type", "concept"),
            "description": c.get("description", ""),
            "parent_id": c.get("parent_id", ""),
            "children_count": c.get("children_count", 0),
            "binding_count": c.get("binding_count", 0),
        }
        for c in concepts
    ]
    node_ids = {n["id"] for n in nodes}

    edges: list[dict] = []
    seen: set[tuple] = set()

    # is-a 层级边（从 parent_id 派生，方向 child → parent）
    for n in nodes:
        pid = n.get("parent_id", "")
        if pid and pid in node_ids:
            key = ("is_a", n["id"], pid)
            if key not in seen:
                seen.add(key)
                edges.append({"source": n["id"], "target": pid, "type": "is_a"})

    # concept_related 关系边（排除 is_a，避免与层级边重复）
    try:
        rows = _q("SELECT in AS source, out AS target, relation_type FROM concept_related")
        for r in _flatten(rows):
            if not isinstance(r, dict):
                continue
            src = wr._extract_id(r.get("source", ""))
            tgt = wr._extract_id(r.get("target", ""))
            rtype = r.get("relation_type") or "related"
            if rtype == "is_a":
                continue  # 已由 parent_id 派生
            if src in node_ids and tgt in node_ids:
                key = (rtype, src, tgt)
                if key not in seen:
                    seen.add(key)
                    edges.append({"source": src, "target": tgt, "type": rtype})
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges}


# ====================================================================== #
#  统计
# ====================================================================== #

def stats() -> dict:
    """本体统计信息。"""
    try:
        rows = _q("SELECT count() FROM concept GROUP ALL")
        r = _flatten(rows)
        concept_count = r[0].get("count", 0) if r and isinstance(r[0], dict) else 0
    except Exception:
        concept_count = 0

    try:
        rows = _q("SELECT count() FROM concept_related GROUP ALL")
        r = _flatten(rows)
        relation_count = r[0].get("count", 0) if r and isinstance(r[0], dict) else 0
    except Exception:
        relation_count = 0

    try:
        rows = _q("SELECT count() FROM concept_binding GROUP ALL")
        r = _flatten(rows)
        binding_count = r[0].get("count", 0) if r and isinstance(r[0], dict) else 0
    except Exception:
        binding_count = 0

    return {
        "concepts": concept_count,
        "relations": relation_count,
        "bindings": binding_count,
    }
