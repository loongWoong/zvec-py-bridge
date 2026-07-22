"""
本体展开与检索计划生成 — 对应设计文档 Step 6。

核心功能：
  expand_concepts(concept_ids, depth) → 沿本体关系展开 N 跳
  generate_search_plan(concepts, domain_context) → 生成多策略并行检索计划
  get_search_scope(concept_ids) → 获取概念相关的文件/模块范围

检索计划包含 5 种策略：
  策略1 [向量] — 概念名拼接为语义查询文本
  策略2 [全文] — 概念相关的关键词/术语精确匹配
  策略3 [图] — 沿本体路径 N 跳图遍历
  策略4 [元数据] — topic/tag 过滤
  策略5 [结构化] — 概念绑定的具体文件/函数路径

集成到 search_documents 调用之前，将 LLM 编排的检索计划翻译为实际参数。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ontology


# ====================================================================== #
#  本体展开
# ====================================================================== #

def expand_concepts(
    concept_names_or_ids: list[str],
    depth: int = 2,
) -> dict:
    """沿本体关系展开概念列表。

    对于每个输入概念：
      - 展开 is-a 子概念（后代）
      - 展开概念间关系边（depends/triggers/contains/contrasts）
      - 获取概念绑定的文档

    Args:
        concept_names_or_ids: 概念名称或 ID 列表
        depth: 展开深度（1-3，默认 2）

    Returns:
        {
            "root_concepts": [...],        # 输入概念
            "expanded_concepts": [...],    # 展开后的所有概念
            "relations": [...],            # 发现的关系边
            "bound_documents": [...],      # 绑定的文档
            "bound_files": [...],          # 绑定的具体文件
            "paths": {...},                # 每个根概念的展开路径
        }
    """
    # 解析为概念 ID
    resolved: list[dict] = []
    for name_or_id in concept_names_or_ids:
        rid = name_or_id if ":" in name_or_id else ontology._concept_rid(name_or_id)
        concept = ontology.get_concept(rid)
        if concept:
            resolved.append({"id": concept["id"], "name": concept.get("name", "")})
        else:
            # 尝试按 name 查找
            all_c = ontology.list_concepts()
            for c in all_c:
                if c["name"] == name_or_id:
                    resolved.append({"id": c["id"], "name": c["name"]})
                    break

    if not resolved:
        return {
            "root_concepts": [],
            "expanded_concepts": [],
            "relations": [],
            "bound_documents": [],
            "bound_files": [],
            "paths": {},
        }

    root_names = [c["name"] for c in resolved]
    root_ids = [c["id"] for c in resolved]

    all_expanded: set[str] = set(root_ids)
    all_relations: list[dict] = []
    all_bindings: list[dict] = []
    all_files: set[str] = set()
    paths: dict[str, dict] = {}

    for rid in root_ids:
        # 获取后代
        descendants = ontology.get_descendants(rid, depth)
        for d in descendants:
            all_expanded.add(d["id"])

        # 获取关系
        rels = ontology.get_relations(rid)
        for r in rels:
            other = r.get("target") if r.get("direction") == "out" else r.get("source")
            if other:
                all_expanded.add(other)
                all_relations.append(r)

        # 获取绑定文档
        bindings = ontology.get_bindings(rid)
        all_bindings.extend(bindings)
        for b in bindings:
            fp = b.get("file_path", "")
            if fp:
                all_files.add(fp)

        # 构建路径
        paths[rid] = {
            "name": (resolved[[c["id"] for c in resolved].index(rid)]["name"]
                     if rid in [c["id"] for c in resolved] else rid),
            "children": len(descendants),
            "relations": len(rels),
            "bindings": len(bindings),
        }

    return {
        "root_concepts": [{"id": c["id"], "name": c["name"]} for c in resolved],
        "expanded_concepts": sorted(all_expanded),
        "relations": all_relations,
        "bound_documents": [b for b in all_bindings if b.get("doc_id")],
        "bound_files": sorted(all_files),
        "paths": paths,
    }


# ====================================================================== #
#  检索计划生成
# ====================================================================== #

def generate_search_plan(
    concept_names_or_ids: list[str],
    depth: int = 2,
    include_vector: bool = True,
    include_fulltext: bool = True,
    include_graph: bool = True,
    include_metadata: bool = True,
    include_file: bool = True,
) -> dict:
    """根据概念生成多策略并行检索计划。

    返回的检索计划可直接翻译为 search_documents / 图查询 / grep 调用。

    Args:
        concept_names_or_ids: 概念名称或 ID 列表
        depth: 展开深度
        include_*: 开关各种策略

    Returns:
        {
            "plan": [
                {"strategy": "vector", "query": "...", "topk": 10, "params": {...}},
                {"strategy": "fulltext", "query": "...", "scope": [...], "params": {...}},
                {"strategy": "graph", "root_concepts": [...], "depth": 2, "params": {...}},
                {"strategy": "metadata", "topics": [...], "tags": [...], "params": {...}},
                {"strategy": "file", "files": [...], "functions": [...], "params": {...}},
            ],
            "stats": {...}
        }
    """
    expansion = expand_concepts(concept_names_or_ids, depth)
    root_concepts = expansion["root_concepts"]
    root_names = [c["name"] for c in root_concepts]
    all_bindings = expansion.get("bound_documents", [])

    plan: list[dict] = []

    # ── 策略1: 向量检索 ──
    if include_vector and root_names:
        vector_query = " ".join(root_names)
        plan.append({
            "strategy": "vector",
            "query": vector_query,
            "topk": 10,
            "params": {
                "description": f"语义检索：{', '.join(root_names)}",
            },
        })

    # ── 策略2: 全文检索 ──
    if include_fulltext and root_names:
        # 用概念名作为关键词
        keywords = "|".join(root_names)
        plan.append({
            "strategy": "fulltext",
            "keywords": keywords,
            "scope": [b.get("doc_id", "") for b in all_bindings[:10] if b.get("doc_id")],
            "params": {
                "description": f"全文检索：{keywords}",
            },
        })

    # ── 策略3: 图查询 ──
    if include_graph and root_concepts:
        plan.append({
            "strategy": "graph",
            "root_concepts": [c["id"] for c in root_concepts],
            "depth": depth,
            "params": {
                "description": f"图展开：{', '.join(root_names)} → {depth}跳",
                "action": "expand_concepts",
            },
        })

    # ── 策略4: 元数据检索 ──
    if include_metadata:
        # 从绑定文档中提取 topic/tag
        topics: set[str] = set()
        tags: set[str] = set()
        for b in all_bindings:
            doc = None
            try:
                import wiki_runtime as wr
                doc = wr.get_document(b.get("doc_id", ""))
            except Exception:
                pass
            if doc:
                topic = doc.get("topic_id", "")
                if topic:
                    topics.add(topic)
                # tag 信息在 get_document 不直接返回，跳过
        if topics:
            plan.append({
                "strategy": "metadata",
                "topics": sorted(topics),
                "params": {
                    "description": f"元数据过滤：topics={topics}",
                },
            })

    # ── 策略5: 文件检索 ──
    if include_file and expansion.get("bound_files"):
        plan.append({
            "strategy": "file",
            "files": expansion["bound_files"],
            "params": {
                "description": f"文件定位：{len(expansion['bound_files'])} 个文件",
            },
        })

    return {
        "plan": plan,
        "stats": {
            "root_concepts": len(root_concepts),
            "expanded_concepts": len(expansion.get("expanded_concepts", [])),
            "relations": len(expansion.get("relations", [])),
            "bound_documents": len(all_bindings),
            "bound_files": len(expansion.get("bound_files", [])),
            "strategies": len(plan),
        },
    }


# ====================================================================== #
#  快捷函数：获取概念相关的搜索范围
# ====================================================================== #

def get_search_scope(concept_names_or_ids: list[str]) -> dict:
    """获取概念相关的文件和文档范围（用于限定检索范围）。

    Args:
        concept_names_or_ids: 概念名称或 ID 列表

    Returns:
        {doc_ids: [...], file_paths: [...], topics: [...], tags: [...]}
    """
    expansion = expand_concepts(concept_names_or_ids, depth=1)
    bindings = expansion.get("bound_documents", [])

    doc_ids: list[str] = []
    file_paths: list[str] = []
    topics: set[str] = set()

    for b in bindings:
        did = b.get("doc_id", "")
        if did:
            doc_ids.append(did)
        fp = b.get("file_path", "")
        if fp:
            file_paths.append(fp)

    return {
        "doc_ids": sorted(set(doc_ids)),
        "file_paths": sorted(set(file_paths)),
        "concept_count": len(expansion.get("expanded_concepts", [])),
    }


# ====================================================================== #
#  工具函数：为 search_documents 生成增强查询
# ====================================================================== #

def build_enhanced_query(
    user_query: str,
    concept_names: list[str],
    depth: int = 1,
) -> str:
    """结合本体概念增强原始查询文本。

    用于替换原始 query 传给 search_documents()，
    提升向量和全文检索的召回率。

    Args:
        user_query: 原始用户问题
        concept_names: 定位到的概念名列表
        depth: 展开深度

    Returns:
        增强后的查询文本
    """
    expansion = expand_concepts(concept_names, depth)
    expanded_names = [c["name"] for c in expansion.get("root_concepts", [])]

    parts = [user_query]
    if expanded_names:
        parts.append("相关概念: " + ", ".join(expanded_names))

    # 附加关系信息
    rels = expansion.get("relations", [])
    if rels:
        rel_summary = "; ".join(
            f"{r.get('source_name', '')} → {r.get('target_name', '')}"
            for r in rels[:5]
        )
        parts.append(f"概念关系: {rel_summary}")

    return " ".join(parts)
