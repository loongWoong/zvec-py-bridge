"""Wiki Semantic Runtime — 核心运行时。

对应 design.md 的四层架构中的 "Wiki Semantic Runtime" 层，包含四个 Service：
  Document Service  — 文档对象 CRUD + Markdown 导出
  Graph Service     — 图关系操作（RELATE / 邻接遍历 / 树）
  Metadata Service  — Topic / Tag / Entity 元数据节点管理
  Search Service    — 四路融合检索（向量 + 全文 + 图 + 元数据）

LLM 永远通过本模块的 Tool 接口操作知识对象，不直接写数据库。
对应 design.md 第十一节。
"""
from __future__ import annotations

import re
import time
from datetime import datetime

import config
import zvec_client
from db import get_db

# ====================================================================== #
#  边类型集中定义（避免多处硬编码列表不一致）
#  对应 design.md 第二节（图结构）、第六节（Document Relation）
# ====================================================================== #
# 文档间语义关系边（design §6）
_DOC_RELATION_TYPES = [
    "related", "depends", "extends", "implements",
    "contradicts", "supersedes", "duplicates", "same_topic", "derived_from",
]
# 文档 → 元数据/原始资料 边
_DOC_META_TYPES = [
    "belongs_to", "has_tag", "mentions", "references", "updated_by", "archived_from",
]
# 元数据节点间边
_META_EDGE_TYPES = ["child_of", "entity_related"]
# 版本链边
_VERSION_EDGE_TYPES = ["previous_version"]
# 对话 → 文档 边（design §8 LLM Memory Graph）
_MEMORY_EDGE_TYPES = ["about"]
# 全部边类型（用于全图遍历/统计）
_ALL_EDGE_TYPES = (
    _DOC_RELATION_TYPES + _DOC_META_TYPES + _META_EDGE_TYPES
    + _VERSION_EDGE_TYPES + _MEMORY_EDGE_TYPES
)
# 全部节点表
_ALL_TABLES = ["document", "entity", "tag", "topic", "raw", "archive", "conversation", "version"]


# ====================================================================== #
#  辅助
# ====================================================================== #
def _rid(table: str, key: str) -> str:
    """构造 SurrealDB RecordID：table:key（key 会被转义）。"""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", key)
    return f"{table}:{safe}"


def _extract_id(record_id) -> str:
    """从 SurrealDB 返回的 id（可能是字符串或 RecordID 对象）提取 table:key 形式。"""
    s = str(record_id)
    # 去掉可能的命名空间前缀如 "wiki:wiki:document:xxx"
    parts = s.split(":")
    if len(parts) >= 2:
        # 取最后两段 table:key
        return f"{parts[-2]}:{parts[-1]}"
    return s


def _now_iso() -> str:
    return datetime.now().isoformat()


def _q(sql: str, params: dict | None = None) -> list:
    """执行 SurrealQL 查询，返回结果列表。"""
    db = get_db()
    res = db.query(sql, params or {})
    # SurrealDB Python SDK 返回 list（每条语句一个元素）
    if isinstance(res, list):
        return res
    return [res]


def _q_one(sql: str, params: dict | None = None) -> dict | None:
    """执行查询，返回第一条结果（dict）或 None。"""
    rows = _q(sql, params)
    if not rows:
        return None
    row = rows[0]
    if isinstance(row, list):
        return row[0] if row else None
    return row


# ====================================================================== #
#  Document Service
# ====================================================================== #
def create_document(
    title: str,
    content: str,
    summary: str | None = None,
    topic_id: str | None = None,
    author: str | None = None,
    doc_key: str | None = None,
    tags: list[str] | None = None,
    entities: list[dict] | None = None,
    relations: list[dict] | None = None,
) -> dict:
    """创建 Wiki 文档对象，可选同时建立 tag/entity/relation 关系边。

    对应 design.md 第一节："数据库是真实来源，Markdown 是导出产物"。
    返回完整文档对象（含 id）。
    """
    key = doc_key or re.sub(r"[^A-Za-z0-9_]", "_", title.lower()).strip("_")
    if not key:
        key = f"doc_{int(time.time())}"
    rid = _rid("document", key)

    db = get_db()
    # 幂等：UPSERT 对同 record ID 安全（存在则更新，不存在则创建）
    db.query(f"""
        UPSERT {rid} SET
            title = $title,
            summary = $summary,
            content = $content,
            topic_id = $topic_id,
            author = $author,
            status = 'active',
            version = 1,
            created = time::now(),
            updated = time::now()
    """, {
        "title": title, "summary": summary,
        "content": content, "topic_id": topic_id, "author": author,
    })

    # 建立关系边
    if topic_id:
        relate(rid, "belongs_to", _rid("topic", topic_id))
    if tags:
        for tag_name in tags:
            tag_key = re.sub(r"[^A-Za-z0-9_]", "_", tag_name.lower()).strip("_")
            ensure_tag(tag_key, tag_name)
            relate(rid, "has_tag", _rid("tag", tag_key))
    if entities:
        for ent in entities:
            ent_name = ent["name"]
            ent_key = re.sub(r"[^A-Za-z0-9_]", "_", ent_name.lower()).strip("_")
            ensure_entity(ent_key, ent_name, ent.get("type"))
            relate(rid, "mentions", _rid("entity", ent_key))
    if relations:
        for rel in relations:
            target_key = re.sub(r"[^A-Za-z0-9_]", "_", rel["target_title"].lower()).strip("_")
            relate(rid, rel.get("type", "related"), _rid("document", target_key))

    return get_document(rid)  # type: ignore[return-value]


def get_document(doc_id: str) -> dict | None:
    """获取单个文档（含关系邻接摘要）。"""
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    doc = _q_one(f"SELECT * FROM {rid}")
    if not doc:
        return None
    return _enrich_document(doc)


def _enrich_document(doc: dict) -> dict:
    """为文档对象附加关系邻接摘要。"""
    rid = _extract_id(doc.get("id", ""))
    doc["id"] = rid
    # 出边关系
    edges = _q(f"""
        SELECT id, in, out, type::table(id) AS edge_type
        FROM (SELECT ->belongs_to, ->has_tag, ->mentions, ->references,
                     ->related, ->depends, ->extends, ->implements,
                     ->contradicts, ->supersedes
              FROM {rid})
    """)
    # 上面复杂查询可能不稳，改用逐类型查询
    doc["out_edges"] = _collect_edges(rid, direction="out")
    doc["in_edges"] = _collect_edges(rid, direction="in")
    return doc


def _collect_edges(rid: str, direction: str) -> list[dict]:
    """收集文档的出/入边关系。"""
    edge_types = _DOC_META_TYPES + _DOC_RELATION_TYPES
    results: list[dict] = []
    arrow_out = "->" if direction == "out" else "<-"
    arrow_in = "->" if direction == "out" else "<-"
    for et in edge_types:
        try:
            if direction == "out":
                rows = _q(f"SELECT out AS target FROM {rid}->{et}")
            else:
                rows = _q(f"SELECT in AS source FROM {rid}<-{et}")
            for r in (rows[0] if rows and isinstance(rows[0], list) else rows):
                if not isinstance(r, dict):
                    continue
                target = r.get("target") or r.get("source")
                if target:
                    results.append({
                        "edge_type": et,
                        "direction": direction,
                        "node": _extract_id(target),
                    })
        except Exception:
            continue
    return results


def list_documents(topic: str | None = None, limit: int = 100) -> list[dict]:
    """列出文档（可按 topic 过滤）。"""
    if topic:
        rows = _q("SELECT * FROM document WHERE topic_id = $topic LIMIT $limit",
                  {"topic": topic, "limit": limit})
    else:
        rows = _q("SELECT * FROM document LIMIT $limit", {"limit": limit})
    docs = rows[0] if rows and isinstance(rows[0], list) else rows
    return [_enrich_document(d) for d in docs if isinstance(d, dict)]


def update_document(doc_id: str, **fields) -> dict | None:
    """更新文档字段（自动保存版本快照）。"""
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    # 先存版本快照
    save_version(rid)

    set_clauses = []
    params: dict = {}
    for k, v in fields.items():
        if k in ("title", "summary", "content", "topic_id", "author", "status"):
            set_clauses.append(f"{k} = ${k}")
            params[k] = v
    if not set_clauses:
        return get_document(rid)
    set_clauses.append("updated = time::now()")

    _q(f"UPDATE {rid} SET {', '.join(set_clauses)}", params)
    return get_document(rid)


def delete_document(doc_id: str) -> bool:
    """删除文档及其关联边。"""
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    try:
        _q(f"DELETE FROM {rid}")
        return True
    except Exception:
        return False


def export_markdown(doc_id: str) -> str | None:
    """将文档导出为 Markdown（对应 design.md："Markdown 是导出产物"）。"""
    doc = get_document(doc_id)
    if not doc:
        return None
    lines = [f"# {doc.get('title', '')}", ""]
    if doc.get("summary"):
        lines += [f"> {doc['summary']}", ""]
    if doc.get("topic_id"):
        lines += [f"**Topic**: `{doc['topic_id']}`", ""]
    if doc.get("author"):
        lines += [f"**Author**: {doc['author']}", ""]
    lines += [doc.get("content", ""), ""]
    # 关系
    out_edges = doc.get("out_edges", [])
    if out_edges:
        lines += ["## Relations", ""]
        for e in out_edges:
            lines.append(f"- {e['edge_type']} → `{e['node']}`")
        lines.append("")
    return "\n".join(lines)


# ====================================================================== #
#  Graph Service
# ====================================================================== #
def relate(src: str, edge_type: str, dst: str, props: dict | None = None) -> None:
    """创建图关系边。对应 design.md 第二节。"""
    props = props or {}
    prop_str = ", ".join(f"{k} = ${k}" for k in props) if props else ""
    set_clause = f"SET {prop_str}" if prop_str else ""
    sql = f"RELATE {src}->{edge_type}->{dst} {set_clause}".strip()
    try:
        _q(sql, props)
    except Exception:
        pass


def neighbors(
    node_id: str,
    direction: str = "both",
    edge_type: str | None = None,
    depth: int = 1,
) -> list[dict]:
    """图邻接遍历。direction: out/in/both。depth>1 时做 BFS 多跳。

    depth=1 返回直接邻居；depth>1 返回 depth 跳内所有可达邻居（含中间节点），
    每条结果额外带 hop 字段表示距离。
    """
    rid = _normalize_rid(node_id)
    if depth <= 1:
        return _single_hop_neighbors(rid, direction, edge_type)

    # BFS 多跳
    visited: set[str] = {rid}
    results: list[dict] = []
    frontier: list[str] = [rid]
    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for current in frontier:
            nbrs = _single_hop_neighbors(current, direction, edge_type)
            for nb in nbrs:
                nb_node = nb["node"]
                nb["hop"] = hop
                results.append(nb)
                if nb_node not in visited:
                    visited.add(nb_node)
                    next_frontier.append(nb_node)
        frontier = next_frontier
        if not frontier:
            break
    return results


def _normalize_rid(node_id: str) -> str:
    """将 node_id 规范化为 record id。"""
    if ":" in node_id:
        return node_id
    return _rid("document", node_id)


def _single_hop_neighbors(rid: str, direction: str, edge_type: str | None) -> list[dict]:
    """单跳邻接查询（内部使用）。"""
    results: list[dict] = []
    edge_types = [edge_type] if edge_type else _ALL_EDGE_TYPES
    for et in edge_types:
        if direction in ("out", "both"):
            try:
                rows = _q(f"SELECT out AS node FROM {rid}->{et}")
                for r in _flatten(rows):
                    if isinstance(r, dict) and r.get("node"):
                        results.append({"edge_type": et, "direction": "out",
                                        "node": _extract_id(r["node"])})
            except Exception:
                pass
        if direction in ("in", "both"):
            try:
                rows = _q(f"SELECT in AS node FROM {rid}<-{et}")
                for r in _flatten(rows):
                    if isinstance(r, dict) and r.get("node"):
                        results.append({"edge_type": et, "direction": "in",
                                        "node": _extract_id(r["node"])})
            except Exception:
                pass
    return results


def _flatten(rows) -> list:
    """将查询结果展平为 list[dict]。"""
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, list):
        return first
    if isinstance(first, dict):
        return rows if isinstance(rows, list) else [rows]
    return []


def related_articles(doc_id: str) -> list[dict]:
    """获取与文档相关的文章（related/extends/depends/implements/supersedes 等边）。"""
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    rel_types = ["related", "extends", "depends", "implements", "supersedes",
                 "duplicates", "same_topic", "derived_from"]
    results: list[dict] = []
    for et in rel_types:
        for direction_arrow in ("->", "<-"):
            try:
                if direction_arrow == "->":
                    rows = _q(f"SELECT out AS node FROM {rid}->{et}")
                else:
                    rows = _q(f"SELECT in AS node FROM {rid}<-{et}")
                for r in _flatten(rows):
                    if isinstance(r, dict) and r.get("node"):
                        node_id = _extract_id(r["node"])
                        doc = get_document(node_id)
                        if doc:
                            results.append({
                                "edge_type": et,
                                "direction": "out" if direction_arrow == "->" else "in",
                                "doc_id": node_id,
                                "title": doc.get("title", ""),
                                "summary": doc.get("summary", ""),
                            })
            except Exception:
                pass
    # 去重
    seen = set()
    unique = []
    for r in results:
        if r["doc_id"] not in seen:
            seen.add(r["doc_id"])
            unique.append(r)
    return unique


def topic_tree() -> list[dict]:
    """列出所有 topic。"""
    rows = _q("SELECT * FROM topic")
    topics = _flatten(rows)
    return [{"id": _extract_id(t.get("id", "")),
             "name": t.get("name", ""),
             "description": t.get("description", "")}
            for t in topics if isinstance(t, dict)]


def tag_tree() -> list[dict]:
    """列出所有 tag（含 parent 关系）。"""
    rows = _q("SELECT * FROM tag")
    tags = _flatten(rows)
    result = []
    for t in tags:
        if not isinstance(t, dict):
            continue
        tid = _extract_id(t.get("id", ""))
        # 查 parent
        parent = None
        try:
            p_rows = _q(f"SELECT in AS parent FROM {tid}<-child_of")
            for p in _flatten(p_rows):
                if isinstance(p, dict) and p.get("parent"):
                    parent = _extract_id(p["parent"])
                    break
        except Exception:
            pass
        result.append({"id": tid, "name": t.get("name", ""), "parent": parent})
    return result


def entity_lookup(name: str) -> dict | None:
    """按名称查找实体。"""
    rows = _q("SELECT * FROM entity WHERE name = $name", {"name": name})
    ents = _flatten(rows)
    if not ents:
        return None
    e = ents[0]
    eid = _extract_id(e.get("id", ""))
    # 反查哪些文档 mention 了这个实体
    docs: list[dict] = []
    try:
        d_rows = _q(f"SELECT in AS doc FROM {eid}<-mentions")
        for d in _flatten(d_rows):
            if isinstance(d, dict) and d.get("doc"):
                doc_id = _extract_id(d["doc"])
                doc = get_document(doc_id)
                if doc:
                    docs.append({"doc_id": doc_id, "title": doc.get("title", "")})
    except Exception:
        pass
    return {"id": eid, "name": e.get("name", ""), "type": e.get("type"),
            "mentioned_by": docs}


def graph_subtree(node_id: str, depth: int = 2) -> dict:
    """以某节点为根，获取局部子图（节点+边）。边去重。"""
    rid = _normalize_rid(node_id)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[frozenset] = set()

    def _visit(current_rid: str, current_depth: int):
        if current_rid in nodes:
            return
        # 获取节点信息
        node = _q_one(f"SELECT * FROM {current_rid}")
        if node:
            nodes[current_rid] = {
                "id": current_rid,
                "label": node.get("title") or node.get("name") or current_rid,
                "type": current_rid.split(":")[0] if ":" in current_rid else "unknown",
            }
        else:
            nodes[current_rid] = {"id": current_rid, "label": current_rid, "type": "unknown"}

        if current_depth < 0:
            return

        nbrs = _single_hop_neighbors(current_rid, "both", None)
        for nb in nbrs:
            nb_node = nb["node"]
            src = current_rid if nb["direction"] == "out" else nb_node
            dst = nb_node if nb["direction"] == "out" else current_rid
            # 边去重：用 frozenset({src, dst, type}) 标识
            edge_key = frozenset({src, dst, nb["edge_type"]})
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({"from": src, "to": dst, "type": nb["edge_type"]})
            if nb_node not in nodes:
                _visit(nb_node, current_depth - 1)

    _visit(rid, depth)
    return {"nodes": list(nodes.values()), "edges": edges}


# ====================================================================== #
#  图查询算法（BFS/DFS/中心性/路径/共现/血缘）
#  对应 design.md 图结构设计，提供多样的知识图谱查询能力
# ====================================================================== #


def _get_node_label(rid: str) -> dict:
    """获取节点的基本信息（label + type）。"""
    node = _q_one(f"SELECT * FROM {rid}")
    if node:
        return {
            "id": rid,
            "label": node.get("title") or node.get("name") or rid,
            "type": rid.split(":")[0] if ":" in rid else "unknown",
        }
    return {"id": rid, "label": rid, "type": rid.split(":")[0] if ":" in rid else "unknown"}


def shortest_path(src_id: str, dst_id: str, max_depth: int = 6) -> dict:
    """BFS 最短路径：两节点间的最短关系路径。

    跨 document/entity/tag 节点遍历，返回路径节点序列 + 边序列。
    """
    src = _normalize_rid(src_id)
    dst = _normalize_rid(dst_id)
    if src == dst:
        return {"path": [src], "edges": [], "length": 0}

    # BFS
    from collections import deque
    queue: deque = deque([(src, [src], [])])
    visited: set[str] = {src}

    while queue:
        current, path, edge_path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        nbrs = _single_hop_neighbors(current, "both", None)
        for nb in nbrs:
            nb_node = nb["node"]
            if nb_node == dst:
                full_path = path + [nb_node]
                full_edges = edge_path + [{
                    "from": current if nb["direction"] == "out" else nb_node,
                    "to": nb_node if nb["direction"] == "out" else current,
                    "type": nb["edge_type"],
                }]
                return {"path": full_path, "edges": full_edges, "length": len(full_path) - 1}
            if nb_node not in visited:
                visited.add(nb_node)
                queue.append((
                    nb_node,
                    path + [nb_node],
                    edge_path + [{
                        "from": current if nb["direction"] == "out" else nb_node,
                        "to": nb_node if nb["direction"] == "out" else current,
                        "type": nb["edge_type"],
                    }],
                ))
    return {"path": [], "edges": [], "length": -1, "reason": "未找到路径"}


def common_neighbors(node_a: str, node_b: str) -> dict:
    """共同邻居：两节点的共同邻居，发现隐含关联。"""
    rid_a = _normalize_rid(node_a)
    rid_b = _normalize_rid(node_b)
    nbrs_a = _single_hop_neighbors(rid_a, "both", None)
    nbrs_b = _single_hop_neighbors(rid_b, "both", None)
    set_a = {nb["node"] for nb in nbrs_a}
    set_b = {nb["node"] for nb in nbrs_b}
    common = set_a & set_b
    # 排除彼此
    common.discard(rid_a)
    common.discard(rid_b)

    result = []
    for node_id in common:
        info = _get_node_label(node_id)
        # 找出 a 和 b 分别通过什么边连到这个共同邻居
        a_edges = [nb for nb in nbrs_a if nb["node"] == node_id]
        b_edges = [nb for nb in nbrs_b if nb["node"] == node_id]
        result.append({
            "node": node_id,
            "label": info["label"],
            "type": info["type"],
            "a_edge": a_edges[0]["edge_type"] if a_edges else None,
            "b_edge": b_edges[0]["edge_type"] if b_edges else None,
        })
    return {"node_a": rid_a, "node_b": rid_b, "common": result, "count": len(result)}


def node_degree(node_id: str) -> dict:
    """度中心性：节点的入度/出度/总度。"""
    rid = _normalize_rid(node_id)
    out_nbrs = _single_hop_neighbors(rid, "out", None)
    in_nbrs = _single_hop_neighbors(rid, "in", None)
    return {
        "node": rid,
        "out_degree": len(out_nbrs),
        "in_degree": len(in_nbrs),
        "total_degree": len(out_nbrs) + len(in_nbrs),
        "out_neighbors": [{"node": nb["node"], "edge": nb["edge_type"]} for nb in out_nbrs],
        "in_neighbors": [{"node": nb["node"], "edge": nb["edge_type"]} for nb in in_nbrs],
    }


def top_central_nodes(limit: int = 10) -> list[dict]:
    """度中心性排序：知识库中度最高的 hub 节点。"""
    degrees: list[dict] = []
    for table in _ALL_TABLES:
        rows = _q(f"SELECT id FROM {table}")
        for r in _flatten(rows):
            if not isinstance(r, dict):
                continue
            rid = _extract_id(r.get("id", ""))
            deg = node_degree(rid)
            degrees.append({
                "node": rid,
                "label": _get_node_label(rid)["label"],
                "type": rid.split(":")[0] if ":" in rid else "unknown",
                "degree": deg["total_degree"],
                "in_degree": deg["in_degree"],
                "out_degree": deg["out_degree"],
            })
    degrees.sort(key=lambda x: -x["degree"])
    return degrees[:limit]


def graph_stats() -> dict:
    """全局图统计：节点数/边数/各边类型计数/平均度/密度。"""
    node_counts: dict[str, int] = {}
    total_nodes = 0
    for table in _ALL_TABLES:
        try:
            rows = _q(f"SELECT count() FROM {table} GROUP ALL")
            r = _flatten(rows)
            cnt = r[0].get("count", 0) if r and isinstance(r[0], dict) else 0
        except Exception:
            cnt = 0
        node_counts[table] = cnt
        total_nodes += cnt

    edge_types = _ALL_EDGE_TYPES
    edge_counts: dict[str, int] = {}
    total_edges = 0
    for et in edge_types:
        try:
            rows = _q(f"SELECT count() FROM {et} GROUP ALL")
            r = _flatten(rows)
            cnt = r[0].get("count", 0) if r and isinstance(r[0], dict) else 0
        except Exception:
            cnt = 0
        edge_counts[et] = cnt
        total_edges += cnt

    avg_degree = (2 * total_edges / total_nodes) if total_nodes > 0 else 0
    # 密度 = 2E / (N*(N-1))，有向图用 E/(N*(N-1))，这里用无向近似
    max_edges = total_nodes * (total_nodes - 1) if total_nodes > 1 else 1
    density = (2 * total_edges / max_edges) if max_edges > 0 else 0

    return {
        "nodes": node_counts,
        "total_nodes": total_nodes,
        "edges": edge_counts,
        "total_edges": total_edges,
        "avg_degree": round(avg_degree, 2),
        "density": round(density, 4),
    }


def entity_co_occurrence(entity_name: str) -> dict:
    """共现分析：与某实体共同出现在文档中的其他实体。

    通过 mentions 反查：找到 mention 该实体的文档，
    再找这些文档 mention 的其他实体。
    """
    # 找到实体
    rows = _q("SELECT * FROM entity WHERE name = $name", {"name": entity_name})
    ents = _flatten(rows)
    if not ents:
        return {"entity": entity_name, "co_occurrences": [], "count": 0}
    eid = _extract_id(ents[0].get("id", ""))

    # 找 mention 该实体的文档
    doc_ids: list[str] = []
    try:
        d_rows = _q(f"SELECT in AS doc FROM {eid}<-mentions")
        for d in _flatten(d_rows):
            if isinstance(d, dict) and d.get("doc"):
                doc_ids.append(_extract_id(d["doc"]))
    except Exception:
        pass

    # 对每个文档，找其 mention 的其他实体
    co_occur: dict[str, int] = {}
    for did in doc_ids:
        try:
            e_rows = _q(f"SELECT out AS ent FROM {did}->mentions")
            for e in _flatten(e_rows):
                if isinstance(e, dict) and e.get("ent"):
                    other_eid = _extract_id(e["ent"])
                    if other_eid != eid:
                        co_occur[other_eid] = co_occur.get(other_eid, 0) + 1
        except Exception:
            pass

    result = []
    for other_eid, count in sorted(co_occur.items(), key=lambda x: -x[1]):
        info = _get_node_label(other_eid)
        result.append({
            "entity": other_eid,
            "name": info["label"],
            "co_occurrence_count": count,
        })
    return {"entity": eid, "entity_name": entity_name, "co_occurrences": result, "count": len(result)}


def knowledge_lineage(doc_id: str, max_depth: int = 3) -> dict:
    """知识血缘：文档的上下游知识链。

    递归遍历 extends/depends/related/supersedes 边，
    对应 design.md "展示Embedding上下游知识" = Graph Traversal。
    """
    rid = _normalize_rid(doc_id)
    lineage_edges = ["extends", "depends", "related", "implements", "supersedes"]
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[frozenset] = set()

    def _trace(current_rid: str, current_depth: int, direction: str):
        if current_depth <= 0:
            return
        if current_rid not in nodes:
            nodes[current_rid] = _get_node_label(current_rid)
        for et in lineage_edges:
            nbrs = _single_hop_neighbors(current_rid, direction, et)
            for nb in nbrs:
                nb_node = nb["node"]
                src = current_rid if nb["direction"] == "out" else nb_node
                dst = nb_node if nb["direction"] == "out" else current_rid
                edge_key = frozenset({src, dst, et})
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({"from": src, "to": dst, "type": et})
                if nb_node not in nodes:
                    _trace(nb_node, current_depth - 1, direction)

    _trace(rid, max_depth, "out")   # 下游
    _trace(rid, max_depth, "in")    # 上游
    return {
        "root": rid,
        "nodes": list(nodes.values()),
        "edges": edges,
        "upstream_count": sum(1 for e in edges if e["to"] == rid),
        "downstream_count": sum(1 for e in edges if e["from"] == rid),
    }


def multi_hop_neighbors(node_id: str, depth: int = 3) -> dict:
    """多跳邻接：BFS N 跳，返回按层级分组的节点。"""
    rid = _normalize_rid(node_id)
    visited: set[str] = {rid}
    levels: list[list[dict]] = []
    frontier: list[str] = [rid]

    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        level_nodes: list[dict] = []
        for current in frontier:
            nbrs = _single_hop_neighbors(current, "both", None)
            for nb in nbrs:
                nb_node = nb["node"]
                if nb_node not in visited:
                    visited.add(nb_node)
                    next_frontier.append(nb_node)
                    info = _get_node_label(nb_node)
                    level_nodes.append({
                        "id": nb_node,
                        "label": info["label"],
                        "type": info["type"],
                        "via": current,
                        "edge": nb["edge_type"],
                        "hop": hop,
                    })
        levels.append({"hop": hop, "nodes": level_nodes})
        frontier = next_frontier
        if not frontier:
            break

    return {"root": rid, "levels": levels, "total_reachable": sum(len(l["nodes"]) for l in levels)}


def graph_full() -> dict:
    """全图数据：所有节点 + 所有边，供前端 D3 渲染。

    节点按类型分组（document/entity/tag/topic），每类含 id/label/type。
    边按类型查询，含 from/to/type。
    """
    nodes: list[dict] = []
    for table in _ALL_TABLES:
        rows = _q(f"SELECT * FROM {table}")
        for r in _flatten(rows):
            if not isinstance(r, dict):
                continue
            rid = _extract_id(r.get("id", ""))
            label = r.get("title") or r.get("name") or rid
            nodes.append({
                "id": rid,
                "label": label,
                "type": table,
                "summary": r.get("summary") or r.get("description") or "",
            })

    edge_types = _ALL_EDGE_TYPES
    edges: list[dict] = []
    for et in edge_types:
        try:
            rows = _q(f"SELECT in, out FROM {et}")
            for r in _flatten(rows):
                if isinstance(r, dict) and r.get("in") and r.get("out"):
                    edges.append({
                        "source": _extract_id(r["in"]),
                        "target": _extract_id(r["out"]),
                        "type": et,
                    })
        except Exception:
            pass

    return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}



def ensure_topic(key: str, name: str, description: str = "") -> str:
    """确保 topic 节点存在（幂等）。返回 record id。"""
    rid = _rid("topic", key)
    _q(f"UPSERT {rid} SET name = $name, description = $desc",
        {"name": name, "desc": description})
    return rid


def ensure_tag(key: str, name: str, parent_key: str | None = None) -> str:
    """确保 tag 节点存在（幂等）。可选建立 child_of 层级。"""
    rid = _rid("tag", key)
    _q(f"UPSERT {rid} SET name = $name", {"name": name})
    if parent_key:
        parent_rid = _rid("tag", parent_key)
        relate(rid, "child_of", parent_rid)
    return rid


def ensure_entity(key: str, name: str, ent_type: str | None = None) -> str:
    """确保 entity 节点存在（幂等）。"""
    rid = _rid("entity", key)
    _q(f"UPSERT {rid} SET name = $name, type = $etype",
        {"name": name, "etype": ent_type})
    return rid


def list_topics() -> list[dict]:
    return topic_tree()


def list_tags() -> list[dict]:
    return tag_tree()


def list_entities() -> list[dict]:
    rows = _q("SELECT * FROM entity")
    ents = _flatten(rows)
    return [{"id": _extract_id(e.get("id", "")),
             "name": e.get("name", ""),
             "type": e.get("type")}
            for e in ents if isinstance(e, dict)]


def docs_by_topic(topic_key: str) -> list[dict]:
    rid = _rid("topic", topic_key)
    rows = _q(f"SELECT in AS doc FROM {rid}<-belongs_to")
    results = []
    for r in _flatten(rows):
        if isinstance(r, dict) and r.get("doc"):
            doc = get_document(_extract_id(r["doc"]))
            if doc:
                results.append({"doc_id": doc["id"], "title": doc.get("title", "")})
    return results


def docs_by_tag(tag_key: str) -> list[dict]:
    rid = _rid("tag", tag_key)
    rows = _q(f"SELECT in AS doc FROM {rid}<-has_tag")
    results = []
    for r in _flatten(rows):
        if isinstance(r, dict) and r.get("doc"):
            doc = get_document(_extract_id(r["doc"]))
            if doc:
                results.append({"doc_id": doc["id"], "title": doc.get("title", "")})
    return results


# ====================================================================== #
#  Version Service（对应 design.md 第七节）
# ====================================================================== #
def save_version(doc_id: str) -> dict | None:
    """保存文档当前版本快照，并建立 previous_version 边形成版本链。

    对应 design.md 第七节：Version Graph。每次更新时保存快照，
    若存在上一版本则通过 previous_version 边链接，形成 v1→v2→v3 链条。
    """
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    doc = get_document(rid)
    if not doc:
        return None
    ver_num = doc.get("version", 1)
    ver_key = f"{rid.split(':')[-1]}_v{ver_num}"
    ver_rid = _rid("version", ver_key)
    _q(f"""
        UPSERT {ver_rid} SET
            doc_id = $doc_id,
            title = $title,
            content = $content,
            summary = $summary,
            version = $ver,
            snapshot = time::now()
    """, {
        "doc_id": rid, "title": doc.get("title", ""),
        "content": doc.get("content", ""), "summary": doc.get("summary"),
        "ver": ver_num,
    })
    # 建立 previous_version 边：当前版本 → 上一版本（形成版本链）
    if ver_num > 1:
        prev_ver_key = f"{rid.split(':')[-1]}_v{ver_num - 1}"
        prev_ver_rid = _rid("version", prev_ver_key)
        relate(ver_rid, "previous_version", prev_ver_rid)
    # 递增文档版本号
    _q(f"UPDATE {rid} SET version = $next", {"next": ver_num + 1})
    return {"version_id": ver_rid, "version": ver_num}


def list_versions(doc_id: str) -> list[dict]:
    """列出文档的所有历史版本。"""
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    key = rid.split(":")[-1]
    rows = _q("SELECT * FROM version WHERE doc_id = $rid ORDER BY version DESC",
              {"rid": rid})
    vers = _flatten(rows)
    return [{"id": _extract_id(v.get("id", "")),
             "version": v.get("version", 0),
             "title": v.get("title", ""),
             "snapshot": str(v.get("snapshot", ""))}
            for v in vers if isinstance(v, dict)]


def version_chain(doc_id: str) -> list[dict]:
    """按 previous_version 边遍历，返回完整版本链（从最新到最旧）。

    对应 design.md 第七节：Version Graph。与 list_versions() 不同，
    本方法通过 previous_version 边遍历，反映真正的版本演进链。
    """
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    doc = get_document(rid)
    if not doc:
        return []
    # 当前版本号 - 1 = 最新快照的 version 字段
    current_ver = doc.get("version", 1)
    chain: list[dict] = []
    # 最新快照
    latest_ver_key = f"{rid.split(':')[-1]}_v{current_ver - 1}" if current_ver > 1 else None
    if not latest_ver_key:
        # 只有一个版本，无历史链
        return [{"version": 1, "title": doc.get("title", ""),
                 "snapshot": str(doc.get("updated", "")), "current": True}]
    # 从最新快照开始，沿 previous_version 边回溯
    ver_rid = _rid("version", latest_ver_key)
    visited: set[str] = set()
    while ver_rid and ver_rid not in visited:
        visited.add(ver_rid)
        v = _q_one(f"SELECT * FROM {ver_rid}")
        if not v or not isinstance(v, dict):
            break
        chain.append({
            "id": _extract_id(v.get("id", "")),
            "version": v.get("version", 0),
            "title": v.get("title", ""),
            "snapshot": str(v.get("snapshot", "")),
        })
        # 沿 previous_version 边回溯
        try:
            rows = _q(f"SELECT out AS prev FROM {ver_rid}->previous_version")
            for r in _flatten(rows):
                if isinstance(r, dict) and r.get("prev"):
                    ver_rid = _extract_id(r["prev"])
                    break
            else:
                break
        except Exception:
            break
    # 标记当前文档版本
    if chain:
        chain[0]["current"] = False
    return chain


# ====================================================================== #
#  Search Service — 四路融合检索（对应 design.md 第九节）
# ====================================================================== #
def search_documents(query: str, topk: int = 5) -> dict:
    """四路融合检索：向量 + 全文 + 图 + 元数据。

    返回 {
        "query": str,
        "results": [{doc_id, title, score, sources: [str], excerpt}, ...],
        "routes": {"vector": N, "fts": N, "graph": N, "meta": N}
    }
    """
    routes_count = {"vector": 0, "fts": 0, "graph": 0, "meta": 0}
    # 每路返回 (doc_id, rank_score) 列表；rank_score = 1/(60+rank)
    ranked: dict[str, dict] = {}  # doc_id -> {score, sources:set, title, excerpt}

    def _add(doc_id: str, source: str, rank: int, title: str = "", excerpt: str = ""):
        rrf = 1.0 / (60 + rank)
        if doc_id not in ranked:
            ranked[doc_id] = {"score": 0.0, "sources": set(), "title": title, "excerpt": excerpt}
        ranked[doc_id]["score"] += rrf
        ranked[doc_id]["sources"].add(source)
        if title and not ranked[doc_id]["title"]:
            ranked[doc_id]["title"] = title
        if excerpt and not ranked[doc_id]["excerpt"]:
            ranked[doc_id]["excerpt"] = excerpt

    # ── ① 向量路（zvec）──
    try:
        chunks = zvec_client.search(query, topk=topk)
        routes_count["vector"] = len(chunks)
        seen_docs: set[str] = set()
        rank = 0
        for ch in chunks:
            did = ch.get("document_id", "")
            if not did or did in seen_docs:
                continue
            seen_docs.add(did)
            _add(did, "vector", rank, ch.get("title", ""), ch.get("content", "")[:120])
            rank += 1
    except Exception as e:
        print(f"  ⚠ 向量检索失败: {e}")

    # ── ② 全文检索路（SurrealDB，降级为 CONTAINS 子串匹配）──
    try:
        fts_docs = _fts_search(query, topk)
        routes_count["fts"] = len(fts_docs)
        for rank, d in enumerate(fts_docs):
            _add(d["doc_id"], "fts", rank, d.get("title", ""), d.get("excerpt", ""))
    except Exception as e:
        print(f"  ⚠ 全文检索失败: {e}")

    # ── ③ 图遍历路（query 命中实体 → mentions 反查文档）──
    try:
        graph_docs = _graph_search(query, topk)
        routes_count["graph"] = len(graph_docs)
        for rank, d in enumerate(graph_docs):
            _add(d["doc_id"], "graph", rank, d.get("title", ""), d.get("excerpt", ""))
    except Exception as e:
        print(f"  ⚠ 图检索失败: {e}")

    # ── ④ 元数据路（解析 author=/topic= 等结构化条件）──
    try:
        meta_docs = _metadata_search(query, topk)
        routes_count["meta"] = len(meta_docs)
        for rank, d in enumerate(meta_docs):
            _add(d["doc_id"], "meta", rank, d.get("title", ""), d.get("excerpt", ""))
    except Exception as e:
        print(f"  ⚠ 元数据检索失败: {e}")

    # 合并排序
    results = []
    for doc_id, info in sorted(ranked.items(), key=lambda x: -x[1]["score"]):
        results.append({
            "doc_id": doc_id,
            "title": info["title"],
            "score": round(info["score"], 4),
            "sources": sorted(info["sources"]),
            "excerpt": info["excerpt"],
        })

    return {"query": query, "results": results[:topk], "routes": routes_count}


def _fts_search(query: str, topk: int) -> list[dict]:
    """全文检索：优先使用 BM25 SEARCH 索引，降级为 CONTAINS 子串匹配。

    对应 design.md 第九节 FullText 路。db.py 定义了 doc_search BM25 索引，
    若 SDK/版本不支持 SEARCH 语法则自动降级到子串匹配，保证向后兼容。
    """
    keywords = [w.strip() for w in re.split(r"[\s,，。、]+", query) if len(w.strip()) >= 2]
    if not keywords:
        return []

    # ── 优先：BM25 全文检索索引 ──
    try:
        # SurrealQL SEARCH 语法：SEARCH '<query>' IN doc_search ON document
        safe_query = query.replace("'", " ")
        sql = f"SELECT id, title, content FROM SEARCH '{safe_query}' IN doc_search ON document LIMIT {topk}"
        rows = _q(sql)
        docs = _flatten(rows)
        results = []
        for d in docs:
            if not isinstance(d, dict):
                continue
            results.append({
                "doc_id": _extract_id(d.get("id", "")),
                "title": d.get("title", ""),
                "excerpt": (d.get("content", "") or "")[:120],
            })
        if results:
            return results
    except Exception:
        pass  # 降级到 CONTAINS

    # ── 降级：CONTAINS 子串匹配（大小写不敏感）──
    where_parts = " OR ".join(
        f"string::lowercase(title) CONTAINS string::lowercase('{k}') "
        f"OR string::lowercase(content) CONTAINS string::lowercase('{k}')"
        for k in keywords
    )
    sql = f"SELECT id, title, content FROM document WHERE {where_parts} LIMIT {topk}"
    rows = _q(sql)
    docs = _flatten(rows)
    results = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        results.append({
            "doc_id": _extract_id(d.get("id", "")),
            "title": d.get("title", ""),
            "excerpt": (d.get("content", "") or "")[:120],
        })
    return results


def _graph_search(query: str, topk: int) -> list[dict]:
    """图检索：实体匹配 → mentions 反查 → 关联文档扩展。

    三步图检索：
    ① 实体名匹配 → mentions 反查文档
    ② 命中文档沿 related/extends/depends 边扩展 1 跳（图扩展检索）
    ③ 实体间 entity_related 边反查关联实体再 mentions 反查
    """
    all_entities = list_entities()
    if not all_entities:
        return []
    query_lower = query.lower()

    # ① 实体名匹配 → mentions 反查
    direct_docs: list[dict] = []
    matched_entities: list[str] = []
    for ent in all_entities:
        if ent["name"].lower() in query_lower:
            matched_entities.append(ent["id"])
            try:
                rows = _q(f"SELECT in AS doc FROM {ent['id']}<-mentions")
                for r in _flatten(rows):
                    if isinstance(r, dict) and r.get("doc"):
                        doc = get_document(_extract_id(r["doc"]))
                        if doc:
                            direct_docs.append({
                                "doc_id": doc["id"],
                                "title": doc.get("title", ""),
                                "excerpt": (doc.get("content", "") or "")[:120],
                                "via": "entity-mention",
                            })
            except Exception:
                pass

    # ② 图扩展：命中文档沿 related/extends/depends 边扩展 1 跳
    expanded_docs: list[dict] = []
    direct_doc_ids = {d["doc_id"] for d in direct_docs}
    for doc_id in list(direct_doc_ids):
        for et in ["related", "extends", "depends", "implements"]:
            try:
                rows = _q(f"SELECT out AS node FROM {doc_id}->{et}")
                for r in _flatten(rows):
                    if isinstance(r, dict) and r.get("node"):
                        nid = _extract_id(r["node"])
                        if nid not in direct_doc_ids:
                            doc = get_document(nid)
                            if doc:
                                expanded_docs.append({
                                    "doc_id": doc["id"],
                                    "title": doc.get("title", ""),
                                    "excerpt": (doc.get("content", "") or "")[:120],
                                    "via": f"graph-expand:{et}",
                                })
            except Exception:
                pass

    # ③ entity_related 边反查关联实体
    for eid in matched_entities:
        try:
            rows = _q(f"SELECT out AS ent FROM {eid}->entity_related")
            for r in _flatten(rows):
                if isinstance(r, dict) and r.get("ent"):
                    related_eid = _extract_id(r["ent"])
                    d_rows = _q(f"SELECT in AS doc FROM {related_eid}<-mentions")
                    for d in _flatten(d_rows):
                        if isinstance(d, dict) and d.get("doc"):
                            did = _extract_id(d["doc"])
                            if did not in direct_doc_ids:
                                doc = get_document(did)
                                if doc:
                                    expanded_docs.append({
                                        "doc_id": doc["id"],
                                        "title": doc.get("title", ""),
                                        "excerpt": (doc.get("content", "") or "")[:120],
                                        "via": "entity-related",
                                    })
        except Exception:
            pass

    # 合并去重（直接命中优先，扩展结果按 via 标注）
    all_results = direct_docs + expanded_docs
    seen = set()
    unique = []
    for m in all_results:
        if m["doc_id"] not in seen:
            seen.add(m["doc_id"])
            unique.append(m)
    return unique[:topk]


def _metadata_search(query: str, topk: int) -> list[dict]:
    """元数据检索：解析 author=/topic= 等结构化条件。"""
    results: list[dict] = []
    # 解析 author=xxx
    author_match = re.search(r"author[=:]?\s*(\S+)", query, re.IGNORECASE)
    if author_match:
        author = author_match.group(1)
        rows = _q("SELECT id, title, content FROM document WHERE author = $author LIMIT $limit",
                  {"author": author, "limit": topk})
        for d in _flatten(rows):
            if isinstance(d, dict):
                results.append({
                    "doc_id": _extract_id(d.get("id", "")),
                    "title": d.get("title", ""),
                    "excerpt": (d.get("content", "") or "")[:120],
                })
    # 解析 topic=xxx
    topic_match = re.search(r"topic[=:]?\s*(\S+)", query, re.IGNORECASE)
    if topic_match:
        topic_key = re.sub(r"[^A-Za-z0-9_]", "_", topic_match.group(1).lower())
        for d in docs_by_topic(topic_key):
            results.append({
                "doc_id": d["doc_id"],
                "title": d.get("title", ""),
                "excerpt": "",
            })
    return results[:topk]


# ====================================================================== #
#  RawSource Service（对应 design.md 第一节："raw 文档也是对象"）
# ====================================================================== #
def create_raw(url: str | None, author: str | None,
               published: str | None, content: str,
               raw_key: str | None = None) -> dict:
    """创建 RawSource 对象。幂等：同 key 则更新。

    对应 design.md 第一节：RawSource 是独立对象，Markdown 只是导出产物。
    """
    key = raw_key or re.sub(r"[^A-Za-z0-9_]", "_", (url or content[:30]).lower()).strip("_")
    if not key:
        key = f"raw_{int(time.time())}"
    rid = _rid("raw", key)
    # published 是字符串日期，转换为 SurrealDB datetime 字面量
    published_clause = "type::datetime($published)" if published else "NONE"
    _q(f"""
        UPSERT {rid} SET
            url = $url,
            author = $author,
            published = {published_clause},
            content = $content,
            collected = time::now()
    """, {"url": url, "author": author, "published": published, "content": content})
    return get_raw(rid) or {"id": rid, "url": url, "author": author}


def get_raw(raw_id: str) -> dict | None:
    """获取单个 raw 对象。"""
    rid = raw_id if ":" in raw_id else _rid("raw", raw_id)
    r = _q_one(f"SELECT * FROM {rid}")
    if not r or not isinstance(r, dict):
        return None
    r["id"] = _extract_id(r.get("id", ""))
    return r


def list_raws() -> list[dict]:
    """列出所有 raw 对象。"""
    rows = _q("SELECT * FROM raw")
    raws = _flatten(rows)
    return [{"id": _extract_id(r.get("id", "")),
             "url": r.get("url", ""),
             "author": r.get("author", ""),
             "content": (r.get("content", "") or "")[:120]}
            for r in raws if isinstance(r, dict)]


def link_raw(doc_id: str, raw_id: str) -> dict:
    """关联文档与 raw：建立 references + updated_by 边。

    对应 design.md 第七节：updated_by 关系（文档由哪个 raw 更新）。
    """
    doc_rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    raw_rid = raw_id if ":" in raw_id else _rid("raw", raw_id)
    relate(doc_rid, "references", raw_rid)
    relate(doc_rid, "updated_by", raw_rid)
    return {"doc_id": doc_rid, "raw_id": raw_rid, "linked": True}


# ====================================================================== #
#  Archive Service（对应 design.md 第一节：ArchiveDocument 对象）
# ====================================================================== #
def create_archive(title: str, content: str, source: str | None = None) -> dict:
    """创建 ArchiveDocument 对象。"""
    key = re.sub(r"[^A-Za-z0-9_]", "_", title.lower()).strip("_") or f"arch_{int(time.time())}"
    rid = _rid("archive", key)
    _q(f"""
        UPSERT {rid} SET
            title = $title,
            content = $content,
            source = $source,
            archived = time::now()
    """, {"title": title, "content": content, "source": source})
    return {"id": rid, "title": title, "source": source}


def list_archives() -> list[dict]:
    """列出所有 archive 对象。"""
    rows = _q("SELECT * FROM archive")
    archs = _flatten(rows)
    return [{"id": _extract_id(a.get("id", "")),
             "title": a.get("title", ""),
             "source": a.get("source", ""),
             "content": (a.get("content", "") or "")[:120]}
            for a in archs if isinstance(a, dict)]


# ====================================================================== #
#  LLM Memory Graph Service（对应 design.md 第八节）
#  Conversation → about → Document
# ====================================================================== #
def record_conversation(question: str, answer: str | None,
                        doc_ids: list[str] | None = None) -> dict:
    """记录一次对话，并建立 about 边指向相关文档。

    对应 design.md 第八节：Query about Document。
    以后可分析"哪些文档用户问得最多"。
    """
    key = f"conv_{int(time.time())}"
    rid = _rid("conversation", key)
    _q(f"""
        UPSERT {rid} SET
            question = $question,
            answer = $answer,
            created = time::now()
    """, {"question": question, "answer": answer})
    if doc_ids:
        for did in doc_ids:
            doc_rid = did if ":" in did else _rid("document", did)
            relate(rid, "about", doc_rid)
    return {"id": rid, "question": question, "linked_docs": len(doc_ids) if doc_ids else 0}


def hot_documents(limit: int = 10) -> list[dict]:
    """统计 about 边入度，返回被问得最多的文档排行。

    对应 design.md 第八节："哪些文档用户问得最多"。
    """
    rows = _q(f"""
        SELECT count() AS cnt, in AS doc FROM about GROUP BY doc
        ORDER BY cnt DESC LIMIT {limit}
    """)
    result = []
    for r in _flatten(rows):
        if not isinstance(r, dict) or not r.get("doc"):
            continue
        doc_id = _extract_id(r["doc"])
        doc = get_document(doc_id)
        result.append({
            "doc_id": doc_id,
            "title": doc.get("title", "") if doc else "",
            "question_count": r.get("cnt", 0),
        })
    return result


def conversations_by_doc(doc_id: str) -> list[dict]:
    """反查某文档关联的对话。"""
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    rows = _q(f"SELECT in AS conv FROM {rid}<-about")
    result = []
    for r in _flatten(rows):
        if not isinstance(r, dict) or not r.get("conv"):
            continue
        conv_id = _extract_id(r["conv"])
        conv = _q_one(f"SELECT * FROM {conv_id}")
        if conv and isinstance(conv, dict):
            result.append({
                "id": conv_id,
                "question": conv.get("question", ""),
                "answer": (conv.get("answer", "") or "")[:120],
                "created": str(conv.get("created", "")),
            })
    return result


# ====================================================================== #
#  Agent 写入类工具（对应 design.md 第十一节：merge/update_metadata/build_graph）
# ====================================================================== #
def merge_document(source_id: str, target_id: str,
                   merged_title: str, merged_content: str,
                   merged_summary: str | None = None) -> dict:
    """合并两篇文档：创建新文档，建 supersedes 边，将源文档标记为 archived。

    对应 design.md 第十一节 merge_document() 工具。
    """
    source_rid = source_id if ":" in source_id else _rid("document", source_id)
    target_rid = target_id if ":" in target_id else _rid("document", target_id)
    # 创建合并后的新文档
    merged = create_document(
        title=merged_title, content=merged_content, summary=merged_summary,
        doc_key=re.sub(r"[^A-Za-z0-9_]", "_", merged_title.lower()).strip("_") + "_merged",
    )
    merged_rid = merged.get("id", "")
    if merged_rid:
        # 新文档 supersedes 源文档和目标文档
        relate(merged_rid, "supersedes", source_rid)
        relate(merged_rid, "supersedes", target_rid)
        # 源文档和目标文档标记为 archived
        _q(f"UPDATE {source_rid} SET status = 'archived'")
        _q(f"UPDATE {target_rid} SET status = 'archived'")
    return {"merged_doc_id": merged_rid, "superseded": [source_rid, target_rid]}


def update_metadata(doc_id: str, tags: list[str] | None = None,
                    entities: list[dict] | None = None,
                    topic: str | None = None,
                    author: str | None = None) -> dict:
    """更新文档元数据并重建 has_tag/mentions/belongs_to 边。

    对应 design.md 第十一节 update_metadata() 工具。
    """
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    # 更新 author 字段
    if author:
        _q(f"UPDATE {rid} SET author = $author", {"author": author})
    # 重建 topic 边
    if topic:
        _q(f"UPDATE {rid} SET topic_id = $topic", {"topic": topic})
        relate(rid, "belongs_to", _rid("topic", topic))
    # 重建 tag 边
    if tags:
        for tag_name in tags:
            tag_key = re.sub(r"[^A-Za-z0-9_]", "_", tag_name.lower()).strip("_")
            ensure_tag(tag_key, tag_name)
            relate(rid, "has_tag", _rid("tag", tag_key))
    # 重建 entity 边
    if entities:
        for ent in entities:
            ent_name = ent["name"]
            ent_key = re.sub(r"[^A-Za-z0-9_]", "_", ent_name.lower()).strip("_")
            ensure_entity(ent_key, ent_name, ent.get("type"))
            relate(rid, "mentions", _rid("entity", ent_key))
    return get_document(rid) or {"id": rid, "updated": True}


def build_graph(doc_id: str) -> dict:
    """对文档抽取实体/关系并建图边。

    对应 design.md 第十一节 build_graph() 工具。
    委托 llm.extract_entities() 抽取，然后落库建边。
    """
    import llm  # 延迟导入避免循环依赖
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    doc = get_document(rid)
    if not doc:
        return {"error": f"文档 {doc_id} 不存在"}
    try:
        result = llm.extract_entities(rid)
    except llm.LLMError as e:
        return {"error": f"抽取失败: {e}"}
    # 落库：创建/更新实体 + mentions 边 + entity 间关系
    for ent in result.get("entities", []):
        ent_key = re.sub(r"[^A-Za-z0-9_]", "_", ent["name"].lower()).strip("_")
        ensure_entity(ent_key, ent["name"], ent.get("type"))
        relate(rid, "mentions", _rid("entity", ent_key))
    for rel in result.get("relations", []):
        from_key = re.sub(r"[^A-Za-z0-9_]", "_", rel["from"].lower()).strip("_")
        to_key = re.sub(r"[^A-Za-z0-9_]", "_", rel["to"].lower()).strip("_")
        relate(f"entity:{from_key}", "entity_related", f"entity:{to_key}")
    return {
        "doc_id": rid,
        "entities_added": len(result.get("entities", [])),
        "relations_added": len(result.get("relations", [])),
    }


# ====================================================================== #
#  统计
# ====================================================================== #
def stats() -> dict:
    """返回知识库统计信息。"""
    def _count(table: str) -> int:
        try:
            rows = _q(f"SELECT count() FROM {table} GROUP ALL")
            r = _flatten(rows)
            if r and isinstance(r[0], dict):
                return r[0].get("count", 0)
        except Exception:
            pass
        return 0
    return {
        "documents": _count("document"),
        "topics": _count("topic"),
        "tags": _count("tag"),
        "entities": _count("entity"),
        "raws": _count("raw"),
        "archives": _count("archive"),
        "conversations": _count("conversation"),
        "versions": _count("version"),
    }
