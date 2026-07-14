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
    edge_types = ["belongs_to", "has_tag", "mentions", "references",
                  "related", "depends", "extends", "implements",
                  "contradicts", "supersedes"]
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
    """图邻接遍历。direction: out/in/both。"""
    rid = node_id if ":" in node_id else _rid(node_id.split(":")[0] if ":" in node_id else "document", node_id)
    results: list[dict] = []

    edge_types = [edge_type] if edge_type else [
        "belongs_to", "has_tag", "mentions", "references",
        "related", "depends", "extends", "implements",
        "contradicts", "supersedes", "child_of", "entity_related",
    ]

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
    """获取与文档相关的文章（related/extends/depends/implements 边）。"""
    rid = doc_id if ":" in doc_id else _rid("document", doc_id)
    rel_types = ["related", "extends", "depends", "implements", "supersedes"]
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
    """以某节点为根，获取局部子图（节点+边）。"""
    rid = node_id if ":" in node_id else _rid("document", node_id)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def _visit(current_rid: str, current_depth: int):
        if current_depth < 0:
            return
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

        nbrs = neighbors(current_rid, direction="both")
        for nb in nbrs:
            nb_node = nb["node"]
            edges.append({
                "from": current_rid if nb["direction"] == "out" else nb_node,
                "to": nb_node if nb["direction"] == "out" else current_rid,
                "type": nb["edge_type"],
            })
            if nb_node not in nodes:
                _visit(nb_node, current_depth - 1)

    _visit(rid, depth)
    return {"nodes": list(nodes.values()), "edges": edges}


# ====================================================================== #
#  Metadata Service
# ====================================================================== #
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
    """保存文档当前版本快照。"""
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
    """全文检索：尝试 SEARCH 索引，降级为 CONTAINS 子串匹配。"""
    keywords = [w.strip() for w in re.split(r"[\s,，。、]+", query) if len(w.strip()) >= 2]
    if not keywords:
        return []
    # 降级方案：CONTAINS 子串匹配（SurrealDB 用 string::lowercase 做大小写不敏感）
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
    """图检索：从 query 中识别已知实体名，反查 mentions 该实体的文档。"""
    all_entities = list_entities()
    if not all_entities:
        return []
    query_lower = query.lower()
    matched: list[dict] = []
    for ent in all_entities:
        if ent["name"].lower() in query_lower:
            # 反查 mentions 该实体的文档
            eid = ent["id"]
            try:
                rows = _q(f"SELECT in AS doc FROM {eid}<-mentions")
                for r in _flatten(rows):
                    if isinstance(r, dict) and r.get("doc"):
                        doc = get_document(_extract_id(r["doc"]))
                        if doc:
                            matched.append({
                                "doc_id": doc["id"],
                                "title": doc.get("title", ""),
                                "excerpt": (doc.get("content", "") or "")[:120],
                            })
            except Exception:
                pass
    # 去重
    seen = set()
    unique = []
    for m in matched:
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
        "versions": _count("version"),
    }
