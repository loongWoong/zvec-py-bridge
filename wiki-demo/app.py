#!/usr/bin/env python3
"""
Semantic Wiki Web 应用
=====================

基于 SurrealDB（图+文档+元数据）+ zvec（向量检索）+ LLM（编译/抽取/问答）的
语义 Wiki 运行时。对应 design.md 的四层架构。

启动方式
--------
    python app.py

浏览器访问 http://localhost:8090 即可使用。

前置条件
--------
1. SurrealDB 嵌入式模式（无需单独起 server，由 surrealdb Python SDK 内置）
2. zvec REST Bridge 服务已启动（默认 http://localhost:8666）—— 可选，缺失时向量检索路降级
3. Ollama 已运行且已拉取 embedding 模型 —— 可选，同上
4. LLM 服务（OpenAI 兼容 / Ollama）—— 可选，缺失时编译/抽取/问答不可用
"""
from __future__ import annotations

import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import db  # noqa: E402
import llm  # noqa: E402
import ontology  # noqa: E402
import ontology_builder  # noqa: E402
import pipeline  # noqa: E402
import seed  # noqa: E402
import wiki_runtime as wr  # noqa: E402
import zvec_client  # noqa: E402

# ====================================================================== #
#  应用
# ====================================================================== #
app = FastAPI(title="Semantic Wiki Runtime", version="1.0.0")

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(DEMO_DIR, "static")

# 运行时状态
_state: dict = {
    "started": False,
    "zvec": False,
    "ollama": False,
    "llm": False,
}

# 是否在启动时后台修复概念绑定稀疏问题（使 Re-Rank 概念距离因子生效）。
# 默认关闭：避免每次启动都触发 LLM 标注，按需用 POST /api/repair-bindings 触发。
REPAIR_BINDINGS_ON_STARTUP = False


@app.on_event("startup")
def startup():
    """启动时初始化 SurrealDB schema + 种子数据（图/文档同步，向量入库后台异步）。"""
    import threading

    print("=" * 60)
    print("  Semantic Wiki Runtime 启动中...")
    print("=" * 60)

    # 1. 初始化 SurrealDB schema
    try:
        db.init()
        print(f"  ✓ SurrealDB 已连接 ({config.SURREAL_DB})")
    except Exception as e:
        print(f"  ✗ SurrealDB 初始化失败: {e}")
        raise

    # 2. 检查外部服务（zvec / Ollama）
    try:
        health = zvec_client.check_health()
        _state["zvec"] = health["zvec"]
        _state["ollama"] = health["ollama"]
        if health["zvec"]:
            print(f"  ✓ zvec REST Bridge 可达 (v{health['zvec_version']})")
        else:
            print("  ⚠ zvec 不可达 — 向量检索路将降级")
        if health["ollama"]:
            print(f"  ✓ Ollama 可达 (embed: {config.EMBED_MODEL})")
        else:
            print("  ⚠ Ollama 不可达 — 向量检索路将降级")
    except Exception as e:
        print(f"  ⚠ 健康检查异常: {e}")

    # 3. 检查 LLM
    try:
        import requests
        if config.LLM_API == "openai":
            headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}
            r = requests.get(f"{config.LLM_URL}/v1/models", headers=headers, timeout=config.LLM_HEALTH_TIMEOUT)
        else:
            r = requests.get(f"{config.LLM_URL}/api/tags", timeout=config.LLM_HEALTH_TIMEOUT)
        _state["llm"] = r.status_code == 200
        if _state["llm"]:
            print(f"  ✓ LLM 可达 ({config.LLM_MODEL})")
        else:
            print(f"  ⚠ LLM 不可达 — 编译/抽取/问答功能不可用")
    except Exception:
        print("  ⚠ LLM 不可达 — 编译/抽取/问答功能不可用")

    # 4. 种子数据灌入（图/文档同步快速完成，向量入库后台异步）
    try:
        result = seed.seed_all_sync()
        if result.get("skipped"):
            print(f"  ✓ 种子数据已存在，跳过 ({result['stats']['documents']} 篇文档)")
        else:
            print(f"  ✓ 图/文档种子已灌入: {result['documents']} 篇文档, "
                  f"{result['entities']} 实体, {result['chunks']} 向量分块待入库")
            # 后台异步灌入向量库（不阻塞 Web 服务启动）
            if _state["zvec"] and _state["ollama"]:
                _state["vector_seeding"] = True
                def _bg_seed():
                    try:
                        seed.seed_vectors(result["chunks"])
                        _state["vector_seeding"] = False
                        _state["zvec_seeded"] = True
                        print("  ✓ 向量库后台灌入完成")
                    except Exception as e:
                        _state["vector_seeding"] = False
                        print(f"  ⚠ 向量库后台灌入失败: {e}")
                threading.Thread(target=_bg_seed, daemon=True).start()
                print("  ♦ 向量库后台灌入中（不阻塞服务）...")
            else:
                print("  ⚠ zvec/Ollama 不可达，跳过向量入库（图/全文/元数据检索仍可用）")
    except Exception as e:
        print(f"  ⚠ 种子数据灌入异常: {e}")

    # 5. 灌入种子本体（ontology/concepts.yaml）—— 与文档种子独立，幂等
    try:
        onto_result = seed.seed_ontology()
        if onto_result.get("skipped"):
            if _state.get("started"):
                pass
        # 成功/跳过信息已在 seed_ontology 内打印
    except Exception as e:
        print(f"  ⚠ 本体种子灌入异常: {e}")

    # 6. （可选）启动后后台修复概念绑定稀疏问题，使 Re-Rank 概念距离因子生效
    if REPAIR_BINDINGS_ON_STARTUP and _state.get("llm"):
        def _bg_repair():
            try:
                res = pipeline.repair_concept_bindings(limit=200)
                print(f"  ✓ 概念绑定修复完成: {res}")
            except Exception as e:
                print(f"  ⚠ 概念绑定修复失败: {e}")
        threading.Thread(target=_bg_repair, daemon=True).start()
        print("  ♦ 概念绑定修复后台进行中（不阻塞服务）...")

    _state["started"] = True
    print("=" * 60)
    print(f"  访问 http://localhost:{config.WIKI_PORT}")
    print("=" * 60)


# ====================================================================== #
#  请求模型
# ====================================================================== #
class SearchRequest(BaseModel):
    query: str
    topk: int = 5
    code: bool = False  # 是否叠加第五路「代码结构化检索」（需已定位概念）


class CodeSearchRequest(BaseModel):
    query: str
    topk: int = 10
    concept_ids: list[str] | None = None   # 直接指定概念 ID
    concept_names: list[str] | None = None  # 或指定概念名（自动解析为 ID）
    auto_locate: bool = True                # 未指定概念时，是否用 LLM 自动定位概念


class AskRequest(BaseModel):
    question: str
    max_iterations: int = 6


class CreateDocumentRequest(BaseModel):
    title: str
    content: str
    summary: str | None = None
    topic_id: str | None = None
    author: str | None = None
    doc_key: str | None = None
    tags: list[str] | None = None
    entities: list[dict] | None = None
    relations: list[dict] | None = None


class UpdateDocumentRequest(BaseModel):
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    topic_id: str | None = None
    author: str | None = None
    status: str | None = None


class CompileRequest(BaseModel):
    raw_content: str
    topic: str | None = None
    title_hint: str | None = None


class ExtractRequest(BaseModel):
    document_id: str


class CreateRawRequest(BaseModel):
    url: str | None = None
    author: str | None = None
    published: str | None = None
    content: str
    raw_key: str | None = None


class CreateArchiveRequest(BaseModel):
    title: str
    content: str
    source: str | None = None


class ConversationRequest(BaseModel):
    question: str
    answer: str | None = None
    doc_ids: list[str] | None = None


class MergeDocumentRequest(BaseModel):
    source_id: str
    target_id: str
    merged_title: str
    merged_content: str
    merged_summary: str | None = None


class UpdateMetadataRequest(BaseModel):
    doc_id: str
    tags: list[str] | None = None
    entities: list[dict] | None = None
    topic: str | None = None
    author: str | None = None


# ====================================================================== #
#  健康检查
# ====================================================================== #
@app.get("/api/health")
def health():
    """健康检查 + 知识库统计。"""
    try:
        stats = wr.stats()
    except Exception:
        stats = {}
    return {
        "status": "UP" if _state["started"] else "STARTING",
        "surrealdb": True,
        "zvec": _state["zvec"],
        "ollama": _state["ollama"],
        "llm": _state["llm"],
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "vector_seeding": _state.get("vector_seeding", False),
        "vector_ready": _state.get("zvec_seeded", False),
        "stats": stats,
    }


# ====================================================================== #
#  文档 CRUD
# ====================================================================== #
@app.get("/api/documents")
def list_documents(topic: str | None = None, limit: int = 100):
    """列出文档（可按 topic 过滤）。"""
    return {"documents": wr.list_documents(topic=topic, limit=limit)}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    """获取单个文档（含关系邻接）。"""
    doc = wr.get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"文档 {doc_id} 不存在")
    return doc


@app.post("/api/documents")
def create_document(req: CreateDocumentRequest):
    """创建文档。"""
    doc = wr.create_document(
        title=req.title, content=req.content, summary=req.summary,
        topic_id=req.topic_id, author=req.author, doc_key=req.doc_key,
        tags=req.tags, entities=req.entities, relations=req.relations,
    )
    return doc


@app.put("/api/documents/{doc_id}")
def update_document(doc_id: str, req: UpdateDocumentRequest):
    """更新文档（自动保存版本快照）。"""
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    doc = wr.update_document(doc_id, **fields)
    if not doc:
        raise HTTPException(404, f"文档 {doc_id} 不存在")
    return doc


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """删除文档及其关联边、版本快照、向量分块。"""
    result = wr.delete_document(doc_id)
    if not result.get("deleted"):
        raise HTTPException(404, result.get("error", f"文档 {doc_id} 删除失败"))
    # 同步删除向量库中的分块
    try:
        vec_deleted = zvec_client.delete_by_document_id(doc_id)
        result["vector_chunks_removed"] = vec_deleted
    except Exception as e:
        result["vector_warning"] = str(e)
    return result


@app.post("/api/documents/check-duplicates")
def check_duplicates(titles: list[str]):
    """批量检查标题是否已有重复文档（用于编译前检测）。

    请求体: ["标题1", "标题2", ...]
    返回: {"duplicates": [{"title": "标题1", "doc_id": "...", "existing_title": "..."}, ...]}
    """
    duplicates = []
    for title in titles:
        dup = wr.check_duplicate(title)
        if dup:
            duplicates.append({
                "title": title,
                "doc_id": dup["id"],
                "existing_title": dup["title"],
            })
    return {"duplicates": duplicates}


@app.get("/api/documents/{doc_id}/export")
def export_markdown(doc_id: str):
    """导出文档为 Markdown。"""
    md = wr.export_markdown(doc_id)
    if md is None:
        raise HTTPException(404, f"文档 {doc_id} 不存在")
    return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")


@app.get("/api/documents/{doc_id}/versions")
def list_versions(doc_id: str):
    """列出文档的历史版本。"""
    return {"versions": wr.list_versions(doc_id)}


# ====================================================================== #
#  图操作
#  注意：固定路径路由（full/stats/central）必须定义在 {node_id} 通配路由之前，
#  否则 "full"/"stats"/"central" 会被当作 node_id 匹配。
# ====================================================================== #
@app.get("/api/graph/full")
def graph_full():
    """全图数据（所有节点+边），供前端 D3 渲染。"""
    return wr.graph_full()


@app.get("/api/graph/stats")
def graph_stats():
    """全局图统计：节点数/边数/各边类型计数/平均度/密度。"""
    return wr.graph_stats()


@app.get("/api/graph/central")
def top_central(limit: int = 10):
    """度中心性排序：度最高的 hub 节点。"""
    return {"nodes": wr.top_central_nodes(limit)}


@app.get("/api/graph/{node_id}")
def graph_neighbors(node_id: str, direction: str = "both", edge_type: str | None = None):
    """图邻接遍历。"""
    return {"node_id": node_id, "neighbors": wr.neighbors(node_id, direction, edge_type)}


@app.get("/api/graph/{node_id}/subtree")
def graph_subtree(node_id: str, depth: int = 2):
    """获取以某节点为根的局部子图。"""
    return wr.graph_subtree(node_id, depth)


@app.get("/api/graph/{node_id}/shortest-path")
def shortest_path(node_id: str, target: str, max_depth: int = 6):
    """BFS 最短路径：两节点间的最短关系路径。"""
    return wr.shortest_path(node_id, target, max_depth)


@app.get("/api/graph/{node_id}/common")
def common_neighbors(node_id: str, other: str):
    """共同邻居：两节点的共同邻居。"""
    return wr.common_neighbors(node_id, other)


@app.get("/api/graph/{node_id}/degree")
def node_degree(node_id: str):
    """度中心性：节点的入度/出度/总度。"""
    return wr.node_degree(node_id)


@app.get("/api/graph/{node_id}/multi-hop")
def multi_hop(node_id: str, depth: int = 3):
    """多跳邻接：BFS N 跳，按层级分组。"""
    return wr.multi_hop_neighbors(node_id, depth)


@app.get("/api/documents/{doc_id}/related")
def related_articles(doc_id: str):
    """获取与文档相关的文章。"""
    return {"related": wr.related_articles(doc_id)}


@app.get("/api/documents/{doc_id}/lineage")
def knowledge_lineage(doc_id: str, max_depth: int = 3):
    """知识血缘：文档的上下游知识链。"""
    return wr.knowledge_lineage(doc_id, max_depth)


# ====================================================================== #
#  元数据
# ====================================================================== #
@app.get("/api/topics")
def list_topics():
    return {"topics": wr.list_topics()}


@app.get("/api/tags")
def list_tags():
    return {"tags": wr.list_tags()}


@app.get("/api/entities")
def list_entities():
    return {"entities": wr.list_entities()}


@app.get("/api/entities/{name}")
def entity_lookup(name: str):
    """按名称查找实体。"""
    result = wr.entity_lookup(name)
    if not result:
        raise HTTPException(404, f"实体 {name} 不存在")
    return result


@app.get("/api/entities/{name}/co-occurrence")
def entity_co_occurrence(name: str):
    """共现分析：与某实体共同出现在文档中的其他实体。"""
    return wr.entity_co_occurrence(name)


# ====================================================================== #
#  RawSource（对应 design.md 第一节：raw 文档也是对象）
# ====================================================================== #
@app.get("/api/raws")
def list_raws():
    """列出所有 raw 源。"""
    return {"raws": wr.list_raws()}


@app.get("/api/raws/{raw_id}")
def get_raw(raw_id: str):
    """获取单个 raw 源。"""
    raw = wr.get_raw(raw_id)
    if not raw:
        raise HTTPException(404, f"raw {raw_id} 不存在")
    return raw


@app.post("/api/raws")
def create_raw(req: CreateRawRequest):
    """创建 raw 源。"""
    return wr.create_raw(req.url, req.author, req.published, req.content, req.raw_key)


@app.post("/api/documents/{doc_id}/link-raw")
def link_raw(doc_id: str, raw_id: str):
    """关联文档与 raw（建 references + updated_by 边）。"""
    return wr.link_raw(doc_id, raw_id)


# ====================================================================== #
#  Archive（对应 design.md 第一节：ArchiveDocument）
# ====================================================================== #
@app.get("/api/archives")
def list_archives():
    """列出所有 archive 文档。"""
    return {"archives": wr.list_archives()}


@app.post("/api/archives")
def create_archive(req: CreateArchiveRequest):
    """创建 archive 文档。"""
    return wr.create_archive(req.title, req.content, req.source)


# ====================================================================== #
#  Version Chain（对应 design.md 第七节）
# ====================================================================== #
@app.get("/api/documents/{doc_id}/version-chain")
def version_chain(doc_id: str):
    """完整版本链（通过 previous_version 边遍历）。"""
    return {"chain": wr.version_chain(doc_id)}


# ====================================================================== #
#  LLM Memory Graph（对应 design.md 第八节）
# ====================================================================== #
@app.get("/api/conversations/hot")
def hot_documents(limit: int = 10):
    """热门文档排行：被问得最多的文档。"""
    return {"documents": wr.hot_documents(limit)}


@app.get("/api/documents/{doc_id}/conversations")
def doc_conversations(doc_id: str):
    """文档关联的对话记录。"""
    return {"conversations": wr.conversations_by_doc(doc_id)}


@app.post("/api/conversations")
def record_conversation(req: ConversationRequest):
    """记录一次对话到 Memory Graph。"""
    return wr.record_conversation(req.question, req.answer, req.doc_ids)


# ====================================================================== #
#  Agent 写入类工具（对应 design.md 第十一节）
# ====================================================================== #
@app.post("/api/documents/merge")
def merge_document(req: MergeDocumentRequest):
    """合并两篇文档。"""
    return wr.merge_document(req.source_id, req.target_id,
                             req.merged_title, req.merged_content, req.merged_summary)


@app.post("/api/documents/update-metadata")
def update_metadata(req: UpdateMetadataRequest):
    """更新文档元数据。"""
    return wr.update_metadata(req.doc_id, req.tags, req.entities, req.topic, req.author)


@app.post("/api/documents/{doc_id}/build-graph")
def build_graph(doc_id: str):
    """对文档抽取实体/关系并建图边。"""
    if not _state["llm"]:
        raise HTTPException(503, "LLM 服务不可达，无法构建图")
    return wr.build_graph(doc_id)


# ====================================================================== #
#  搜索（四路融合）
# ====================================================================== #
@app.get("/api/search")
def search(q: str, topk: int = 5, code: bool = False):
    """四路融合检索（可选叠加第五路代码结构化检索）。"""
    return wr.search_documents(q, topk=topk, code_route=code)


@app.post("/api/search")
def search_post(req: SearchRequest):
    """四路融合检索（POST，可选叠加第五路代码结构化检索）。"""
    return wr.search_documents(req.query, topk=req.topk, code_route=req.code)


# ====================================================================== #
#  代码结构化检索（索引 D / 策略5 [调用链]）
# ====================================================================== #
def _resolve_concept_ids(req: CodeSearchRequest) -> tuple[list[str], list[str]]:
    """将请求中的概念名/ID 解析为概念 ID 列表，返回 (ids, notes)。

    优先级：concept_ids > concept_names > 自动定位（需 LLM）。
    """
    ids: list[str] = list(req.concept_ids or [])
    notes: list[str] = []
    try:
        import ontology
        if req.concept_names:
            name_to_id = {c["name"]: c["id"] for c in ontology.list_concepts()}
            for n in req.concept_names:
                if n in name_to_id:
                    ids.append(name_to_id[n])
                else:
                    notes.append(f"概念名未命中: {n}")
        if not ids and req.auto_locate and _state.get("llm"):
            try:
                import concept_locator
                located = concept_locator.locate(req.query)
                names = [c["name"] for c in located.get("located_concepts", [])]
                names += [c["name"] for c in located.get("implicit_concepts", [])]
                name_to_id = {c["name"]: c["id"] for c in ontology.list_concepts()}
                for n in names:
                    if n in name_to_id and name_to_id[n] not in ids:
                        ids.append(name_to_id[n])
                notes.append(f"自动定位概念: {names}")
            except Exception as e:
                notes.append(f"自动定位失败: {e}")
    except Exception as e:
        notes.append(f"本体不可用: {e}")
    # 去重
    seen = set()
    unique = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique, notes


@app.post("/api/code-search")
def code_search_post(req: CodeSearchRequest):
    """结构化代码检索：按概念绑定的文件路径，提取符号表/调用链并匹配查询。

    对应 AI推理引擎.md Step 4 索引 D（结构化索引/代码专用）与策略5 [调用链]。
    不依赖 tree-sitter：用 code_analyzer 正则提取函数/类/调用关系。
    """
    concept_ids, notes = _resolve_concept_ids(req)
    if not concept_ids:
        return {"query": req.query, "results": [], "notes": notes,
                "reason": "未解析到概念，无法定位代码文件"}
    results = wr.search_documents(
        req.query, topk=req.topk, use_rerank=False,
        concept_ids=concept_ids, code_route=True,
    ).get("results", [])
    # 仅保留代码路线命中
    code_results = [r for r in results if "code" in (r.get("sources") or [])]
    return {"query": req.query, "concept_ids": concept_ids,
            "results": code_results, "notes": notes}


@app.get("/api/code-search")
def code_search_get(q: str, topk: int = 10,
                    concept_ids: str | None = None,
                    concept_names: str | None = None):
    """结构化代码检索（GET）。concept_ids / concept_names 用逗号分隔。"""
    parsed = CodeSearchRequest(
        query=q, topk=topk,
        concept_ids=[s.strip() for s in concept_ids.split(",")] if concept_ids else None,
        concept_names=[s.strip() for s in concept_names.split(",")] if concept_names else None,
    )
    return code_search_post(parsed)


@app.get("/api/documents/{doc_id}/code-symbols")
def document_code_symbols(doc_id: str):
    """返回文档落库时提取的代码符号索引（索引 D，仅代码文档有）。"""
    doc = wr.get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"文档 {doc_id} 不存在")
    symbols = doc.get("code_symbols")
    if not symbols:
        return {"doc_id": doc_id, "has_code_symbols": False,
                "reason": "非代码文档或未提取符号"}
    return {"doc_id": doc_id, "has_code_symbols": True, **symbols}


@app.post("/api/repair-bindings")
def repair_bindings(limit: int = 200):
    """重新标注现有文档的概念绑定（LLM 多选 + 关键词兜底）。

    用于修复概念绑定稀疏问题，使 Re-Rank 的概念距离因子真正生效。
    返回 {scanned, updated, errors}。依赖 LLM（标注阶段）。
    """
    if not _state.get("llm"):
        raise HTTPException(503, "LLM 不可达，无法重新标注概念")
    return pipeline.repair_concept_bindings(limit=limit)


# ====================================================================== #
#  LLM 功能
# ====================================================================== #
@app.post("/api/ask")
def ask(req: AskRequest):
    """Agent 问答。"""
    if not _state["llm"]:
        raise HTTPException(503, "LLM 服务不可达，无法执行问答")
    try:
        result = llm.run_agent(req.question, max_iterations=req.max_iterations)
        return result
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


@app.post("/api/compile")
def compile_doc(req: CompileRequest):
    """LLM 编译 raw → wiki 文档。"""
    if not _state["llm"]:
        raise HTTPException(503, "LLM 服务不可达，无法编译文档")
    try:
        compiled = llm.compile_document(req.raw_content, req.topic, req.title_hint)
        # 落库
        doc = wr.create_document(
            title=compiled.get("title", "未命名"),
            content=compiled.get("content", ""),
            summary=compiled.get("summary"),
            topic_id=req.topic,
            tags=compiled.get("suggested_tags"),
            entities=compiled.get("entities"),
            relations=compiled.get("suggested_relations"),
        )
        # 自动同步向量
        vec_result = None
        if _state["zvec"] and _state["ollama"] and doc:
            try:
                vec_result = pipeline.sync_vectors_for_document(doc["id"])
            except Exception as e:
                vec_result = {"status": "warning", "reason": str(e)}
        return {"compiled": compiled, "document": doc, "vector_sync": vec_result}
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


@app.post("/api/compile-batch")
async def compile_batch(
    files: list[UploadFile] = File(...),
    topic: str | None = Form(default=None),
):
    """批量编译：上传多个文件（.md/.txt），逐个 LLM 编译为 Wiki 文档。

    支持文件上传和文件夹遍历上传。文件名作为 title_hint 传入。
    返回每个文件的编译结果（成功/失败分别记录）。
    """
    if not _state["llm"]:
        raise HTTPException(503, "LLM 服务不可达，无法编译文档")

    results: list[dict] = []
    for f in files:
        filename = f.filename or "unknown"
        # 跳过非文本文件
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".md", ".txt", ".markdown", ""):
            results.append({"file": filename, "status": "skipped", "reason": f"不支持的文件类型: {ext}"})
            continue
        try:
            raw_bytes = await f.read()
            raw_content = raw_bytes.decode("utf-8", errors="replace").strip()
            if not raw_content:
                results.append({"file": filename, "status": "skipped", "reason": "文件为空"})
                continue
            # 用文件名（去扩展名）作为 title_hint
            title_hint = os.path.splitext(filename)[0]
            compiled = llm.compile_document(raw_content, topic, title_hint)
            doc = wr.create_document(
                title=compiled.get("title", title_hint),
                content=compiled.get("content", ""),
                summary=compiled.get("summary"),
                topic_id=topic,
                tags=compiled.get("suggested_tags"),
                entities=compiled.get("entities"),
                relations=compiled.get("suggested_relations"),
            )
            results.append({
                "file": filename,
                "status": "ok",
                "title": doc.get("title", ""),
                "doc_id": doc.get("id", ""),
            })
            # 后台异步同步向量
            if _state["zvec"] and _state["ollama"] and doc:
                try:
                    pipeline.sync_vectors_for_document(doc["id"])
                except Exception:
                    pass
        except llm.LLMError as e:
            results.append({"file": filename, "status": "error", "reason": str(e)})
        except Exception as e:
            results.append({"file": filename, "status": "error", "reason": str(e)})
        finally:
            await f.close()

    succeeded = sum(1 for r in results if r["status"] == "ok")
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    }


@app.post("/api/extract")
def extract(req: ExtractRequest):
    """对已有文档抽取实体/关系。"""
    if not _state["llm"]:
        raise HTTPException(503, "LLM 服务不可达，无法抽取实体")
    try:
        result = llm.extract_entities(req.document_id)
        # 落库：创建/更新实体 + mentions 边 + entity 间关系
        doc = wr.get_document(req.document_id)
        if not doc:
            raise HTTPException(404, f"文档 {req.document_id} 不存在")
        rid = doc["id"]
        for ent in result.get("entities", []):
            import re as _re
            ent_key = wr._safe_key(ent["name"].lower())
            wr.ensure_entity(ent_key, ent["name"], ent.get("type"))
            wr.relate(rid, "mentions", f"entity:{ent_key}")
        for rel in result.get("relations", []):
            from_key = wr._safe_key(rel["from"].lower())
            to_key = wr._safe_key(rel["to"].lower())
            wr.relate(f"entity:{from_key}", "entity_related", f"entity:{to_key}")
        return {"extracted": result, "doc_id": rid}
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


# ====================================================================== #
#  入库管道（Phase 1：文件/目录导入 + 向量同步）
# ====================================================================== #
class IngestFileRequest(BaseModel):
    file_path: str
    topic: str | None = None
    author: str | None = None
    skip_existing: bool = True
    skip_zvec: bool = False


class IngestDirectoryRequest(BaseModel):
    dir_path: str
    topic: str | None = None
    author: str | None = None
    recursive: bool = True
    skip_existing: bool = True
    skip_zvec: bool = False
    max_files: int = 500


@app.post("/api/pipeline/ingest-file")
def api_ingest_file(req: IngestFileRequest):
    """导入单个文件到知识库（自动切分 + 写 SurrealDB + 写 zvec）。"""
    return pipeline.ingest_file(
        req.file_path, topic=req.topic, author=req.author,
        skip_existing=req.skip_existing, skip_zvec=req.skip_zvec,
    )


@app.post("/api/pipeline/ingest-directory")
def api_ingest_directory(req: IngestDirectoryRequest):
    """批量导入目录到知识库。"""
    return pipeline.ingest_directory(
        req.dir_path, topic=req.topic, author=req.author,
        recursive=req.recursive, skip_existing=req.skip_existing,
        skip_zvec=req.skip_zvec, max_files=req.max_files,
    )


@app.post("/api/documents/{doc_id}/sync-vectors")
def api_sync_vectors(doc_id: str):
    """为已有文档重建向量索引（删除旧向量 + 重新切分 + 入库）。"""
    if not _state["zvec"] or not _state["ollama"]:
        raise HTTPException(503, "zvec/Ollama 不可达，无法同步向量")
    result = pipeline.sync_vectors_for_document(doc_id)
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "同步失败"))
    return result


# ====================================================================== #
#  本体层（Phase 2：概念层级 + 关系 + 文档绑定）
#  对应 AI推理引擎.md Step 2 本体构建 / Step 5 概念定位 / Step 6 本体展开
# ====================================================================== #
class CreateConceptRequest(BaseModel):
    name: str
    concept_type: str = "concept"
    description: str = ""
    parent_id: str | None = None
    key: str | None = None


class AddRelationRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str = "related"


class BindConceptRequest(BaseModel):
    concept_id: str
    document_id: str
    binding_type: str = "primary"
    file_path: str = ""
    function_name: str = ""


class ImportOntologyRequest(BaseModel):
    file_path: str | None = None
    yaml_content: str | None = None
    clear_existing: bool = False


class BuildOntologyRequest(BaseModel):
    domain_hint: str = ""
    output_yaml: str | None = None


@app.get("/api/ontology/stats")
def ontology_stats():
    """本体统计：概念数 / 关系数 / 绑定数。"""
    return ontology.stats()


@app.get("/api/ontology/concepts")
def ontology_list_concepts(concept_type: str | None = None):
    """列出所有概念（可按类型过滤）。"""
    return {"concepts": ontology.list_concepts(concept_type)}


@app.get("/api/ontology/graph")
def ontology_graph():
    """完整本体图（概念节点 + is-a 层级边 + 关系边），供前端 D3 渲染。"""
    return ontology.get_full_graph()


@app.get("/api/ontology/concepts/{concept_id}")
def ontology_get_concept(concept_id: str):
    """获取概念详情（含子概念、父概念、关系、绑定文档）。"""
    concept = ontology.get_concept(concept_id)
    if not concept:
        raise HTTPException(404, f"概念 {concept_id} 不存在")
    return concept


@app.get("/api/ontology/concepts/{concept_id}/expand")
def ontology_expand(concept_id: str, depth: int = 2):
    """以概念为中心展开本体子图（N 跳），对应 Step 6 本体展开。"""
    result = ontology.expand_concept(concept_id, depth)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.post("/api/ontology/concepts")
def ontology_create_concept(req: CreateConceptRequest):
    """创建概念节点。"""
    return ontology.create_concept(
        name=req.name, concept_type=req.concept_type,
        description=req.description, parent_id=req.parent_id, key=req.key,
    )


@app.post("/api/ontology/relations")
def ontology_add_relation(req: AddRelationRequest):
    """建立概念间关系边（depends/triggers/contains/contrasts/related/...）。"""
    result = ontology.add_relation(req.source_id, req.target_id, req.relation_type)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/ontology/bindings")
def ontology_bind(req: BindConceptRequest):
    """将概念绑定到文档（primary/secondary/inferred）。"""
    result = ontology.bind_concept(
        req.concept_id, req.document_id, req.binding_type,
        req.file_path, req.function_name,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.delete("/api/ontology/concepts/{concept_id}")
def ontology_delete(concept_id: str):
    """删除概念及其关联边、绑定。"""
    return ontology.delete_concept(concept_id)


@app.get("/api/ontology/export")
def ontology_export():
    """导出全部本体为 YAML（Git 版本管理 / 人工编辑回流）。"""
    yaml_str = ontology.export_to_yaml()
    return PlainTextResponse(yaml_str, media_type="text/yaml; charset=utf-8")


@app.post("/api/ontology/import")
def ontology_import(req: ImportOntologyRequest):
    """从 YAML 导入本体。

    可传 file_path（服务器路径）或 yaml_content（直接内容）。
    不传则加载内置种子 ontology/concepts.yaml。
    """
    import tempfile
    if req.yaml_content:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(req.yaml_content)
            tmp_path = f.name
        try:
            return ontology.import_from_yaml(tmp_path, clear_existing=req.clear_existing)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    file_path = req.file_path or os.path.join(DEMO_DIR, "ontology", "concepts.yaml")
    if not os.path.exists(file_path):
        raise HTTPException(404, f"YAML 文件不存在: {file_path}")
    return ontology.import_from_yaml(file_path, clear_existing=req.clear_existing)


@app.post("/api/ontology/build")
def ontology_build(req: BuildOntologyRequest):
    """LLM 辅助本体构建：扫描文档 → 提议概念/关系/绑定 → 输出 YAML。

    对应 AI推理引擎.md Step 2「LLM 辅助抽取 + 人工校验」。
    """
    if not _state["llm"]:
        raise HTTPException(503, "LLM 服务不可达，无法构建本体")
    try:
        return ontology_builder.propose_all(
            domain_hint=req.domain_hint,
            output_yaml=req.output_yaml,
        )
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


# ====================================================================== #
#  静态文件
# ====================================================================== #
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ====================================================================== #
#  入口
# ====================================================================== #
if __name__ == "__main__":
    uvicorn.run(app, host=config.WIKI_HOST, port=config.WIKI_PORT)
