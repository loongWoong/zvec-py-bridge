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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import db  # noqa: E402
import llm  # noqa: E402
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
            r = requests.get(f"{config.LLM_URL}/v1/models", headers=headers, timeout=3)
        else:
            r = requests.get(f"{config.LLM_URL}/api/tags", timeout=3)
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
    """删除文档。"""
    ok = wr.delete_document(doc_id)
    if not ok:
        raise HTTPException(404, f"文档 {doc_id} 删除失败")
    return {"deleted": True, "doc_id": doc_id}


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
# ====================================================================== #
@app.get("/api/graph/{node_id}")
def graph_neighbors(node_id: str, direction: str = "both", edge_type: str | None = None):
    """图邻接遍历。"""
    return {"node_id": node_id, "neighbors": wr.neighbors(node_id, direction, edge_type)}


@app.get("/api/graph/{node_id}/subtree")
def graph_subtree(node_id: str, depth: int = 2):
    """获取以某节点为根的局部子图。"""
    return wr.graph_subtree(node_id, depth)


@app.get("/api/documents/{doc_id}/related")
def related_articles(doc_id: str):
    """获取与文档相关的文章。"""
    return {"related": wr.related_articles(doc_id)}


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


# ====================================================================== #
#  搜索（四路融合）
# ====================================================================== #
@app.get("/api/search")
def search(q: str, topk: int = 5):
    """四路融合检索。"""
    return wr.search_documents(q, topk=topk)


@app.post("/api/search")
def search_post(req: SearchRequest):
    """四路融合检索（POST）。"""
    return wr.search_documents(req.query, topk=req.topk)


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
        return {"compiled": compiled, "document": doc}
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


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
            ent_key = _re.sub(r"[^A-Za-z0-9_]", "_", ent["name"].lower()).strip("_")
            wr.ensure_entity(ent_key, ent["name"], ent.get("type"))
            wr.relate(rid, "mentions", f"entity:{ent_key}")
        for rel in result.get("relations", []):
            from_key = _re.sub(r"[^A-Za-z0-9_]", "_", rel["from"].lower()).strip("_")
            to_key = _re.sub(r"[^A-Za-z0-9_]", "_", rel["to"].lower()).strip("_")
            wr.relate(f"entity:{from_key}", "entity_related", f"entity:{to_key}")
        return {"extracted": result, "doc_id": rid}
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
