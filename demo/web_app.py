#!/usr/bin/env python3
"""
RAG 知识库 Web 应用
====================

提供可交互的 Web 界面，通过 zvec REST Bridge + Ollama qwen3-embedding:4b
实现完整的 RAG 知识库体验：知识入库 → 语义检索 → 生成回答。

启动方式
--------
    python web_app.py

浏览器访问 http://localhost:8080 即可使用。

前置条件
--------
1. zvec REST Bridge 服务已启动（默认 http://localhost:8666）
2. Ollama 已运行且已拉取 qwen3-embedding:4b 模型
3. 服务端已安装 openai 依赖（pip install openai）
"""
from __future__ import annotations

import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保能 import 同目录下的 kb_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kb_data as kb  # noqa: E402
import agent  # noqa: E402
import document_loader  # noqa: E402

# ====================================================================== #
#  应用
# ====================================================================== #
app = FastAPI(title="RAG 知识库 Web 演示", version="1.0.0")

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(DEMO_DIR, "static")

# 运行时状态
_state: dict = {
    "initialized": False,
    "dimension": 0,
    "doc_count": 0,
    "upload_count": 0,
}


# ====================================================================== #
#  请求模型
# ====================================================================== #
class SearchRequest(BaseModel):
    query: str
    topk: int = 3


class AskRequest(BaseModel):
    query: str
    topk: int = 3


class AgentAskRequest(BaseModel):
    query: str
    max_iterations: int = 6


# ====================================================================== #
#  页面
# ====================================================================== #
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# 静态资源（如有额外 JS/CSS 文件）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ====================================================================== #
#  API
# ====================================================================== #
@app.get("/api/config")
def get_config():
    """返回当前配置信息（不含密钥）。"""
    return {
        "zvec_url": kb.ZVEC_URL,
        "ollama_url": kb.OLLAMA_URL,
        "embed_model": kb.EMBED_MODEL,
        "llm_model": kb.LLM_MODEL,
        "ocr_model": kb.OCR_MODEL,
        "collection": kb.COLLECTION_NAME,
        "corpus_size": len(kb.CORPUS),
        "sample_questions": kb.SAMPLE_QUESTIONS,
    }


@app.get("/api/status")
def get_status():
    """检查服务状态和知识库初始化情况。"""
    health = kb.check_health()
    return {
        "zvec": health["zvec"],
        "zvec_version": health["zvec_version"],
        "ollama": health["ollama"],
        "has_embed_model": health["has_embed_model"],
        "kb_initialized": _state["initialized"],
        "dimension": _state["dimension"],
        "doc_count": _state["doc_count"],
    }


@app.get("/api/corpus")
def get_corpus():
    """返回知识库语料列表。"""
    return {"documents": kb.CORPUS}


@app.get("/api/documents")
def get_documents():
    """返回所有文档（内置语料 + 上传文档）。"""
    if agent.DOCUMENTS:
        return {"documents": [
            {
                "id": d["document_id"],
                "title": d["title"],
                "category": d["category"],
                "char_count": d["char_count"],
                "chunk_count": len(d["chunks"]),
                "source_file": d.get("source_file", ""),
            }
            for d in agent.DOCUMENTS
        ]}
    # 未初始化时返回内置语料
    return {"documents": [
        {
            "id": d["id"],
            "title": d["title"],
            "category": "RAG/向量数据库",
            "char_count": len(d["content"]),
            "chunk_count": 0,
            "source_file": "",
        }
        for d in kb.CORPUS
    ]}


@app.post("/api/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """上传文档（Markdown/DOCX/PDF/XLSX），解析→切分→嵌入入库→更新本体。"""
    if not _state["initialized"]:
        raise HTTPException(status_code=400, detail="知识库未初始化，请先点击「初始化知识库」")

    results = []
    documents_to_ingest = []
    parsed_docs = []

    for file in files:
        content = await file.read()
        try:
            parsed = document_loader.parse_file(file.filename, content)
            doc_id = f"upload_{_state['upload_count'] + 1}"
            _state["upload_count"] += 1
            parsed["id"] = doc_id
            parsed_docs.append(parsed)

            # 按 chunk 准备入库文档
            for i, chunk in enumerate(parsed["chunks"]):
                chunk_id = f"{doc_id}_c{i + 1}"
                embed_text = f"{chunk['heading']}。{chunk['content']}" if chunk["heading"] else chunk["content"]
                documents_to_ingest.append({
                    "id": chunk_id,
                    "text": embed_text,
                    "fields": {
                        "title": parsed["title"],
                        "content": chunk["content"],
                        "heading": chunk["heading"],
                        "document_id": doc_id,
                    },
                })

            results.append({
                "document_id": doc_id,
                "title": parsed["title"],
                "source_file": file.filename,
                "source_type": parsed["source_type"],
                "chunk_count": len(parsed["chunks"]),
            })
        except Exception as e:
            results.append({
                "source_file": file.filename,
                "error": str(e),
            })

    # 批量入库 + 更新本体
    if documents_to_ingest:
        try:
            kb.ingest_documents(documents_to_ingest)
        except kb.KBError as e:
            raise HTTPException(status_code=400, detail=str(e))
        agent.add_documents(parsed_docs)

    _state["doc_count"] = len(agent.DOCUMENTS)
    return {"uploaded": results, "total_chunks": len(documents_to_ingest)}


@app.post("/api/init")
def init_kb():
    """初始化知识库：发现维度 → 注册嵌入 → 创建集合 → 入库。"""
    if _state["initialized"]:
        return {"dimension": _state["dimension"], "doc_count": _state["doc_count"],
                "message": "知识库已初始化"}
    try:
        result = kb.init_knowledge_base()
        agent.build_ontology()
        _state["initialized"] = True
        _state["dimension"] = result["dimension"]
        _state["doc_count"] = result["doc_count"]
        _state["upload_count"] = 0
        return result
    except kb.KBError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
def api_search(req: SearchRequest):
    """语义检索：查询文本 → 嵌入 → Top-K 检索。"""
    if not _state["initialized"]:
        raise HTTPException(status_code=400, detail="知识库未初始化，请先点击「初始化知识库」")
    try:
        docs = kb.search(req.query, topk=req.topk)
        return {"query": req.query, "topk": req.topk, "documents": docs}
    except kb.KBError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
def api_ask(req: AskRequest):
    """RAG 问答：检索 + 生成。"""
    if not _state["initialized"]:
        raise HTTPException(status_code=400, detail="知识库未初始化，请先点击「初始化知识库」")
    try:
        result = kb.rag_ask(req.query, topk=req.topk)
        return {"query": req.query, "topk": req.topk,
                "documents": result["documents"], "answer": result["answer"]}
    except kb.KBError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/ask")
def api_agent_ask(req: AgentAskRequest):
    """OAG Agent 问答：LLM 自主选择工具 → runtime 执行 → 整合结果。"""
    if not _state["initialized"]:
        raise HTTPException(status_code=400, detail="知识库未初始化，请先点击「初始化知识库」")
    try:
        result = agent.run_agent(req.query, max_iterations=req.max_iterations)
        return {"query": req.query, **result}
    except kb.KBError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cleanup")
def api_cleanup():
    """清理知识库资源。"""
    try:
        kb.cleanup()
        _state["initialized"] = False
        _state["dimension"] = 0
        _state["doc_count"] = 0
        _state["upload_count"] = 0
        return {"message": "清理完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================== #
#  启动
# ====================================================================== #
if __name__ == "__main__":
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "8080"))
    print(f"RAG 知识库 Web 应用启动中...")
    print(f"  访问地址: http://localhost:{port}")
    print(f"  zvec 服务: {kb.ZVEC_URL}")
    print(f"  Ollama   : {kb.OLLAMA_URL}")
    uvicorn.run(app, host=host, port=port)
