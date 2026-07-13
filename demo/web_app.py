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


@app.on_event("startup")
def startup():
    """启动时检查是否已有持久化的知识库，避免每次重启都重新初始化。"""
    os.makedirs(agent.UPLOADS_DIR, exist_ok=True)
    extra = agent.load_state()
    if extra is not None:
        # 本体已从磁盘恢复，验证 zvec 集合是否仍然存在
        try:
            r = kb.zvec_api("GET", f"/collections/{kb.COLLECTION_NAME}", timeout=5)
            if r.status_code == 200:
                # 集合存在，但嵌入函数注册是内存态（zvec 重启后丢失），需重新注册
                dimension = extra.get("dimension", 2560)
                try:
                    kb.register_embedding(dimension)
                    print(f"  嵌入函数已重新注册 (dimension={dimension})")
                except Exception as e:
                    print(f"  ⚠ 嵌入函数注册失败: {e}，请检查 Ollama 是否运行")
                _state["initialized"] = True
                _state["dimension"] = dimension
                _state["doc_count"] = len(agent.DOCUMENTS)
                _state["upload_count"] = extra.get("upload_count", 0)
                print(f"  知识库已从持久化状态恢复：{len(agent.DOCUMENTS)} 篇文档")
                return
        except Exception:
            pass
        # zvec 集合不存在，清除过期状态
        agent.clear_state()
        agent.reset()
        print("  持久化状态已失效（zvec 集合不存在），需要重新初始化")
    else:
        print("  未找到持久化状态，需要初始化知识库")


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
                "preview": d["content"][:100] + "..." if len(d["content"]) > 100 else d["content"],
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
            "preview": d["content"][:100] + "..." if len(d["content"]) > 100 else d["content"],
        }
        for d in kb.CORPUS
    ]}


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    """返回单篇文档的完整内容（含分块）。"""
    doc = agent.get_document(document_id)
    if doc:
        return {
            "document_id": doc["document_id"],
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "char_count": doc["char_count"],
            "source_file": doc.get("source_file", ""),
            "source_path": doc.get("source_path", ""),
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "heading": c["heading"],
                    "content": c["content"],
                    "ordinal": c["ordinal"],
                }
                for c in doc["chunks"]
            ],
        }
    # 内置语料 fallback
    for d in kb.CORPUS:
        if d["id"] == document_id:
            return {
                "document_id": d["id"],
                "title": d["title"],
                "content": d["content"],
                "category": "RAG/向量数据库",
                "char_count": len(d["content"]),
                "source_file": "",
                "source_path": "",
                "chunks": [],
            }
    raise HTTPException(status_code=404, detail="文档不存在")


@app.get("/api/files/{document_id}")
def get_file(document_id: str):
    """返回上传的原始文件（图片/PDF/DOCX 等）。"""
    for fname in os.listdir(agent.UPLOADS_DIR):
        if fname.startswith(document_id + "_"):
            return FileResponse(
                os.path.join(agent.UPLOADS_DIR, fname),
                filename=fname.split("_", 1)[1] if "_" in fname else fname,
            )
    raise HTTPException(status_code=404, detail="原始文件不存在（可能是内置语料）")


@app.post("/api/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """上传文档（Markdown/DOCX/PDF/XLSX/图片），解析→切分→嵌入入库→更新本体。"""
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

            # 保存原始文件到 uploads/ 目录
            safe_name = file.filename.replace("/", "_").replace("\\", "_")
            saved_name = f"{doc_id}_{safe_name}"
            saved_path = os.path.join(agent.UPLOADS_DIR, saved_name)
            with open(saved_path, "wb") as f:
                f.write(content)
            parsed["source_path"] = saved_path

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
        agent.save_state({"dimension": _state["dimension"], "upload_count": _state["upload_count"]})

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
        agent.save_state({"dimension": result["dimension"], "upload_count": 0})
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
        agent.reset()
        agent.clear_state()
        # 清空 uploads 目录
        for fname in os.listdir(agent.UPLOADS_DIR):
            os.remove(os.path.join(agent.UPLOADS_DIR, fname))
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
