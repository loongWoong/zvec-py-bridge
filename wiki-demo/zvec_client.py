"""zvec REST Bridge 客户端 — 向量语义检索。

精简自 demo/kb_data.py，只保留 wiki 需要的能力：
注册嵌入函数 → 建集合 → chunk 级入库 → 语义检索。

zvec 负责"四路融合检索"中的向量路；SurrealDB 负责图/全文/元数据三路。
"""
from __future__ import annotations

import requests

import config


class ZvecError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


# ====================================================================== #
#  底层封装
# ====================================================================== #
def _api(method: str, path: str, **kwargs) -> requests.Response:
    return requests.request(method, f"{config.ZVEC_URL}{path}", **kwargs)


# ====================================================================== #
#  健康检查与维度探测
# ====================================================================== #
def check_health() -> dict:
    """检查 zvec 与 Ollama 状态。

    返回 {"zvec": bool, "zvec_version": str, "ollama": bool, "has_embed_model": bool}
    """
    result: dict = {
        "zvec": False, "zvec_version": "",
        "ollama": False, "has_embed_model": False,
    }
    try:
        r = _api("GET", "/health", timeout=3)
        if r.status_code == 200:
            info = r.json()
            result["zvec"] = info.get("status") == "UP"
            result["zvec_version"] = info.get("zvec_version", "")
    except requests.RequestException:
        pass

    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            result["ollama"] = True
            models = [m["name"] for m in r.json().get("models", [])]
            result["has_embed_model"] = any(config.EMBED_MODEL in m for m in models)
    except requests.RequestException:
        pass

    return result


def discover_dimension() -> int:
    """调用 Ollama embed 接口探测模型输出维度。"""
    r = requests.post(f"{config.OLLAMA_URL}/api/embed", json={
        "model": config.EMBED_MODEL,
        "input": "维度探测样本",
    }, timeout=30)
    if r.status_code != 200:
        raise ZvecError(f"Ollama embed 失败: ({r.status_code}) {r.text[:200]}", r.status_code)
    embeddings = r.json().get("embeddings", [])
    dim = len(embeddings[0]) if embeddings else 0
    if dim == 0:
        raise ZvecError("Ollama 返回空向量")
    return dim


# ====================================================================== #
#  嵌入函数注册与集合创建（幂等）
# ====================================================================== #
def register_embedding(dimension: int) -> None:
    """向 zvec 注册嵌入函数（指向 Ollama）。幂等。"""
    _api("DELETE", f"/embeddings/{config.EMBED_FUNC_NAME}")
    r = _api("POST", "/embeddings", json={
        "name": config.EMBED_FUNC_NAME,
        "type": "openai",
        "model": config.EMBED_MODEL,
        "dimension": dimension,
        "base_url": f"{config.OLLAMA_URL}/v1",
        "api_key": "ollama",
    })
    if r.status_code not in (200, 201):
        raise ZvecError(f"注册嵌入函数失败: ({r.status_code}) {r.text[:200]}", r.status_code)


def create_collection(dimension: int) -> None:
    """创建向量集合。幂等：已存在则先删除。"""
    _api("DELETE", f"/collections/{config.COLLECTION_NAME}")
    r = _api("POST", "/collections", json={
        "schema": {
            "name": config.COLLECTION_NAME,
            "fields": [
                {"name": "document_id", "data_type": "STRING", "nullable": True},
                {"name": "title", "data_type": "STRING", "nullable": True},
                {"name": "heading", "data_type": "STRING", "nullable": True},
                {"name": "content", "data_type": "STRING", "nullable": True},
            ],
            "vectors": [
                {
                    "name": config.VECTOR_FIELD,
                    "data_type": "VECTOR_FP32",
                    "dimension": dimension,
                    "index_param": {"type": "FLAT", "metric_type": "IP"},
                },
            ],
        },
    })
    if r.status_code not in (200, 201):
        raise ZvecError(f"创建集合失败: ({r.status_code}) {r.text[:200]}", r.status_code)


# ====================================================================== #
#  入库与检索
# ====================================================================== #
def ingest_chunks(chunks: list[dict]) -> int:
    """批量插入文档分块（文本自动嵌入为向量）。

    chunks: [{id, text, fields:{document_id, title, heading, content}}, ...]
    返回插入数量。
    """
    if not chunks:
        return 0
    r = _api("POST", f"/collections/{config.COLLECTION_NAME}/documents", json={
        "embedding": {"function": config.EMBED_FUNC_NAME, "field": config.VECTOR_FIELD},
        "documents": chunks,
    })
    if r.status_code != 200:
        raise ZvecError(f"插入分块失败: ({r.status_code}) {r.text[:200]}", r.status_code)
    return r.json().get("count", 0)


def search(query: str, topk: int = 5) -> list[dict]:
    """语义检索：返回 Top-K 相关分块。

    返回 [{id, score, document_id, title, heading, content}, ...]
    """
    r = _api("POST", f"/collections/{config.COLLECTION_NAME}/search", json={
        "embedding": {"function": config.EMBED_FUNC_NAME, "field": config.VECTOR_FIELD},
        "queries": [{"field_name": config.VECTOR_FIELD, "text": query}],
        "topk": topk,
        "output_fields": ["document_id", "title", "heading", "content", "concept_ids"],
    })
    if r.status_code != 200:
        raise ZvecError(f"检索失败: ({r.status_code}) {r.text[:200]}", r.status_code)
    docs = r.json().get("documents", [])
    return [
        {
            "id": d.get("id", ""),
            "score": d.get("score", 0),
            "document_id": d.get("fields", {}).get("document_id", ""),
            "title": d.get("fields", {}).get("title", ""),
            "heading": d.get("fields", {}).get("heading", ""),
            "content": d.get("fields", {}).get("content", ""),
        }
        for d in docs
    ]


def cleanup() -> None:
    """删除集合、注销嵌入函数。"""
    _api("DELETE", f"/collections/{config.COLLECTION_NAME}")
    _api("DELETE", f"/embeddings/{config.EMBED_FUNC_NAME}")


def delete_by_document_id(document_id: str) -> int:
    """删除指定文档的所有向量分块。

    通过 filter 表达式匹配 document_id 字段批量删除。
    返回删除数量。
    """
    r = _api("POST", f"/collections/{config.COLLECTION_NAME}/documents/delete", json={
        "filter": f"document_id == '{document_id}'",
    })
    if r.status_code != 200:
        # 可能接口不支持 filter delete，尝试逐个删除
        return _delete_by_document_id_fallback(document_id)
    return r.json().get("count", 0)


def _delete_by_document_id_fallback(document_id: str) -> int:
    """fallback：先检索再逐个 ID 删除。"""
    deleted = 0
    try:
        # 查询该文档的所有 chunk
        r = _api("POST", f"/collections/{config.COLLECTION_NAME}/search", json={
            "embedding": {"function": config.EMBED_FUNC_NAME, "field": config.VECTOR_FIELD},
            "queries": [{"field_name": config.VECTOR_FIELD, "text": "dummy"}],
            "topk": 1000,
            "output_fields": ["document_id"],
        })
        if r.status_code == 200:
            docs = r.json().get("documents", [])
            for d in docs:
                if d.get("fields", {}).get("document_id") == document_id:
                    chunk_id = d.get("id", "")
                    if chunk_id:
                        _api("DELETE", f"/collections/{config.COLLECTION_NAME}/documents/{chunk_id}")
                        deleted += 1
    except Exception:
        pass
    return deleted
