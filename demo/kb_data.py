"""RAG 知识库共享模块 — 配置、语料、核心操作函数。

供 rag_demo.py（CLI 验证）和 web_app.py（Web 界面）共用，
确保两端的配置与行为完全一致。
"""
from __future__ import annotations

import os
import requests

# ====================================================================== #
#  配置（可通过环境变量覆盖）
# ====================================================================== #
ZVEC_URL = os.environ.get("ZVEC_URL", "http://localhost:8666")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:4b")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")  # 生成回答用，可选
OCR_MODEL = os.environ.get("OCR_MODEL", "glm-ocr")    # OCR 识别用

COLLECTION_NAME = "rag_demo"
EMBED_FUNC_NAME = "ollama_qwen3"
VECTOR_FIELD = "embedding"

# ====================================================================== #
#  知识库语料
# ====================================================================== #
CORPUS = [
    {
        "id": "doc_01",
        "title": "什么是向量数据库",
        "content": (
            "向量数据库是一种专门用于存储和检索高维向量的数据库系统。"
            "它通过近似最近邻搜索（ANN）算法，在海量向量中快速找到与查询向量最相似的结果，"
            "广泛应用于语义搜索、推荐系统和检索增强生成（RAG）等场景。"
            "与传统关系型数据库不同，向量数据库的查询基于向量相似度而非精确匹配。"
        ),
    },
    {
        "id": "doc_02",
        "title": "文本嵌入模型",
        "content": (
            "文本嵌入模型将自然语言文本映射为固定维度的稠密向量，"
            "使语义相近的文本在向量空间中距离更近。"
            "常见的嵌入模型包括 OpenAI text-embedding 系列、BGE、Qwen3-Embedding 等，"
            "输出维度通常在 768 到 4096 之间。嵌入质量直接影响下游检索的效果。"
        ),
    },
    {
        "id": "doc_03",
        "title": "检索增强生成 RAG",
        "content": (
            "检索增强生成（Retrieval-Augmented Generation, RAG）是一种结合检索与生成的技术。"
            "其核心思路是：先从知识库中检索与用户问题相关的文档片段，"
            "再将这些片段作为上下文拼入提示词，交由大语言模型生成回答。"
            "RAG 能有效缓解大模型的幻觉问题，并支持基于私有知识的问答。"
        ),
    },
    {
        "id": "doc_04",
        "title": "HNSW 索引算法",
        "content": (
            "HNSW（分层可导航小世界图）是一种高效的近似最近邻搜索索引结构。"
            "它通过构建多层图来加速检索：上层稀疏图用于快速定位粗粒度区域，"
            "下层稠密图用于精细搜索。HNSW 在召回率和查询速度之间取得了良好的平衡，"
            "是向量数据库中最常用的索引之一。关键参数包括 M（图度数）和 ef_construction（建图候选集大小）。"
        ),
    },
    {
        "id": "doc_05",
        "title": "余弦相似度与距离度量",
        "content": (
            "余弦相似度通过测量两个向量夹角的余弦值来衡量相似程度，取值范围为 -1 到 1，"
            "值越大表示越相似。在向量检索中，当嵌入向量经过 L2 归一化后，"
            "余弦相似度等价于内积（Inner Product）。"
            "其他常用距离度量包括欧氏距离（L2）和曼哈顿距离（L1）。"
        ),
    },
    {
        "id": "doc_06",
        "title": "文档分块策略",
        "content": (
            "在构建 RAG 知识库时，需要将长文档切分为较小的文本块（chunk）再进行嵌入。"
            "常见的分块策略包括固定长度切分、按句子或段落切分、以及递归字符切分。"
            "分块大小通常在 200 到 1000 个 token 之间，需要在检索精度和上下文完整性之间权衡。"
            "适当的重叠（overlap）可以避免语义在切分边界处断裂。"
        ),
    },
    {
        "id": "doc_07",
        "title": "混合检索",
        "content": (
            "混合检索结合稠密向量检索与稀疏检索（如 BM25），取长补短。"
            "稠密检索擅长捕捉语义相似性，而稀疏检索基于关键词匹配，擅长精确匹配专有名词。"
            "两路结果可通过倒数排名融合（RRF）或加权融合进行合并，从而提升整体召回率。"
        ),
    },
    {
        "id": "doc_08",
        "title": "重排序 Reranking",
        "content": (
            "重排序是 RAG 流程中的可选环节：先用轻量检索器快速召回 Top-K 候选，"
            "再用更强大的交叉编码器（Cross-Encoder）对候选逐一打分并重新排序。"
            "这种两阶段策略在保证检索速度的同时，显著提升了最终结果的相关性。"
            "常用的重排序模型包括 BGE-Reranker、Cohere Rerank 等。"
        ),
    },
]

# 用于验证语义检索的测试查询（期望命中的文档 ID）
TEST_QUERIES = [
    ("向量数据库的定义和用途是什么？", "doc_01"),
    ("嵌入模型是怎么工作的？", "doc_02"),
    ("什么是检索增强生成技术？", "doc_03"),
    ("HNSW 索引的原理是什么？", "doc_04"),
    ("文档分块有哪些策略？", "doc_06"),
]

# 示例问题（Web UI 快捷按钮）
SAMPLE_QUESTIONS = [
    "向量数据库是什么？有什么用？",
    "RAG 技术的原理是什么？",
    "HNSW 索引是怎么工作的？",
    "文档分块有哪些常见策略？",
    "余弦相似度和内积有什么关系？",
]


# ====================================================================== #
#  异常
# ====================================================================== #
class KBError(Exception):
    """知识库操作异常。"""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


# ====================================================================== #
#  核心操作
# ====================================================================== #
def zvec_api(method: str, path: str, **kwargs) -> requests.Response:
    """调用 zvec REST API。"""
    return requests.request(method, f"{ZVEC_URL}{path}", **kwargs)


def check_health() -> dict:
    """检查 zvec 和 Ollama 服务状态。

    返回 {"zvec": bool, "zvec_version": str, "ollama": bool, "has_embed_model": bool}
    """
    result: dict = {"zvec": False, "zvec_version": "", "ollama": False, "has_embed_model": False}

    try:
        r = zvec_api("GET", "/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            result["zvec"] = info.get("status") == "UP"
            result["zvec_version"] = info.get("zvec_version", "")
    except requests.ConnectionError:
        pass

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            result["ollama"] = True
            models = [m["name"] for m in r.json().get("models", [])]
            result["has_embed_model"] = any(EMBED_MODEL in m for m in models)
            result["ollama_models"] = models
    except requests.ConnectionError:
        pass

    return result


def discover_dimension() -> int:
    """直接调用 Ollama embed 接口获取模型输出维度。"""
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL,
        "input": "维度探测样本",
    }, timeout=30)
    if r.status_code != 200:
        raise KBError(f"Ollama embed 失败: ({r.status_code}) {r.text[:200]}", r.status_code)
    embeddings = r.json().get("embeddings", [])
    dim = len(embeddings[0]) if embeddings else 0
    if dim == 0:
        raise KBError("Ollama 返回空向量")
    return dim


def register_embedding(dimension: int) -> None:
    """向 zvec 注册嵌入函数（指向 Ollama）。"""
    zvec_api("DELETE", f"/embeddings/{EMBED_FUNC_NAME}")  # 幂等：已存在则先删除
    r = zvec_api("POST", "/embeddings", json={
        "name": EMBED_FUNC_NAME,
        "type": "openai",
        "model": EMBED_MODEL,
        "dimension": dimension,
        "base_url": f"{OLLAMA_URL}/v1",
        "api_key": "ollama",
    })
    if r.status_code not in (200, 201):
        raise KBError(f"注册嵌入函数失败: ({r.status_code}) {r.text[:200]}", r.status_code)


def create_collection(dimension: int) -> None:
    """创建向量集合。"""
    zvec_api("DELETE", f"/collections/{COLLECTION_NAME}")  # 幂等
    r = zvec_api("POST", "/collections", json={
        "schema": {
            "name": COLLECTION_NAME,
            "fields": [
                {"name": "title", "data_type": "STRING", "nullable": True},
                {"name": "content", "data_type": "STRING", "nullable": True},
                {"name": "heading", "data_type": "STRING", "nullable": True},
                {"name": "document_id", "data_type": "STRING", "nullable": True},
            ],
            "vectors": [
                {
                    "name": VECTOR_FIELD,
                    "data_type": "VECTOR_FP32",
                    "dimension": dimension,
                    "index_param": {"type": "FLAT", "metric_type": "IP"},
                },
            ],
        },
    })
    if r.status_code not in (200, 201):
        raise KBError(f"创建集合失败: ({r.status_code}) {r.text[:200]}", r.status_code)


def ingest_documents(documents: list[dict]) -> int:
    """批量插入文档（文本自动嵌入为向量）。

    documents: [{id, text, fields:{title, content, heading, document_id}}, ...]
    返回插入数量。
    """
    r = zvec_api("POST", f"/collections/{COLLECTION_NAME}/documents", json={
        "embedding": {"function": EMBED_FUNC_NAME, "field": VECTOR_FIELD},
        "documents": documents,
    })
    if r.status_code != 200:
        raise KBError(f"插入文档失败: ({r.status_code}) {r.text[:200]}", r.status_code)
    body = r.json()
    count = body.get("count", 0)
    if count != len(documents):
        raise KBError(f"插入数量不匹配: 期望 {len(documents)}, 实际 {count}")
    return count


def ingest_corpus() -> int:
    """批量插入知识库语料（文本自动嵌入为向量）。"""
    documents = [
        {
            "id": doc["id"],
            "text": f"{doc['title']}。{doc['content']}",
            "fields": {
                "title": doc["title"],
                "content": doc["content"],
                "heading": doc["title"],
                "document_id": doc["id"],
            },
        }
        for doc in CORPUS
    ]
    return ingest_documents(documents)


def search(query: str, topk: int = 3) -> list[dict]:
    """语义检索：将查询文本嵌入为向量，返回 Top-K 相关文档。

    返回 [{id, score, title, content, heading, document_id}, ...]
    """
    r = zvec_api("POST", f"/collections/{COLLECTION_NAME}/search", json={
        "embedding": {"function": EMBED_FUNC_NAME, "field": VECTOR_FIELD},
        "queries": [{"field_name": VECTOR_FIELD, "text": query}],
        "topk": topk,
        "output_fields": ["title", "content", "heading", "document_id"],
    })
    if r.status_code != 200:
        raise KBError(f"检索失败: ({r.status_code}) {r.text[:200]}", r.status_code)
    docs = r.json().get("documents", [])
    return [
        {
            "id": d.get("id", ""),
            "score": d.get("score", 0),
            "title": d.get("fields", {}).get("title", ""),
            "content": d.get("fields", {}).get("content", ""),
            "heading": d.get("fields", {}).get("heading", ""),
            "document_id": d.get("fields", {}).get("document_id", ""),
        }
        for d in docs
    ]


def rag_ask(question: str, topk: int = 3) -> dict:
    """RAG 问答：检索相关文档 → 拼接上下文 → 调用 Ollama 生成回答。

    返回 {"documents": [...], "answer": str}
    """
    docs = search(question, topk=topk)
    context = "\n\n".join(
        f"[{d['id']}] {d['title']}\n{d['content']}" for d in docs
    )
    prompt = (
        f"请根据以下检索到的知识回答问题。如果知识中没有相关信息，请说明。\n\n"
        f"知识：\n{context}\n\n"
        f"问题：{question}\n\n"
        f"回答："
    )
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=120)
    if r.status_code != 200:
        raise KBError(
            f"Ollama 生成失败 (模型 {LLM_MODEL}): ({r.status_code}) {r.text[:200]}",
            r.status_code,
        )
    answer = r.json()["message"]["content"].strip()
    return {"documents": docs, "answer": answer}


def cleanup() -> None:
    """删除集合、注销嵌入函数。"""
    zvec_api("DELETE", f"/collections/{COLLECTION_NAME}")
    zvec_api("DELETE", f"/embeddings/{EMBED_FUNC_NAME}")


def init_knowledge_base() -> dict:
    """完整初始化：发现维度 → 注册嵌入 → 创建集合 → 入库。

    返回 {"dimension": int, "doc_count": int}
    """
    health = check_health()
    if not health["zvec"]:
        raise KBError("zvec REST Bridge 不可达，请确认服务已启动")
    if not health["has_embed_model"]:
        raise KBError(f"Ollama 未安装 {EMBED_MODEL}，请运行 ollama pull {EMBED_MODEL}")

    dimension = discover_dimension()
    register_embedding(dimension)
    create_collection(dimension)
    doc_count = ingest_corpus()
    return {"dimension": dimension, "doc_count": doc_count}
