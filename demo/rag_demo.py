#!/usr/bin/env python3
"""
RAG 知识库演示程序
====================

通过 zvec REST Bridge 连接向量数据库，使用本地 Ollama 的
qwen3-embedding:4b 模型进行文本向量化，完整验证向量库的
RAG 流程：

    知识入库（文本→向量→存储） → 语义检索 → 生成回答

前置条件
--------
1. zvec REST Bridge 服务已启动（默认 http://localhost:8666）
2. Ollama 已运行且已拉取嵌入模型：
       ollama pull qwen3-embedding:4b
3. 服务端已安装 openai 依赖（Ollama 兼容 OpenAI 接口）：
       pip install openai

运行方式
--------
    python rag_demo.py

可通过环境变量覆盖默认地址：
    ZVEC_URL=http://localhost:8666 OLLAMA_URL=http://localhost:11434 python rag_demo.py
"""
from __future__ import annotations

import os
import sys
import time

import requests

# ====================================================================== #
#  配置（可通过环境变量覆盖）
# ====================================================================== #
ZVEC_URL = os.environ.get("ZVEC_URL", "http://localhost:8666")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:4b")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")  # 生成回答用，可选

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

# ====================================================================== #
#  工具函数
# ====================================================================== #
_passed = 0
_failed = 0


def check(label: str, cond: bool, extra: str = "") -> None:
    """断言一个条件，打印通过/失败。"""
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✅ {label}")
    else:
        _failed += 1
        print(f"  ❌ {label}  {extra}")


def zvec(method: str, path: str, **kwargs) -> requests.Response:
    """调用 zvec REST API。"""
    return requests.request(method, f"{ZVEC_URL}{path}", **kwargs)


def section(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


# ====================================================================== #
#  步骤 1：健康检查
# ====================================================================== #
def step_health_check() -> bool:
    section("步骤 1  健康检查")
    # 检查 zvec REST Bridge
    try:
        r = zvec("GET", "/health", timeout=5)
    except requests.ConnectionError:
        print(f"  ❌ 无法连接 zvec REST Bridge ({ZVEC_URL})，请确认服务已启动。")
        return False
    check("zvec 服务可达", r.status_code == 200, f"HTTP {r.status_code}")
    if r.status_code == 200:
        info = r.json()
        check("zvec 引擎在线", info.get("status") == "UP")
        print(f"     zvec 版本: {info.get('zvec_version', 'N/A')}")

    # 检查 Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        check("Ollama 服务可达", r.status_code == 200, f"HTTP {r.status_code}")
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            has_embed = any(EMBED_MODEL in m for m in models)
            check(f"Ollama 已安装 {EMBED_MODEL}", has_embed,
                  f"可用模型: {models}")
            if not has_embed:
                print(f"     提示: 运行  ollama pull {EMBED_MODEL}  拉取模型")
    except requests.ConnectionError:
        print(f"  ❌ 无法连接 Ollama ({OLLAMA_URL})，请确认 Ollama 已启动。")
        return False

    return True


# ====================================================================== #
#  步骤 2：发现向量维度（直接调用 Ollama）
# ====================================================================== #
def step_discover_dimension() -> int | None:
    section("步骤 2  发现向量维度")
    # 直接调用 Ollama 的 embed 接口获取模型输出维度，
    # 这样在注册嵌入函数时就能传入正确的 dimension。
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL,
        "input": "维度探测样本",
    }, timeout=30)
    check("Ollama embed 调用成功", r.status_code == 200, f"({r.status_code}) {r.text[:200]}")
    if r.status_code != 200:
        return None

    embeddings = r.json().get("embeddings", [])
    dim = len(embeddings[0]) if embeddings else 0
    check("返回非空向量", dim > 0)
    print(f"     {EMBED_MODEL} 输出维度 = {dim}")
    return dim


# ====================================================================== #
#  步骤 3：注册嵌入函数（指向 Ollama，携带正确维度）
# ====================================================================== #
def step_register_embedding(dimension: int) -> bool:
    section("步骤 3  注册嵌入函数")
    # 若已存在则先删除，保证可重复运行
    zvec("DELETE", f"/embeddings/{EMBED_FUNC_NAME}")

    r = zvec("POST", "/embeddings", json={
        "name": EMBED_FUNC_NAME,
        "type": "openai",                       # Ollama 兼容 OpenAI 接口
        "model": EMBED_MODEL,
        "dimension": dimension,                 # 必须与模型实际输出维度一致
        "base_url": f"{OLLAMA_URL}/v1",         # Ollama 的 OpenAI 兼容端点
        "api_key": "ollama",                    # Ollama 不校验 key，填任意值即可
    })
    check("注册嵌入函数", r.status_code in (200, 201), f"({r.status_code}) {r.text[:200]}")
    if r.status_code not in (200, 201):
        print("     提示: 服务端需安装 openai 依赖  →  pip install openai")
        return False

    info = r.json()
    check("函数名正确", info.get("name") == EMBED_FUNC_NAME)
    check("类型为 openai", info.get("type") == "openai")

    # 通过 zvec 转发验证嵌入通路是否打通
    r = zvec("POST", f"/embeddings/{EMBED_FUNC_NAME}/embed", json={
        "texts": ["验证嵌入通路"],
    })
    check("zvec 嵌入通路验证", r.status_code == 200, f"({r.status_code}) {r.text[:200]}")
    return True


# ====================================================================== #
#  步骤 4：创建集合
# ====================================================================== #
def step_create_collection(dimension: int) -> bool:
    section("步骤 4  创建向量集合")
    # 若已存在则先删除，保证可重复运行
    zvec("DELETE", f"/collections/{COLLECTION_NAME}")

    r = zvec("POST", "/collections", json={
        "schema": {
            "name": COLLECTION_NAME,
            "fields": [
                {"name": "title", "data_type": "STRING", "nullable": True},
                {"name": "content", "data_type": "STRING", "nullable": True},
            ],
            "vectors": [
                {
                    "name": VECTOR_FIELD,
                    "data_type": "VECTOR_FP32",
                    "dimension": dimension,
                    # FLAT 暴力检索，适合小规模演示；度量用 IP（内积）
                    "index_param": {"type": "FLAT", "metric_type": "IP"},
                },
            ],
        },
    })
    check("创建集合", r.status_code in (200, 201), f"({r.status_code}) {r.text[:200]}")
    return r.status_code in (200, 201)


# ====================================================================== #
#  步骤 5：知识入库（自动嵌入）
# ====================================================================== #
def step_ingest_corpus() -> bool:
    section("步骤 5  知识入库（文本自动嵌入）")
    documents = [
        {
            "id": doc["id"],
            "text": f"{doc['title']}。{doc['content']}",  # 嵌入用文本
            "fields": {"title": doc["title"], "content": doc["content"]},
        }
        for doc in CORPUS
    ]

    r = zvec("POST", f"/collections/{COLLECTION_NAME}/documents", json={
        "embedding": {"function": EMBED_FUNC_NAME, "field": VECTOR_FIELD},
        "documents": documents,
    })
    check("批量插入", r.status_code == 200, f"({r.status_code}) {r.text[:200]}")
    if r.status_code != 200:
        return False

    body = r.json()
    count = body.get("count", 0)
    all_ok = all(x["status"]["ok"] for x in body.get("results", []))
    check(f"插入 {len(CORPUS)} 篇文档", count == len(CORPUS) and all_ok)

    # 验证向量确实被填充
    r = zvec("POST", f"/collections/{COLLECTION_NAME}/documents:fetch", json={
        "ids": [CORPUS[0]["id"]], "include_vector": True,
    })
    if r.status_code == 200:
        vec = r.json()["documents"][0].get("vectors", {}).get(VECTOR_FIELD)
        check("向量已自动填充", vec is not None and len(vec) > 0)
    return True


# ====================================================================== #
#  步骤 6：语义检索验证
# ====================================================================== #
def step_semantic_search() -> None:
    section("步骤 6  语义检索验证")
    print(f"  共 {len(TEST_QUERIES)} 个测试查询\n")

    for query, expected_id in TEST_QUERIES:
        r = zvec("POST", f"/collections/{COLLECTION_NAME}/search", json={
            "embedding": {"function": EMBED_FUNC_NAME, "field": VECTOR_FIELD},
            "queries": [{"field_name": VECTOR_FIELD, "text": query}],
            "topk": 3,
            "output_fields": ["title", "content"],
        })
        if r.status_code != 200:
            check(f"「{query}」", False, f"({r.status_code}) {r.text[:150]}")
            continue

        docs = r.json().get("documents", [])
        top = docs[0] if docs else {}
        top_id = top.get("id", "")
        top_score = top.get("score", 0)
        top_title = top.get("fields", {}).get("title", "")

        hit = top_id == expected_id
        status = "✅" if hit else "❌"
        print(f"  {status} 查询: {query}")
        print(f"     Top1: [{top_id}] {top_title}  (score={top_score:.4f})")
        if not hit:
            print(f"     期望: {expected_id}")
        check(f"「{query}」命中预期文档", hit)

    # 额外：验证 fetch 取回标量字段
    r = zvec("POST", f"/collections/{COLLECTION_NAME}/documents:fetch", json={
        "ids": [CORPUS[0]["id"]], "output_fields": ["title", "content"],
        "include_vector": False,
    })
    if r.status_code == 200:
        doc = r.json()["documents"][0]
        check("fetch 取回标量字段", doc["fields"].get("title") == CORPUS[0]["title"])


# ====================================================================== #
#  步骤 7：RAG 生成回答（可选）
# ====================================================================== #
def step_rag_generate() -> None:
    section("步骤 7  RAG 端到端问答（检索 + 生成）")
    question = "向量数据库和传统数据库有什么区别？RAG 又是什么？"

    # 1) 检索相关文档
    r = zvec("POST", f"/collections/{COLLECTION_NAME}/search", json={
        "embedding": {"function": EMBED_FUNC_NAME, "field": VECTOR_FIELD},
        "queries": [{"field_name": VECTOR_FIELD, "text": question}],
        "topk": 3,
        "output_fields": ["title", "content"],
    })
    if r.status_code != 200:
        print(f"  ⚠ 检索失败，跳过生成: ({r.status_code}) {r.text[:150]}")
        return

    docs = r.json().get("documents", [])
    context = "\n\n".join(
        f"[{d['id']}] {d['fields'].get('title', '')}\n{d['fields'].get('content', '')}"
        for d in docs
    )
    print(f"  问题: {question}")
    print(f"  检索到 {len(docs)} 篇相关文档:")
    for d in docs:
        print(f"    • [{d['id']}] {d['fields'].get('title', '')}  (score={d.get('score', 0):.4f})")

    # 2) 调用 Ollama 生成回答
    prompt = (
        f"请根据以下检索到的知识回答问题。如果知识中没有相关信息，请说明。\n\n"
        f"知识：\n{context}\n\n"
        f"问题：{question}\n\n"
        f"回答："
    )
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }, timeout=60)
        if r.status_code == 200:
            answer = r.json()["message"]["content"].strip()
            check("Ollama 生成回答", len(answer) > 0)
            print(f"\n  📝 回答:\n  {'-' * 54}")
            for line in answer.splitlines():
                print(f"  {line}")
            print(f"  {'-' * 54}")
        else:
            print(f"  ⚠ Ollama 生成失败: ({r.status_code}) {r.text[:150]}")
            print(f"     提示: 运行  ollama pull {LLM_MODEL}  拉取生成模型")
    except requests.ConnectionError:
        print("  ⚠ 无法连接 Ollama，跳过生成步骤。")
    except Exception as exc:
        print(f"  ⚠ 生成异常: {exc}")


# ====================================================================== #
#  步骤 8：清理
# ====================================================================== #
def step_cleanup() -> None:
    section("步骤 8  清理资源")
    r = zvec("DELETE", f"/collections/{COLLECTION_NAME}")
    check("删除集合", r.status_code in (200, 204, 404), f"({r.status_code})")
    r = zvec("DELETE", f"/embeddings/{EMBED_FUNC_NAME}")
    check("注销嵌入函数", r.status_code in (200, 204, 404), f"({r.status_code})")


# ====================================================================== #
#  主函数
# ====================================================================== #
def main() -> int:
    print("╔" + "═" * 60 + "╗")
    print("║" + " RAG 知识库演示 — zvec REST Bridge + Ollama ".center(54) + "║")
    print("╚" + "═" * 60 + "╝")
    print(f"  zvec 服务 : {ZVEC_URL}")
    print(f"  Ollama    : {OLLAMA_URL}")
    print(f"  嵌入模型  : {EMBED_MODEL}")
    print(f"  生成模型  : {LLM_MODEL}")

    if not step_health_check():
        print("\n❌ 前置条件不满足，请检查服务状态后重试。")
        return 1

    dimension = step_discover_dimension()
    if not dimension:
        return 1

    if not step_register_embedding(dimension):
        return 1

    if not step_create_collection(dimension):
        return 1

    step_ingest_corpus()
    step_semantic_search()
    step_rag_generate()
    step_cleanup()

    print(f"\n{'━' * 60}")
    print(f"  验证完成:  通过 {_passed} 项, 失败 {_failed} 项")
    print(f"{'━' * 60}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
