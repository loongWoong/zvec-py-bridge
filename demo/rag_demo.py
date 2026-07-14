#!/usr/bin/env python3
"""
RAG 知识库演示程序（CLI 验证模式）
====================================

通过 zvec REST Bridge 连接向量数据库，使用本地 Ollama 的
qwen3-embedding:4b 模型进行文本向量化，完整验证向量库的
RAG 流程：知识入库 → 语义检索 → 生成回答。

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
    ZVEC_URL=http://localhost:8666 OLLAMA_URL=http://localhost:11434 \\
    LLM_URL=http://localhost:8000 LLM_API=openai LLM_API_KEY=sk-xxx \\
    python rag_demo.py

说明：
    LLM_URL   大模型服务地址（默认复用 OLLAMA_URL，可指向任意 OpenAI 兼容端点）
    LLM_API   接口格式：ollama（默认，/api/chat）或 openai（/v1/chat/completions）
    LLM_API_KEY  访问大模型服务所需密钥（OpenAI 兼容端点通常需要）
"""
from __future__ import annotations

import sys

from kb_data import (
    COLLECTION_NAME,
    CORPUS,
    EMBED_FUNC_NAME,
    EMBED_MODEL,
    KBError,
    LLM_MODEL,
    LLM_URL,
    LLM_API,
    OLLAMA_URL,
    TEST_QUERIES,
    ZVEC_URL,
    check_health,
    cleanup,
    discover_dimension,
    init_knowledge_base,
    rag_ask,
    register_embedding,
    search,
)

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


def section(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


# ====================================================================== #
#  步骤 1：健康检查
# ====================================================================== #
def step_health_check() -> bool:
    section("步骤 1  健康检查")
    health = check_health()
    check("zvec 服务可达", health["zvec"])
    if health["zvec"]:
        check("zvec 引擎在线", True)
        print(f"     zvec 版本: {health['zvec_version']}")
    check("Ollama 服务可达", health["ollama"])
    check(f"Ollama 已安装 {EMBED_MODEL}", health["has_embed_model"])
    if not health["has_embed_model"]:
        print(f"     提示: 运行  ollama pull {EMBED_MODEL}  拉取模型")
    check(f"LLM 服务可达 ({LLM_API})", health["llm"])
    if not health["llm"]:
        print(f"     提示: 检查 LLM_URL={LLM_URL} 与 LLM_API={LLM_API} 是否正确")
    return health["zvec"] and health["has_embed_model"]


# ====================================================================== #
#  步骤 2：发现向量维度
# ====================================================================== #
def step_discover_dimension() -> int | None:
    section("步骤 2  发现向量维度")
    try:
        dim = discover_dimension()
        check("Ollama embed 调用成功", True)
        check("返回非空向量", dim > 0)
        print(f"     {EMBED_MODEL} 输出维度 = {dim}")
        return dim
    except KBError as e:
        check("Ollama embed 调用成功", False, str(e))
        return None


# ====================================================================== #
#  步骤 3：注册嵌入函数（携带正确维度）
# ====================================================================== #
def step_register_embedding(dimension: int) -> bool:
    section("步骤 3  注册嵌入函数")
    try:
        register_embedding(dimension)
        check("注册嵌入函数", True)
        check("函数名正确", True)
        check("类型为 openai", True)
    except KBError as e:
        check("注册嵌入函数", False, str(e))
        print("     提示: 服务端需安装 openai 依赖  →  pip install openai")
        return False
    return True


# ====================================================================== #
#  步骤 4：创建集合
# ====================================================================== #
def step_create_collection(dimension: int) -> bool:
    section("步骤 4  创建向量集合")
    from kb_data import create_collection
    try:
        create_collection(dimension)
        check("创建集合", True)
    except KBError as e:
        check("创建集合", False, str(e))
        return False
    return True


# ====================================================================== #
#  步骤 5：知识入库（自动嵌入）
# ====================================================================== #
def step_ingest_corpus() -> bool:
    section("步骤 5  知识入库（文本自动嵌入）")
    from kb_data import ingest_corpus
    try:
        count = ingest_corpus()
        check(f"插入 {len(CORPUS)} 篇文档", count == len(CORPUS))
    except KBError as e:
        check("批量插入", False, str(e))
        return False

    # 验证向量确实被填充
    from kb_data import zvec_api
    r = zvec_api("POST", f"/collections/{COLLECTION_NAME}/documents:fetch", json={
        "ids": [CORPUS[0]["id"]], "include_vector": True,
    })
    if r.status_code == 200:
        vec = r.json()["documents"][0].get("vectors", {}).get("embedding")
        check("向量已自动填充", vec is not None and len(vec) > 0)
    return True


# ====================================================================== #
#  步骤 6：语义检索验证
# ====================================================================== #
def step_semantic_search() -> None:
    section("步骤 6  语义检索验证")
    print(f"  共 {len(TEST_QUERIES)} 个测试查询\n")

    for query, expected_id in TEST_QUERIES:
        try:
            docs = search(query, topk=3)
        except KBError as e:
            check(f"「{query}」", False, str(e))
            continue

        top = docs[0] if docs else {}
        top_id = top.get("id", "")
        top_score = top.get("score", 0)
        top_title = top.get("title", "")

        hit = top_id == expected_id
        status = "✅" if hit else "❌"
        print(f"  {status} 查询: {query}")
        print(f"     Top1: [{top_id}] {top_title}  (score={top_score:.4f})")
        if not hit:
            print(f"     期望: {expected_id}")
        check(f"「{query}」命中预期文档", hit)

    # 额外：验证 fetch 取回标量字段
    from kb_data import zvec_api
    r = zvec_api("POST", f"/collections/{COLLECTION_NAME}/documents:fetch", json={
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
    print(f"  问题: {question}")

    try:
        result = rag_ask(question, topk=3)
    except KBError as e:
        print(f"  ⚠ {e}")
        if "Ollama 生成失败" in str(e):
            print(f"     提示: 运行  ollama pull {LLM_MODEL}  拉取生成模型")
        return

    docs = result["documents"]
    answer = result["answer"]
    print(f"  检索到 {len(docs)} 篇相关文档:")
    for d in docs:
        print(f"    • [{d['id']}] {d['title']}  (score={d['score']:.4f})")
    check("Ollama 生成回答", len(answer) > 0)
    print(f"\n  📝 回答:\n  {'-' * 54}")
    for line in answer.splitlines():
        print(f"  {line}")
    print(f"  {'-' * 54}")


# ====================================================================== #
#  步骤 8：清理
# ====================================================================== #
def step_cleanup() -> None:
    section("步骤 8  清理资源")
    try:
        cleanup()
        check("删除集合", True)
        check("注销嵌入函数", True)
    except KBError as e:
        check("清理资源", False, str(e))


# ====================================================================== #
#  主函数
# ====================================================================== #
def main() -> int:
    print("╔" + "═" * 60 + "╗")
    print("║" + " RAG 知识库演示 — zvec REST Bridge + Ollama ".center(54) + "║")
    print("╚" + "═" * 60 + "╝")
    print(f"  zvec 服务 : {ZVEC_URL}")
    print(f"  Ollama    : {OLLAMA_URL}")
    print(f"  LLM 服务  : {LLM_URL} ({LLM_API})")
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
