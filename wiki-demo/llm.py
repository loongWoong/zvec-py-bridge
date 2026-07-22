"""LLM 客户端 — 三职责：编译文档 / 抽取实体 / Agent 问答。

复用 demo/agent.py 的 LLM 调用模式（Ollama 原生 + OpenAI 兼容双格式），
通过 Wiki Runtime Tool 操作知识对象，对应 design.md 第十一节。
"""
from __future__ import annotations

import json
import re
import time

import requests

import config
import wiki_runtime as wr

# 可选导入：本体相关模块（需 surrealdb）。缺失时 use_ontology/use_rerank 自动降级。
try:
    import concept_locator
    import ontology_traversal
    import reranker
    _ONTOLOGY_DEPS = True
except ImportError:
    concept_locator = None     # type: ignore
    ontology_traversal = None  # type: ignore
    reranker = None            # type: ignore
    _ONTOLOGY_DEPS = False

# ====================================================================== #
#  底层 LLM 调用（复用 demo/agent.py 模式）
# ====================================================================== #
def _build_llm_request(messages: list[dict], tools: list[dict] | None = None):
    """根据 config.LLM_API 构造请求，返回 (url, json_body, headers)。"""
    body: dict = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    if config.LLM_API == "openai":
        headers = (
            {"Authorization": f"Bearer {config.LLM_API_KEY}"}
            if config.LLM_API_KEY else {}
        )
        return f"{config.LLM_URL}/v1/chat/completions", body, headers
    # 默认 Ollama 原生格式
    return f"{config.LLM_URL}/api/chat", body, {}


def _parse_llm_response(data: dict):
    """解析响应，返回 (content, tool_calls)。"""
    if config.LLM_API == "openai":
        msg = data["choices"][0]["message"]
    else:
        msg = data["message"]

    content = msg.get("content") or ""
    raw_calls = msg.get("tool_calls") or []
    tool_calls = []
    for idx, tc in enumerate(raw_calls):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        tool_calls.append({
            "id": tc.get("id") or f"call_{idx}",
            "name": fn.get("name", ""),
            "arguments": args,
        })
    return content, tool_calls


def _to_provider_messages(messages: list[dict]) -> list[dict]:
    """将归一化消息转换为当前 LLM_API 所需的请求格式。"""
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            if config.LLM_API == "openai":
                out.append({
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                            },
                        }
                        for tc in m["tool_calls"]
                    ],
                })
            else:
                tool_calls_out = []
                for tc in m["tool_calls"]:
                    item = {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    if tc.get("id"):
                        item["id"] = tc["id"]
                    tool_calls_out.append(item)
                out.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": tool_calls_out,
                })
        elif role == "tool":
            if config.LLM_API == "openai":
                out.append({
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                })
            else:
                out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def _call_llm(messages: list[dict], tools: list[dict] | None = None,
              temperature: float = 0.7) -> dict:
    """调用 LLM，返回原始 JSON 响应（失败抛 LLMError）。"""
    url, body, headers = _build_llm_request(messages, tools)
    body["temperature"] = temperature
    r = requests.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT)
    if r.status_code != 200:
        raise LLMError(
            f"LLM 调用失败 (模型 {config.LLM_MODEL}): ({r.status_code}) {r.text[:200]}"
        )
    return r.json()


def _clean_answer(text: str) -> str:
    """清理最终回答：去除 <think>...</think> 推理块。"""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON（可能被 ```json 包裹或前后有文字）。"""
    # 先尝试提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (ValueError, TypeError):
            pass
    # 再尝试直接解析整个文本中的 JSON 对象
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (ValueError, TypeError):
            pass
    return None


def _json_safe_dumps(obj) -> str:
    """JSON 序列化，安全处理 datetime 等不可序列化类型。"""
    def _default(o):
        from datetime import datetime as _dt
        if isinstance(o, _dt):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=_default)


class LLMError(Exception):
    pass


def _call_llm_safe(messages: list[dict], tools: list[dict] | None = None,
                   temperature: float = 0.7) -> dict:
    """调用 LLM，失败时抛 LLMError。"""
    url, body, headers = _build_llm_request(messages, tools)
    body["temperature"] = temperature
    r = requests.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT)
    if r.status_code != 200:
        raise LLMError(
            f"LLM 调用失败 (模型 {config.LLM_MODEL}): ({r.status_code}) {r.text[:200]}"
        )
    return r.json()


# ====================================================================== #
#  职责一：编译文档（raw → WikiDocument）
# ====================================================================== #
COMPILE_SYSTEM_PROMPT = """\
你是一个 Wiki 编译器。你的任务是将原始资料编译成结构化的 Wiki 文档。

请将给定的原始资料编译为一篇 Wiki 文档，输出 JSON 格式：
{
  "title": "文档标题",
  "summary": "一句话摘要（不超过50字）",
  "content": "Markdown 格式正文，使用 ## 作为章节标题",
  "entities": [{"name": "实体名", "type": "类型(mechanism/component/concept/method/algorithm)"}],
  "suggested_tags": ["标签1", "标签2"],
  "suggested_relations": [{"target_title": "目标文档标题", "type": "related|extends|depends|implements|contradicts|supersedes"}]
}

要求：
- content 必须是结构化的 Markdown，包含多个 ## 章节
- entities 提取文中提到的关键技术概念
- suggested_tags 建议合适的标签
- suggested_relations 建议与已有文档的关系（仅当确信目标存在时才建议）
- 只输出 JSON，不要输出其他内容
"""


def compile_document(raw_content: str, topic: str | None = None,
                    title_hint: str | None = None) -> dict:
    """从 raw 源编译生成 WikiDocument（含 summary + entities + tags + relations）。

    返回编译结果 dict（尚未落库）。
    """
    user_msg = f"原始资料：\n{raw_content}"
    if title_hint:
        user_msg = f"建议标题：{title_hint}\n\n{user_msg}"
    if topic:
        user_msg = f"所属主题：{topic}\n\n{user_msg}"

    messages = [
        {"role": "system", "content": COMPILE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    data = _call_llm_safe(messages, temperature=0.3)
    content, _ = _parse_llm_response(data)
    result = _extract_json(_clean_answer(content))
    if not result:
        raise LLMError(f"LLM 编译输出无法解析为 JSON: {content[:200]}")
    return result


# ====================================================================== #
#  职责二：抽取实体/关系
# ====================================================================== #
EXTRACT_SYSTEM_PROMPT = """\
你是一个知识图谱构建器。你的任务是从 Wiki 文档中抽取实体和关系。

请分析给定文档，输出 JSON 格式：
{
  "entities": [
    {"name": "实体名", "type": "类型(mechanism/component/concept/method/algorithm/metric/parameter/problem)"}
  ],
  "relations": [
    {"from": "实体A名", "to": "实体B名", "type": "related"}
  ]
}

要求：
- entities 提取文档中提到的所有关键技术概念、机制、组件
- relations 描述实体之间的关系（如 Attention related Encoder）
- 只输出 JSON，不要输出其他内容
"""


def extract_entities(document_id: str) -> dict:
    """对已有文档抽取实体/关系，返回抽取结果（尚未落库）。"""
    doc = wr.get_document(document_id)
    if not doc:
        raise LLMError(f"文档 {document_id} 不存在")

    user_msg = f"文档标题：{doc.get('title', '')}\n\n文档内容：\n{doc.get('content', '')}"
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    data = _call_llm_safe(messages, temperature=0.3)
    content, _ = _parse_llm_response(data)
    result = _extract_json(_clean_answer(content))
    if not result:
        raise LLMError(f"LLM 抽取输出无法解析为 JSON: {content[:200]}")
    return result


# ====================================================================== #
#  职责三：Agent 问答（通过 Wiki Runtime Tool）
# ====================================================================== #
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "四路融合检索文档（向量+全文+图+元数据）。"
                "适用于定位性问题：哪些文档提到 X、X 出现在哪里。"
                "返回结果含 doc_id、title、score、sources（来源路）、excerpt。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询文本"},
                    "topk": {"type": "integer", "description": "返回结果数量（默认5）", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "读取单篇文档全文及其关系邻接。用于核对出处、补充上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档 ID，如 document:transformer"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出知识库中所有文档的元数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "可选：按主题过滤"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_neighbors",
            "description": (
                "图邻接遍历：获取某节点的出/入边关系。"
                "适用于查询文档的上下游关系、实体被哪些文档提及。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "节点 ID，如 document:rag 或 entity:attention"},
                    "direction": {"type": "string", "description": "方向：out/in/both（默认both）", "default": "both"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "related_articles",
            "description": "获取与某文档相关的文章（related/extends/depends/implements 边）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档 ID"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "entity_lookup",
            "description": "按名称查找实体，返回该实体被哪些文档提及。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "实体名称"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "topic_tree",
            "description": "列出所有主题（topic）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tag_tree",
            "description": "列出所有标签（tag）及其层级关系。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_document",
            "description": (
                "合并两篇文档为一篇新文档。创建合并后的新文档，建立 supersedes 边，"
                "并将源文档标记为 archived。适用于发现重复内容时整合知识。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "description": "源文档 ID"},
                    "target_id": {"type": "string", "description": "目标文档 ID"},
                    "merged_title": {"type": "string", "description": "合并后文档标题"},
                    "merged_content": {"type": "string", "description": "合并后文档内容（Markdown）"},
                    "merged_summary": {"type": "string", "description": "合并后文档摘要"},
                },
                "required": ["source_id", "target_id", "merged_title", "merged_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_metadata",
            "description": (
                "更新文档元数据（标签、实体、主题、作者）并重建对应图边。"
                "适用于修正或补充文档的元数据信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档 ID"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"},
                    "entities": {"type": "array", "items": {"type": "object"}, "description": "实体列表 [{name, type}]"},
                    "topic": {"type": "string", "description": "主题 ID"},
                    "author": {"type": "string", "description": "作者"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_graph",
            "description": (
                "对指定文档抽取实体和关系并建立图边。"
                "适用于文档已有内容但尚未建立知识图谱连接时。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档 ID"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hot_documents",
            "description": "返回被用户问得最多的文档排行（基于对话记录统计）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回数量（默认10）", "default": 10},
                },
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_documents": lambda query, topk=5: wr.search_documents(query, topk),
    "get_document": lambda doc_id: wr.get_document(doc_id) or {"error": f"文档 {doc_id} 不存在"},
    "list_documents": lambda topic=None: {"documents": wr.list_documents(topic=topic)},
    "graph_neighbors": lambda node_id, direction="both": {"neighbors": wr.neighbors(node_id, direction)},
    "related_articles": lambda doc_id: {"related": wr.related_articles(doc_id)},
    "entity_lookup": lambda name: wr.entity_lookup(name) or {"error": f"实体 {name} 不存在"},
    "topic_tree": lambda: {"topics": wr.topic_tree()},
    "tag_tree": lambda: {"tags": wr.tag_tree()},
    "merge_document": lambda source_id, target_id, merged_title, merged_content, merged_summary=None: wr.merge_document(source_id, target_id, merged_title, merged_content, merged_summary),
    "update_metadata": lambda doc_id, tags=None, entities=None, topic=None, author=None: wr.update_metadata(doc_id, tags, entities, topic, author),
    "build_graph": lambda doc_id: wr.build_graph(doc_id),
    "hot_documents": lambda limit=10: {"documents": wr.hot_documents(limit)},
}

AGENT_SYSTEM_PROMPT = """\
你是一个 Semantic Wiki 问答 Agent。你通过调用工具检索知识图谱中的信息来回答问题。

## 工作流程（闭环检索）
1. 收到问题 → 分析需要什么信息
2. 调用工具检索 → 观察结果
3. 自评：检索结果是否足够回答问题？
   - 够了 → 生成带引用的最终回答
   - 不够 → 明确缺什么信息 → 补充检索 → 再次自评
   - 最多补充检索 2 次（总共 3 轮检索）

## 自评标准
检索结果足够，当：
- 找到了问题的核心概念定义和解释
- 找到了相关的因果关系或排查步骤
- 没有明显的矛盾信息

需要继续检索，当：
- 关键概念未被覆盖
- 结果中存在矛盾需要验证
- 排查类问题缺少某个关键环节

## 可用工具
1. search_documents — 四路融合检索（向量+全文+图+元数据），返回 doc_id/title/score/sources/excerpt
2. get_document — 读取单篇文档全文及关系邻接
3. list_documents — 列出所有文档元数据
4. graph_neighbors — 图邻接遍历
5. related_articles — 获取文档相关文章
6. entity_lookup — 按名称查找实体
7. topic_tree — 列出所有主题
8. tag_tree — 列出所有标签层级
9. merge_document — 合并重复文档
10. update_metadata — 更新文档元数据
11. build_graph — 对文档抽取实体关系建图边
12. hot_documents — 热门文档排行

## 检索策略
- 第 1 轮：用 search_documents 广撒网，查看最高分结果的标题和摘要
- 若第 1 轮信息不够 → 针对缺失信息用更精确的关键词再搜
- 第 2 轮后仍不够 → 用 get_document 精读最相关的文档全文
- 简单定义类问题（"什么是 X"）通常 1 轮就够了

## 引用要求（必须遵守）
- 回答中必须标注来源：[doc_id] title
- 不能编造文档中不存在的信息
- 如果工具返回的结果不足以完整回答问题，明确说明信息不足，并给出已有信息中能得到的部分结论

## 错误守门
- 如果工具返回 error 字段，如实说明失败原因
- 不能把工具调用失败描述为成功

## 知识库
知识库包含关于 LLM、Transformer、Attention、BERT、RAG、Embedding、向量数据库、HNSW、Agent、Prompt、Fine-tuning 的中文 Wiki 文档，以及它们之间的关系图。
"""


def run_agent(question: str, max_iterations: int = 8,
              use_ontology: bool = True,
              use_rerank: bool = True) -> dict:
    """运行闭环 Wiki Agent。

    流程（P3 快速通道 + P1-3 检索计划 + P2-1 硬约束 + P2-2 增强答案）：
      0. classify_query → fast/slow path
      1. 概念定位 + 本体展开 → 检索计划
      2. Tool-calling 循环（代码级闭环控制）
      3. Re-Rank 精选
      4. 答案附加 ontology_path + confidence

    返回 {"answer": str, "trace": [...], "iterations": int, "elapsed": float,
           "concept_location": dict | None, "ontology_path": list | None,
           "confidence": float | None, "path_type": "fast" | "deep"}
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    ]

    # ── Step 0: 复杂度判定（P3 快速通道）──
    path_type = "deep"
    is_simple = False
    if concept_locator:
        try:
            classification = concept_locator.classify_query(question)
            is_simple = classification.get("complexity") == "simple"
            path_type = "fast" if is_simple else "deep"
        except Exception:
            pass

    # ── Step 1: 概念定位 + 检索计划（P1-3）──
    concept_info: dict | None = None
    search_plan: dict | None = None
    concept_names: list[str] = []
    if use_ontology and concept_locator:
        try:
            concept_info = concept_locator.locate(question)
            located = concept_info.get("located_concepts", [])
            implicit = concept_info.get("implicit_concepts", [])
            concept_names = [c["name"] for c in located if c.get("confidence", 0) > 0.5]
            concept_names += [c["name"] for c in implicit if c.get("confidence", 0) > 0.4]

            if concept_names and ontology_traversal:
                # 生成检索计划
                try:
                    search_plan = ontology_traversal.generate_search_plan(concept_names, depth=2)
                except Exception:
                    search_plan = None

                # 概念提示注入
                concept_hint = []
                if located:
                    names = [c["name"] for c in located if c.get("confidence", 0) > 0.5]
                    if names:
                        concept_hint.append(f"相关概念: {', '.join(names)}")
                if implicit:
                    names = [c["name"] for c in implicit if c.get("confidence", 0) > 0.4]
                    if names:
                        concept_hint.append(f"可能相关的概念: {', '.join(names)}")
                if search_plan:
                    scope = search_plan.get("stats", {})
                    concept_hint.append(
                        f"检索范围: {scope.get('bound_documents', 0)} 文档, "
                        f"{scope.get('bound_files', 0)} 文件"
                    )
                if concept_hint:
                    question = f"{question}\n\n[系统提示] {'; '.join(concept_hint)}"
        except Exception:
            concept_info = None

    # 简单问题走快速通道：限制 max_iterations 为 3
    if is_simple:
        max_iterations = min(max_iterations, 4)

    messages.append({"role": "user", "content": question})

    trace: list[dict] = []
    start_time = time.time()
    all_retrieved_docs: list[dict] = []
    search_rounds = 0
    last_new_docs = 0
    prev_retrieved_ids: set[str] = set()

    # P2-1: 硬性约束
    TIME_LIMIT = 45     # 总时间上限 (秒)
    TOKEN_LIMIT = 40000 # 总 token 预算上限 (估算: 1字≈0.3 token)

    def _est_tokens(text: str) -> int:
        """粗略 token 估算：中文字≈1 token/字，英文≈0.25 token/字"""
        cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en = len(text) - cn
        return cn + int(en * 0.25)

    def _total_tokens(msgs: list[dict]) -> int:
        return sum(_est_tokens(m.get("content", "") or "") for m in msgs)

    for i in range(max_iterations):
        # P2-1: 时间超时检查
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            messages.append({
                "role": "user",
                "content": f"[系统] 已超时({TIME_LIMIT}秒)。请基于已有信息直接给出最佳回答。",
            })
            # 让 LLM 最后一轮直接回答
            max_iterations = i + 2

        # P2-1: token 预算检查
        if _total_tokens(messages) > TOKEN_LIMIT:
            messages.append({
                "role": "user",
                "content": "[系统] 已达 token 预算上限。请基于已有信息直接给出最佳回答。",
            })
            max_iterations = i + 2
        url, body, headers = _build_llm_request(_to_provider_messages(messages), TOOL_DEFS)
        r = requests.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT)

        if r.status_code != 200:
            raise LLMError(
                f"LLM 调用失败 (模型 {config.LLM_MODEL}): ({r.status_code}) {r.text[:200]}"
            )

        content, tool_calls = _parse_llm_response(r.json())

        if tool_calls:
            # P2-1: 追踪检索轮次
            is_search_call = any(tc["name"] == "search_documents" for tc in tool_calls)
            if is_search_call:
                search_rounds += 1
                # P2-1: 超过 3 轮检索，强制注入终止提示
                if search_rounds > 3:
                    messages.append({
                        "role": "user",
                        "content": "[系统] 已达到最大检索轮次(3轮)。请基于已有信息直接给出最佳回答，明确标注不确定的部分。",
                    })
                    continue

                # P2-1: 每轮检索范围收窄提示
                if search_rounds == 2:
                    messages.append({
                        "role": "user",
                        "content": "[系统] 第2轮检索：请针对第1轮缺失的关键信息做精确补充检索，不要重复之前的查询。",
                    })
                elif search_rounds == 3:
                    messages.append({
                        "role": "user",
                        "content": "[系统] 第3轮检索（最后一轮）：仅验证矛盾点或补充最关键的缺失环节。",
                    })

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

            new_docs_this_round = 0
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]

                tool_fn = TOOL_FUNCTIONS.get(tool_name)
                t0 = time.time()
                if tool_fn:
                    try:
                        result = tool_fn(**tool_args)
                    except Exception as exc:
                        result = {"error": f"工具执行异常: {exc}"}
                else:
                    result = {"error": f"未知工具: {tool_name}"}
                duration = round(time.time() - t0, 3)

                # 收集 search_documents 结果
                if tool_name == "search_documents" and isinstance(result, dict):
                    docs = result.get("results", [])
                    for d in docs:
                        did = d.get("doc_id", "")
                        if did and did not in prev_retrieved_ids:
                            new_docs_this_round += 1
                            prev_retrieved_ids.add(did)
                    all_retrieved_docs.extend(docs)

                trace.append({
                    "iteration": i + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                    "duration": duration,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tool_name,
                    "content": _json_safe_dumps(result),
                })

            # P2-1: 连续两轮无新增有效信息 → 强制终止
            if is_search_call and search_rounds >= 2 and new_docs_this_round == 0 and last_new_docs == 0:
                messages.append({
                    "role": "user",
                    "content": "[系统] 最近两轮检索均无新增信息。请直接基于已有证据给出回答。",
                })
            # P2-1: 每轮检索后告知 Agent 已检索的文档，防止重复查询
            if is_search_call and prev_retrieved_ids:
                retrieved_titles = [d.get("title", d.get("doc_id", ""))
                                   for d in all_retrieved_docs[-5:]]
                messages.append({
                    "role": "user",
                    "content": f"[系统] 已检索文档({len(prev_retrieved_ids)}篇，最新5篇): "
                               f"{'; '.join(retrieved_titles)}。补充检索时避免重复查询已覆盖的内容。",
                })
            last_new_docs = new_docs_this_round
        else:
            answer = _clean_answer(content)

            # ── Re-Rank 后处理 ──
            if use_rerank and reranker and all_retrieved_docs:
                try:
                    reranked = reranker.rerank(all_retrieved_docs, question,
                                               target_concept_ids=concept_names, topk=5)
                    trace.append({
                        "iteration": i + 1,
                        "tool": "_rerank",
                        "args": {"candidates": len(all_retrieved_docs)},
                        "result": {"top_docs": [(r["title"], r.get("final_score", r.get("score", 0))) for r in reranked[:5]]},
                        "duration": 0,
                    })
                except Exception:
                    pass

            # P2-2: 构建 ontology_path 和 confidence
            ontology_path = None
            confidence = None
            if concept_info and concept_info.get("located_concepts"):
                ontology_path = [
                    {
                        "step": "概念定位",
                        "concepts": [c["name"] for c in concept_info.get("located_concepts", [])],
                    }
                ]
                if search_plan:
                    ontology_path.append({
                        "step": "检索计划",
                        "strategies": [s["strategy"] for s in search_plan.get("plan", [])],
                    })
                ontology_path.append({
                    "step": "检索执行",
                    "rounds": search_rounds,
                    "docs_retrieved": len(all_retrieved_docs),
                })
                # 简单置信度：概念匹配数 / 结果覆盖
                if concept_info.get("located_concepts"):
                    confidence = min(0.9, 0.5 + 0.1 * len(concept_info["located_concepts"]))

            # 记录对话到 Memory Graph
            doc_ids: list[str] = []
            for t in trace:
                result = t.get("result", {})
                if isinstance(result, dict):
                    for r in (result.get("results") or []):
                        did = r.get("doc_id", "")
                        if did and did not in doc_ids:
                            doc_ids.append(did)
                    did = result.get("id", "") or result.get("doc_id", "")
                    if did and did not in doc_ids:
                        doc_ids.append(did)
            try:
                wr.record_conversation(question, answer, doc_ids)
            except Exception:
                pass

            return {
                "answer": answer,
                "trace": trace,
                "iterations": i + 1,
                "elapsed": round(time.time() - start_time, 2),
                "concept_location": concept_info,
                "ontology_path": ontology_path,
                "confidence": confidence,
                "path_type": path_type,
            }

    return {
        "answer": "达到最大迭代次数，未能生成完整回答。请尝试更具体的问题。",
        "trace": trace,
        "iterations": max_iterations,
        "elapsed": round(time.time() - start_time, 2),
        "concept_location": concept_info,
        "ontology_path": None,
        "confidence": None,
        "path_type": path_type,
    }
