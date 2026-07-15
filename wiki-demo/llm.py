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

## 工作模式
你不是直接从自身知识回答，而是通过调用工具检索证据，再基于检索结果整合答案。
工作流程：分析问题 → 选择合适工具 → 查看结果 → 决定是否继续检索或直接回答 → 生成带引用的回答。

## 可用工具
1. search_documents — 四路融合检索（向量+全文+图+元数据），返回 doc_id/title/score/sources/excerpt
2. get_document — 读取单篇文档全文及关系邻接
3. list_documents — 列出所有文档元数据
4. graph_neighbors — 图邻接遍历（查询文档上下游关系、实体被哪些文档提及）
5. related_articles — 获取与某文档相关的文章
6. entity_lookup — 按名称查找实体，返回被哪些文档提及
7. topic_tree — 列出所有主题
8. tag_tree — 列出所有标签层级
9. merge_document — 合并两篇重复文档为一篇新文档（建 supersedes 边，源文档标记 archived）
10. update_metadata — 更新文档元数据（标签/实体/主题/作者）并重建图边
11. build_graph — 对文档抽取实体和关系并建立图边
12. hot_documents — 返回被用户问得最多的文档排行

## 引用要求（必须遵守）
- 回答中必须标注来源：[doc_id] title
- 不能编造文档中不存在的信息
- 如果工具返回的结果不足以完整回答问题，应明确说明信息不足

## 错误守门
- 如果工具返回 error 字段，必须在回答中如实说明失败原因
- 不能把工具调用失败描述为成功

## 知识库
知识库包含关于 LLM、Transformer、Attention、BERT、RAG、Embedding、向量数据库、HNSW、Agent、Prompt、Fine-tuning 的中文 Wiki 文档，以及它们之间的关系图。
"""


def run_agent(question: str, max_iterations: int = 6) -> dict:
    """运行 Wiki Agent：LLM 选择工具 → runtime 执行 → LLM 整合结果。

    返回 {"answer": str, "trace": [...], "iterations": int, "elapsed": float}
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    start_time = time.time()

    for i in range(max_iterations):
        url, body, headers = _build_llm_request(_to_provider_messages(messages), TOOL_DEFS)
        r = requests.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT)

        if r.status_code != 200:
            raise LLMError(
                f"LLM 调用失败 (模型 {config.LLM_MODEL}): ({r.status_code}) {r.text[:200]}"
            )

        content, tool_calls = _parse_llm_response(r.json())

        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

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
        else:
            answer = _clean_answer(content)
            # 记录对话到 Memory Graph（design §8）：从 trace 中提取引用的 doc_id
            doc_ids: list[str] = []
            for t in trace:
                result = t.get("result", {})
                if isinstance(result, dict):
                    # search_documents 返回 results 列表
                    for r in (result.get("results") or []):
                        did = r.get("doc_id", "")
                        if did and did not in doc_ids:
                            doc_ids.append(did)
                    # get_document 返回单个 doc
                    did = result.get("id", "") or result.get("doc_id", "")
                    if did and did not in doc_ids:
                        doc_ids.append(did)
            try:
                wr.record_conversation(question, answer, doc_ids)
            except Exception:
                pass  # 记录失败不影响主流程
            return {
                "answer": answer,
                "trace": trace,
                "iterations": i + 1,
                "elapsed": round(time.time() - start_time, 2),
            }

    return {
        "answer": "达到最大迭代次数，未能生成完整回答。请尝试更具体的问题。",
        "trace": trace,
        "iterations": max_iterations,
        "elapsed": round(time.time() - start_time, 2),
    }
