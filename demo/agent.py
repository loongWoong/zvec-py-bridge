"""OAG Agent — 最小化实现

基于设计文档的 OAG（Ontology-Agent-Generation）模式：
    结构化本体 → LLM 自主选择工具 → 确定性 runtime 执行 → LLM 整合结果

与传统 RAG 的区别：
    RAG:  检索文本 → LLM 总结（单步，固定流程）
    OAG:  LLM 自主选择检索工具 → runtime 执行 → LLM 整合（多步，可追溯）

本模块实现：
  1. 本体层 — 将语料结构化为 Document + DocumentChunk 对象
  2. 工具层 — 4 个领域专用函数（search/read/list/prepare），各带 usage_prompt
  3. Agent 循环 — 支持 Ollama 原生与 OpenAI 兼容 tool calling（由 kb.LLM_API 决定），LLM 自主决策调用链
  4. 可追溯 — trace 记录每次工具调用的名称、参数、结果、耗时
  5. 引用约束 — 系统提示词要求回答标注来源 [document_id] title
  6. 错误守门 — 工具错误返回 {error: ...}，提示词要求不得掩盖失败
"""
from __future__ import annotations

import json
import os
import time

import requests

import kb_data as kb

# ====================================================================== #
#  本体（Ontology）：Document + DocumentChunk
# ====================================================================== #
DOCUMENTS: list[dict] = []
CHUNKS: dict[str, dict] = {}

# 持久化状态文件
_STATE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_STATE_DIR, "kb_state.json")
UPLOADS_DIR = os.path.join(_STATE_DIR, "uploads")


# ====================================================================== #
#  持久化 — save / load / clear / reset
# ====================================================================== #
def save_state(extra: dict | None = None) -> None:
    """将 DOCUMENTS / CHUNKS 序列化到 JSON 文件，实现重启后恢复。"""
    data = {
        "documents": DOCUMENTS,
        "chunks": CHUNKS,
    }
    if extra:
        data.update(extra)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_state() -> dict | None:
    """从 JSON 文件恢复 DOCUMENTS / CHUNKS。返回 extra 字段或 None。"""
    global DOCUMENTS, CHUNKS
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        DOCUMENTS = data.get("documents", [])
        CHUNKS = data.get("chunks", {})
        if not DOCUMENTS:
            return None
        # 返回 extra 字段（dimension, upload_count 等）
        return {k: v for k, v in data.items() if k not in ("documents", "chunks")}
    except Exception:
        return None


def clear_state() -> None:
    """删除持久化状态文件。"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def reset() -> None:
    """清空内存中的本体（cleanup 时调用）。"""
    global DOCUMENTS, CHUNKS
    DOCUMENTS = []
    CHUNKS = {}


def get_document(document_id: str) -> dict | None:
    """按 ID 获取单个文档（含分块）。"""
    for doc in DOCUMENTS:
        if doc["document_id"] == document_id:
            return doc
    return None


def build_ontology() -> None:
    """将 kb.CORPUS 结构化为 Document + DocumentChunk 对象。

    按中文句号切分，保留 heading / ordinal，避免固定长度切碎语义结构。
    """
    global DOCUMENTS, CHUNKS
    DOCUMENTS = []
    CHUNKS = {}
    for doc in kb.CORPUS:
        doc_id = doc["id"]
        content = doc["content"]
        # 按句号切分，保留完整句子
        sentences = [s.strip() + "。" for s in content.split("。") if s.strip()]
        doc_chunks = []
        for i, sent in enumerate(sentences):
            chunk_id = f"{doc_id}_c{i + 1}"
            chunk = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "heading": doc["title"],
                "content": sent,
                "ordinal": i + 1,
            }
            doc_chunks.append(chunk)
            CHUNKS[chunk_id] = chunk
        DOCUMENTS.append({
            "document_id": doc_id,
            "title": doc["title"],
            "content": content,
            "category": "RAG/向量数据库",
            "char_count": len(content),
            "chunks": doc_chunks,
            "source_file": "",
            "source_path": "",
        })
    save_state()


def add_documents(parsed_docs: list[dict]) -> None:
    """将上传的文档追加到本体（DOCUMENTS / CHUNKS）。

    parsed_docs: [{id, title, content, chunks:[{heading, content}], source_type, source_file}]
    """
    for doc in parsed_docs:
        doc_id = doc["id"]
        doc_chunks = []
        for i, chunk in enumerate(doc["chunks"]):
            chunk_id = f"{doc_id}_c{i + 1}"
            chunk_obj = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "heading": chunk["heading"],
                "content": chunk["content"],
                "ordinal": i + 1,
            }
            doc_chunks.append(chunk_obj)
            CHUNKS[chunk_id] = chunk_obj
        DOCUMENTS.append({
            "document_id": doc_id,
            "title": doc["title"],
            "content": doc["content"],
            "category": doc.get("source_type", "uploaded"),
            "char_count": len(doc["content"]),
            "chunks": doc_chunks,
            "source_file": doc.get("source_file", ""),
            "source_path": doc.get("source_path", ""),
        })
    save_state()


# ====================================================================== #
#  工具实现（确定性 runtime）
#  每个函数对应设计幻灯片 2.4 中的一个检索分层
# ====================================================================== #
def tool_search_documents(query: str, limit: int = 5) -> dict:
    """定位层：语义检索文档，返回轻量证据片段。

    调用 zvec 向量库进行语义检索，返回 chunk 级证据片段（含相关度评分）。
    """
    docs = kb.search(query, topk=limit)
    return {
        "count": len(docs),
        "results": [
            {
                "document_id": d.get("document_id") or d["id"],
                "chunk_id": d["id"],
                "title": d["title"],
                "heading": d.get("heading", ""),
                "score": round(d["score"], 4),
                "excerpt": d["content"][:120] + "..." if len(d["content"]) > 120 else d["content"],
            }
            for d in docs
        ],
    }


def tool_read_document(document_id: str) -> dict:
    """核验层：读取单篇文档全文及其分块。

    用于核对出处、补充上下文。从内存本体读取，无网络调用。
    """
    for doc in DOCUMENTS:
        if doc["document_id"] == document_id:
            return {
                "document_id": doc["document_id"],
                "title": doc["title"],
                "content": doc["content"],
                "category": doc["category"],
                "char_count": doc["char_count"],
                "source_file": doc.get("source_file", ""),
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
    return {"error": f"文档 {document_id} 不存在，可用文档: {[d['document_id'] for d in DOCUMENTS]}"}


def tool_list_documents() -> dict:
    """定位层：列出知识库中所有文档的元数据。"""
    return {
        "count": len(DOCUMENTS),
        "documents": [
            {
                "document_id": d["document_id"],
                "title": d["title"],
                "category": d["category"],
                "char_count": d["char_count"],
                "chunk_count": len(d["chunks"]),
                "source_file": d.get("source_file", ""),
            }
            for d in DOCUMENTS
        ],
    }


def tool_prepare_answer_context(query: str, limit: int = 5) -> dict:
    """综合层：检索多文档证据包，用于综合回答。

    返回多文档证据包 + 综合提纲，适用于汇总/比较/归纳/趋势判断类问题。
    """
    docs = kb.search(query, topk=limit)
    return {
        "query": query,
        "document_count": len(docs),
        "documents": [
            {
                "document_id": d["id"],
                "title": d["title"],
                "score": round(d["score"], 4),
                "excerpts": [d["content"]],
            }
            for d in docs
        ],
        "synthesis_outline": (
            "建议按以下角度组织回答："
            "1) 直接回答用户问题；"
            "2) 引用相关文档作为依据（标注 document_id 和 title）；"
            "3) 如需比较或多角度分析，综合多篇文档内容。"
        ),
    }


# 工具注册表：name → (function, description)
TOOL_FUNCTIONS = {
    "search_documents": tool_search_documents,
    "read_document": tool_read_document,
    "list_documents": tool_list_documents,
    "prepare_answer_context": tool_prepare_answer_context,
}


# ====================================================================== #
#  工具定义（Ollama tool calling 格式，含 usage_prompt）
# ====================================================================== #
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "语义检索文档，返回轻量证据片段（含相关度评分）。"
                "适用于定位性问题：哪些文档提到 X、X 出现在哪里。"
                "usage_prompt: 若用户要的是答案本身且需归纳/比较/趋势，"
                "不要仅凭本工具结果作答，应继续调用 read_document 或 prepare_answer_context。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询文本"},
                    "limit": {"type": "integer", "description": "返回结果数量（默认5）", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "读取单篇文档全文及其分块。"
                "适用于核对出处、补充上下文。"
                "usage_prompt: 搜索片段不够完整或需核对出处时调用；"
                "不要只根据搜索片段作答，也不要盲目遍历大量全文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "文档 ID，如 doc_01"},
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "列出知识库中所有文档的元数据（标题、分类、字符数、分块数）。"
                "适用于列文档、按分类筛选。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_answer_context",
            "description": (
                "检索多文档证据包，用于综合回答。"
                "适用于汇总、比较、归纳、趋势判断等需要多文档支持的问题。"
                "usage_prompt: 回答必须基于返回的 documents[].excerpts，"
                "先看 synthesis_outline，不能声称覆盖检索结果之外的内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询文本"},
                    "limit": {"type": "integer", "description": "返回结果数量（默认5）", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


# ====================================================================== #
#  系统提示词
# ====================================================================== #
SYSTEM_PROMPT = """\
你是一个文档问答 Agent，基于 OAG（本体-工具-执行）模式工作。

## 工作模式
你不是直接从自身知识回答，而是通过调用工具检索证据，再基于检索结果整合答案。
工作流程：分析问题 → 选择合适工具 → 查看结果 → 决定是否继续检索或直接回答 → 生成带引用的回答。

## 可用工具

1. **search_documents** — 语义检索，返回 chunk 级证据片段（document_id, chunk_id, title, heading, score, excerpt）
   - 何时使用：用户问"哪些文档提到 X""X 出现在哪里"等定位性问题
   - 注意：结果中的 document_id 可用于 read_document 获取完整文档
   - 不要：若用户需要归纳/比较/趋势分析，不要仅凭此工具结果作答，应继续调用 read_document 或 prepare_answer_context

2. **read_document** — 读取单篇文档全文及分块（参数为 document_id）
   - 何时使用：搜索片段不够完整、需要核对出处或补充上下文时
   - 不要：不要盲目遍历大量全文

3. **list_documents** — 列出所有文档元数据
   - 何时使用：用户要列文档、按分类筛选

4. **prepare_answer_context** — 多文档证据包 + 综合提纲
   - 何时使用：汇总、比较、归纳、趋势判断等需要多文档支持的问题
   - 回答必须基于返回的 documents[].excerpts

## 引用要求（必须遵守）
- 回答中必须标注来源：[document_id] title
- 不能编造文档中不存在的信息
- 如果工具返回的结果不足以完整回答问题，应明确说明信息不足

## 错误守门
- 如果工具返回 error 字段，必须在回答中如实说明失败原因
- 不能把工具调用失败描述为成功

## 知识库
知识库包含 8 篇关于 RAG、向量数据库、嵌入模型、HNSW 索引、余弦相似度、文档分块、混合检索、重排序的中文文档。
用户也可能上传了额外的文档（Markdown、DOCX、PDF、XLSX），可通过 list_documents 查看全部文档。
"""


# ====================================================================== #
#  Agent 循环
# ====================================================================== #
def _build_llm_request(messages: list[dict], tools: list[dict]):
    """根据 kb.LLM_API 构造请求，返回 (url, json_body, headers)。"""
    body = {
        "model": kb.LLM_MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False,
    }
    if kb.LLM_API == "openai":
        headers = (
            {"Authorization": f"Bearer {kb.LLM_API_KEY}"}
            if kb.LLM_API_KEY else {}
        )
        return f"{kb.LLM_URL}/v1/chat/completions", body, headers
    # 默认 Ollama 原生格式
    return f"{kb.LLM_URL}/api/chat", body, {}


def _parse_llm_response(data: dict):
    """解析响应，返回 (content, tool_calls)。

    tool_calls 归一化为 [{id, name, arguments(dict)}, ...]，
    屏蔽 Ollama（arguments 为 dict）与 OpenAI（arguments 为 JSON 字符串）差异。
    """
    if kb.LLM_API == "openai":
        msg = data["choices"][0]["message"]
    else:
        msg = data["message"]

    content = msg.get("content") or ""
    raw_calls = msg.get("tool_calls") or []
    tool_calls = []
    for tc in raw_calls:
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
            "id": tc.get("id", ""),
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
            if kb.LLM_API == "openai":
                out.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
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
                out.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {
                            "id": tc.get("id", ""),
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in m["tool_calls"]
                    ],
                })
        elif role == "tool":
            if kb.LLM_API == "openai":
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


def run_agent(question: str, max_iterations: int = 6) -> dict:
    """运行 OAG Agent：LLM 选择工具 → runtime 执行 → LLM 整合结果。

    兼容 Ollama 原生与 OpenAI 兼容两种接口（由 kb.LLM_API 决定）。

    返回 {"answer": str, "trace": [...], "iterations": int, "elapsed": float}
    """
    if not DOCUMENTS:
        build_ontology()

    # 归一化消息：tool_calls 用 {id, name, arguments}，tool 结果用 {tool_call_id, content}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    start_time = time.time()

    for i in range(max_iterations):
        url, body, headers = _build_llm_request(
            _to_provider_messages(messages), TOOL_DEFS
        )
        r = requests.post(url, json=body, headers=headers, timeout=120)

        if r.status_code != 200:
            raise kb.KBError(
                f"LLM 调用失败 (模型 {kb.LLM_MODEL}): ({r.status_code}) {r.text[:200]}",
                r.status_code,
            )

        content, tool_calls = _parse_llm_response(r.json())

        if tool_calls:
            # 保留 assistant 消息（含 tool_calls）到对话历史
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]

                # 确定性 runtime 执行工具
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

                # 将工具结果送回 LLM（OpenAI 需要 tool_call_id 关联）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            # LLM 生成最终回答
            answer = content
            return {
                "answer": answer,
                "trace": trace,
                "iterations": i + 1,
                "elapsed": round(time.time() - start_time, 2),
            }

    # 达到最大迭代次数
    return {
        "answer": "达到最大迭代次数，未能生成完整回答。请尝试更具体的问题。",
        "trace": trace,
        "iterations": max_iterations,
        "elapsed": round(time.time() - start_time, 2),
    }
