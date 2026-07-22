"""
概念定位器 — 查询时将用户自然语言问题映射到本体概念。

核心功能：
  locate(query) → LLM 从本体概念列表中选择匹配概念 + 置信度
  classify_query(query) → 判断问题复杂度（简单→快速通道 / 复杂→闭环）

工作方式：
  给 LLM 本体概念列表作为"选项"，让 LLM 从列表中做多选题，
  而不是自由发挥。大幅降低幻觉。

集成到 Agent 问答的 Step 5（意图理解 + 概念定位）。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm as llm_client
import ontology


# ====================================================================== #
#  Prompt
# ====================================================================== #

LOCATE_SYSTEM_PROMPT = """\
你是一个知识检索导航员。你的任务是将用户的自然语言问题映射到本体概念。

你会收到一个概念列表（本体中的核心概念）。你需要从列表中选出与用户问题相关的概念。

规则：
- 只能从给定的概念列表中选择，不要自己编造概念
- 为每个匹配的概念标注置信度 (0.0-1.0)
- 同时推断"隐含概念"：用户没直接说但很可能相关的概念
- 提取约束条件（如"偶尔"、"登录后"、"生产环境"）

输出 JSON 格式：
{
  "intent_type": "故障排查|概念解释|操作指南|对比分析|概述浏览",
  "located_concepts": [
    {"name": "概念名", "confidence": 0.9, "reason": "匹配原因简述"}
  ],
  "implicit_concepts": [
    {"name": "隐含概念名", "confidence": 0.6, "reason": "推测原因"}
  ],
  "constraints": {"频率": "偶尔", "触发点": "登录后"},
  "expected_output": "用户期望的回答类型：排查步骤/概念解释/操作指南"
}
"""

CLASSIFY_SYSTEM_PROMPT = """\
你是一个查询分类器。判断用户问题的复杂度。

简单问题（complexity: "simple"）：定义类、概述类、FAQ 类，已知概念明确，不需要多轮检索
复杂问题（complexity: "complex"）：排障类、分析类、多概念关联类，需要深度检索和多步推理

输出 JSON：
{
  "complexity": "simple|complex",
  "reason": "判断原因",
  "suggested_approach": "fast|deep"
}
"""


# ====================================================================== #
#  概念定位
# ====================================================================== #

def _get_concept_list_text() -> str:
    """生成概念列表文本（供 LLM 阅读）。"""
    concepts = ontology.list_concepts()
    if not concepts:
        return "(概念列表为空)"

    lines = []
    for c in concepts:
        parent_info = f" → 父: {c.get('parent_id', '')}" if c.get("parent_id") else ""
        lines.append(
            f"- {c['name']} [{c.get('type', '')}]"
            f" | {c.get('description', '')[:100]}"
            f"{parent_info}"
        )

    return "\n".join(lines)


def locate(query: str) -> dict:
    """将用户查询映射到本体概念。

    Args:
        query: 用户自然语言问题

    Returns:
        {
            intent_type: str,
            located_concepts: [{name, confidence, reason}],
            implicit_concepts: [{name, confidence, reason}],
            constraints: dict,
            expected_output: str
        }
    """
    concepts_text = _get_concept_list_text()

    messages = [
        {"role": "system", "content": LOCATE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"## 概念列表\n{concepts_text}\n\n"
            f"## 用户问题\n{query}\n\n"
            f"请从概念列表中选择与问题相关的概念。"
        )},
    ]

    data = llm_client._call_llm_safe(messages, temperature=0.3)
    content, _ = llm_client._parse_llm_response(data)
    result = llm_client._extract_json(llm_client._clean_answer(content))

    if not result:
        return {
            "intent_type": "概述浏览",
            "located_concepts": [],
            "implicit_concepts": [],
            "constraints": {},
            "expected_output": "一般回答",
            "error": f"LLM 输出无法解析: {content[:200]}",
        }

    return result


def classify_query(query: str) -> dict:
    """判断查询复杂度，决定走快速通道还是深度检索。

    Args:
        query: 用户问题

    Returns:
        {complexity: "simple"|"complex", reason, suggested_approach: "fast"|"deep"}
    """
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户问题：{query}"},
    ]

    data = llm_client._call_llm_safe(messages, temperature=0.3)
    content, _ = llm_client._parse_llm_response(data)
    result = llm_client._extract_json(llm_client._clean_answer(content))

    if not result:
        return {"complexity": "simple", "reason": "fallback", "suggested_approach": "fast"}

    return result


# ====================================================================== #
#  辅助：从定位结果生成增强的检索查询
# ====================================================================== #

def build_search_query(query: str, location_result: dict) -> str:
    """结合概念定位结果，增强检索查询。

    原始 query + 定位到的概念名 + 隐含概念名，
    拼接为更丰富的查询文本，提高检索召回率。

    Args:
        query: 原始用户查询
        location_result: locate() 的返回结果

    Returns:
        增强后的检索查询文本
    """
    parts = [query]

    located = location_result.get("located_concepts", [])
    if located:
        parts.append(" ".join(c["name"] for c in located if c.get("confidence", 0) > 0.5))

    implicit = location_result.get("implicit_concepts", [])
    if implicit:
        parts.append(" ".join(c["name"] for c in implicit if c.get("confidence", 0) > 0.4))

    return " ".join(parts)
