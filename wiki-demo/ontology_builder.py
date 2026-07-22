"""
LLM 辅助本体构建器 — 从文档库中自动抽取/提议概念、关系和绑定。

提供三个核心能力：
  1. propose_concepts()     — LLM 扫描文档摘要，提取候选概念列表
  2. propose_relations()    — LLM 分析概念间的层级/依赖关系
  3. propose_bindings()     — LLM 判断概念应该绑定到哪些文档
  4. propose_all()          — 一键运行上述三步，输出完整本体 YAML

工作流：
  扫描所有文档 → LLM 提议概念 → 人工 review YAML → import_from_yaml() → 入库

使用 llm.py 中的 LLM 客户端（支持 OpenAI 兼容 / Ollama）。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import llm as llm_client
import ontology
import wiki_runtime as wr


# ====================================================================== #
#  Prompt 模板
# ====================================================================== #

CONCEPT_PROPOSAL_SYSTEM = """\
你是一个知识本体专家。你的任务是从一组文档中抽取核心概念，构建领域本体。

请分析以下文档标题和摘要，抽取出核心概念。每个概念需要：
- name: 概念名称（简洁的技术术语，中英文均可）
- type: 类型，从以下选择：mechanism(机制/算法), component(组件/模块), process(流程/方法), 
        problem(问题/故障模式), configuration(配置/参数), concept(通用概念), 
        metric(度量), infrastructure(基础设施), pattern(模式), tool(工具)
- description: 一句话描述
- parent: 父概念名称（如果有层级关系，不要编造不存在的关系）

要求：
- 每个概念必须能在给定文档中找到依据
- 父概念必须是同时被提出的概念之一，不能引用不存在的概念
- 概念之间有明显的 is-a 层级关系时才填写 parent
- 数量控制在 20-50 个核心概念
- 只输出 JSON，不要输出其他内容

输出格式：
{
  "concepts": [
    {
      "name": "Self-Attention",
      "type": "mechanism",
      "description": "自注意力机制，通过 Q/K/V 计算序列内部依赖关系",
      "parent": "Attention"
    }
  ]
}
"""

RELATION_PROPOSAL_SYSTEM = """\
你是一个知识本体专家。你的任务是为已有的概念列表推断语义关系。

给定概念列表，请推断概念之间的关系。关系类型：
- depends: A 依赖 B（A 的实现/理解需要 B）
- triggers: A 触发 B（A 的发生导致 B）
- contains: A 包含 B（A 是 B 的上位概念/组成部分）
- contrasts: A 与 B 对比/互补（两者经常一起讨论但概念不同）
- related: 一般关联

要求：
- 只输出 JSON，不要输出其他内容
- 只提出确信的关系，不确定的不要编造

输出格式：
{
  "relations": [
    {"source": "RAG", "target": "Embedding", "type": "depends"},
    {"source": "Multi-Head Attention", "target": "Self-Attention", "type": "contains"}
  ]
}
"""

BINDING_PROPOSAL_SYSTEM = """\
你是一个知识本体专家。你的任务是将概念绑定到文档。

给定概念列表和文档列表（标题+摘要），判断每个概念应该绑定到哪些文档。
- primary: 该文档是此概念的核心解释文档
- secondary: 该文档提到了此概念，但不是主要讲解

要求：
- 只输出 JSON，不要输出其他内容
- 每个概念最多绑定 3 个主要文档

输出格式：
{
  "bindings": [
    {"concept": "Self-Attention", "document": "Transformer", "type": "primary"},
    {"concept": "Self-Attention", "document": "Attention", "type": "primary"}
  ]
}
"""


# ====================================================================== #
#  核心函数
# ====================================================================== #

def propose_concepts(
    documents: list[dict] | None = None,
    domain_hint: str = "",
) -> dict:
    """扫描文档列表，LLM 提取候选概念。

    Args:
        documents: 文档列表 [{title, summary, id}, ...]，None 则自动获取
        domain_hint: 领域提示（如 "机器学习"、"后端架构"）

    Returns:
        {"concepts": [{name, type, description, parent}, ...]}
    """
    if documents is None:
        docs = wr.list_documents()
        documents = [
            {"title": d.get("title", ""), "summary": d.get("summary", ""), "id": d.get("id", "")}
            for d in docs
        ]

    if not documents:
        return {"concepts": []}

    # 构建文档摘要文本
    docs_text = "\n".join(
        f"- [{d.get('title', '')}]: {d.get('summary', '')[:150]}"
        for d in documents[:50]  # 限制数量避免 token 溢出
    )

    hint_text = f"\n领域提示：{domain_hint}" if domain_hint else ""

    messages = [
        {"role": "system", "content": CONCEPT_PROPOSAL_SYSTEM},
        {"role": "user", "content": f"文档列表：\n{docs_text}{hint_text}\n\n请抽取核心概念。"},
    ]

    data = llm_client._call_llm_safe(messages, temperature=0.3)
    content, _ = llm_client._parse_llm_response(data)
    result = llm_client._extract_json(llm_client._clean_answer(content))

    if not result:
        return {"concepts": [], "error": f"LLM 输出无法解析: {content[:300]}"}

    return result


def propose_relations(concepts: list[dict]) -> dict:
    """对已有概念列表推断关系。

    Args:
        concepts: 概念列表 [{name, type, description}, ...]

    Returns:
        {"relations": [{source, target, type}, ...]}
    """
    if not concepts:
        return {"relations": []}

    concepts_text = "\n".join(
        f"- {c['name']} ({c.get('type', 'concept')}): {c.get('description', '')[:100]}"
        for c in concepts
    )

    messages = [
        {"role": "system", "content": RELATION_PROPOSAL_SYSTEM},
        {"role": "user", "content": f"概念列表：\n{concepts_text}\n\n请推断概念间关系。"},
    ]

    data = llm_client._call_llm_safe(messages, temperature=0.3)
    content, _ = llm_client._parse_llm_response(data)
    result = llm_client._extract_json(llm_client._clean_answer(content))

    if not result:
        return {"relations": [], "error": f"LLM 输出无法解析: {content[:300]}"}

    return result


def propose_bindings(
    concepts: list[dict],
    documents: list[dict] | None = None,
) -> dict:
    """将概念绑定到文档。

    Args:
        concepts: 概念列表 [{name, ...}]
        documents: 文档列表，None 则自动获取

    Returns:
        {"bindings": [{concept, document, type}, ...]}
    """
    if documents is None:
        docs = wr.list_documents()
        documents = [
            {"title": d.get("title", ""), "summary": d.get("summary", ""), "id": d.get("id", "")}
            for d in docs
        ]

    if not concepts or not documents:
        return {"bindings": []}

    concepts_text = "\n".join(
        f"- {c['name']}: {c.get('description', '')[:80]}"
        for c in concepts
    )
    docs_text = "\n".join(
        f"- {d['title']}: {d.get('summary', '')[:120]}"
        for d in documents[:50]
    )

    messages = [
        {"role": "system", "content": BINDING_PROPOSAL_SYSTEM},
        {"role": "user", "content": (
            f"概念列表：\n{concepts_text}\n\n"
            f"文档列表：\n{docs_text}\n\n"
            f"请判断每个概念应该绑定到哪些文档。"
        )},
    ]

    data = llm_client._call_llm_safe(messages, temperature=0.3)
    content, _ = llm_client._parse_llm_response(data)
    result = llm_client._extract_json(llm_client._clean_answer(content))

    if not result:
        return {"bindings": [], "error": f"LLM 输出无法解析: {content[:300]}"}

    return result


# ====================================================================== #
#  一键构建：全流程
# ====================================================================== #

def propose_all(
    domain_hint: str = "",
    output_yaml: str | None = None,
) -> dict:
    """一键运行：LLM 提取概念 → 推断关系 → 绑定文档 → 输出 YAML。

    Args:
        domain_hint: 领域提示
        output_yaml: 若提供则写入 YAML 文件

    Returns:
        {concepts: [...], relations: [...], bindings: [...], yaml_path: str|None}
    """
    # Step 1: 获取文档
    docs = wr.list_documents()
    documents = [
        {"title": d.get("title", ""), "summary": d.get("summary", ""), "id": d.get("id", "")}
        for d in docs
    ]

    if not documents:
        return {"error": "知识库中没有文档，请先入库一些文档"}

    # Step 2: 提取概念
    print(f"  → 正在从 {len(documents)} 篇文档中提取概念...")
    concept_result = propose_concepts(documents, domain_hint)
    concepts = concept_result.get("concepts", [])
    print(f"  ✓ 提取到 {len(concepts)} 个概念")

    if concept_result.get("error"):
        return {"error": f"概念提取失败: {concept_result['error']}"}

    # Step 3: 推断关系
    print(f"  → 正在推断概念间关系...")
    relation_result = propose_relations(concepts)
    relations = relation_result.get("relations", [])
    print(f"  ✓ 推断出 {len(relations)} 条关系")

    # Step 4: 绑定文档
    print(f"  → 正在绑定概念到文档...")
    binding_result = propose_bindings(concepts, documents)
    bindings = binding_result.get("bindings", [])
    print(f"  ✓ 生成 {len(bindings)} 条绑定")

    # Step 5: 组装为完整 YAML
    concepts_yaml = []
    name_to_parent = {}
    for c in concepts:
        if c.get("parent"):
            name_to_parent[c["name"]] = c["parent"]
    for c in concepts:
        entry = {
            "name": c["name"],
            "type": c.get("type", "concept"),
            "description": c.get("description", ""),
        }
        if c.get("parent"):
            entry["parent"] = c["parent"]
        # 添加关系（从 relations 中查找）
        entry_rels = [
            {"target": r["target"], "type": r["type"], "direction": "out"}
            for r in relations if r.get("source") == c["name"]
        ]
        if entry_rels:
            entry["relations"] = entry_rels
        # 添加绑定
        entry_binds = [
            {"document": b["document"], "type": b.get("type", "primary")}
            for b in bindings if b.get("concept") == c["name"]
        ]
        if entry_binds:
            entry["bindings"] = entry_binds
        concepts_yaml.append(entry)

    yaml_data = {"concepts": concepts_yaml}
    yaml_str = yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

    if output_yaml:
        Path(output_yaml).write_text(yaml_str, encoding="utf-8")
        print(f"  ✓ 已写入 {output_yaml}")

    return {
        "concept_count": len(concepts),
        "relation_count": len(relations),
        "binding_count": len(bindings),
        "concepts": concepts,
        "relations": relations,
        "bindings": bindings,
        "yaml": yaml_str,
        "yaml_path": output_yaml,
    }


# ====================================================================== #
#  增量更新：新增文档后补充本体
# ====================================================================== #

def update_for_documents(document_ids: list[str]) -> dict:
    """对新增文档集增量提议本体更新。

    已有概念不会被覆盖，只提议新的概念和绑定。

    Args:
        document_ids: 新增文档 ID 列表

    Returns:
        {new_concepts: [...], new_bindings: [...], yaml: str}
    """
    # 获取新增文档
    docs = []
    for did in document_ids:
        doc = wr.get_document(did)
        if doc:
            docs.append({"title": doc.get("title", ""), "summary": doc.get("summary", ""), "id": doc["id"]})

    if not docs:
        return {"error": "未找到有效文档"}

    # 获取已有概念名
    existing_concepts = ontology.list_concepts()
    existing_names = {c["name"] for c in existing_concepts}

    # 提取新概念
    result = propose_concepts(docs)
    new_concepts = [c for c in result.get("concepts", []) if c["name"] not in existing_names]

    # 绑定
    if new_concepts:
        binding_result = propose_bindings(new_concepts, docs)
        new_bindings = binding_result.get("bindings", [])
    else:
        new_bindings = []

    # 生成增量 YAML
    concepts_yaml = []
    for c in new_concepts:
        entry = {
            "name": c["name"],
            "type": c.get("type", "concept"),
            "description": c.get("description", ""),
        }
        if c.get("parent"):
            entry["parent"] = c["parent"]
        entry["bindings"] = [
            {"document": b["document"], "type": b.get("type", "primary")}
            for b in new_bindings if b.get("concept") == c["name"]
        ]
        concepts_yaml.append(entry)

    yaml_str = yaml.dump(
        {"concepts": concepts_yaml},
        allow_unicode=True, default_flow_style=False, sort_keys=False, width=120,
    )

    return {
        "new_concepts": new_concepts,
        "new_bindings": new_bindings,
        "yaml": yaml_str,
    }
