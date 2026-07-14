"""种子数据 — LLM/向量库主题知识库。

预置约 10 篇文档 + 关系边 + 实体 + 标签 + topic，
覆盖 Transformer/Attention/BERT/RAG/Embedding/向量数据库/HNSW/Agent/Prompt/Fine-tuning。
启动时若库为空则一键灌入（幂等）。
"""
from __future__ import annotations

import re

import config
import wiki_runtime as wr
import zvec_client

# ====================================================================== #
#  Topic 定义
# ====================================================================== #
TOPICS = [
    ("machine_learning", "机器学习", "机器学习与深度学习基础理论"),
    ("retrieval", "检索增强", "检索增强生成与向量检索技术"),
    ("infrastructure", "基础设施", "向量数据库与索引基础设施"),
]

# ====================================================================== #
#  Tag 层级（parent → child）
# ====================================================================== #
TAG_TREE = [
    # (key, name, parent_key)
    ("ai", "AI", None),
    ("llm", "LLM", "ai"),
    ("transformer", "Transformer", "llm"),
    ("agent", "Agent", "ai"),
    ("prompt", "Prompt", "llm"),
    ("fine_tuning", "Fine-tuning", "llm"),
    ("retrieval", "Retrieval", "ai"),
    ("rag", "RAG", "retrieval"),
    ("embedding", "Embedding", "retrieval"),
    ("vector_db", "Vector DB", "retrieval"),
    ("index", "Index", "vector_db"),
]

# ====================================================================== #
#  文档语料（key, title, topic, summary, content, tags, entities, relations）
# ====================================================================== #
DOCUMENTS = [
    {
        "key": "transformer",
        "title": "Transformer",
        "topic": "machine_learning",
        "summary": "基于自注意力机制的序列建模架构，是现代大语言模型的基石。",
        "content": (
            "Transformer 是 2017 年在论文《Attention Is All You Need》中提出的序列建模架构。"
            "它完全摒弃了循环神经网络（RNN）和卷积神经网络（CNN），"
            "仅依赖自注意力（Self-Attention）机制来捕捉序列内部的依赖关系。\n\n"
            "## 核心组件\n\n"
            "Transformer 由 Encoder 和 Decoder 两部分组成。"
            "Encoder 负责将输入序列编码为连续表示，Decoder 负责自回归地生成输出序列。"
            "每一层包含多头注意力（Multi-Head Attention）和前馈网络（Feed-Forward Network），"
            "并辅以残差连接（Residual Connection）和层归一化（Layer Normalization）。\n\n"
            "## 自注意力机制\n\n"
            "自注意力通过 Query、Key、Value 三个矩阵将输入映射，"
            "计算注意力权重：Attention(Q,K,V) = softmax(QK^T/√d_k)V。"
            "缩放因子 √d_k 防止内积过大导致梯度消失。\n\n"
            "## 位置编码\n\n"
            "由于注意力机制本身不具备位置感知能力，Transformer 引入位置编码（Positional Encoding）"
            "将位置信息注入输入嵌入。原始论文使用正弦/余弦函数生成位置编码。\n\n"
            "## 影响\n\n"
            "Transformer 成为 BERT、GPT 系列等大语言模型的基础架构，"
            "推动了自然语言处理领域的范式转变。"
        ),
        "tags": ["transformer", "llm", "ai"],
        "entities": [
            {"name": "Attention", "type": "mechanism"},
            {"name": "Encoder", "type": "component"},
            {"name": "Decoder", "type": "component"},
            {"name": "Multi-Head Attention", "type": "mechanism"},
            {"name": "Positional Encoding", "type": "mechanism"},
            {"name": "Layer Normalization", "type": "mechanism"},
            {"name": "Residual Connection", "type": "mechanism"},
        ],
        "relations": [
            {"target_title": "Attention", "type": "extends"},
            {"target_title": "BERT", "type": "related"},
            {"target_title": "Fine-tuning", "type": "related"},
        ],
    },
    {
        "key": "attention",
        "title": "Attention",
        "topic": "machine_learning",
        "summary": "允许模型动态关注输入序列中不同部分的机制，是 Transformer 的核心。",
        "content": (
            "注意力机制（Attention Mechanism）允许模型在处理序列时，"
            "动态地将注意力分配到输入的不同位置，从而捕捉长距离依赖关系。\n\n"
            "## 缩放点积注意力\n\n"
            "Transformer 使用的核心公式为："
            "Attention(Q,K,V) = softmax(QK^T/√d_k)V。"
            "其中 Q（Query）、K（Key）、V（Value）由输入经过线性变换得到。"
            "缩放因子 √d_k 用于防止点积结果过大导致 softmax 梯度消失。\n\n"
            "## 多头注意力\n\n"
            "多头注意力（Multi-Head Attention）将 Q、K、V 映射到多个子空间并行计算注意力，"
            "再拼接结果。这使得模型能同时关注不同位置和不同表示子空间的信息。\n\n"
            "## 交叉注意力\n\n"
            "在 Encoder-Decoder 架构中，Decoder 的每一层还包含交叉注意力，"
            "Query 来自 Decoder 上层输出，Key 和 Value 来自 Encoder 输出，"
            "实现解码时对编码信息的动态关注。"
        ),
        "tags": ["transformer", "llm", "ai"],
        "entities": [
            {"name": "Attention", "type": "mechanism"},
            {"name": "Multi-Head Attention", "type": "mechanism"},
            {"name": "Query", "type": "concept"},
            {"name": "Key", "type": "concept"},
            {"name": "Value", "type": "concept"},
        ],
        "relations": [],
    },
    {
        "key": "bert",
        "title": "BERT",
        "topic": "machine_learning",
        "summary": "基于 Transformer Encoder 的双向预训练语言模型，通过掩码语言建模学习上下文表示。",
        "content": (
            "BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的"
            "预训练语言模型。它仅使用 Transformer 的 Encoder 部分，"
            "通过掩码语言建模（Masked Language Modeling, MLM）和下一句预测（Next Sentence Prediction, NSP）"
            "两个任务进行预训练。\n\n"
            "## 双向编码\n\n"
            "与 GPT 的单向自回归不同，BERT 是双向的：它在预测被掩码的词时能同时看到左右上下文，"
            "因此能学习更丰富的上下文表示。\n\n"
            "## 微调范式\n\n"
            "BERT 采用预训练+微调（Pre-train then Fine-tune）的范式："
            "在大规模无标注语料上预训练，再在下游任务（分类、问答、序列标注）上微调。"
            "只需在 BERT 顶部添加任务特定的输出层即可适配。\n\n"
            "## 影响\n\n"
            "BERT 推动了预训练语言模型的普及，其双向编码思想影响了后续众多模型设计。"
        ),
        "tags": ["llm", "ai", "fine_tuning"],
        "entities": [
            {"name": "Encoder", "type": "component"},
            {"name": "Masked Language Modeling", "type": "task"},
            {"name": "Fine-tuning", "type": "method"},
        ],
        "relations": [
            {"target_title": "Transformer", "type": "implements"},
            {"target_title": "Fine-tuning", "type": "depends"},
        ],
    },
    {
        "key": "rag",
        "title": "RAG",
        "topic": "retrieval",
        "summary": "检索增强生成：先检索相关文档，再将其作为上下文拼入提示词，由 LLM 生成回答。",
        "content": (
            "检索增强生成（Retrieval-Augmented Generation, RAG）是一种结合检索与生成的技术。"
            "其核心思路是：先从知识库中检索与用户问题相关的文档片段，"
            "再将这些片段作为上下文拼入提示词，交由大语言模型生成回答。\n\n"
            "## 工作流程\n\n"
            "1. 用户提问 → 将问题嵌入为向量\n"
            "2. 向量检索 → 从向量数据库召回 Top-K 相关文档分块\n"
            "3. 上下文拼接 → 将检索结果作为上下文注入提示词\n"
            "4. LLM 生成 → 基于上下文生成最终回答\n\n"
            "## 优势\n\n"
            "RAG 能有效缓解大模型的幻觉（Hallucination）问题，"
            "支持基于私有知识的问答，且无需重新训练模型即可更新知识库。\n\n"
            "## 依赖\n\n"
            "RAG 依赖 Embedding 模型将文本向量化，依赖向量数据库存储和检索向量，"
            "依赖 LLM 进行最终生成。"
        ),
        "tags": ["rag", "retrieval", "ai"],
        "entities": [
            {"name": "Embedding", "type": "concept"},
            {"name": "Vector Database", "type": "infrastructure"},
            {"name": "Hallucination", "type": "problem"},
            {"name": "LLM", "type": "concept"},
        ],
        "relations": [
            {"target_title": "Embedding", "type": "depends"},
            {"target_title": "向量数据库", "type": "depends"},
            {"target_title": "Agent", "type": "related"},
        ],
    },
    {
        "key": "embedding",
        "title": "Embedding",
        "topic": "retrieval",
        "summary": "将文本映射为固定维度的稠密向量，使语义相近的文本在向量空间中距离更近。",
        "content": (
            "文本嵌入模型（Embedding Model）将自然语言文本映射为固定维度的稠密向量，"
            "使语义相近的文本在向量空间中距离更近。\n\n"
            "## 常见模型\n\n"
            "常见的嵌入模型包括 OpenAI text-embedding 系列、BGE、Qwen3-Embedding 等，"
            "输出维度通常在 768 到 4096 之间。嵌入质量直接影响下游检索的效果。\n\n"
            "## 距离度量\n\n"
            "常用的向量距离度量包括：\n"
            "- 余弦相似度（Cosine Similarity）：测量向量夹角，取值 -1 到 1\n"
            "- 欧氏距离（L2）：测量向量直线距离\n"
            "- 内积（Inner Product）：当向量归一化后等价于余弦相似度\n\n"
            "## 与向量数据库的关系\n\n"
            "嵌入模型负责生成向量，向量数据库负责存储和检索向量。"
            "两者共同构成语义检索的基础设施。"
        ),
        "tags": ["embedding", "retrieval", "ai"],
        "entities": [
            {"name": "Cosine Similarity", "type": "metric"},
            {"name": "Euclidean Distance", "type": "metric"},
            {"name": "Inner Product", "type": "metric"},
        ],
        "relations": [
            {"target_title": "向量数据库", "type": "related"},
            {"target_title": "RAG", "type": "related"},
        ],
    },
    {
        "key": "vector_db",
        "title": "向量数据库",
        "topic": "infrastructure",
        "summary": "专门用于存储和检索高维向量的数据库系统，通过 ANN 算法实现快速相似度搜索。",
        "content": (
            "向量数据库（Vector Database）是一种专门用于存储和检索高维向量的数据库系统。"
            "它通过近似最近邻搜索（ANN）算法，在海量向量中快速找到与查询向量最相似的结果，"
            "广泛应用于语义搜索、推荐系统和检索增强生成（RAG）等场景。\n\n"
            "## 与传统数据库的区别\n\n"
            "与传统关系型数据库不同，向量数据库的查询基于向量相似度而非精确匹配。"
            "它通常同时存储向量字段和标量字段，支持向量检索+标量过滤的混合查询。\n\n"
            "## 核心能力\n\n"
            "- 向量插入与存储\n"
            "- 近似最近邻搜索（ANN）\n"
            "- 标量过滤（metadata filtering）\n"
            "- 索引构建（HNSW、IVF、FLAT 等）\n\n"
            "## zvec\n\n"
            "zvec 是一个高性能向量数据库引擎，具有原生 C++ 后端和 Python 绑定，"
            "支持 HNSW、IVF、FLAT 等多种索引类型，以及稠密向量和稀疏向量的混合检索。"
        ),
        "tags": ["vector_db", "index", "retrieval"],
        "entities": [
            {"name": "ANN", "type": "algorithm"},
            {"name": "HNSW", "type": "index"},
            {"name": "IVF", "type": "index"},
            {"name": "zvec", "type": "product"},
        ],
        "relations": [
            {"target_title": "HNSW", "type": "related"},
            {"target_title": "Embedding", "type": "related"},
            {"target_title": "RAG", "type": "related"},
        ],
    },
    {
        "key": "hnsw",
        "title": "HNSW",
        "topic": "infrastructure",
        "summary": "分层可导航小世界图索引，通过多层图结构加速近似最近邻搜索。",
        "content": (
            "HNSW（Hierarchical Navigable Small World）是一种高效的近似最近邻搜索索引结构。"
            "它通过构建多层图来加速检索：上层稀疏图用于快速定位粗粒度区域，"
            "下层稠密图用于精细搜索。\n\n"
            "## 关键参数\n\n"
            "- M（图度数）：每个节点的最大连接数，影响索引精度和内存\n"
            "- ef_construction（建图候选集大小）：建图时的搜索宽度，影响索引质量\n"
            "- ef_search（查询候选集大小）：查询时的搜索宽度，影响召回率\n\n"
            "## 工作原理\n\n"
            "HNSW 从最上层稀疏图开始搜索，逐层下沉到最下层稠密图。"
            "每层使用贪心搜索找到局部最近邻，再进入下一层精细搜索。"
            "这种分层策略在召回率和查询速度之间取得了良好平衡。\n\n"
            "## 应用\n\n"
            "HNSW 是向量数据库中最常用的索引之一，被 zvec、Milvus、Qdrant 等广泛支持。"
        ),
        "tags": ["index", "vector_db", "retrieval"],
        "entities": [
            {"name": "HNSW", "type": "index"},
            {"name": "ANN", "type": "algorithm"},
            {"name": "M", "type": "parameter"},
            {"name": "ef_construction", "type": "parameter"},
        ],
        "relations": [
            {"target_title": "向量数据库", "type": "related"},
        ],
    },
    {
        "key": "agent",
        "title": "Agent",
        "topic": "machine_learning",
        "summary": "LLM 驱动的自主智能体，通过工具调用循环实现多步推理与任务执行。",
        "content": (
            "Agent（智能体）是以大语言模型为核心驱动的自主系统。"
            "它能理解用户意图，自主选择并调用工具，基于工具返回的结果进行多步推理，"
            "最终完成复杂任务。\n\n"
            "## 工具调用循环\n\n"
            "Agent 的核心是 ReAct 循环（Reason + Act）：\n"
            "1. 推理：分析当前状态，决定下一步行动\n"
            "2. 行动：调用工具执行操作\n"
            "3. 观察：接收工具返回结果\n"
            "4. 重复，直到生成最终回答\n\n"
            "## 与 RAG 的关系\n\n"
            "RAG 是单步检索+生成，而 Agent 是多步自主决策。"
            "Agent 可以将 RAG 检索作为其工具之一，"
            "也可以调用图查询、元数据查询等多种工具，实现更灵活的知识访问。\n\n"
            "## 可追溯性\n\n"
            "Agent 的每次工具调用都会记录 trace（工具名、参数、结果、耗时），"
            "使得推理过程可追溯、可审计。"
        ),
        "tags": ["agent", "ai", "llm"],
        "entities": [
            {"name": "ReAct", "type": "pattern"},
            {"name": "Tool Calling", "type": "mechanism"},
            {"name": "Trace", "type": "concept"},
        ],
        "relations": [
            {"target_title": "RAG", "type": "related"},
            {"target_title": "Prompt", "type": "depends"},
        ],
    },
    {
        "key": "prompt",
        "title": "Prompt",
        "topic": "machine_learning",
        "summary": "引导 LLM 生成特定输出的输入文本，包括系统提示、指令、上下文和示例。",
        "content": (
            "Prompt（提示词）是引导大语言模型生成特定输出的输入文本。"
            "良好的提示词工程（Prompt Engineering）能显著提升模型输出质量。\n\n"
            "## 组成要素\n\n"
            "一个完整的 Prompt 通常包含：\n"
            "- 系统提示（System Prompt）：定义模型角色和行为约束\n"
            "- 指令（Instruction）：明确任务要求\n"
            "- 上下文（Context）：提供背景知识（如 RAG 检索结果）\n"
            "- 示例（Few-shot Examples）：展示期望的输入输出格式\n\n"
            "## 提示策略\n\n"
            "常见的提示策略包括：\n"
            "- 零样本（Zero-shot）：不提供示例，直接指令\n"
            "- 少样本（Few-shot）：提供少量示例引导格式\n"
            "- 思维链（Chain-of-Thought）：要求模型逐步推理\n\n"
            "## 与 Agent 的关系\n\n"
            "Agent 的系统提示词定义了可用工具、引用要求和错误守门规则，"
            "是 Agent 行为的关键控制点。"
        ),
        "tags": ["prompt", "llm", "ai"],
        "entities": [
            {"name": "System Prompt", "type": "concept"},
            {"name": "Few-shot", "type": "strategy"},
            {"name": "Chain-of-Thought", "type": "strategy"},
        ],
        "relations": [
            {"target_title": "Agent", "type": "related"},
        ],
    },
    {
        "key": "fine_tuning",
        "title": "Fine-tuning",
        "topic": "machine_learning",
        "summary": "在预训练模型基础上，用领域数据继续训练以适配特定任务或领域。",
        "content": (
            "Fine-tuning（微调）是在已预训练的大语言模型基础上，"
            "使用领域特定数据继续训练，使其适配特定任务或领域的过程。\n\n"
            "## 全量微调 vs 参数高效微调\n\n"
            "- 全量微调（Full Fine-tuning）：更新所有参数，效果最好但成本高\n"
            "- LoRA（Low-Rank Adaptation）：只训练低秩适配矩阵，参数量小\n"
            "- QLoRA：在量化基础上做 LoRA，进一步降低显存需求\n\n"
            "## 与 RAG 的对比\n\n"
            "Fine-tuning 将知识内化到模型参数中，适合需要改变模型行为风格的场景；"
            "RAG 将知识外置于检索库中，适合需要频繁更新知识、要求可溯源的场景。"
            "两者可以结合使用。\n\n"
            "## 流程\n\n"
            "1. 准备领域训练数据（指令-回答对）\n"
            "2. 选择基座模型（如 BERT、LLaMA）\n"
            "3. 配置训练参数（学习率、epoch、batch size）\n"
            "4. 训练并评估\n"
            "5. 部署微调后的模型"
        ),
        "tags": ["fine_tuning", "llm", "ai"],
        "entities": [
            {"name": "LoRA", "type": "method"},
            {"name": "QLoRA", "type": "method"},
            {"name": "Full Fine-tuning", "type": "method"},
        ],
        "relations": [
            {"target_title": "BERT", "type": "related"},
            {"target_title": "RAG", "type": "related"},
        ],
    },
]


# ====================================================================== #
#  入库逻辑
# ====================================================================== #
def _chunk_document(doc: dict) -> list[dict]:
    """将文档按句号切分为 chunk（用于 zvec 向量入库）。"""
    content = doc["content"]
    # 按双换行（段落）或句号切分
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    ordinal = 0
    for para in paragraphs:
        sentences = [s.strip() + "。" for s in para.replace("。", "。\n").split("\n") if s.strip()]
        # 每 N 句合并为一个 chunk
        for i in range(0, len(sentences), config.CHUNK_SENTENCES):
            chunk_text = "".join(sentences[i:i + config.CHUNK_SENTENCES])
            if not chunk_text:
                continue
            ordinal += 1
            chunk_id = f"{doc['key']}_c{ordinal}"
            chunks.append({
                "id": chunk_id,
                "text": f"{doc['title']}。{chunk_text}",
                "fields": {
                    "document_id": f"document:{doc['key']}",
                    "title": doc["title"],
                    "heading": doc["title"],
                    "content": chunk_text,
                },
            })
    return chunks


def seed_all_sync() -> dict:
    """灌入图/文档/实体/标签种子数据（SurrealDB，快速同步完成）。

    向量入库由 seed_vectors() 异步执行，不阻塞调用方。
    幂等：若已有文档则跳过。
    """
    # 检查是否已灌入
    st = wr.stats()
    if st["documents"] > 0:
        return {"skipped": True, "reason": "已有文档，跳过种子灌入", "stats": st}

    # 1. 灌入 topics
    for key, name, desc in TOPICS:
        wr.ensure_topic(key, name, desc)

    # 2. 灌入 tag 层级
    for key, name, parent in TAG_TREE:
        wr.ensure_tag(key, name, parent)

    # 3. 灌入文档 + 关系 + 实体
    all_chunks: list[dict] = []
    for doc in DOCUMENTS:
        wr.create_document(
            title=doc["title"],
            content=doc["content"],
            summary=doc["summary"],
            topic_id=doc["topic"],
            doc_key=doc["key"],
            tags=doc["tags"],
            entities=doc["entities"],
            relations=doc["relations"],
        )
        all_chunks.extend(_chunk_document(doc))

    st = wr.stats()
    return {
        "skipped": False,
        "documents": st["documents"],
        "topics": st["topics"],
        "tags": st["tags"],
        "entities": st["entities"],
        "chunks": all_chunks,  # 返回 chunk 列表，供 seed_vectors 使用
    }


def seed_vectors(chunks: list[dict]) -> int:
    """异步向量入库：注册嵌入函数 → 建集合 → 批量嵌入入库。

    chunks: seed_all_sync() 返回的 chunks 列表（[{id, text, fields}, ...]）。
    """
    dimension = zvec_client.discover_dimension()
    zvec_client.register_embedding(dimension)
    zvec_client.create_collection(dimension)
    count = zvec_client.ingest_chunks(chunks)
    print(f"  向量库已灌入 {count} 个分块 (dim={dimension})")
    return count


def seed_all() -> dict:
    """完整灌入（图/文档同步 + 向量入库）。幂等。

    注意：向量入库可能耗时较长（每个 chunk 一次嵌入调用）。
    生产环境建议用 seed_all_sync() + seed_vectors() 异步分离。
    """
    result = seed_all_sync()
    if result.get("skipped"):
        return result
    chunks = result.pop("chunks")
    result["chunks"] = len(chunks)
    result["zvec_seeded"] = False
    try:
        health = zvec_client.check_health()
        if health["zvec"] and health["has_embed_model"]:
            seed_vectors(chunks)
            result["zvec_seeded"] = True
        else:
            print("  ⚠ zvec 或 Ollama 嵌入模型不可达，跳过向量入库（图/全文/元数据检索仍可用）")
    except Exception as e:
        print(f"  ⚠ 向量入库失败: {e}（图/全文/元数据检索仍可用）")
    return result
