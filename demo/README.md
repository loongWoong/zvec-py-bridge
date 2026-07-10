# RAG 知识库演示

通过 **zvec REST Bridge** 连接向量数据库，使用本地 **Ollama** 的 `qwen3-embedding:4b` 模型进行文本向量化，端到端验证向量库的 RAG 流程：

```
知识入库（文本→向量→存储） → 语义检索 → 生成回答
```

## 架构

```
┌──────────────┐      HTTP/JSON      ┌──────────────────┐      OpenAI API      ┌────────┐
│  rag_demo.py │  ───────────────►   │  zvec REST Bridge │  ────────────────►  │ Ollama │
│  (本程序)     │   注册/插入/检索     │  (FastAPI:8666)   │   qwen3-embedding    │ :11434 │
└──────────────┘                     └──────────────────┘                      └────────┘
                                           │
                                           ▼
                                     zvec 向量引擎
                                     (持久化存储)
```

## 前置条件

### 1. 启动 zvec REST Bridge

```bash
cd server
pip install -r requirements.txt
pip install openai          # Ollama 兼容 OpenAI 接口，服务端需要此依赖
python main.py              # 默认监听 0.0.0.0:8666
```

验证服务就绪：

```bash
curl http://localhost:8666/health
# {"status":"UP","zvec_version":"0.5.1"}
```

### 2. 启动 Ollama 并拉取模型

```bash
# 拉取嵌入模型
ollama pull qwen3-embedding:4b

# 拉取生成模型（可选，用于 RAG 回答生成步骤）
ollama pull qwen3:4b
```

## 运行

```bash
cd demo
pip install -r requirements.txt
python rag_demo.py
```

### 配置

通过环境变量覆盖默认地址：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ZVEC_URL` | `http://localhost:8666` | zvec REST Bridge 地址 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_MODEL` | `qwen3-embedding:4b` | 嵌入模型名 |
| `LLM_MODEL` | `qwen3:4b` | 生成模型名（RAG 回答用） |

示例：

```bash
ZVEC_URL=http://localhost:8000 python rag_demo.py
```

## 验证流程

| 步骤 | 说明 |
|------|------|
| 1. 健康检查 | 确认 zvec 服务和 Ollama 均在线，嵌入模型已安装 |
| 2. 注册嵌入函数 | 向 zvec 注册 `openai` 类型嵌入函数，指向 Ollama 的 `/v1` 端点 |
| 3. 发现向量维度 | 嵌入一段样本文本，自动获取 `qwen3-embedding:4b` 的输出维度 |
| 4. 创建集合 | 创建含 `VECTOR_FP32` 向量字段和 `STRING` 标量字段的集合 |
| 5. 知识入库 | 批量插入 8 篇知识文档，文本自动嵌入为向量存储 |
| 6. 语义检索 | 用 5 个测试查询验证 Top-1 命中预期文档 |
| 7. RAG 问答 | 检索相关文档 → 拼接上下文 → 调用 Ollama 生成回答 |
| 8. 清理 | 删除集合、注销嵌入函数，保证可重复运行 |

## 知识库内容

内置 8 篇 RAG / 向量数据库相关中文文档：向量数据库、嵌入模型、RAG、HNSW、余弦相似度、分块策略、混合检索、重排序。
