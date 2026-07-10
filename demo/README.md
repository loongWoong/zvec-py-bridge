# RAG 知识库演示

通过 **zvec REST Bridge** 连接向量数据库，使用本地 **Ollama** 的 `qwen3-embedding:4b` 模型进行文本向量化，端到端验证向量库的 RAG 流程：

```
知识入库（文本→向量→存储） → 语义检索 → 生成回答
```

提供两种使用方式：**Web 界面**（可交互）和 **CLI 验证**（自动化测试）。

## 架构

```
┌────────────────┐     HTTP/JSON     ┌──────────────────┐    OpenAI API    ┌────────┐
│  浏览器 / CLI   │ ──────────────►  │  zvec REST Bridge │ ──────────────► │ Ollama │
│                │   注册/插入/检索   │  (FastAPI:8666)   │  qwen3-embedding │ :11434 │
└────────────────┘                   └──────────────────┘                  └────────┘
       │                                      │
       │ Web UI (:8080)                        ▼
       └─────────────► web_app.py (FastAPI)  zvec 向量引擎
```

## 文件结构

```
demo/
├── web_app.py          # Web 后端（FastAPI，端口 8080）
├── agent.py            # OAG Agent：本体 + 工具 + agent 循环
├── rag_demo.py         # CLI 自动化验证脚本
├── kb_data.py          # 共享模块：配置、语料、核心操作
├── static/
│   └── index.html      # Web 前端（单页面应用）
├── requirements.txt    # 依赖
└── README.md
```

## 前置条件

### 1. 启动 zvec REST Bridge

```bash
cd server
pip install -r requirements.txt
pip install openai          # Ollama 兼容 OpenAI 接口，服务端需要此依赖
python main.py              # 默认监听 0.0.0.0:8666
```

验证：`curl http://localhost:8666/health` → `{"status":"UP",...}`

### 2. 启动 Ollama 并拉取模型

```bash
ollama pull qwen3-embedding:4b    # 嵌入模型（必需）
ollama pull qwen3:4b              # 生成模型（RAG 回答用，可选）
```

## 方式一：Web 界面（推荐）

```bash
cd demo
pip install -r requirements.txt
python web_app.py              # 默认端口 8080
```

浏览器访问 **http://localhost:8080**，操作流程：

1. 点击 **「初始化知识库」** — 自动发现维度、注册嵌入函数、创建集合、入库 8 篇文档
2. 在输入框输入问题，或点击示例问题
3. 切换 **RAG 模式 / Agent 模式**：
   - **RAG 模式**：`检索 → 拼接上下文 → 生成回答`（单步，固定流程）
   - **Agent 模式（OAG）**：`LLM 自主选择工具 → runtime 执行 → 整合结果`（多步，可追溯）
4. RAG 模式下点击 **「🔍 检索」** 查看语义检索结果，或 **「🤖 RAG 问答」** 获取 AI 回答
5. Agent 模式下点击 **「🤖 Agent 问答」**，查看 Agent 执行轨迹（工具调用链）+ 带引用的回答
6. 点击 **「🗑 清理知识库」** 释放资源

### Agent 模式（OAG）

Agent 模式实现了设计文档中的 OAG（Ontology-Agent-Generation）模式，与传统 RAG 的区别：

| | 传统 RAG | OAG Agent |
|---|---|---|
| 流程 | 检索→总结（单步） | LLM 选工具→执行→整合（多步） |
| 工具选择 | 固定（总是检索+生成） | LLM 自主决策 |
| 可追溯 | 仅检索结果 | 完整工具调用轨迹 |
| 引用 | 可选 | 强制要求 `[doc_id] title` |

**本体层**：将语料结构化为 `Document` + `DocumentChunk` 对象（按句切分，保留 heading/ordinal）。

**工具层**（4 个领域专用函数，各含 `usage_prompt`）：

| 工具 | 层 | 产物 |
|------|-----|------|
| `search_documents` | 定位 | 轻量证据片段（FTS/语义/rerank） |
| `list_documents` | 定位 | 文档元数据列表 |
| `read_document` | 核验 | 单篇全文/分块 |
| `prepare_answer_context` | 综合 | 多文档证据包 + 综合提纲 |

**Agent 循环**：Ollama 原生 tool calling，LLM 自主选择工具 → 确定性 runtime 执行 → 结果送回 LLM → 生成带引用的回答。最多 6 轮迭代，每步记录 trace（工具名、参数、结果、耗时）。

## 方式二：CLI 验证

```bash
cd demo
pip install -r requirements.txt
python rag_demo.py
```

运行 8 个步骤的自动化验证（23 项断言），输出每步的 ✅/❌ 结果。

## 配置

通过环境变量覆盖默认地址（对所有方式生效）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ZVEC_URL` | `http://localhost:8666` | zvec REST Bridge 地址 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_MODEL` | `qwen3-embedding:4b` | 嵌入模型名 |
| `LLM_MODEL` | `qwen3:4b` | 生成模型名（RAG 回答用） |
| `WEB_HOST` | `0.0.0.0` | Web 服务监听地址 |
| `WEB_PORT` | `8080` | Web 服务端口 |

示例：

```bash
LLM_MODEL=gemma4:e4b python web_app.py
```

## 知识库内容

内置 8 篇 RAG / 向量数据库相关中文文档：向量数据库、嵌入模型、RAG、HNSW、余弦相似度、分块策略、混合检索、重排序。
