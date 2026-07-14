# Semantic Wiki Runtime

一个基于 **SurrealDB（图+文档存储）+ zvec（向量检索）+ LLM（编译/抽取/问答）** 的语义 Wiki 运行时。

将 Karpathy 式的 Markdown Wiki 升级为 **Semantic Wiki**：数据库是真实来源，Markdown 是导出产物；文档、图、本体、向量四位一体。

## 架构

```
                +----------------------------+
                |          LLM Agent         |
                +----------------------------+
                           │
              Natural Language / Tool Calls
                           │
──────────────────────────────────────────────────────
                Wiki Semantic Runtime
──────────────────────────────────────────────────────

      Document Service    Graph Service
      Metadata Service    Search Service (四路融合)

──────────────────────────────────────────────────────
             SurrealDB              zvec REST Bridge
        (图 / 全文 / 元数据)        (向量语义检索)
```

## 快速开始

### 1. 安装依赖

```bash
cd wiki-demo
pip install -r requirements.txt
```

### 2. 启动外部服务（可选，但推荐）

- **zvec REST Bridge**（向量检索路）：启动 `server/` 下的 zvec 桥接服务，默认 `http://localhost:8666`
- **Ollama**（嵌入模型）：`ollama pull qwen3-embedding:0.6b`
- **LLM 服务**（编译/抽取/问答）：OpenAI 兼容端点或 Ollama

> 若以上服务不可达，应用仍可启动 — 图/全文/元数据三路检索可用，仅向量检索路和 LLM 功能降级。

### 3. 启动

```bash
python app.py
```

浏览器访问 **http://localhost:8090**

## 功能

| 视图 | 能力 |
|------|------|
| **文档库** | 浏览/搜索文档，查看全文与关系邻接，导出 Markdown，编译文档（raw→Wiki），版本历史 |
| **知识图谱** | 全局图可视化（文档/实体/标签为节点，关系为边），按类型过滤，图查询（最短路径/共同邻居/知识血缘/度中心性） |
| **元数据** | 浏览主题、标签树、实体列表、原始资料（RawSource） |
| **问答** | LLM Agent 通过 Wiki Runtime Tool 自主检索并回答，带可追溯 trace，热门文档排行 |

### 四路融合检索

搜索栏触发四路融合检索，结果标注来源路：

| 路 | 引擎 | 用途 |
|----|------|------|
| **vector** | zvec + Ollama embedding | 语义相似度召回 |
| **fts** | SurrealDB 全文索引 | 关键词匹配 |
| **graph** | SurrealDB 图遍历 | 实体 → mentions 反查文档 |
| **meta** | SurrealDB 元数据 | author=/topic= 结构化过滤 |

各路结果按倒数排名融合（RRF）去重排序。

### LLM 三职责

1. **编译文档**（`POST /api/compile`）：从 raw 源编译生成 WikiDocument（含 summary + entities + tags + relations）
2. **抽取实体**（`POST /api/extract`）：对已有文档抽取 Entity/Tag 并建立关系边
3. **Agent 问答**（`POST /api/ask`）：LLM 自主选择工具检索，生成带引用的回答

## 配置

所有配置通过环境变量覆盖（见 `config.py`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SURREAL_DB` | `file://wiki_data.db` | SurrealDB 嵌入式连接串（`memory` 或 `file://...`） |
| `ZVEC_URL` | `http://localhost:8666` | zvec REST Bridge 地址 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | 嵌入模型 |
| `LLM_URL` | `http://127.0.0.1:8000` | LLM 服务地址 |
| `LLM_API` | `openai` | `openai` 或 `ollama` |
| `LLM_MODEL` | `hy3` | 生成模型 |
| `WIKI_PORT` | `8090` | Web 服务端口 |

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 统计 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/{id}` | 文档详情（含关系邻接） |
| POST | `/api/documents` | 创建文档 |
| PUT | `/api/documents/{id}` | 更新文档（自动存版本） |
| DELETE | `/api/documents/{id}` | 删除文档 |
| GET | `/api/documents/{id}/export` | 导出 Markdown |
| GET | `/api/documents/{id}/versions` | 版本历史（离散快照） |
| GET | `/api/documents/{id}/version-chain` | 完整版本链（previous_version 边遍历） |
| GET | `/api/documents/{id}/related` | 相关文章 |
| GET | `/api/documents/{id}/lineage` | 知识血缘（上下游知识链） |
| GET | `/api/documents/{id}/conversations` | 文档关联对话记录 |
| POST | `/api/documents/{id}/link-raw` | 关联文档与 raw 源 |
| POST | `/api/documents/{id}/build-graph` | 对文档抽取实体并建图边 |
| POST | `/api/documents/merge` | 合并两篇文档 |
| POST | `/api/documents/update-metadata` | 更新文档元数据 |
| GET | `/api/graph/full` | 全图数据（D3 渲染） |
| GET | `/api/graph/stats` | 全图统计 |
| GET | `/api/graph/central` | 度中心性排行 |
| GET | `/api/graph/{id}` | 图邻接遍历 |
| GET | `/api/graph/{id}/subtree` | 局部子图 |
| GET | `/api/graph/{id}/shortest-path` | BFS 最短路径 |
| GET | `/api/graph/{id}/common` | 共同邻居 |
| GET | `/api/graph/{id}/degree` | 度中心性 |
| GET | `/api/graph/{id}/multi-hop` | 多跳邻接 |
| GET | `/api/topics` `/api/tags` `/api/entities` | 元数据列表 |
| GET | `/api/entities/{name}` | 实体查询（含被哪些文档提及） |
| GET | `/api/entities/{name}/co-occurrence` | 实体共现分析 |
| GET | `/api/raws` | raw 源列表 |
| GET | `/api/raws/{id}` | 获取单个 raw 源 |
| POST | `/api/raws` | 创建 raw 源 |
| GET | `/api/archives` | archive 文档列表 |
| POST | `/api/archives` | 创建 archive 文档 |
| GET | `/api/conversations/hot` | 热门文档排行（被问得最多） |
| POST | `/api/conversations` | 记录对话到 Memory Graph |
| GET | `/api/search?q=...` | 四路融合检索 |
| POST | `/api/search` | 四路融合检索（POST） |
| POST | `/api/ask` | Agent 问答（自动记录对话） |
| POST | `/api/compile` | LLM 编译文档 |
| POST | `/api/extract` | LLM 抽取实体 |

## 种子数据

启动时若库为空，自动灌入 10 篇 LLM 主题文档（Transformer/Attention/BERT/RAG/Embedding/向量数据库/HNSW/Agent/Prompt/Fine-tuning），含 topic、tag 层级、entity 节点、关系边，2 条 raw 源（关联到 Transformer 和 RAG），并向量入库。

## 文件结构

```
wiki-demo/
├── design.md           # 设计方案
├── README.md           # 本文件
├── requirements.txt    # 依赖
├── config.py           # 配置
├── db.py               # SurrealDB 连接与 Schema
├── zvec_client.py      # zvec 向量检索客户端
├── wiki_runtime.py     # 核心 Runtime（CRUD + 图 + 四路检索）
├── llm.py              # LLM 三职责
├── seed.py             # 种子数据
├── app.py              # FastAPI Web 应用
└── static/
    └── index.html      # 单页前端
```
