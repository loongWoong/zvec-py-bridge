# Semantic Wiki Runtime

一个基于 **SurrealDB（图+文档存储）+ zvec（向量检索）+ LLM（编译/抽取/问答）+ 本体层（概念/关系/绑定）** 的语义 Wiki 运行时。

将 Karpathy 式的 Markdown Wiki 升级为 **Semantic Wiki**：数据库是真实来源，Markdown 是导出产物；文档、图、本体、向量、代码结构五位一体，并提供 **AI 推理引擎**（设计 `docs/AI推理引擎.md` 的 Step 1–10）驱动的可追溯问答。

## 架构

```
                          用户提问（自然语言）
                                │
                 ┌──────────────▼──────────────┐
                 │      AI 推理引擎（在线）       │  Step 5 概念定位
                 │  concept_locator → ontology   │  Step 6 检索计划
                 │  _traversal → reranker        │  Step 7 并行检索
                 │  → LLM 闭环评估               │  Step 8 重排+过滤
                 └──────────────┬──────────────┘  Step 9 闭环迭代
                                │  Step 10 合成输出
──────────────────────────────────────────────────────────────
                Wiki Semantic Runtime
──────────────────────────────────────────────────────────────
   Document    Graph      Ontology     Search(四路融合)   Code
   Service     Service    Service     + Re-Rank         Analyzer
──────────────────────────────────────────────────────────────
            SurrealDB                  zvec REST Bridge
       (图 / 全文 / 元数据 / 本体)        (向量语义检索)
```

离线阶段（Step 1–4）把原始资料变成可检索的结构化知识；在线阶段（Step 5–10）把用户问题沿推理链变成带引用的答案。

## 快速开始

### 1. 安装依赖

```bash
cd wiki-demo
pip install -r requirements.txt
# 可选：PDF 解析（路径导入 PDF 时需要）
pip install pdfplumber
```

### 2. 启动外部服务（可选，但推荐）

- **zvec REST Bridge**（向量检索路）：启动 `server/` 下的 zvec 桥接服务，默认 `http://localhost:8666`
- **Ollama**（嵌入模型）：`ollama pull qwen3-embedding:0.6b`
- **LLM 服务**（编译/抽取/问答/本体构建）：OpenAI 兼容端点或 Ollama

> 若以上服务不可达，应用仍可启动 — 图/全文/元数据三路检索可用，仅向量检索路和 LLM 功能降级。

### 3. 启动

```bash
python app.py
```

浏览器访问 **http://localhost:8090**。首次启动若库为空，自动灌入种子数据（10 篇主题文档 + 34 个本体概念，见下文「种子数据」）。

## 两阶段心智模型

前端 5 个标签按 **建图 → 导航** 两阶段排列，对应设计的离线/在线阶段：

| 阶段 | 标签 | 对应设计 Step |
|------|------|---------------|
| 离线·建图 | 文档库 / 本体 / 知识图谱 | Step 1 解析切分 / Step 2 本体 / Step 3 嵌入 / Step 4 多索引 |
| 在线·导航 | 问答 / 分析 | Step 5 概念定位 → Step 10 合成输出 |

## 核心功能与用户使用流程（示例）

### 流程 A：离线建图——从代码仓库建知识库

> 目标：把一个 Python 项目导入知识库，让 Agent 能回答"认证模块依赖了哪些概念"。

1. **打开「文档库」→ `+ 编译文档`**，选择模式：
   - **LLM 编译**：上传 `.md/.txt`（或选文件夹），LLM 把原始资料编译为结构化 Wiki（抽取摘要/标签/实体/关系）。适合把散乱笔记变成 Wiki。
   - **路径导入（多格式自动索引）**：点 **选择文件夹** 选整个项目目录，或 **选择文件** 多选。支持 `.pdf .py .js .html .md .yaml .json …` 全格式，自动按类型切分（代码按函数/类边界、Markdown 按标题、YAML 整文件）→ 写 SurrealDB + 写向量库。也可填服务端绝对路径走 `ingest-directory`。
2. **切到「本体」**：点 **LLM 构建**，让 LLM 扫描文档摘要提议概念/关系/绑定；在待审核区 review，或手动 **+ 新建概念** 定义层级（如 `auth → depends_on → session`）。概念可 **绑定文档**，使在线问答能按概念定位。
3. **切到「知识图谱」**：可视化文档/实体/标签/概念为节点。点节点看邻接，用图查询（最短路径/共同邻居/知识血缘）验证建图质量。空状态会引导你先建图。

完成后即拥有：可语义检索的文档库 + 概念本体 + 图结构 + 向量索引。

### 流程 B：在线问答——沿推理链得到可追溯答案

> 目标：问"Transformer 的注意力机制和 RAG 的检索增强有什么关系？"，并看清 Agent 是怎么推理的。

1. **切到「问答」**，输入问题（或点示例问题/热门文档）。Agent 执行：
   - **Step 5 概念定位**：`concept_locator` 把问题映射到本体概念（如 `attention`、`rag`），带置信度与隐含概念。
   - **Step 6 检索计划**：`ontology_traversal` 沿本体关系展开，生成 5 种并行策略（向量/全文/图/调用链/元数据）。
   - **Step 7 检索执行**：四路融合检索 + 代码结构化检索并行召回候选 chunk。
   - **Step 8 重排+过滤**：`reranker` 多因子打分（语义+概念距离+结构+新鲜度），并按设计规则过滤（概念距离过远且语义弱→丢弃；同概念 footprint 去重）。
   - **Step 9 闭环迭代**：LLM 评估证据是否充分，不足则补检索（广撒网→精确补充→验证性），多轮直到满足。
   - **Step 10 合成**：生成带引用的答案。
2. 答案下方 **推理链时间线** 可视化上述每步：概念定位（紫）、检索计划（蓝）、检索执行（绿）、推理路径（琥珀，含可点击证据文档）、闭环迭代（红，按轮次分组工具调用）。点证据文档 chip 直接跳文档库检索。
3. **跨视图导航**：在「本体」点某概念 → "以此概念提问"；在「文档库」点某文档 → "就此文档提问"；检索结果 → "提问"。一键把上下文带进问答。

### 流程 C：本体维护与知识库巡检

> 目标：定期让本体跟上文档变化，清理漂移绑定。

1. **切到「分析」**：查看查询日志排行、增量索引（`reindex-changed` 检测文件变化重切分）、本体版本管理入口（已统一到本体视图）。
2. **切到「本体」→ 工具栏**：
   - **修复绑定**（`repair-bindings`）：重新计算概念↔文档绑定，清理失效项。
   - **快照 / 回滚**：把当前本体导出 YAML 快照版本化，出错可回滚。
   - **导出 / 导入 YAML**：人工 Git 管理本体。
3. **定时维护**：`maintenance_scheduler` 可周期跑本体重建/绑定修复/向量同步，状态与报告在分析页查看。

## 功能视图

| 视图 | 能力 |
|------|------|
| **文档库** | 浏览/搜索文档；查看全文与关系邻接；导出 Markdown；LLM 编译（raw→Wiki）；路径导入（多格式自动索引）；代码结构化检索；元数据浏览；版本历史 |
| **本体** | 概念 CRUD + 关系 + 文档绑定；LLM 构建提议；待审核 review；快照/回滚；YAML 导入导出；按概念提问 |
| **知识图谱** | 全局图可视化（文档/实体/标签/概念为节点）；按类型过滤；图查询（最短路径/共同邻居/知识血缘/度中心性/多跳） |
| **问答** | LLM Agent 沿 Step 5–9 推理链自主检索并回答；推理链时间线可视化；可点击证据文档；跨视图导航；热门文档排行 |
| **分析** | 查询日志；增量索引；本体版本管理入口；维护调度（启动/停止/报告/提议） |

### 四路融合检索 + Re-Rank

搜索栏触发四路融合检索，结果标注来源路，再经多因子 Re-Rank：

| 路 | 引擎 | 用途 |
|----|------|------|
| **vector** | zvec + Ollama embedding | 语义相似度召回 |
| **fts** | SurrealDB 全文索引 | 关键词匹配 |
| **graph** | SurrealDB 图遍历 | 实体 → mentions 反查文档 |
| **meta** | SurrealDB 元数据 | author=/topic= 结构化过滤 |

`reranker` 在融合基础上叠加：语义分 + 概念距离分（本体图上与目标概念的距离）+ 结构相关性（同文件/模块/调用链）+ 新鲜度，并执行 Step 8 过滤规则。

### LLM 职责

1. **编译文档**（`POST /api/compile`、`/api/compile-batch`）：raw 源 → WikiDocument（summary + entities + tags + relations）
2. **抽取实体**（`POST /api/extract`）：对已有文档抽取 Entity/Tag 并建关系边
3. **Agent 问答**（`POST /api/ask`）：自主选工具检索，生成带引用回答，返回完整推理链（concept_location / ontology_path / trace）
4. **本体构建**（`POST /api/ontology/build`）：扫描文档提议概念/关系/绑定
5. **概念定位**（集成于 `/api/ask` Step 5）：问题 → 本体概念多选

## 配置

所有配置通过环境变量覆盖（见 `config.py`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SURREAL_DB` | `file://wiki_data.db` | SurrealDB 连接串（`memory` 或 `file://...`） |
| `SURREAL_NS` | `wiki` | 命名空间 |
| `SURREAL_DB_NAME` | `wiki` | 数据库 |
| `SURREAL_USER` / `SURREAL_PASS` | `root` / `root` | 凭据 |
| `ZVEC_URL` | `http://localhost:8666` | zvec REST Bridge 地址 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | 嵌入模型 |
| `LLM_URL` | `http://127.0.0.1:8000` | LLM 服务地址 |
| `LLM_API` | `openai` | `openai` 或 `ollama` |
| `LLM_API_KEY` | `sk-123` | LLM API Key |
| `LLM_MODEL` | `deepseek-v4-flash` | 生成模型 |
| `LLM_TIMEOUT` | `120` | LLM 请求超时（秒） |
| `LLM_HEALTH_TIMEOUT` | `5` | LLM 健康检查超时（秒） |
| `WIKI_HOST` | `0.0.0.0` | Web 监听地址 |
| `WIKI_PORT` | `8090` | Web 服务端口 |
| `CHUNK_SENTENCES` | `3` | 文本分块句子数 |

## API 一览

### 基础与文档
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 统计 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/{id}` | 文档详情（含关系邻接） |
| POST | `/api/documents` | 创建文档 |
| PUT | `/api/documents/{id}` | 更新文档（自动存版本） |
| DELETE | `/api/documents/{id}` | 删除文档 |
| POST | `/api/documents/check-duplicates` | 按标题查重 |
| GET | `/api/documents/{id}/export` | 导出 Markdown |
| GET | `/api/documents/{id}/versions` | 版本历史（离散快照） |
| GET | `/api/documents/{id}/version-chain` | 完整版本链 |
| GET | `/api/documents/{id}/related` | 相关文章 |
| GET | `/api/documents/{id}/lineage` | 知识血缘 |
| GET | `/api/documents/{id}/conversations` | 文档关联对话记录 |
| POST | `/api/documents/{id}/link-raw` | 关联文档与 raw 源 |
| POST | `/api/documents/{id}/build-graph` | 对文档抽取实体并建图边 |
| POST | `/api/documents/{id}/sync-vectors` | 重建文档向量索引 |
| GET | `/api/documents/{id}/code-symbols` | 文档代码符号 |
| POST | `/api/documents/merge` | 合并两篇文档 |
| POST | `/api/documents/update-metadata` | 更新文档元数据 |

### 入库管道（Step 1+3+4）
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/compile` | LLM 编译单文档 |
| POST | `/api/compile-batch` | 批量上传 .md/.txt LLM 编译 |
| POST | `/api/pipeline/ingest-file` | 服务端路径单文件入库（多格式） |
| POST | `/api/pipeline/ingest-directory` | 服务端路径目录批量入库 |
| POST | `/api/pipeline/ingest-upload` | 浏览器上传文件/文件夹批量入库（多格式） |
| POST | `/api/pipeline/reindex-changed` | 增量重索引变更文件 |

### 图与元数据
| 方法 | 路径 | 说明 |
|------|------|------|
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
| GET | `/api/raws` `/api/raws/{id}` | raw 源列表/详情 |
| POST | `/api/raws` | 创建 raw 源 |
| GET | `/api/archives` | archive 文档列表 |
| POST | `/api/archives` | 创建 archive 文档 |

### 本体层（Step 2 / Step 5 / Step 6）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/ontology/stats` | 本体统计 |
| GET | `/api/ontology/concepts` | 概念列表 |
| GET | `/api/ontology/graph` | 本体图 |
| GET | `/api/ontology/concepts/{id}` | 概念详情 |
| GET | `/api/ontology/concepts/{id}/expand` | 沿关系展开 N 跳 |
| POST | `/api/ontology/concepts` | 创建概念 |
| POST | `/api/ontology/relations` | 创建关系 |
| POST | `/api/ontology/bindings` | 创建概念↔文档绑定 |
| DELETE | `/api/ontology/concepts/{id}` | 删除概念 |
| GET | `/api/ontology/export` | 导出 YAML |
| POST | `/api/ontology/import` | 导入 YAML |
| POST | `/api/ontology/build` | LLM 提议本体 |
| GET | `/api/ontology/review/pending` | 待审核提议 |
| POST | `/api/ontology/review` | 审核提议 |
| POST | `/api/ontology/snapshot` | 创建快照 |
| POST | `/api/ontology/rollback` | 从快照回滚 |

### 检索与问答（Step 5–10）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/search?q=...` | 四路融合检索 |
| POST | `/api/search` | 四路融合检索（POST） |
| GET | `/api/code-search` | 代码结构化检索（函数/调用链） |
| POST | `/api/code-search` | 代码结构化检索（POST） |
| GET | `/api/search/grep` | grep 全文检索 |
| POST | `/api/repair-bindings` | 修复概念↔文档绑定 |
| POST | `/api/ask` | Agent 问答（返回推理链） |

### 对话与分析维护
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/conversations/hot` | 热门文档排行 |
| POST | `/api/conversations` | 记录对话到 Memory Graph |
| GET | `/api/analytics/query-log` | 查询日志 |
| POST | `/api/maintenance/run` | 立即运行维护 |
| GET | `/api/maintenance/status` | 调度状态 |
| POST | `/api/maintenance/start` | 启动定时维护 |
| POST | `/api/maintenance/stop` | 停止定时维护 |
| GET | `/api/maintenance/report` | 维护报告 |
| GET | `/api/maintenance/proposals` | 维护提议 |

## 种子数据

启动时若库为空，自动灌入：
- **10 篇 LLM 主题文档**：Transformer / Attention / BERT / RAG / Embedding / 向量数据库 / HNSW / Agent / Prompt / Fine-tuning，含 topic、tag 层级、entity 节点、关系边，2 条 raw 源（关联到 Transformer 和 RAG），并向量入库。
- **34 个本体概念**：`ontology/concepts.yaml` 定义的概念层级 + 关系 + 绑定，使开箱即可体验 Step 5 概念定位。

## 文件结构

```
wiki-demo/
├── docs/                    # 设计与分析文档
│   └── AI推理引擎.md         # 推理引擎设计（Step 1–10）
├── ontology/
│   └── concepts.yaml        # 种子本体（36 概念）
├── requirements.txt         # 依赖
├── config.py                # 配置（环境变量）
├── db.py                    # SurrealDB 连接与 Schema
├── zvec_client.py           # zvec 向量检索客户端
├── wiki_runtime.py          # 核心 Runtime（CRUD + 图 + 四路检索）
├── pipeline.py              # 入库管道（文件/目录/上传 → 切分 → DB+向量）
├── chunker.py               # 文档智能切分（Markdown/代码/文本/YAML）
├── code_analyzer.py         # 代码结构化索引（函数/调用链，策略5）
├── llm.py                   # LLM 职责（编译/抽取/问答/本体/概念定位）
├── ontology.py              # 本体层（概念/关系/绑定 CRUD + YAML）
├── ontology_builder.py      # LLM 辅助本体构建（提议概念/关系/绑定）
├── ontology_traversal.py    # 本体展开 + 检索计划生成（Step 6）
├── concept_locator.py       # 概念定位器（Step 5，问题→本体概念）
├── reranker.py              # 多因子 Re-Rank + Step 8 过滤规则
├── maintenance_scheduler.py # 本体定期维护调度（W4.2）
├── seed.py                  # 种子数据（文档 + 本体）
├── app.py                   # FastAPI Web 应用（78 端点）
└── static/
    └── index.html           # 单页前端（5 标签 + 推理链可视化）
```
