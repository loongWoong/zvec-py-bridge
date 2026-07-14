# Semantic Wiki Runtime — 全问题修复计划

## 目标
解决前序分析发现的全部 10 个问题，使 `wiki-demo` 对 `design.md` 的完成度从 ~75% 提升至 ~95%+。

---

## 修改清单（按文件分组）

### 1. `db.py` — Schema 补全

**新增对象表：**
```sql
DEFINE TABLE archive SCHEMALESS;
DEFINE FIELD title     ON archive TYPE string;
DEFINE FIELD content   ON archive TYPE string;
DEFINE FIELD source    ON archive TYPE option<string>;
DEFINE FIELD archived  ON archive TYPE datetime DEFAULT time::now();

DEFINE TABLE conversation SCHEMALESS;
DEFINE FIELD question  ON conversation TYPE string;
DEFINE FIELD answer    ON conversation TYPE option<string>;
DEFINE FIELD created   ON conversation TYPE datetime DEFAULT time::now();
```

**新增边表（补 design §6 缺失的 4 种 + §7 的 updated_by + §8 的 about）：**
```sql
-- design §6 补全
DEFINE TABLE duplicates    TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE same_topic    TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE derived_from  TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE archived_from TYPE RELATION FROM document TO archive SCHEMALESS;
-- design §7 知识血缘
DEFINE TABLE updated_by    TYPE RELATION FROM document TO raw SCHEMALESS;
-- design §8 LLM Memory Graph
DEFINE TABLE about        TYPE RELATION FROM conversation TO document SCHEMALESS;
```

**新增全文检索索引（解决 §9 降级问题）：**
```sql
DEFINE ANALYZER IF NOT EXISTS simple_bm25 TOKENIZERS blank, class FILTERS lowercase;
DEFINE INDEX IF NOT EXISTS doc_search ON document FIELDS title, content, summary SEARCH ANALYZER simple_bm25 BM25 HIGHLIGHTS;
```

---

### 2. `wiki_runtime.py` — Runtime 功能补全

**2a. RawSource 管理（design §1 缺失）**
- 新增 `create_raw(url, author, published, content) -> dict` — UPSERT raw 节点
- 新增 `get_raw(raw_id) -> dict | None`
- 新增 `list_raws() -> list[dict]`
- 新增 `link_raw(doc_id, raw_id)` — 创建 `references` 和 `updated_by` 边

**2b. ArchiveDocument 管理（design §1 缺失）**
- 新增 `create_archive(title, content, source) -> dict`
- 新增 `list_archives() -> list[dict]`
- `archived_from` 边由 `relate()` 使用

**2c. Version Graph 链条修复（design §7）**
- 修改 `save_version()`：保存快照后，若存在上一版本（`ver_num > 1`），调用 `relate(ver_rid, "previous_version", prev_ver_rid)` 建链
- 新增 `version_chain(doc_id) -> list[dict]` — 按 `previous_version` 边遍历返回完整版本链

**2d. LLM Memory Graph（design §8 完全缺失）**
- 新增 `record_conversation(question, answer, doc_ids: list[str]) -> dict` — 创建 conversation 节点 + 对每个 doc_id 建 `about` 边
- 新增 `hot_documents(limit=10) -> list[dict]` — 统计 `about` 边入度，返回"被问得最多"的文档
- 新增 `conversations_by_doc(doc_id) -> list[dict]` — 反查某文档关联的对话

**2e. Agent 写入类工具（design §11 缺 3 个）**
- `merge_document(source_id, target_id, merged_content, merged_title) -> dict`：
  - 创建新文档（merged_content），建 `supersedes` 边指向 source 和 target，将 source/target 标记为 `status='archived'`
- `update_metadata(doc_id, tags=None, entities=None, topic=None, author=None) -> dict`：
  - 更新文档元数据 + 重建 `has_tag`/`mentions`/`belongs_to` 边
- `build_graph(doc_id) -> dict`：
  - 对文档内容调用 LLM 抽取实体/关系并建边（委托 `llm.extract_entities` + 落库逻辑，复用 `app.py:extract` 端点已有逻辑）

**2f. 全文检索升级（design §9 降级问题）**
- 修改 `_fts_search()`：优先使用 `SEARCH ... IN doc_search` SurrealQL 语法；若查询失败（索引不存在/SDK 不支持），降级为现有 CONTAINS 逻辑
- 保持降级路径不变，确保向后兼容

**2g. 边类型列表集中化**
- 新增模块级常量 `_ALL_EDGE_TYPES` 和 `_DOC_RELATION_TYPES`，替换 `_collect_edges`(L160)、`_single_hop_neighbors`(L312)、`graph_stats`(L628)、`graph_full`(L803) 中重复硬编码的列表
- 在列表中加入新增的 `duplicates`/`same_topic`/`derived_from`/`archived_from`/`updated_by`/`about`/`previous_version`

---

### 3. `llm.py` — Agent Tool 补全

**3a. 新增 3 个 Tool 定义**（加入 `TOOL_DEFS` + `TOOL_FUNCTIONS` + `AGENT_SYSTEM_PROMPT`）：
- `merge_document(source_id, target_id, merged_title, merged_content)` 
- `update_metadata(doc_id, tags?, entities?, topic?, author?)`
- `build_graph(doc_id)` — 对文档抽取实体并建图边

**3b. 新增 Memory Graph 查询工具：**
- `hot_documents(limit=10)` — 返回被问得最多的文档（design §8 "哪些文档用户问得最多"）

**3c. 更新 `AGENT_SYSTEM_PROMPT`** — 在"可用工具"清单中补充上述 4 个工具的说明

---

### 4. `app.py` — API 端点补全

**4a. RawSource API：**
- `POST /api/raws` — 创建 raw（请求模型 `CreateRawRequest`）
- `GET /api/raws` — 列出 raw
- `GET /api/raws/{raw_id}` — 获取单个 raw
- `POST /api/documents/{doc_id}/link-raw` — 关联文档与 raw（建 references + updated_by 边）

**4b. Archive API：**
- `POST /api/archives` — 创建 archive
- `GET /api/archives` — 列出 archive

**4c. Version Chain API：**
- `GET /api/documents/{doc_id}/version-chain` — 完整版本链（区别于现有 `/versions` 返回离散快照）

**4d. Memory Graph API：**
- `GET /api/conversations/hot` — 热门文档排行
- `GET /api/documents/{doc_id}/conversations` — 文档关联对话
- `POST /api/conversations` — 记录对话（请求模型 `ConversationRequest`）

**4e. 图算法端点已有但前端未用** — 不新增端点，前端补 UI 即可

**4f. 请求模型新增：**
- `CreateRawRequest(url, author, published, content)`
- `CreateArchiveRequest(title, content, source)`
- `ConversationRequest(question, answer, doc_ids)`
- `MergeDocumentRequest(source_id, target_id, merged_title, merged_content)`
- `UpdateMetadataRequest(doc_id, tags, entities, topic, author)`

---

### 5. `seed.py` — 种子数据补全

**5a. 新增 Raw 种子数据：**
- 为 Transformer 文档关联一条 raw（模拟 "Attention Is All You Need" 论文）
- 为 RAG 文档关联一条 raw（模拟检索增强论文）
- 在 `seed_all_sync()` 中灌入 raw + 建立 `references` 边

**5b. 不新增 archive/conversation 种子** — 这些是运行时产生的，种子无意义

---

### 6. `static/index.html` — 前端补全

**6a. 新增"元数据"Tab**（`data-tab="metadata"` → `view-metadata`）：
- 展示 Topics 列表（调用 `/api/topics`）
- 展示 Tags 树形结构（调用 `/api/tags`，按 parent 渲染缩进）
- 展示 Entities 列表（调用 `/api/entities`）
- 点击 entity → 跳转文档库搜索

**6b. 文档详情增强 — 版本历史面板：**
- 在 `showDocDetail()` 中新增"版本历史"按钮
- 点击调用 `/api/documents/{id}/version-chain`，在详情面板内展开版本时间线

**6c. 知识图谱视图增强 — 图算法查询面板：**
- 在 graph-toolbar 旁新增"图查询"折叠面板，包含：
  - 最短路径查询（选两个节点 → `/api/graph/{id}/shortest-path?target=`）
  - 共同邻居（选两个节点 → `/api/graph/{id}/common?other=`）
  - 度中心性排行（→ `/api/graph/central`）
  - 知识血缘（选文档 → `/api/documents/{id}/lineage`）
- 结果以子图叠加或列表形式展示

**6d. 新增"编译文档"入口：**
- 在文档库 Tab 顶部新增"编译文档"按钮
- 点击展开文本框 + 提交按钮 → `POST /api/compile`（raw_content + topic）
- 编译成功后刷新文档列表

**6e. 新增"热门文档"展示：**
- 在问答 Tab 底部或元数据 Tab 中展示 `/api/conversations/hot` 结果
- 显示文档被问次数排行

**6f. CSS 复用现有 token** — 所有新增 UI 使用 `--surface`/`--border`/`--radius`/`--font-*` 等，不引入新色值

---

### 7. 清理

**7a. 删除 `_tmp_test.py`** — 临时调试脚本，无保留价值

**7b. 更新 `README.md`** — 补充新增 API 端点、新增功能说明

---

## 实施顺序（依赖关系）

1. **db.py** — Schema 先行（其他都依赖表/边存在）
2. **wiki_runtime.py** — Runtime 函数（API 和前端都依赖）
3. **llm.py** — Agent 工具（依赖 wiki_runtime 新函数）
4. **app.py** — API 端点（依赖 wiki_runtime + llm）
5. **seed.py** — 种子数据（依赖 wiki_runtime raw 函数）
6. **static/index.html** — 前端（依赖全部 API 就绪）
7. **清理** — 删除临时文件 + 更新 README

## 验证方式
- 启动 `python app.py`，确认无启动错误
- 访问 `/api/health` 确认 stats 含 raws/archives/conversations 计数
- 逐个调用新增 API 端点验证返回
- 前端逐个 Tab 验证新功能可交互
- 确认 `_tmp_test.py` 已删除

## 不修改的文件
- `config.py` — 无需新增配置
- `zvec_client.py` — 向量检索部分无变化
- `.gitignore` — 无需修改