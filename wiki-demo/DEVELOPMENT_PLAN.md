# AI 推理引擎 — 开发计划

> 基于当前 Semantic Wiki Runtime 能力，对照 AI推理引擎.md 设计，分阶段实施。
> 每完成一个阶段更新本文档进度。

---

## 现状基线

| 能力 | 状态 | 说明 |
|------|------|------|
| 文档入库 | ✅ | compile_document LLM 编译 + create_document 落库 |
| 向量检索 | ✅ | zvec REST Bridge + Ollama embedding |
| 全文检索 | ✅ | SurrealDB BM25 SEARCH 索引 |
| 图检索 | ✅ | entity mentions 反查 + 图扩展 |
| 元数据检索 | ✅ | author/topic/tag 结构化查询 |
| 四路融合 | ✅ | search_documents RRF 融合 |
| Agent 问答 | ✅ | run_agent tool-calling 检索+合成 |
| 文档切分 | ⚠️ | seed 有简单句子切分，无通用管道 |
| 本体层 | ❌ | 有 entity 但无层级/关系体系 |
| 概念定位 | ❌ | Agent 直接从 query 调工具 |
| 闭环评估 | ❌ | 开环 "查→答" 模式 |
| Re-Rank | ⚠️ | 仅简单 RRF，无多因子打分 |
| 代码 AST | ❌ | 无 tree-sitter/调用链分析 |

---

## Phase 1：补齐基础设施（目标 1 周）

> 把当前 80% 的 Phase 1 补到 100%，让"扔文件进去 → 自动切分/索引 → 能检索/问答"成为一条完整管道。

### 1.1 通用文档入库管道
- [ ] `pipeline.py` — 统一入库入口
  - [ ] `ingest_file(path)` — 单文件：读取 → 切 chunk → 写 SurrealDB → 写 zvec
  - [ ] `ingest_directory(path)` — 批量：遍历目录 → 逐个 ingest_file
  - [ ] 支持 .md / .txt / .py / .js / .java / .yaml / .json

### 1.2 Chunk 切分器
- [ ] `chunker.py` — 按文档类型智能切分
  - [ ] Markdown 按 ## 标题 + 段落切分
  - [ ] 代码按函数/类边界切分（简单正则，Phase 3 再升级 tree-sitter）
  - [ ] 纯文本按段落切分
  - [ ] 每个 chunk 带元数据：file_path, heading, chunk_index, chunk_type

### 1.3 编译后自动同步向量
- [ ] compile_document 后自动触发 chunk 重切 + 向量入库
- [ ] update_document（内容变更时）同步更新 chunk 和向量

### 1.4 验证
- [ ] 创建测试文档集（5 篇不同类型的文档）
- [ ] 跑 ingest_directory → 验证搜索/问答

---

## Phase 2：本体骨架（目标 2-3 周）

> 给 LLM 一张"领域地图"：概念层级 + 关系 + 概念→文档绑定。

### 2.1 本体 Schema
- [ ] `ontology.py` — 本体数据模型与 CRUD
  - [ ] Concept 节点：name, type, description, parent_id
  - [ ] ConceptRelation 边：source, target, relation_type
  - [ ] ConceptBinding：concept → document_id / file_path / function_name
  - [ ] 存储层：SurrealDB 的 concept 表 + concept_relation 边
  - [ ] YAML 导入/导出（人工编辑 + Git 版本管理）

### 2.2 LLM 辅助本体构建
- [ ] `ontology_builder.py`
  - [ ] `propose_concepts(documents)` — LLM 扫描所有文档摘要 → 提议概念列表
  - [ ] `propose_relations(concepts)` — LLM 分析概念间的层级/依赖关系
  - [ ] `propose_bindings(concept, documents)` — LLM 判断概念绑定到哪些文档
  - [ ] 输出 YAML → 人工 review → 入库

### 2.3 Chunk 概念标注
- [ ] 改造 seed.py / pipeline.py — 入库时 LLM 标注每个 chunk 的所属概念
- [ ] chunk 元数据增加 concept_ids 字段
- [ ] 迁移现有 SurrealDB document 表，增加 concept_ids 字段

### 2.4 查询时概念定位（Step 5）
- [ ] `concept_locator.py`
  - [ ] `locate(query, concepts)` — 给 LLM 概念列表作为选项，输出匹配的概念 + 置信度
  - [ ] 置信度 < 阈值时反问用户确认
  - [ ] 集成到 run_agent 的第一步

### 2.5 图查询增强（Step 6）
- [ ] `ontology_traversal.py`
  - [ ] `expand_concept(concept_id, depth)` — 沿本体关系 1-2 跳展开
  - [ ] `generate_search_plan(concepts)` — 根据概念生成多策略检索计划
  - [ ] 集成到 search_documents 调用前

### 2.6 验证
- [ ] 构建 20-30 个核心概念的测试本体
- [ ] 验证概念定位准确性（10 个典型查询）
- [ ] 验证本体引导检索 vs 纯关键词检索的改善

---

## Phase 3：闭环 Agent + Re-Rank（目标 2-3 周）

> 让 Agent 有"方向感"和"止损能力"。

### 3.1 Re-Rank 模块
- [ ] `reranker.py`
  - [ ] 多因子打分：语义相似度(向量分) + 概念距离(本体图) + 结构相关性(同一调用链/模块) + 新鲜度
  - [ ] 替代当前 RRF 简单融合
  - [ ] 可配置权重

### 3.2 闭环 Agent
- [ ] 改造 `llm.py` run_agent
  - [ ] 增加评估步骤：检索后判断是否足够回答
  - [ ] 评估输出 JSON：{decision: "answer"|"continue", reason, missing_info, next_search}
  - [ ] 循环控制：最多 3 轮，每轮收窄范围
  - [ ] 终止条件：总 token 预算 / 总时间 / 连续两轮无新增有效信息

### 3.3 快速通道
- [ ] FAQ 类简单问题跳过闭环，直接单轮回答
- [ ] 查询分类器：判定问题复杂度（定义/概述类 → 快速通道；排障/分析类 → 闭环）

### 3.4 最终合成增强（Step 10）
- [ ] 答案附推理路径（沿本体哪条路径得出的结论）
- [ ] 答案附置信度 + 不确定说明
- [ ] 追溯链：每个断言可追溯到源 chunk/doc

### 3.5 验证
- [ ] 对比开环 vs 闭环的答案质量（复杂排障类问题）
- [ ] 测量延迟和 token 消耗

---

## Phase 4：代码场景 + 自动化维护（按需）

### 4.1 代码 AST 解析
- [ ] `code_analyzer.py` — tree-sitter 解析
  - [ ] Python/JavaScript/Java 函数/类提取
  - [ ] 调用链分析
  - [ ] 作为"结构化索引路"加入 search_documents

### 4.2 增量索引更新
- [ ] Git hook 触发变更文件重新索引
- [ ] commit hash 绑定 chunk，支持过滤

### 4.3 本体自动化维护
- [ ] LLM 定期扫描新增文档 → 提议本体更新
- [ ] 查询日志分析 → 发现本体盲区

---

## 文件结构（目标）

```
wiki-demo/
├── app.py              # FastAPI Web 应用（不变）
├── config.py           # 配置（不变）
├── db.py               # SurrealDB 初始化（不变）
├── llm.py              # LLM 客户端（Phase 3 改造）
├── wiki_runtime.py     # 核心运行时（逐步增强）
├── zvec_client.py      # zvec 客户端（不变）
├── seed.py             # 种子数据（Phase 1.3 改造）
│
├── chunker.py          # [NEW Phase 1] 智能文档切分
├── pipeline.py         # [NEW Phase 1] 统一入库管道
│
├── ontology.py         # [NEW Phase 2] 本体数据模型与 CRUD
├── ontology_builder.py # [NEW Phase 2] LLM 辅助本体构建
├── concept_locator.py  # [NEW Phase 2] 查询时概念定位
├── ontology_traversal.py # [NEW Phase 2] 本体展开/检索计划
│
├── reranker.py         # [NEW Phase 3] 多因子 Re-Rank
│
├── code_analyzer.py    # [NEW Phase 4] AST/调用链分析
│
├── ontology/            # 本体 YAML 文件（人工编辑，Git 管理）
│   └── concepts.yaml
│
├── AI推理引擎.md        # 设计文档
└── DEVELOPMENT_PLAN.md  # 本文件
```

---

## 进度追踪

| Phase | 开始 | 完成 | 状态 |
|-------|------|------|------|
| Phase 1 | 2026-07-21 | 2026-07-21 | ✅ 已完成 |
| Phase 2 | 2026-07-21 | 2026-07-21 | ✅ 已完成 |
| Phase 3 | 2026-07-21 | 2026-07-21 | ✅ 已完成 |
| Phase 4 | — | — | ⬜ 按需 |

### Phase 1 完成总结

- ✅ 1.1 chunker.py — Markdown/代码/纯文本/YAML 智能切分，支持二级切分
- ✅ 1.2 pipeline.py — ingest_file/ingest_directory/ingest_text/sync_vectors_for_document
- ✅ 1.3 compile/compile-batch 自动同步向量 + sync-vectors API + pipeline API endpoints
- ✅ 1.4 测试文档集验证（4 种类型，切分正确）

### Phase 2 完成总结

- ✅ 2.1 ontology.py — Concept CRUD + 关系边 + 文档绑定 + YAML 导入/导出 + 层级遍历
- ✅ 2.2 ontology_builder.py — LLM 辅助本体构建（propose_concepts/relations/bindings/propose_all）
- ✅ 2.3 种子本体 YAML — 36 个核心概念（ML/向量检索领域），存入 ontology/concepts.yaml
- ✅ 2.4 concept_locator.py — 查询时概念定位（LLM 多选题模式）+ 查询复杂度分类
- ✅ 2.5 ontology_traversal.py — 本体展开（N 跳）+ 5 策略检索计划生成 + 检索范围限定

### Phase 3 完成总结

- ✅ 3.1 reranker.py — 多因子打分（语义/概念距离/结构/新鲜度）+ 过滤规则 + 去重
- ✅ 3.2 llm.py run_agent 改造 — 概念定位注入 + Re-Rank 后处理 + 闭环 Prompt（自评+补充检索）
- ✅ 3.3 快速通道 — classify_query() 判定 simple/complex，简单问题直接单轮

---

## 新增文件清单

```
wiki-demo/
├── chunker.py              # [NEW] 智能文档切分器
├── pipeline.py             # [NEW] 统一入库管道
├── ontology.py             # [NEW] 本体数据模型与 CRUD
├── ontology_builder.py     # [NEW] LLM 辅助本体构建
├── concept_locator.py      # [NEW] 查询时概念定位
├── ontology_traversal.py   # [NEW] 本体展开/检索计划
├── reranker.py             # [NEW] 多因子 Re-Rank
├── ontology/
│   └── concepts.yaml       # [NEW] 种子本体（36 个核心概念）
├── test_docs/
│   ├── test_markdown.md    # [NEW] Markdown 切分测试
│   ├── test_code.py        # [NEW] Python 代码切分测试
│   ├── test_text.txt       # [NEW] 纯文本切分测试
│   └── test_config.yaml    # [NEW] YAML 切分测试
├── AI推理引擎.md            # 设计文档
└── DEVELOPMENT_PLAN.md      # 本文件
```

## 改造文件

```
wiki-demo/
├── app.py    — 新增 pipeline API + compile 后自动同步向量
├── db.py     — 新增 concept/concept_related/concept_binding 表
└── llm.py    — 闭环 Agent（概念定位 + 自评 + Re-Rank）
```

---
*最后更新：2026-07-21*
