# AI 推理引擎 — 全量功能差距分析与完善计划

> 需求基线：`AI推理引擎.md`（离线 Step 1-4 + 在线 Step 5-10 + MVP Phase 1-4 + 风险对策节）
> 实现范围：`wiki-demo/` 全部代码（含未提交变更）
> 衔接文档：`需求完成度分析.md`（初评 82%）、`问题修复总结.md`（自报 95%）、`未提交代码Review_差距与实现计划.md`（断点详查，修正为 ~88%）
> 分析日期：2026-07-23

---

## 一、总体结论

**wiki-demo 未实现 AI推理引擎.md 定义的全部功能。全量口径完成度约 78%。**

- **10 个 Step 全部有对应代码骨架**，核心链路（概念定位 → 本体展开 → 多路并行检索 → 多因子 Re-Rank → 结构化闭环 → 合成输出）真实串通，不是空壳；
- 但有 **3 类差距**：
  1. **断点型**（6 处，已详查）：数据链中途断裂，已调用但未闭环——概念距离因子 β、新鲜度因子 δ 实际失效，索引 D 落库失败（详见 `未提交代码Review_差距与实现计划.md`）；
  2. **缺失型**（11 项）：设计明确定义但代码中完全不存在的功能；
  3. **降级型**（5 项）：有简化实现，能力弱于设计意图。

**MVP 阶段口径**：Phase 1 ✅ / Phase 2 ✅ / Phase 3 ⚠️ 85%（受断点拖累）/ **Phase 4 ❌ 0%**。

---

## 二、全量功能对照表

### 离线阶段（Step 1-4）

| 设计要求 | 实现 | 状态 | 差距 |
|---|---|---|---|
| Step 1: Markdown 解析（标题层级） | chunker.py 正则切分 | ✅ | — |
| Step 1: 代码 AST | code_analyzer.py（正则+缩进启发式，非 tree-sitter） | ⚠️ 降级 | 精度受限（泛型/装饰器/嵌套场景），但可接受 |
| Step 1: PDF 解析（pdfplumber） | 无 | ❌ | 缺失 |
| Step 1: HTML 解析 | 无 | ❌ | 缺失 |
| Step 1: 数据库 Schema 解析 | 无 | ❌ | 缺失 |
| Step 1: 表格提取 | 无（MD 表格按普通文本切） | ❌ | 缺失 |
| Step 2: 概念/关系/层级/属性绑定 | ontology.py 全套 + concepts.yaml（36 概念） | ✅ | — |
| Step 2: LLM 辅助抽取 + 人工校验 | ontology_builder.py 有 propose；**无 review/approve 工作流** | ⚠️ | 提议直接入库，缺人工确认环节 |
| Step 2: 代码 AST → 本体骨架自动提取 | 无管道（code_analyzer 的 symbols 未反哺本体） | ❌ | 缺失 |
| Step 3: 不切断函数体/章节 | ✅ | ✅ | — |
| Step 3: 超长二次切分 + 上下文摘要 | parent_chunk_id 有；**无上下文摘要** | ⚠️ | 半实现 |
| Step 3: **chunk 级概念标注**（所属/父/关联概念） | 仅文档级标注（pipeline._annotate_concepts） | ❌ | 缺失 |
| Step 3: chunk 间关系（前置/依赖/示例/对比） | 无 | ❌ | 缺失 |
| Step 4: 索引 A 向量 / B 倒排 / C 本体图 | zvec + SurrealDB doc_search + 图边 | ✅ | — |
| Step 4: 索引 D 结构化（符号表/调用链落库） | **落库断（P0-1）**；运行时 `_code_search` 路通 | ⚠️ 断点 | SET 子句漏字段 |
| Step 4: 统一元数据（行号/概念/类型/时间） | 行号 ✅（chunk 级）；概念 ❌（chunk 级）；类型/时间仅文档级 | ⚠️ | chunk 元数据不全 |

### 在线阶段（Step 5-10）

| 设计要求 | 实现 | 状态 | 差距 |
|---|---|---|---|
| Step 5: 从概念列表中选（约束幻觉） | concept_locator.locate（多选题 JSON） | ✅ | — |
| Step 5: 意图类型/定位/隐含/约束/期望输出 | 输出五元组 | ✅ | — |
| Step 5: **约束条件参与检索**（频率/触发点） | 输出后**无人消费** | ❌ | constraints 只进 prompt 文本 |
| Step 5: 低置信度反问用户确认 | 无 | ❌ | 缺失 |
| Step 5: 检索结果修正定位（重定位） | 无 | ❌ | 缺失 |
| Step 6: 本体展开 + 5 策略计划 | expand_concepts + generate_search_plan + _execute_search_plan | ✅ 主体 | — |
| Step 6/7: 策略1 向量 / 策略4 图 | search_documents 四路并行 | ✅ | — |
| Step 6/7: 策略2 grep | 仅 _code_search 内对绑定文件 fallback；**无跨库 grep 策略** | ⚠️ 降级 | 范围限于绑定文件 |
| Step 6/7: 策略3 结构化（查配置规则等） | 无 | ❌ | 缺失 |
| Step 6/7: 策略5 调用链 | code_route 第五路 | ✅ | 依赖概念绑定 file_path |
| Step 8: α语义 + β概念距离 + γ结构 + δ新鲜度 | 框架完整；**β/δ 断点失效（P0-2/P0-3/P1-5）**；γ 参考集自匹配（P1-6） | ⚠️ 断点 | 详见上轮报告 |
| Step 8: 过滤规则（距离>阈值丢弃/低相似降权/同概念合并） | 仅 min_score 阈值 + 按 doc_id 去重 | ⚠️ 降级 | 无概念距离硬过滤；无按概念合并 |
| Step 9: 结构化评估 JSON | evaluate_retrieval（answer/continue/missing_info/next_search） | ✅ 主体 | — |
| Step 9: **"矛盾，需验证 Y" → 验证性检索** | 无 verify 决策分支（只有 answer/continue） | ❌ | 缺失 |
| Step 9: 终止条件（3轮/时间/token/无新增） | search_rounds<3 / 45s / 40k token / 无新增强制停 | ✅ | — |
| Step 10: 结构化答案 + 引用来源 | prompt 强制 `[doc_id]` | ✅ | — |
| Step 10: **推理路径（沿本体哪条关系路径）** | ontology_path 只是步骤摘要（定位/计划/执行），**无关系边遍历路径** | ⚠️ 降级 | 非"沿 is-a/depends 路径" |
| Step 10: 置信度 + **不确定说明** | confidence 启发式（0.5+0.1×概念数）；无不确定说明 | ⚠️ 降级 | 无矛盾点/盲区声明 |

### Phase 4 与风险对策（设计第 二、三 节明确定义）

| 设计要求 | 状态 |
|---|---|
| Phase 4: Git commit 触发增量索引（diff 变更文件 → 只重处理变更 chunk） | ❌ |
| Phase 4: chunk 绑定 commit hash，查询可按 commit 过滤 | ❌ |
| Phase 4: LLM 定期扫描新文档提议本体更新（人工 review 合入） | ❌（builder 有能力，无调度与 review 流） |
| Phase 4: 查询日志分析发现本体盲区 | ❌ |
| 风险对策: 相同概念组合检索结果短期缓存 | ❌ |
| 风险对策: 本体版本化（回溯） | ⚠️ YAML 导出/导入有，版本管理无 |
| 风险对策: 简单问题快速通道 | ✅ classify_query/is_simple |

---

## 三、完善计划（分 5 个 Wave，按 ROI 排序）

> Wave 0-1 是"让已有功能真正生效"（断点修复），Wave 2-4 是"补齐设计功能"。
> 每个任务标注：改动位置、工作量（S<半天 / M<2天 / L≥2天）、验收标准。

### Wave 0：P0 断点修复（≈10 行，0.5 天）—— 沿用上轮计划

| # | 任务 | 位置 | 量 | 验收 |
|---|---|---|---|---|
| 0.1 | UPSERT SET 补 `code_symbols = $code_symbols` | wiki_runtime.py create_document | S | ingest .py 后 `/api/documents/{id}/code-symbols` 非空 |
| 0.2 | `_update_document_concepts` 写入前规范化 `document:{id}` | pipeline.py | S | 二跑 repair-bindings，updated 数显著下降（收敛） |
| 0.3 | `candidates_for_rerank` 补 `concept_ids` + `updated_at` | wiki_runtime.py search_documents | S | rerank 输出 concept_score 出现 0.3 以外的区分值 |

### Wave 1：P1 数据链补全（≈30 行，1 天）

| # | 任务 | 位置 | 量 | 验收 |
|---|---|---|---|---|
| 1.1 | 降级路径 results 输出 concept_ids | wiki_runtime.py | S | `use_rerank=False` 返回含 concept_ids |
| 1.2 | `Candidate.to_dict()` 补 concept_ids/function_name/updated_at | reranker.py | S | run_agent 二次 rerank 概念因子生效 |
| 1.3 | reference 集改从目标概念 get_bindings 收集 | wiki_runtime.py | S | 绑定文件候选 structural_score > 非绑定 |
| 1.4 | Re-Rank 硬过滤：概念距离=0.1（远距）且语义<阈值 → 丢弃；同概念多 chunk 取最高分 | reranker.py | M | 过滤前后候选数变化可观测 |

### Wave 2：在线推理补全（2-3 天）—— 对齐 Step 5/9/10 与风险对策

| # | 任务 | 位置 | 量 | 验收 |
|---|---|---|---|---|
| 2.1 | **矛盾验证分支**：EVAL prompt 增加 `decision: "verify"`，输出 verify_target；run_agent 对 verify 生成验证性检索（沿本体绑定文件 grep/定向 search） | llm.py | M | 构造矛盾场景，trace 出现 verify 轮 |
| 2.2 | **真实推理路径**：基于 expand_concepts 的 relations/paths，输出 `概念A -[关系]-> 概念B → 命中文档` 链路，替代步骤摘要 | llm.py + ontology_traversal.py | M | ontology_path 含关系边类型 |
| 2.3 | **不确定说明**：合成 prompt 要求输出"证据盲区/矛盾点/假设"；结构化进返回字段 `uncertainties` | llm.py | S | 返回含 uncertainties 列表 |
| 2.4 | **低置信度反问**：located 置信度全 <0.4 时返回 `clarification_needed` + 候选概念，由前端/API 调用方确认后继续（不阻塞，带默认行为） | app.py + llm.py | M | 模糊问题返回澄清请求 |
| 2.5 | **constraints 消费**：定位输出的约束（频率/触发点）注入检索 filter 与评估 prompt | llm.py + wiki_runtime.py | S | 含"偶尔/登录后"的问题检索用到约束词 |
| 2.6 | **检索结果短期缓存**：（概念组合+query 哈希）→ results，TTL 5min，省重复 LLM/DB 调用 | wiki_runtime.py | M | 相同查询二次延迟显著下降 |

### Wave 3：离线索引补全（3-5 天）—— 对齐 Step 1/3/4

| # | 任务 | 位置 | 量 | 验收 |
|---|---|---|---|---|
| 3.1 | **chunk 级概念标注**：ingest 时对每个 chunk 调 annotate_document_concepts（chunk 文本+所属标题），写入 chunk 元数据；检索候选可取 chunk 概念 | pipeline.py + zvec_client.py | M | chunk payload 含 concept_ids |
| 3.2 | **PDF 解析**：pipeline 增加 .pdf 分支（pdfplumber，懒加载可选依赖） | pipeline.py + chunker.py | M | ingest PDF 成功且按页/标题切分 |
| 3.3 | **HTML 解析**：.html 分支（去标签取正文+标题层级） | pipeline.py + chunker.py | S | ingest HTML 成功 |
| 3.4 | **MD 表格识别**：表格块作为独立 chunk（不拆行），metadata 标 type=table | chunker.py | S | 表格完整不切断 |
| 3.5 | **超长 chunk 上下文摘要**：二次切分时将父 chunk 前 100 字摘要写入子 chunk metadata | chunker.py | S | 子 chunk 含 parent_summary |
| 3.6 | **AST → 本体骨架**：code_analyzer 的 symbols/calls 转本体提议（模块=概念、calls=depends 关系），走 builder propose 管道 | ontology_builder.py + code_analyzer.py | L | 对代码目录一键生成本体草案 |
| 3.7 | **跨库 grep 策略**：策略2 独立成工具——对 SurrealDB content 全文正则扫描（不限绑定文件），纳入 _execute_search_plan | wiki_runtime.py | M | 计划中含独立 grep 路命中 |

### Wave 4：Phase 4 自动化维护（5-8 天，可独立排期）

| # | 任务 | 量 | 验收 |
|---|---|---|---|
| 4.1 | **Git 增量索引**：`git diff --name-only <old> <new>` → 变更文件重 ingest；chunk metadata 绑定 commit hash；查询支持 `since_commit` 过滤 | L | 提交后只重建变更文件索引 |
| 4.2 | **本体定期维护**：定时任务扫描近 N 天新文档 → builder.propose_concepts/relations → 生成 review 清单（YAML PR 或 API 待确认队列） | L | 产生可人工确认的提议列表 |
| 4.3 | **查询日志分析**：conversation 表统计未命中概念的高频问题 → 本体盲区报告 | M | 输出盲区 TOP-N |
| 4.4 | **本体版本化**：concepts.yaml 变更历史 + 回滚接口 | M | 可回滚到指定版本 |
| 4.5 | **本体 review 工作流**：concept 增加 status(proposed/approved/rejected)，检索只用 approved | M | proposed 概念不参与检索 |

---

## 四、执行建议

1. **Wave 0+1 先行**（1.5 天）：不修这两波，本体引导与结构因子等于没接电——这是"已有但未生效"，ROI 最高；
2. **Wave 2 提升推理可信度**（矛盾验证、真实推理路径、不确定说明）——这是设计文档"方向感与止损能力"卖点的直接体现；
3. **Wave 3 按数据来源取舍**：若知识库无 PDF/HTML 来源，3.2/3.3 可缓；chunk 级标注（3.1）对检索精度提升最大，优先；
4. **Wave 4 属长期运营能力**，建议在 Wave 0-2 上线并积累 2-4 周查询日志后启动（4.3 依赖数据积累）；
5. 每 Wave 完成后跑端到端验收（`/api/ask` 深度路径 + `/api/code-search` + `repair-bindings` 收敛测试）。

**完成 Wave 0-3 后预计全量完成度 ≈95%；Wave 4 完成后 100%（对齐设计全量）。**
