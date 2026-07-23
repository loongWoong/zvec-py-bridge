# 未提交代码 Review：实际使用但未实现的功能分析 + 实现计划

> Review 对象：`git status` 未提交变更（app.py / config.py / llm.py / pipeline.py / wiki_runtime.py 修改 + code_analyzer.py 新增）
> 需求基线：`AI推理引擎.md`（设计）、`需求完成度分析.md`（4 偏差 + 2 简化）、`问题修复总结.md`（自报已全部修复，完成度 ~95%）
> 分析日期：2026-07-23
> 分析方法：对未提交代码中**每一个跨模块调用点**逐一验证被调方是否真实存在、数据链路是否闭环（静态分析，未实跑）

---

## 一、总体结论

未提交代码方向正确、骨架完整：`code_analyzer.py`（索引 D）、`_code_search` 第五路检索、`_execute_search_plan` 检索计划激活、`evaluate_retrieval` 结构化闭环、LLM 概念标注，**模块本身都已真实实现**，且依赖的既有接口（`reranker.rerank` 的 reference 参数、`ontology.get_bindings` 的 file_path、`concept_locator.locate` 返回结构等）**全部验证存在**。

但存在 **6 处"实际使用却未闭环"的数据断点**——函数调了、参数传了，数据却在链路中途丢失。这导致 `问题修复总结.md` 自报的"完成度 ~95%"**偏乐观**：偏差①（概念距离因子）名义上修了 3 处代码，但数据链上仍有 3 个断点，因子实际仍未生效；偏差⑤（索引 D）运行时检索路通，但落库路断。**修正后评估完成度约 88%**。

---

## 二、确认的数据断点（实际使用但未闭环）

### P0-1 `code_symbols` 传而未存 —— 索引 D 落库断链

| 环节 | 位置 | 状态 |
|------|------|------|
| 提取 | `pipeline.py` ingest_file/ingest_text 调 `code_analyzer.analyze_code` | ✅ 正常 |
| 传递 | `wr.create_document(..., code_symbols=code_symbols)` | ✅ 参数已加 |
| **落库** | `wiki_runtime.py:135-150` UPSERT 的 SET 子句 | ❌ **缺 `code_symbols = $code_symbols`** |
| 读取 | `GET /api/documents/{doc_id}/code-symbols` 读 `doc.get("code_symbols")` | 永远为空 |

参数字典里传了 `"code_symbols": code_symbols`（wiki_runtime.py:149），但 SQL SET 只有 title/summary/content/topic_id/author/status/version/created/updated —— SurrealDB 忽略未引用参数，**符号索引永远不会写库**。
后果：`/api/documents/{id}/code-symbols` 恒返回 `has_code_symbols: false`；"索引 D 落库"实际只有运行时 `_code_search` 实时分析一条路。

### P0-2 Re-Rank 主路径 `concept_ids` 传而未递 —— 偏差①修复未闭环

`wiki_runtime.py:1164` `_attach_concept_ids(rrf_ranked)` 确实把 `concept_ids` 写进了 `rrf_ranked[did]` 的 info 里，但 **1181-1192 行构造 `candidates_for_rerank` 的字典字面量没有 `"concept_ids"` 键**：

```python
candidates_for_rerank = [
    { "doc_id": did, "title": ..., "excerpt": ..., "score": ...,
      "sources": ..., "file_path": ..., "function_name": ...,
      # ❌ 缺 "concept_ids": info.get("concept_ids", [])
      # ❌ 缺 "updated_at"（新鲜度因子 δ 同样恒为中性 0.5）
    } ...]
```

`reranker.rerank`（reranker.py:269）`r.get("concept_ids", [])` 读到空 → `_concept_distance_score` 走 148-150 行提前返回中性 0.3 → **β=0.30 权重仍是所有候选一致的常量，偏差①实际未修复**。`_attach_concept_ids` 的 DB 查询白跑一次。

### P0-3 `document.concept_ids` 写入静默失败 —— 潜伏既有 bug 被新功能继承

`pipeline.py:468`：

```python
db.query(f"UPDATE {doc_id} SET concept_ids = $ids", {"ids": concept_ids})
```

调用点（pipeline.py:174/365/505）传入的 `doc_id` 均来自 `doc.get("id")`——`create_document`/`list_documents` 经 `_enrich_document` 处理后 id 已**去掉 `document:` 前缀**（纯 key）。SurrealDB 的 record ID 必须是 `table:key` 格式，裸 key 被解析为**表名** → `UPDATE ml_basics SET ...` 对一个不存在的表静默空操作，异常被 `except: pass` 吞掉。

对比佐证：同函数内的 `ontology.bind_concept(cid, doc_id)` 做了规范化（ontology.py:387 `f"document:{document_id}"`）→ **concept_binding 边能建、document.concept_ids 字段写不进**。这解释了需求分析中"大量文档 concept_ids 为空"的另一半原因（不只是标注稀疏，写入路径本身就是断的）。

连锁后果：
- `_fetch_concept_ids` 读 `document.concept_ids` → 恒为空 → 即使 P0-2 修好，候选仍无概念可匹配（断点上移）；
- `repair_concept_bindings` 每次执行 `old_ids` 恒为 `[]` → `updated` 计数虚高，重复"修复"永不收敛；
- 好的一面：`_code_search` 走 `concept_binding` 边（ontology.get_bindings），**不受此 bug 影响**，代码检索路可用。

### P1-4 降级路径 results 未输出 `concept_ids`

`wiki_runtime.py:1218` `_attach_concept_ids(ranked)` 之后，1220-1228 行 `results.append` 同样没有 `"concept_ids"` 键 → 附加结果再次丢弃。影响面：`use_rerank=False` 的调用方（`_execute_search_plan` 预召回、`/api/code-search`）拿不到概念信息，调试/展示不可见。

### P1-5 `Candidate.to_dict()` 丢三字段 —— run_agent 二次 rerank 断链

`reranker.py:84-97` `to_dict()` 输出无 `concept_ids` / `function_name` / `updated_at`。`llm.py:1019` run_agent 最终重排 `reranker.rerank(all_retrieved_docs, ..., target_concept_ids=concept_ids)` 时，若 docs 来自 `search_documents` 的 reranked 结果，`r.get("concept_ids", [])` 又为空 → **概念因子在 Agent 最终重排环节第二次失效**（即使 P0-2 修好也会在这里再断一次）。

### P1-6 `_structural_score` 参考集取自候选自身 —— 结构因子有分无区分度

`wiki_runtime.py:1167-1179` 从 `all_candidates` **自身**收集 `reference_files`/`reference_functions` → 候选的 `file_path` 必然在参考集中（自匹配）→ 凡带 file_path 的候选一律得 0.4 基准分，γ=0.20 区分度弱。
设计意图（AI推理引擎.md Step 8"是否在同一调用链/同一模块"）：参考集应来自**目标概念绑定的文件**（`ontology.get_bindings(cid)` 的 file_path/function_name），这样"候选文件 ∈ 概念绑定文件"才构成真正的相关性信号。

---

## 三、需求对齐修正表

| 问题修复总结声称 | 实际验证结果 | 状态 |
|---|---|---|
| ① 概念距离因子已修复 | 数据链 3 断点：写入失败（P0-3）→ 主路径未递（P0-2）→ 二次 rerank 丢失（P1-5） | ⚠️ 未闭环 |
| ② 检索计划激活 | `_execute_search_plan` 真实执行本体展开 + 增强查询预召回，引用接口全部存在 | ✅ 闭环 |
| ③ 结构化闭环评估 | `evaluate_retrieval` 输出 answer/continue JSON，`force_answer`/`is_simple`/`is_search_call` 均有定义 | ✅ 闭环 |
| ④ LLM 概念标注 | `annotate_document_concepts` LLM 多选 + 关键词兜底，pipeline 优先 LLM | ✅ 闭环（但写入受 P0-3 拖累） |
| ⑤ 索引 D / 代码结构化 | `code_analyzer.py` 完整；`_code_search` 走 concept_binding 边路通；**落库断（P0-1）** | ⚠️ 半闭环 |
| ⑥ 种子文档概念绑定 | 复核认可：seed 经实体名绑定，bind_concept 规范化正确 | ✅ 无需改 |

**真实完成度：约 88%**（自报 95% 需下修）。P0 三项合计约 10 行改动即可让两个核心卖点真正闭环。

---

## 四、实现计划（按 ROI 排序）

### 第 1 批：P0 断点修复（约 10 行，当天可完成）

**任务 1.1** — `wiki_runtime.py` create_document 的 UPSERT SET 增加：
```
code_symbols = $code_symbols
```
验收：ingest 一个 .py 文件后 `GET /api/documents/{id}/code-symbols` 返回非空 symbols。

**任务 1.2** — `pipeline.py` `_update_document_concepts` 写入前规范化：
```python
rid = doc_id if ":" in doc_id else f"document:{doc_id}"
db.query(f"UPDATE {rid} SET concept_ids = $ids", ...)
```
验收：ingest 一篇文档后 `SELECT concept_ids FROM document` 非空；`repair_concept_bindings` 二次运行 updated 数下降。

**任务 1.3** — `wiki_runtime.py` search_documents 的 `candidates_for_rerank` 增加：
```python
"concept_ids": info.get("concept_ids", []),
"updated_at": info.get("updated_at", ""),
```
（`_attach_concept_ids` 顺手按同一 rid 查询补 `updated` 字段）
验收：rerank 输出中 `concept_score` 出现非 0.3 的区分值（有绑定文档的候选 = 1.0/0.5）。

### 第 2 批：P1 数据链补全（约 20 行，1 天内）

**任务 2.1** — 降级路径 `results.append` 增加 `"concept_ids": info.get("concept_ids", [])`。
**任务 2.2** — `reranker.py` `to_dict()` 增加 `concept_ids` / `function_name` / `updated_at` 三字段输出。
**任务 2.3** — `wiki_runtime.py` reference 集改从目标概念绑定收集：
```python
for cid in (concept_ids or []):
    for b in ontology.get_bindings(cid):
        if b.get("file_path"): reference_files.append(b["file_path"])
        if b.get("function_name"): reference_functions.append(b["function_name"])
```
验收：非绑定文件的候选 structural_score < 绑定文件候选。

### 第 3 批：端到端验证（需 SurrealDB + zvec + Ollama + LLM 运行时）

1. `python -m py_compile` 全部改动文件（静态把关）；
2. `code_analyzer.py` 自测（`python code_analyzer.py`）回归；
3. 启动服务跑通三条链路：
   - `POST /api/ask`（深度路径）→ trace 中出现 `_evaluate` 且 rerank 结果 concept_score 有区分；
   - `POST /api/code-search`（concept_names 指定）→ 命中带 file_path/function_name；
   - `POST /api/repair-bindings` 跑两遍 → 第二遍 updated 显著下降（收敛证明）。

### 第 4 批（可选，Phase 4 剩余项）

- `_update_document_concepts` 对空 `concept_ids` 也应允许清空旧绑定（当前 `not concept_ids: return` 会保留脏绑定）；
- `repair_concept_bindings` 增加 dry-run 模式与进度日志；
- 自动维护（Git 变更触发 AST 重解析/本体更新）——AI推理引擎.md Phase 4，不在本次范围。

---

## 五、附：已验证无问题的调用点（排除项）

| 调用点 | 验证结果 |
|---|---|
| `reranker.rerank(reference_files/reference_functions)` | reranker.py:234-243 签名已支持，267-271 行正确回填 Candidate |
| `ontology.get_bindings` 返回结构 | 含 doc_id/title/file_path/function_name，供 `_code_search` 使用 ✅ |
| `concept_locator.locate` 返回结构 | located_concepts/implicit_concepts 键名与 app.py/llm.py 使用一致 ✅ |
| `expand_concepts` / `build_enhanced_query` / `generate_search_plan` | 签名与返回结构匹配 ✅ |
| `app.py` threading | startup() 内局部 import（app.py:70），无 NameError ✅ |
| `/api/repair-bindings` 端点 | app.py:680-689 已实现 ✅ |
| `is_simple` / `is_search_call` / `search_rounds` | llm.py:758/897/854 定义先于 Step 9 评估使用 ✅ |
| db.py code_symbols 字段 | document 表 SCHEMALESS，无需预定义（但 P0-1 SQL 仍需补）✅ |
| `llm.py` 内部函数 `_call_llm_safe` 等 | 全部存在（53/142/150/183 行）✅ |
