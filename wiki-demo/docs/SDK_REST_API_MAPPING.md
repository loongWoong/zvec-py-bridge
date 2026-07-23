# zvec-py-bridge：SDK 与 REST API 对应关系说明

本文档说明 **zvec 0.5.x Python SDK** 与本桥接服务暴露的 **REST API** 之间的逐项对应关系，并详细描述每个 REST 端点的用法与特性。

> 阅读前请先了解：本服务是一个**无状态**的 FastAPI 门面（façade），它把 zvec 的强类型 Python 绑定（`DataType`/`MetricType` 等枚举与一组按索引类型区分的 `*Param` 子类）包装成纯 JSON 的 HTTP 接口，使 Java / Go / JS 等任何语言都能通过 JSON 驱动 zvec。

---

## 1. 总体架构与映射原理

### 1.1 分层与“唯一桥接点”

```
HTTP JSON  ──►  api/        （薄路由，只解析 HTTP，不 import zvec）
                 │
                 ▼
              service/      （编排：manager + mapper，不解析 JSON）
                 │
                 ▼
              model/mapper  （★ 唯一桥接点：JSON ⇄ zvec 强类型对象）
                 │
                 ▼
              core/manager  （线程安全的 Collection / Embedding 注册表）
                 │
                 ▼
              zvec SDK      （C++ 后端的 Python 绑定）
```

- `api/` 层**绝不**直接 `import zvec`，只调用 `service/`。
- `service/` **绝不**解析 JSON，只调用 `core.manager` + `model.mapper`。
- `model.mapper` 是**唯一**知道如何把 JSON 转成 zvec 枚举/参数/文档（以及反向序列化）的模块，因此其余代码库无需任何 zvec 专属分支。
- `core.manager` 用 `RLock` 串行化 create/open/close/destroy，使 REST 层可被并发调用而不会破坏内存注册表。

### 1.2 映射总览表

| SDK 对象 / 方法 | REST 端点 | 桥接实现位置 |
|---|---|---|
| `zvec.init(...)` | （启动时自动调用，无端点） | `main.py` lifespan |
| `zvec.create_and_open(path, schema)` | `POST /collections` | `collection_service.create` |
| `zvec.open(path, option)` | `POST /collections/{name}/open` | `collection_service.open` |
| `Collection.schema / .path / .option` | `GET /collections/{name}` | `collection_service.info` |
| （注册表 + 磁盘扫描） | `GET /collections` | `manager.list_collections` |
| `Collection.flush()` + 释放句柄 | `POST /collections/{name}/close` | `collection_service.close` |
| `Collection.destroy()` | `DELETE /collections/{name}` | `collection_service.destroy` |
| `Collection.add_column(...)` | `POST /collections/{name}/columns` | `collection_service.add_column` |
| `Collection.alter_column(...)` | `PUT /collections/{name}/columns/{old_name}` | `collection_service.alter_column` |
| `Collection.drop_column(name)` | `DELETE /collections/{name}/columns/{field_name}` | `collection_service.drop_column` |
| `Collection.insert(docs)` | `POST /collections/{name}/documents` | `document_service.insert` |
| `Collection.upsert(docs)` | `PUT /collections/{name}/documents` | `document_service.upsert` |
| `Collection.update(docs)` | `PATCH /collections/{name}/documents` | `document_service.update` |
| `Collection.delete(ids)` | `DELETE /collections/{name}/documents` | `document_service.delete` |
| `Collection.delete_by_filter(filter)` | `POST /collections/{name}/documents:deleteByFilter` | `document_service.delete_by_filter` |
| `Collection.fetch(ids, ...)` | `POST /collections/{name}/documents:fetch` | `document_service.fetch` |
| `Collection.query(queries, topk, filter, ...)` | `POST /collections/{name}/search` | `query_service.search` |
| `Collection.create_index(field_name, index_param, option)` | `POST /collections/{name}/indexes/{field_name}` | `index_service.create_index` |
| `Collection.drop_index(field_name)` | `DELETE /collections/{name}/indexes/{field_name}` | `index_service.drop_index` |
| `Collection.optimize(option)` | `POST /collections/{name}:optimize` | `index_service.optimize` |
| `Collection.stats` | `GET /collections/{name}/stats` | `admin_service.stats` |
| `Collection.flush()` | `POST /collections/{name}:flush` | `admin_service.flush` |
| `EmbeddingFunction.embed(text)` | `POST /embeddings/{name}/embed` | `embedding_service.embed` |
| （注册表：构建 + 校验） | `POST /embeddings` | `embedding_service.register` |
| （注册表：列出，脱敏） | `GET /embeddings` | `embedding_service.list` |
| （注册表：移除） | `DELETE /embeddings/{name}` | `embedding_service.remove` |
| `zvec.__version__` | `GET /health`、`GET /engine` | `main.py` / `admin_service.engine_info` |
| `zvec.is_diskann_plugin_loaded()` 等 | `GET /engine` | `admin_service.engine_info` |
| `zvec.get/set_default_jieba_dict_dir()` | `GET/PUT /admin/jieba-dict` | `admin_service` |
| `zvec.load_diskann_plugin()` | `POST /admin/diskann:load` | `admin_service.load_diskann_plugin` |

> **命名约定**：路径中带冒号的 `:verb` 后缀（如 `:fetch`、`:optimize`、`:flush`、`:deleteByFilter`、`:load`）是 Google AIP 风格的自定义动作，用于在同一个资源路径上区分“非 CRUD 动词”，避免与标准 `POST/PUT/DELETE` 冲突。

---

## 2. Collections — 集合生命周期与 Schema

### 2.1 `POST /collections` — 创建集合并打开

- **SDK**：`zvec.create_and_open(path=path, schema=schema)`（经 `manager.create`）。
- **请求体**：`CreateCollectionDTO`
  ```jsonc
  {
    "schema": {
      "name": "docs",
      "fields": [
        { "name": "title", "data_type": "STRING", "nullable": true },
        { "name": "content", "data_type": "STRING", "nullable": true,
          "index_param": { "type": "FTS", "tokenizer_name": "standard" } },
        { "name": "tag", "data_type": "STRING", "nullable": true,
          "index_param": { "type": "INVERT", "enable_range_optimization": true } }
      ],
      "vectors": [
        { "name": "embedding", "data_type": "VECTOR_FP32", "dimension": 1536,
          "index_param": { "type": "HNSW", "metric_type": "COSINE",
                           "m": 16, "ef_construction": 200 } }
      ]
    },
    "read_only": false
  }
  ```
- **响应**：`{ "name", "status":"created", "path", "schema": {...} }`
- **特性**：
  - 路径中的 `{name}` 必须与 `schema.name` 一致，否则 `INVALID_ARGUMENT(400)`。
  - 集合**必须至少定义一个向量字段**，否则 400。
  - 若集合名已打开或磁盘上已存在同名目录 → `ALREADY_EXISTS(409)`。
  - 创建失败时会清理半建目录，避免脏状态。
  - 集合名禁止包含路径分隔符（`/`、`\`、`os.sep`），保证文件系统安全。
  - `index_param` 是**判别联合**：`type` 字段决定具体参数类，多余字段被忽略。详见 [§7 索引参数](#7-索引参数index_param)。

### 2.2 `GET /collections` — 列出所有集合

- **SDK**：无直接 SDK 调用；由 `manager.list_collections()` 合并“内存注册表”与“数据目录磁盘扫描”。
- **响应**：`[ { "name", "opened": bool, "path" }, ... ]`
- **特性**：
  - 同时返回**已打开**和**仅在磁盘上**（未打开）的集合。
  - `opened` 标识该集合当前是否在内存注册表中。

### 2.3 `GET /collections/{name}` — 集合信息 + Schema

- **SDK**：读取 `Collection.schema / .path / .option`。
- **响应**：`{ "name", "opened": true, "path", "read_only", "enable_mmap", "schema": {...} }`
- **特性**：
  - 若 `ZVEC_AUTO_OPEN=true`（默认），未打开时会**自动打开**；否则返回 `COLLECTION_NOT_OPEN(409)`。
  - 磁盘上不存在时返回 `NOT_FOUND(404)`。

### 2.4 `POST /collections/{name}/open` — 打开已有集合

- **SDK**：`zvec.open(path, option=zvec.CollectionOption(read_only=...))`（经 `manager.open`）。
- **请求体**：`OpenCollectionDTO` → `{ "read_only": false }`
- **响应**：`{ "name", "status":"opened", "path", "schema": {...} }`
- **特性**：
  - 若集合已在注册表中，**幂等**直接返回（不重复打开）。
  - 磁盘不存在 → `NOT_FOUND(404)`。

### 2.5 `POST /collections/{name}/close` — 卸载集合（保留数据）

- **SDK**：先 `Collection.flush()` 持久化挂起写入，再从注册表移除句柄（C++ 对象由 pybind 引用计数回收）。
- **响应**：`{ "name", "status":"closed" }`
- **特性**：
  - **数据保留在磁盘**，仅卸载内存句柄；后续可 `open` 重新加载。
  - flush 失败被忽略（best-effort），仍会卸载句柄。
  - 未打开 → `NOT_FOUND(404)`。

### 2.6 `DELETE /collections/{name}` — 永久销毁集合

- **SDK**：`Collection.destroy()`；若句柄不在内存则手动 `shutil.rmtree` 磁盘目录。
- **响应**：`{ "name", "status":"destroyed" }`
- **特性**：
  - **不可恢复**：同时清除内存句柄与磁盘数据。
  - 若 `destroy()` 抛错，回退到删除目录并报 `ENGINE_ERROR(500)`。
  - 磁盘上不存在 → `NOT_FOUND(404)`。

---

## 3. Columns — 列（标量字段）DDL

> `alter_column` 受 zvec 引擎限制，**仅支持基础数值类型**（INT32/64、UINT32/64、FLOAT、DOUBLE）。

### 3.1 `POST /collections/{name}/columns` — 新增列（含回填表达式）

- **SDK**：`Collection.add_column(field_schema=..., expression=..., option=zvec.AddColumnOption(concurrency=...))`。
- **请求体**：`AddColumnDTO`
  ```jsonc
  {
    "field": { "name": "score", "data_type": "FLOAT", "nullable": true },
    "expression": "0.0",      // 回填表达式，对已有行求值
    "concurrency": 0          // 0 表示用引擎默认并发
  }
  ```
- **响应**：`{ "name", "column", "status":"added" }`
- **特性**：
  - `expression` 用于为**已有文档**回填新列的初值（zvec SQL-like 表达式）。
  - `field` 可带 `index_param`（如 INVERT/FTS），随列一起建索引。

### 3.2 `PUT /collections/{name}/columns/{old_name}` — 改名 / 改 Schema

- **SDK**：`Collection.alter_column(old_name=..., new_name=..., field_schema=..., option=zvec.AlterColumnOption(concurrency=...))`。
- **请求体**：`AlterColumnDTO` → `{ "new_name": "pop", "field": {...}|null, "concurrency": 0 }`
- **响应**：`{ "name", "old_name", "new_name", "status":"altered" }`
- **特性**：
  - 可只改名（仅 `new_name`），也可同时改 Schema（带 `field`）。
  - 数值类型以外的 `alter` 会被引擎拒绝 → `ENGINE_ERROR(500)`。

### 3.3 `DELETE /collections/{name}/columns/{field_name}` — 删除列

- **SDK**：`Collection.drop_column(field_name)`。
- **响应**：`{ "name", "column", "status":"dropped" }`

---

## 4. Documents — 文档 DML

> 写操作（insert/upsert/update）共享同一实现 `document_service._apply`：先按需自动嵌入文本，再 `getattr(collection, op)(docs)`。返回每个文档的 `Status`（`ok/code/message`，注意 zvec 的 `Status` 暴露的是**方法**而非属性）。

### 4.1 `POST /collections/{name}/documents` — 插入

- **SDK**：`Collection.insert(docs)`。
- **请求体**：`DocumentBatchDTO`
  ```jsonc
  {
    "documents": [
      { "id": "d1",
        "vectors": { "embedding": [0.1, 0.2, /* ... */] },
        "fields":  { "title": "hello", "tag": "a" } }
    ],
    "embedding": null      // 可选：见 §4.7 文本→向量自动嵌入
  }
  ```
- **响应**：`{ "name", "op":"insert", "count", "results": [ { "id", "status": {ok,code,message} } ] }`
- **特性**：
  - **稀疏向量**用 JSON 对象表示，键是字符串（JSON 无整型键），桥接自动转为 `uint32`：
    `{ "sparse": { "1": 0.5, "2": 1.0 } }`。
  - 插入已存在的 `id` 行为由 zvec 引擎决定（通常需用 upsert 覆盖）。

### 4.2 `PUT /collections/{name}/documents` — Upsert（插入或覆盖）

- **SDK**：`Collection.upsert(docs)`。
- **请求/响应**：同 insert，`op` 为 `"upsert"`。
- **特性**：存在则覆盖、不存在则插入，适合幂等写入。

### 4.3 `PATCH /collections/{name}/documents` — Update（部分更新）

- **SDK**：`Collection.update(docs)`。
- **请求/响应**：同 insert，`op` 为 `"update"`。
- **特性**：仅更新请求中出现的字段/向量，未出现的保持不变。

### 4.4 `DELETE /collections/{name}/documents` — 按 id 删除

- **SDK**：`Collection.delete(ids)`。
- **请求体**：`{ "ids": ["d1", "d2"] }`（DELETE 方法带 body）。
- **响应**：`{ "name", "op":"delete", "count", "results": [ { "id", "status": {...} } ] }`

### 4.5 `POST /collections/{name}/documents:deleteByFilter` — 按过滤表达式删除

- **SDK**：`Collection.delete_by_filter(filter)`。
- **请求体**：`DeleteByFilterDTO` → `{ "filter": "tag = 'b'" }`
- **响应**：`{ "name", "op":"delete_by_filter", "filter", "status":"ok" }`
- **特性**：`filter` 使用 zvec SQL-like 语法（`tag = 'b'`、`age > 30`、`cat IN (...)`，等号为单 `=`）。

### 4.6 `POST /collections/{name}/documents:fetch` — 按 id 取回

- **SDK**：`Collection.fetch(ids, output_fields=..., include_vector=...)`。
- **请求体**：`FetchDTO`
  ```jsonc
  { "ids": ["d1", "missing"], "output_fields": ["title"], "include_vector": true }
  ```
- **响应**：`{ "name", "count", "documents": [ {id, score, fields, vectors} ] }`
- **特性**：
  - 不存在的 id 静默跳过（`count` 只含命中数）。
  - `include_vector=false` 时 `vectors` 为 `null`，节省带宽。
  - `output_fields` 限定返回的标量字段；为空则返回全部。
  - 向量结果中的 ndarray 经 `tolist()` 转 JSON 安全结构。

### 4.7 文本 → 向量（自动嵌入）

写操作与查询都支持：当批次/请求带顶层 `embedding` 引用时，桥接自动把每条文档的 `text` 嵌入到指定向量字段，**无需预先算向量**。

```jsonc
POST /collections/docs/documents
{
  "embedding": { "function": "my_emb", "field": "embedding",
                 "encoding_type": "document" },   // sparse/BM25 才需要；可省略
  "documents": [ { "id": "d1", "text": "machine learning is fun", "fields": {"tag":"a"} } ]
}
```

- **优先级**：显式 `vectors` > `text` 自动嵌入；若某文档既无该字段向量又无 `text` → `INVALID_ARGUMENT(400)`。
- **稀疏/BM25** 的 `encoding_type`（`query` | `document`）选择编码策略：插入时默认 `document`，查询时默认 `query`，也可显式指定。
- 嵌入函数须先经 `POST /embeddings` 注册（见 [§10 Embeddings](#10-embeddings-embeddings)）。

---

## 5. Search — 检索（向量 / FTS / 混合）

### 5.1 `POST /collections/{name}/search` — 单/多路检索 + 重排

- **SDK**：`Collection.query(queries=..., topk=..., filter=..., include_vector=..., output_fields=..., reranker=...)`。
- **请求体**：`SearchDTO`
  ```jsonc
  {
    "queries": [
      { "field_name": "embedding", "vector": [0.1, 0.2, /* ... */],
        "param": { "type": "HNSW", "ef": 100 } },
      { "field_name": "content",
        "fts": { "match_string": "machine learning" },
        "param": { "type": "FTS", "default_operator": "AND" } }
    ],
    "topk": 10,
    "filter": "tag = 'a'",
    "include_vector": false,
    "output_fields": ["title"],
    "reranker": { "type": "rrf", "rank_constant": 60 },
    "embedding": { "function": "my_emb", "field": "embedding" }   // 可选
  }
  ```
- **响应**：`{ "name", "topk", "filter", "count", "documents": [ {id, score, fields, vectors} ] }`

#### 查询类型（`queries[]` 每项三选一）
| 形式 | 字段 | 说明 |
|---|---|---|
| 向量查询 | `vector` 或 `id` | 用向量或已有文档 id 作为查询向量 |
| 文本查询 | `text` + 顶层 `embedding` | 桥接先嵌入成向量再查（须 `field_name` 与 `embedding.field` 一致） |
| 全文查询 | `fts: {query_string, match_string}` | 走 FTS 索引 |

- **优先级**：`vector` / `id` > `text`；若都无 → 400。
- **多路融合**：`queries` 含多项时由 `reranker` 融合；默认 RRF（`rank_constant=60`）。
- **`filter`**：zvec SQL-like 语法，作用于标量字段（依赖 INVERT 索引加速）。
- **`param.type` 必须与字段索引类型匹配**（HNSW 字段用 HNSW 查询参数等）。`param` 可省略（如 FLAT 索引）。
- 查询参数详情见 [§8 查询参数](#8-查询参数query_param)，重排器见 [§9 重排器](#9-重排器reranker)。

---

## 6. Indexes — 索引管理

### 6.1 `POST /collections/{name}/indexes/{field_name}` — 创建索引

- **SDK**：`Collection.create_index(field_name=..., index_param=..., option=zvec.IndexOption(concurrency=...))`。
- **请求体**：`CreateIndexDTO`
  ```jsonc
  {
    "index_param": { "type": "HNSW", "metric_type": "COSINE", "m": 16, "ef_construction": 200 },
    "concurrency": 0
  }
  ```
- **响应**：`{ "name", "field", "index_param": {...}, "status":"created" }`
- **特性**：
  - 支持的 `type`：`HNSW`、`HNSW_RABITQ`、`IVF`、`FLAT`、`INVERT`、`FTS`、`VAMANA`、`DISKANN`。
  - **DISKANN**（磁盘 ANN）需要 libaio 插件，先 `POST /admin/diskann:load` 加载；否则引擎报错。
  - 同一字段换索引类型需先 `DELETE` 旧索引。

### 6.2 `DELETE /collections/{name}/indexes/{field_name}` — 删除索引

- **SDK**：`Collection.drop_index(field_name)`。
- **响应**：`{ "name", "field", "status":"dropped" }`

### 6.3 `POST /collections/{name}:optimize` — 优化（合并/重建）

- **SDK**：`Collection.optimize(option=zvec.OptimizeOption(concurrency=...))`。
- **请求体**：`OptimizeDTO` → `{ "concurrency": 0 }`
- **响应**：`{ "name", "status":"optimized" }`
- **特性**：触发索引段合并与重建，提升后续查询性能；大批量写入后建议调用。

---

## 7. 索引参数（`index_param`）

`IndexParamDTO` 是判别联合：`type` 决定具体 zvec 参数类，`mapper.build_index_param` 只转发该类型允许的字段（`_drop_none` 去空）。下表列出各类型与对应 SDK 类及专有字段。

| `type` | SDK 类 | 专有字段 |
|---|---|---|
| `HNSW` | `zvec.HnswIndexParam` | `metric_type`, `m`, `ef_construction`, `quantize_type`, `use_contiguous_memory` |
| `HNSW_RABITQ` | `zvec.HnswRabitqIndexParam` | `metric_type`, `total_bits`, `num_clusters`, `m`, `ef_construction`, `sample_count` |
| `IVF` | `zvec.IVFIndexParam` | `metric_type`, `n_list`, `n_iters`, `use_soar`, `quantize_type` |
| `FLAT` | `zvec.FlatIndexParam` | `metric_type`, `quantize_type` |
| `INVERT` | `zvec.InvertIndexParam` | `enable_range_optimization`, `enable_extended_wildcard` |
| `FTS` | `zvec.FtsIndexParam` | `tokenizer_name`, `filters`, `extra_params` |
| `VAMANA` | `zvec.VamanaIndexParam` | `metric_type`, `max_degree`, `search_list_size`, `alpha`, `saturate_graph`, `use_contiguous_memory`, `use_id_map`, `quantize_type` |
| `DISKANN` | `zvec.DiskAnnIndexParam` | `metric_type`, `max_degree`, `list_size`, `pq_chunk_num`, `quantize_type` |

- `metric_type` 取值：`zvec.MetricType` 成员（如 `COSINE`、`L2`、`IP`）。
- `quantize_type` 取值：`zvec.QuantizeType` 成员。
- 枚举解析用 `getattr(zvec.DataType, name.upper(), None)`（zvec 枚举是 pybind11 类型，非标准库 `Enum`，不可下标），未知值 → `INVALID_ARGUMENT(400)` 并列出合法值。

---

## 8. 查询参数（`query_param`）

`QueryParamDTO` 同样按 `type` 判别，`mapper.build_query_param` 转发对应字段。`type` 为空字符串时返回 `None`（用字段默认查询行为）。

| `type` | SDK 类 | 专有字段 |
|---|---|---|
| `HNSW` | `zvec.HnswQueryParam` | `ef`, `radius`, `is_linear`, `is_using_refiner`, `extra_params`（含 `prefetch_offset`/`prefetch_lines`） |
| `HNSW_RABITQ` | `zvec.HnswRabitqQueryParam` | `ef`, `radius`, `is_linear`, `is_using_refiner` |
| `IVF` | `zvec.IVFQueryParam` | `nprobe` |
| `FTS` | `zvec.FtsQueryParam` | `default_operator` |
| `VAMANA` | `zvec.VamanaQueryParam` | `ef_search`, `radius`, `is_linear`, `is_using_refiner`, `extra_params` |
| `DISKANN` | `zvec.DiskAnnQueryParam` | `list_size`, `radius`, `is_linear`, `is_using_refiner` |

- **FTS 查询体** `FtsDTO`：`{ query_string, match_string }`，二者至少一个非空才生效。
- 合法的查询 `type` 集合是索引 `type` 集合的子集（无 `FLAT`/`INVERT` 查询参数——FLAT 用默认，INVERT 通过 `filter` 体现）。

---

## 9. 重排器（`reranker`）

`RerankerDTO`，`mapper.build_reranker` 构建。多路查询时融合结果。

| `type` | SDK 类 | 必填/字段 | 说明 |
|---|---|---|---|
| `rrf` | `RrfReRanker` | `rank_constant`（默认 60） | 倒数排名融合，无需 query |
| `weighted` | `WeightedReRanker` | `weights: [0.7, 0.3]`（非空） | 加权分数融合，长度需与 queries 数一致 |
| `local_model` | `DefaultLocalReRanker` | `query`, `rerank_field`（必填）；可选 `model_name`, `model_source`, `device`, `batch_size` | 本地 cross-encoder 重排 |
| `qwen_model` | `QwenReRanker` | `query`, `rerank_field`（必填）；可选 `model`, `api_key` | Qwen API 重排 |

- `local_model` / `qwen_model` 缺 `query` 或 `rerank_field` → 400（在 mapper 校验阶段失败）。
- 模型类重排在**构建时**校验配置（如缺 torch/openai 依赖），实际网络推理可能在运行时才报 `ENGINE_ERROR(500)`。

---

## 10. Embeddings — `/embeddings`

嵌入函数注册表（`EmbeddingManager`）存储**配置**而非实例，按 `(name, encoding_type)` 懒加载并缓存：稠密类型按 `name` 缓存、稀疏类型按 `(name, encoding_type)` 缓存（query/document 编码不同）。

### 10.1 `POST /embeddings` — 注册嵌入函数

- **SDK**：`mapper.build_embedding_function(dto)` 即时构建一次以**校验配置 + 尽早暴露缺失依赖**。
- **请求体**：`EmbeddingConfigDTO`
  ```jsonc
  {
    "name": "bm25_doc",
    "type": "bm25",
    "encoding_type": "document",
    "language": "en",
    "b": 0.75, "k1": 1.2, "corpus": [/* 可选语料 */]
  }
  ```
- **响应**：脱敏配置 `{ name, type, has_api_key, model, dimension, encoding_type, language, base_url }`
- **特性**：
  - 重名 → `ALREADY_EXISTS(409)`。
  - 重型可选依赖（torch / dashtext / openai / jina / qwen …）由 zvec **懒加载**；缺失时 `ImportError` 被包装为 `INVALID_ARGUMENT(400)`，提示具体依赖。
  - `api_key` 永不回显（list 响应只暴露 `has_api_key` 布尔）。

支持类型与 SDK 类：

| `type` | SDK 类 | 接受字段 |
|---|---|---|
| `bm25` | `BM25EmbeddingFunction` | `corpus`, `encoding_type`, `language`, `b`, `k1` |
| `default_local_dense` | `DefaultLocalDenseEmbedding` | `model_source`, `device`, `normalize_embeddings`, `batch_size` |
| `default_local_sparse` | `DefaultLocalSparseEmbedding` | `model_source`, `device`, `encoding_type` |
| `openai` | `OpenAIDenseEmbedding` | `model`, `dimension`, `api_key`, `base_url` |
| `qwen_dense` | `QwenDenseEmbedding` | `dimension`, `model`, `api_key` |
| `qwen_sparse` | `QwenSparseEmbedding` | `dimension`, `model`, `api_key` |
| `jina` | `JinaDenseEmbedding` | `model`, `dimension`, `api_key`, `task` |
| `http` | `HTTPDenseEmbedding` | `base_url`, `model`, `api_key`, `timeout` |

### 10.2 `GET /embeddings` — 列出已注册函数

- **响应**：脱敏配置数组（同 10.1 响应结构）。

### 10.3 `DELETE /embeddings/{name}` — 移除函数

- **响应**：`{ "name", "status":"removed" }`
- **特性**：同时清理该 name 下所有缓存的 `(name, encoding_type)` 实例。未注册 → `NOT_FOUND(404)`。

### 10.4 `POST /embeddings/{name}/embed` — 独立嵌入文本

- **SDK**：`inst.embed(text)`（经 `embedding_manager.embed`）。
- **请求体**：`EmbedTextDTO` → `{ "texts": ["..."], "encoding_type": "query" }`
- **响应**：`{ "function", "count", "vectors": [ ... ] }`
- **特性**：
  - 稠密函数忽略 `encoding_type`；稀疏/BM25 据其选择 query/document 编码。
  - 空文本或非字符串 → 400；空 `texts` 列表直接返回空数组。
  - ndarray / 稀疏 dict 结果经 `embed_to_jsonable` 转 JSON 安全结构（稀疏键为字符串形式的整型）。
  - 主要用于调试与预计算；插入/查询的自动嵌入走 `embedding` 引用而非此端点。

---

## 11. Admin — 统计 / 刷盘 / 引擎信息 / 插件

### 11.1 `GET /collections/{name}/stats` — 集合统计

- **SDK**：读取 `Collection.stats`（`doc_count`、`index_completeness`）与 `Collection.schema`。
- **响应**：`{ "name", "path", "stats": {doc_count, index_completeness}, "schema": {...} }`

### 11.2 `POST /collections/{name}:flush` — 刷盘挂起写入

- **SDK**：`Collection.flush()`。
- **响应**：`{ "name", "status":"flushed" }`
- **特性**：把内存中未持久化的写入落盘；close 前会自动调用，但也可显式触发以保证持久性。

### 11.3 `GET /engine` — 引擎 + 桥接信息

- **SDK**：`zvec.__version__`、`zvec.is_diskann_plugin_loaded()`、`zvec.is_libaio_available()`、`zvec.get_default_jieba_dict_dir()`。
- **响应**：
  ```jsonc
  { "zvec_version", "data_dir", "auto_open",
    "diskann_plugin_loaded": bool|null, "libaio_available": bool|null,
    "jieba_dict_dir": str|null }
  ```
- **特性**：各项插件探测用 try/except 包裹，平台不支持时返回 `null` 而非报错。

### 11.4 `GET /admin/jieba-dict` / `PUT /admin/jieba-dict` — jieba 分词词典目录

- **SDK**：`zvec.get_default_jieba_dict_dir()` / `zvec.set_default_jieba_dict_dir(path)`。
- **PUT 请求体**：`JiebaDictDTO` → `{ "dir": "/path/to/dict" }`
- **响应**：`{ "jieba_dict_dir": "..." }`
- **特性**：影响中文 FTS（`tokenizer_name` 为 jieba 时）的分词词典。

### 11.5 `POST /admin/diskann:load` — 加载 DiskANN 插件

- **SDK**：`zvec.load_diskann_plugin()`。
- **响应**：成功 `{ "diskann_plugin_loaded": true, "status":"loaded" }`；失败 `{ "diskann_plugin_loaded": <当前状态>, "status":"failed", "error": "..." }`。
- **特性**：使用 DISKANN 索引前必须先加载（依赖 libaio）。失败不抛 500，而是返回结构化失败信息，便于客户端探测环境。

### 11.6 `GET /health` — 存活探针

- **SDK**：`zvec.__version__`。
- **响应**：`{ "status":"UP", "zvec_version":"0.5.x" }`
- **特性**：轻量探针，适合 K8s liveness/readiness；不触碰集合或磁盘。

---

## 12. 错误模型

所有错误返回统一形状与语义化 HTTP 状态码：

```json
{ "error": { "code": "NOT_FOUND", "message": "no collection found ..." } }
```

| code | HTTP | 含义 |
|---|---|---|
| `INVALID_ARGUMENT` | 400 | 请求非法 / 未知枚举 / 缺失依赖 / 重排器缺字段 |
| `NOT_FOUND` | 404 | 集合或嵌入函数不存在 |
| `ALREADY_EXISTS` | 409 | 重复创建集合 / 嵌入函数 |
| `COLLECTION_NOT_OPEN` | 409 | auto-open 关闭且集合未打开 |
| `ENGINE_ERROR` | 500 | zvec 引擎底层抛错（已包装原始信息） |
| `INTERNAL_ERROR` | 500 | 未预期异常 |

- 由 `core/errors.py` 的 `ZvecBridgeError` 体系 + FastAPI exception handler 统一产出。
- `HTTPException` 透传给 FastAPI 自身处理器，不吞没。

---

## 13. 配置（环境变量）

引擎在启动 lifespan 中经 `zvec.init(...)` 初始化一次（`--reload` 下二次调用会被捕获忽略）。全部配置走环境变量（12-factor）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `ZVEC_DATA_DIR` | `./data` | 集合磁盘存储目录 |
| `ZVEC_AUTO_OPEN` | `true` | 首次引用集合时是否自动打开 |
| `ZVEC_LOG_LEVEL` | `WARN` | DEBUG/INFO/WARN/ERROR/FATAL |
| `ZVEC_LOG_TYPE` | `CONSOLE` | CONSOLE 或 FILE |
| `ZVEC_LOG_DIR` / `ZVEC_LOG_BASENAME` | `./logs` / null | 日志目录与文件名 |
| `ZVEC_QUERY_THREADS` / `ZVEC_OPTIMIZE_THREADS` | auto | 查询/优化线程池大小 |
| `ZVEC_MEMORY_LIMIT_MB` | auto | 软内存上限 |
| `ZVEC_LOG_FILE_SIZE` / `ZVEC_LOG_OVERDUE_DAYS` | 2048 / 7 | 日志轮转 |
| `ZVEC_INVERT_TO_FORWARD_SCAN_RATIO` | 0.9 | invert→正向扫描阈值 |
| `ZVEC_BRUTE_FORCE_BY_KEYS_RATIO` | 0.1 | 键查找暴力阈值 |
| `ZVEC_FTS_BRUTE_FORCE_BY_KEYS_RATIO` | 0.05 | FTS 暴力阈值 |
| `ZVEC_JIEBA_DICT_DIR` | bundled | jieba 中文 FTS 词典目录 |
| `ZVEC_HOST` / `ZVEC_PORT` | `0.0.0.0` / `8666` | 绑定地址与端口 |

---

## 14. 典型端到端流程

```text
1. GET  /health                                   # 探活
2. POST /embeddings  {name:"bm25_doc", type:"bm25", encoding_type:"document"}   # 注册嵌入
3. POST /collections {schema:{name:"docs", fields:[...], vectors:[...]}}        # 建集合
4. POST /collections/docs/documents  {embedding:{...}, documents:[{id,text}]}   # 文本自动嵌入+插入
5. POST /collections/docs:flush                                                  # 刷盘
6. POST /collections/docs/search  {queries:[{field_name,text}], embedding:{...}, topk:10}  # 文本检索
7. POST /collections/docs:optimize                                               # 优化索引
8. GET  /collections/docs/stats                                                  # 查看统计
9. DELETE /collections/docs                                                      # 销毁
```

交互式文档：启动后访问 `/docs`（Swagger）与 `/redoc`。
