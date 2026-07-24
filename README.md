# zvec-py-bridge

A REST façade that exposes the full **zvec 0.5.x** vector-database Python SDK
over HTTP, so any language (Java, Go, JS, …) can drive zvec through plain JSON.

> zvec is a high-performance vector database engine with a native C++ backend
> and Python bindings. This bridge wraps those bindings in a stateless FastAPI
> service.

---

## Architecture

```
server/
├── main.py                 # FastAPI app: lifespan (zvec.init), routers, error handlers
├── config.py               # env-driven settings (data dir, threads, log, port)
├── requirements.txt
│
├── core/
│   ├── errors.py           # typed exceptions + uniform JSON error responses
│   ├── manager.py          # thread-safe collection registry + lifecycle
│   └── embedding_manager.py# thread-safe embedding-function registry
│
├── model/
│   ├── dto.py              # pydantic request/response models (the wire format)
│   └── mapper.py           # DTO  ⇄  zvec typed objects (the single bridge point)
│
├── service/                # orchestration: manager + mapper per domain
│   ├── collection_service.py
│   ├── document_service.py
│   ├── embedding_service.py
│   ├── query_service.py
│   ├── index_service.py
│   └── admin_service.py
│
├── api/                    # thin FastAPI routers (one per domain)
│   ├── collection.py
│   ├── document.py
│   ├── embedding.py
│   ├── query.py
│   ├── index.py
│   └── admin.py
│
├── test_e2e.py             # core flow tests (39 assertions)
├── test_e2e_advanced.py    # index/reranker/multi-vector/column tests (19 assertions)
└── test_e2e_embedding.py   # embedding/text-insert/jieba/diskann tests (32 assertions)
```

**Layering rule:** the `api/` layer never imports `zvec`; it only talks to
`service/`. `service/` never parses JSON; it only calls `core.manager` +
`model.mapper`. `model.mapper` is the *only* module that knows how to turn
JSON into zvec enums/params/docs and back, so the rest of the codebase stays
free of zvec-specific branching.

---

## Quick start

```bash
cd server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8666
```

Health check:

```bash
curl http://localhost:8666/health
# {"status":"UP","zvec_version":"0.5.1"}
```

Configuration is entirely environment-variable driven:

| Variable | Default | Meaning |
|---|---|---|
| `ZVEC_DATA_DIR` | `./data` | where collections are stored on disk |
| `ZVEC_AUTO_OPEN` | `true` | auto-open a collection on first reference |
| `ZVEC_LOG_LEVEL` | `WARN` | DEBUG/INFO/WARN/ERROR/FATAL |
| `ZVEC_LOG_TYPE` | `CONSOLE` | CONSOLE or FILE |
| `ZVEC_QUERY_THREADS` | auto | query thread pool size |
| `ZVEC_MEMORY_LIMIT_MB` | auto | soft memory cap |
| `ZVEC_LOG_FILE_SIZE` / `ZVEC_LOG_OVERDUE_DAYS` | 2048 / 7 | log rotation |
| `ZVEC_INVERT_TO_FORWARD_SCAN_RATIO` | 0.9 | invert→forward-scan threshold |
| `ZVEC_BRUTE_FORCE_BY_KEYS_RATIO` | 0.1 | brute-force key-lookup threshold |
| `ZVEC_FTS_BRUTE_FORCE_BY_KEYS_RATIO` | 0.05 | FTS brute-force threshold |
| `ZVEC_JIEBA_DICT_DIR` | bundled | jieba Chinese FTS tokenizer dict |
| `ZVEC_CORS_ORIGINS` | `*` | comma-separated allowed CORS origins; set to a concrete list to lock down before exposing beyond a trusted network |
| `ZVEC_HOST` / `ZVEC_PORT` | `0.0.0.0` / `8666` | bind address |

---

## REST API

Interactive docs are available at `/docs` (Swagger) and `/redoc` once running.

### Collections — `/collections`

| Method | Path | Description |
|---|---|---|
| `POST` | `/collections` | create a collection from a full schema |
| `GET` | `/collections` | list all collections (opened + on-disk) |
| `GET` | `/collections/{name}` | collection info + schema |
| `POST` | `/collections/{name}/open` | open an existing collection |
| `POST` | `/collections/{name}/close` | unload a collection (data stays) |
| `DELETE` | `/collections/{name}` | permanently destroy a collection |

**Create** — the schema describes scalar `fields` and `vectors`, each with an
optional `index_param` discriminated by `type`:

```jsonc
POST /collections
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
  }
}
```

### Columns — `/collections/{name}/columns`

| Method | Path | Description |
|---|---|---|
| `POST` | `/collections/{name}/columns` | add a column (with backfill `expression`) |
| `PUT` | `/collections/{name}/columns/{old_name}` | rename / re-schema a column |
| `DELETE` | `/collections/{name}/columns/{field_name}` | drop a column |

> `alter_column` only supports basic numeric types (INT32/64, UINT32/64,
> FLOAT, DOUBLE) — a zvec engine limitation.

### Documents — `/collections/{name}/documents`

| Method | Path | Description |
|---|---|---|
| `POST` | `/.../documents` | insert |
| `PUT` | `/.../documents` | upsert |
| `PATCH` | `/.../documents` | update (partial) |
| `DELETE` | `/.../documents` | delete by id (body: `{"ids":[...]}`) |
| `POST` | `/.../documents:deleteByIds` | delete by id — POST variant (prefer this when clients/proxies strip DELETE bodies) |
| `POST` | `/.../documents:deleteByFilter` | delete by filter expression |
| `POST` | `/.../documents:fetch` | fetch by id |

```jsonc
POST /collections/docs/documents
{
  "documents": [
    { "id": "d1",
      "vectors": { "embedding": [0.1, 0.2, ...] },
      "fields":  { "title": "hello", "tag": "a" } }
  ]
}
```

Sparse vectors use **integer keys** (JSON object keys arrive as strings; the
bridge casts them to `uint32` automatically):

```jsonc
{ "id": "s1", "vectors": { "sparse": { "1": 0.5, "2": 1.0 } } }
```

#### Text → vector (auto-embed)

Instead of pre-computing vectors, send `text` and an `embedding` reference to a
[registered embedding function](#embeddings--embeddings). The bridge embeds the
text into the named vector field:

```jsonc
POST /collections/docs/documents
{
  "embedding": { "function": "my_emb", "field": "embedding" },
  "documents": [
    { "id": "d1", "text": "machine learning is fun",
      "fields": { "tag": "a" } }
  ]
}
```

For sparse/BM25 functions, `encoding_type` (`query` | `document`) selects the
encoding strategy; it is auto-defaulted (document on insert, query on search)
but can be set explicitly in the `embedding` ref.

### Search — `/collections/{name}/search`

A single request carries one or more `queries`; each targets one field and is
either a **vector** query (`vector` or `id`), a **text** query (`text` + an
`embedding` ref), or a **full-text** query (`fts`). Multiple queries are fused
by a `reranker`.

```jsonc
POST /collections/docs/search
{
  "queries": [
    { "field_name": "embedding", "vector": [0.1, 0.2, ...],
      "param": { "type": "HNSW", "ef": 100 } },
    { "field_name": "content", "fts": { "match_string": "machine learning" },
      "param": { "type": "FTS", "default_operator": "AND" } }
  ],
  "topk": 10,
  "filter": "tag = 'a'",
  "include_vector": false,
  "output_fields": ["title"],
  "reranker": { "type": "rrf", "rank_constant": 60 }
}
```

- `filter` uses zvec's SQL-like syntax: `tag = 'a'`, `age > 30`, `cat IN (...)`.
  Equality is a single `=`.
- **Text query**: send `text` instead of `vector` plus a top-level `embedding`
  ref; the bridge embeds it first.
- `reranker.type`:
  - `rrf` — reciprocal rank fusion (`rank_constant`)
  - `weighted` — weighted score fusion (`weights: [0.7, 0.3]`)
  - `local_model` — `DefaultLocalReRanker` cross-encoder (needs `query` +
    `rerank_field`)
  - `qwen_model` — `QwenReRanker` API reranker (needs `query` + `rerank_field`)

### Indexes — `/collections/{name}/indexes/{field_name}`

| Method | Path | Description |
|---|---|---|
| `POST` | `/.../indexes/{field_name}` | create an index |
| `DELETE` | `/.../indexes/{field_name}` | drop an index |
| `POST` | `/collections/{name}:optimize` | optimize (merge/rebuild) |

Supported index `type`s: `HNSW`, `HNSW_RABITQ`, `IVF`, `FLAT`, `INVERT`,
`FTS`, `VAMANA`, `DISKANN`. The query `param.type` must match the field's index
type. `DISKANN` (disk-based ANN) requires the libaio plugin — load it via
`POST /admin/diskann:load`.

### Embeddings — `/embeddings`

| Method | Path | Description |
|---|---|---|
| `POST` | `/embeddings` | register a named embedding function |
| `GET` | `/embeddings` | list registered functions |
| `DELETE` | `/embeddings/{name}` | remove a function |
| `POST` | `/embeddings/{name}/embed` | embed text(s) into vectors |

Supported `type`s: `bm25`, `default_local_dense`, `default_local_sparse`,
`openai`, `qwen_dense`, `qwen_sparse`, `jina`, `http`. Heavy optional deps
(torch, dashtext, openai, …) are imported lazily; a missing dep is reported at
registration time. Registered functions feed the `embedding` ref on
insert/search for end-to-end text→vector retrieval.

### Admin

| Method | Path | Description |
|---|---|---|
| `GET` | `/collections/{name}/stats` | doc count + index completeness |
| `POST` | `/collections/{name}:flush` | flush pending writes |
| `GET` | `/engine` | bridge + engine info (libaio, jieba, diskann) |
| `GET` | `/admin/jieba-dict` | get the jieba FTS tokenizer dict dir |
| `PUT` | `/admin/jieba-dict` | set the jieba dict dir |
| `POST` | `/admin/diskann:load` | load the DiskANN plugin |
| `GET` | `/health` | liveness |

---

## Error model

Every error returns a uniform shape with an HTTP-appropriate status code:

```json
{ "error": { "code": "NOT_FOUND", "message": "no collection found ..." } }
```

| code | HTTP | meaning |
|---|---|---|
| `INVALID_ARGUMENT` | 400 | bad request / unknown enum |
| `NOT_FOUND` | 404 | collection or doc missing |
| `ALREADY_EXISTS` | 409 | duplicate create |
| `COLLECTION_NOT_OPEN` | 409 | auto-open disabled and not opened |
| `ENGINE_ERROR` | 500 | zvec raised underneath |

---

## Testing

```bash
cd server
python test_e2e.py          # 39 assertions: lifecycle, DML, search, FTS, filter
python test_e2e_advanced.py # 19 assertions: index DDL, RRF/weighted, multi-vector, columns
python test_e2e_embedding.py# 32 assertions: embedding, text-insert/search, jieba, diskann
```

All three suites use FastAPI's `TestClient` and a throwaway data dir, so they
run in-process without a standing server. The embedding suite uses BM25 (needs
`pip install dashtext`); model-based rerankers/embeddings (OpenAI, Qwen, local
sentence-transformer) are validated at construction but require their own
optional deps / API keys to run end-to-end.
