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
│   └── manager.py          # thread-safe collection registry + lifecycle
│
├── model/
│   ├── dto.py              # pydantic request/response models (the wire format)
│   └── mapper.py           # DTO  ⇄  zvec typed objects (the single bridge point)
│
├── service/                # orchestration: manager + mapper per domain
│   ├── collection_service.py
│   ├── document_service.py
│   ├── query_service.py
│   ├── index_service.py
│   └── admin_service.py
│
├── api/                    # thin FastAPI routers (one per domain)
│   ├── collection.py
│   ├── document.py
│   ├── query.py
│   ├── index.py
│   └── admin.py
│
├── test_e2e.py             # core flow tests (39 assertions)
└── test_e2e_advanced.py    # index/reranker/multi-vector/column tests (19 assertions)
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
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
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
| `ZVEC_HOST` / `ZVEC_PORT` | `0.0.0.0` / `8000` | bind address |

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

### Search — `/collections/{name}/search`

A single request carries one or more `queries`; each targets one field and is
either a **vector** query (`vector` or `id`) or a **full-text** query (`fts`).
Multiple queries are fused by a `reranker`.

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
- `reranker.type` is `rrf` (reciprocal rank fusion) or `weighted` (needs
  `weights: [0.7, 0.3]`).

### Indexes — `/collections/{name}/indexes/{field_name}`

| Method | Path | Description |
|---|---|---|
| `POST` | `/.../indexes/{field_name}` | create an index |
| `DELETE` | `/.../indexes/{field_name}` | drop an index |
| `POST` | `/collections/{name}:optimize` | optimize (merge/rebuild) |

Supported index `type`s: `HNSW`, `HNSW_RABITQ`, `IVF`, `FLAT`, `INVERT`,
`FTS`, `VAMANA`. The query `param.type` must match the field's index type.

### Admin

| Method | Path | Description |
|---|---|---|
| `GET` | `/collections/{name}/stats` | doc count + index completeness |
| `POST` | `/collections/{name}:flush` | flush pending writes |
| `GET` | `/engine` | bridge + engine info |
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
```

Both suites use FastAPI's `TestClient` and a throwaway data dir, so they run
in-process without a standing server.
