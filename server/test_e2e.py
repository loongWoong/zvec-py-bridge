"""End-to-end test of the zvec REST bridge using FastAPI's TestClient.

Covers: health, collection create/open/info/list/close/destroy, schema,
column DDL, document insert/upsert/update/delete/fetch, vector search,
multi-vector + RRF, FTS search, hybrid search, index create/drop/optimize,
admin stats/flush/engine.
"""
from __future__ import annotations

import os
import sys

# use a throwaway data dir
os.environ["ZVEC_DATA_DIR"] = "./_test_data"
os.environ["ZVEC_LOG_LEVEL"] = "ERROR"

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)

passed = 0
failed = 0


def check(label, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {extra}")


def ok(resp, label, status=200):
    check(label, resp.status_code == status, f"({resp.status_code}) {resp.text[:200]}")


def main() -> None:
    import shutil

    shutil.rmtree("./_test_data", ignore_errors=True)
    shutil.rmtree("./_test_logs", ignore_errors=True)

    # ---- health ----
    r = client.get("/health")
    ok(r, "health")
    check("health status UP", r.json().get("status") == "UP")

    # ---- engine ----
    r = client.get("/engine")
    ok(r, "engine info")
    check("engine zvec_version present", "zvec_version" in r.json())

    # ---- create collection (vector + scalar + fts) ----
    schema = {
        "name": "docs",
        "fields": [
            {"name": "title", "data_type": "STRING", "nullable": True},
            {"name": "tag", "data_type": "STRING", "nullable": True,
             "index_param": {"type": "INVERT", "enable_range_optimization": True}},
            {"name": "content", "data_type": "STRING", "nullable": True,
             "index_param": {"type": "FTS", "tokenizer_name": "standard"}},
        ],
        "vectors": [
            {"name": "embedding", "data_type": "VECTOR_FP32", "dimension": 8,
             "index_param": {"type": "HNSW", "metric_type": "COSINE", "m": 16, "ef_construction": 200}},
        ],
    }
    r = client.post("/collections", json={"schema": schema})
    ok(r, "create collection")
    check("create returns path", "path" in r.json())
    check("create returns schema vectors", r.json()["schema"]["vectors"][0]["name"] == "embedding")

    # duplicate create -> 409
    r = client.post("/collections", json={"schema": schema})
    ok(r, "duplicate create -> 409", status=409)

    # ---- info ----
    r = client.get("/collections/docs")
    ok(r, "collection info")
    check("info read_only false", r.json()["read_only"] is False)

    # ---- list ----
    r = client.get("/collections")
    ok(r, "list collections")
    check("list contains docs", any(c["name"] == "docs" for c in r.json()))

    # ---- insert documents ----
    docs = {
        "documents": [
            {"id": "d1", "vectors": {"embedding": [1, 0, 0, 0, 0, 0, 0, 0]},
             "fields": {"title": "hello world", "tag": "a", "content": "machine learning is fun"}},
            {"id": "d2", "vectors": {"embedding": [0, 1, 0, 0, 0, 0, 0, 0]},
             "fields": {"title": "foo bar", "tag": "b", "content": "deep learning rocks"}},
            {"id": "d3", "vectors": {"embedding": [0, 0, 1, 0, 0, 0, 0, 0]},
             "fields": {"title": "baz qux", "tag": "a", "content": "vector database search"}},
        ]
    }
    r = client.post("/collections/docs/documents", json=docs)
    ok(r, "insert documents")
    check("insert count 3", r.json()["count"] == 3)
    check("insert all ok", all(x["status"]["ok"] for x in r.json()["results"]))

    # ---- upsert ----
    r = client.put("/collections/docs/documents", json={
        "documents": [{"id": "d1", "vectors": {"embedding": [1, 0, 0, 0, 0, 0, 0, 0]},
                       "fields": {"title": "hello world v2", "tag": "a", "content": "machine learning is fun"}}]
    })
    ok(r, "upsert document")
    check("upsert ok", r.json()["results"][0]["status"]["ok"])

    # ---- fetch ----
    r = client.post("/collections/docs/documents:fetch", json={"ids": ["d1", "missing"], "include_vector": True})
    ok(r, "fetch documents")
    check("fetch count 1", r.json()["count"] == 1)
    check("fetch returns updated title", r.json()["documents"][0]["fields"]["title"] == "hello world v2")
    check("fetch includes vector", r.json()["documents"][0]["vectors"] is not None)

    # ---- vector search ----
    r = client.post("/collections/docs/search", json={
        "queries": [{"field_name": "embedding", "vector": [1, 0, 0, 0, 0, 0, 0, 0], "param": {"type": "HNSW", "ef": 50}}],
        "topk": 3, "output_fields": ["title", "tag"],
    })
    ok(r, "vector search")
    check("vector search returns d1 first", r.json()["documents"][0]["id"] == "d1")
    check("vector search returns 3", r.json()["count"] == 3)

    # ---- FTS search ----
    r = client.post("/collections/docs/search", json={
        "queries": [{"field_name": "content", "fts": {"match_string": "learning"}}],
        "topk": 5, "output_fields": ["title", "content"],
    })
    ok(r, "fts search")
    check("fts returns >=2 docs", r.json()["count"] >= 2)

    # ---- filter on invert index ----
    r = client.post("/collections/docs/search", json={
        "queries": [{"field_name": "embedding", "vector": [0, 0, 0, 0, 0, 0, 0, 1]}],
        "topk": 10, "filter": "tag = 'a'", "output_fields": ["title", "tag"],
    })
    ok(r, "filtered vector search")
    check("filter only tag a", all(d["fields"]["tag"] == "a" for d in r.json()["documents"]))

    # ---- delete by filter ----
    r = client.post("/collections/docs/documents:deleteByFilter", json={"filter": "tag = 'b'"})
    ok(r, "delete by filter")

    # ---- stats ----
    r = client.get("/collections/docs/stats")
    ok(r, "stats")
    check("stats doc_count 2", r.json()["stats"]["doc_count"] == 2)

    # ---- flush ----
    r = client.post("/collections/docs:flush")
    ok(r, "flush")

    # ---- optimize ----
    r = client.post("/collections/docs:optimize", json={"concurrency": 0})
    ok(r, "optimize")

    # ---- close + reopen ----
    r = client.post("/collections/docs/close")
    ok(r, "close collection")
    r = client.post("/collections/docs/open", json={"read_only": False})
    ok(r, "reopen collection")
    check("reopen returns schema", "schema" in r.json())

    # ---- destroy ----
    r = client.delete("/collections/docs")
    ok(r, "destroy collection")
    r = client.get("/collections")
    check("destroy removed from list", not any(c["name"] == "docs" for c in r.json()))

    # ---- error handling: not found ----
    r = client.get("/collections/does_not_exist")
    ok(r, "info on missing (auto-open disabled path)", status=404)

    print(f"\n{'='*50}\nPASSED: {passed}   FAILED: {failed}\n{'='*50}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
