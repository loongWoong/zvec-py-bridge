"""Advanced end-to-end tests: index DDL, multi-vector + RRF, update, delete,
column alter/drop, IVF/FLAT index types.
"""
from __future__ import annotations

import os
import sys

os.environ["ZVEC_DATA_DIR"] = "./_test_data2"
os.environ["ZVEC_LOG_LEVEL"] = "ERROR"

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)
passed = failed = 0


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

    shutil.rmtree("./_test_data2", ignore_errors=True)

    # collection with two vector fields + fts for hybrid/RRF
    schema = {
        "name": "hybrid",
        "fields": [
            {"name": "content", "data_type": "STRING", "nullable": True,
             "index_param": {"type": "FTS", "tokenizer_name": "standard"}},
            {"name": "cat", "data_type": "STRING", "nullable": True},
        ],
        "vectors": [
            {"name": "dense", "data_type": "VECTOR_FP32", "dimension": 4},
            {"name": "sparse", "data_type": "SPARSE_VECTOR_FP32", "dimension": 0},
        ],
    }
    r = client.post("/collections", json={"schema": schema})
    ok(r, "create hybrid collection")

    # insert with two vectors (dense + sparse)
    r = client.post("/collections/hybrid/documents", json={"documents": [
        {"id": "h1", "vectors": {"dense": [1, 0, 0, 0], "sparse": {1: 1.0, 2: 0.5}},
         "fields": {"content": "machine learning ai", "cat": "tech"}},
        {"id": "h2", "vectors": {"dense": [0, 1, 0, 0], "sparse": {3: 1.0}},
         "fields": {"content": "vector database search", "cat": "tech"}},
        {"id": "h3", "vectors": {"dense": [0, 0, 1, 0], "sparse": {4: 1.0}},
         "fields": {"content": "cooking recipe food", "cat": "life"}},
    ]})
    ok(r, "insert multi-vector docs")
    check("insert 3 ok", r.json()["count"] == 3)

    # ---- create index on dense (HNSW) ----
    r = client.post("/collections/hybrid/indexes/dense", json={
        "index_param": {"type": "HNSW", "metric_type": "COSINE", "m": 16, "ef_construction": 200},
        "concurrency": 0,
    })
    ok(r, "create HNSW index on dense")

    # ---- create IVF index (replace) -> drop first ----
    r = client.delete("/collections/hybrid/indexes/dense")
    ok(r, "drop HNSW index")
    r = client.post("/collections/hybrid/indexes/dense", json={
        "index_param": {"type": "IVF", "metric_type": "L2", "n_list": 4, "n_iters": 10},
    })
    ok(r, "create IVF index on dense")

    # ---- create FLAT index ----
    r = client.delete("/collections/hybrid/indexes/dense")
    r = client.post("/collections/hybrid/indexes/dense", json={
        "index_param": {"type": "FLAT", "metric_type": "COSINE"},
    })
    ok(r, "create FLAT index on dense")

    # ---- create INVERT index on cat ----
    r = client.post("/collections/hybrid/indexes/cat", json={
        "index_param": {"type": "INVERT", "enable_range_optimization": True},
    })
    ok(r, "create INVERT index on cat")

    # ---- multi-vector + RRF reranker (dense vector + fts) ----
    # dense field has a FLAT index, so we omit the query param type.
    r = client.post("/collections/hybrid/search", json={
        "queries": [
            {"field_name": "dense", "vector": [1, 0, 0, 0]},
            {"field_name": "content", "fts": {"match_string": "machine learning"}},
        ],
        "topk": 3,
        "reranker": {"type": "rrf", "rank_constant": 60},
        "output_fields": ["content", "cat"],
    })
    ok(r, "hybrid multi-query + RRF")
    check("rrf returns results", r.json()["count"] >= 1)

    # ---- weighted reranker ----
    r = client.post("/collections/hybrid/search", json={
        "queries": [
            {"field_name": "dense", "vector": [1, 0, 0, 0]},
            {"field_name": "content", "fts": {"match_string": "learning"}},
        ],
        "topk": 3,
        "reranker": {"type": "weighted", "weights": [0.7, 0.3]},
    })
    ok(r, "hybrid + weighted reranker")

    # ---- update via PATCH ----
    r = client.patch("/collections/hybrid/documents", json={
        "documents": [{"id": "h1", "fields": {"cat": "ai-tech"}}]
    })
    ok(r, "patch update")
    r = client.post("/collections/hybrid/documents:fetch", json={"ids": ["h1"], "include_vector": False})
    check("patch updated cat", r.json()["documents"][0]["fields"]["cat"] == "ai-tech")

    # ---- delete by id ----
    r = client.request("DELETE", "/collections/hybrid/documents", json={"ids": ["h3"]})
    ok(r, "delete by id")
    r = client.get("/collections/hybrid/stats")
    check("delete reduced count to 2", r.json()["stats"]["doc_count"] == 2)

    # ---- column alter (rename) + drop ----
    # zvec alter_column only supports basic numeric types, so add one first.
    r = client.post("/collections/hybrid/columns", json={
        "field": {"name": "popularity", "data_type": "INT64", "nullable": True},
        "expression": "0",
    })
    ok(r, "add numeric column popularity")
    r = client.put("/collections/hybrid/columns/popularity", json={"new_name": "pop"})
    ok(r, "alter/rename column popularity->pop")
    r = client.delete("/collections/hybrid/columns/pop")
    ok(r, "drop column pop")

    # ---- add column ----
    r = client.post("/collections/hybrid/columns", json={
        "field": {"name": "score", "data_type": "FLOAT", "nullable": True},
        "expression": "0.0",
    })
    ok(r, "add column score")

    client.delete("/collections/hybrid")
    print(f"\n{'='*50}\nPASSED: {passed}   FAILED: {failed}\n{'='*50}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
