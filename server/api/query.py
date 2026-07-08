"""Query / search routes."""
from __future__ import annotations

from fastapi import APIRouter

from model.dto import SearchDTO
from service import query_service

router = APIRouter(prefix="/collections", tags=["query"])


@router.post("/{name}/search", summary="Vector / FTS / hybrid search")
def search(name: str, dto: SearchDTO):
    """Perform one or more queries against a collection.

    Each entry in ``queries`` targets a single field and is either a vector
    query (``vector`` or ``id``) or a full-text query (``fts``). Multiple
    queries are fused by the configured ``reranker`` (RRF by default when more
    than one query is supplied)."""
    return query_service.search(name, dto)
