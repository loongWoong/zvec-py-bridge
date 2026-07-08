"""Query: vector search, multi-vector, FTS, hybrid, filter, reranker."""
from __future__ import annotations

from typing import Any

from core.errors import InvalidArgumentError, ZvecRuntimeError
from core.manager import ZvecManager, manager
from model.dto import SearchDTO
from model.mapper import build_query, build_reranker, doc_to_dict
from service.embedding_service import EmbeddingService


class QueryService:
    def __init__(self, mgr: ZvecManager = manager, emb: EmbeddingService | None = None) -> None:
        self._mgr = mgr
        self._emb = emb

    @property
    def embedding(self) -> EmbeddingService:
        if self._emb is None:
            from service import embedding_service

            self._emb = embedding_service
        return self._emb

    def search(self, name: str, dto: SearchDTO) -> dict[str, Any]:
        if not dto.queries:
            raise InvalidArgumentError("at least one query is required")
        # auto-embed text -> vector when an embedding reference is supplied
        if dto.embedding is not None:
            for q in dto.queries:
                if q.text and q.vector is None and q.id is None:
                    self.embedding.resolve_query_vector(q, dto.embedding)
        collection = self._mgr.get(name)
        queries = [build_query(q) for q in dto.queries]
        reranker = build_reranker(dto.reranker)
        try:
            results = collection.query(
                queries=queries,
                topk=dto.topk,
                filter=dto.filter,
                include_vector=dto.include_vector,
                output_fields=dto.output_fields,
                reranker=reranker,
            )
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"query failed: {exc}") from exc
        return {
            "name": name,
            "topk": dto.topk,
            "filter": dto.filter,
            "count": len(results),
            "documents": [
                doc_to_dict(d, include_vector=dto.include_vector) for d in results
            ],
        }
