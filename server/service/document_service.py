"""Document DML: insert / upsert / update / delete / fetch."""
from __future__ import annotations

from typing import Any

from core.errors import ZvecRuntimeError
from core.manager import ZvecManager, manager
from model.dto import DeleteByFilterDTO, DocumentBatchDTO, FetchDTO
from model.mapper import build_doc, doc_to_dict, status_to_dict
from service.embedding_service import EmbeddingService


class DocumentService:
    def __init__(self, mgr: ZvecManager = manager, emb: EmbeddingService | None = None) -> None:
        self._mgr = mgr
        self._emb = emb  # lazily resolved to avoid a circular import at module load

    @property
    def embedding(self) -> EmbeddingService:
        if self._emb is None:
            from service import embedding_service

            self._emb = embedding_service
        return self._emb

    def _apply(self, name: str, op: str, dto: DocumentBatchDTO) -> dict[str, Any]:
        # auto-embed text -> vector when an embedding reference is supplied
        if dto.embedding is not None:
            self.embedding.populate_documents(dto.documents, dto.embedding)
        collection = self._mgr.get(name)
        docs = [build_doc(d) for d in dto.documents]
        method = getattr(collection, op)
        # Serialise engine mutations per collection (see ZvecManager.lock_for).
        with self._mgr.lock_for(name):
            try:
                result = method(docs)
            except Exception as exc:  # noqa: BLE001
                raise ZvecRuntimeError(f"{op} failed: {exc}") from exc
        statuses = result if isinstance(result, list) else [result]
        return {
            "name": name,
            "op": op,
            "count": len(dto.documents),
            "results": [
                {
                    "id": dto.documents[i].id,
                    "status": status_to_dict(s),
                }
                for i, s in enumerate(statuses)
            ],
        }

    def insert(self, name: str, dto: DocumentBatchDTO) -> dict[str, Any]:
        return self._apply(name, "insert", dto)

    def upsert(self, name: str, dto: DocumentBatchDTO) -> dict[str, Any]:
        return self._apply(name, "upsert", dto)

    def update(self, name: str, dto: DocumentBatchDTO) -> dict[str, Any]:
        return self._apply(name, "update", dto)

    def delete(self, name: str, ids: list[str]) -> dict[str, Any]:
        collection = self._mgr.get(name)
        with self._mgr.lock_for(name):
            try:
                result = collection.delete(ids)
            except Exception as exc:  # noqa: BLE001
                raise ZvecRuntimeError(f"delete failed: {exc}") from exc
        statuses = result if isinstance(result, list) else [result]
        return {
            "name": name,
            "op": "delete",
            "count": len(ids),
            "results": [
                {"id": ids[i], "status": status_to_dict(s)}
                for i, s in enumerate(statuses)
            ],
        }

    def delete_by_filter(self, name: str, dto: DeleteByFilterDTO) -> dict[str, Any]:
        collection = self._mgr.get(name)
        with self._mgr.lock_for(name):
            try:
                collection.delete_by_filter(dto.filter)
            except Exception as exc:  # noqa: BLE001
                raise ZvecRuntimeError(f"delete_by_filter failed: {exc}") from exc
        return {"name": name, "op": "delete_by_filter", "filter": dto.filter, "status": "ok"}

    def fetch(self, name: str, dto: FetchDTO) -> dict[str, Any]:
        collection = self._mgr.get(name)
        with self._mgr.lock_for(name):
            try:
                docs = collection.fetch(
                    dto.ids,
                    output_fields=dto.output_fields,
                    include_vector=dto.include_vector,
                )
            except Exception as exc:  # noqa: BLE001
                raise ZvecRuntimeError(f"fetch failed: {exc}") from exc
        # zvec's fetch returns either a mapping (id -> Doc) or a list of Docs
        # depending on the engine version; normalise both into a list.
        doc_list = list(docs.values()) if hasattr(docs, "values") else list(docs)
        return {
            "name": name,
            "count": len(doc_list),
            "documents": [
                doc_to_dict(d, include_vector=dto.include_vector) for d in doc_list
            ],
        }
