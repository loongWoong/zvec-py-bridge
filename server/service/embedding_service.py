"""Embedding function registration, standalone embed, and text→vector
integration helpers used by the document and query services."""
from __future__ import annotations

from typing import Any

from core.embedding_manager import EmbeddingManager, embedding_manager
from core.errors import InvalidArgumentError
from model.dto import DocumentDTO, EmbeddingConfigDTO, EmbeddingRefDTO, EmbedTextDTO, QueryDTO


class EmbeddingService:
    def __init__(self, mgr: EmbeddingManager = embedding_manager) -> None:
        self._mgr = mgr

    # ------------------------------------------------------------------ #
    # registry
    # ------------------------------------------------------------------ #
    def register(self, dto: EmbeddingConfigDTO) -> dict[str, Any]:
        return self._mgr.register(dto)

    def remove(self, name: str) -> dict[str, Any]:
        return self._mgr.remove(name)

    def list(self) -> list[dict[str, Any]]:
        return self._mgr.list()

    # ------------------------------------------------------------------ #
    # standalone embed
    # ------------------------------------------------------------------ #
    def embed(self, name: str, dto: EmbedTextDTO) -> dict[str, Any]:
        vectors = self._mgr.embed(name, dto.texts, encoding_type=dto.encoding_type)
        return {"function": name, "count": len(vectors), "vectors": vectors}

    # ------------------------------------------------------------------ #
    # integration helpers
    # ------------------------------------------------------------------ #
    def populate_documents(
        self, documents: list[DocumentDTO], ref: EmbeddingRefDTO
    ) -> None:
        """Mutate ``documents`` in place: embed each doc's ``text`` into the
        vector field named by ``ref.field`` (only for docs that lack a vector
        for that field but carry ``text``)."""
        texts, indices = [], []
        for i, doc in enumerate(documents):
            if doc.vectors and doc.vectors.get(ref.field) is not None:
                continue  # explicit vector wins
            if not doc.text:
                raise InvalidArgumentError(
                    f"document {doc.id!r} has no vector for field {ref.field!r} "
                    f"and no `text` to embed"
                )
            texts.append(doc.text)
            indices.append(i)
        if not texts:
            return
        vectors = self._mgr.embed(ref.function, texts, encoding_type=ref.encoding_type)
        for i, vec in zip(indices, vectors):
            documents[i].vectors = {**(documents[i].vectors or {}), ref.field: vec}

    def resolve_query_vector(self, query: QueryDTO, ref: EmbeddingRefDTO) -> None:
        """If a query carries ``text`` but no explicit ``vector``, embed the
        text and set ``query.vector`` so the downstream mapper can build it."""
        if query.vector is not None or query.id is not None:
            return  # explicit vector / id wins
        if not query.text:
            raise InvalidArgumentError(
                f"query on field {query.field_name!r} has no vector, id, or text"
            )
        if query.field_name != ref.field:
            raise InvalidArgumentError(
                f"embedding ref targets field {ref.field!r} but query targets "
                f"{query.field_name!r}"
            )
        vec = self._mgr.embed(ref.function, [query.text], encoding_type=ref.encoding_type)[0]
        query.vector = vec
