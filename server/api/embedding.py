"""Embedding function registration and standalone embed routes."""
from __future__ import annotations

from fastapi import APIRouter

from model.dto import EmbeddingConfigDTO, EmbedTextDTO
from service import embedding_service

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("", summary="Register an embedding function")
def register_embedding(dto: EmbeddingConfigDTO):
    """Register a named embedding function (BM25, OpenAI, Qwen, Jina, local
    sentence-transformer, or a custom HTTP endpoint). The function is built
    immediately so missing optional dependencies are reported at registration."""
    return embedding_service.register(dto)


@router.get("", summary="List registered embedding functions")
def list_embeddings():
    return embedding_service.list()


@router.delete("/{name}", summary="Remove an embedding function")
def remove_embedding(name: str):
    return embedding_service.remove(name)


@router.post("/{name}/embed", summary="Embed text(s) into vectors")
def embed(name: str, dto: EmbedTextDTO):
    """Embed a list of texts using a registered function.

    For sparse/BM25 functions, ``encoding_type`` (``query`` | ``document``)
    selects the encoding strategy; dense functions ignore it."""
    return embedding_service.embed(name, dto)
