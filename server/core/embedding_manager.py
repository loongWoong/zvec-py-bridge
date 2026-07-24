"""Thread-safe registry for zvec embedding functions.

Embedding functions are stateful (model handles, API clients, tokenizers) and
some — notably BM25 / sparse encoders — encode *differently* for documents
versus queries (``encoding_type``). This manager:

* stores *configs* (not just instances) so it can rebuild with a different
  ``encoding_type`` on demand;
* lazily instantiates and caches instances keyed by ``(name, encoding_type)``
  for sparse types, and by ``name`` alone for dense types (which have no
  encoding_type concept);
* fails fast at registration time so a missing optional dependency (torch,
  dashtext, openai, ...) is reported immediately rather than at first embed.
"""
from __future__ import annotations

import threading
from typing import Any

from core.errors import (
    AlreadyExistsError,
    InvalidArgumentError,
    NotFoundError,
    ZvecRuntimeError,
)
from model.dto import EmbeddingConfigDTO
from model.mapper import build_embedding_function, embedding_to_dict, embed_to_jsonable

# Types whose ``embed()`` depends on an ``encoding_type`` (query vs document).
_SPARSE_TYPES = {"bm25", "default_local_sparse", "qwen_sparse"}


class EmbeddingManager:
    def __init__(self) -> None:
        self._configs: dict[str, EmbeddingConfigDTO] = {}
        self._instances: dict[tuple[str, str | None], Any] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #
    def register(self, dto: EmbeddingConfigDTO) -> dict[str, Any]:
        with self._lock:
            if dto.name in self._configs:
                raise AlreadyExistsError(f"embedding function {dto.name!r} already registered")
            # build once to validate config + surface missing dependencies early
            build_embedding_function(dto)
            self._configs[dto.name] = dto
            # drop only this function's cached instances — previously the whole
            # cache (incl. already-loaded heavy models) was cleared on every
            # registration, forcing needless reloads.
            for key in [k for k in self._instances if k[0] == dto.name]:
                self._instances.pop(key, None)
            return embedding_to_dict(dto.name, dto)

    def remove(self, name: str) -> dict[str, Any]:
        with self._lock:
            if name not in self._configs:
                raise NotFoundError(f"embedding function {name!r} not registered")
            self._configs.pop(name)
            # drop cached instances for this name
            for key in [k for k in self._instances if k[0] == name]:
                self._instances.pop(key, None)
            return {"name": name, "status": "removed"}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [embedding_to_dict(n, c) for n, c in self._configs.items()]

    def get_config(self, name: str) -> EmbeddingConfigDTO:
        with self._lock:
            if name not in self._configs:
                raise NotFoundError(f"embedding function {name!r} not registered")
            return self._configs[name]

    # ------------------------------------------------------------------ #
    # embed
    # ------------------------------------------------------------------ #
    def _instance_for(self, name: str, encoding_type: str | None) -> Any:
        config = self.get_config(name)
        is_sparse = (config.type or "").lower() in _SPARSE_TYPES
        # dense types ignore encoding_type entirely
        key = (name, encoding_type) if (is_sparse and encoding_type) else (name, None)
        # Fast path: already cached (checked under the lock).
        with self._lock:
            cached = self._instances.get(key)
            if cached is not None:
                return cached
        # Build outside the global lock so loading a heavy model (torch /
        # sentence-transformers / …) does not block embeds on *other* functions.
        # Another thread may build the same key while we load; we double-check
        # and keep whichever instance landed first.
        cfg = config
        if is_sparse and encoding_type:
            # rebuild with the requested encoding_type (query vs document)
            cfg = config.model_copy(update={"encoding_type": encoding_type})
        built = build_embedding_function(cfg)
        with self._lock:
            existing = self._instances.get(key)
            if existing is not None:
                return existing
            self._instances[key] = built
            return built

    def embed(self, name: str, texts: list[str], encoding_type: str | None = None) -> list[Any]:
        if not texts:
            return []
        # _instance_for manages its own locking and may load a model outside the
        # lock; we must not hold the global lock across the (potentially slow)
        # build or the embedding loop.
        inst = self._instance_for(name, encoding_type)
        results: list[Any] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise InvalidArgumentError("embed input must be a non-empty string")
            try:
                results.append(embed_to_jsonable(inst.embed(text)))
            except Exception as exc:  # noqa: BLE001
                raise ZvecRuntimeError(f"embed failed for {name!r}: {exc}") from exc
        return results


# process-wide singleton
embedding_manager = EmbeddingManager()
