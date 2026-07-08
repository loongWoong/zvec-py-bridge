"""Index management: create_index / drop_index / optimize."""
from __future__ import annotations

from typing import Any

from core.errors import ZvecRuntimeError
from core.manager import ZvecManager, manager
from model.dto import CreateIndexDTO, OptimizeDTO
from model.mapper import build_index_param, index_param_to_dict


class IndexService:
    def __init__(self, mgr: ZvecManager = manager) -> None:
        self._mgr = mgr

    def create_index(self, name: str, field_name: str, dto: CreateIndexDTO) -> dict[str, Any]:
        collection = self._mgr.get(name)
        index_param = build_index_param(dto.index_param)
        try:
            import zvec

            collection.create_index(
                field_name=field_name,
                index_param=index_param,
                option=zvec.IndexOption(concurrency=dto.concurrency),
            )
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"create_index failed: {exc}") from exc
        return {
            "name": name,
            "field": field_name,
            "index_param": index_param_to_dict(index_param),
            "status": "created",
        }

    def drop_index(self, name: str, field_name: str) -> dict[str, Any]:
        collection = self._mgr.get(name)
        try:
            collection.drop_index(field_name)
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"drop_index failed: {exc}") from exc
        return {"name": name, "field": field_name, "status": "dropped"}

    def optimize(self, name: str, dto: OptimizeDTO) -> dict[str, Any]:
        collection = self._mgr.get(name)
        try:
            import zvec

            collection.optimize(option=zvec.OptimizeOption(concurrency=dto.concurrency))
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"optimize failed: {exc}") from exc
        return {"name": name, "status": "optimized"}
