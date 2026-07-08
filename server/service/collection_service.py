"""Collection lifecycle + schema DDL operations."""
from __future__ import annotations

from typing import Any

from core.errors import InvalidArgumentError, ZvecRuntimeError
from core.manager import ZvecManager, manager
from model.dto import (
    AddColumnDTO,
    AlterColumnDTO,
    CollectionSchemaDTO,
    CreateCollectionDTO,
    OpenCollectionDTO,
)
from model.mapper import (
    build_field_schema,
    build_vector_schema,
    schema_to_dict,
)


class CollectionService:
    def __init__(self, mgr: ZvecManager = manager) -> None:
        self._mgr = mgr

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def create(self, name: str, dto: CreateCollectionDTO) -> dict[str, Any]:
        schema_dto: CollectionSchemaDTO = dto.schema_
        if schema_dto.name != name:
            raise InvalidArgumentError(
                f"collection name in path ({name!r}) must match schema name "
                f"({schema_dto.name!r})"
            )
        if not schema_dto.vectors:
            raise InvalidArgumentError("a collection must define at least one vector field")
        fields = [build_field_schema(f) for f in (schema_dto.fields or [])]
        vectors = [build_vector_schema(v) for v in schema_dto.vectors]
        schema = _build_schema(schema_dto.name, fields, vectors)
        collection = self._mgr.create(name, schema)
        return {
            "name": name,
            "status": "created",
            "path": collection.path,
            "schema": schema_to_dict(collection.schema),
        }

    def open(self, name: str, dto: OpenCollectionDTO) -> dict[str, Any]:
        collection = self._mgr.open(name, read_only=dto.read_only)
        return {
            "name": name,
            "status": "opened",
            "path": collection.path,
            "schema": schema_to_dict(collection.schema),
        }

    def close(self, name: str) -> dict[str, Any]:
        self._mgr.close(name)
        return {"name": name, "status": "closed"}

    def destroy(self, name: str) -> dict[str, Any]:
        self._mgr.destroy(name)
        return {"name": name, "status": "destroyed"}

    def list(self) -> list[dict[str, Any]]:
        return self._mgr.list_collections()

    def info(self, name: str) -> dict[str, Any]:
        collection = self._mgr.get(name)
        return {
            "name": name,
            "opened": True,
            "path": collection.path,
            "read_only": collection.option.read_only,
            "enable_mmap": collection.option.enable_mmap,
            "schema": schema_to_dict(collection.schema),
        }

    # ------------------------------------------------------------------ #
    # column DDL
    # ------------------------------------------------------------------ #
    def add_column(self, name: str, dto: AddColumnDTO) -> dict[str, Any]:
        collection = self._mgr.get(name)
        field = build_field_schema(dto.field)
        try:
            collection.add_column(
                field_schema=field,
                expression=dto.expression,
                option=_add_column_option(dto.concurrency),
            )
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"add_column failed: {exc}") from exc
        return {"name": name, "column": dto.field.name, "status": "added"}

    def alter_column(self, name: str, old_name: str, dto: AlterColumnDTO) -> dict[str, Any]:
        collection = self._mgr.get(name)
        field = build_field_schema(dto.field) if dto.field else None
        try:
            collection.alter_column(
                old_name=old_name,
                new_name=dto.new_name,
                field_schema=field,
                option=_alter_column_option(dto.concurrency),
            )
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"alter_column failed: {exc}") from exc
        return {
            "name": name,
            "old_name": old_name,
            "new_name": dto.new_name,
            "status": "altered",
        }

    def drop_column(self, name: str, field_name: str) -> dict[str, Any]:
        collection = self._mgr.get(name)
        try:
            collection.drop_column(field_name)
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"drop_column failed: {exc}") from exc
        return {"name": name, "column": field_name, "status": "dropped"}


# ---------------------------------------------------------------------- #
# private helpers
# ---------------------------------------------------------------------- #
def _build_schema(name, fields, vectors):
    import zvec

    return zvec.CollectionSchema(name=name, fields=fields or None, vectors=vectors or None)


def _add_column_option(concurrency: int):
    import zvec

    return zvec.AddColumnOption(concurrency=concurrency)


def _alter_column_option(concurrency: int):
    import zvec

    return zvec.AlterColumnOption(concurrency=concurrency)
