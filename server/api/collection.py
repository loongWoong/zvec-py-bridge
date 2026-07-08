"""Collection lifecycle + schema DDL routes."""
from __future__ import annotations

from fastapi import APIRouter

from model.dto import AddColumnDTO, AlterColumnDTO, CreateCollectionDTO, OpenCollectionDTO
from service import collection_service

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", summary="Create a collection")
def create_collection(dto: CreateCollectionDTO):
    """Create a new collection. The collection name in the JSON body schema
    is used as the collection identity."""
    return collection_service.create(dto.schema_.name, dto)


@router.get("", summary="List all collections")
def list_collections():
    return collection_service.list()


@router.post("/{name}/open", summary="Open an existing collection")
def open_collection(name: str, dto: OpenCollectionDTO):
    return collection_service.open(name, dto)


@router.post("/{name}/close", summary="Close (unload) a collection")
def close_collection(name: str):
    return collection_service.close(name)


@router.delete("/{name}", summary="Permanently destroy a collection")
def destroy_collection(name: str):
    return collection_service.destroy(name)


@router.get("/{name}", summary="Collection info + schema")
def collection_info(name: str):
    return collection_service.info(name)


# ---------------------------------------------------------------------- #
# column DDL
# ---------------------------------------------------------------------- #
@router.post("/{name}/columns", summary="Add a column")
def add_column(name: str, dto: AddColumnDTO):
    return collection_service.add_column(name, dto)


@router.put("/{name}/columns/{old_name}", summary="Alter/rename a column")
def alter_column(name: str, old_name: str, dto: AlterColumnDTO):
    return collection_service.alter_column(name, old_name, dto)


@router.delete("/{name}/columns/{field_name}", summary="Drop a column")
def drop_column(name: str, field_name: str):
    return collection_service.drop_column(name, field_name)
