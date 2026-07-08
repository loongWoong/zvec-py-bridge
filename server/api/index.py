"""Index management routes."""
from __future__ import annotations

from fastapi import APIRouter

from model.dto import CreateIndexDTO, OptimizeDTO
from service import index_service

router = APIRouter(prefix="/collections", tags=["index"])


@router.post("/{name}/indexes/{field_name}", summary="Create an index on a field")
def create_index(name: str, field_name: str, dto: CreateIndexDTO):
    return index_service.create_index(name, field_name, dto)


@router.delete("/{name}/indexes/{field_name}", summary="Drop an index")
def drop_index(name: str, field_name: str):
    return index_service.drop_index(name, field_name)


@router.post("/{name}:optimize", summary="Optimize a collection")
def optimize(name: str, dto: OptimizeDTO):
    return index_service.optimize(name, dto)
