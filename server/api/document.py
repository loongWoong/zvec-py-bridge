"""Document DML routes."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from model.dto import DeleteByFilterDTO, DocumentBatchDTO, FetchDTO
from service import document_service

router = APIRouter(prefix="/collections", tags=["documents"])


class DeleteIdsDTO(BaseModel):
    ids: list[str]


@router.post("/{name}/documents", summary="Insert documents")
def insert_documents(name: str, dto: DocumentBatchDTO):
    return document_service.insert(name, dto)


@router.put("/{name}/documents", summary="Upsert documents")
def upsert_documents(name: str, dto: DocumentBatchDTO):
    return document_service.upsert(name, dto)


@router.patch("/{name}/documents", summary="Update documents")
def update_documents(name: str, dto: DocumentBatchDTO):
    return document_service.update(name, dto)


@router.delete("/{name}/documents", summary="Delete documents by id")
def delete_documents(name: str, dto: DeleteIdsDTO):
    return document_service.delete(name, dto.ids)


@router.post("/{name}/documents:deleteByFilter", summary="Delete documents by filter")
def delete_by_filter(name: str, dto: DeleteByFilterDTO):
    return document_service.delete_by_filter(name, dto)


@router.post("/{name}/documents:fetch", summary="Fetch documents by id")
def fetch_documents(name: str, dto: FetchDTO):
    return document_service.fetch(name, dto)
