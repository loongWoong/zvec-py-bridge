"""Admin routes: stats / flush / engine info."""
from __future__ import annotations

from fastapi import APIRouter

from service import admin_service

router = APIRouter(prefix="/collections", tags=["admin"])


@router.get("/{name}/stats", summary="Collection statistics")
def collection_stats(name: str):
    return admin_service.stats(name)


@router.post("/{name}:flush", summary="Flush pending writes to disk")
def flush(name: str):
    return admin_service.flush(name)
