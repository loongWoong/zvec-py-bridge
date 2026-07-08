"""FastAPI application entrypoint for the zvec REST bridge.

Run with::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

The zvec engine is initialised exactly once during startup (zvec.init raises
if called twice), and the shared :class:`core.manager.ZvecManager` is used by
every service.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import zvec
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import (
    admin_router,
    collection_router,
    document_router,
    embedding_router,
    index_router,
    query_router,
)
from config import settings
from core.errors import ZvecBridgeError, zvec_bridge_exception_handler, unhandled_exception_handler
from model.dto import JiebaDictDTO
from service import admin_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Initialise the zvec engine once.
    init_kwargs: dict = {
        "log_level": _log_level(settings.log_level),
        "log_type": _log_type(settings.log_type),
        "log_dir": settings.log_dir,
    }
    if settings.query_threads is not None:
        init_kwargs["query_threads"] = settings.query_threads
    if settings.optimize_threads is not None:
        init_kwargs["optimize_threads"] = settings.optimize_threads
    if settings.memory_limit_mb is not None:
        init_kwargs["memory_limit_mb"] = settings.memory_limit_mb
    if settings.log_basename is not None:
        init_kwargs["log_basename"] = settings.log_basename
    if settings.log_file_size is not None:
        init_kwargs["log_file_size"] = settings.log_file_size
    if settings.log_overdue_days is not None:
        init_kwargs["log_overdue_days"] = settings.log_overdue_days
    if settings.invert_to_forward_scan_ratio is not None:
        init_kwargs["invert_to_forward_scan_ratio"] = settings.invert_to_forward_scan_ratio
    if settings.brute_force_by_keys_ratio is not None:
        init_kwargs["brute_force_by_keys_ratio"] = settings.brute_force_by_keys_ratio
    if settings.fts_brute_force_by_keys_ratio is not None:
        init_kwargs["fts_brute_force_by_keys_ratio"] = settings.fts_brute_force_by_keys_ratio
    if settings.jieba_dict_dir is not None:
        init_kwargs["jieba_dict_dir"] = settings.jieba_dict_dir
    try:
        zvec.init(**init_kwargs)
    except RuntimeError:
        # already initialised (e.g. under --reload); safe to continue
        pass
    yield


def _log_level(name: str):
    try:
        return zvec.LogLevel[name.upper()]
    except KeyError:
        return zvec.LogLevel.WARN


def _log_type(name: str):
    try:
        return zvec.LogType[name.upper()]
    except KeyError:
        return zvec.LogType.CONSOLE


app = FastAPI(
    title="Zvec REST Bridge",
    description="A REST façade exposing the full zvec 0.5.x vector-database SDK.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(ZvecBridgeError, zvec_bridge_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(collection_router)
app.include_router(document_router)
app.include_router(query_router)
app.include_router(index_router)
app.include_router(admin_router)
app.include_router(embedding_router)


@app.get("/health", tags=["health"], summary="Health check")
def health():
    return {"status": "UP", "zvec_version": zvec.__version__}


@app.get("/engine", tags=["admin"], summary="Engine + bridge info")
def engine_info():
    return admin_service.engine_info()


@app.get("/admin/jieba-dict", tags=["admin"], summary="Get the jieba FTS tokenizer dict dir")
def get_jieba_dict():
    return admin_service.get_jieba_dict_dir()


@app.put("/admin/jieba-dict", tags=["admin"], summary="Set the jieba FTS tokenizer dict dir")
def set_jieba_dict(dto: JiebaDictDTO):
    return admin_service.set_jieba_dict_dir(dto.dir)


@app.post("/admin/diskann:load", tags=["admin"], summary="Load the DiskANN plugin")
def load_diskann():
    return admin_service.load_diskann_plugin()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
