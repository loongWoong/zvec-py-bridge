"""Custom exceptions and a uniform FastAPI exception handler.

Every error raised inside the service layer is translated into a structured
JSON error response here, so callers always get a consistent shape:

    {"error": {"code": "NOT_FOUND", "message": "..."}}
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ZvecBridgeError(Exception):
    """Base class for all bridge-level errors."""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(ZvecBridgeError):
    code = "NOT_FOUND"
    status_code = 404


class AlreadyExistsError(ZvecBridgeError):
    code = "ALREADY_EXISTS"
    status_code = 409


class InvalidArgumentError(ZvecBridgeError):
    code = "INVALID_ARGUMENT"
    status_code = 400


class CollectionNotOpenError(ZvecBridgeError):
    code = "COLLECTION_NOT_OPEN"
    status_code = 409


class ZvecRuntimeError(ZvecBridgeError):
    """Wraps an error that originated from the zvec engine itself."""

    code = "ENGINE_ERROR"
    status_code = 500


async def zvec_bridge_exception_handler(_request: Request, exc: ZvecBridgeError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Re-raise if it looks like an HTTPException so FastAPI's own handler runs.
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
    )
