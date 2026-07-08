"""Core layer: lifecycle management and error handling."""
from core.embedding_manager import embedding_manager  # noqa: F401
from core.errors import (  # noqa: F401
    AlreadyExistsError,
    CollectionNotOpenError,
    InvalidArgumentError,
    NotFoundError,
    ZvecBridgeError,
    ZvecRuntimeError,
)
from core.manager import manager  # noqa: F401
