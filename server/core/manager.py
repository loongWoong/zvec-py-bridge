"""Thread-safe registry and lifecycle manager for zvec collections.

A single :class:`ZvecManager` instance owns every opened collection and
serialises create/open/close/destroy operations with a lock so the REST layer
can be called concurrently without corrupting the in-memory registry.

The manager only deals with zvec ``Collection`` objects; all schema/param/doc
translation happens in :mod:`model.mapper`.
"""
from __future__ import annotations

import os
import shutil
import threading
from typing import Any

import zvec

from core.errors import (
    AlreadyExistsError,
    InvalidArgumentError,
    NotFoundError,
    ZvecRuntimeError,
)


class ZvecManager:
    """Owns the lifecycle of every opened zvec collection."""

    def __init__(self, data_dir: str = "./data", auto_open: bool = True) -> None:
        self.base_path = os.path.abspath(data_dir)
        self.auto_open = auto_open
        self._collections: dict[str, zvec.Collection] = {}
        self._lock = threading.RLock()
        # Per-collection operation lock. The registry lock above only guards the
        # in-memory registry; the actual engine calls (insert/query/optimize/…)
        # run outside it, so we serialise them per collection to avoid racing a
        # close/destroy that is flushing or finalising a handle another request
        # is still using.
        self._op_locks: dict[str, threading.RLock] = {}
        self._op_locks_guard = threading.Lock()
        os.makedirs(self.base_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # per-collection operation lock
    # ------------------------------------------------------------------ #
    def lock_for(self, name: str) -> threading.RLock:
        """Return (creating if needed) the operation lock for a collection."""
        with self._op_locks_guard:
            lock = self._op_locks.get(name)
            if lock is None:
                lock = threading.RLock()
                self._op_locks[name] = lock
            return lock

    def _drop_op_lock(self, name: str) -> None:
        with self._op_locks_guard:
            self._op_locks.pop(name, None)

    # ------------------------------------------------------------------ #
    # path helpers
    # ------------------------------------------------------------------ #
    def _path(self, name: str) -> str:
        if not name:
            raise InvalidArgumentError("collection name must not be empty")
        # keep names filesystem-safe; reject path separators
        if os.sep in name or "/" in name or "\\" in name:
            raise InvalidArgumentError(
                f"collection name must not contain path separators: {name!r}"
            )
        return os.path.join(self.base_path, name)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def create(self, name: str, schema: zvec.CollectionSchema) -> zvec.Collection:
        """Create a brand-new collection on disk and register it."""
        with self._lock:
            if name in self._collections:
                raise AlreadyExistsError(f"collection {name!r} is already open")
            path = self._path(name)
            if os.path.exists(path):
                raise AlreadyExistsError(
                    f"a collection already exists on disk at {path!r}; "
                    "open it instead of creating"
                )
            try:
                collection = zvec.create_and_open(path=path, schema=schema)
            except Exception as exc:  # noqa: BLE001
                # clean up a half-created directory if possible
                shutil.rmtree(path, ignore_errors=True)
                if "already" in str(exc).lower():
                    raise AlreadyExistsError(str(exc)) from exc
                raise ZvecRuntimeError(f"failed to create collection {name!r}: {exc}") from exc
            self._collections[name] = collection
            return collection

    def open(self, name: str, *, read_only: bool = False) -> zvec.Collection:
        """Open an existing collection from disk and register it."""
        with self._lock:
            existing = self._collections.get(name)
            if existing is not None:
                return existing
            path = self._path(name)
            if not os.path.exists(path):
                raise NotFoundError(f"no collection found on disk at {path!r}")
            option = zvec.CollectionOption(read_only=read_only)
            try:
                collection = zvec.open(path, option)
            except Exception as exc:  # noqa: BLE001
                raise ZvecRuntimeError(f"failed to open collection {name!r}: {exc}") from exc
            self._collections[name] = collection
            return collection

    def close(self, name: str) -> None:
        """Drop a collection from the in-memory registry (data stays on disk)."""
        with self._lock:
            if name not in self._collections:
                raise NotFoundError(f"collection {name!r} is not open")
            collection = self._collections.pop(name)
        # Wait for any in-flight operation on this collection to finish (the op
        # lock is held by callers around their engine calls), then flush so
        # pending writes are durable before we let the handle go.
        with self.lock_for(name):
            try:
                collection.flush()
            except Exception:  # noqa: BLE001 - best effort
                pass
            # The C++ object is ref-counted by pybind; letting it go out of
            # scope here releases the open handle.
        self._drop_op_lock(name)

    def destroy(self, name: str) -> None:
        """Permanently delete a collection and its on-disk data."""
        with self._lock:
            collection = self._collections.pop(name, None)
            path = self._path(name)
        # Serialise against any in-flight operation on this collection.
        with self.lock_for(name):
            if collection is not None:
                try:
                    collection.destroy()
                except Exception as exc:  # noqa: BLE001
                    # fall back to removing the directory manually
                    shutil.rmtree(path, ignore_errors=True)
                    raise ZvecRuntimeError(
                        f"failed to destroy collection {name!r}: {exc}"
                    ) from exc
            else:
                if not os.path.exists(path):
                    raise NotFoundError(f"no collection found at {path!r}")
                shutil.rmtree(path, ignore_errors=True)
        self._drop_op_lock(name)

    # ------------------------------------------------------------------ #
    # access
    # ------------------------------------------------------------------ #
    def get(self, name: str) -> zvec.Collection:
        """Return an opened collection, auto-opening it when configured to."""
        with self._lock:
            collection = self._collections.get(name)
            if collection is not None:
                return collection
            if self.auto_open:
                return self.open(name)
            from core.errors import CollectionNotOpenError

            raise CollectionNotOpenError(
                f"collection {name!r} is not open and auto-open is disabled"
            )

    def get_opened(self, name: str) -> zvec.Collection | None:
        """Like :meth:`get` but never auto-opens; returns ``None`` if absent."""
        with self._lock:
            return self._collections.get(name)

    def list_collections(self) -> list[dict[str, Any]]:
        """List every collection known to the registry plus on-disk ones."""
        with self._lock:
            opened = sorted(self._collections.keys())
            on_disk: list[str] = []
            if os.path.isdir(self.base_path):
                for entry in os.listdir(self.base_path):
                    if os.path.isdir(os.path.join(self.base_path, entry)):
                        on_disk.append(entry)
            return [
                {
                    "name": n,
                    "opened": n in self._collections,
                    "path": self._path(n),
                }
                for n in sorted(set(opened) | set(on_disk))
            ]


# A process-wide singleton. ``main.py`` initialises zvec before this is used.
manager = ZvecManager(
    data_dir=__import__("config").settings.data_dir,
    auto_open=__import__("config").settings.auto_open,
)
