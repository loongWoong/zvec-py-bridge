"""Admin operations: stats / flush / destroy / engine info."""
from __future__ import annotations

from typing import Any

from core.errors import ZvecRuntimeError
from core.manager import ZvecManager, manager
from model.mapper import schema_to_dict, stats_to_dict


class AdminService:
    def __init__(self, mgr: ZvecManager = manager) -> None:
        self._mgr = mgr

    def stats(self, name: str) -> dict[str, Any]:
        collection = self._mgr.get(name)
        return {
            "name": name,
            "path": collection.path,
            "stats": stats_to_dict(collection.stats),
            "schema": schema_to_dict(collection.schema),
        }

    def flush(self, name: str) -> dict[str, Any]:
        collection = self._mgr.get(name)
        try:
            collection.flush()
        except Exception as exc:  # noqa: BLE001
            raise ZvecRuntimeError(f"flush failed: {exc}") from exc
        return {"name": name, "status": "flushed"}

    def engine_info(self) -> dict[str, Any]:
        import zvec

        info: dict[str, Any] = {
            "zvec_version": zvec.__version__,
            "data_dir": self._mgr.base_path,
            "auto_open": self._mgr.auto_open,
        }
        # diskann plugin status (optional, may be unsupported on some platforms)
        try:
            info["diskann_plugin_loaded"] = zvec.is_diskann_plugin_loaded()
        except Exception:  # noqa: BLE001
            info["diskann_plugin_loaded"] = None
        try:
            info["libaio_available"] = zvec.is_libaio_available()
        except Exception:  # noqa: BLE001
            info["libaio_available"] = None
        try:
            info["jieba_dict_dir"] = zvec.get_default_jieba_dict_dir()
        except Exception:  # noqa: BLE001
            info["jieba_dict_dir"] = None
        return info

    # ------------------------------------------------------------------ #
    # jieba / diskann plugin management
    # ------------------------------------------------------------------ #
    def set_jieba_dict_dir(self, path: str) -> dict[str, Any]:
        import zvec

        zvec.set_default_jieba_dict_dir(path)
        return {"jieba_dict_dir": zvec.get_default_jieba_dict_dir()}

    def get_jieba_dict_dir(self) -> dict[str, Any]:
        import zvec

        return {"jieba_dict_dir": zvec.get_default_jieba_dict_dir()}

    def load_diskann_plugin(self) -> dict[str, Any]:
        import zvec

        try:
            zvec.load_diskann_plugin()
            return {"diskann_plugin_loaded": True, "status": "loaded"}
        except Exception as exc:  # noqa: BLE001
            return {
                "diskann_plugin_loaded": zvec.is_diskann_plugin_loaded(),
                "status": "failed",
                "error": str(exc),
            }
