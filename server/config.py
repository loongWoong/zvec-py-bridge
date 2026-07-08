"""Application configuration for the zvec REST bridge.

All values can be overridden through environment variables, which makes the
service container-friendly (12-factor style).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


@dataclass
class Settings:
    # Where collections are physically stored on disk.
    data_dir: str = field(default_factory=lambda: os.environ.get("ZVEC_DATA_DIR", "./data"))

    # Zvec engine init options (None => engine default).
    log_level: str = field(default_factory=lambda: os.environ.get("ZVEC_LOG_LEVEL", "WARN"))
    log_type: str = field(default_factory=lambda: os.environ.get("ZVEC_LOG_TYPE", "CONSOLE"))
    log_dir: str = field(default_factory=lambda: os.environ.get("ZVEC_LOG_DIR", "./logs"))
    query_threads: int | None = field(
        default_factory=lambda: (int(v) if (v := os.environ.get("ZVEC_QUERY_THREADS")) else None)
    )
    optimize_threads: int | None = field(
        default_factory=lambda: (int(v) if (v := os.environ.get("ZVEC_OPTIMIZE_THREADS")) else None)
    )
    memory_limit_mb: int | None = field(
        default_factory=lambda: (int(v) if (v := os.environ.get("ZVEC_MEMORY_LIMIT_MB")) else None)
    )
    # log rotation / tuning
    log_basename: str | None = field(
        default_factory=lambda: os.environ.get("ZVEC_LOG_BASENAME") or None
    )
    log_file_size: int | None = field(
        default_factory=lambda: (int(v) if (v := os.environ.get("ZVEC_LOG_FILE_SIZE")) else None)
    )
    log_overdue_days: int | None = field(
        default_factory=lambda: (int(v) if (v := os.environ.get("ZVEC_LOG_OVERDUE_DAYS")) else None)
    )
    # query heuristics (0.0-1.0)
    invert_to_forward_scan_ratio: float | None = field(
        default_factory=lambda: (float(v) if (v := os.environ.get("ZVEC_INVERT_TO_FORWARD_SCAN_RATIO")) else None)
    )
    brute_force_by_keys_ratio: float | None = field(
        default_factory=lambda: (float(v) if (v := os.environ.get("ZVEC_BRUTE_FORCE_BY_KEYS_RATIO")) else None)
    )
    fts_brute_force_by_keys_ratio: float | None = field(
        default_factory=lambda: (float(v) if (v := os.environ.get("ZVEC_FTS_BRUTE_FORCE_BY_KEYS_RATIO")) else None)
    )
    jieba_dict_dir: str | None = field(
        default_factory=lambda: os.environ.get("ZVEC_JIEBA_DICT_DIR") or None
    )

    # Whether collections are auto-opened when first referenced by an operation.
    auto_open: bool = field(default_factory=lambda: _env_bool("ZVEC_AUTO_OPEN", True))

    # HTTP server.
    host: str = field(default_factory=lambda: os.environ.get("ZVEC_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("ZVEC_PORT", 8666))


settings = Settings()
