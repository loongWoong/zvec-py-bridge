"""Semantic Wiki Runtime 配置。

所有值可通过环境变量覆盖，与 demo/kb_data.py 保持同一风格。
"""
from __future__ import annotations

import os


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


# ====================================================================== #
#  SurrealDB（嵌入式，无需单独起 server）
#  支持 "memory"（纯内存）、"file://wiki_data.db"（文件持久化）
# ====================================================================== #
SURREAL_DB = os.environ.get("SURREAL_DB", "file://wiki_data.db")
SURREAL_NS = os.environ.get("SURREAL_NS", "wiki")
SURREAL_DB_NAME = os.environ.get("SURREAL_DB_NAME", "wiki")
SURREAL_USER = os.environ.get("SURREAL_USER", "root")
SURREAL_PASS = os.environ.get("SURREAL_PASS", "root")

# ====================================================================== #
#  zvec REST Bridge（向量语义检索）
# ====================================================================== #
ZVEC_URL = os.environ.get("ZVEC_URL", "http://localhost:8666")

# ====================================================================== #
#  Ollama（嵌入模型）
# ====================================================================== #
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")

# zvec 集合与嵌入函数名
COLLECTION_NAME = "wiki_chunks"
EMBED_FUNC_NAME = "wiki_ollama"
VECTOR_FIELD = "embedding"

# ====================================================================== #
#  生成式大模型（OpenAI 兼容 / Ollama 原生，复用 demo/kb_data.py 约定）
# ====================================================================== #
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8000")
LLM_API = os.environ.get("LLM_API", "openai")          # "openai" | "ollama"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-123")
LLM_MODEL = os.environ.get("LLM_MODEL", "hy3")

# ====================================================================== #
#  Web 服务
# ====================================================================== #
WIKI_HOST = os.environ.get("WIKI_HOST", "0.0.0.0")
WIKI_PORT = _env_int("WIKI_PORT", 8090)

# 文档分块大小（按句号切分，每块最多 N 个句子）
CHUNK_SENTENCES = 3
