"""SurrealDB 连接与 Schema 初始化。

使用嵌入式模式（file:// 或 memory），无需单独起 server 进程。
Schema 对应 design.md 第十节：document / raw / topic / tag / entity / version
及图关系边表。
"""
from __future__ import annotations

from surrealdb import Surreal

import config

# 全局连接（由 init() 创建，get_db() 复用）
_db: Surreal | None = None

# ====================================================================== #
#  Schema 定义（SurrealQL）
#  对应 design.md 第十节。SCHEMALESS 允许灵活扩展字段。
# ====================================================================== #
SCHEMA_SQL = """
-- ─────────── 对象表（节点） ───────────

DEFINE TABLE document SCHEMALESS;
DEFINE FIELD title    ON document TYPE string;
DEFINE FIELD summary  ON document TYPE option<string>;
DEFINE FIELD content  ON document TYPE string;
DEFINE FIELD topic_id ON document TYPE option<string>;
DEFINE FIELD author   ON document TYPE option<string>;
DEFINE FIELD status   ON document TYPE string DEFAULT 'active';
DEFINE FIELD version  ON document TYPE int DEFAULT 1;
DEFINE FIELD created  ON document TYPE datetime DEFAULT time::now();
DEFINE FIELD updated  ON document TYPE datetime DEFAULT time::now();

DEFINE TABLE raw SCHEMALESS;
DEFINE FIELD url       ON raw TYPE option<string>;
DEFINE FIELD author    ON raw TYPE option<string>;
DEFINE FIELD published ON raw TYPE option<datetime>;
DEFINE FIELD content   ON raw TYPE string;
DEFINE FIELD collected  ON raw TYPE datetime DEFAULT time::now();

DEFINE TABLE topic SCHEMALESS;
DEFINE FIELD name        ON topic TYPE string;
DEFINE FIELD description ON topic TYPE option<string>;

DEFINE TABLE tag SCHEMALESS;
DEFINE FIELD name      ON tag TYPE string;
DEFINE FIELD parent_id ON tag TYPE option<string>;

DEFINE TABLE entity SCHEMALESS;
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD type ON entity TYPE option<string>;

DEFINE TABLE version SCHEMALESS;
DEFINE FIELD doc_id    ON version TYPE string;
DEFINE FIELD title     ON version TYPE string;
DEFINE FIELD content   ON version TYPE string;
DEFINE FIELD summary   ON version TYPE option<string>;
DEFINE FIELD version   ON version TYPE int;
DEFINE FIELD snapshot  ON version TYPE datetime DEFAULT time::now();

-- ─────────── 图关系边表（TYPE RELATION） ───────────
-- document → topic
DEFINE TABLE belongs_to  TYPE RELATION FROM document TO topic SCHEMALESS;
-- document → tag
DEFINE TABLE has_tag     TYPE RELATION FROM document TO tag SCHEMALESS;
-- document → entity
DEFINE TABLE mentions   TYPE RELATION FROM document TO entity SCHEMALESS;
-- document → raw
DEFINE TABLE references TYPE RELATION FROM document TO raw SCHEMALESS;

-- document → document（多种语义关系，对应 design.md 第六节）
DEFINE TABLE related     TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE depends    TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE extends    TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE implements TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE contradicts TYPE RELATION FROM document TO document SCHEMALESS;
DEFINE TABLE supersedes TYPE RELATION FROM document TO document SCHEMALESS;

-- tag → tag（标签层级）
DEFINE TABLE child_of   TYPE RELATION FROM tag TO tag SCHEMALESS;
-- entity → entity（实体间关系）
DEFINE TABLE entity_related TYPE RELATION FROM entity TO entity SCHEMALESS;
-- version → version（版本链）
DEFINE TABLE previous_version TYPE RELATION FROM version TO version SCHEMALESS;
"""


def init() -> Surreal:
    """初始化 SurrealDB 连接并创建 Schema。幂等：重复调用安全。"""
    global _db
    if _db is not None:
        return _db

    db = Surreal(config.SURREAL_DB)
    db.use(config.SURREAL_NS, config.SURREAL_DB_NAME)
    try:
        db.signin({"username": config.SURREAL_USER, "password": config.SURREAL_PASS})
    except Exception:
        # 嵌入式 memory/file 模式可能无需认证，忽略错误
        pass

    # 执行 Schema（逐条执行，单条失败不中断整体——例如 ANALYZER 已存在）
    for stmt in SCHEMA_SQL.split(";"):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("--"):
            continue
        try:
            db.query(stmt + ";")
        except Exception as e:
            # 幂等：已存在的定义会报错，忽略即可
            _warn(f"Schema 语句跳过: {stmt[:60]}... → {e}")

    _db = db
    return _db


def get_db() -> Surreal:
    """返回已初始化的连接。若未初始化则自动 init()。"""
    if _db is None:
        return init()
    return _db


def close() -> None:
    """关闭连接。"""
    global _db
    if _db is not None:
        try:
            _db.close()
        except Exception:
            pass
        _db = None


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")
