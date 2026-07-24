"""Bidirectional translation between JSON DTOs and zvec typed objects.

The REST layer only ever sees plain JSON. zvec, by contrast, works with
strongly-typed enums (``DataType``, ``MetricType``, ...) and a family of
param subclasses selected by index type. This module is the single place that
knows how to bridge the two worlds, so the service/api layers stay free of
zvec-specific branching.

Every ``build_*`` function takes a DTO and returns the matching zvec object;
every ``to_*`` function serialises a zvec object back into JSON-safe data.
"""
from __future__ import annotations

from typing import Any

import zvec
from zvec.extension.bm25_embedding_function import BM25EmbeddingFunction
from zvec.extension.embedding_function import DenseEmbeddingFunction, SparseEmbeddingFunction
from zvec.extension.http_embedding_function import HTTPDenseEmbedding
from zvec.extension.jina_embedding_function import JinaDenseEmbedding
from zvec.extension.multi_vector_reranker import RrfReRanker, WeightedReRanker
from zvec.extension.openai_embedding_function import OpenAIDenseEmbedding
from zvec.extension.qwen_embedding_function import QwenDenseEmbedding, QwenSparseEmbedding
from zvec.extension.qwen_rerank_function import QwenReRanker
from zvec.extension.sentence_transformer_embedding_function import (
    DefaultLocalDenseEmbedding,
    DefaultLocalSparseEmbedding,
)
from zvec.extension.sentence_transformer_rerank_function import DefaultLocalReRanker

from core.errors import InvalidArgumentError
from model.dto import (
    DocumentDTO,
    EmbeddingConfigDTO,
    FtsDTO,
    FieldSchemaDTO,
    IndexParamDTO,
    QueryDTO,
    QueryParamDTO,
    RerankerDTO,
    VectorSchemaDTO,
)

# ---------------------------------------------------------------------- #
# enum helpers
# ---------------------------------------------------------------------- #
def _data_type(name: str) -> zvec.DataType:
    if not name:
        raise InvalidArgumentError("data_type must not be empty")
    key = name.upper()
    # zvec enums are pybind11 types (not stdlib Enum), so they are not
    # subscriptable; resolve members via getattr instead.
    member = getattr(zvec.DataType, key, None)
    if member is None:
        raise InvalidArgumentError(
            f"unknown data_type {name!r}; valid: {[d.name for d in zvec.DataType]}"
        )
    return member


def _metric_type(name: str | None) -> zvec.MetricType | None:
    if name is None:
        return None
    member = getattr(zvec.MetricType, name.upper(), None)
    if member is None:
        raise InvalidArgumentError(
            f"unknown metric_type {name!r}; valid: {[m.name for m in zvec.MetricType]}"
        )
    return member


def _quantize_type(name: str | None) -> zvec.QuantizeType | None:
    if name is None:
        return None
    member = getattr(zvec.QuantizeType, name.upper(), None)
    if member is None:
        raise InvalidArgumentError(
            f"unknown quantize_type {name!r}; valid: {[q.name for q in zvec.QuantizeType]}"
        )
    return member


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------- #
# index params
# ---------------------------------------------------------------------- #
def build_index_param(dto: IndexParamDTO) -> Any:
    """Map an :class:`IndexParamDTO` to the matching zvec index param object."""
    t = (dto.type or "").upper()
    if t == "HNSW":
        return zvec.HnswIndexParam(**_drop_none({
            "metric_type": _metric_type(dto.metric_type),
            "m": dto.m,
            "ef_construction": dto.ef_construction,
            "quantize_type": _quantize_type(dto.quantize_type),
            "use_contiguous_memory": dto.use_contiguous_memory,
        }))
    if t == "HNSW_RABITQ":
        return zvec.HnswRabitqIndexParam(**_drop_none({
            "metric_type": _metric_type(dto.metric_type),
            "total_bits": dto.total_bits,
            "num_clusters": dto.num_clusters,
            "m": dto.m,
            "ef_construction": dto.ef_construction,
            "sample_count": dto.sample_count,
        }))
    if t == "IVF":
        return zvec.IVFIndexParam(**_drop_none({
            "metric_type": _metric_type(dto.metric_type),
            "n_list": dto.n_list,
            "n_iters": dto.n_iters,
            "use_soar": dto.use_soar,
            "quantize_type": _quantize_type(dto.quantize_type),
        }))
    if t == "FLAT":
        return zvec.FlatIndexParam(**_drop_none({
            "metric_type": _metric_type(dto.metric_type),
            "quantize_type": _quantize_type(dto.quantize_type),
        }))
    if t == "INVERT":
        return zvec.InvertIndexParam(**_drop_none({
            "enable_range_optimization": dto.enable_range_optimization,
            "enable_extended_wildcard": dto.enable_extended_wildcard,
        }))
    if t == "FTS":
        return zvec.FtsIndexParam(**_drop_none({
            "tokenizer_name": dto.tokenizer_name,
            "filters": dto.filters,
            "extra_params": dto.extra_params,
        }))
    if t == "VAMANA":
        return zvec.VamanaIndexParam(**_drop_none({
            "metric_type": _metric_type(dto.metric_type),
            "max_degree": dto.max_degree,
            "search_list_size": dto.search_list_size,
            "alpha": dto.alpha,
            "saturate_graph": dto.saturate_graph,
            "use_contiguous_memory": dto.use_contiguous_memory,
            "use_id_map": dto.use_id_map,
            "quantize_type": _quantize_type(dto.quantize_type),
        }))
    if t == "DISKANN":
        return zvec.DiskAnnIndexParam(**_drop_none({
            "metric_type": _metric_type(dto.metric_type),
            "max_degree": dto.max_degree,
            "list_size": dto.list_size,
            "pq_chunk_num": dto.pq_chunk_num,
            "quantize_type": _quantize_type(dto.quantize_type),
        }))
    raise InvalidArgumentError(
        f"unknown index type {dto.type!r}; valid: HNSW, HNSW_RABITQ, IVF, FLAT, "
        "INVERT, FTS, VAMANA, DISKANN"
    )


def index_param_to_dict(param: Any) -> dict[str, Any] | None:
    """Serialise a zvec index param back to a JSON-safe dict."""
    if param is None:
        return None
    if hasattr(param, "to_dict"):
        try:
            return param.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return {"type": getattr(param, "type", None)}


# ---------------------------------------------------------------------- #
# schema
# ---------------------------------------------------------------------- #
def build_field_schema(dto: FieldSchemaDTO) -> zvec.FieldSchema:
    index_param = build_index_param(dto.index_param) if dto.index_param else None
    return zvec.FieldSchema(
        name=dto.name,
        data_type=_data_type(dto.data_type),
        nullable=dto.nullable,
        index_param=index_param,
    )


def build_vector_schema(dto: VectorSchemaDTO) -> zvec.VectorSchema:
    index_param = build_index_param(dto.index_param) if dto.index_param else None
    return zvec.VectorSchema(
        name=dto.name,
        data_type=_data_type(dto.data_type),
        dimension=dto.dimension,
        index_param=index_param,
    )


def field_schema_to_dict(field: zvec.FieldSchema) -> dict[str, Any]:
    return {
        "name": field.name,
        "data_type": field.data_type.name,
        "nullable": field.nullable,
        "index_param": index_param_to_dict(field.index_param),
    }


def vector_schema_to_dict(vector: zvec.VectorSchema) -> dict[str, Any]:
    return {
        "name": vector.name,
        "data_type": vector.data_type.name,
        "dimension": vector.dimension,
        "index_param": index_param_to_dict(vector.index_param),
    }


def schema_to_dict(schema: zvec.CollectionSchema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "fields": [field_schema_to_dict(f) for f in schema.fields],
        "vectors": [vector_schema_to_dict(v) for v in schema.vectors],
    }


# ---------------------------------------------------------------------- #
# documents
# ---------------------------------------------------------------------- #
def build_doc(dto: DocumentDTO) -> zvec.Doc:
    vectors = None
    if dto.vectors:
        vectors = {name: _normalise_vector(v) for name, v in dto.vectors.items()}
    # `score` is an *output* concept (assigned by the engine on search), so it
    # is intentionally not forwarded on insert/upsert/update.
    return zvec.Doc(
        id=dto.id,
        vectors=vectors or None,
        fields=dto.fields or None,
    )


def _normalise_vector(v: Any) -> Any:
    """Coerce a JSON-decoded vector into the form zvec expects.

    Dense vectors arrive as lists and need no change. Sparse vectors arrive as
    JSON objects whose keys are *strings* (JSON has no integer keys), but zvec
    requires ``uint32`` keys, so we cast them here.
    """
    if isinstance(v, dict):
        try:
            return {int(k): float(val) for k, val in v.items()}
        except (ValueError, TypeError) as exc:
            raise InvalidArgumentError(
                f"sparse vector keys must be integers: {exc}"
            ) from exc
    return v


def doc_to_dict(doc: zvec.Doc, *, include_vector: bool = True) -> dict[str, Any]:
    vectors = None
    if include_vector and doc.vectors:
        vectors = {k: _to_jsonable(v) for k, v in doc.vectors.items()}
    return {
        "id": doc.id,
        "score": doc.score,
        "fields": dict(doc.fields) if doc.fields else {},
        "vectors": vectors,
    }


def _to_jsonable(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


# ---------------------------------------------------------------------- #
# query params + queries
# ---------------------------------------------------------------------- #
def build_query_param(dto: QueryParamDTO) -> Any:
    t = (dto.type or "").upper()
    extra: dict[str, int] = {}
    if dto.prefetch_offset is not None:
        extra["prefetch_offset"] = dto.prefetch_offset
    if dto.prefetch_lines is not None:
        extra["prefetch_lines"] = dto.prefetch_lines
    if t == "HNSW":
        return zvec.HnswQueryParam(**_drop_none({
            "ef": dto.ef,
            "radius": dto.radius,
            "is_linear": dto.is_linear,
            "is_using_refiner": dto.is_using_refiner,
            "extra_params": extra or None,
        }))
    if t == "HNSW_RABITQ":
        return zvec.HnswRabitqQueryParam(**_drop_none({
            "ef": dto.ef,
            "radius": dto.radius,
            "is_linear": dto.is_linear,
            "is_using_refiner": dto.is_using_refiner,
        }))
    if t == "IVF":
        return zvec.IVFQueryParam(**_drop_none({"nprobe": dto.nprobe}))
    if t == "FTS":
        return zvec.FtsQueryParam(**_drop_none({"default_operator": dto.default_operator}))
    if t == "VAMANA":
        return zvec.VamanaQueryParam(**_drop_none({
            "ef_search": dto.ef_search,
            "radius": dto.radius,
            "is_linear": dto.is_linear,
            "is_using_refiner": dto.is_using_refiner,
            "extra_params": extra or None,
        }))
    if t == "DISKANN":
        return zvec.DiskAnnQueryParam(**_drop_none({
            "list_size": dto.list_size,
            "radius": dto.radius,
            "is_linear": dto.is_linear,
            "is_using_refiner": dto.is_using_refiner,
        }))
    if t == "":
        return None
    raise InvalidArgumentError(
        f"unknown query param type {dto.type!r}; valid: HNSW, HNSW_RABITQ, IVF, FTS, VAMANA, DISKANN"
    )


def build_fts(dto: FtsDTO) -> zvec.Fts | None:
    if dto is None:
        return None
    if not (dto.query_string or dto.match_string):
        return None
    return zvec.Fts(query_string=dto.query_string, match_string=dto.match_string)


def build_query(dto: QueryDTO) -> zvec.Query:
    param = build_query_param(dto.param) if dto.param else None
    fts = build_fts(dto.fts)
    vector = _normalise_vector(dto.vector) if dto.vector is not None else None
    return zvec.Query(
        field_name=dto.field_name,
        id=dto.id,
        vector=vector,
        param=param,
        fts=fts,
    )


def build_reranker(dto: RerankerDTO | None) -> Any:
    if dto is None:
        return None
    t = (dto.type or "").lower()
    if t == "rrf":
        return RrfReRanker(rank_constant=dto.rank_constant if dto.rank_constant is not None else 60)
    if t == "weighted":
        if not dto.weights:
            raise InvalidArgumentError("weighted reranker requires non-empty 'weights'")
        return WeightedReRanker(weights=dto.weights)
    if t == "local_model":
        _require_rerank_fields(dto, "local_model")
        return DefaultLocalReRanker(**_drop_none({
            "query": dto.query,
            "rerank_field": dto.rerank_field,
            "model_name": dto.model_name,
            "model_source": dto.model_source,
            "device": dto.device,
            "batch_size": dto.batch_size,
        }))
    if t == "qwen_model":
        _require_rerank_fields(dto, "qwen_model")
        return QwenReRanker(**_drop_none({
            "query": dto.query,
            "rerank_field": dto.rerank_field,
            "model": dto.model,
            "api_key": dto.api_key,
        }))
    raise InvalidArgumentError(
        f"unknown reranker type {dto.type!r}; valid: rrf, weighted, local_model, qwen_model"
    )


def _require_rerank_fields(dto: RerankerDTO, kind: str) -> None:
    if not dto.query:
        raise InvalidArgumentError(f"{kind} reranker requires 'query'")
    if not dto.rerank_field:
        raise InvalidArgumentError(f"{kind} reranker requires 'rerank_field'")


# ---------------------------------------------------------------------- #
# status / stats
# ---------------------------------------------------------------------- #
def status_to_dict(status: zvec.Status) -> dict[str, Any]:
    # zvec Status exposes ok/code/message as *methods*, not properties.
    code = status.code()
    return {
        "ok": bool(status.ok()),
        "code": code.name if hasattr(code, "name") else str(code),
        "message": status.message(),
    }


def stats_to_dict(stats: zvec.CollectionStats) -> dict[str, Any]:
    return {
        "doc_count": stats.doc_count,
        "index_completeness": _to_jsonable(stats.index_completeness),
    }


# ---------------------------------------------------------------------- #
# embedding functions
# ---------------------------------------------------------------------- #
# Map our wire ``type`` to the constructor + the keyword args it accepts.
_EMBEDDING_SPECS: dict[str, tuple[type, set[str]]] = {
    "bm25": (BM25EmbeddingFunction, {"corpus", "encoding_type", "language", "b", "k1"}),
    "default_local_dense": (
        DefaultLocalDenseEmbedding,
        {"model_source", "device", "normalize_embeddings", "batch_size"},
    ),
    "default_local_sparse": (
        DefaultLocalSparseEmbedding,
        {"model_source", "device", "encoding_type"},
    ),
    "openai": (OpenAIDenseEmbedding, {"model", "dimension", "api_key", "base_url"}),
    "qwen_dense": (QwenDenseEmbedding, {"dimension", "model", "api_key"}),
    "qwen_sparse": (QwenSparseEmbedding, {"dimension", "model", "api_key"}),
    "jina": (JinaDenseEmbedding, {"model", "dimension", "api_key", "task"}),
    "http": (HTTPDenseEmbedding, {"base_url", "model", "api_key", "timeout"}),
}


def build_embedding_function(dto: EmbeddingConfigDTO) -> Any:
    """Instantiate a zvec embedding function from a wire config.

    Only the keyword args declared for the chosen ``type`` are forwarded, so a
    client can safely send a superset of fields. Heavy optional dependencies
    (torch, openai, dashtext, ...) are imported lazily by zvec; a missing dep
    surfaces as an :class:`ImportError` wrapped in :class:`InvalidArgumentError`.
    """
    spec = _EMBEDDING_SPECS.get((dto.type or "").lower())
    if spec is None:
        raise InvalidArgumentError(
            f"unknown embedding type {dto.type!r}; valid: "
            f"{', '.join(sorted(_EMBEDDING_SPECS))}"
        )
    ctor, allowed = spec
    kwargs = {k: getattr(dto, k) for k in allowed if getattr(dto, k) is not None}
    try:
        return ctor(**kwargs)
    except ImportError as exc:
        raise InvalidArgumentError(
            f"embedding type {dto.type!r} requires a missing dependency: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise InvalidArgumentError(f"failed to build embedding {dto.name!r}: {exc}") from exc


def embedding_to_dict(name: str, dto: EmbeddingConfigDTO) -> dict[str, Any]:
    """Serialise a registered embedding config (without secrets) for listing."""
    return {
        "name": name,
        "type": dto.type,
        # mask api_key so secrets never leak through the list endpoint
        "has_api_key": dto.api_key is not None,
        "model": dto.model,
        "dimension": dto.dimension,
        "encoding_type": dto.encoding_type,
        "language": dto.language,
        "base_url": dto.base_url,
    }


def embed_to_jsonable(vec: Any) -> Any:
    """Make a single embed() result JSON-safe (ndarray/dict -> list/dict)."""
    return _to_jsonable(vec)
