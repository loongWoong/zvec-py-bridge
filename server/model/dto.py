"""Request/response data-transfer objects for the REST API.

These are intentionally thin pydantic models that mirror the JSON wire format.
Conversion between these DTOs and the strongly-typed zvec objects lives in
:mod:`model.mapper`.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ====================================================================== #
# Schema building blocks
# ====================================================================== #
class IndexParamDTO(BaseModel):
    """A discriminated index configuration. The ``type`` field selects the
    concrete index; remaining fields are forwarded to the matching zvec param
    constructor (unknown keys are ignored)."""

    type: str = Field(..., description="HNSW | HNSW_RABITQ | IVF | FLAT | INVERT | FTS | VAMANA | DISKANN")
    # HNSW / HNSW_RABITQ / IVF / FLAT / VAMANA
    metric_type: str | None = None
    quantize_type: str | None = None
    # HNSW / HNSW_RABITQ
    m: int | None = None
    ef_construction: int | None = None
    # HNSW_RABITQ
    total_bits: int | None = None
    num_clusters: int | None = None
    sample_count: int | None = None
    # HNSW
    use_contiguous_memory: bool | None = None
    # IVF
    n_list: int | None = None
    n_iters: int | None = None
    use_soar: bool | None = None
    # INVERT
    enable_range_optimization: bool | None = None
    enable_extended_wildcard: bool | None = None
    # FTS
    tokenizer_name: str | None = None
    filters: list[str] | None = None
    extra_params: str | None = None
    # VAMANA
    max_degree: int | None = None
    search_list_size: int | None = None
    alpha: float | None = None
    saturate_graph: bool | None = None
    use_id_map: bool | None = None
    # DISKANN
    list_size: int | None = None
    pq_chunk_num: int | None = None


class FieldSchemaDTO(BaseModel):
    name: str
    data_type: str
    nullable: bool = False
    index_param: IndexParamDTO | None = None


class VectorSchemaDTO(BaseModel):
    name: str
    data_type: str = "VECTOR_FP32"
    dimension: int = 0
    index_param: IndexParamDTO | None = None


class CollectionSchemaDTO(BaseModel):
    name: str
    fields: list[FieldSchemaDTO] | None = None
    vectors: list[VectorSchemaDTO] | None = None


# ====================================================================== #
# Collection lifecycle
# ====================================================================== #
class CreateCollectionDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: CollectionSchemaDTO = Field(alias="schema")
    read_only: bool = False


class OpenCollectionDTO(BaseModel):
    read_only: bool = False


# ====================================================================== #
# Column DDL
# ====================================================================== #
class AddColumnDTO(BaseModel):
    field: FieldSchemaDTO
    expression: str = ""
    concurrency: int = 0


class AlterColumnDTO(BaseModel):
    new_name: str | None = None
    field: FieldSchemaDTO | None = None
    concurrency: int = 0


# ====================================================================== #
# Document DML
# ====================================================================== #
class DocumentDTO(BaseModel):
    """A single document. ``id`` is the primary key. ``vectors`` maps vector
    field name -> list of floats (or sparse dict). ``fields`` maps scalar field
    name -> value. ``text`` is an alternative to ``vectors``: when the batch
    carries an ``embedding`` reference, the bridge embeds ``text`` into the
    named vector field automatically."""

    id: str
    score: float | None = None
    vectors: dict[str, Any] | None = None
    fields: dict[str, Any] | None = None
    text: str | None = None


class EmbeddingRefDTO(BaseModel):
    """References a registered embedding function and the vector field it should
    populate. Used by insert/search to auto-embed ``text``."""

    function: str
    field: str
    encoding_type: str | None = Field(
        None, description="query | document (sparse/BM25 only); auto-defaulted if omitted"
    )


class DocumentBatchDTO(BaseModel):
    documents: list[DocumentDTO]
    embedding: EmbeddingRefDTO | None = None


class FetchDTO(BaseModel):
    ids: list[str]
    output_fields: list[str] | None = None
    include_vector: bool = True


class DeleteByFilterDTO(BaseModel):
    filter: str


# ====================================================================== #
# Query
# ====================================================================== #
class FtsDTO(BaseModel):
    query_string: str | None = None
    match_string: str | None = None


class QueryParamDTO(BaseModel):
    """Discriminated query param. ``type`` selects HNSW | HNSW_RABITQ | IVF |
    FTS | VAMANA | DISKANN."""

    type: str | None = None
    ef: int | None = None
    ef_search: int | None = None
    nprobe: int | None = None
    radius: float | None = None
    is_linear: bool | None = None
    is_using_refiner: bool | None = None
    default_operator: str | None = None
    prefetch_offset: int | None = None
    prefetch_lines: int | None = None
    # DISKANN
    list_size: int | None = None


class QueryDTO(BaseModel):
    field_name: str
    id: str | None = None
    vector: list[float] | dict[str, float] | None = None
    text: str | None = None
    param: QueryParamDTO | None = None
    fts: FtsDTO | None = None


class RerankerDTO(BaseModel):
    """Reranker configuration.

    - ``rrf``      : reciprocal rank fusion (fusion)
    - ``weighted`` : weighted score fusion (needs ``weights``)
    - ``local_model``: DefaultLocalReRanker cross-encoder (needs ``query`` +
      ``rerank_field``)
    - ``qwen_model`` : QwenReRanker API reranker (needs ``query`` +
      ``rerank_field``)
    """

    type: str = Field("rrf", description="rrf | weighted | local_model | qwen_model")
    rank_constant: int | None = None
    weights: list[float] | None = None
    # model rerankers
    query: str | None = None
    rerank_field: str | None = None
    model_name: str | None = None
    model_source: str | None = None
    device: str | None = None
    batch_size: int | None = None
    model: str | None = None
    api_key: str | None = None


class SearchDTO(BaseModel):
    queries: list[QueryDTO]
    topk: int = 10
    filter: str | None = None
    include_vector: bool = False
    output_fields: list[str] | None = None
    reranker: RerankerDTO | None = None
    embedding: EmbeddingRefDTO | None = None


# ====================================================================== #
# Index / Admin
# ====================================================================== #
class CreateIndexDTO(BaseModel):
    index_param: IndexParamDTO
    concurrency: int = 0


class OptimizeDTO(BaseModel):
    concurrency: int = 0


# ====================================================================== #
# Embedding functions
# ====================================================================== #
class EmbeddingConfigDTO(BaseModel):
    """Configuration for registering a named embedding function.

    ``type`` selects the backend; only the relevant fields for that type are
    forwarded to the constructor."""

    name: str
    type: str = Field(
        ...,
        description="bm25 | default_local_dense | default_local_sparse | "
        "openai | qwen_dense | qwen_sparse | jina | http",
    )
    # BM25
    encoding_type: str | None = None
    language: str | None = None
    b: float | None = None
    k1: float | None = None
    corpus: list[str] | None = None
    # local (sentence-transformer)
    model_source: str | None = None
    device: str | None = None
    normalize_embeddings: bool | None = None
    batch_size: int | None = None
    # openai / qwen / jina / http
    model: str | None = None
    dimension: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    # jina
    task: str | None = None
    # http
    timeout: int | None = None


class EmbedTextDTO(BaseModel):
    texts: list[str]
    encoding_type: str | None = None


# ====================================================================== #
# Admin: jieba / diskann
# ====================================================================== #
class JiebaDictDTO(BaseModel):
    dir: str
