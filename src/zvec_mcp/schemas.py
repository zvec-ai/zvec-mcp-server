"""Pydantic input models for Zvec MCP server tools."""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zvec_mcp.types import (
    DataTypeEnum,
    MetricTypeEnum,
    QuantizeTypeEnum,
    ResponseFormat,
)

# ============================================================================
# Index Parameter Models
# ============================================================================


class HnswIndexParamInput(BaseModel):
    """Parameters for HNSW (Hierarchical Navigable Small World) index."""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["HNSW"] = "HNSW"
    metric_type: MetricTypeEnum = Field(
        default=MetricTypeEnum.IP,
        description="Distance metric: COSINE, IP, or L2 (default: IP)",
    )
    m: int = Field(
        default=50,
        ge=1,
        description="Number of bi-directional links per node. Higher = better accuracy, more memory (default: 50)",
    )
    ef_construction: int = Field(
        default=500,
        ge=1,
        description="Candidate list size during index construction. Higher = better quality, slower build (default: 500)",
    )
    quantize_type: QuantizeTypeEnum = Field(
        default=QuantizeTypeEnum.UNDEFINED,
        description="Vector quantization type for compression: UNDEFINED (none), FP16, INT8 (default: UNDEFINED)",
    )


class FlatIndexParamInput(BaseModel):
    """Parameters for Flat (brute-force exact search) index."""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["FLAT"] = "FLAT"
    metric_type: MetricTypeEnum = Field(
        default=MetricTypeEnum.IP,
        description="Distance metric: COSINE, IP, or L2 (default: IP)",
    )
    quantize_type: QuantizeTypeEnum = Field(
        default=QuantizeTypeEnum.UNDEFINED,
        description="Vector quantization type for compression: UNDEFINED (none), FP16, INT8 (default: UNDEFINED)",
    )


class IVFIndexParamInput(BaseModel):
    """Parameters for IVF (Inverted File Index) index."""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["IVF"] = "IVF"
    metric_type: MetricTypeEnum = Field(
        default=MetricTypeEnum.IP,
        description="Distance metric: COSINE, IP, or L2 (default: IP)",
    )
    nlist: int = Field(
        default=128,
        ge=1,
        description="Number of Voronoi cells (clusters). More = finer partitioning, slower build (default: 128)",
    )
    quantize_type: QuantizeTypeEnum = Field(
        default=QuantizeTypeEnum.UNDEFINED,
        description="Vector quantization type for compression: UNDEFINED (none), FP16, INT8 (default: UNDEFINED)",
    )


class InvertIndexParamInput(BaseModel):
    """Parameters for Inverted index on scalar fields."""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["INVERT"] = "INVERT"
    enable_range_optimization: bool = Field(
        default=False,
        description="Enable range query optimization for this scalar field (default: False)",
    )


# Annotated union types for use in field definitions
VectorIndexParam = Annotated[
    HnswIndexParamInput | FlatIndexParamInput | IVFIndexParamInput,
    Field(discriminator="type"),
]
ScalarOrVectorIndexParam = Annotated[
    HnswIndexParamInput | FlatIndexParamInput | IVFIndexParamInput | InvertIndexParamInput,
    Field(discriminator="type"),
]


class VectorFieldInput(BaseModel):
    """Vector field definition."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Name of the vector field", min_length=1)
    data_type: DataTypeEnum = Field(..., description="Vector data type (e.g., VECTOR_FP32)")
    dimension: int = Field(..., description="Dimensionality of the vector", ge=1)
    index_param: VectorIndexParam | None = Field(
        default=None,
        description="Optional index to create on this field when the collection is initialized. "
        "Supported types: HnswIndexParamInput, FlatIndexParamInput, IVFIndexParamInput",
    )


class ScalarFieldInput(BaseModel):
    """Scalar field definition."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Name of the scalar field", min_length=1)
    data_type: DataTypeEnum = Field(..., description="Data type (e.g., INT64, STRING, FLOAT)")
    nullable: bool = Field(default=False, description="Whether the field can be null")
    index_param: InvertIndexParamInput | None = Field(
        default=None,
        description="Optional inverted index to create on this scalar field when the collection is initialized",
    )


class CreateCollectionInput(BaseModel):
    """Input for creating a collection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    path: str = Field(
        ...,
        description="Filesystem path where the collection will be created",
        min_length=1,
    )
    collection_name: str = Field(
        ...,
        description="Name of the collection (also used as unique key in the session cache)",
        min_length=1,
    )
    vector_fields: list[VectorFieldInput] = Field(
        ..., description="List of vector fields", min_length=1
    )
    scalar_fields: list[ScalarFieldInput] | None = Field(
        default=None, description="List of scalar fields (optional)"
    )


class OpenCollectionInput(BaseModel):
    """Input for opening an existing collection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    path: str = Field(..., description="Filesystem path of the existing collection", min_length=1)
    collection_name: str = Field(
        ..., description="Unique name for this collection in the session", min_length=1
    )
    read_only: bool = Field(default=False, description="Open in read-only mode")


class GetCollectionInfoInput(BaseModel):
    """Input for getting collection information."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'"
    )


class DestroyCollectionInput(BaseModel):
    """Input for destroying a collection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)


class DocumentInput(BaseModel):
    """Document input for insert/upsert/update."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    id: str = Field(..., description="Unique document ID", min_length=1)
    vectors: dict[str, list[float] | dict[int, float]] | None = Field(
        default=None,
        description="Dict of vector field names to vector values (dense: List[float], sparse: Dict[int, float])",
    )
    fields: dict[str, Any] | None = Field(
        default=None, description="Dict of scalar field names to field values"
    )


class InsertDocumentsInput(BaseModel):
    """Input for inserting documents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    documents: list[DocumentInput] = Field(
        ..., description="List of documents to insert", min_length=1
    )


class UpsertDocumentsInput(BaseModel):
    """Input for upserting documents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    documents: list[DocumentInput] = Field(
        ..., description="List of documents to upsert", min_length=1
    )


class UpdateDocumentsInput(BaseModel):
    """Input for updating documents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    documents: list[DocumentInput] = Field(
        ..., description="List of documents with updates", min_length=1
    )


class DeleteDocumentsInput(BaseModel):
    """Input for deleting documents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    document_ids: list[str] = Field(..., description="List of document IDs to delete", min_length=1)


class FetchDocumentsInput(BaseModel):
    """Input for fetching documents by ID."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    document_ids: list[str] = Field(..., description="List of document IDs to fetch", min_length=1)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'"
    )


class VectorQueryInput(BaseModel):
    """Input for vector similarity search."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    field_name: str = Field(..., description="Name of the vector field to query", min_length=1)
    vector: list[float] = Field(..., description="Query vector", min_length=1)
    topk: int = Field(default=10, description="Number of results to return", ge=1, le=1000)
    filter: str | None = Field(
        default=None,
        description="Optional filter expression (e.g., 'age > 25 AND city == \"NYC\"'),",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'"
    )


class MultiVectorQuerySpec(BaseModel):
    """Single vector query specification for multi-vector search."""

    model_config = ConfigDict(str_strip_whitespace=True)

    field_name: str = Field(..., description="Name of the vector field to query", min_length=1)
    vector: list[float] | dict[int, float] = Field(
        ..., description="Query vector (dense: List[float], sparse: Dict[int, float])"
    )


class RerankerType(str, Enum):
    """Re-ranker types for multi-vector fusion."""

    WEIGHTED = "weighted"
    RRF = "rrf"  # Reciprocal Rank Fusion


class MultiVectorQueryInput(BaseModel):
    """Input for multi-vector similarity search."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    vectors: list[MultiVectorQuerySpec] = Field(
        ...,
        description="List of vector queries (one for each embedding space)",
        min_length=2,
    )
    topk: int = Field(
        default=10,
        description="Number of candidates to retrieve from each vector field",
        ge=1,
        le=1000,
    )
    topn: int = Field(
        default=5,
        description="Number of final documents to return after re-ranking",
        ge=1,
        le=100,
    )
    reranker_type: RerankerType = Field(
        default=RerankerType.WEIGHTED,
        description="Re-ranking strategy: 'weighted' or 'rrf'",
    )
    weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for each vector field (for weighted re-ranker). Example: {'dense': 1.2, 'sparse': 1.0}",
    )
    rank_constant: int = Field(
        default=60,
        description="Rank constant for RRF re-ranker (higher = less top-rank dominance)",
        ge=1,
    )
    metric_type: MetricTypeEnum = Field(
        default=MetricTypeEnum.IP,
        description="Similarity metric for score normalization (for weighted re-ranker)",
    )
    filter: str | None = Field(
        default=None,
        description="Optional filter expression to apply before search",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'"
    )


class CreateIndexInput(BaseModel):
    """Input for creating an index on a collection field."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    field_name: str = Field(..., description="Name of the field to index", min_length=1)
    index_param: ScalarOrVectorIndexParam = Field(
        ...,
        description="Index parameters. Use HnswIndexParamInput / FlatIndexParamInput / IVFIndexParamInput "
        "for vector fields, or InvertIndexParamInput for scalar fields. "
        'Example: {"type": "HNSW", "metric_type": "COSINE", "m": 16, "ef_construction": 200}',
    )


class DropIndexInput(BaseModel):
    """Input for dropping an index."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    field_name: str = Field(..., description="Name of the indexed field", min_length=1)


class OptimizeCollectionInput(BaseModel):
    """Input for optimizing a collection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)


# ============================================================================
# AI Extension Models (OpenAI Embedding)
# ============================================================================


class GenerateDenseEmbeddingInput(BaseModel):
    """Input for generating dense embeddings using OpenAIDenseEmbedding."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        ...,
        description="Text to generate dense embedding for",
        min_length=1,
    )
    api_key: str | None = Field(
        default=None,
        description="OpenAI API key. If not provided, uses OPENAI_API_KEY env var",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom OpenAI-compatible API base URL (e.g. for local or proxy endpoints). "
        "If not provided, uses OPENAI_BASE_URL env var or the default OpenAI endpoint",
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name (default: text-embedding-3-small)",
    )
    dimension: int = Field(
        default=1536,
        ge=1,
        description="Embedding dimension (default: 1536)",
    )


class TextDocumentInput(BaseModel):
    """A document with text content to be embedded."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., description="Unique document ID", min_length=1)
    text: str = Field(
        ...,
        description="Text content to be converted to a dense vector embedding",
        min_length=1,
    )
    fields: dict[str, Any] | None = Field(
        default=None,
        description='Optional scalar field values (e.g. {"title": "...", "score": 0.9})',
    )


class EmbeddingWriteInput(BaseModel):
    """Input for writing documents with auto-generated dense embeddings.

    OpenAI connection is configured via environment variables:
      OPENAI_API_KEY, OPENAI_BASE_URL (optional), OPENAI_EMBEDDING_MODEL (optional).
    The embedding dimension is inferred from the collection schema automatically.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    field_name: str = Field(
        ...,
        description="Name of the vector field in the collection to write into",
        min_length=1,
    )
    documents: list[TextDocumentInput] = Field(
        ...,
        description="List of documents whose text will be embedded and inserted",
        min_length=1,
    )


class EmbeddingSearchInput(BaseModel):
    """Input for semantic search using auto-generated dense embeddings.

    OpenAI connection is configured via environment variables:
      OPENAI_API_KEY, OPENAI_BASE_URL (optional), OPENAI_EMBEDDING_MODEL (optional).
    The embedding dimension is inferred from the collection schema automatically.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    collection_name: str = Field(..., description="Collection name", min_length=1)
    field_name: str = Field(
        ...,
        description="Name of the vector field in the collection to search against",
        min_length=1,
    )
    query_text: str = Field(
        ...,
        description="Natural language query to be converted to a vector for similarity search",
        min_length=1,
    )
    topk: int = Field(
        default=10,
        description="Number of results to return (default: 10)",
        ge=1,
        le=1000,
    )
    filter: str | None = Field(
        default=None,
        description="Optional filter expression (e.g., 'score > 0.8 AND category == \"news\"'),",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'"
    )
