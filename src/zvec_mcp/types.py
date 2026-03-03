"""Type definitions and enums for Zvec MCP server."""

from enum import Enum


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class DataTypeEnum(str, Enum):
    """Zvec data types."""

    STRING = "STRING"
    BOOL = "BOOL"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    VECTOR_FP16 = "VECTOR_FP16"
    VECTOR_FP32 = "VECTOR_FP32"
    VECTOR_FP64 = "VECTOR_FP64"
    VECTOR_INT8 = "VECTOR_INT8"
    SPARSE_VECTOR_FP32 = "SPARSE_VECTOR_FP32"
    SPARSE_VECTOR_FP16 = "SPARSE_VECTOR_FP16"


class MetricTypeEnum(str, Enum):
    """Distance/similarity metrics."""

    COSINE = "COSINE"
    IP = "IP"  # Inner Product
    L2 = "L2"  # Euclidean Distance


class IndexTypeEnum(str, Enum):
    """Index types."""

    HNSW = "HNSW"
    IVF = "IVF"
    FLAT = "FLAT"


class QuantizeTypeEnum(str, Enum):
    """Quantization types for vector compression."""

    UNDEFINED = "UNDEFINED"  # No quantization (default)
    FP16 = "FP16"  # FP16 quantization
    INT8 = "INT8"  # INT8 quantization
