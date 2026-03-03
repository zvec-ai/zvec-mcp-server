"""Utility functions for Zvec MCP server."""

import json

try:
    import zvec
except ImportError:
    raise ImportError("zvec is not installed. Please install it with: pip install zvec")

from zvec_mcp.types import ResponseFormat

# Global cache for opened collections
_open_collections: dict[str, zvec.Collection] = {}


def get_zvec_data_type(type_str: str) -> zvec.DataType:
    """Convert string to zvec.DataType.

    Args:
        type_str: String representation of data type

    Returns:
        zvec.DataType enum value

    Raises:
        ValueError: If type_str is not a valid data type
    """
    try:
        return getattr(zvec.DataType, type_str.upper())
    except AttributeError:
        raise ValueError(f"Invalid data type: {type_str}")


def get_zvec_metric_type(metric_str: str) -> zvec.MetricType:
    """Convert string to zvec.MetricType.

    Args:
        metric_str: String representation of metric type

    Returns:
        zvec.MetricType enum value

    Raises:
        ValueError: If metric_str is not a valid metric type
    """
    try:
        return getattr(zvec.MetricType, metric_str.upper())
    except AttributeError:
        raise ValueError(f"Invalid metric type: {metric_str}")


def get_zvec_quantize_type(quantize_str: str) -> object:
    """Convert string to zvec.QuantizeType.

    Args:
        quantize_str: String representation of quantize type (UNDEFINED, FP16, INT8)

    Returns:
        zvec.QuantizeType enum value

    Raises:
        ValueError: If quantize_str is not a valid quantize type
    """
    try:
        return getattr(zvec.QuantizeType, quantize_str.upper())
    except AttributeError:
        raise ValueError(f"Invalid quantize type: {quantize_str}")


def format_doc_list(docs: list[zvec.Doc], format_type: ResponseFormat) -> str:
    """Format document list as Markdown or JSON.

    Args:
        docs: List of zvec documents
        format_type: Output format (JSON or Markdown)

    Returns:
        Formatted string
    """
    if format_type == ResponseFormat.JSON:
        return _format_docs_json(docs)
    return _format_docs_markdown(docs)


def _format_docs_json(docs: list[zvec.Doc]) -> str:
    """Format documents as JSON."""
    doc_dicts = [
        {
            "id": doc.id,
            **({"score": doc.score} if doc.score is not None else {}),
            **({"vectors": doc.vectors} if doc.vectors else {}),
            **({"fields": doc.fields} if doc.fields else {}),
        }
        for doc in docs
    ]
    return json.dumps(doc_dicts, indent=2)


def _format_docs_markdown(docs: list[zvec.Doc]) -> str:
    """Format documents as Markdown."""
    lines = [f"# Documents (Total: {len(docs)})", ""]

    for i, doc in enumerate(docs, 1):
        lines.extend(_format_single_doc_markdown(doc, i))

    return "\n".join(lines)


def _format_single_doc_markdown(doc: zvec.Doc, index: int) -> list[str]:
    """Format a single document as Markdown lines."""
    lines = [f"## {index}. Document ID: {doc.id}"]

    if doc.score is not None:
        lines.append(f"- **Score**: {doc.score:.4f}")

    if doc.fields:
        lines.append("- **Fields**:")
        for k, v in doc.fields.items():
            lines.append(f"  - `{k}`: {v}")

    if doc.vectors:
        lines.append("- **Vectors**:")
        for k, v in doc.vectors.items():
            vec_preview = str(v[:5]) + "..." if len(v) > 5 else str(v)
            lines.append(f"  - `{k}`: {vec_preview} (dim: {len(v)})")

    lines.append("")
    return lines


def handle_error(e: Exception) -> str:
    """Consistent error formatting.

    Args:
        e: Exception to format

    Returns:
        Formatted error message
    """
    error_msg = str(e)
    if "not found" in error_msg.lower():
        return f"Error: Resource not found. {error_msg}"
    elif "already exists" in error_msg.lower():
        return f"Error: Resource already exists. {error_msg}"
    elif "invalid" in error_msg.lower():
        return f"Error: Invalid argument. {error_msg}"
    return f"Error: {error_msg}"


def get_collection(collection_name: str) -> zvec.Collection | None:
    """Get collection from cache.

    Args:
        collection_name: Collection name

    Returns:
        Collection instance or None
    """
    return _open_collections.get(collection_name)


def list_cached_collections() -> dict[str, zvec.Collection]:
    """Return a copy of all cached collections.

    Key is collection_name, value is zvec.Collection.
    """
    return dict(_open_collections)


def cache_collection(collection_name: str, collection: zvec.Collection) -> None:
    """Cache an opened collection.

    Args:
        collection_name: Collection name
        collection: Collection instance to cache
    """
    _open_collections[collection_name] = collection


def remove_collection_from_cache(collection_name: str) -> None:
    """Remove collection from cache.

    Args:
        collection_name: Collection name
    """
    if collection_name in _open_collections:
        del _open_collections[collection_name]
