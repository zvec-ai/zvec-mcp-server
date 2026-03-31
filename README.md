# Zvec MCP Server

A Model Context Protocol (MCP) server for [Zvec](https://github.com/alibaba/zvec), a high-performance embedded vector database by Alibaba.

## Overview

This MCP server enables LLMs to interact with Zvec vector database through well-designed tools. It provides comprehensive functionality for:

- **Collection Management**: Create, open, and manage vector database collections
- **Document Operations**: Insert, update, delete, and fetch documents with full CRUD support
- **Vector Search**: Single-vector and multi-vector similarity search with re-ranking
- **Index Management**: Create and manage vector indexes (HNSW, IVF, FLAT) for fast retrieval
- **AI Embedding**: OpenAI-powered dense embedding with automatic text-to-vector conversion

## Features

- 🚀 **17 Comprehensive Tools**: Full API coverage for common vector database operations
- 🤖 **AI-Powered Embedding**: Built-in OpenAI embedding for semantic search
- 📊 **Multiple Response Formats**: Support both JSON and Markdown output formats
- 🔍 **Multi-Vector Search**: Combine multiple embeddings with advanced re-ranking
- 🎯 **Hybrid Search**: Combine vector similarity with scalar filters
- 🛡️ **Type Safety**: Full Pydantic v2 validation for all inputs
- 📝 **Rich Documentation**: Detailed tool descriptions with examples

## Installation

### Requirements

- Python 3.10 - 3.14
- Supported platforms: Linux (x86_64, ARM64), macOS (ARM64)

### Install from PyPI

```bash
# Using uv (recommended)
uv pip install zvec-mcp-server

# Or using pip
pip install zvec-mcp-server
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/zvec-ai/zvec-mcp-server.git
cd zvec-mcp-server

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### Running the Server

```bash
# Using the installed package
python -m zvec_mcp

# Or with uv
uv run python -m zvec_mcp

# Test with MCP Inspector
npx @modelcontextprotocol/inspector python -m zvec_mcp
```

### IDE Integration (Qoder/Cursor/Claude Desktop)

Add to your IDE's MCP configuration file:

**Qoder MCP Config** (`~/.qoder/mcp.json` or `~/.config/qoder/mcp.json`):

```json
{
  "mcpServers": {
    "zvec-mcp": {
      "command": "uvx",
      "args": ["zvec-mcp-server"],
      "env": {
        "OPENAI_API_KEY": "your-api-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
      }
    }
  }
}
```

**Claude Desktop Config** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "zvec-mcp": {
      "command": "uvx",
      "args": ["zvec-mcp-server"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

**Environment Variables:**

- `OPENAI_API_KEY` (required): OpenAI API key for embedding generation
- `OPENAI_BASE_URL` (optional): Custom API endpoint (e.g., for DashScope)
- `OPENAI_EMBEDDING_MODEL` (optional): Model name, default is `text-embedding-3-small`

### Basic Usage Example

```python
# 1. Create and open a collection
create_and_open_collection({
    "path": "./my_vectors",
    "collection_name": "docs_col",
    "vector_fields": [
        {
            "name": "embedding",
            "data_type": "VECTOR_FP32",
            "dimension": 1536
        }
    ],
    "scalar_fields": [
        {
            "name": "title",
            "data_type": "STRING",
            "nullable": False
        }
    ]
})

# 2. Insert documents with auto-generated embeddings (requires OPENAI_API_KEY)
embedding_write({
    "collection_name": "docs_col",
    "field_name": "embedding",
    "documents": [
        {
            "id": "doc1",
            "text": "This is a sample document about machine learning.",
            "fields": {"title": "ML Introduction"}
        }
    ]
})

# 3. Semantic search with natural language query
embedding_search({
    "collection_name": "docs_col",
    "field_name": "embedding",
    "query_text": "artificial intelligence and neural networks",
    "topk": 10
})
```

## Available Tools

### Collection Management (4 tools)
- `create_and_open_collection` - Create new collection with schema and auto-create indexes
- `open_collection` - Open existing collection into session cache
- `get_collection_info` - Get schema and statistics
- `destroy_collection` - Permanently delete collection

### Document Operations (5 tools)
- `insert_documents` - Insert new documents (fail if exists)
- `upsert_documents` - Insert or update documents
- `update_documents` - Update existing documents
- `delete_documents` - Delete documents by ID
- `fetch_documents` - Retrieve documents by ID

### Vector Search (2 tools)
- `vector_query` - Single-vector similarity search with optional filtering
- `multi_vector_query` - Multi-vector search with re-ranking (Weighted/RRF)

### Index Management (3 tools)
- `create_index` - Create vector index (HNSW/IVF/FLAT) or scalar index (INVERT)
- `drop_index` - Remove index from field
- `optimize_collection` - Optimize collection for better performance

### AI Embedding (3 tools)
- `generate_dense_embedding` - Generate embedding for text using OpenAI API
- `embedding_write` - Auto-embed text documents and upsert to collection
- `embedding_search` - Natural language semantic search with auto-embedding

## Tool Details

### Vector Data Types
- `VECTOR_FP32`, `VECTOR_FP64`, `VECTOR_FP16` - Dense float vectors
- `VECTOR_INT8` - Dense integer vectors
- `SPARSE_VECTOR_FP32`, `SPARSE_VECTOR_FP16` - Sparse vectors (Dict[int, float])

### Scalar Data Types
- `INT32`, `INT64`, `UINT32`, `UINT64` - Integer types
- `FLOAT`, `DOUBLE` - Floating point types
- `STRING`, `BOOL` - Text and boolean

### Index Types

**Vector Indexes:**
- `HNSW` - Hierarchical Navigable Small World (recommended for most cases)
- `IVF` - Inverted File Index (good for large datasets)
- `FLAT` - Brute-force exact search (small datasets)

**Scalar Indexes:**
- `INVERT` - Inverted index for scalar fields with optional range optimization

### Distance Metrics
- `COSINE` - Cosine similarity
- `IP` - Inner product
- `L2` - Euclidean distance

### Re-ranking Strategies (Multi-Vector Query)
- `WEIGHTED` - Weighted score fusion with custom weights per field
- `RRF` - Reciprocal Rank Fusion (rank-based fusion)

## Architecture

### Modular Structure

```
zvec-mcp-server/
├── src/
│   └── zvec_mcp/
│       ├── __init__.py       # Package entry point
│       ├── server.py         # MCP server implementation (17 tools)
│       ├── schemas.py        # Pydantic input validation models
│       ├── types.py          # Enums and type definitions
│       └── utils.py          # Helper functions and formatters
├── tests/
│   └── test_server.py        # Pytest test suite
├── pyproject.toml            # Project configuration
├── README.md                 # This file
├── CONTRIBUTING.md           # Contribution guidelines
└── LICENSE                   # Apache 2.0 License
```

### MCP Resources
The server exposes two MCP resources for introspection:
- `zvec://collections` - List all opened collections in the current session
- `zvec://collection/{collection_name}` - Get detailed schema and stats for a specific collection

### Error Handling
All tools provide clear, actionable error messages:
- Resource not found errors with suggestions
- Validation errors from Pydantic v2
- Zvec API errors with context

### Response Formats
Tools support two output formats:
- **JSON**: Structured data for programmatic processing
- **Markdown**: Human-readable formatted text with headers and lists

## Development

### Running Tests

The project includes a comprehensive pytest test suite with 21 test cases covering all functionality.

```bash
# Install dev dependencies (includes pytest and pytest-asyncio)
uv pip install -e ".[dev]"

# Run all tests
pytest tests/test_server.py -v
```
## References

- [Zvec GitHub](https://github.com/alibaba/zvec)
- [Zvec Documentation](https://zvec.org/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/python-sdk)

## License

[Apache 2.0](LICENSE)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
