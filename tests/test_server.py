#!/usr/bin/env python3
"""
Pytest test suite for zvec_mcp server.

Tests:
- Basic collection operations
- Document CRUD operations
- Single vector query
- Multi-vector query with re-ranking
- Helper functions

Run with:
    pytest tests/test_server.py -v
    pytest tests/test_server.py -v -s  # with print output
"""

import json
import os
import shutil
from collections.abc import Generator

import pytest
import zvec

from zvec_mcp.schemas import (
    CreateCollectionInput,
    CreateIndexInput,
    DeleteDocumentsInput,
    DocumentInput,
    FetchDocumentsInput,
    FlatIndexParamInput,
    GenerateDenseEmbeddingInput,
    GetCollectionInfoInput,
    HnswIndexParamInput,
    InsertDocumentsInput,
    InvertIndexParamInput,
    MultiVectorQueryInput,
    MultiVectorQuerySpec,
    RerankerType,
    ScalarFieldInput,
    UpdateDocumentsInput,
    UpsertDocumentsInput,
    VectorFieldInput,
    VectorQueryInput,
)
from zvec_mcp.server import (
    collection_details_resource,
    create_and_open_collection,
    create_index,
    delete_documents,
    fetch_documents,
    generate_dense_embedding,
    get_collection_info,
    insert_documents,
    list_collections_resource,
    multi_vector_query,
    update_documents,
    upsert_documents,
    vector_query,
)
from zvec_mcp.types import (
    DataTypeEnum,
    MetricTypeEnum,
    QuantizeTypeEnum,
    ResponseFormat,
)
from zvec_mcp.utils import get_collection, get_zvec_data_type, get_zvec_metric_type

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_collection_path() -> Generator[str, None, None]:
    """Provide a clean test collection path."""
    import random
    import string

    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    path = f"./test_collection_{random_suffix}"
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    yield path
    # Cleanup: use destroy_collection if cached, otherwise rmtree
    try:
        from zvec_mcp.utils import get_collection, remove_collection_from_cache

        collection = get_collection("test_collection")
        if collection is not None:
            collection.destroy()
            remove_collection_from_cache("test_collection")
    except Exception:
        pass
    # Always try to remove directory
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def multi_vector_path() -> Generator[str, None, None]:
    """Provide a clean multi-vector test collection path."""
    import random
    import string

    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    path = f"./test_multi_vector_{random_suffix}"
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    yield path
    # Cleanup: use destroy_collection if cached, otherwise rmtree
    try:
        from zvec_mcp.utils import list_cached_collections, remove_collection_from_cache

        for name, collection in list_cached_collections().items():
            if name.startswith("multi_col_"):
                collection.destroy()
                remove_collection_from_cache(name)
                break
    except Exception:
        pass
    # Always try to remove directory
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
async def basic_collection(test_collection_path: str) -> str:
    """Create a basic test collection with single vector field."""
    collection_name = "test_collection"

    result = await create_and_open_collection(
        CreateCollectionInput(
            path=test_collection_path,
            vector_fields=[
                VectorFieldInput(
                    name="embedding",
                    data_type=DataTypeEnum.VECTOR_FP32,
                    dimension=4,
                )
            ],
            scalar_fields=[
                ScalarFieldInput(name="title", data_type=DataTypeEnum.STRING, nullable=False)
            ],
            collection_name=collection_name,
        )
    )
    assert "Successfully created" in result
    return collection_name


@pytest.fixture
async def collection_with_docs(basic_collection: str) -> str:
    """Create collection and insert test documents."""
    result = await insert_documents(
        InsertDocumentsInput(
            collection_name=basic_collection,
            documents=[
                DocumentInput(
                    id="doc1",
                    vectors={"embedding": [0.1, 0.2, 0.3, 0.4]},
                    fields={"title": "First Document"},
                ),
                DocumentInput(
                    id="doc2",
                    vectors={"embedding": [0.2, 0.3, 0.4, 0.1]},
                    fields={"title": "Second Document"},
                ),
                DocumentInput(
                    id="doc3",
                    vectors={"embedding": [0.4, 0.3, 0.2, 0.1]},
                    fields={"title": "Third Document"},
                ),
            ],
        )
    )
    assert "3/3" in result
    return basic_collection


@pytest.fixture
async def multi_vector_collection(multi_vector_path: str) -> str:
    """Create collection with multiple vector fields."""
    import random
    import string

    collection_name = "multi_col_" + "".join(random.choices(string.ascii_lowercase, k=6))

    result = await create_and_open_collection(
        CreateCollectionInput(
            path=multi_vector_path,
            vector_fields=[
                VectorFieldInput(name="dense", data_type=DataTypeEnum.VECTOR_FP32, dimension=3),
                VectorFieldInput(
                    name="sparse",
                    data_type=DataTypeEnum.SPARSE_VECTOR_FP32,
                    dimension=50,
                ),
            ],
            scalar_fields=[ScalarFieldInput(name="category", data_type=DataTypeEnum.STRING)],
            collection_name=collection_name,
        )
    )
    assert "Successfully created" in result

    # Insert documents with multiple embeddings
    insert_result = await insert_documents(
        InsertDocumentsInput(
            collection_name=collection_name,
            documents=[
                DocumentInput(
                    id="doc1",
                    vectors={"dense": [0.9, 0.1, 0.2], "sparse": {1: 0.8, 5: 0.6}},
                    fields={"category": "A"},
                ),
                DocumentInput(
                    id="doc2",
                    vectors={"dense": [0.1, 0.9, 0.3], "sparse": {2: 0.7, 5: 0.5}},
                    fields={"category": "B"},
                ),
                DocumentInput(
                    id="doc3",
                    vectors={"dense": [0.2, 0.3, 0.9], "sparse": {3: 0.9, 7: 0.4}},
                    fields={"category": "A"},
                ),
            ],
        )
    )
    assert "3/3" in insert_result
    return collection_name


# ============================================================================
# Test Cases - Collection Management
# ============================================================================


@pytest.mark.asyncio
class TestCollectionManagement:
    """Test collection creation and management."""

    async def test_resources_list_and_read(self, collection_with_docs: str):
        """Test FastMCP resources list and read behavior (collections & single collection)."""
        # list_collections_resource should return JSON with at least one collection
        collections_json = await list_collections_resource()
        data = json.loads(collections_json)
        assert "collections" in data
        assert any(c["collection_name"] == collection_with_docs for c in data["collections"])

        # collection_details_resource should return detailed schema info for target collection
        details_json = await collection_details_resource(collection_name=collection_with_docs)
        details = json.loads(details_json)
        assert details["name"] == "test_collection"
        assert any(vf["name"] == "embedding" for vf in details["vector_fields"])

    async def test_create_collection(self, test_collection_path: str):
        """Test creating a new collection."""
        result = await create_and_open_collection(
            CreateCollectionInput(
                path=test_collection_path,
                vector_fields=[
                    VectorFieldInput(name="vec", data_type=DataTypeEnum.VECTOR_FP32, dimension=3)
                ],
                collection_name="create_test",
            )
        )
        assert "Successfully created" in result
        assert "create_test" in result
        assert "Indexes created: 0" in result

    async def test_create_collection_with_index_param(self, test_collection_path: str):
        """Test creating a collection with typed index_param on vector and scalar fields."""
        result = await create_and_open_collection(
            CreateCollectionInput(
                path=test_collection_path,
                collection_name="indexed_col",
                vector_fields=[
                    VectorFieldInput(
                        name="embedding",
                        data_type=DataTypeEnum.VECTOR_FP32,
                        dimension=4,
                        index_param=HnswIndexParamInput(
                            metric_type=MetricTypeEnum.COSINE,
                            m=16,
                            ef_construction=200,
                        ),
                    )
                ],
                scalar_fields=[
                    ScalarFieldInput(
                        name="title",
                        data_type=DataTypeEnum.STRING,
                        index_param=InvertIndexParamInput(
                            enable_range_optimization=False,
                        ),
                    )
                ],
            )
        )
        assert "Successfully created" in result
        assert "Indexes created: 2" in result

    async def test_create_collection_with_flat_index(self, test_collection_path: str):
        """Test creating a collection with FlatIndexParam and quantization."""
        result = await create_and_open_collection(
            CreateCollectionInput(
                path=test_collection_path,
                collection_name="flat_col",
                vector_fields=[
                    VectorFieldInput(
                        name="vec",
                        data_type=DataTypeEnum.VECTOR_FP32,
                        dimension=4,
                        index_param=FlatIndexParamInput(
                            metric_type=MetricTypeEnum.L2,
                            quantize_type=QuantizeTypeEnum.FP16,
                        ),
                    )
                ],
            )
        )
        assert "Successfully created" in result
        assert "Indexes created: 1" in result

    async def test_get_collection_info_json(self, collection_with_docs: str):
        """Test getting collection info in JSON format."""
        result = await get_collection_info(
            GetCollectionInfoInput(
                collection_name=collection_with_docs, response_format=ResponseFormat.JSON
            )
        )
        assert "test_collection" in result
        assert "embedding" in result

    async def test_get_collection_info_markdown(self, collection_with_docs: str):
        """Test getting collection info in Markdown format."""
        result = await get_collection_info(
            GetCollectionInfoInput(
                collection_name=collection_with_docs,
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        assert "Collection Information" in result
        assert "test_collection" in result


# ============================================================================
# Test Cases - Document Operations
# ============================================================================


@pytest.mark.asyncio
class TestDocumentOperations:
    """Test document CRUD operations."""

    async def test_insert_documents(self, basic_collection: str):
        """Test inserting documents."""
        result = await insert_documents(
            InsertDocumentsInput(
                collection_name=basic_collection,
                documents=[
                    DocumentInput(
                        id="test1",
                        vectors={"embedding": [0.1, 0.2, 0.3, 0.4]},
                        fields={"title": "Test Doc"},
                    )
                ],
            )
        )
        assert "1/1" in result

    async def test_upsert_documents(self, collection_with_docs: str):
        """Test upserting documents."""
        result = await upsert_documents(
            UpsertDocumentsInput(
                collection_name=collection_with_docs,
                documents=[
                    DocumentInput(
                        id="doc1",  # Existing
                        vectors={"embedding": [0.5, 0.5, 0.5, 0.5]},
                        fields={"title": "Updated First"},
                    ),
                    DocumentInput(
                        id="doc_new",  # New
                        vectors={"embedding": [0.1, 0.1, 0.1, 0.1]},
                        fields={"title": "New Doc"},
                    ),
                ],
            )
        )
        assert "2/2" in result

    async def test_update_documents(self, collection_with_docs: str):
        """Test updating existing documents."""
        result = await update_documents(
            UpdateDocumentsInput(
                collection_name=collection_with_docs,
                documents=[
                    DocumentInput(
                        id="doc2",
                        vectors={"embedding": [0.9, 0.8, 0.7, 0.6]},
                        fields={"title": "Updated Second"},
                    )
                ],
            )
        )
        assert "1/1" in result

    async def test_delete_documents(self, collection_with_docs: str):
        """Test deleting documents."""
        result = await delete_documents(
            DeleteDocumentsInput(collection_name=collection_with_docs, document_ids=["doc3"])
        )
        assert "1/1" in result

    async def test_fetch_documents(self, collection_with_docs: str):
        """Test fetching documents by ID."""
        result = await fetch_documents(
            FetchDocumentsInput(
                collection_name=collection_with_docs,
                document_ids=["doc1", "doc2"],
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        assert "doc1" in result
        assert "doc2" in result
        assert "Documents" in result


# ============================================================================
# Test Cases - Vector Query
# ============================================================================


@pytest.mark.asyncio
class TestVectorQuery:
    """Test single vector query operations."""

    async def test_vector_query_basic(self, collection_with_docs: str):
        """Test basic vector similarity search."""
        result = await vector_query(
            VectorQueryInput(
                collection_name=collection_with_docs,
                field_name="embedding",
                vector=[0.15, 0.25, 0.35, 0.25],
                topk=2,
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        assert "Documents" in result
        assert "Total: 2" in result

    async def test_vector_query_json_format(self, collection_with_docs: str):
        """Test vector query with JSON response format."""
        result = await vector_query(
            VectorQueryInput(
                collection_name=collection_with_docs,
                field_name="embedding",
                vector=[0.2, 0.3, 0.4, 0.1],
                topk=3,
                response_format=ResponseFormat.JSON,
            )
        )
        assert "[" in result  # JSON array
        assert "score" in result


# ============================================================================
# Test Cases - Multi-Vector Query
# ============================================================================


@pytest.mark.asyncio
class TestMultiVectorQuery:
    """Test multi-vector query with re-ranking."""

    async def test_weighted_reranker(self, multi_vector_collection: str):
        """Test multi-vector query with Weighted re-ranker."""
        result = await multi_vector_query(
            MultiVectorQueryInput(
                collection_name=multi_vector_collection,
                vectors=[
                    MultiVectorQuerySpec(field_name="dense", vector=[0.8, 0.2, 0.3]),
                    MultiVectorQuerySpec(field_name="sparse", vector={1: 0.7, 5: 0.6}),
                ],
                topk=3,
                topn=2,
                reranker_type=RerankerType.WEIGHTED,
                weights={"dense": 1.5, "sparse": 1.0},
                metric_type=MetricTypeEnum.IP,
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        assert "Documents" in result
        assert "Total: 2" in result

    async def test_rrf_reranker(self, multi_vector_collection: str):
        """Test multi-vector query with RRF re-ranker."""
        result = await multi_vector_query(
            MultiVectorQueryInput(
                collection_name=multi_vector_collection,
                vectors=[
                    MultiVectorQuerySpec(field_name="dense", vector=[0.8, 0.2, 0.3]),
                    MultiVectorQuerySpec(field_name="sparse", vector={1: 0.7, 5: 0.6}),
                ],
                topk=3,
                topn=2,
                reranker_type=RerankerType.RRF,
                rank_constant=60,
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        assert "Documents" in result

    async def test_multi_vector_with_filter(self, multi_vector_collection: str):
        """Test multi-vector query with filter."""
        result = await multi_vector_query(
            MultiVectorQueryInput(
                collection_name=multi_vector_collection,
                vectors=[
                    MultiVectorQuerySpec(field_name="dense", vector=[0.8, 0.2, 0.3]),
                    MultiVectorQuerySpec(field_name="sparse", vector={1: 0.7}),
                ],
                topk=3,
                topn=2,
                reranker_type=RerankerType.WEIGHTED,
                filter='category = "A"',
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        assert "Documents" in result

    async def test_multi_vector_default_weights(self, multi_vector_collection: str):
        """Test multi-vector query with default equal weights."""
        result = await multi_vector_query(
            MultiVectorQueryInput(
                collection_name=multi_vector_collection,
                vectors=[
                    MultiVectorQuerySpec(field_name="dense", vector=[0.9, 0.1, 0.2]),
                    MultiVectorQuerySpec(field_name="sparse", vector={1: 0.8}),
                ],
                topk=3,
                topn=2,
                reranker_type=RerankerType.WEIGHTED,
                # weights not provided - should use default equal weights
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        assert "Documents" in result


# ============================================================================
# Test Cases - Index Management
# ============================================================================


@pytest.mark.asyncio
class TestIndexManagement:
    """Test index creation and management."""

    async def test_create_hnsw_index(self, collection_with_docs: str):
        """Test creating HNSW index with full parameters."""
        result = await create_index(
            CreateIndexInput(
                collection_name=collection_with_docs,
                field_name="embedding",
                index_param=HnswIndexParamInput(
                    metric_type=MetricTypeEnum.COSINE,
                    m=32,
                    ef_construction=300,
                    quantize_type=QuantizeTypeEnum.UNDEFINED,
                ),
            )
        )
        assert "Successfully created" in result
        assert "HNSW" in result

    async def test_create_invert_index(self, collection_with_docs: str):
        """Test creating inverted index on a scalar field."""
        result = await create_index(
            CreateIndexInput(
                collection_name=collection_with_docs,
                field_name="title",
                index_param=InvertIndexParamInput(
                    enable_range_optimization=True,
                ),
            )
        )
        assert "Successfully created" in result
        assert "INVERT" in result


# ============================================================================
# Test Cases - Helper Functions
# ============================================================================


@pytest.mark.asyncio
class TestHelperFunctions:
    """Test utility helper functions."""

    async def test_get_zvec_data_type(self):
        """Test data type conversion."""
        dt = get_zvec_data_type("VECTOR_FP32")
        assert dt == zvec.DataType.VECTOR_FP32

    async def test_get_zvec_metric_type(self):
        """Test metric type conversion."""
        mt = get_zvec_metric_type("COSINE")
        assert mt == zvec.MetricType.COSINE

    async def test_collection_cache(self, basic_collection: str):
        """Test collection caching mechanism."""
        cached = get_collection(basic_collection)
        assert cached is not None
        assert cached.schema.name == "test_collection"

    async def test_invalid_collection_cache(self):
        """Test fetching non-existent collection from cache."""
        cached = get_collection("non_existent")
        assert cached is None


# ============================================================================
# Test Cases - Error Handling
# ============================================================================


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_query_nonexistent_collection(self):
        """Test querying a non-existent collection."""
        result = await vector_query(
            VectorQueryInput(
                collection_name="nonexistent",
                field_name="embedding",
                vector=[0.1, 0.2, 0.3],
                topk=5,
            )
        )
        assert "Error" in result
        assert "not found" in result

    async def test_fetch_nonexistent_documents(self, collection_with_docs: str):
        """Test fetching non-existent documents."""
        result = await fetch_documents(
            FetchDocumentsInput(
                collection_name=collection_with_docs,
                document_ids=["nonexistent1", "nonexistent2"],
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        # Should return empty or error message
        assert result is not None


# ============================================================================
# Test Cases - AI Extension (Qwen Embedding & Reranker)
# ============================================================================


@pytest.mark.asyncio
class TestAIExtension:
    """Test AI extension tools (embedding and reranking)."""

    async def test_dense_embedding_no_api_key(self):
        """Test dense embedding without API key returns error."""
        # Ensure no env var is set
        import os

        original_key = os.environ.pop("DASHSCOPE_API_KEY", None)
        try:
            result = await generate_dense_embedding(
                GenerateDenseEmbeddingInput(
                    text="Test text for embedding",
                    dimension=256,
                )
            )
            assert "error" in result.lower() or "api key" in result.lower()
        finally:
            if original_key:
                os.environ["DASHSCOPE_API_KEY"] = original_key
