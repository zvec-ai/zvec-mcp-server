"""MCP Server implementation for Zvec vector database."""

import json

try:
    import zvec
except ImportError:
    raise ImportError("zvec is not installed. Please install it with: pip install zvec")

from mcp.server.fastmcp import FastMCP

from zvec_mcp.schemas import (
    CreateCollectionInput,
    CreateIndexInput,
    DeleteDocumentsInput,
    DestroyCollectionInput,
    DropIndexInput,
    EmbeddingSearchInput,
    EmbeddingWriteInput,
    FetchDocumentsInput,
    FlatIndexParamInput,
    GenerateDenseEmbeddingInput,
    GetCollectionInfoInput,
    HnswIndexParamInput,
    InsertDocumentsInput,
    InvertIndexParamInput,
    IVFIndexParamInput,
    MultiVectorQueryInput,
    OpenCollectionInput,
    OptimizeCollectionInput,
    RerankerType,
    UpdateDocumentsInput,
    UpsertDocumentsInput,
    VectorQueryInput,
)
from zvec_mcp.types import ResponseFormat
from zvec_mcp.utils import (
    cache_collection,
    format_doc_list,
    get_collection,
    get_zvec_data_type,
    get_zvec_metric_type,
    get_zvec_quantize_type,
    handle_error,
    list_cached_collections,
    remove_collection_from_cache,
)

# Initialize the MCP server
mcp = FastMCP("zvec_mcp")


@mcp.resource(
    uri="zvec://collections",
    name="ZvecCollections",
    description="List all currently opened Zvec collections in this MCP session.",
    mime_type="application/json",
    annotations={
        "title": "Opened Zvec Collections",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def list_collections_resource() -> str:
    """Resource: list currently cached collections."""
    try:
        collections_payload = []
        for collection_name, collection in list_cached_collections().items():
            stats = collection.stats

            collections_payload.append(
                {
                    "collection_name": collection_name,
                    "path": collection.path,
                    "doc_count": getattr(stats, "doc_count", None),
                }
            )

        return json.dumps({"collections": collections_payload}, indent=2)
    except Exception as e:
        return json.dumps({"error": handle_error(e)}, indent=2)


@mcp.resource(
    uri="zvec://collection/{collection_name}",
    name="ZvecCollectionDetails",
    description="Detailed schema and stats for a specific Zvec collection.",
    mime_type="application/json",
    annotations={
        "title": "Zvec Collection Details",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def collection_details_resource(collection_name: str) -> str:
    """Resource: detailed info for a single collection."""
    try:
        collection = get_collection(collection_name)
        if collection is None:
            return json.dumps(
                {"error": f"Collection '{collection_name}' not found. Please open it first."},
                indent=2,
            )

        schema = collection.schema
        stats = collection.stats

        info = {
            "path": collection.path,
            "name": schema.name,
            "vector_fields": [
                {
                    "name": v.name,
                    "data_type": str(v.data_type),
                    "dimension": v.dimension,
                    "index_param": v.index_param.to_dict()
                    if getattr(v, "index_param", None) is not None
                    else None,
                }
                for v in schema.vectors
            ],
            "scalar_fields": [
                {
                    "name": f.name,
                    "data_type": str(f.data_type),
                    "nullable": f.nullable,
                    "index_param": f.index_param.to_dict()
                    if getattr(f, "index_param", None) is not None
                    else None,
                }
                for f in (schema.fields or [])
            ],
            "stats": {
                "doc_count": getattr(stats, "doc_count", None),
            },
        }

        return json.dumps(info, indent=2)
    except Exception as e:
        return json.dumps({"error": handle_error(e)}, indent=2)


def _build_zvec_index_param(index_param):
    """Convert an IndexParamInput model to the corresponding zvec index param object."""
    if isinstance(index_param, HnswIndexParamInput):
        return zvec.HnswIndexParam(
            metric_type=get_zvec_metric_type(index_param.metric_type.value),
            m=index_param.m,
            ef_construction=index_param.ef_construction,
            quantize_type=get_zvec_quantize_type(index_param.quantize_type.value),
        )
    elif isinstance(index_param, FlatIndexParamInput):
        return zvec.FlatIndexParam(
            metric_type=get_zvec_metric_type(index_param.metric_type.value),
            quantize_type=get_zvec_quantize_type(index_param.quantize_type.value),
        )
    elif isinstance(index_param, IVFIndexParamInput):
        return zvec.IVFIndexParam(
            metric_type=get_zvec_metric_type(index_param.metric_type.value),
            nlist=index_param.nlist,
            quantize_type=get_zvec_quantize_type(index_param.quantize_type.value),
        )
    elif isinstance(index_param, InvertIndexParamInput):
        return zvec.InvertIndexParam(
            enable_range_optimization=index_param.enable_range_optimization,
        )
    else:
        raise ValueError(f"Unsupported index param type: {type(index_param)}")


# ============================================================================
# MCP Tools - Collection Management
# ============================================================================


@mcp.tool(
    name="create_and_open_collection",
    annotations={
        "title": "Create and Open Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def create_and_open_collection(params: CreateCollectionInput) -> str:
    """
    Create a new Zvec collection and open it for use.

    This tool creates a new vector database collection at the specified path with the
    given schema definition. The collection is automatically opened and cached for
    subsequent operations. Use this when you need to initialize a new vector database.

    Args:
        params (CreateCollectionInput): Validated input parameters containing:
            - path (str): Filesystem path where collection will be created (e.g., './my_vectors')
            - collection_name (str): Name of the collection (also used as unique session key)
            - vector_fields (List[VectorFieldInput]): Vector field definitions (required, min 1);
              each field may include an optional `index_param` to auto-create its index
            - scalar_fields (Optional[List[ScalarFieldInput]]): Scalar field definitions;
              each field may also include an optional `index_param`

    Returns:
        str: Success message with collection details or error message

    Examples:
        - Use when: "Create a new collection for storing document embeddings"
        - Use when: "Initialize a vector database at ./embeddings with 768-dim vectors"
        - Don't use when: Collection already exists (use open_collection instead)
    """
    try:
        # Build vector schemas
        vector_schemas = []
        for vf in params.vector_fields:
            vector_schemas.append(
                zvec.VectorSchema(
                    name=vf.name,
                    data_type=get_zvec_data_type(vf.data_type.value),
                    dimension=vf.dimension,
                )
            )

        # Build scalar field schemas
        field_schemas = []
        if params.scalar_fields:
            for sf in params.scalar_fields:
                field_schemas.append(
                    zvec.FieldSchema(
                        name=sf.name,
                        data_type=get_zvec_data_type(sf.data_type.value),
                        nullable=sf.nullable,
                    )
                )

        # Create collection schema
        schema = zvec.CollectionSchema(
            name=params.collection_name,
            vectors=vector_schemas,
            fields=field_schemas if field_schemas else None,
        )

        # Create and open collection
        collection = zvec.create_and_open(path=params.path, schema=schema)

        # Cache the collection
        cache_collection(params.collection_name, collection)

        # Create indexes for any fields that specify index_param
        indexed_count = 0
        all_fields = list(params.vector_fields) + list(params.scalar_fields or [])
        for field in all_fields:
            if field.index_param is not None:
                zvec_index_param = _build_zvec_index_param(field.index_param)
                collection.create_index(field_name=field.name, index_param=zvec_index_param)
                indexed_count += 1

        return (
            f"Successfully created and opened collection '{params.collection_name}' at path '{params.path}'.\n"
            f"Collection ID: {params.collection_name}\n"
            f"Vector fields: {len(params.vector_fields)}\n"
            f"Scalar fields: {len(params.scalar_fields) if params.scalar_fields else 0}\n"
            f"Indexes created: {indexed_count}\n"
            f"Use this collection_name for subsequent operations."
        )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="open_collection",
    annotations={
        "title": "Open Existing Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def open_collection(params: OpenCollectionInput) -> str:
    """
    Open an existing Zvec collection from disk.

    This tool opens a previously created collection and caches it for subsequent
    operations. The collection must have been created with zvec_create_and_open_collection.

    Args:
        params (OpenCollectionInput): Validated input parameters containing:
            - path (str): Filesystem path of the existing collection
            - collection_name (str): Unique session identifier for caching
            - read_only (bool): Open in read-only mode (default: False)

    Returns:
        str: Success message with collection details or error message

    Examples:
        - Use when: "Open the collection at ./my_vectors"
        - Use when: "Load existing vector database from ./embeddings"
        - Don't use when: Collection doesn't exist (use create_and_open_collection)
    """
    try:
        option = zvec.CollectionOption(read_only=params.read_only)
        collection = zvec.open(path=params.path, option=option)

        # Cache the collection
        cache_collection(params.collection_name, collection)

        stats = collection.stats
        schema = collection.schema

        return (
            f"Successfully opened collection at path '{params.path}'.\n"
            f"Collection ID: {params.collection_name}\n"
            f"Collection name: {schema.name}\n"
            f"Document count: {stats.doc_count if hasattr(stats, 'doc_count') else 'N/A'}\n"
            f"Read-only mode: {params.read_only}\n"
            f"Use this collection_name for subsequent operations."
        )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="get_collection_info",
    annotations={
        "title": "Get Zvec Collection Information",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_collection_info(params: GetCollectionInfoInput) -> str:
    """
    Get detailed information about an opened collection.

    Retrieves schema definition, statistics, and configuration of a collection.

    Args:
        params (GetCollectionInfoInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - response_format (ResponseFormat): Output format ('markdown' or 'json')

    Returns:
        str: Collection information in the requested format or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        schema = collection.schema
        stats = collection.stats

        if params.response_format == ResponseFormat.JSON:
            info = {
                "path": collection.path,
                "name": schema.name,
                "vector_fields": [
                    {
                        "name": v.name,
                        "data_type": str(v.data_type),
                        "dimension": v.dimension,
                        "index_param": v.index_param.to_dict()
                        if v.index_param is not None
                        else None,
                    }
                    for v in schema.vectors
                ],
                "scalar_fields": [
                    {
                        "name": f.name,
                        "data_type": str(f.data_type),
                        "nullable": f.nullable,
                        "index_param": f.index_param.to_dict()
                        if getattr(f, "index_param", None) is not None
                        else None,
                    }
                    for f in schema.fields
                ],
                "stats": {
                    "doc_count": stats.doc_count if hasattr(stats, "doc_count") else None,
                },
            }
            return json.dumps(info, indent=2)

        # Markdown format
        lines = [
            f"# Collection Information: {schema.name}",
            "",
            f"**Path**: {collection.path}",
            f"**Document Count**: {stats.doc_count if hasattr(stats, 'doc_count') else 'N/A'}",
            "",
            "## Vector Fields",
            "",
        ]

        for v in schema.vectors:
            lines.append(f"- **{v.name}**")
            lines.append(f"  - Type: {v.data_type}")
            lines.append(f"  - Dimension: {v.dimension}")
            if v.index_param is not None:
                ip = v.index_param
                lines.append(f"  - Index: {ip.type} | {ip.to_dict()}")
            else:
                lines.append("  - Index: None")

        if schema.fields:
            lines.append("")
            lines.append("## Scalar Fields")
            lines.append("")
            for f in schema.fields:
                lines.append(f"- **{f.name}**")
                lines.append(f"  - Type: {f.data_type}")
                lines.append(f"  - Nullable: {f.nullable}")
                f_index = getattr(f, "index_param", None)
                if f_index is not None:
                    lines.append(f"  - Index: {f_index.type} | {f_index.to_dict()}")
                else:
                    lines.append("  - Index: None")

        return "\n".join(lines)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="destroy_collection",
    annotations={
        "title": "Destroy Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def destroy_collection(params: DestroyCollectionInput) -> str:
    """
    Permanently delete a collection from disk.

    WARNING: This operation is irreversible. All data will be permanently lost.

    Args:
        params (DestroyCollectionInput): Validated input parameters containing:
            - collection_name (str): Collection identifier

    Returns:
        str: Success confirmation or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        path = collection.path
        collection.destroy()

        # Remove from cache
        remove_collection_from_cache(params.collection_name)

        return f"Successfully destroyed collection at path '{path}'. All data has been permanently deleted."
    except Exception as e:
        return handle_error(e)


# ============================================================================
# MCP Tools - Document Operations
# ============================================================================


@mcp.tool(
    name="insert_documents",
    annotations={
        "title": "Insert Documents into Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def insert_documents(params: InsertDocumentsInput) -> str:
    """
    Insert new documents into a collection.

    Documents must have unique IDs and conform to the collection schema. This operation
    fails if a document with the same ID already exists.

    Args:
        params (InsertDocumentsInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - documents (List[DocumentInput]): Documents to insert

    Returns:
        str: Success message with insertion count or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        docs = []
        for doc_input in params.documents:
            doc = zvec.Doc(id=doc_input.id, vectors=doc_input.vectors, fields=doc_input.fields)
            docs.append(doc)

        statuses = collection.insert(docs)

        if isinstance(statuses, list):
            success_count = sum(1 for s in statuses if s.ok())
            return f"Successfully inserted {success_count}/{len(docs)} documents."
        else:
            return (
                "Successfully inserted 1 document."
                if statuses.ok()
                else f"Error: {statuses.message()}"
            )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="upsert_documents",
    annotations={
        "title": "Upsert Documents in Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def upsert_documents(params: UpsertDocumentsInput) -> str:
    """
    Insert new documents or update existing ones by ID.

    This operation inserts documents if they don't exist, or updates them if they do.

    Args:
        params (UpsertDocumentsInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - documents (List[DocumentInput]): Documents to upsert

    Returns:
        str: Success message with upsert count or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        docs = []
        for doc_input in params.documents:
            doc = zvec.Doc(id=doc_input.id, vectors=doc_input.vectors, fields=doc_input.fields)
            docs.append(doc)

        statuses = collection.upsert(docs)

        if isinstance(statuses, list):
            success_count = sum(1 for s in statuses if s.ok())
            return f"Successfully upserted {success_count}/{len(docs)} documents."
        else:
            return (
                "Successfully upserted 1 document."
                if statuses.ok()
                else f"Error: {statuses.message()}"
            )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="update_documents",
    annotations={
        "title": "Update Documents in Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def update_documents(params: UpdateDocumentsInput) -> str:
    """
    Update existing documents by ID.

    Only specified fields are updated; others remain unchanged. Documents must already exist.

    Args:
        params (UpdateDocumentsInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - documents (List[DocumentInput]): Documents with updates

    Returns:
        str: Success message with update count or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        docs = []
        for doc_input in params.documents:
            doc = zvec.Doc(id=doc_input.id, vectors=doc_input.vectors, fields=doc_input.fields)
            docs.append(doc)

        statuses = collection.update(docs)

        if isinstance(statuses, list):
            success_count = sum(1 for s in statuses if s.ok())
            return f"Successfully updated {success_count}/{len(docs)} documents."
        else:
            return (
                "Successfully updated 1 document."
                if statuses.ok()
                else f"Error: {statuses.message()}"
            )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="delete_documents",
    annotations={
        "title": "Delete Documents from Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def delete_documents(params: DeleteDocumentsInput) -> str:
    """
    Delete documents by their IDs.

    Args:
        params (DeleteDocumentsInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - document_ids (List[str]): Document IDs to delete

    Returns:
        str: Success message with deletion count or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        statuses = collection.delete(params.document_ids)

        if isinstance(statuses, list):
            success_count = sum(1 for s in statuses if s.ok())
            return f"Successfully deleted {success_count}/{len(params.document_ids)} documents."
        else:
            return (
                "Successfully deleted 1 document."
                if statuses.ok()
                else f"Error: {statuses.message()}"
            )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="fetch_documents",
    annotations={
        "title": "Fetch Documents from Zvec Collection",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def fetch_documents(params: FetchDocumentsInput) -> str:
    """
    Retrieve documents by their IDs.

    Args:
        params (FetchDocumentsInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - document_ids (List[str]): Document IDs to fetch
            - response_format (ResponseFormat): Output format

    Returns:
        str: Documents in the requested format or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        # fetch() returns dict[str, Doc]
        doc_dict = collection.fetch(params.document_ids)

        if not doc_dict:
            return "No documents found with the specified IDs."

        # Convert dict to list of Doc objects
        docs = list(doc_dict.values())

        return format_doc_list(docs, params.response_format)
    except Exception as e:
        return handle_error(e)


# ============================================================================
# MCP Tools - Vector Query
# ============================================================================


@mcp.tool(
    name="vector_query",
    annotations={
        "title": "Perform Vector Similarity Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def vector_query(params: VectorQueryInput) -> str:
    """
    Perform vector similarity search with optional filtering.

    This tool searches for the most similar documents based on vector similarity.
    Optionally apply scalar filters to restrict results to a subset of documents.

    Args:
        params (VectorQueryInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - field_name (str): Name of the vector field to query
            - vector (List[float]): Query vector
            - topk (int): Number of results to return (default: 10, max: 1000)
            - filter (Optional[str]): Filter expression (e.g., 'age > 25 AND city == "NYC"')
            - response_format (ResponseFormat): Output format

    Returns:
        str: Search results sorted by similarity score or error message

    Examples:
        - Use when: "Find the 10 most similar documents to this embedding"
        - Use when: "Search for similar vectors with age > 30"
        - Filter syntax: "field_name > value", "field == 'string'", combined with AND/OR
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        # Create vector query
        vector_query = zvec.VectorQuery(field_name=params.field_name, vector=params.vector)

        # Execute query
        if params.filter:
            results = collection.query(vector_query, filter=params.filter, topk=params.topk)
        else:
            results = collection.query(vector_query, topk=params.topk)

        if not results:
            return "No results found matching the query."

        return format_doc_list(results, params.response_format)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="multi_vector_query",
    annotations={
        "title": "Perform Multi-Vector Similarity Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def multi_vector_query(params: MultiVectorQueryInput) -> str:
    """
    Perform multi-vector similarity search with score fusion and re-ranking.

    This tool searches across multiple vector embeddings simultaneously and combines
    their results using a re-ranking strategy. This is useful when documents have
    multiple types of embeddings (e.g., dense + sparse, text + image).

    Args:
        params (MultiVectorQueryInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - vectors (List[MultiVectorQuerySpec]): List of vector queries (min 2)
            - topk (int): Candidates to retrieve from each vector field (default: 10)
            - topn (int): Final documents to return after re-ranking (default: 5)
            - reranker_type (str): 'weighted' or 'rrf' (default: weighted)
            - weights (Optional[Dict[str, float]]): Field weights for weighted re-ranker
            - rank_constant (int): RRF rank constant (default: 60)
            - metric_type (str): Metric for weighted re-ranker (default: IP)
            - filter (Optional[str]): Filter expression
            - response_format (str): Output format

    Returns:
        str: Re-ranked search results or error message

    Examples:
        - Use when: "Search using both dense and sparse embeddings"
        - Use when: "Combine text and image similarity for multi-modal search"

    Re-ranking Strategies:
        - Weighted: Combines normalized scores with custom weights per field
          Best when scores are comparable and you know field importance
        - RRF (Reciprocal Rank Fusion): Combines based on rank positions only
          Best when scores use different metrics/scales or prefer tuning-free approach
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        # Build list of VectorQuery objects
        vector_queries = []
        for vq in params.vectors:
            vector_queries.append(zvec.VectorQuery(field_name=vq.field_name, vector=vq.vector))

        # Create re-ranker based on type
        if params.reranker_type == RerankerType.WEIGHTED:
            # Weighted re-ranker: combines normalized scores with custom weights
            metric = get_zvec_metric_type(params.metric_type.value)

            # Use provided weights or default to equal weights
            if params.weights:
                weights = params.weights
            else:
                # Default: equal weights for all fields
                weights = {vq.field_name: 1.0 for vq in params.vectors}

            reranker = zvec.WeightedReRanker(topn=params.topn, metric=metric, weights=weights)
        else:  # RRF
            # RRF re-ranker: combines based on rank positions only
            reranker = zvec.RrfReRanker(topn=params.topn, rank_constant=params.rank_constant)

        # Execute multi-vector query
        if params.filter:
            results = collection.query(
                vectors=vector_queries, topk=params.topk, reranker=reranker, filter=params.filter
            )
        else:
            results = collection.query(vectors=vector_queries, topk=params.topk, reranker=reranker)

        if not results:
            return "No results found matching the query."

        return format_doc_list(results, params.response_format)
    except Exception as e:
        return handle_error(e)


# ============================================================================
# MCP Tools - Index Management
# ============================================================================


@mcp.tool(
    name="create_index",
    annotations={
        "title": "Create Index on Zvec Collection Field",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def create_index(params: CreateIndexInput) -> str:
    """
    Create an index on a field to accelerate queries.

    Use HnswIndexParamInput / FlatIndexParamInput / IVFIndexParamInput for vector fields,
    and InvertIndexParamInput for scalar fields.

    Args:
        params (CreateIndexInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - field_name (str): Name of the field to index
            - index_param: One of HnswIndexParamInput, FlatIndexParamInput, IVFIndexParamInput,
              or InvertIndexParamInput (use 'type' field as discriminator)

    Returns:
        str: Success message or error message

    Examples:
        - Use when: "Create an HNSW index on the embedding field"
        - Use when: "Build an inverted index on the category scalar field"
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        zvec_index_param = _build_zvec_index_param(params.index_param)
        collection.create_index(field_name=params.field_name, index_param=zvec_index_param)

        return (
            f"Successfully created {params.index_param.type} index on field '{params.field_name}'."
        )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="drop_index",
    annotations={
        "title": "Drop Index from Zvec Collection Field",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def drop_index(params: DropIndexInput) -> str:
    """
    Remove the index from a field.

    Args:
        params (DropIndexInput): Validated input parameters containing:
            - collection_name (str): Collection identifier
            - field_name (str): Name of the indexed field

    Returns:
        str: Success message or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        collection.drop_index(field_name=params.field_name)

        return f"Successfully dropped index from field '{params.field_name}'."
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="optimize_collection",
    annotations={
        "title": "Optimize Zvec Collection",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def optimize_collection(params: OptimizeCollectionInput) -> str:
    """
    Optimize the collection (e.g., merge segments, rebuild index).

    This operation improves query performance and reduces storage overhead.

    Args:
        params (OptimizeCollectionInput): Validated input parameters containing:
            - collection_name (str): Collection identifier

    Returns:
        str: Success message or error message
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        collection.optimize()

        return "Successfully optimized collection. Query performance should be improved."
    except Exception as e:
        return handle_error(e)


# ============================================================================
# MCP Tools - AI Extension (OpenAI Embedding)
# ============================================================================


def _build_openai_embedding(api_key, base_url, model, dimension):
    """Instantiate OpenAIDenseEmbedding with resolved credentials."""
    import os

    from zvec.extension import OpenAIDenseEmbedding

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None

    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key required. Provide via api_key parameter or OPENAI_API_KEY env var."
        )

    kwargs = dict(api_key=resolved_api_key, model=model, dimension=dimension)
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return OpenAIDenseEmbedding(**kwargs)


def _build_openai_embedding_from_env(dimension: int):
    """Instantiate OpenAIDenseEmbedding using MCP server environment configuration.

    Reads:
      OPENAI_API_KEY      — required
      OPENAI_BASE_URL     — optional, for compatible endpoints
      OPENAI_EMBEDDING_MODEL — optional, defaults to text-embedding-3-small
    """
    import os

    from zvec.extension import OpenAIDenseEmbedding

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Configure it before starting the MCP server."
        )

    kwargs = dict(api_key=api_key, model=model, dimension=dimension)
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIDenseEmbedding(**kwargs)


def _get_vector_field_dimension(collection, field_name: str) -> int:
    """Retrieve the declared dimension of a vector field from the collection schema."""
    schema = collection.schema
    for v in schema.vectors:
        if v.name == field_name:
            return v.dimension
    raise ValueError(
        f"Vector field '{field_name}' not found in collection schema. "
        f"Available fields: {[v.name for v in schema.vectors]}"
    )


@mcp.tool(
    name="generate_dense_embedding",
    annotations={
        "title": "Generate Dense Embedding using OpenAI",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def generate_dense_embedding(params: GenerateDenseEmbeddingInput) -> str:
    """
    Generate a dense embedding vector for a piece of text using OpenAIDenseEmbedding.

    Converts text into a fixed-length dense vector via the OpenAI (or compatible)
    embedding API. The resulting vector can be directly used for similarity search.

    Args:
        params (GenerateDenseEmbeddingInput):
            - text: Text to embed
            - api_key: OpenAI API key (or OPENAI_API_KEY env var)
            - base_url: Custom API base URL for OpenAI-compatible endpoints
            - model: Embedding model name (default: text-embedding-3-small)
            - dimension: Output vector dimension (default: 1536)

    Returns:
        str: JSON with text preview, model, dimension, and the dense vector
    """
    try:
        embedding_func = _build_openai_embedding(
            params.api_key, params.base_url, params.model, params.dimension
        )
        vector = embedding_func.embed(params.text)
        return json.dumps(
            {
                "text": params.text[:100] + "..." if len(params.text) > 100 else params.text,
                "model": params.model,
                "dimension": params.dimension,
                "vector": list(vector),
            },
            indent=2,
        )
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except ImportError:
        return json.dumps(
            {"error": "zvec.extension not available. Install with: pip install openai"},
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": handle_error(e)}, indent=2)


@mcp.tool(
    name="embedding_write",
    annotations={
        "title": "Write Documents with Auto Embedding",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def embedding_write(params: EmbeddingWriteInput) -> str:
    """
    Embed text documents and upsert them into a Zvec collection.

    Converts each document's text field to a dense vector using OpenAIDenseEmbedding,
    then upserts all documents into the specified collection. This is the high-level
    write interface: supply plain text, get vectors stored automatically.

    OpenAI connection is read from environment variables:
      OPENAI_API_KEY, OPENAI_BASE_URL (optional), OPENAI_EMBEDDING_MODEL (optional).
    The embedding dimension is inferred from the collection schema automatically.

    Args:
        params (EmbeddingWriteInput):
            - collection_name: Target collection
            - field_name: Vector field to populate
            - documents: List of {id, text, fields} — text is auto-embedded

    Returns:
        str: Success message with upsert count, or error
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        dimension = _get_vector_field_dimension(collection, params.field_name)
        embedding_func = _build_openai_embedding_from_env(dimension)

        docs = []
        for doc_input in params.documents:
            vector = embedding_func.embed(doc_input.text)
            docs.append(
                zvec.Doc(
                    id=doc_input.id,
                    vectors={params.field_name: list(vector)},
                    fields=doc_input.fields,
                )
            )

        statuses = collection.upsert(docs)

        if isinstance(statuses, list):
            success_count = sum(1 for s in statuses if s.ok())
            return f"Successfully upserted {success_count}/{len(docs)} documents into '{params.collection_name}'."
        else:
            return (
                f"Successfully upserted 1 document into '{params.collection_name}'."
                if statuses.ok()
                else f"Error: {statuses.message()}"
            )
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except ImportError:
        return json.dumps(
            {"error": "zvec.extension not available. Install with: pip install openai"},
            indent=2,
        )
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="embedding_search",
    annotations={
        "title": "Semantic Search with Auto Embedding",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def embedding_search(params: EmbeddingSearchInput) -> str:
    """
    Convert a natural language query to a vector and perform similarity search.

    Embeds query_text using OpenAIDenseEmbedding, then runs a vector similarity
    search against the specified field in the collection. This is the high-level
    search interface: supply a natural language query, get ranked results.

    OpenAI connection is read from environment variables:
      OPENAI_API_KEY, OPENAI_BASE_URL (optional), OPENAI_EMBEDDING_MODEL (optional).
    The embedding dimension is inferred from the collection schema automatically.

    Args:
        params (EmbeddingSearchInput):
            - collection_name: Target collection
            - field_name: Vector field to search
            - query_text: Natural language query to embed and search with
            - topk: Number of results (default: 10)
            - filter: Optional scalar filter expression
            - response_format: Output format ('markdown' or 'json')

    Returns:
        str: Search results sorted by similarity, or error
    """
    try:
        collection = get_collection(params.collection_name)
        if collection is None:
            return f"Error: Collection '{params.collection_name}' not found. Please open it first."

        dimension = _get_vector_field_dimension(collection, params.field_name)
        embedding_func = _build_openai_embedding_from_env(dimension)

        query_vector = embedding_func.embed(params.query_text)

        vq = zvec.VectorQuery(field_name=params.field_name, vector=list(query_vector))

        if params.filter:
            results = collection.query(vq, filter=params.filter, topk=params.topk)
        else:
            results = collection.query(vq, topk=params.topk)

        if not results:
            return "No results found matching the query."

        return format_doc_list(results, params.response_format)
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except ImportError:
        return json.dumps(
            {"error": "zvec.extension not available. Install with: pip install openai"},
            indent=2,
        )
    except Exception as e:
        return handle_error(e)
