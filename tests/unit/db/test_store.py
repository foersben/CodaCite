"""Tests for the SurrealDB store implementations.

This module validates the low-level database interactions and SurrealQL generation
within the Infrastructure layer (Persistence).
"""

from datetime import datetime
from typing import Any

import pytest
from surrealdb import RecordID

from app.db.store import SurrealDocumentStore, SurrealGraphStore
from app.models.models import Chunk, Community, Document, Edge, Node, Notebook


@pytest.fixture
def mock_db(mocker: Any) -> Any:
    """Fixture providing a mocked database instance.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked database instance.
    """
    return mocker.AsyncMock()


@pytest.mark.asyncio
async def test_save_document(mock_db: Any) -> None:
    """Tests that saving a document generates the correct SurrealQL query.

    Given:
        A Document instance.
    When:
        save_document is called.
    Then:
        The database should receive an UPDATE query with CONTENT.
    """
    store = SurrealDocumentStore(mock_db)
    doc = Document(id="doc1", filename="test.pdf", metadata={"author": "Alice"})

    await store.save_document(doc)

    mock_db.query.assert_called_once()
    args, _ = mock_db.query.call_args
    sql = args[0]
    assert "UPSERT $id CONTENT" in sql


@pytest.mark.asyncio
async def test_update_document_status(mock_db: Any) -> None:
    """Tests updating a document's status.

    Given:
        A document ID and a new status.
    When:
        update_document_status is called.
    Then:
        The database should receive an UPDATE query targeting the status field.
    """
    store = SurrealDocumentStore(mock_db)
    await store.update_document_status("doc1", "processed")
    mock_db.query.assert_called_once()
    assert (
        "UPDATE type::record('document', $id) SET status = $status" in mock_db.query.call_args[0][0]
    )


@pytest.mark.asyncio
async def test_save_document_with_summary(mock_db: Any) -> None:
    """Tests that updating a document with a summary generates the correct SurrealQL.

    Given:
        A document ID and a pre-computed summary.
    When:
        save_document_with_summary is called.
    Then:
        The database should receive an UPDATE query with type::record.
    """
    store = SurrealDocumentStore(mock_db)
    await store.save_document_with_summary("doc1", "This is a summary")

    mock_db.query.assert_called_once()
    sql = mock_db.query.call_args[0][0]
    params = mock_db.query.call_args[0][1]

    assert "UPDATE type::record('document', $doc_id)" in sql
    assert "global_summary: $summary" in sql
    assert params["doc_id"] == "doc1"
    assert params["summary"] == "This is a summary"


@pytest.mark.asyncio
async def test_get_all_documents(mock_db: Any) -> None:
    """Tests retrieving all documents from the store.

    Given:
        A database containing documents.
    When:
        get_all_documents is called.
    Then:
        It should return a list of Document instances.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        [{"id": RecordID("document", "doc1"), "filename": "f1.pdf", "status": "active"}]
    ]
    docs = await store.get_all_documents()
    assert len(docs) == 1
    assert docs[0].id == "doc1"


@pytest.mark.asyncio
async def test_get_document_success(mock_db: Any) -> None:
    """Tests retrieving a specific document by ID.

    Given:
        A document ID that exists in the database.
    When:
        get_document is called.
    Then:
        It should return the correct Document instance.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        [{"id": RecordID("document", "doc1"), "filename": "test.pdf", "status": "active"}]
    ]
    doc = await store.get_document("doc1")
    assert doc is not None
    assert doc.id == "doc1"
    assert doc.filename == "test.pdf"


@pytest.mark.asyncio
async def test_get_document_missing(mock_db: Any) -> None:
    """Tests retrieving a document that does not exist.

    Given:
        A document ID not in the database.
    When:
        get_document is called.
    Then:
        It should return None.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [[]]
    doc = await store.get_document("missing")
    assert doc is None


@pytest.mark.asyncio
async def test_get_notebook_documents(mock_db: Any) -> None:
    """Tests retrieving documents associated with a specific notebook.

    Given:
        A notebook ID.
    When:
        get_notebook_documents is called.
    Then:
        The database should receive a query with a BELONGS_TO relationship filter.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        [{"id": RecordID("document", "doc1"), "filename": "f1.pdf", "status": "active"}]
    ]
    docs = await store.get_notebook_documents("nb1")
    assert len(docs) == 1
    assert (
        "SELECT * FROM document WHERE ->belongs_to->notebook.id CONTAINS $notebook"
        in mock_db.query.call_args[0][0]
    )


@pytest.mark.asyncio
async def test_notebook_management(mock_db: Any) -> None:
    """Tests notebook CRUD operations and relationship management.

    Given:
        A notebook and associated document IDs.
    When:
        CRUD operations and relationship updates are performed.
    Then:
        The database should receive the corresponding SurrealQL queries (UPDATE, RELATE, DELETE).
    """
    store = SurrealDocumentStore(mock_db)

    # 1. Save Notebook
    nb = Notebook(id="nb1", title="My Notebook", created_at=datetime.now().isoformat())
    await store.save_notebook(nb)
    update_call = [
        c
        for c in mock_db.query.call_args_list
        if "UPSERT type::record('notebook', $id) CONTENT $data;" in c[0][0]
    ][0]
    assert update_call[0][1]["data"]["title"] == "My Notebook"

    # 2. Add Document to Notebook
    await store.add_document_to_notebook("doc1", "nb1")
    relate_call = [c for c in mock_db.query.call_args_list if "RELATE" in c[0][0]][0]
    assert "RELATE $doc -> belongs_to -> $notebook" in relate_call[0][0]

    # 3. List Notebooks
    mock_db.query.return_value = [[{"id": RecordID("notebook", "nb1"), "title": "NB1"}]]
    notebooks = await store.get_all_notebooks()
    assert len(notebooks) == 1
    assert notebooks[0].id == "nb1"

    # 4. Remove Document from Notebook
    await store.remove_document_from_notebook("doc1", "nb1")
    delete_rel_call = [c for c in mock_db.query.call_args_list if "DELETE belongs_to" in c[0][0]][0]
    assert "DELETE belongs_to WHERE in = $doc AND out = $notebook" in delete_rel_call[0][0]

    # 5. Delete Notebook
    await store.delete_notebook("nb1")
    del_nb_call = [c for c in mock_db.query.call_args_list if "DELETE $id" in c[0][0]][0]
    assert "DELETE $id" in del_nb_call[0][0]


@pytest.mark.asyncio
async def test_delete_document(mock_db: Any) -> None:
    """Tests deleting a document and its cascading effects.

    Given:
        A document ID.
    When:
        delete_document is called.
    Then:
        The database should receive a SELECT query for file_path followed by a TRANSACTION.
    """
    store = SurrealDocumentStore(mock_db)

    # Mock file_path lookup and deletion transaction
    mock_db.query.side_effect = [
        [[{"file_path": "/data/blobs/test.pdf"}]],  # SELECT file_path
        [],  # BEGIN TRANSACTION...
    ]

    success = await store.delete_document("doc1")

    assert success is True
    assert mock_db.query.call_count == 2

    # Check the first call (SELECT)
    args, _ = mock_db.query.call_args_list[0]
    assert "SELECT file_path FROM type::record('document', $id)" in args[0]

    # Check the second call (TRANSACTION)
    args, _ = mock_db.query.call_args_list[1]
    assert "BEGIN TRANSACTION" in args[0]


@pytest.mark.asyncio
async def test_save_chunks(mock_db: Any) -> None:
    """Tests that saving chunks generates the correct SurrealQL queries.

    Given:
        A list of Chunk instances.
    When:
        save_chunks is called.
    Then:
        The database should receive multiple queries for the batch update.
    """
    store = SurrealDocumentStore(mock_db)
    chunks = [
        Chunk(id="c1", document_id="doc1", text="Chunk 1 text", index=0, embedding=[0.1, 0.2]),
    ]
    await store.save_chunks(chunks)
    assert mock_db.query.call_count >= 2


@pytest.mark.asyncio
async def test_graph_store_queries(mock_db: Any) -> None:
    """Tests basic GraphStore queries for nodes, edges, and communities.

    Given:
        A functional graph store.
    When:
        Nodes, edges, or communities are queried or saved.
    Then:
        It should return the expected domain objects or execute the correct SurrealQL.
    """
    store = SurrealGraphStore(mock_db)

    # 1. Get all nodes
    mock_db.query.return_value = [
        [{"id": RecordID("entity", "n1"), "label": "L", "name": "N", "source_chunk_ids": ["c1"]}]
    ]
    nodes = await store.get_all_nodes()
    assert len(nodes) == 1
    assert nodes[0].source_chunk_ids == ["c1"]

    # 2. Get all edges
    mock_db.query.return_value = [
        [
            {
                "id": RecordID("rel", "r1"),
                "in": RecordID("entity", "n2"),
                "out": RecordID("entity", "n1"),
                "source_id": RecordID("entity", "n1"),
                "target_id": RecordID("entity", "n2"),
                "relation": "K",
                "source_chunk_ids": ["c1"],
            }
        ]
    ]
    edges = await store.get_all_edges()
    assert len(edges) == 1
    assert edges[0].id == "rel:r1"
    assert edges[0].source_id == "n1"
    assert edges[0].target_id == "n2"
    assert edges[0].source_chunk_ids == ["c1"]

    # 3. Save community
    community = Community(id="c1", summary="S", node_ids=["n1", "n2"])
    await store.save_community(community)
    mock_db.query.assert_called()
    assert "UPSERT type::record('community', $id)" in mock_db.query.call_args[0][0]


@pytest.mark.asyncio
async def test_save_nodes_edges(mock_db: Any) -> None:
    """Tests saving nodes and edges to the graph store.

    Given:
        A list of nodes or edges.
    When:
        save_nodes or save_edges is called.
    Then:
        The database should receive the corresponding update queries.
    """
    store = SurrealGraphStore(mock_db)
    nodes = [Node(id="n1", label="PERSON", name="Alice")]
    await store.save_nodes(nodes)
    assert mock_db.query.call_count >= 1

    mock_db.query.reset_mock()
    edges = [Edge(source_id="n1", target_id="n2", relation="KNOWS")]
    await store.save_edges(edges)
    mock_db.query.assert_called_once()


@pytest.mark.asyncio
async def test_search_chunks_hybrid_unfiltered(mock_db: Any) -> None:
    """Tests hybrid (BM25 + HNSW) search without notebook filters.

    Given:
        An embedding vector and a query text.
    When:
        search_chunks is called with query_text and no notebook filter.
    Then:
        The database should receive a query using both the BM25 ``@1@`` operator
        and the HNSW ``<|K|>`` operator with a combined ``hybrid_score``.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        [
            {
                "id": RecordID("chunk", "c1"),
                "text": "T",
                "index": 0,
                "embedding": [0.1],
                "hybrid_score": 0.9,
            }
        ]
    ]
    chunks = await store.search_chunks([0.1], query_text="machine learning", top_k=1)
    assert len(chunks) == 1
    sql = mock_db.query.call_args[0][0]
    assert "@1@ $query_text" in sql
    assert "embedding <|1,150|>" in sql
    assert "search::score(1)" in sql
    assert "vector::similarity::cosine" in sql
    assert "hybrid_score" in sql
    assert "ORDER BY hybrid_score DESC" in sql
    # Verify params
    params = mock_db.query.call_args[0][1]
    assert params["query_text"] == "machine learning"
    assert params["alpha"] == 0.5  # default


@pytest.mark.asyncio
async def test_search_chunks_hybrid_alpha_weighting(mock_db: Any) -> None:
    """Tests that a custom alpha value is correctly forwarded to the query.

    Given:
        An embedding vector, query text, and a custom alpha of 0.8.
    When:
        search_chunks is called.
    Then:
        The query params should contain alpha=0.8.
    """
    store = SurrealDocumentStore(mock_db)
    # Mock both the count diagnostic query and the actual hybrid search
    mock_db.query.side_effect = [
        [[{"count": 10}]],  # Result for count query
        [
            [{"id": "chunk:1", "text": "...", "index": 0, "document_id": "doc:1"}]
        ],  # Result for search
    ]
    await store.search_chunks([0.1], query_text="deep learning", alpha=0.8, top_k=3)

    # The actual search is the second call (index 1)
    # args is at [0], kwargs is at [1]
    # search query is query(sql, params)
    params = mock_db.query.call_args_list[1][0][1]
    assert params["alpha"] == 0.8


@pytest.mark.asyncio
async def test_search_chunks_hybrid_filtered(mock_db: Any) -> None:
    """Tests hybrid search with active notebook filtering.

    Given:
        An embedding vector, a query text, and active notebook IDs.
    When:
        search_chunks is called.
    Then:
        The database should receive a query using BM25, HNSW, and a
        CONTAINSANY notebook membership filter.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        [
            {
                "id": RecordID("chunk", "c1"),
                "text": "T",
                "index": 0,
                "embedding": [0.1],
                "hybrid_score": 0.7,
            }
        ]
    ]
    chunks = await store.search_chunks(
        [0.1], query_text="neural networks", top_k=1, active_notebook_ids=["nb1"]
    )
    assert len(chunks) == 1
    sql = mock_db.query.call_args[0][0]
    assert "@1@ $query_text" in sql
    assert "CONTAINSANY $notebook_ids" in sql
    assert "hybrid_score" in sql


@pytest.mark.asyncio
async def test_search_chunks_vector_only_fallback(mock_db: Any) -> None:
    """Tests that omitting query_text falls back to pure HNSW vector search.

    Given:
        An embedding vector with no query text.
    When:
        search_chunks is called without query_text.
    Then:
        The database should receive a pure HNSW query without BM25 operators.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        [{"id": "chunk:c1", "text": "T", "index": 0, "embedding": [0.1], "document_id": "doc:1"}]
    ]
    chunks = await store.search_chunks([0.1], top_k=1)
    assert len(chunks) == 1
    sql = mock_db.query.call_args[0][0]
    assert "@1@" not in sql
    assert "search::score" not in sql
    assert "embedding <|1,150|>" in sql


@pytest.mark.asyncio
async def test_search_chunks_vector_only_filtered(mock_db: Any) -> None:
    """Tests pure HNSW vector search filtered by notebook IDs (no query text).

    Given:
        An embedding vector and active notebook IDs but no query text.
    When:
        search_chunks is called.
    Then:
        The database should receive a vector search query with notebook membership filters
        and no BM25 operators.
    """
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        [{"id": "chunk:c1", "text": "T", "index": 0, "embedding": [0.1], "document_id": "doc:1"}]
    ]
    chunks = await store.search_chunks([0.1], top_k=1, active_notebook_ids=["nb1"])
    assert len(chunks) == 1
    sql = mock_db.query.call_args[0][0]
    assert "@1@" not in sql
    assert "CONTAINSANY $notebook_ids" in sql
    assert "embedding <|1,150|>" in sql


@pytest.mark.asyncio
async def test_initialize_schema(mock_db: Any) -> None:
    """Tests schema initialization for both document and graph stores.

    Given:
        A database connection.
    When:
        initialize_schema is called.
    Then:
        The database should receive DEFINE ANALYZER, DEFINE INDEX for BM25,
        DEFINE INDEX for HNSW, and entity embedding index queries.
    """
    doc_store = SurrealDocumentStore(mock_db)
    await doc_store.initialize_schema()
    all_calls = [c[0][0] for c in mock_db.query.call_args_list]
    assert any("DEFINE ANALYZER OVERWRITE standard" in s for s in all_calls)
    assert any("DEFINE INDEX OVERWRITE chunk_text_idx" in s for s in all_calls)
    assert any("DEFINE INDEX OVERWRITE chunk_embedding_idx" in s for s in all_calls)

    mock_db.query.reset_mock()
    graph_store = SurrealGraphStore(mock_db)
    await graph_store.initialize_schema()
    # Delegation now sends multiple blocks; check all of them.
    all_graph_calls = [c[0][0] for c in mock_db.query.call_args_list]
    assert any("DEFINE INDEX OVERWRITE entity_embedding_idx" in s for s in all_graph_calls)


@pytest.mark.asyncio
async def test_extract_rows_edge_cases() -> None:
    """Tests the _extract_rows utility with various malformed or empty inputs.

    Given:
        Various input types (None, empty list, malformed dict).
    When:
        _extract_rows is called.
    Then:
        It should return a normalized list or an empty list as appropriate.
    """
    from app.db.store import _extract_rows

    assert _extract_rows(None) == []
    assert _extract_rows([]) == []
    assert _extract_rows([{}]) == [{}]
    assert _extract_rows({"result": "not a list"}) == [{"result": "not a list"}]
    assert _extract_rows([{"result": {"id": "1"}}]) == [{"result": {"id": "1"}}]
    assert _extract_rows(123) == []  # Not a list or dict
    assert _extract_rows({"result": [{"id": "1"}, 123]}) == [{"result": [{"id": "1"}, 123]}]


@pytest.mark.asyncio
async def test_traverse_logic(mock_db: Any) -> None:
    """Tests complex graph traversal logic including multi-depth and batch edge fetching.

    Given:
        A seed node ID and a multi-level graph structure mocked in the database.
    When:
        The traverse method is called with depth > 1.
    Then:
        It should correctly discover nodes and edges through recursive traversal
        using SurrealDB 3.x batch query patterns (INSIDE for edges, $ids for nodes).
    """
    store = SurrealGraphStore(mock_db)

    # Mock side effect to handle different batch queries in the traversal loop
    async def side_effect(query: str, vars: dict[str, Any] | None = None) -> list[Any]:
        if vars is None:
            vars = {}
        # 1. Edge traversal query
        if "relation WHERE in INSIDE $ids OR out INSIDE $ids" in query:
            ids = vars.get("ids", [])
            raw_ids = [str(rid.id) if hasattr(rid, "id") else str(rid) for rid in ids]

            if "n1" in raw_ids:
                return [
                    [
                        {
                            "id": RecordID("relation", "e1"),
                            "in": RecordID("entity", "n2"),
                            "out": RecordID("entity", "n1"),
                            "source_id": RecordID("entity", "n1"),
                            "target_id": RecordID("entity", "n2"),
                            "relation": "KNOWS",
                            "source_chunk_ids": ["c1"],
                            "weight": 0.8,
                        },
                        {
                            "id": RecordID("relation", "e2"),
                            "in": RecordID("entity", "n3"),
                            "out": RecordID("entity", "n1"),
                            "source_id": RecordID("entity", "n1"),
                            "target_id": RecordID("entity", "n3"),
                            "relation": "KNOWS",
                            "source_chunk_ids": ["c2"],
                            "weight": 0.5,
                        },
                    ]
                ]
            elif "n2" in raw_ids or "n3" in raw_ids:
                return [
                    [
                        {
                            "id": RecordID("relation", "e3"),
                            "in": RecordID("entity", "n4"),
                            "out": RecordID("entity", "n2"),
                            "source_id": RecordID("entity", "n2"),
                            "target_id": RecordID("entity", "n4"),
                            "relation": "WORKS_AT",
                            "source_chunk_ids": ["c3"],
                            "weight": 0.9,
                        },
                    ]
                ]
            return [[]]

        # 2. Final batch node fetch query
        elif "SELECT * FROM $ids" in query:
            return [
                [
                    {
                        "id": RecordID("entity", "n1"),
                        "label": "PERSON",
                        "name": "n1",
                        "source_chunk_ids": ["c1"],
                    },
                    {"id": RecordID("entity", "n2"), "label": "PERSON", "name": "n2"},
                    {"id": RecordID("entity", "n3"), "label": "PERSON", "name": "n3"},
                    {"id": RecordID("entity", "n4"), "label": "PERSON", "name": "n4"},
                ]
            ]

        return [[]]

    mock_db.query.side_effect = side_effect

    # Traverse from n1 with depth 2
    nodes, edges = await store.traverse(seed_node_ids=["n1"], depth=2)

    assert len(nodes) == 4
    assert len(edges) == 3

    # Verify edge contents
    e1 = [e for e in edges if e.source_id == "n1" and e.target_id == "n2"][0]
    assert e1.id == "relation:e1"
    assert e1.relation == "KNOWS"
    assert e1.weight == 0.8

    e3 = [e for e in edges if e.relation == "WORKS_AT"][0]
    assert e3.source_id == "n2"
    assert e3.target_id == "n4"
    assert e3.source_chunk_ids == ["c3"]


@pytest.mark.asyncio
async def test_traverse_with_umlauts(mock_db: Any) -> None:
    """Tests that traversal correctly handles IDs with special characters."""
    store = SurrealGraphStore(mock_db)
    from surrealdb import RecordID

    # Mock query response for the single node fetch (depth=0)
    mock_db.query.return_value = [
        [
            {
                "id": RecordID("entity", "benjamin_förster"),
                "label": "Person",
                "name": "Benjamin Förster",
                "description": "Expert",
            }
        ]
    ]

    nodes, edges = await store.traverse(seed_node_ids=["benjamin_förster"], depth=0)

    assert len(nodes) == 1
    assert nodes[0].id == "benjamin_förster"
    assert nodes[0].name == "Benjamin Förster"

    # Verify the final node query format
    final_query_call = mock_db.query.call_args_list[-1]
    query_str = final_query_call[0][0]
    query_vars = final_query_call[0][1]

    assert "SELECT * FROM $ids" in query_str
    assert any(str(rid.id) == "benjamin_förster" for rid in query_vars["ids"])
