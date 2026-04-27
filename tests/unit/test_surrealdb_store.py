"""Tests for the SurrealDB store implementations.

This module validates the low-level database interactions and SurrealQL generation
within the Infrastructure layer (Persistence).
"""

from datetime import datetime
from typing import Any

import pytest

from app.domain.models import Chunk, Community, Document, Edge, Node, Notebook
from app.infrastructure.database.store import SurrealDocumentStore, SurrealGraphStore


@pytest.fixture
def mock_db(mocker) -> Any:
    """Mock database instance."""
    return mocker.AsyncMock()


@pytest.mark.asyncio
async def test_save_document(mock_db: Any) -> None:
    """Test saving a document generates the correct SurrealQL query."""
    store = SurrealDocumentStore(mock_db)
    doc = Document(id="doc1", filename="test.pdf", metadata={"author": "Alice"})

    await store.save_document(doc)

    mock_db.query.assert_called_once()
    args, _ = mock_db.query.call_args
    sql = args[0]
    assert "UPDATE $id CONTENT" in sql


@pytest.mark.asyncio
async def test_update_document_status(mock_db: Any) -> None:
    """Test updating document status."""
    store = SurrealDocumentStore(mock_db)
    await store.update_document_status("doc1", "processed")
    mock_db.query.assert_called_once()
    assert (
        "UPDATE type::thing('document', $id) SET status = $status" in mock_db.query.call_args[0][0]
    )


@pytest.mark.asyncio
async def test_get_all_documents(mock_db: Any) -> None:
    """Test retrieving all documents."""
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        {"result": [{"id": "document:doc1", "filename": "f1.pdf", "status": "active"}]}
    ]
    docs = await store.get_all_documents()
    assert len(docs) == 1
    assert docs[0].id == "doc1"


@pytest.mark.asyncio
async def test_get_notebook_documents(mock_db: Any) -> None:
    """Test retrieving documents for a notebook."""
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        {"result": [{"id": "document:doc1", "filename": "f1.pdf", "status": "active"}]}
    ]
    docs = await store.get_notebook_documents("nb1")
    assert len(docs) == 1
    assert (
        "SELECT * FROM document WHERE ->belongs_to->notebook.id CONTAINS $notebook"
        in mock_db.query.call_args[0][0]
    )


@pytest.mark.asyncio
async def test_notebook_management(mock_db: Any) -> None:
    """Test notebook CRUD operations and document relations."""
    store = SurrealDocumentStore(mock_db)

    # 1. Save Notebook
    nb = Notebook(id="nb1", title="My Notebook", created_at=datetime.now().isoformat())
    await store.save_notebook(nb)
    update_call = [
        c for c in mock_db.query.call_args_list if "UPDATE type::thing('notebook', $id)" in c[0][0]
    ][0]
    assert update_call[0][1]["title"] == "My Notebook"

    # 2. Add Document to Notebook
    await store.add_document_to_notebook("doc1", "nb1")
    relate_call = [c for c in mock_db.query.call_args_list if "RELATE" in c[0][0]][0]
    assert "RELATE $doc -> belongs_to -> $notebook" in relate_call[0][0]

    # 3. List Notebooks
    mock_db.query.return_value = [{"result": [{"id": "notebook:nb1", "title": "NB1"}]}]
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
    """Test deleting a document and its cascading effects."""
    store = SurrealDocumentStore(mock_db)
    await store.delete_document("doc1")
    mock_db.query.assert_called_once()
    args, _ = mock_db.query.call_args
    assert "BEGIN TRANSACTION" in args[0]


@pytest.mark.asyncio
async def test_save_chunks(mock_db: Any) -> None:
    """Test saving chunks generates the correct SurrealQL queries."""
    store = SurrealDocumentStore(mock_db)
    chunks = [
        Chunk(id="c1", document_id="doc1", text="Chunk 1 text", index=0, embedding=[0.1, 0.2]),
    ]
    await store.save_chunks(chunks)
    assert mock_db.query.call_count >= 2


@pytest.mark.asyncio
async def test_graph_store_queries(mock_db: Any) -> None:
    """Test basic GraphStore queries."""
    store = SurrealGraphStore(mock_db)

    # 1. Get all nodes
    mock_db.query.return_value = [{"result": [{"id": "entity:n1", "label": "L", "name": "N"}]}]
    nodes = await store.get_all_nodes()
    assert len(nodes) == 1

    # 2. Get all edges
    mock_db.query.return_value = [
        {"result": [{"id": "rel:r1", "in": "n1", "out": "n2", "relation": "K"}]}
    ]
    edges = await store.get_all_edges()
    assert len(edges) == 1

    # 3. Save community
    community = Community(id="c1", summary="S", node_ids=["n1", "n2"])
    await store.save_community(community)
    mock_db.query.assert_called()
    assert "UPDATE type::thing('community', $id)" in mock_db.query.call_args[0][0]


@pytest.mark.asyncio
async def test_save_nodes_edges(mock_db: Any) -> None:
    """Test saving nodes and edges."""
    store = SurrealGraphStore(mock_db)
    nodes = [Node(id="n1", label="PERSON", name="Alice")]
    await store.save_nodes(nodes)
    assert mock_db.query.call_count >= 1

    mock_db.query.reset_mock()
    edges = [Edge(source_id="n1", target_id="n2", relation="KNOWS")]
    await store.save_edges(edges)
    mock_db.query.assert_called_once()


@pytest.mark.asyncio
async def test_search_chunks_unfiltered(mock_db: Any) -> None:
    """Test unfiltered similarity search."""
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        {"result": [{"id": "chunk:c1", "text": "T", "index": 0, "embedding": [0.1]}]}
    ]
    chunks = await store.search_chunks([0.1], top_k=1)
    assert len(chunks) == 1
    assert "embedding <|5|> $embedding" in mock_db.query.call_args[0][0]


@pytest.mark.asyncio
async def test_search_chunks_filtered(mock_db: Any) -> None:
    """Test filtered similarity search with notebook IDs."""
    store = SurrealDocumentStore(mock_db)
    mock_db.query.return_value = [
        {"result": [{"id": "chunk:c1", "text": "T", "index": 0, "embedding": [0.1]}]}
    ]
    chunks = await store.search_chunks([0.1], top_k=1, active_notebook_ids=["nb1"])
    assert len(chunks) == 1
    assert "CONTAINSANY $notebook_ids" in mock_db.query.call_args[0][0]


@pytest.mark.asyncio
async def test_initialize_schema(mock_db: Any) -> None:
    """Test schema initialization queries."""
    doc_store = SurrealDocumentStore(mock_db)
    await doc_store.initialize_schema()
    assert "DEFINE INDEX chunk_embedding_idx" in mock_db.query.call_args[0][0]

    mock_db.query.reset_mock()
    graph_store = SurrealGraphStore(mock_db)
    await graph_store.initialize_schema()
    assert "DEFINE INDEX entity_embedding_idx" in mock_db.query.call_args[0][0]


@pytest.mark.asyncio
async def test_extract_rows_edge_cases(mock_db: Any) -> None:
    """Test _extract_rows with various malformed inputs."""
    from app.infrastructure.database.store import _extract_rows

    assert _extract_rows(None) == []
    assert _extract_rows([]) == []
    assert _extract_rows([{}]) == [{}]
    assert _extract_rows({"result": "not a list"}) == [{"result": "not a list"}]
    assert _extract_rows([{"result": {"id": "1"}}]) == []  # result is not a list


@pytest.mark.asyncio
async def test_traverse_logic(mock_db: Any) -> None:
    """Test complex graph traversal logic including multi-depth and incoming edges."""
    store = SurrealGraphStore(mock_db)

    # Mock side effect to handle different queries in the traversal loop
    async def side_effect(query: str, vars: dict[str, Any] | None = None):
        """Docstring generated to satisfy ruff D103."""
        # 1. Outgoing edges query
        if "FROM $node->relation" in query:
            node_id = str(vars["node"].id) if vars and "node" in vars else ""
            if node_id == "n1":
                return [
                    {
                        "result": [
                            {
                                "id": "rel:e1",
                                "in": "entity:n1",
                                "out": "entity:n2",
                                "source_id": "entity:n1",
                                "target_id": "entity:n2",
                                "relation": "KNOWS",
                                "weight": 0.8,
                            }
                        ]
                    }
                ]
            if node_id == "n2":
                return [
                    {
                        "result": [
                            {
                                "id": "rel:e2",
                                "in": "entity:n2",
                                "out": "entity:n3",
                                "source_id": "entity:n2",
                                "target_id": "entity:n3",
                                "relation": "LIKES",
                            }
                        ]
                    }
                ]

        # 2. Incoming edges query
        elif "FROM <-relation<-entity" in query:
            node_id = str(vars["node"].id) if vars and "node" in vars else ""
            if node_id == "n2":
                return [
                    {
                        "result": [
                            {
                                "id": "rel:e3",
                                "in": "entity:n4",
                                "out": "entity:n2",
                                "source_id": "entity:n4",
                                "target_id": "entity:n2",
                                "relation": "FOLLOWS",
                                "source_chunk_ids": ["chunk1"],
                            }
                        ]
                    }
                ]

        # 3. Fetch nodes query (at the end)
        elif "SELECT * FROM entity:" in query or "SELECT * FROM $node" in query:
            if vars and "node" in vars:
                nid = str(vars["node"].id)
                return [{"result": [{"id": f"entity:{nid}", "label": "ENTITY", "name": nid}]}]
            return [
                {
                    "result": [
                        {"id": "entity:n1", "label": "PERSON", "name": "n1"},
                        {"id": "entity:n2", "label": "PERSON", "name": "n2"},
                        {"id": "entity:n3", "label": "PERSON", "name": "n3"},
                        {"id": "entity:n4", "label": "PERSON", "name": "n4"},
                    ]
                }
            ]

        return [{"result": []}]

    mock_db.query.side_effect = side_effect

    # Traverse from n1 with depth 2
    nodes, edges = await store.traverse(seed_node_ids=["n1"], depth=2)

    assert len(nodes) == 4
    assert len(edges) == 3

    e3 = [e for e in edges if e.source_id == "n4"][0]
    assert e3.target_id == "n2"
    assert e3.source_chunk_ids == ["chunk1"]

    e1 = [e for e in edges if e.source_id == "n1"][0]
    assert e1.weight == 0.8

    e2 = [e for e in edges if e.source_id == "n2"][0]
    assert e2.weight == 1.0
