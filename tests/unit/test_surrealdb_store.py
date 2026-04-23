"""Tests for the SurrealDB store implementations.

This module validates the low-level database interactions and SurrealQL generation
within the Infrastructure layer (Persistence).
"""

from typing import Any

import pytest

from app.domain.models import Chunk, Document, Edge, Node
from app.infrastructure.database.store import SurrealDocumentStore, SurrealGraphStore


@pytest.fixture
def mock_db(mocker) -> Any:
    """Mock database instance."""
    return mocker.AsyncMock()


@pytest.mark.asyncio
async def test_save_document(mock_db: Any) -> None:
    """Test saving a document generates the correct SurrealQL query.

    Given: A SurrealDocumentStore and a Document domain model.
    When: save_document is called.
    Then: It should execute the correct SurrealQL UPDATE query with appropriate parameters.
    """
    # Arrange
    store = SurrealDocumentStore(mock_db)
    doc = Document(id="doc1", filename="test.pdf", metadata={"author": "Alice"})

    # Act
    await store.save_document(doc)

    # Assert
    mock_db.query.assert_called_once()
    args, kwargs = mock_db.query.call_args
    sql = args[0]
    variables = args[1] if len(args) > 1 else kwargs

    assert "UPDATE type::thing('document', $id) CONTENT" in sql
    assert variables["id"] == "doc1"
    assert variables["content"] == "test.pdf"
    assert variables["metadata"] == {"author": "Alice"}


@pytest.mark.asyncio
async def test_save_chunks(mock_db: Any) -> None:
    """Test saving chunks generates the correct SurrealQL queries.

    Given: A list of Chunk domain models.
    When: save_chunks is called on the SurrealDocumentStore.
    Then: It should execute an UPDATE query for each chunk in the list.
    """
    # Arrange
    store = SurrealDocumentStore(mock_db)
    chunks = [
        Chunk(id="c1", document_id="doc1", text="Chunk 1 text", index=0, embedding=[0.1, 0.2]),
        Chunk(id="c2", document_id="doc1", text="Chunk 2 text", index=1, embedding=[0.3, 0.4]),
    ]

    # Act
    await store.save_chunks(chunks)

    # Assert
    assert mock_db.query.call_count == 2
    for i, chunk in enumerate(chunks):
        args, kwargs = mock_db.query.call_args_list[i]
        sql = args[0]
        vars = args[1] if len(args) > 1 else kwargs
        assert "UPDATE type::thing('chunk', $id) CONTENT" in sql
        assert vars["id"] == chunk.id
        assert vars["text"] == chunk.text


@pytest.mark.asyncio
async def test_save_nodes(mock_db: Any) -> None:
    """Test saving nodes generates the correct UPSERT queries.

    Given: A list of Node domain models.
    When: save_nodes is called on the SurrealGraphStore.
    Then: It should execute a SurrealQL UPDATE query for each node.
    """
    # Arrange
    store = SurrealGraphStore(mock_db)
    nodes = [Node(id="n1", label="PERSON", name="Alice")]

    # Act
    await store.save_nodes(nodes)

    # Assert
    mock_db.query.assert_called_once()
    args, kwargs = mock_db.query.call_args
    sql = args[0]
    vars = args[1] if len(args) > 1 else kwargs

    assert "UPDATE type::thing('entity', $id) CONTENT" in sql
    assert vars["id"] == "n1"
    assert vars["name"] == "Alice"


@pytest.mark.asyncio
async def test_save_edges(mock_db: Any) -> None:
    """Test saving edges generates the correct RELATE queries.

    Given: A list of Edge domain models.
    When: save_edges is called on the SurrealGraphStore.
    Then: It should execute a SurrealQL RELATE query linking the source and target entities.
    """
    # Arrange
    store = SurrealGraphStore(mock_db)
    edges = [Edge(source_id="n1", target_id="n2", relation="KNOWS")]

    # Act
    await store.save_edges(edges)

    # Assert
    mock_db.query.assert_called_once()
    args, kwargs = mock_db.query.call_args
    sql = args[0]
    vars = args[1] if len(args) > 1 else kwargs

    assert "RELATE $source->relation->$target CONTENT" in sql
    assert vars["source"] == "entity:n1"
    assert vars["target"] == "entity:n2"
    assert vars["relation"] == "KNOWS"


@pytest.mark.asyncio
async def test_traverse_depth_2(mock_db: Any) -> None:
    """Test graph traversal up to depth 2.

    Given: A seed node and a graph structure in SurrealDB.
    When: traverse is called with a depth of 2.
    Then: It should correctly traverse the graph and return all nodes and edges within the specified depth.
    """
    # Arrange
    store = SurrealGraphStore(mock_db)

    # Mock behavior of db.query.
    # traverse runs queries in order:
    # 1. out-edges for n1
    # 2. in-edges for n1
    # 3. out-edges for n2
    # 4. in-edges for n2
    # 5. node queries for n1, n2, n3

    def side_effect(query: str, *args, **kwargs):
        if "entity:n1->relation" in query:
            return [
                {
                    "result": [
                        {
                            "id": "rel1",
                            "source_id": "entity:n1",
                            "target_id": "entity:n2",
                            "relation": "KNOWS",
                            "description": None,
                            "source_chunk_ids": [],
                            "weight": 1.0,
                        }
                    ]
                }
            ]
        elif "out = entity:n1" in query:
            return [{"result": []}]
        elif "entity:n2->relation" in query:
            return [
                {
                    "result": [
                        {
                            "id": "rel2",
                            "source_id": "entity:n2",
                            "target_id": "entity:n3",
                            "relation": "KNOWS",
                            "description": None,
                            "source_chunk_ids": [],
                            "weight": 1.0,
                        }
                    ]
                }
            ]
        elif "out = entity:n2" in query:
            return [{"result": []}]
        elif "FROM entity:n1" in query:
            return [
                {
                    "result": [
                        {
                            "id": "entity:n1",
                            "label": "PERSON",
                            "name": "Alice",
                            "description": None,
                            "description_embedding": None,
                            "source_chunk_ids": [],
                        }
                    ]
                }
            ]
        elif "FROM entity:n2" in query:
            return [
                {
                    "result": [
                        {
                            "id": "entity:n2",
                            "label": "PERSON",
                            "name": "Bob",
                            "description": None,
                            "description_embedding": None,
                            "source_chunk_ids": [],
                        }
                    ]
                }
            ]
        elif "FROM entity:n3" in query:
            return [
                {
                    "result": [
                        {
                            "id": "entity:n3",
                            "label": "PERSON",
                            "name": "Charlie",
                            "description": None,
                            "description_embedding": None,
                            "source_chunk_ids": [],
                        }
                    ]
                }
            ]
        return [{"result": []}]

    mock_db.query.side_effect = side_effect

    # Act
    nodes, edges = await store.traverse(seed_node_ids=["n1"], depth=2)

    # Assert
    assert len(nodes) == 3
    node_ids = {n.id for n in nodes}
    assert node_ids == {"n1", "n2", "n3"}

    assert len(edges) == 2
    edge_targets = {e.target_id for e in edges}
    assert edge_targets == {"n2", "n3"}
