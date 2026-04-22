"""Unit tests for SurrealDB Store implementations."""

import pytest
from pytest_mock import MockerFixture

from app.domain.models import Chunk, Document, Edge, Node
from app.infrastructure.database.store import SurrealDocumentStore, SurrealGraphStore


@pytest.fixture
def mock_db(mocker: MockerFixture):
    """Mock database instance."""
    return mocker.AsyncMock()


@pytest.mark.asyncio
async def test_save_document(mock_db) -> None:
    """Test saving a document generates the correct SurrealQL query.

    Arrange: Set up SurrealDocumentStore with a mock DB and create a Document.
    Act: Call save_document.
    Assert: the db.query was called with correct SQL and variables.
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

    assert "CREATE document CONTENT" in sql
    assert variables["id"] == "doc1"
    assert variables["content"] == "test.pdf"
    assert variables["metadata"] == {"author": "Alice"}


@pytest.mark.asyncio
async def test_save_chunks(mock_db) -> None:
    """Test saving chunks generates the correct SurrealQL queries.

    Arrange: Set up SurrealDocumentStore and create a list of Chunks.
    Act: Call save_chunks.
    Assert: db.query was called for each chunk with correct params.
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
        assert "CREATE chunk CONTENT" in sql
        assert vars["id"] == chunk.id
        assert vars["text"] == chunk.text


@pytest.mark.asyncio
async def test_save_nodes(mock_db) -> None:
    """Test saving nodes generates the correct UPSERT queries.

    Arrange: Set up SurrealGraphStore and create a list of Nodes.
    Act: Call save_nodes.
    Assert: db.query was called for each node with correct params.
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

    assert "UPSERT entity CONTENT" in sql
    assert vars["id"] == "n1"
    assert vars["name"] == "Alice"


@pytest.mark.asyncio
async def test_save_edges(mock_db) -> None:
    """Test saving edges generates the correct RELATE queries.

    Arrange: Set up SurrealGraphStore and create a list of Edges.
    Act: Call save_edges.
    Assert: db.query was called for each edge with correct params.
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
