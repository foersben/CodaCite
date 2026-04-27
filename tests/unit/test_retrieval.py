"""Unit tests for the GraphRAGRetrievalUseCase.

Validates the retrieval logic, including vector search, entity linking,
and graph-based traversal.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.models import Chunk, Edge, Node
from app.domain.ports import DocumentStore, Embedder, GraphStore


@pytest.fixture
def mock_document_store():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock(spec=DocumentStore)


@pytest.fixture
def mock_graph_store():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock(spec=GraphStore)


@pytest.fixture
def mock_embedder():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock(spec=Embedder)


@pytest.fixture
def mock_entity_linker():
    """Docstring generated to satisfy ruff D103."""
    linker = MagicMock()
    linker.link_entities = AsyncMock()
    return linker


@pytest.fixture
def mock_reranker():
    """Docstring generated to satisfy ruff D103."""
    reranker = MagicMock()
    reranker.rerank = AsyncMock()
    return reranker


@pytest.mark.asyncio
async def test_retrieval_execute_basic(
    mock_document_store, mock_graph_store, mock_embedder, mock_entity_linker, mock_reranker
):
    """Test basic retrieval flow.

    Given: A query.
    When: GraphRAGRetrievalUseCase.execute is called.
    Then: It should embed query, search chunks, link entities, traverse, and return context.
    """
    # Arrange
    use_case = GraphRAGRetrievalUseCase(
        mock_document_store, mock_graph_store, mock_embedder, mock_entity_linker, mock_reranker
    )
    query = "What is quantum computing?"

    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

    chunk = Chunk(id="chunk:1", text="Quantum computing uses qubits.", document_id="doc1", index=0)
    mock_document_store.search_chunks.return_value = [chunk]

    # Entity linking
    node1 = Node(
        id="node:1", name="Quantum Computing", label="Concept", description="A field of study"
    )
    mock_graph_store.get_all_nodes.return_value = [node1]
    mock_entity_linker.link_entities.return_value = [node1]

    # Traversal
    node2 = Node(
        id="node:2", name="Qubit", label="Concept", description="Basic unit of quantum info"
    )
    edge = Edge(source_id="node:1", target_id="node:2", relation="uses")
    mock_graph_store.traverse.return_value = ([node2], [edge])

    # Reranker fallback (returning top context as is)
    mock_reranker.rerank.side_effect = Exception("No reranker")

    # Act
    results = await use_case.execute(query, top_k=5)

    # Assert
    assert len(results) > 0
    texts = [r["text"] for r in results]
    assert "Quantum computing uses qubits." in texts
    assert "Entity: Qubit (Concept). Basic unit of quantum info" in texts
    assert "Relationship: node:1 uses node:2." in texts

    mock_embedder.embed.assert_called_once()
    mock_document_store.search_chunks.assert_called_once()
    mock_graph_store.traverse.assert_called_once_with(["node:1"], depth=2)


@pytest.mark.asyncio
async def test_retrieval_with_reranking(
    mock_document_store, mock_graph_store, mock_embedder, mock_entity_linker, mock_reranker
):
    """Test retrieval with successful reranking.

    Given: A query and context snippets.
    When: Reranker successfully processes snippets.
    Then: Reranked results should be returned.
    """
    # Arrange
    use_case = GraphRAGRetrievalUseCase(
        mock_document_store, mock_graph_store, mock_embedder, mock_entity_linker, mock_reranker
    )

    mock_embedder.embed.return_value = [0.1]
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", text="text1", document_id="d1", index=0)
    ]
    mock_graph_store.get_all_nodes.return_value = []
    mock_entity_linker.link_entities.return_value = []

    mock_reranker.rerank.return_value = [{"text": "text1", "score": 0.95}]

    # Act
    results = await use_case.execute("query")

    # Assert
    assert results == [{"text": "text1", "score": 0.95}]
    mock_reranker.rerank.assert_called_once()
