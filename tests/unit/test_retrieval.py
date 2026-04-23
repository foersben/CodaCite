"""Tests for GraphRAGRetrievalUseCase."""

from typing import Any

import pytest

from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.models import Chunk, Edge, Node


@pytest.fixture
def mock_linker(mocker: Any) -> Any:
    """Provide a mock entity linker."""
    linker = mocker.AsyncMock()
    linker.link_entities = mocker.AsyncMock(return_value=[])
    return linker


@pytest.fixture
def mock_reranker(mocker: Any) -> Any:
    """Provide a mock reranker."""
    reranker = mocker.AsyncMock()
    reranker.rerank = mocker.AsyncMock(
        return_value=[{"text": "result", "score": 0.95}]
    )
    return reranker


@pytest.mark.asyncio
async def test_retrieval_no_results(
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_linker: Any,
    mock_reranker: Any,
) -> None:
    """Test retrieval returns empty when no chunks or linked entities.

    Arrange: All stores return empty results.
    Act: Execute retrieval.
    Assert: Returns empty list.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1] * 768
    mock_document_store.search_chunks.return_value = []
    mock_graph_store.get_all_nodes.return_value = []
    mock_linker.link_entities.return_value = []

    use_case = GraphRAGRetrievalUseCase(
        document_store=mock_document_store,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
        entity_linker=mock_linker,
        reranker=mock_reranker,
    )

    # Act
    results = await use_case.execute("What is AI?")

    # Assert
    assert results == []


@pytest.mark.asyncio
async def test_retrieval_vector_only(
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_linker: Any,
    mock_reranker: Any,
) -> None:
    """Test retrieval with vector chunks only (no linked entities).

    Arrange: Document store returns chunks, entity linker returns nothing.
    Act: Execute retrieval.
    Assert: Results contain reranked chunk text.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1] * 768
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", document_id="d1", text="AI is the future.", index=0),
    ]
    mock_graph_store.get_all_nodes.return_value = []
    mock_linker.link_entities.return_value = []
    mock_reranker.rerank.return_value = [
        {"text": "AI is the future.", "score": 0.9},
    ]

    use_case = GraphRAGRetrievalUseCase(
        document_store=mock_document_store,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
        entity_linker=mock_linker,
        reranker=mock_reranker,
    )

    # Act
    results = await use_case.execute("What is AI?")

    # Assert
    assert len(results) == 1
    assert results[0]["text"] == "AI is the future."
    mock_embedder.embed.assert_called_once_with("What is AI?")


@pytest.mark.asyncio
async def test_retrieval_with_graph_traversal(
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_linker: Any,
    mock_reranker: Any,
) -> None:
    """Test retrieval with both chunks and graph traversal.

    Arrange: Document store returns chunks, linker finds entities, graph is traversed.
    Act: Execute retrieval.
    Assert: Reranker is called with combined contexts.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1] * 768
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", document_id="d1", text="Apple is a tech company.", index=0),
    ]

    alice_node = Node(id="alice", label="PERSON", name="Alice")
    apple_node = Node(id="apple", label="COMPANY", name="Apple")
    mock_graph_store.get_all_nodes.return_value = [alice_node, apple_node]
    mock_linker.link_entities.return_value = [apple_node]

    # Graph traversal
    mock_graph_store.traverse.return_value = (
        [apple_node, alice_node],
        [Edge(source_id="alice", target_id="apple", relation="WORKS_AT")],
    )

    mock_reranker.rerank.return_value = [
        {"text": "Apple is a tech company.", "score": 0.95},
        {"text": "Entity: Apple (COMPANY). ", "score": 0.85},
    ]

    use_case = GraphRAGRetrievalUseCase(
        document_store=mock_document_store,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
        entity_linker=mock_linker,
        reranker=mock_reranker,
    )

    # Act
    results = await use_case.execute("Tell me about Apple")

    # Assert
    assert len(results) == 2
    mock_graph_store.traverse.assert_called_once()
    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_reranker_failure_fallback(
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_linker: Any,
    mock_reranker: Any,
) -> None:
    """Test retrieval falls back when reranker raises an exception.

    Arrange: Reranker raises an exception.
    Act: Execute retrieval.
    Assert: Returns fallback results with score=1.0.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1] * 768
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", document_id="d1", text="Chunk text.", index=0),
    ]
    mock_graph_store.get_all_nodes.return_value = []
    mock_linker.link_entities.return_value = []
    mock_reranker.rerank.side_effect = RuntimeError("Reranker failed")

    use_case = GraphRAGRetrievalUseCase(
        document_store=mock_document_store,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
        entity_linker=mock_linker,
        reranker=mock_reranker,
    )

    # Act
    results = await use_case.execute("Some query")

    # Assert
    assert len(results) == 1
    assert results[0]["text"] == "Chunk text."
    assert results[0]["score"] == 1.0
