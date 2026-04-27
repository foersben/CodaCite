"""Unit tests for the GraphRAGRetrievalUseCase.

This module validates the retrieval orchestration logic, ensuring that document search,
entity linking, graph traversal, and reranking are correctly coordinated.
"""

from typing import Any

import pytest

from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.models import Chunk, Edge, Node


@pytest.fixture
def use_case(
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
) -> GraphRAGRetrievalUseCase:
    """Initialize the GraphRAGRetrievalUseCase with mocked dependencies.

    Args:
        mock_document_store: Mock document store fixture.
        mock_graph_store: Mock graph store fixture.
        mock_embedder: Mock embedder fixture.
        mock_entity_linker: Mock entity linker fixture.
        mock_reranker: Mock reranker fixture.

    Returns:
        An instance of GraphRAGRetrievalUseCase.
    """
    return GraphRAGRetrievalUseCase(
        document_store=mock_document_store,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
        entity_linker=mock_entity_linker,
        reranker=mock_reranker,
    )


@pytest.mark.asyncio
async def test_retrieval_execute_success(
    use_case: GraphRAGRetrievalUseCase,
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
) -> None:
    """Tests the full retrieval pipeline for success.

    Given:
        A query string and a top_k value.
    When:
        The execute method is called.
    Then:
        It should coordinate search, linking, traversal, and reranking.
    """
    # Arrange
    query = "What is Entity A?"
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", text="Document text", document_id="doc1", index=0)
    ]
    mock_graph_store.get_all_nodes.return_value = [
        Node(id="n1", name="Entity A", label="Concept", description="Description A")
    ]
    mock_graph_store.traverse.return_value = (
        [Node(id="n1", name="Entity A", label="Concept", description="Description A")],
        [Edge(source_id="n1", target_id="n2", relation="related_to")],
    )
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_embedder.query_prefix = "Query: "
    mock_entity_linker.link_entities.return_value = [
        Node(id="n1", name="Entity A", label="Concept")
    ]
    mock_reranker.rerank.side_effect = lambda q, ctx, top_k: [
        {"text": c, "score": 1.0} for c in ctx
    ]

    # Act
    results = await use_case.execute(query, top_k=5)

    # Assert
    assert len(results) > 0
    mock_document_store.search_chunks.assert_called_once()
    mock_entity_linker.link_entities.assert_called_once()
    mock_graph_store.traverse.assert_called_once()
    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_with_reranking(
    use_case: GraphRAGRetrievalUseCase,
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
) -> None:
    """Tests retrieval flow when a reranker successfully reorders snippets."""
    # Arrange
    mock_embedder.embed.return_value = [0.1]
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", text="text1", document_id="d1", index=0)
    ]
    mock_graph_store.get_all_nodes.return_value = []
    mock_entity_linker.link_entities.return_value = []

    mock_reranker_output = [{"text": "text1", "score": 0.95}]
    mock_reranker.rerank.return_value = mock_reranker_output

    # Act
    results = await use_case.execute("query")

    # Assert
    assert results == mock_reranker_output
    mock_reranker.rerank.assert_called_once()
