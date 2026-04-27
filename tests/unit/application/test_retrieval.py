from unittest.mock import AsyncMock

import pytest

from app.application.retrieval import GraphRAGRetrievalUseCase


@pytest.mark.asyncio
async def test_retrieval_execute():
    """Docstring generated to satisfy ruff D103."""
    # Arrange
    from app.domain.models import Chunk, Edge, Node

    mock_doc_store = AsyncMock()
    mock_doc_store.search_chunks.return_value = [
        Chunk(id="c1", text="Document text", document_id="doc1", index=0)
    ]

    mock_graph_store = AsyncMock()
    mock_graph_store.get_all_nodes.return_value = [
        Node(id="n1", name="Entity A", label="Concept", description="Description A")
    ]
    mock_graph_store.traverse.return_value = (
        [Node(id="n1", name="Entity A", label="Concept", description="Description A")],
        [Edge(id="e1", source_id="n1", target_id="n2", relation="related_to")],
    )

    mock_embedder = AsyncMock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_embedder.query_prefix = "Query: "

    mock_linker = AsyncMock()
    mock_linker.link_entities.return_value = [Node(id="n1", name="Entity A", label="Concept")]

    mock_reranker = AsyncMock()
    mock_reranker.rerank.side_effect = lambda q, ctx, top_k: [
        {"text": c, "score": 1.0} for c in ctx
    ]

    use_case = GraphRAGRetrievalUseCase(
        mock_doc_store, mock_graph_store, mock_embedder, mock_linker, mock_reranker
    )

    # Act
    results = await use_case.execute("What is Entity A?", top_k=5)

    # Assert
    assert len(results) > 0
    mock_doc_store.search_chunks.assert_called_once()
    mock_linker.link_entities.assert_called_once()
    mock_graph_store.traverse.assert_called_once()
    mock_reranker.rerank.assert_called_once()
