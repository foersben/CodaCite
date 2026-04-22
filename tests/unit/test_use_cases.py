"""Tests for Use Cases."""

from unittest.mock import AsyncMock

import pytest

from app.application.extraction import GraphExtractionUseCase
from app.domain.models import Chunk, Edge, Node


@pytest.mark.asyncio
async def test_extract_entities_success(
    mock_entity_extractor: AsyncMock,
    mock_entity_resolver: AsyncMock,
    mock_graph_store: AsyncMock,
    mock_embedder: AsyncMock,
) -> None:
    """Tests the entity extraction pipeline with valid text.

    Arrange: Setup the extractor with mocked ports and sample chunk.
    Act: Call the extraction execution method with the sample chunk.
    Assert: Verify the returned nodes/edges match expected output and methods were called.
    """
    # Arrange
    sample_chunk = Chunk(
        id="chunk_1", document_id="doc_1", text="Tim Cook is the CEO of Apple.", index=0
    )

    # Mock extractor returns
    mock_node_1 = Node(id="tim_cook", label="PERSON", name="Tim Cook")
    mock_node_2 = Node(id="apple", label="ORGANIZATION", name="Apple")
    mock_edge = Edge(source_id="tim_cook", target_id="apple", relation="IS_CEO_OF")

    mock_entity_extractor.extract.return_value = ([mock_node_1, mock_node_2], [mock_edge])

    # Mock store existing nodes
    mock_graph_store.get_all_nodes.return_value = []

    # Mock resolver
    mock_entity_resolver.resolve_entities.return_value = [mock_node_1, mock_node_2]

    # Mock embedder
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

    usecase = GraphExtractionUseCase(
        extractor=mock_entity_extractor,
        resolver=mock_entity_resolver,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
    )

    # Act
    final_nodes, final_edges = await usecase.execute([sample_chunk])

    # Assert
    assert len(final_nodes) == 2
    assert len(final_edges) == 1
    # Check normalization logic inside use case
    assert final_edges[0].relation == "CEO_OF"

    mock_entity_extractor.extract.assert_called_once_with(sample_chunk.text)
    mock_graph_store.save_nodes.assert_called_once()
    mock_graph_store.save_edges.assert_called_once()
