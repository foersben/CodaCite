"""Unit tests for the GraphExtractionUseCase.

Verifies the knowledge graph extraction pipeline including entity/relationship
extraction, resolution, embedding, and storage.
"""

from typing import Any

import pytest

from app.application.extraction import GraphExtractionUseCase
from app.domain.models import Chunk, Edge, Node


@pytest.fixture
def use_case(
    mock_entity_extractor: Any,
    mock_entity_resolver: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
) -> GraphExtractionUseCase:
    """Initialize the GraphExtractionUseCase with mocked dependencies.

    Args:
        mock_entity_extractor: Mock entity extractor fixture.
        mock_entity_resolver: Mock entity resolver fixture.
        mock_graph_store: Mock graph store fixture.
        mock_embedder: Mock embedder fixture.

    Returns:
        An instance of GraphExtractionUseCase.
    """
    return GraphExtractionUseCase(
        extractor=mock_entity_extractor,
        resolver=mock_entity_resolver,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
    )


@pytest.mark.asyncio
async def test_extraction_success_full_pipeline(
    use_case: GraphExtractionUseCase,
    mock_entity_extractor: Any,
    mock_entity_resolver: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
) -> None:
    """Test successful graph extraction and resolution.

    Given:
        A list of document chunks.
    When:
        execute is called.
    Then:
        It should extract nodes/edges, resolve them, generate embeddings, and save.
    """
    # Arrange
    chunk = Chunk(id="chunk:1", document_id="doc:1", text="Alice works at Acme.", index=0)

    node_alice = Node(id="alice", name="Alice", label="PERSON")
    node_acme = Node(id="acme", name="Acme", label="ORG")
    edge_works = Edge(source_id="alice", target_id="acme", relation="works at")

    mock_entity_extractor.extract.return_value = ([node_alice, node_acme], [edge_works])
    mock_graph_store.get_all_nodes.return_value = []
    mock_entity_resolver.resolve_entities.return_value = [node_alice, node_acme]
    mock_embedder.embed.return_value = [0.1, 0.2]

    # Act
    nodes, edges = await use_case.execute([chunk])

    # Assert
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].relation == "WORKS_FOR"  # Normalized in code

    # Verify tagging
    assert "chunk:1" in nodes[0].source_chunk_ids
    assert "chunk:1" in edges[0].source_chunk_ids

    # Verify calls
    mock_entity_extractor.extract.assert_called_once_with("Alice works at Acme.")
    mock_entity_resolver.resolve_entities.assert_called_once()
    mock_embedder.embed.assert_called()
    mock_graph_store.save_nodes.assert_called_once_with(nodes)
    mock_graph_store.save_edges.assert_called_once_with(edges)


@pytest.mark.asyncio
async def test_extraction_normalization_logic(
    use_case: GraphExtractionUseCase,
    mock_entity_extractor: Any,
    mock_entity_resolver: Any,
    mock_graph_store: Any,
) -> None:
    """Test the relation normalization logic."""
    # Arrange
    chunk = Chunk(id="c1", document_id="d1", text="Bob is CEO of Globex", index=0)
    edge = Edge(source_id="bob", target_id="globex", relation="IS_CEO_OF")

    mock_entity_extractor.extract.return_value = ([], [edge])
    mock_entity_resolver.resolve_entities.return_value = []
    mock_graph_store.get_all_nodes.return_value = []

    # Act
    _, edges = await use_case.execute([chunk])

    # Assert
    assert edges[0].relation == "CEO_OF"


@pytest.mark.asyncio
async def test_extraction_empty_chunks(use_case: GraphExtractionUseCase) -> None:
    """Test extraction with no chunks."""
    # Act
    nodes, edges = await use_case.execute([])

    # Assert
    assert nodes == []
    assert edges == []
