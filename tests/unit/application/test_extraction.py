"""Unit tests for the GraphExtractionUseCase.

Verifies the knowledge graph extraction pipeline including entity/relationship
extraction, resolution, embedding, and storage.
"""

from unittest.mock import AsyncMock

import pytest

from app.application.extraction import GraphExtractionUseCase
from app.domain.models import Chunk, Edge, Node


@pytest.fixture
def mock_deps():
    """Create mock dependencies for the extraction use case."""
    return {
        "extractor": AsyncMock(),
        "resolver": AsyncMock(),
        "graph_store": AsyncMock(),
        "embedder": AsyncMock(),
    }


@pytest.fixture
def use_case(mock_deps):
    """Initialize the GraphExtractionUseCase with mocked dependencies."""
    return GraphExtractionUseCase(**mock_deps)


@pytest.mark.asyncio
async def test_extraction_success_full_pipeline(use_case, mock_deps):
    """Test successful graph extraction and resolution.

    Given: A list of document chunks.
    When: execute is called.
    Then: It should extract nodes/edges, resolve them, generate embeddings, and save.
    """
    # Arrange
    chunk = Chunk(id="chunk:1", document_id="doc:1", text="Alice works at Acme.", index=0)

    # 1. Extraction Mock
    node_alice = Node(id="alice", name="Alice", label="PERSON")
    node_acme = Node(id="acme", name="Acme", label="ORG")
    edge_works = Edge(id="edge:1", source_id="alice", target_id="acme", relation="works at")
    mock_deps["extractor"].extract.return_value = ([node_alice, node_acme], [edge_works])

    # 2. Store Mock
    mock_deps["graph_store"].get_all_nodes.return_value = []

    # 3. Resolver Mock
    mock_deps["resolver"].resolve_entities.return_value = [node_alice, node_acme]

    # 4. Embedder Mock
    mock_deps["embedder"].embed.return_value = [0.1, 0.2]

    # Act
    nodes, edges = await use_case.execute([chunk])

    # Assert
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].relation == "WORKS_FOR"  # Normalized

    # Verify tagging
    assert "chunk:1" in nodes[0].source_chunk_ids
    assert "chunk:1" in edges[0].source_chunk_ids

    # Verify calls
    mock_deps["extractor"].extract.assert_called_once_with("Alice works at Acme.")
    mock_deps["resolver"].resolve_entities.assert_called_once()
    mock_deps["embedder"].embed.assert_called()
    mock_deps["graph_store"].save_nodes.assert_called_once_with(nodes)
    mock_deps["graph_store"].save_edges.assert_called_once_with(edges)


@pytest.mark.asyncio
async def test_extraction_normalization_logic(use_case, mock_deps):
    """Test the relation normalization logic.

    Given: A relationship with a varied name like 'IS_CEO_OF'.
    When: execute is called.
    Then: It should normalize it to 'CEO_OF'.
    """
    # Arrange
    chunk = Chunk(id="c1", document_id="d1", text="Bob is CEO of Globex", index=0)
    edge = Edge(id="e1", source_id="bob", target_id="globex", relation="IS_CEO_OF")
    mock_deps["extractor"].extract.return_value = ([], [edge])
    mock_deps["resolver"].resolve_entities.return_value = []
    mock_deps["graph_store"].get_all_nodes.return_value = []

    # Act
    _, edges = await use_case.execute([chunk])

    # Assert
    assert edges[0].relation == "CEO_OF"


@pytest.mark.asyncio
async def test_extraction_empty_chunks(use_case):
    """Test extraction with no chunks.

    Given: An empty list of chunks.
    When: execute is called.
    Then: It should return empty lists.
    """
    # Act
    nodes, edges = await use_case.execute([])

    # Assert
    assert nodes == []
    assert edges == []
