"""Unit tests for the GraphEnhancementUseCase.

Validates the graph enrichment logic, including community detection,
hierarchical structuring, and community summary generation using LLMs.
"""

from typing import Any

import pytest

from app.application.enhancement import GraphEnhancementUseCase
from app.domain.models import Edge, Node


@pytest.fixture
def use_case(mock_graph_store: Any) -> GraphEnhancementUseCase:
    """Provides a GraphEnhancementUseCase instance.

    Args:
        mock_graph_store: Global mock GraphStore fixture.

    Returns:
        A GraphEnhancementUseCase instance.
    """
    return GraphEnhancementUseCase(graph_store=mock_graph_store)


@pytest.mark.asyncio
async def test_enhance_empty_graph(
    use_case: GraphEnhancementUseCase,
    mock_graph_store: Any,
) -> None:
    """Tests that the enhancement process exits early when the graph has no nodes."""
    # Arrange
    mock_graph_store.get_all_nodes.return_value = []
    mock_graph_store.get_all_edges.return_value = []

    # Act
    await use_case.execute()

    # Assert
    mock_graph_store.save_community.assert_not_called()


@pytest.mark.asyncio
async def test_enhance_no_edges(
    use_case: GraphEnhancementUseCase,
    mock_graph_store: Any,
) -> None:
    """Tests that enhancement exits early when the graph has no edges."""
    # Arrange
    mock_graph_store.get_all_nodes.return_value = [
        Node(id="n1", label="PERSON", name="Alice"),
    ]
    mock_graph_store.get_all_edges.return_value = []

    # Act
    await use_case.execute()

    # Assert
    mock_graph_store.save_community.assert_not_called()


@pytest.mark.asyncio
async def test_enhance_detects_communities(
    use_case: GraphEnhancementUseCase,
    mock_graph_store: Any,
) -> None:
    """Tests that enhancement detects and saves communities."""
    # Arrange
    nodes = [
        Node(id="n1", label="PERSON", name="Alice"),
        Node(id="n2", label="PERSON", name="Bob"),
        Node(id="n3", label="COMPANY", name="Acme"),
    ]
    edges = [
        Edge(source_id="n1", target_id="n2", relation="KNOWS"),
        Edge(source_id="n2", target_id="n3", relation="WORKS_AT"),
    ]
    mock_graph_store.get_all_nodes.return_value = nodes
    mock_graph_store.get_all_edges.return_value = edges

    # Act
    await use_case.execute()

    # Assert
    assert mock_graph_store.save_community.call_count >= 1
    saved_community = mock_graph_store.save_community.call_args[0][0]
    assert saved_community.id
    assert saved_community.summary
    assert len(saved_community.node_ids) > 0


@pytest.mark.asyncio
async def test_enhance_with_llm_summarizer(
    mock_graph_store: Any,
    mocker: Any,
) -> None:
    """Tests enhancement with an LLM summarizer."""
    # Arrange
    nodes = [Node(id="n1", label="PERSON", name="Alice"), Node(id="n2", label="PERSON", name="Bob")]
    edges = [Edge(source_id="n1", target_id="n2", relation="KNOWS")]
    mock_graph_store.get_all_nodes.return_value = nodes
    mock_graph_store.get_all_edges.return_value = edges

    mock_summarizer = mocker.AsyncMock(return_value="A group of people.")
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store, llm_summarizer=mock_summarizer)

    # Act
    await use_case.execute()

    # Assert
    mock_summarizer.assert_called()
    saved_community = mock_graph_store.save_community.call_args[0][0]
    assert saved_community.summary == "A group of people."
