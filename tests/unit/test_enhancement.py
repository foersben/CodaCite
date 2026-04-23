"""Tests for GraphEnhancementUseCase."""

from typing import Any

import pytest

from app.application.enhancement import GraphEnhancementUseCase
from app.domain.models import Edge, Node


@pytest.mark.asyncio
async def test_enhance_empty_graph(mock_graph_store: Any) -> None:
    """Test enhancement exits early when graph has no nodes.

    Arrange: Mock graph_store returns empty nodes/edges.
    Act: Execute the use case.
    Assert: save_community is never called.
    """
    # Arrange
    mock_graph_store.get_all_nodes.return_value = []
    mock_graph_store.get_all_edges.return_value = []
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store)

    # Act
    await use_case.execute()

    # Assert
    mock_graph_store.save_community.assert_not_called()


@pytest.mark.asyncio
async def test_enhance_no_edges(mock_graph_store: Any) -> None:
    """Test enhancement exits early when graph has nodes but no edges.

    Arrange: Mock graph_store returns nodes but no edges.
    Act: Execute the use case.
    Assert: save_community is never called.
    """
    # Arrange
    mock_graph_store.get_all_nodes.return_value = [
        Node(id="n1", label="PERSON", name="Alice"),
    ]
    mock_graph_store.get_all_edges.return_value = []
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store)

    # Act
    await use_case.execute()

    # Assert
    mock_graph_store.save_community.assert_not_called()


@pytest.mark.asyncio
async def test_enhance_detects_communities(mock_graph_store: Any) -> None:
    """Test enhancement detects communities and saves them.

    Arrange: Mock graph_store returns a connected graph.
    Act: Execute the use case.
    Assert: save_community is called at least once with a Community model.
    """
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
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store)

    # Act
    await use_case.execute()

    # Assert
    assert mock_graph_store.save_community.call_count >= 1
    saved_community = mock_graph_store.save_community.call_args[0][0]
    assert saved_community.id  # UUID string
    assert saved_community.summary  # non-empty summary
    assert len(saved_community.node_ids) > 0


@pytest.mark.asyncio
async def test_enhance_with_llm_summarizer(mock_graph_store: Any, mocker: Any) -> None:
    """Test enhancement uses LLM summarizer when provided.

    Arrange: Mock graph_store with a small graph and provide an async summarizer.
    Act: Execute the use case.
    Assert: The summarizer was called and the community summary uses LLM output.
    """
    # Arrange
    nodes = [
        Node(id="n1", label="PERSON", name="Alice"),
        Node(id="n2", label="PERSON", name="Bob"),
    ]
    edges = [
        Edge(source_id="n1", target_id="n2", relation="KNOWS"),
    ]
    mock_graph_store.get_all_nodes.return_value = nodes
    mock_graph_store.get_all_edges.return_value = edges

    mock_summarizer = mocker.AsyncMock(return_value="A group of people who know each other.")
    use_case = GraphEnhancementUseCase(
        graph_store=mock_graph_store, llm_summarizer=mock_summarizer
    )

    # Act
    await use_case.execute()

    # Assert
    mock_summarizer.assert_called()
    saved_community = mock_graph_store.save_community.call_args[0][0]
    assert saved_community.summary == "A group of people who know each other."
