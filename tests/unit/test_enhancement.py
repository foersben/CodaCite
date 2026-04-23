"""Tests for GraphEnhancementUseCase.

This module validates the graph enrichment logic, such as community detection
and summary generation, within the Application layer.
"""

from typing import Any

import pytest

from app.application.enhancement import GraphEnhancementUseCase
from app.domain.models import Edge, Node


@pytest.mark.asyncio
async def test_enhance_empty_graph(mock_graph_store: Any) -> None:
    """Test enhancement exits early when graph has no nodes.

    Given: An empty graph with no nodes or edges.
    When: The GraphEnhancementUseCase is executed.
    Then: It should exit early without saving any communities.
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

    Given: A graph with isolated nodes and no edges.
    When: The GraphEnhancementUseCase is executed.
    Then: It should not detect any communities or save any results.
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

    Given: A connected graph with several nodes and edges.
    When: The GraphEnhancementUseCase is executed.
    Then: It should detect at least one community and persist it to the graph store.
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

    Given: A connected graph and an available LLM summarizer service.
    When: The GraphEnhancementUseCase is executed.
    Then: It should use the LLM to generate descriptive community summaries.
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
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store, llm_summarizer=mock_summarizer)

    # Act
    await use_case.execute()

    # Assert
    mock_summarizer.assert_called()
    saved_community = mock_graph_store.save_community.call_args[0][0]
    assert saved_community.summary == "A group of people who know each other."
