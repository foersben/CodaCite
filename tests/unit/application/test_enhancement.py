"""Unit tests for the GraphEnhancementUseCase.

Validates community detection, LLM-based summarization, and persistence logic.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.enhancement import GraphEnhancementUseCase
from app.domain.models import Edge, Node


@pytest.fixture
def mock_graph_store():
    """Provides a mocked GraphStore."""
    store = MagicMock()
    store.get_all_nodes = AsyncMock(
        return_value=[
            Node(id="n1", label="P", name="Alice"),
            Node(id="n2", label="P", name="Bob"),
            Node(id="n3", label="P", name="Charlie"),
        ]
    )
    store.get_all_edges = AsyncMock(
        return_value=[
            Edge(source_id="n1", target_id="n2", relation="KNOWS"),
            Edge(source_id="n2", target_id="n3", relation="KNOWS"),
        ]
    )
    store.save_community = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_execute_success(mock_graph_store):
    """Tests successful community detection and storage."""
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store)

    await use_case.execute()

    # Should have called get_all_nodes and get_all_edges
    mock_graph_store.get_all_nodes.assert_called_once()
    mock_graph_store.get_all_edges.assert_called_once()

    # Should have called save_community at least once ( Louvain should find 1 or more)
    assert mock_graph_store.save_community.called


@pytest.mark.asyncio
async def test_execute_empty_graph(mock_graph_store):
    """Tests enhancement behavior on an empty graph."""
    mock_graph_store.get_all_nodes = AsyncMock(return_value=[])
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store)

    await use_case.execute()

    mock_graph_store.save_community.assert_not_called()


@pytest.mark.asyncio
async def test_execute_with_llm_summarizer(mock_graph_store):
    """Tests community detection with LLM-based summarization."""
    mock_summarizer = AsyncMock(return_value="LLM Summary")
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store, llm_summarizer=mock_summarizer)

    await use_case.execute()

    assert mock_summarizer.called
    # Check that the saved community has the LLM summary
    args = mock_graph_store.save_community.call_args[0][0]
    assert args.summary == "LLM Summary"


@pytest.mark.asyncio
async def test_execute_llm_failure_fallback(mock_graph_store):
    """Tests fallback to naive summary when LLM summarization fails."""
    mock_summarizer = AsyncMock(side_effect=Exception("LLM Error"))
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store, llm_summarizer=mock_summarizer)

    await use_case.execute()

    # Should have called save_community with fallback summary
    args = mock_graph_store.save_community.call_args[0][0]
    assert "Community" in args.summary
    assert "nodes" in args.summary


@pytest.mark.asyncio
async def test_execute_louvain_failure(mock_graph_store, mocker):
    """Tests handling of algorithm failures."""
    mocker.patch("networkx.community.louvain_communities", side_effect=Exception("Algo error"))
    use_case = GraphEnhancementUseCase(graph_store=mock_graph_store)

    with pytest.raises(Exception, match="Algo error"):
        await use_case.execute()
