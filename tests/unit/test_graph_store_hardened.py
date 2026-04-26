"""Hardened unit tests for SurrealGraphStore.

This module provides deep coverage for graph traversal and record normalization logic.
"""

from unittest.mock import MagicMock

import pytest

from app.domain.models import Node
from app.infrastructure.database.store import SurrealGraphStore


@pytest.fixture
def mock_db_client():
    """Mock SurrealDB client."""
    client = MagicMock()
    # Ensure query returns an awaitable
    client.query.return_value = MagicMock()
    client.query.return_value.__await__ = MagicMock(return_value=iter([]))
    return client


@pytest.fixture
def graph_store(mock_db_client):
    """SurrealGraphStore instance with mocked client."""
    return SurrealGraphStore(db=mock_db_client)


@pytest.mark.asyncio
async def test_traverse_normalization_logic(graph_store, mock_db_client):
    """Test that traversal correctly normalizes record IDs.

    Given: A start node ID with special characters.
    When: Traverse is called.
    Then: It should normalize the ID before executing the SurrealQL query.
    """
    # Arrange
    start_node = "user:123-456"
    mock_db_client.query.return_value = []

    # Act
    await graph_store.traverse(seed_node_ids=[start_node], depth=1)

    # Assert
    # Verify the query string contains the node ID
    called_query = ""
    for call in mock_db_client.query.call_args_list:
        if "entity:" in call[0][0]:
            called_query = call[0][0]
            break

    assert start_node in called_query


@pytest.mark.asyncio
async def test_traverse_result_mapping(graph_store, mock_db_client):
    """Test that traversal results are correctly mapped to domain models.

    Given: A SurrealDB query result with raw nodes and edges.
    When: Traverse is called.
    Then: It should return a list of (Node, Edge) tuples.
    """
    # Arrange
    # seed_node_ids=["start"]
    # 1. Edge query (outgoing)
    # 2. Edge query (incoming)
    # 3. Node query for "start"
    # 4. Node query for "n2" (target of edge)

    edge_result = [
        {
            "result": [
                {
                    "id": "relation:e1",
                    "source_id": "entity:start",
                    "target_id": "entity:n2",
                    "relation": "RELATES_TO",
                }
            ]
        }
    ]

    node_result_start = [
        {
            "result": [
                {
                    "id": "entity:start",
                    "label": "Concept",
                    "name": "Start Node",
                }
            ]
        }
    ]

    node_result_n2 = [
        {
            "result": [
                {
                    "id": "entity:n2",
                    "label": "Entity",
                    "name": "Target Node",
                }
            ]
        }
    ]

    mock_db_client.query.side_effect = [
        edge_result,  # Outgoing edges for "start"
        [],  # Incoming edges for "start"
        node_result_start,  # Node fetch for "start"
        node_result_n2,  # Node fetch for "n2"
    ]

    # Act
    nodes, edges = await graph_store.traverse(seed_node_ids=["start"], depth=1)

    # Assert
    assert len(nodes) == 2
    assert any(n.id == "start" for n in nodes)
    assert any(n.id == "n2" for n in nodes)
    assert len(edges) == 1
    assert edges[0].source_id == "start"
    assert edges[0].target_id == "n2"


@pytest.mark.asyncio
async def test_save_nodes_edges_transaction_fail(graph_store, mock_db_client):
    """Test that saving fails if the database query raises an error.

    Given: A list of nodes and edges to save.
    When: The query execution fails.
    Then: It should raise the exception.
    """
    # Arrange
    mock_db_client.query.side_effect = Exception("SurrealDB Connection Error")

    # Act & Assert
    with pytest.raises(Exception, match="SurrealDB Connection Error"):
        await graph_store.save_nodes([Node(id="1", label="L", name="N")])
