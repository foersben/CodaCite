"""Hardened unit tests for the SurrealGraphStore infrastructure adapter.

Validates deep graph traversal logic, record ID normalization, result mapping
to domain models, and transactional error handling in SurrealDB.
"""

from typing import Any

import pytest

from app.domain.models import Node
from app.infrastructure.database.store import SurrealGraphStore


@pytest.fixture
def mock_db_client(mocker: Any) -> Any:
    """Provides a mocked SurrealDB client.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked SurrealDB client with an async query method.
    """
    client = mocker.MagicMock()
    # Ensure query is an AsyncMock for async/await support
    client.query = mocker.AsyncMock()
    return client


@pytest.fixture
def graph_store(mock_db_client: Any) -> SurrealGraphStore:
    """Provides a SurrealGraphStore instance with a mocked client.

    Args:
        mock_db_client: The mocked SurrealDB client fixture.

    Returns:
        A SurrealGraphStore instance for testing.
    """
    return SurrealGraphStore(db=mock_db_client)


@pytest.mark.asyncio
async def test_traverse_normalization_logic(
    graph_store: SurrealGraphStore, mock_db_client: Any
) -> None:
    """Tests that traversal correctly normalizes record IDs in queries.

    Given:
        A start node ID containing special characters (e.g., colons, dashes).
    When:
        The traverse method is called.
    Then:
        It should correctly incorporate the ID into the SurrealQL query string.

    Args:
        graph_store: The SurrealGraphStore fixture.
        mock_db_client: The mocked SurrealDB client fixture.
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
async def test_traverse_result_mapping(graph_store: SurrealGraphStore, mock_db_client: Any) -> None:
    """Tests that raw SurrealDB traversal results are correctly mapped to domain models.

    Given:
        A set of mocked SurrealDB query responses representing nodes and edges.
    When:
        The traverse method is called.
    Then:
        It should return valid Node and Edge domain objects with correct attributes.

    Args:
        graph_store: The SurrealGraphStore fixture.
        mock_db_client: The mocked SurrealDB client fixture.
    """
    # Arrange
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
async def test_save_nodes_edges_transaction_fail(
    graph_store: SurrealGraphStore, mock_db_client: Any
) -> None:
    """Tests that saving nodes fails gracefully if the database query raises an error.

    Given:
        A list of nodes to save and a database client that raises an exception.
    When:
        The save_nodes method is called.
    Then:
        It should propagate the exception to the caller.

    Args:
        graph_store: The SurrealGraphStore fixture.
        mock_db_client: The mocked SurrealDB client fixture.
    """
    # Arrange
    mock_db_client.query.side_effect = Exception("SurrealDB Connection Error")

    # Act & Assert
    with pytest.raises(Exception, match="SurrealDB Connection Error"):
        await graph_store.save_nodes([Node(id="1", label="L", name="N")])
