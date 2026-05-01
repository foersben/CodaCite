"""Unit tests for the SimpleEntityLinker infrastructure adapter.

Validates entity resolution and linking logic, ensuring that entities
mentioned in queries are correctly mapped to existing graph nodes.
"""

from typing import Any

import pytest

from app.domain.models import Node
from app.infrastructure.linker import SimpleEntityLinker


@pytest.mark.asyncio
async def test_linker_exact_match() -> None:
    """Tests that a node is linked when its name matches the query exactly.

    Given:
        An existing node named "Alice".
    When:
        The link_entities method is called with a query containing "Alice".
    Then:
        It should return the "Alice" node.
    """
    linker = SimpleEntityLinker()
    existing_nodes = [Node(id="n1", label="PERSON", name="Alice")]

    # Query contains "Alice"
    linked = await linker.link_entities("Is Alice here?", existing_nodes)

    assert len(linked) == 1
    assert linked[0].id == "n1"


@pytest.mark.asyncio
async def test_linker_no_match() -> None:
    """Tests that no nodes are returned when no match is found in the query.

    Given:
        An existing node named "Bob".
    When:
        The link_entities method is called with a query mentioning "Alice".
    Then:
        It should return an empty list.
    """
    linker = SimpleEntityLinker()
    existing_nodes = [Node(id="n1", label="PERSON", name="Bob")]

    linked = await linker.link_entities("Is Alice here?", existing_nodes)

    assert len(linked) == 0


@pytest.mark.asyncio
async def test_linker_with_extractor(mocker: Any) -> None:
    """Tests entity linking using an extractor to parse mentions from the query.

    Given:
        An extractor that identifies "Alice" in a vague query and an existing "Alice" node.
    When:
        The link_entities method is called.
    Then:
        It should use the extractor and successfully link to the existing node.

    Args:
        mocker: The pytest-mock fixture.
    """
    mock_extractor = mocker.AsyncMock()
    # mock_extractor.extract returns (nodes, edges)
    mock_extractor.extract.return_value = ([Node(id="temp", label="PERSON", name="Alice")], [])

    linker = SimpleEntityLinker(gliner_extractor=mock_extractor)
    existing_nodes = [Node(id="n1", label="PERSON", name="Alice")]

    linked = await linker.link_entities("Tell me about her.", existing_nodes)

    assert len(linked) == 1
    assert linked[0].id == "n1"
    mock_extractor.extract.assert_called_once_with("Tell me about her.")
