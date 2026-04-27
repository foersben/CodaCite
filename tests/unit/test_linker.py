"""Unit tests for the SimpleEntityLinker.

Validates entity resolution and linking logic.
"""

from unittest.mock import AsyncMock

import pytest

from app.domain.models import Node
from app.infrastructure.linker import SimpleEntityLinker


@pytest.mark.asyncio
async def test_linker_exact_match():
    """Test that it links a node when the name matches exactly in the query."""
    linker = SimpleEntityLinker()
    existing_nodes = [Node(id="n1", label="PERSON", name="Alice")]

    # Query contains "Alice"
    linked = await linker.link_entities("Is Alice here?", existing_nodes)

    assert len(linked) == 1
    assert linked[0].id == "n1"


@pytest.mark.asyncio
async def test_linker_no_match():
    """Test that it returns empty list when no match is found."""
    linker = SimpleEntityLinker()
    existing_nodes = [Node(id="n1", label="PERSON", name="Bob")]

    linked = await linker.link_entities("Is Alice here?", existing_nodes)

    assert len(linked) == 0


@pytest.mark.asyncio
async def test_linker_with_extractor():
    """Test linking using the extractor to parse query entities."""
    mock_extractor = AsyncMock()
    # mock_extractor.extract returns (nodes, edges)
    mock_extractor.extract.return_value = ([Node(id="temp", label="PERSON", name="Alice")], [])

    linker = SimpleEntityLinker(gliner_extractor=mock_extractor)
    existing_nodes = [Node(id="n1", label="PERSON", name="Alice")]

    linked = await linker.link_entities("Tell me about her.", existing_nodes)

    assert len(linked) == 1
    assert linked[0].id == "n1"
    mock_extractor.extract.assert_called_once_with("Tell me about her.")
