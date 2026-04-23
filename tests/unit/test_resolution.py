"""Tests for JaroWinklerResolver."""

from typing import Any

import pytest

from app.domain.models import Node
from app.infrastructure.resolution import JaroWinklerResolver


@pytest.mark.asyncio
async def test_resolve_no_existing_nodes(mock_embedder: Any) -> None:
    """Test resolution with no existing nodes returns all new nodes.

    Arrange: Empty existing nodes list.
    Act: Resolve new nodes.
    Assert: All new nodes are returned as-is.
    """
    # Arrange
    resolver = JaroWinklerResolver(embedder=mock_embedder, threshold=0.85)
    new_nodes = [
        Node(id="n1", label="PERSON", name="Alice"),
        Node(id="n2", label="COMPANY", name="Acme Corp"),
    ]

    # Act
    result = await resolver.resolve_entities(new_nodes, existing_nodes=[])

    # Assert
    assert len(result) == 2
    assert result[0].id == "n1"
    assert result[1].id == "n2"


@pytest.mark.asyncio
async def test_resolve_exact_match(mock_embedder: Any) -> None:
    """Test resolution merges an exact name match with the existing node.

    Arrange: New node has same name as existing node.
    Act: Resolve new nodes.
    Assert: Merged node uses the existing node's ID.
    """
    # Arrange
    resolver = JaroWinklerResolver(embedder=mock_embedder, threshold=0.85)
    existing = [Node(id="existing_alice", label="PERSON", name="Alice")]
    new = [Node(id="new_alice", label="PERSON", name="Alice", source_chunk_ids=["c1"])]

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert len(result) == 1
    assert result[0].id == "existing_alice"  # kept existing ID
    assert "c1" in result[0].source_chunk_ids


@pytest.mark.asyncio
async def test_resolve_similar_names(mock_embedder: Any) -> None:
    """Test resolution merges nodes with very similar names.

    Arrange: New node name is a slight variation of existing.
    Act: Resolve new nodes.
    Assert: Merged to existing node.
    """
    # Arrange
    resolver = JaroWinklerResolver(embedder=mock_embedder, threshold=0.85)
    existing = [Node(id="existing_alice", label="PERSON", name="Alice Johnson")]
    new = [Node(id="new_alice", label="PERSON", name="Alice Jonhson")]  # typo

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert len(result) == 1
    assert result[0].id == "existing_alice"  # merged


@pytest.mark.asyncio
async def test_resolve_dissimilar_names(mock_embedder: Any) -> None:
    """Test resolution does not merge very different names.

    Arrange: New node name is completely different from existing.
    Act: Resolve new nodes.
    Assert: New node is kept with its original ID.
    """
    # Arrange
    resolver = JaroWinklerResolver(embedder=mock_embedder, threshold=0.85)
    existing = [Node(id="existing_bob", label="PERSON", name="Bob Smith")]
    new = [Node(id="new_alice", label="PERSON", name="Alice Johnson")]

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert len(result) == 1
    assert result[0].id == "new_alice"  # NOT merged


@pytest.mark.asyncio
async def test_resolve_preserves_description(mock_embedder: Any) -> None:
    """Test resolution keeps the existing description when merging.

    Arrange: Existing node has a description, new node does not.
    Act: Resolve new nodes.
    Assert: Merged node has the existing description.
    """
    # Arrange
    resolver = JaroWinklerResolver(embedder=mock_embedder, threshold=0.85)
    existing = [
        Node(
            id="existing",
            label="PERSON",
            name="Alice",
            description="CEO of Acme",
        )
    ]
    new = [Node(id="new", label="PERSON", name="Alice")]

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert result[0].description == "CEO of Acme"


@pytest.mark.asyncio
async def test_resolve_new_description_fills_gap(mock_embedder: Any) -> None:
    """Test resolution uses new node's description when existing has none.

    Arrange: Existing node has no description, new node does.
    Act: Resolve new nodes.
    Assert: Merged node gets the new description.
    """
    # Arrange
    resolver = JaroWinklerResolver(embedder=mock_embedder, threshold=0.85)
    existing = [Node(id="existing", label="PERSON", name="Alice")]
    new = [Node(id="new", label="PERSON", name="Alice", description="An engineer")]

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert result[0].description == "An engineer"


def test_string_similarity(mock_embedder: Any) -> None:
    """Test the internal Jaro-Winkler similarity calculation.

    Arrange: Create resolver instance.
    Act: Compare various string pairs.
    Assert: Exact matches score 1.0, dissimilar strings score low.
    """
    # Arrange
    resolver = JaroWinklerResolver(embedder=mock_embedder)

    # Act & Assert
    assert resolver._string_similarity("Alice", "Alice") == 1.0
    assert resolver._string_similarity("Alice", "alice") == 1.0  # case insensitive
    assert resolver._string_similarity("Alice", "Bob") < 0.7
    assert resolver._string_similarity("Apple Inc", "Apple Inc.") > 0.9
