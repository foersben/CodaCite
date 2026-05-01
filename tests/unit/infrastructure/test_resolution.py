"""Unit tests for the JaroWinklerResolver infrastructure adapter.

Validates the entity resolution logic (deduplication) based on
name similarity using the Jaro-Winkler distance algorithm.
"""

import pytest

from app.domain.models import Node
from app.infrastructure.resolution import JaroWinklerResolver


@pytest.mark.asyncio
async def test_resolve_no_existing_nodes() -> None:
    """Tests resolution with no existing nodes returns all new nodes as is.

    Given:
        A set of new nodes and an empty list of existing nodes.
    When:
        The resolve_entities method is executed.
    Then:
        It should return all new nodes as separate entities.
    """
    # Arrange
    resolver = JaroWinklerResolver(threshold=0.85)
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
async def test_resolve_exact_match() -> None:
    """Tests that resolution merges an exact name match with an existing node.

    Given:
        A new node with an exact name match to an existing node.
    When:
        The resolve_entities method is executed.
    Then:
        It should merge the new node into the existing entity, preserving the existing ID.
    """
    # Arrange
    resolver = JaroWinklerResolver(threshold=0.85)
    existing = [Node(id="existing_alice", label="PERSON", name="Alice")]
    new = [Node(id="new_alice", label="PERSON", name="Alice", source_chunk_ids=["c1"])]

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert len(result) == 1
    assert result[0].id == "existing_alice"  # kept existing ID
    assert "c1" in result[0].source_chunk_ids


@pytest.mark.asyncio
async def test_resolve_similar_names() -> None:
    """Tests that resolution merges nodes with names exceeding the similarity threshold.

    Given:
        A new node with a misspelled name that is very similar to an existing node.
    When:
        The resolve_entities method is executed.
    Then:
        It should merge the nodes to consolidate the entity.
    """
    # Arrange
    resolver = JaroWinklerResolver(threshold=0.85)
    existing = [Node(id="existing_alice", label="PERSON", name="Alice Johnson")]
    new = [Node(id="new_alice", label="PERSON", name="Alice Jonhson")]  # typo

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert len(result) == 1
    assert result[0].id == "existing_alice"  # merged


@pytest.mark.asyncio
async def test_resolve_dissimilar_names() -> None:
    """Tests that resolution does not merge names that are below the similarity threshold.

    Given:
        A new node with a name completely different from existing nodes.
    When:
        The resolve_entities method is executed.
    Then:
        It should treat the new node as a distinct entity.
    """
    # Arrange
    resolver = JaroWinklerResolver(threshold=0.85)
    existing = [Node(id="existing_bob", label="PERSON", name="Bob Smith")]
    new = [Node(id="new_alice", label="PERSON", name="Alice Johnson")]

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert len(result) == 1
    assert result[0].id == "new_alice"  # NOT merged


@pytest.mark.asyncio
async def test_resolve_preserves_description() -> None:
    """Tests that resolution keeps the existing description when merging entities.

    Given:
        An existing node with a description and a matching new node without one.
    When:
        The resolver merges the nodes.
    Then:
        The consolidated node should retain the original description.
    """
    # Arrange
    resolver = JaroWinklerResolver(threshold=0.85)
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
async def test_resolve_new_description_fills_gap() -> None:
    """Tests that resolution uses the new node's description when the existing one is empty.

    Given:
        An existing node without a description and a matching new node with one.
    When:
        The resolver merges the nodes.
    Then:
        The consolidated node should adopt the new description.
    """
    # Arrange
    resolver = JaroWinklerResolver(threshold=0.85)
    existing = [Node(id="existing", label="PERSON", name="Alice")]
    new = [Node(id="new", label="PERSON", name="Alice", description="An engineer")]

    # Act
    result = await resolver.resolve_entities(new, existing)

    # Assert
    assert result[0].description == "An engineer"


def test_string_similarity() -> None:
    """Tests the internal Jaro-Winkler similarity calculation utility.

    Given:
        Pairs of strings with varying similarity levels.
    When:
        Calculating similarity scores.
    Then:
        It should return 1.0 for exact matches and high scores for close variations.
    """
    # Arrange
    resolver = JaroWinklerResolver()

    # Act & Assert
    assert resolver._string_similarity("Alice", "Alice") == 1.0
    assert resolver._string_similarity("Alice", "alice") == 1.0  # case insensitive
    assert resolver._string_similarity("Alice", "Bob") < 0.7
    assert resolver._string_similarity("Apple Inc", "Apple Inc.") > 0.9
