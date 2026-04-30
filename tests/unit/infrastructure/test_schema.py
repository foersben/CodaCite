"""Unit tests for the SurrealDB schema generation.

Validates that the schema queries are correctly generated with the expected
embedding dimensions.
"""

from app.infrastructure.database.schema import get_schema_queries


def test_get_schema_queries_default() -> None:
    """Tests schema generation with default embedding dimension."""
    queries = get_schema_queries()
    assert len(queries) == 4
    # Check if vector index has default dimension
    assert "DIMENSION 1024" in queries[1]
    assert "DIMENSION 1024" in queries[2]


def test_get_schema_queries_custom_dim() -> None:
    """Tests schema generation with a custom embedding dimension."""
    queries = get_schema_queries(embedding_dim=768)
    assert "DIMENSION 768" in queries[1]
    assert "DIMENSION 768" in queries[2]
