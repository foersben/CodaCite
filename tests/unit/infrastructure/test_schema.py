"""Unit tests for the SurrealDB schema generation.

Validates that the schema queries are correctly generated with the expected
embedding dimensions.
"""

from app.infrastructure.database.schema import get_schema_queries


def test_get_schema_queries_default() -> None:
    """Tests schema generation with default embedding dimension."""
    queries = get_schema_queries()
    # base_queries(11) + chunk_queries(10) + graph_queries(11) + maintenance_queries(2) = 34
    assert len(queries) == 34
    # Check if vector index has default dimension
    assert "DIMENSION 1024" in queries[19]
    assert "DIMENSION 1024" in queries[27]


def test_get_schema_queries_custom_dim() -> None:
    """Tests schema generation with a custom embedding dimension."""
    queries = get_schema_queries(embedding_dim=768)
    assert "DIMENSION 768" in queries[19]
    assert "DIMENSION 768" in queries[27]
