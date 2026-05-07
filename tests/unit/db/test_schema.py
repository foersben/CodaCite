"""Unit tests for the SurrealDB schema generation.

Validates that the schema queries are correctly generated with the expected
embedding dimensions.
"""

from app.db.schema import get_schema_queries


def test_get_schema_queries_default() -> None:
    """Tests schema generation with default embedding dimension."""
    queries = get_schema_queries()
    # Check for presence of key field definitions
    assert any("DEFINE FIELD start_char ON chunk" in q for q in queries)
    assert any("DEFINE FIELD end_char ON chunk" in q for q in queries)
    # Check if vector index has default dimension
    vector_queries = [q for q in queries if "DIMENSION 1024" in q]
    assert len(vector_queries) == 2


def test_get_schema_queries_custom_dim() -> None:
    """Tests schema generation with a custom embedding dimension."""
    queries = get_schema_queries(embedding_dim=768)
    vector_queries = [q for q in queries if "DIMENSION 768" in q]
    # One for chunk embedding, one for graph node embedding
    assert len(vector_queries) == 2
