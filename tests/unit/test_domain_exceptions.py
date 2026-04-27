"""Unit tests for domain exceptions.

Ensures exceptions can be raised and carry proper information.
"""

from app.domain.exceptions import (
    DatabaseConnectionError,
    DocumentProcessingError,
    ExtractionError,
    GraphRAGError,
    GraphTraversalError,
    ModelLoadError,
)


def test_graph_rag_error():
    """Test base GraphRAG exception."""
    exc = GraphRAGError("test error")
    assert str(exc) == "test error"


def test_extraction_error():
    """Test extraction exception."""
    exc = ExtractionError("extraction error")
    assert isinstance(exc, GraphRAGError)


def test_graph_traversal_error():
    """Test graph traversal exception."""
    exc = GraphTraversalError("traversal error")
    assert isinstance(exc, GraphRAGError)


def test_model_load_error():
    """Test model load exception."""
    exc = ModelLoadError("load error")
    assert isinstance(exc, GraphRAGError)


def test_document_processing_error():
    """Test document processing exception."""
    exc = DocumentProcessingError("process error")
    assert isinstance(exc, GraphRAGError)


def test_database_connection_error():
    """Test database connection exception."""
    exc = DatabaseConnectionError("conn error")
    assert isinstance(exc, GraphRAGError)
