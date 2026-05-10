"""Unit tests for domain exceptions.

Validates that domain-specific exceptions correctly inherit from the base
GraphRAGError and preserve message content for debugging and reporting.
"""

from app.domain.exceptions import (
    DatabaseConnectionError,
    DocumentProcessingError,
    ExtractionError,
    GraphRAGError,
    GraphTraversalError,
    ModelLoadError,
)


def test_graph_rag_error() -> None:
    """Tests the base GraphRAGError exception.

    Given:
        A message string.
    When:
        A GraphRAGError is raised with that message.
    Then:
        The string representation of the exception should match the message.
    """
    exc = GraphRAGError("test error")
    assert str(exc) == "test error"


def test_extraction_error() -> None:
    """Tests the ExtractionError exception.

    Given:
        An extraction error message.
    When:
        An ExtractionError is raised.
    Then:
        It should be an instance of GraphRAGError.
    """
    exc = ExtractionError("extraction error")
    assert isinstance(exc, GraphRAGError)


def test_graph_traversal_error() -> None:
    """Tests the GraphTraversalError exception.

    Given:
        A traversal error message.
    When:
        A GraphTraversalError is raised.
    Then:
        It should be an instance of GraphRAGError.
    """
    exc = GraphTraversalError("traversal error")
    assert isinstance(exc, GraphRAGError)


def test_model_load_error() -> None:
    """Tests the ModelLoadError exception.

    Given:
        A model load error message.
    When:
        A ModelLoadError is raised.
    Then:
        It should be an instance of GraphRAGError.
    """
    exc = ModelLoadError("load error")
    assert isinstance(exc, GraphRAGError)


def test_document_processing_error() -> None:
    """Tests the DocumentProcessingError exception.

    Given:
        A document processing error message.
    When:
        A DocumentProcessingError is raised.
    Then:
        It should be an instance of GraphRAGError.
    """
    exc = DocumentProcessingError("process error")
    assert isinstance(exc, GraphRAGError)


def test_database_connection_error() -> None:
    """Tests the DatabaseConnectionError exception.

    Given:
        A database connection error message.
    When:
        A DatabaseConnectionError is raised.
    Then:
        It should be an instance of GraphRAGError.
    """
    exc = DatabaseConnectionError("conn error")
    assert isinstance(exc, GraphRAGError)
