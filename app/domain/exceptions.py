"""Domain exceptions for the GraphRAG application."""


class GraphRAGError(Exception):
    """Base exception for all GraphRAG domain errors."""

    pass


class ExtractionError(GraphRAGError):
    """Raised when entity or relationship extraction fails."""

    pass


class GraphTraversalError(GraphRAGError):
    """Raised when a graph traversal operation fails."""

    pass


class ModelLoadError(GraphRAGError):
    """Raised when an NLP or LLM model fails to load or initialize."""

    pass


class DocumentProcessingError(GraphRAGError):
    """Raised when a document cannot be processed or chunked."""

    pass


class DatabaseConnectionError(GraphRAGError):
    """Raised when the connection to the database (e.g., SurrealDB) fails."""

    pass
