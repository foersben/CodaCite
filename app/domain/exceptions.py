"""Domain exceptions for the GraphRAG application.

This module defines the custom exception hierarchy used throughout the domain layer
to signal specific failure modes in a consistent, architecture-compliant manner.
"""


class GraphRAGError(Exception):
    """Base exception for all GraphRAG domain errors.

    All custom domain exceptions should inherit from this class to allow
    for generic error handling at the interface layer.
    """

    pass


class ExtractionError(GraphRAGError):
    """Raised when entity or relationship extraction fails.

    This typically occurs during the processing of semantic chunks where
    the LLM or extraction model fails to produce structured data.
    """

    pass


class GraphTraversalError(GraphRAGError):
    """Raised when a graph traversal operation fails.

    This may be due to malformed queries or issues reaching depth limits
    during multi-hop retrieval.
    """

    pass


class ModelLoadError(GraphRAGError):
    """Raised when an NLP or LLM model fails to load or initialize.

    This is usually related to environmental constraints (e.g., CUDA OOM)
    or missing model artifacts.
    """

    pass


class DocumentProcessingError(GraphRAGError):
    """Raised when a document cannot be processed or chunked.

    Occurs if the document format is unsupported or if the text normalization
    process encounters unrecoverable errors.
    """

    pass


class DatabaseConnectionError(GraphRAGError):
    """Raised when the connection to the database (e.g., SurrealDB) fails.

    This error bridges the infrastructure failure into a domain-readable format.
    """

    pass
