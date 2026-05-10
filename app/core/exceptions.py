"""Domain exceptions for the GraphRAG application.

This module defines the custom exception hierarchy used throughout the domain layer
to signal specific failure modes in a consistent, architecture-compliant manner.
"""


class GraphRAGError(Exception):
    """Base exception for all GraphRAG domain errors.

    All custom domain exceptions should inherit from this class to allow
    for generic error handling at the interface layer (e.g., FastAPI exception handlers).
    """

    pass


class ExtractionError(GraphRAGError):
    """Raised when entity or relationship extraction fails.

    This typically occurs during the Phase 5 (Knowledge Graph Extraction) of the
    ingestion pipeline where the LLM or extraction model (e.g., GeminiExtractor)
    fails to produce structured data from a text chunk.
    """

    pass


class GraphTraversalError(GraphRAGError):
    """Raised when a graph traversal operation fails.

    Occurs during multi-hop retrieval in the retrieval pipeline if the
    underlying graph store (e.g., SurrealDB) encounters malformed queries
    or depth limit issues.
    """

    pass


class ModelLoadError(GraphRAGError):
    """Raised when an NLP or LLM model fails to load or initialize.

    Usually relates to hardware constraints like CUDA Out-of-Memory (OOM)
    or missing local model artifacts for offline implementations (e.g., FastCoref).
    """

    pass


class DocumentProcessingError(GraphRAGError):
    """Raised when a document cannot be processed or chunked.

    Occurs during Phase 2 (Chunking) of the ingestion pipeline if the
    document format is corrupted or if the RecursiveCharacterTextSplitter
    encounters unrecoverable errors.
    """

    pass


class DatabaseConnectionError(GraphRAGError):
    """Raised when the connection to the database (e.g., SurrealDB) fails.

    Bridges infrastructure-level connection failures into a domain-readable format
    to prevent infrastructure leakage into the application layer.
    """

    pass
