"""Global pytest fixtures."""

import logging
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    """Setup basic logging for tests to suppress noise or capture it."""
    logging.getLogger().setLevel(logging.CRITICAL)


@pytest.fixture(scope="function")
def mock_document_store() -> AsyncMock:
    """Provide a mock DocumentStore."""
    from app.domain.ports import DocumentStore

    return AsyncMock(spec=DocumentStore)


@pytest.fixture(scope="function")
def mock_graph_store() -> AsyncMock:
    """Provide a mock GraphStore."""
    from app.domain.ports import GraphStore

    return AsyncMock(spec=GraphStore)


@pytest.fixture(scope="function")
def mock_coref_resolver() -> AsyncMock:
    """Provide a mock CoreferenceResolver."""
    from app.domain.ports import CoreferenceResolver

    return AsyncMock(spec=CoreferenceResolver)


@pytest.fixture(scope="function")
def mock_entity_extractor() -> AsyncMock:
    """Provide a mock EntityExtractor."""
    from app.domain.ports import EntityExtractor

    return AsyncMock(spec=EntityExtractor)


@pytest.fixture(scope="function")
def mock_entity_resolver() -> AsyncMock:
    """Provide a mock EntityResolver."""
    from app.domain.ports import EntityResolver

    return AsyncMock(spec=EntityResolver)


@pytest.fixture(scope="function")
def mock_embedder() -> AsyncMock:
    """Provide a mock Embedder."""
    from app.domain.ports import Embedder

    return AsyncMock(spec=Embedder)


@pytest.fixture(scope="function")
def mock_ingestion_use_case() -> AsyncMock:
    """Provide a mock DocumentIngestionUseCase."""
    from app.application.ingestion import DocumentIngestionUseCase

    return AsyncMock(spec=DocumentIngestionUseCase)


@pytest.fixture(scope="function")
def mock_extraction_use_case() -> AsyncMock:
    """Provide a mock GraphExtractionUseCase."""
    from app.application.extraction import GraphExtractionUseCase

    return AsyncMock(spec=GraphExtractionUseCase)


@pytest.fixture(scope="function")
def mock_retrieval_use_case() -> AsyncMock:
    """Provide a mock GraphRAGRetrievalUseCase."""
    from app.application.retrieval import GraphRAGRetrievalUseCase

    return AsyncMock(spec=GraphRAGRetrievalUseCase)
