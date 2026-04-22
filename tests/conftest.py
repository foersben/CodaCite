"""Global pytest fixtures."""

import logging

import pytest
from pytest_mock import MockerFixture


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    """Setup basic logging for tests to suppress noise or capture it."""
    logging.getLogger().setLevel(logging.CRITICAL)


@pytest.fixture(scope="function")
def mock_document_store(mocker: MockerFixture):
    """Provide a mock DocumentStore."""
    from app.domain.ports import DocumentStore

    return mocker.AsyncMock(spec=DocumentStore)


@pytest.fixture(scope="function")
def mock_graph_store(mocker: MockerFixture):
    """Provide a mock GraphStore."""
    from app.domain.ports import GraphStore

    return mocker.AsyncMock(spec=GraphStore)


@pytest.fixture(scope="function")
def mock_coref_resolver(mocker: MockerFixture):
    """Provide a mock CoreferenceResolver."""
    from app.domain.ports import CoreferenceResolver

    return mocker.AsyncMock(spec=CoreferenceResolver)


@pytest.fixture(scope="function")
def mock_entity_extractor(mocker: MockerFixture):
    """Provide a mock EntityExtractor."""
    from app.domain.ports import EntityExtractor

    return mocker.AsyncMock(spec=EntityExtractor)


@pytest.fixture(scope="function")
def mock_entity_resolver(mocker: MockerFixture):
    """Provide a mock EntityResolver."""
    from app.domain.ports import EntityResolver

    return mocker.AsyncMock(spec=EntityResolver)


@pytest.fixture(scope="function")
def mock_embedder(mocker: MockerFixture):
    """Provide a mock Embedder."""
    from app.domain.ports import Embedder

    return mocker.AsyncMock(spec=Embedder)


@pytest.fixture(scope="function")
def mock_ingestion_use_case(mocker: MockerFixture):
    """Provide a mock DocumentIngestionUseCase."""
    from app.application.ingestion import DocumentIngestionUseCase

    return mocker.AsyncMock(spec=DocumentIngestionUseCase)


@pytest.fixture(scope="function")
def mock_extraction_use_case(mocker: MockerFixture):
    """Provide a mock GraphExtractionUseCase."""
    from app.application.extraction import GraphExtractionUseCase

    return mocker.AsyncMock(spec=GraphExtractionUseCase)


@pytest.fixture(scope="function")
def mock_retrieval_use_case(mocker: MockerFixture):
    """Provide a mock GraphRAGRetrievalUseCase."""
    from app.application.retrieval import GraphRAGRetrievalUseCase

    return mocker.AsyncMock(spec=GraphRAGRetrievalUseCase)
