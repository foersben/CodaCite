"""Global pytest fixtures."""

import logging
from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    """Setup basic logging for tests to suppress noise or capture it."""
    logging.getLogger().setLevel(logging.CRITICAL)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None]:
    """Clear FastAPI dependency overrides after each test."""
    from app.main import app

    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def mock_document_store(mocker: Any) -> Any:
    """Provide a mock DocumentStore."""
    from app.core.interfaces import DocumentStore

    return mocker.AsyncMock(spec=DocumentStore)


@pytest.fixture(scope="function")
def mock_graph_store(mocker: Any) -> Any:
    """Provide a mock GraphStore."""
    from app.core.interfaces import GraphStore

    return mocker.AsyncMock(spec=GraphStore)


@pytest.fixture(scope="function")
def mock_coref_resolver(mocker: Any) -> Any:
    """Provide a mock CoreferenceResolver."""
    from app.core.interfaces import CoreferenceResolver

    return mocker.AsyncMock(spec=CoreferenceResolver)


@pytest.fixture(scope="function")
def mock_entity_extractor(mocker: Any) -> Any:
    """Provide a mock EntityExtractor."""
    from app.core.interfaces import EntityExtractor

    return mocker.AsyncMock(spec=EntityExtractor)


@pytest.fixture(scope="function")
def mock_entity_resolver(mocker: Any) -> Any:
    """Provide a mock EntityResolver."""
    from app.core.interfaces import EntityResolver

    return mocker.AsyncMock(spec=EntityResolver)


@pytest.fixture(scope="function")
def mock_embedder(mocker: Any) -> Any:
    """Provide a mock Embedder."""
    from app.core.interfaces import Embedder

    return mocker.AsyncMock(spec=Embedder)


@pytest.fixture(scope="function")
def mock_ingestion_use_case(mocker: Any) -> Any:
    """Provide a mock DocumentIngestionUseCase."""
    from app.pipelines.ingestion.ingestion import DocumentIngestionUseCase

    return mocker.AsyncMock(spec=DocumentIngestionUseCase)


@pytest.fixture(scope="function")
def mock_extraction_use_case(mocker: Any) -> Any:
    """Provide a mock GraphExtractionUseCase."""
    from app.pipelines.extraction.extraction_logic import GraphExtractionUseCase

    return mocker.AsyncMock(spec=GraphExtractionUseCase)


@pytest.fixture(scope="function")
def mock_retrieval_use_case(mocker: Any) -> Any:
    """Provide a mock GraphRAGRetrievalUseCase."""
    from app.pipelines.retrieval.retrieval import GraphRAGRetrievalUseCase

    return mocker.AsyncMock(spec=GraphRAGRetrievalUseCase)


@pytest.fixture(scope="function")
def mock_enhancement_use_case(mocker: Any) -> Any:
    """Provide a mock GraphEnhancementUseCase."""
    from app.pipelines.extraction.enhancement import GraphEnhancementUseCase

    return mocker.AsyncMock(spec=GraphEnhancementUseCase)


@pytest.fixture(scope="function")
def mock_notebook_use_case(mocker: Any) -> Any:
    """Provide a mock NotebookUseCase."""
    from app.pipelines.notebooks.notebook_manager import NotebookUseCase

    return mocker.AsyncMock(spec=NotebookUseCase)


@pytest.fixture(scope="function")
def mock_llm_generator(mocker: Any) -> Any:
    """Provide a mock LLMGenerator."""
    from app.core.interfaces import LLMGenerator

    return mocker.AsyncMock(spec=LLMGenerator)


@pytest.fixture(scope="function")
def mock_entity_linker(mocker: Any) -> Any:
    """Provide a mock entity linker."""
    return mocker.AsyncMock()


@pytest.fixture(scope="function")
def mock_reranker(mocker: Any) -> Any:
    """Provide a mock reranker."""
    return mocker.AsyncMock()
