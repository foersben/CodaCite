"""Unit test fixtures."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(scope="function")
def mock_document_store() -> AsyncMock:
    """Provide a mock DocumentStore."""
    from app.domain.ports import DocumentStore
    mock = AsyncMock(spec=DocumentStore)
    return mock

@pytest.fixture(scope="function")
def mock_graph_store() -> AsyncMock:
    """Provide a mock GraphStore."""
    from app.domain.ports import GraphStore
    mock = AsyncMock(spec=GraphStore)
    return mock

@pytest.fixture(scope="function")
def mock_coref_resolver() -> AsyncMock:
    """Provide a mock CoreferenceResolver."""
    from app.domain.ports import CoreferenceResolver
    mock = AsyncMock(spec=CoreferenceResolver)
    return mock

@pytest.fixture(scope="function")
def mock_entity_extractor() -> AsyncMock:
    """Provide a mock EntityExtractor."""
    from app.domain.ports import EntityExtractor
    mock = AsyncMock(spec=EntityExtractor)
    return mock

@pytest.fixture(scope="function")
def mock_entity_resolver() -> AsyncMock:
    """Provide a mock EntityResolver."""
    from app.domain.ports import EntityResolver
    mock = AsyncMock(spec=EntityResolver)
    return mock

@pytest.fixture(scope="function")
def mock_embedder() -> AsyncMock:
    """Provide a mock Embedder."""
    from app.domain.ports import Embedder
    mock = AsyncMock(spec=Embedder)
    return mock
