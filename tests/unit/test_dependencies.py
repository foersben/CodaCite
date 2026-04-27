"""Tests for the application dependencies.

This module validates the dependency injection providers in app/interfaces/dependencies.py.
"""

from unittest.mock import MagicMock

import pytest

from app.application.chat import ChatUseCase
from app.application.ingestion import DocumentIngestionUseCase
from app.infrastructure.database.store import SurrealDocumentStore, SurrealGraphStore
from app.interfaces import dependencies


def test_get_db():
    """Test get_db returns the global surreal_db instance."""
    db = dependencies.get_db()
    assert db == dependencies.surreal_db


def test_get_document_store():
    """Test get_document_store returns a SurrealDocumentStore."""
    mock_db = MagicMock()
    store = dependencies.get_document_store(mock_db)
    assert isinstance(store, SurrealDocumentStore)
    assert store.db == mock_db


def test_get_graph_store():
    """Test get_graph_store returns a SurrealGraphStore."""
    mock_db = MagicMock()
    store = dependencies.get_graph_store(mock_db)
    assert isinstance(store, SurrealGraphStore)
    assert store.db == mock_db


def test_get_coref_resolver(mocker):
    """Test get_coref_resolver returns the correct implementation."""
    mocker.patch("app.interfaces.dependencies.FastCorefResolver", return_value=MagicMock())
    resolver = dependencies.get_coref_resolver()
    assert isinstance(resolver, MagicMock)


def test_get_embedder(mocker):
    """Test get_embedder returns a singleton HuggingFaceEmbedder."""
    # Patch the HuggingFaceEmbedder to avoid actual loading
    mocker.patch("app.interfaces.dependencies.HuggingFaceEmbedder", return_value=MagicMock())

    # Reset singleton for test
    dependencies._embedder = None

    embedder1 = dependencies.get_embedder()
    embedder2 = dependencies.get_embedder()

    assert embedder1 == embedder2


def test_get_extractor_gemini(mocker):
    """Test get_extractor returns GeminiEntityExtractor when API key is present."""
    mocker.patch("app.interfaces.dependencies.settings.gemini_api_key", "test_key")
    mocker.patch("app.interfaces.dependencies.GeminiEntityExtractor", return_value=MagicMock())

    extractor = dependencies.get_extractor()
    assert extractor.__class__.__name__ != "GLiNERFallbackExtractor"


def test_get_extractor_fallback(mocker):
    """Test get_extractor returns GLiNERFallbackExtractor when API key is missing."""
    mocker.patch("app.interfaces.dependencies.settings.gemini_api_key", None)
    mocker.patch("app.interfaces.dependencies.GLiNERFallbackExtractor", return_value=MagicMock())

    extractor = dependencies.get_extractor()
    # If the patch worked, it should be the mock
    assert isinstance(extractor, MagicMock)


def test_get_resolver():
    """Test get_resolver returns JaroWinklerResolver."""
    resolver = dependencies.get_resolver()
    assert resolver.__class__.__name__ == "JaroWinklerResolver"


def test_get_linker():
    """Test get_linker returns SimpleEntityLinker."""
    mock_extractor = MagicMock()
    linker = dependencies.get_linker(mock_extractor)
    assert linker.__class__.__name__ == "SimpleEntityLinker"
    assert linker.extractor == mock_extractor


def test_get_reranker():
    """Test get_reranker returns MockReranker."""
    reranker = dependencies.get_reranker()
    assert reranker.__class__.__name__ == "MockReranker"


def test_get_ingestion_use_case():
    """Test get_ingestion_use_case returns DocumentIngestionUseCase."""
    use_case = dependencies.get_ingestion_use_case(
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    assert isinstance(use_case, DocumentIngestionUseCase)


def test_get_chat_use_case():
    """Test get_chat_use_case returns ChatUseCase."""
    use_case = dependencies.get_chat_use_case(MagicMock(), MagicMock())
    assert isinstance(use_case, ChatUseCase)


@pytest.mark.asyncio
async def test_init_db(mocker):
    """Test init_db calls expected methods on surreal_db."""
    mock_db = mocker.patch("app.interfaces.dependencies.surreal_db", autospec=True)
    mock_db.connect = mocker.AsyncMock()
    mock_db.signin = mocker.AsyncMock()
    mock_db.use = mocker.AsyncMock()

    # Mock stores to avoid actual schema initialization
    mocker.patch(
        "app.interfaces.dependencies.SurrealDocumentStore.initialize_schema",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "app.interfaces.dependencies.SurrealGraphStore.initialize_schema",
        new_callable=mocker.AsyncMock,
    )

    await dependencies.init_db()

    mock_db.connect.assert_called_once()
    mock_db.signin.assert_called_once()
    mock_db.use.assert_called_once()
