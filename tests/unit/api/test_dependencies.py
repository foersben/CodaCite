"""Unit tests for the application dependencies.

Validates the dependency injection providers, ensuring that all components
are correctly instantiated and singletons are managed properly.
"""

from typing import Any

import pytest

from app.api import dependencies
from app.db.store import SurrealDocumentStore, SurrealGraphStore
from app.pipelines.generation.chat import ChatUseCase
from app.pipelines.ingestion.ingestion import DocumentIngestionUseCase


def test_get_db() -> None:
    """Tests that get_db returns the global surreal_db instance.

    Given:
        The dependencies module is imported.
    When:
        get_db is called.
    Then:
        It should return the global surreal_db instance.
    """
    db = dependencies.get_db()
    assert db == dependencies.surreal_db


def test_get_document_store(mocker: Any) -> None:
    """Tests that get_document_store returns a SurrealDocumentStore.

    Given:
        A mocked database instance.
    When:
        get_document_store is called.
    Then:
        It should return a SurrealDocumentStore instance linked to that database.

    Args:
        mocker: The pytest-mock fixture.
    """
    mock_db = mocker.MagicMock()
    store = dependencies.get_document_store(mock_db)
    assert isinstance(store, SurrealDocumentStore)
    assert store.db == mock_db


def test_get_graph_store(mocker: Any) -> None:
    """Tests that get_graph_store returns a SurrealGraphStore.

    Given:
        A mocked database instance.
    When:
        get_graph_store is called.
    Then:
        It should return a SurrealGraphStore instance linked to that database.

    Args:
        mocker: The pytest-mock fixture.
    """
    mock_db = mocker.MagicMock()
    store = dependencies.get_graph_store(mock_db)
    assert isinstance(store, SurrealGraphStore)
    assert store.db == mock_db


def test_get_coref_resolver(mocker: Any) -> None:
    """Tests that get_coref_resolver returns the correct implementation.

    Given:
        The dependencies module.
    When:
        get_coref_resolver is called.
    Then:
        It should return a mocked FastCorefResolver instance.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Reset singleton
    dependencies._coref_resolver = None

    mock_resolver = mocker.MagicMock()
    mocker.patch("app.api.dependencies.FastCorefResolver", return_value=mock_resolver)
    resolver = dependencies.get_coref_resolver()
    assert resolver == mock_resolver


def test_get_embedder(mocker: Any) -> None:
    """Tests that get_embedder returns a singleton HuggingFaceEmbedder.

    Given:
        The dependencies module with a reset singleton.
    When:
        get_embedder is called multiple times.
    Then:
        It should return the same instance every time.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Patch the SentenceTransformerEmbedder to avoid actual loading
    mocker.patch(
        "app.api.dependencies.SentenceTransformerEmbedder", return_value=mocker.MagicMock()
    )

    # Reset singleton for test
    dependencies._embedder = None

    embedder1 = dependencies.get_embedder()
    embedder2 = dependencies.get_embedder()

    assert embedder1 == embedder2


def test_get_extractor_gemini(mocker: Any) -> None:
    """Tests that get_extractor returns GeminiEntityExtractor when API key is present.

    Given:
        A settings mock with a valid gemini_api_key.
    When:
        get_extractor is called.
    Then:
        It should return a GeminiEntityExtractor instance.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Reset singleton
    dependencies._extractor = None

    mocker.patch("app.api.dependencies.settings.gemini_api_key", "test_key")
    mocker.patch("app.api.dependencies.settings.use_local_nlp_models", False)
    mocker.patch("app.api.dependencies.GeminiEntityExtractor", return_value=mocker.MagicMock())

    extractor = dependencies.get_extractor()
    assert extractor.__class__.__name__ != "GLiNERFallbackExtractor"


def test_get_extractor_fallback(mocker: Any) -> None:
    """Tests that get_extractor returns GLiNERFallbackExtractor when API key is missing.

    Given:
        A settings mock with no gemini_api_key.
    When:
        get_extractor is called.
    Then:
        It should return a GLiNERFallbackExtractor instance.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Reset singleton
    dependencies._extractor = None

    mocker.patch("app.api.dependencies.settings.gemini_api_key", None)
    mock_fallback = mocker.MagicMock()
    mocker.patch("app.api.dependencies.GLiNERFallbackExtractor", return_value=mock_fallback)

    extractor = dependencies.get_extractor()
    assert extractor == mock_fallback


def test_get_resolver() -> None:
    """Tests that get_resolver returns JaroWinklerResolver.

    Given:
        The dependencies module.
    When:
        get_resolver is called.
    Then:
        It should return a JaroWinklerResolver instance.
    """
    resolver = dependencies.get_resolver()
    assert resolver.__class__.__name__ == "JaroWinklerResolver"


def test_get_linker(mocker: Any) -> None:
    """Tests that get_linker returns SimpleEntityLinker.

    Given:
        A mocked extractor.
    When:
        get_linker is called.
    Then:
        It should return a SimpleEntityLinker instance linked to that extractor.

    Args:
        mocker: The pytest-mock fixture.
    """
    mock_extractor = mocker.MagicMock()
    linker = dependencies.get_linker(mock_extractor)
    assert linker.__class__.__name__ == "SimpleEntityLinker"
    assert linker.extractor == mock_extractor


def test_get_reranker(mocker: Any) -> None:
    """Tests that get_reranker returns a Reranker singleton.

    Given:
        The dependencies module.
    When:
        get_reranker is called.
    Then:
        It should return a ModernBertReranker instance (mocked).
    """
    dependencies._reranker = None
    mock_reranker = mocker.AsyncMock()
    mocker.patch("app.api.dependencies.ModernBertReranker", return_value=mock_reranker)

    reranker = dependencies.get_reranker()
    assert reranker == mock_reranker


def test_get_ingestion_use_case(mocker: Any) -> None:
    """Tests that get_ingestion_use_case returns DocumentIngestionUseCase.

    Given:
        Mocked dependencies for ingestion.
    When:
        get_ingestion_use_case is called.
    Then:
        It should return a DocumentIngestionUseCase instance.

    Args:
        mocker: The pytest-mock fixture.
    """
    use_case = dependencies.get_ingestion_use_case(
        mocker.MagicMock(),  # coref_resolver
        mocker.MagicMock(),  # document_store
        mocker.MagicMock(),  # embedder
        mocker.MagicMock(),  # chunker
        mocker.MagicMock(),  # graph_extraction_use_case
        mocker.MagicMock(),  # graph_store
        mocker.MagicMock(),  # llm_generator
    )
    assert isinstance(use_case, DocumentIngestionUseCase)


def test_get_chat_use_case(mocker: Any) -> None:
    """Tests that get_chat_use_case returns ChatUseCase.

    Given:
        Mocked dependencies for chat.
    When:
        get_chat_use_case is called.
    Then:
        It should return a ChatUseCase instance.

    Args:
        mocker: The pytest-mock fixture.
    """
    use_case = dependencies.get_chat_use_case(mocker.MagicMock(), mocker.MagicMock())
    assert isinstance(use_case, ChatUseCase)


@pytest.mark.asyncio
async def test_init_db(mocker: Any) -> None:
    """Tests that init_db calls expected connection methods on surreal_db.

    Given:
        The dependencies module with a mocked database.
    When:
        init_db is called.
    Then:
        It should connect, sign in, and use the correct namespace/database.

    Args:
        mocker: The pytest-mock fixture.
    """
    mock_db = mocker.patch("app.api.dependencies.surreal_db", autospec=True)
    mock_db.connect = mocker.AsyncMock()
    mock_db.signin = mocker.AsyncMock()
    mock_db.use = mocker.AsyncMock()

    # Mock stores to avoid actual schema initialization
    mocker.patch(
        "app.api.dependencies.SurrealDocumentStore.initialize_schema",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "app.api.dependencies.SurrealGraphStore.initialize_schema",
        new_callable=mocker.AsyncMock,
    )

    await dependencies.init_db()

    mock_db.connect.assert_called_once()
    mock_db.signin.assert_called_once()
    mock_db.use.assert_called_once()


def test_get_extraction_use_case(mocker: Any) -> None:
    """Tests get_extraction_use_case provider."""
    use_case = dependencies.get_extraction_use_case(
        mocker.MagicMock(), mocker.MagicMock(), mocker.MagicMock(), mocker.MagicMock()
    )
    assert use_case.__class__.__name__ == "GraphExtractionUseCase"


def test_get_retrieval_use_case(mocker: Any) -> None:
    """Tests get_retrieval_use_case provider."""
    use_case = dependencies.get_retrieval_use_case(
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
    )
    assert use_case.__class__.__name__ == "GraphRAGRetrievalUseCase"


def test_get_enhancement_use_case(mocker: Any) -> None:
    """Tests get_enhancement_use_case provider."""
    use_case = dependencies.get_enhancement_use_case(mocker.MagicMock())
    assert use_case.__class__.__name__ == "GraphEnhancementUseCase"


def test_get_notebook_use_case(mocker: Any) -> None:
    """Tests get_notebook_use_case provider."""
    use_case = dependencies.get_notebook_use_case(mocker.MagicMock())
    assert use_case.__class__.__name__ == "NotebookUseCase"


def test_get_generator_gemini(mocker: Any) -> None:
    """Tests get_generator provider for Gemini path."""
    # Reset singleton
    dependencies._generator = None

    mocker.patch("app.api.dependencies.settings.use_local_nlp_models", False)
    mocker.patch("app.api.dependencies.GeminiGenerator", return_value=mocker.MagicMock())
    gen = dependencies.get_generator()
    assert gen is not None


def test_get_generator_local(mocker: Any) -> None:
    """Tests get_generator provider for Local path."""
    # Reset singleton
    dependencies._generator = None

    mocker.patch("app.api.dependencies.settings.use_local_nlp_models", True)
    mocker.patch("app.api.dependencies.settings.local_llm_path", "/path/to/model.gguf")
    mocker.patch("app.api.dependencies.Path.exists", return_value=True)
    mocker.patch("app.api.dependencies.LocalLlamaGenerator", return_value=mocker.MagicMock())
    gen = dependencies.get_generator()
    assert gen is not None


def test_get_generator_error(mocker: Any) -> None:
    """Tests get_generator raises error if local is enabled but path is missing."""
    # Reset singleton
    dependencies._generator = None

    mocker.patch("app.api.dependencies.settings.use_local_nlp_models", True)
    mocker.patch("app.api.dependencies.settings.local_llm_path", "")
    with pytest.raises(RuntimeError, match="LOCAL_LLM_PATH"):
        dependencies.get_generator()
