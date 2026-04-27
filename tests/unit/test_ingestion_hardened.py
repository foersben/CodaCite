"""Hardened unit tests for DocumentIngestionUseCase.

This module focuses on error handling, edge cases, and robust verification
of the document ingestion pipeline.
"""

from unittest.mock import AsyncMock

import pytest

from app.application.ingestion import DocumentIngestionUseCase


@pytest.fixture
def mock_coref_resolver():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock()


@pytest.fixture
def mock_document_store():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock()


@pytest.fixture
def mock_embedder():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock()


@pytest.fixture
def mock_extractor():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock()


@pytest.fixture
def mock_resolver():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock()


@pytest.fixture
def mock_graph_store():
    """Docstring generated to satisfy ruff D103."""
    store = AsyncMock()
    store.get_all_nodes.return_value = []
    return store


@pytest.fixture
def use_case(
    mock_coref_resolver,
    mock_document_store,
    mock_embedder,
    mock_extractor,
    mock_resolver,
    mock_graph_store,
):
    """Fixture for DocumentIngestionUseCase."""
    return DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        extractor=mock_extractor,
        resolver=mock_resolver,
        graph_store=mock_graph_store,
    )


@pytest.mark.asyncio
async def test_ingestion_coref_failure_fallback(
    use_case, mock_coref_resolver, mock_document_store, mock_embedder, mock_resolver
) -> None:
    """Test that ingestion continues if coreference resolution fails.

    Given: A document where the coref resolver raises an exception.
    When: The ingestion pipeline is executed.
    Then: It should log the error and proceed using the original text.
    """
    # Arrange
    mock_coref_resolver.resolve.side_effect = Exception("Coreference resolution service down")
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]
    mock_resolver.resolve_entities.return_value = []

    # Act
    text = "The user said hello."
    chunks = await use_case.execute(text=text, filename="test.txt")

    # Assert
    assert len(chunks) == 1
    assert chunks[0].text == text
    mock_document_store.save_document.assert_called_once()
    mock_document_store.save_chunks.assert_called_once()


@pytest.mark.asyncio
async def test_ingestion_embedding_failure_raises(
    use_case, mock_coref_resolver, mock_document_store, mock_embedder
) -> None:
    """Test that ingestion fails if embedding generation raises an error.

    Given: A document where the embedder raises an exception.
    When: The ingestion pipeline is executed.
    Then: It should raise the exception to the caller.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = "Resolved text"
    mock_embedder.embed_batch.side_effect = Exception("GPU out of memory")

    # Act & Assert
    with pytest.raises(Exception, match="GPU out of memory"):
        await use_case.execute(text="Input text", filename="test.txt")

    # Verify document was saved before the failure in Phase 3
    mock_document_store.save_document.assert_called_once()
    # Chunks should NOT have been saved
    mock_document_store.save_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_ingestion_manual_chunking_fallback(
    use_case, mock_coref_resolver, mock_document_store, mock_embedder, mock_resolver, mocker
) -> None:
    """Test manual chunking fallback when langchain is missing.

    Given: langchain-text-splitters is not available in the environment.
    When: chunk_text is called during ingestion.
    Then: It should fallback to manual fixed-size splitting.
    """
    # Arrange
    # Patch sys.modules to simulate missing package
    mocker.patch.dict("sys.modules", {"langchain_text_splitters": None})
    mock_coref_resolver.resolve.return_value = "A" * 2000
    mock_embedder.embed_batch.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    mock_resolver.resolve_entities.return_value = []

    # Act
    chunks = await use_case.execute(text="A" * 2000, filename="test.txt")

    # Assert
    assert len(chunks) > 1
    # Manual fallback uses 1024 chunk size
    assert len(chunks[0].text) == 1024
