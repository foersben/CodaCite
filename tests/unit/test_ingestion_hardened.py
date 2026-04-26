"""Hardened unit tests for DocumentIngestionUseCase.

This module focuses on error handling, edge cases, and robust verification
of the document ingestion pipeline.
"""

from unittest.mock import AsyncMock

import pytest

from app.application.ingestion import DocumentIngestionUseCase


@pytest.mark.asyncio
async def test_ingestion_coref_failure_fallback(
    mock_coref_resolver: AsyncMock, mock_document_store: AsyncMock, mock_embedder: AsyncMock
) -> None:
    """Test that ingestion continues if coreference resolution fails.

    Given: A document where the coref resolver raises an exception.
    When: The ingestion pipeline is executed.
    Then: It should log the error and proceed using the original text.
    """
    # Arrange
    mock_coref_resolver.resolve.side_effect = Exception("Coreference resolution service down")
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]

    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
    )

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
    mock_coref_resolver: AsyncMock, mock_document_store: AsyncMock, mock_embedder: AsyncMock
) -> None:
    """Test that ingestion fails if embedding generation raises an error.

    Given: A document where the embedder raises an exception.
    When: The ingestion pipeline is executed.
    Then: It should raise the exception to the caller.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = "Resolved text"
    mock_embedder.embed_batch.side_effect = Exception("GPU out of memory")

    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
    )

    # Act & Assert
    with pytest.raises(Exception, match="GPU out of memory"):
        await use_case.execute(text="Input text", filename="test.txt")

    # Verify document was saved before the failure in Phase 3
    mock_document_store.save_document.assert_called_once()
    # Chunks should NOT have been saved
    mock_document_store.save_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_ingestion_manual_chunking_fallback(
    mock_coref_resolver: AsyncMock, mock_document_store: AsyncMock, mock_embedder: AsyncMock, mocker
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

    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
    )

    # Act
    chunks = await use_case.execute(text="A" * 2000, filename="test.txt")

    # Assert
    assert len(chunks) > 1
    # Manual fallback uses 1024 chunk size
    assert len(chunks[0].text) == 1024
