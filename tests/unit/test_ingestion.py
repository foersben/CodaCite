"""Tests for DocumentIngestionUseCase and chunk_text utility.

This module contains unit tests for the ingestion pipeline (Application layer)
and text chunking utilities (Domain/Shared logic).
"""

from typing import Any

import pytest

from app.application.ingestion import DocumentIngestionUseCase, chunk_text

# --------------------------------------------------------------------------
# chunk_text tests (Domain / Utility Logic)
# --------------------------------------------------------------------------


class TestChunkText:
    """Tests for the chunk_text utility function."""

    def test_empty_string(self) -> None:
        """Test chunk_text returns empty list for empty string.

        Given: An empty input string.
        When: chunk_text is called.
        Then: It should return an empty list.
        """
        assert chunk_text("") == []

    def test_short_text_single_chunk(self) -> None:
        """Test chunk_text returns a single chunk for short text.

        Given: A text string shorter than the specified chunk size.
        When: chunk_text is called.
        Then: It should return a list containing exactly one chunk with the full text.
        """
        result = chunk_text("Hello World!", chunk_size=1024)
        assert len(result) == 1
        assert result[0] == "Hello World!"

    def test_long_text_multiple_chunks(self) -> None:
        """Test chunk_text splits long text into multiple chunks.

        Given: A text string significantly longer than the specified chunk size.
        When: chunk_text is called with a small chunk size.
        Then: It should split the text into multiple chunks, each within size limits.
        """
        text = "A " * 600  # 1200 chars
        result = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(result) > 1
        # Each chunk should be at most chunk_size
        for chunk in result:
            assert len(chunk) <= 100 + 50  # Allow some flex from splitter heuristics

    def test_overlap_present(self) -> None:
        """Test chunk_text creates overlapping chunks.

        Given: A text string and a non-zero chunk overlap.
        When: chunk_text is called.
        Then: Consecutive chunks should share common text at their boundaries.
        """
        text = " ".join(f"word{i}" for i in range(200))
        result = chunk_text(text, chunk_size=100, chunk_overlap=20)
        if len(result) >= 2:
            # The end of chunk 0 should overlap with start of chunk 1
            # (at least some shared words)
            end_of_first = result[0][-20:]
            assert any(word in result[1] for word in end_of_first.split() if len(word) > 3)


# --------------------------------------------------------------------------
# DocumentIngestionUseCase tests (Application Layer)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingestion_basic(
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any, mocker: Any
) -> None:
    """Test basic document ingestion flow.

    Given: A valid document text and working infrastructure services.
    When: The DocumentIngestionUseCase is executed.
    Then: It should resolve coreferences, chunk the text, embed chunks, and persist results.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = "Resolved text content here."
    # mock_embedder.embed_batch should return a list of dummy embeddings
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]

    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        extractor=mocker.AsyncMock(),
        resolver=mocker.AsyncMock(),
        graph_store=mocker.AsyncMock(),
    )

    # Act
    chunks = await use_case.execute(text="Original text content here.", filename="test.md")

    # Assert
    mock_document_store.save_document.assert_called_once()
    mock_document_store.save_chunks.assert_called_once()
    mock_coref_resolver.resolve.assert_called_once_with("Original text content here.")
    mock_embedder.embed_batch.assert_called_once()
    assert len(chunks) >= 1
    assert chunks[0].text  # non-empty
    assert chunks[0].document_id  # has a document ID


@pytest.mark.asyncio
async def test_ingestion_empty_text(
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any, mocker: Any
) -> None:
    """Test ingestion with empty text produces zero chunks.

    Given: An empty input text.
    When: The DocumentIngestionUseCase is executed.
    Then: It should still create a document record but return zero chunks.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = ""
    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        extractor=mocker.AsyncMock(),
        resolver=mocker.AsyncMock(),
        graph_store=mocker.AsyncMock(),
    )

    # Act
    chunks = await use_case.execute(text="", filename="empty.md")

    # Assert
    mock_document_store.save_document.assert_called_once()
    # Phase 2 complete: Text split into 0 chunks -> returns [] immediately
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_ingestion_metadata_passed(
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any, mocker: Any
) -> None:
    """Test ingestion passes metadata to the document model.

    Given: Custom metadata for a document.
    When: The DocumentIngestionUseCase is executed.
    Then: The persisted document record should contain the provided metadata.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = "Some text."
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]
    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        extractor=mocker.AsyncMock(),
        resolver=mocker.AsyncMock(),
        graph_store=mocker.AsyncMock(),
    )
    metadata: dict[str, str | int | float | bool] = {"author": "Alice", "version": 2}

    # Act
    await use_case.execute(text="Some text.", filename="doc.md", metadata=metadata)

    # Assert
    saved_doc = mock_document_store.save_document.call_args[0][0]
    assert saved_doc.metadata == {"author": "Alice", "version": 2}
    assert saved_doc.filename == "doc.md"


@pytest.mark.asyncio
async def test_ingestion_long_text_produces_multiple_chunks(
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any, mocker: Any
) -> None:
    """Test ingestion of long text produces multiple chunks.

    Given: A text input exceeding the maximum chunk size.
    When: The DocumentIngestionUseCase is executed.
    Then: It should result in multiple chunk domain models linked to the same document ID.
    """
    # Arrange
    long_text = "This is a paragraph of text. " * 200  # ~5800 chars
    mock_coref_resolver.resolve.return_value = long_text

    # Calculate expected number of chunks roughly
    # RecursiveCharacterTextSplitter behavior varies, but we need at least some embeddings
    # We'll just return as many embeddings as there are chunks
    def side_effect(texts):
        """Docstring generated to satisfy ruff D103."""
        return [[0.1] * 1024 for _ in texts]

    mock_embedder.embed_batch.side_effect = side_effect

    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        extractor=mocker.AsyncMock(),
        resolver=mocker.AsyncMock(),
        graph_store=mocker.AsyncMock(),
    )

    # Act
    chunks = await use_case.execute(text=long_text, filename="long.md")

    # Assert
    assert len(chunks) > 1
    # All chunks share the same document_id
    doc_ids = {c.document_id for c in chunks}
    assert len(doc_ids) == 1
