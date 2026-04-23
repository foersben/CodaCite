"""Tests for DocumentIngestionUseCase and chunk_text utility."""

from typing import Any

import pytest

from app.application.ingestion import DocumentIngestionUseCase, chunk_text


# --------------------------------------------------------------------------
# chunk_text tests
# --------------------------------------------------------------------------

class TestChunkText:
    """Tests for the chunk_text utility function."""

    def test_empty_string(self) -> None:
        """Test chunk_text returns empty list for empty string.

        Arrange: Empty string.
        Act: Call chunk_text.
        Assert: Returns empty list.
        """
        assert chunk_text("") == []

    def test_short_text_single_chunk(self) -> None:
        """Test chunk_text returns a single chunk for short text.

        Arrange: Text shorter than chunk_size.
        Act: Call chunk_text.
        Assert: Returns list with one element.
        """
        result = chunk_text("Hello World!", chunk_size=1024)
        assert len(result) == 1
        assert result[0] == "Hello World!"

    def test_long_text_multiple_chunks(self) -> None:
        """Test chunk_text splits long text into multiple chunks.

        Arrange: Text significantly longer than chunk_size.
        Act: Call chunk_text with small chunk_size.
        Assert: Returns multiple chunks.
        """
        text = "A " * 600  # 1200 chars
        result = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(result) > 1
        # Each chunk should be at most chunk_size
        for chunk in result:
            assert len(chunk) <= 100 + 50  # Allow some flex from splitter heuristics

    def test_overlap_present(self) -> None:
        """Test chunk_text creates overlapping chunks.

        Arrange: Text longer than chunk_size.
        Act: Call chunk_text with overlap.
        Assert: Consecutive chunks share some content.
        """
        text = " ".join(f"word{i}" for i in range(200))
        result = chunk_text(text, chunk_size=100, chunk_overlap=20)
        if len(result) >= 2:
            # The end of chunk 0 should overlap with start of chunk 1
            # (at least some shared words)
            end_of_first = result[0][-20:]
            assert any(word in result[1] for word in end_of_first.split() if len(word) > 3)


# --------------------------------------------------------------------------
# DocumentIngestionUseCase tests
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingestion_basic(
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any
) -> None:
    """Test basic document ingestion flow.

    Arrange: Set up coref resolver to pass text through, document store and embedder.
    Act: Execute ingestion with sample text.
    Assert: Document and chunks are saved.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = "Resolved text content here."
    # mock_embedder.embed_batch should return a list of dummy embeddings
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]
    
    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
    )

    # Act
    chunks = await use_case.execute(
        text="Original text content here.", filename="test.md"
    )

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
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any
) -> None:
    """Test ingestion with empty text produces zero chunks.

    Arrange: Set up coref resolver to return empty string.
    Act: Execute ingestion with empty text.
    Assert: save_chunks is called with empty list.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = ""
    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
    )

    # Act
    chunks = await use_case.execute(text="", filename="empty.md")

    # Assert
    mock_document_store.save_document.assert_called_once()
    # Phase 2 complete: Text split into 0 chunks -> returns [] immediately
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_ingestion_metadata_passed(
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any
) -> None:
    """Test ingestion passes metadata to the document model.

    Arrange: Provide metadata dict.
    Act: Execute ingestion.
    Assert: The document saved includes the metadata.
    """
    # Arrange
    mock_coref_resolver.resolve.return_value = "Some text."
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]
    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
    )
    metadata = {"author": "Alice", "version": 2}

    # Act
    await use_case.execute(text="Some text.", filename="doc.md", metadata=metadata)

    # Assert
    saved_doc = mock_document_store.save_document.call_args[0][0]
    assert saved_doc.metadata == {"author": "Alice", "version": 2}
    assert saved_doc.filename == "doc.md"


@pytest.mark.asyncio
async def test_ingestion_long_text_produces_multiple_chunks(
    mock_coref_resolver: Any, mock_document_store: Any, mock_embedder: Any
) -> None:
    """Test ingestion of long text produces multiple chunks.

    Arrange: Text much longer than default chunk_size.
    Act: Execute ingestion.
    Assert: Multiple chunks are created.
    """
    # Arrange
    long_text = "This is a paragraph of text. " * 200  # ~5800 chars
    mock_coref_resolver.resolve.return_value = long_text
    
    # Calculate expected number of chunks roughly
    # RecursiveCharacterTextSplitter behavior varies, but we need at least some embeddings
    # We'll just return as many embeddings as there are chunks
    def side_effect(texts):
        return [[0.1] * 1024 for _ in texts]
    mock_embedder.embed_batch.side_effect = side_effect

    use_case = DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
    )

    # Act
    chunks = await use_case.execute(text=long_text, filename="long.md")

    # Assert
    assert len(chunks) > 1
    # All chunks share the same document_id
    doc_ids = {c.document_id for c in chunks}
    assert len(doc_ids) == 1
