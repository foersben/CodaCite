"""Unit tests for the DocumentIngestionUseCase.

Validates the ingestion pipeline, including text chunking (with fallbacks),
embedding generation, persistence coordination, and background task management.
"""

from typing import Any

import pytest

from app.application.ingestion import DocumentIngestionUseCase, chunk_text


class TestChunkText:
    """Tests for the chunk_text utility function."""

    def test_empty_string(self) -> None:
        """Tests that empty string returns empty list."""
        assert chunk_text("") == []

    def test_short_text_single_chunk(self) -> None:
        """Tests that short text returns a single chunk."""
        result = chunk_text("Hello World!", chunk_size=1024)
        assert len(result) == 1
        assert result[0] == "Hello World!"

    def test_long_text_multiple_chunks(self) -> None:
        """Tests that long text is split correctly."""
        text = "A " * 600  # 1200 chars
        result = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(result) > 1

    def test_manual_chunking_fallback(self, mocker: Any) -> None:
        """Tests manual chunking fallback when langchain is missing."""
        mocker.patch.dict("sys.modules", {"langchain_text_splitters": None})
        text = "A" * 2000
        result = chunk_text(text, chunk_size=1024, chunk_overlap=128)
        # step = 1024 - 128 = 896
        # range(0, 2000, 896) -> [0, 896, 1792] -> 3 chunks
        assert len(result) == 3
        assert len(result[0]) == 1024


@pytest.fixture
def use_case(
    mock_coref_resolver: Any,
    mock_document_store: Any,
    mock_embedder: Any,
    mock_extraction_use_case: Any,
    mock_graph_store: Any,
) -> DocumentIngestionUseCase:
    """Provides a DocumentIngestionUseCase instance with mocked dependencies.

    Args:
        mock_coref_resolver: Mock coreference resolver fixture.
        mock_document_store: Mock document store fixture.
        mock_embedder: Mock embedder fixture.
        mock_extraction_use_case: Mock graph extraction use case fixture.
        mock_graph_store: Mock graph store fixture.

    Returns:
        An initialized DocumentIngestionUseCase.
    """
    return DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        graph_extraction_use_case=mock_extraction_use_case,
        graph_store=mock_graph_store,
    )


@pytest.mark.asyncio
async def test_ingestion_basic_flow(
    use_case: DocumentIngestionUseCase,
    mock_coref_resolver: Any,
    mock_document_store: Any,
    mock_embedder: Any,
) -> None:
    """Tests the basic document ingestion flow."""
    # Arrange
    mock_coref_resolver.resolve.return_value = "Resolved text."
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]

    # Act
    await use_case.process_background(
        document_id="doc:1", text="Original text.", filename="test.md"
    )

    # Assert
    # Assert
    mock_document_store.save_chunks.assert_called_once()
    mock_document_store.update_document_status.assert_called_with("doc:1", "active")


@pytest.mark.asyncio
async def test_ingestion_coref_failure_fallback(
    use_case: DocumentIngestionUseCase,
    mock_document_store: Any,
    mock_coref_resolver: Any,
    mock_embedder: Any,
) -> None:
    """Tests that ingestion continues if coreference resolution fails."""
    # Arrange
    mock_coref_resolver.resolve.side_effect = Exception("Service down")
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]

    # Act
    text = "Original text."
    await use_case.process_background(document_id="doc:2", text=text, filename="test.txt")

    # Assert
    mock_document_store.update_document_status.assert_called_with("doc:2", "active")


@pytest.mark.asyncio
async def test_ingest_and_queue_starts_task(
    use_case: DocumentIngestionUseCase,
    mock_document_store: Any,
) -> None:
    """Tests that ingest_and_queue correctly initializes and persists a document."""
    # Act
    doc_id = await use_case.ingest_and_queue(
        text="Sample text", filename="test.txt", notebook_id="notebook:1"
    )

    # Assert
    assert isinstance(doc_id, str)
    mock_document_store.save_document.assert_called_once()
    mock_document_store.add_document_to_notebook.assert_called_once_with(doc_id, "notebook:1")


@pytest.mark.asyncio
async def test_process_background_success(
    use_case: DocumentIngestionUseCase,
    mock_document_store: Any,
    mock_coref_resolver: Any,
    mock_embedder: Any,
    mock_extraction_use_case: Any,
    mock_graph_store: Any,
) -> None:
    """Tests successful background processing of a document."""
    # Arrange
    doc_id = "doc:123"
    text = "Alice knows Bob."
    mock_coref_resolver.resolve.return_value = text
    mock_embedder.embed_batch.return_value = [[0.1]]
    mock_embedder.embed.return_value = [0.1]

    # Act
    await use_case.process_background(doc_id, text=text, filename="test.md")

    # Assert
    mock_extraction_use_case.execute.assert_called_once()
    mock_document_store.update_document_status.assert_called_with(doc_id, "active")


@pytest.mark.asyncio
async def test_process_background_failure_updates_status(
    use_case: DocumentIngestionUseCase,
    mock_document_store: Any,
    mock_coref_resolver: Any,
) -> None:
    """Tests that background processing failure updates status to 'failed'."""
    # Arrange
    doc_id = "doc:666"
    mock_coref_resolver.resolve.side_effect = RuntimeError("Crash")

    # Act
    await use_case.process_background(doc_id, text="Fail", filename="bad.txt")

    # Assert
    mock_document_store.update_document_status.assert_called_with(doc_id, "failed")
