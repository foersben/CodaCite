"""Unit tests for the DocumentIngestionUseCase.

Validates the ingestion pipeline, including text chunking,
embedding generation, persistence coordination, and background task management.
"""

from typing import Any

import pytest

from app.core.interfaces import Chunker
from app.pipelines.ingestion.ingestion import DocumentIngestionUseCase


@pytest.fixture
def mock_chunker(mocker: Any) -> Any:
    """Mock chunker fixture."""
    return mocker.AsyncMock(spec=Chunker)


@pytest.fixture
def use_case(
    mock_coref_resolver: Any,
    mock_document_store: Any,
    mock_embedder: Any,
    mock_chunker: Any,
    mock_extraction_use_case: Any,
    mock_graph_store: Any,
    mock_llm_generator: Any,
) -> DocumentIngestionUseCase:
    """Provides a DocumentIngestionUseCase instance with mocked dependencies."""
    return DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        chunker=mock_chunker,
        graph_extraction_use_case=mock_extraction_use_case,
        graph_store=mock_graph_store,
        llm_generator=mock_llm_generator,
    )


@pytest.mark.asyncio
async def test_ingestion_basic_flow(
    use_case: DocumentIngestionUseCase,
    mock_coref_resolver: Any,
    mock_document_store: Any,
    mock_embedder: Any,
    mock_chunker: Any,
) -> None:
    """Tests the basic document ingestion flow."""
    # Arrange
    mock_coref_resolver.resolve.return_value = "Resolved text."
    mock_chunker.chunk.return_value = [{"text": "chunk1", "start_char": 0, "end_char": 6}]
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]

    # Act
    await use_case.process_background(
        document_id="doc:1", text="Original text.", filename="test.md"
    )

    # Assert
    mock_document_store.save_chunks.assert_called_once()
    mock_document_store.update_document_status.assert_called_with("doc:1", "active")


@pytest.mark.asyncio
async def test_ingestion_coref_failure_fallback(
    use_case: DocumentIngestionUseCase,
    mock_document_store: Any,
    mock_coref_resolver: Any,
    mock_embedder: Any,
    mock_chunker: Any,
) -> None:
    """Tests that ingestion continues if coreference resolution fails."""
    # Arrange
    mock_coref_resolver.resolve.side_effect = Exception("Service down")
    mock_chunker.chunk.return_value = [{"text": "chunk1", "start_char": 0, "end_char": 6}]
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
    mock_chunker: Any,
    mock_extraction_use_case: Any,
    mock_graph_store: Any,
) -> None:
    """Tests successful background processing of a document."""
    # Arrange
    doc_id = "doc:123"
    text = "Alice knows Bob."
    mock_coref_resolver.resolve.return_value = text
    mock_chunker.chunk.return_value = [{"text": text, "start_char": 0, "end_char": len(text)}]
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
    mock_chunker: Any,
) -> None:
    """Tests that background processing failure updates status to 'failed'."""
    # Arrange
    doc_id = "doc:666"
    # Make it fail by returning empty chunks
    mock_chunker.chunk.return_value = []

    # Act
    await use_case.process_background(doc_id, text="Fail", filename="bad.txt")

    # Assert
    mock_document_store.update_document_status.assert_called_with(doc_id, "failed")
