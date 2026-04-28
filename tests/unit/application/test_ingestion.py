"""Unit tests for the DocumentIngestionUseCase.

Validates the ingestion pipeline, including text chunking (with fallbacks),
embedding generation, persistence coordination, and background task management.
"""

from typing import Any

import pytest

from app.application.ingestion import DocumentIngestionUseCase, chunk_text
from app.domain.models import Edge, Node


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
    mock_entity_extractor: Any,
    mock_entity_resolver: Any,
    mock_graph_store: Any,
) -> DocumentIngestionUseCase:
    """Provides a DocumentIngestionUseCase instance with mocked dependencies.

    Args:
        mock_coref_resolver: Mock coreference resolver fixture.
        mock_document_store: Mock document store fixture.
        mock_embedder: Mock embedder fixture.
        mock_entity_extractor: Mock entity extractor fixture.
        mock_entity_resolver: Mock entity resolver fixture.
        mock_graph_store: Mock graph store fixture.

    Returns:
        An initialized DocumentIngestionUseCase.
    """
    return DocumentIngestionUseCase(
        coref_resolver=mock_coref_resolver,
        document_store=mock_document_store,
        embedder=mock_embedder,
        extractor=mock_entity_extractor,
        resolver=mock_entity_resolver,
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
    chunks = await use_case.execute(text="Original text.", filename="test.md")

    # Assert
    assert len(chunks) == 1
    mock_document_store.save_document.assert_called_once()
    mock_document_store.save_chunks.assert_called_once()


@pytest.mark.asyncio
async def test_ingestion_coref_failure_fallback(
    use_case: DocumentIngestionUseCase,
    mock_coref_resolver: Any,
    mock_embedder: Any,
) -> None:
    """Tests that ingestion continues if coreference resolution fails."""
    # Arrange
    mock_coref_resolver.resolve.side_effect = Exception("Service down")
    mock_embedder.embed_batch.return_value = [[0.1] * 1024]

    # Act
    text = "Original text."
    chunks = await use_case.execute(text=text, filename="test.txt")

    # Assert
    assert len(chunks) == 1
    assert chunks[0].text == text


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
    mock_entity_extractor: Any,
    mock_entity_resolver: Any,
    mock_graph_store: Any,
) -> None:
    """Tests successful background processing of a document."""
    # Arrange
    doc_id = "doc:123"
    text = "Alice knows Bob."
    mock_coref_resolver.resolve.return_value = text
    mock_embedder.embed_batch.return_value = [[0.1]]
    mock_embedder.embed.return_value = [0.1]

    node1 = Node(id="alice", label="PERSON", name="Alice")
    node2 = Node(id="bob", label="PERSON", name="Bob")
    edge = Edge(source_id="alice", target_id="bob", relation="KNOWS")

    mock_entity_extractor.extract.return_value = ([node1, node2], [edge])
    mock_graph_store.get_all_nodes.return_value = []
    mock_entity_resolver.resolve_entities.return_value = [node1, node2]

    # Act
    await use_case.process_background(doc_id, text=text, filename="test.md")

    # Assert
    assert f"{doc_id}_0" in node1.source_chunk_ids
    mock_graph_store.save_nodes.assert_called_once()
    mock_graph_store.save_edges.assert_called_once()
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
