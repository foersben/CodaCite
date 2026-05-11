"""Unit tests for document summarization helpers."""

from typing import Any

import pytest

from app.pipelines.ingestion.summarization import DocumentSummarizer


@pytest.fixture
def summarizer(mock_llm_generator: Any) -> DocumentSummarizer:
    """Provide a summarizer with a mocked LLM generator."""
    return DocumentSummarizer(llm_generator=mock_llm_generator)


def test_create_super_chunks_does_not_prefix_first_chunk(summarizer: DocumentSummarizer) -> None:
    """Tests that the first combined chunk does not start with separator whitespace."""
    # Arrange
    chunks = ["alpha", "beta"]

    # Act
    super_chunks = summarizer._create_super_chunks(chunks, max_tokens=10)

    # Assert
    assert super_chunks == ["alpha\n\nbeta"]


def test_create_super_chunks_handles_initial_overflow(summarizer: DocumentSummarizer) -> None:
    """Tests that an oversized first chunk does not create an empty super-chunk."""
    # Arrange
    chunks = ["A" * 10, "B"]

    # Act
    super_chunks = summarizer._create_super_chunks(chunks, max_tokens=2)

    # Assert
    assert super_chunks == ["A" * 10, "B"]
