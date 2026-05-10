"""Unit tests for the SemanticChunker.

Verifies that text is split into semantic chunks with correct character offsets
and cosine similarity groupings using BGE-M3.
"""

from typing import Any

import pytest

from app.pipelines.ingestion.chunkers import SemanticChunker


@pytest.mark.asyncio
async def test_semantic_chunker_basic(mocker: Any) -> None:
    """Tests that SemanticChunker groups sentences and returns offsets."""
    from app.core.interfaces import Embedder

    mock_embedder = mocker.AsyncMock(spec=Embedder)

    # Mock embeddings for sentences
    # Sent 1: "Hello."
    # Sent 2: "World."
    # Sent 3: "Something different."
    # We'll make Sent 1 and 2 similar, Sent 3 different.
    mock_embedder.embed_batch.return_value = [
        [1.0, 0.0],  # Sent 1
        [0.9, 0.1],  # Sent 2 (similar to 1)
        [0.0, 1.0],  # Sent 3 (different)
    ]

    chunker = SemanticChunker(mock_embedder, similarity_threshold=0.5, min_chunk_size=0)
    text = "Hello. World. Something different."

    chunks = await chunker.chunk(text)

    # Should result in 2 chunks:
    # 1: "Hello. World."
    # 2: "Something different."
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Hello. World."
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == 13  # "Hello. World." is 13 chars

    assert chunks[1]["text"] == "Something different."
    assert chunks[1]["start_char"] == 14
    assert chunks[1]["end_char"] == 34


@pytest.mark.asyncio
async def test_semantic_chunker_empty_string(mocker: Any) -> None:
    """Tests that empty string returns empty list."""
    from app.core.interfaces import Embedder

    mock_embedder = mocker.AsyncMock(spec=Embedder)
    chunker = SemanticChunker(mock_embedder)

    assert await chunker.chunk("") == []


@pytest.mark.asyncio
async def test_semantic_chunker_single_sentence(mocker: Any) -> None:
    """Tests that a single sentence returns a single chunk."""
    from app.core.interfaces import Embedder

    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_embedder.embed_batch.return_value = [[1.0]]

    chunker = SemanticChunker(mock_embedder)
    text = "Just one sentence."

    chunks = await chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Just one sentence."
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len(text)
