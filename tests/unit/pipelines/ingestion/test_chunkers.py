"""Unit tests for the StructuralContextChunker.

Verifies that text is split into structural chunks with correct character offsets
while optionally prepending document context.
"""

import pytest

from app.pipelines.ingestion.chunkers import StructuralContextChunker


@pytest.mark.asyncio
async def test_structural_chunker_basic() -> None:
    """Tests that StructuralContextChunker splits by paragraph/sentence."""
    # Text with clear structural boundaries
    text = "Paragraph 1 line 1.\nParagraph 1 line 2.\n\nParagraph 2 line 1.\nParagraph 2 line 2."

    # Small max_size to force a split at the paragraph break
    chunker = StructuralContextChunker(max_chunk_size=45, chunk_overlap=0)

    chunks = await chunker.chunk(text)

    # Should split at \n\n
    assert len(chunks) >= 2
    assert "Paragraph 1" in chunks[0]["text"]
    assert "Paragraph 2" in chunks[1]["text"]

    # Verify offsets map to original text
    c1_text = text[chunks[0]["start_char"] : chunks[0]["end_char"]]
    assert c1_text == chunks[0]["text"]

    c2_text = text[chunks[1]["start_char"] : chunks[1]["end_char"]]
    assert c2_text == chunks[1]["text"]


@pytest.mark.asyncio
async def test_structural_chunker_with_context() -> None:
    """Tests that context_prefix is prepended without breaking offsets."""
    text = "This is a sentence. This is another sentence."
    prefix = "Document: test.txt\n"

    # Force split after first sentence
    chunker = StructuralContextChunker(max_chunk_size=25, chunk_overlap=0)

    chunks = await chunker.chunk(text, context_prefix=prefix)

    assert len(chunks) == 2

    # Check first chunk
    assert chunks[0]["text"].startswith(prefix)
    original_slice_0 = text[chunks[0]["start_char"] : chunks[0]["end_char"]]
    assert chunks[0]["text"] == prefix + original_slice_0
    assert original_slice_0 == "This is a sentence. "

    # Check second chunk
    assert chunks[1]["text"].startswith(prefix)
    original_slice_1 = text[chunks[1]["start_char"] : chunks[1]["end_char"]]
    assert chunks[1]["text"] == prefix + original_slice_1
    assert original_slice_1 == "This is another sentence."


@pytest.mark.asyncio
async def test_structural_chunker_empty_string() -> None:
    """Tests that empty string returns empty list."""
    chunker = StructuralContextChunker()
    assert await chunker.chunk("") == []


@pytest.mark.asyncio
async def test_structural_chunker_hard_slice() -> None:
    """Tests that it falls back to hard slice if no boundaries found."""
    text = "A" * 100  # No periods or newlines
    chunker = StructuralContextChunker(max_chunk_size=40, chunk_overlap=10)

    chunks = await chunker.chunk(text)

    assert len(chunks) > 1
    # Check that chunks actually move forward and overlap
    assert chunks[1]["start_char"] == chunks[0]["end_char"] - 10
