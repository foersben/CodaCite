"""Structural and context-aware text partitioning implementations.

This module provides high-performance strategies for decomposing documents into
retrieval-optimized fragments. Unlike semantic chunkers which rely on expensive
model inference, these implementations utilize structural markers to maintain
high throughput and perfect character-level provenance.
"""

import logging
from typing import cast

from app.core.interfaces import Chunker, ChunkMetadata

logger = logging.getLogger(__name__)


class StructuralContextChunker(Chunker):
    """Fast sliding-window chunker with structural snapping and context injection.

    This chunker implements a hybrid strategy: it uses a sliding window to ensure
    even coverage while 'snapping' boundaries to the nearest paragraph or
    sentence break. To enhance retrieval relevance, it prepends document-level
    metadata (e.g., source filename) to each chunk's text without corrupting
    the underlying character offsets required for source highlighting.
    Design Goals:
        - **Provenance Integrity**: Exact mapping between chunks and raw text.
        - **Low Latency**: Pure Python string manipulation for O(n) performance.
        - **Context Enrichment**: Injection of parent metadata into leaf nodes.
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ):
        """Initialize the structural chunker with specific window constraints.

        Args:
            max_chunk_size: Maximum character count from the original document
                to include in a single chunk.
            chunk_overlap: The number of characters from the end of one chunk
                to include at the start of the next to prevent semantic shearing.
        """
        if max_chunk_size <= 0:
            msg = "max_chunk_size must be greater than 0."
            raise ValueError(msg)
        if chunk_overlap < 0 or chunk_overlap >= max_chunk_size:
            msg = "chunk_overlap must satisfy 0 <= chunk_overlap < max_chunk_size."
            raise ValueError(msg)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, text: str, context_prefix: str = "") -> list[ChunkMetadata]:
        r"""Decompose text into structural fragments with injected context.

        This method identifies optimal split points (prioritizing \n\n then . )
        within the text. It returns metadata that maps the enriched chunk text
        back to the precise character offsets in the original document.

        Args:
            text: The raw, normalized document content.
            context_prefix: A metadata string (e.g., 'Document: paper.pdf\n')
                to be prepended to the 'text' field of every resulting chunk.

        Returns:
            A sequence of ChunkMetadata objects containing the enriched text
            and original character spans.
        """
        if not text.strip():
            return []
        chunks: list[ChunkMetadata] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            # 1. Determine the hard boundary for this window
            end = min(start + self.max_chunk_size, text_len)
            # 2. Try to snap to structural boundaries if not at the very end
            if end < text_len:
                # Minimum search window to avoid tiny chunks (50% of max_chunk_size)
                search_start = start + (self.max_chunk_size // 2)
                # Priority 1: Paragraph break (\n\n)
                paragraph_break = text.rfind("\n\n", search_start, end)
                if paragraph_break != -1:
                    end = paragraph_break + 2  # Include the newlines
                else:
                    # Priority 2: Sentence break (. )
                    sentence_break = text.rfind(". ", search_start, end)
                    if sentence_break != -1:
                        end = sentence_break + 2  # Include the period and space
            # 3. Extract the original slice
            original_slice = text[start:end]
            # 4. Create metadata with context prepended to the text field
            chunks.append(
                cast(
                    ChunkMetadata,
                    {
                        "text": f"{context_prefix}{original_slice}",
                        "start_char": start,
                        "end_char": end,
                        "metadata": None,
                    },
                )
            )
            # 5. Move start pointer, accounting for overlap
            if end >= text_len:
                break
            # Slide the window
            start = max(0, end - self.chunk_overlap)
            # Safety: Ensure we always move forward
            if start <= chunks[-1]["start_char"]:
                start = end
        logger.info(
            "[CHUNKER] Generated %d structural chunks (Prefix: %r)",
            len(chunks),
            context_prefix.strip(),
        )
        return chunks
