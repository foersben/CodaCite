"""Text chunking using RecursiveCharacterTextSplitter."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


class TextChunker:
    """Splits text into overlapping chunks using :class:`RecursiveCharacterTextSplitter`.

    Falls back to character-level splitting when no natural separator is found,
    making it suitable for dense technical prose and code-heavy documents.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters of overlap between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=False,
        )

    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of string chunks.

        Args:
            text: The document text to split.

        Returns:
            A list of non-empty string chunks.  Returns an empty list for
            empty or whitespace-only input.
        """
        if not text or not text.strip():
            return []
        return self._splitter.split_text(text)
