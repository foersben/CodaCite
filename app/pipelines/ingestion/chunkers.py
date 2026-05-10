"""Semantic chunking implementations for document ingestion.

This module provides strategies for splitting documents into meaningful
fragments while maintaining character-level provenance.
"""

import logging
import re
from typing import Any, cast

from app.core.interfaces import Chunker, ChunkMetadata, Embedder

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    """Implementation of semantic chunking using sentence embeddings.

    Groups sentences together based on their semantic similarity to preserve
    contextual integrity and minimize topic fragmentation.
    """

    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
    ):
        """Initialize the semantic chunker.

        Args:
            embedder: The embedding provider (e.g., BGE-M3).
            similarity_threshold: Cosine similarity threshold to split chunks.
            max_chunk_size: Maximum characters per chunk as a safety limit.
            min_chunk_size: Minimum characters to avoid tiny fragments.
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    async def chunk(self, text: str) -> list[ChunkMetadata]:
        """Split text into semantic chunks with provenance tracking.

        Args:
            text: Raw document text.

        Returns:
            List of dictionaries with 'text', 'start_char', and 'end_char'.
        """
        if not text.strip():
            return []

        # 1. Split into sentences while preserving offsets
        # This regex handles most English sentence boundaries
        sentence_pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")

        sentences: list[dict[str, Any]] = []
        last_idx = 0
        for match in sentence_pattern.finditer(text):
            raw_slice = text[last_idx : match.start()]
            sentence_text = raw_slice.strip()
            if sentence_text:
                # Adjust start offset to exclude leading whitespace.
                # match.start() already points to the separator whitespace, so raw_slice
                # ends at the sentence-ending punctuation with no trailing whitespace.
                leading = len(raw_slice) - len(raw_slice.lstrip())
                start = last_idx + leading
                sentences.append({"text": sentence_text, "start": start, "end": match.start()})
            last_idx = match.end()

        # Add the last sentence
        raw_final = text[last_idx:]
        final_text = raw_final.strip()
        if final_text:
            leading = len(raw_final) - len(raw_final.lstrip())
            trailing = len(raw_final) - len(raw_final.rstrip())
            sentences.append(
                {
                    "text": final_text,
                    "start": last_idx + leading,
                    "end": len(text) - trailing,
                }
            )

        if not sentences:
            return [cast(ChunkMetadata, {"text": text, "start_char": 0, "end_char": len(text)})]

        # 2. Get embeddings for all sentences in a batch
        sentence_texts = [str(s["text"]) for s in sentences]
        embeddings = await self.embedder.embed_batch(sentence_texts)

        # 3. Group sentences semantically
        chunks: list[dict[str, Any]] = []
        current_sentences = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i]

            # Calculate cosine similarity (assuming normalized embeddings)
            # similarity = dot(current_chunk_mean, next_sentence)
            similarity = sum(a * b for a, b in zip(current_embedding, embedding, strict=False))

            current_chunk_len = int(sentence["end"]) - int(current_sentences[0]["start"])

            # Split logic:
            # - If similarity is too low (new topic)
            # - OR if chunk is already too big (safety limit)
            # BUT only split if the current chunk is at least min_chunk_size
            should_split = (
                similarity < self.similarity_threshold and current_chunk_len > self.min_chunk_size
            ) or (current_chunk_len > self.max_chunk_size)

            if should_split:
                # Flush current chunk
                start_c = int(current_sentences[0]["start"])
                end_c = int(current_sentences[-1]["end"])
                full_text = text[start_c:end_c]
                chunks.append(
                    {
                        "text": full_text,
                        "start_char": start_c,
                        "end_char": end_c,
                    }
                )
                # Reset
                current_sentences = [sentence]
                current_embedding = embedding
            else:
                # Merge into current chunk
                current_sentences.append(sentence)
                # Running average for chunk embedding, then re-normalize to unit length
                # so subsequent dot-product comparisons remain cosine similarities.
                weight = 1.0 / len(current_sentences)
                avg = [
                    (1.0 - weight) * ce + weight * ne
                    for ce, ne in zip(current_embedding, embedding, strict=False)
                ]
                norm = sum(v * v for v in avg) ** 0.5
                current_embedding = [v / norm for v in avg] if norm > 0 else avg

        # Flush final chunk
        if current_sentences:
            start_c = int(current_sentences[0]["start"])
            end_c = int(current_sentences[-1]["end"])
            full_text = text[start_c:end_c]
            chunks.append(
                {
                    "text": full_text,
                    "start_char": start_c,
                    "end_char": end_c,
                }
            )

        logger.info("[CHUNKER] Generated %d semantic chunks", len(chunks))
        return cast(list[ChunkMetadata], chunks)
