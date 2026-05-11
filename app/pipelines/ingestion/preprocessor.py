"""Text preprocessing: Unicode normalization, whitespace compression, artifact removal.

This module provides tools for cleaning and normalizing raw text extracted from
documents to ensure consistent processing by downstream NLP components.
"""

from __future__ import annotations

import re
import unicodedata


class TextPreprocessor:
    """Normalizes and cleans raw document text for stable downstream NLP.

    This class provides deterministic text cleaning. It ensures that the input
    to the Coreference Resolver and Chunking stages is free of encoding
    artifacts and excessive noise that could confuse LLM tokenization.

    Operations applied in order:
    1. **NFKC Normalization**: Compatibility decomposition (e.g., combining characters).
    2. **Control Char Removal**: Strips non-printable ASCII/Unicode artifacts.
    3. **Whitespace Compression**: Normalizes tabs/spaces and reduces newline clutter.

    Pipeline Role:
        Phase 1: Normalization. Sanitizes raw text extracted from documents
        (e.g., via Docling) to ensure deterministic processing by downstream
        NLP components (Phase 2-9).
    """

    # Control characters to remove (form-feed, null byte, vertical-tab, etc.)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    # Multiple horizontal whitespace → single space
    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    # More than two consecutive newlines → two newlines
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

    def process(self, text: str) -> str:
        """Apply the normalization pipeline to the input text.

        This method executes the following transformation sequence:
            - **NFKC Normalization**: Resolves visual identicalness (e.g.,
              combining 'e' and '´' into 'é') and ensures consistent
              representation of ligatures and symbols.
            - **Control Character Filtering**: Removes non-printable characters
              that often contaminate text extracted from legacy PDFs.
            - **Whitespace Compression**: Collapses redundant tabs and spaces
              into single spaces, and limits paragraph breaks to a maximum
              of two newlines to prevent sparsity in chunks.

        Args:
            text: The raw, potentially contaminated string.

        Returns:
            A sanitized, UTF-8 normalized string ready for NLP tasks.
        """
        if not text:
            return text

        # Step 1: NFKC normalization
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Remove control/artifact characters
        text = self._CONTROL_CHAR_RE.sub("", text)

        # Step 3: Compress horizontal whitespace
        text = self._MULTI_SPACE_RE.sub(" ", text)

        # Step 4: Compress multiple consecutive newlines
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)

        # Step 5: Strip
        return text.strip()
