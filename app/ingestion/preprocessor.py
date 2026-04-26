"""Text preprocessing: Unicode normalization, whitespace compression, artifact removal.

This module provides tools for cleaning and normalizing raw text extracted from
documents to ensure consistent processing by downstream NLP components.
"""

from __future__ import annotations

import re
import unicodedata


class TextPreprocessor:
    """Normalizes and cleans raw document text.

    Operations applied in order:
    1. NFKC Unicode normalization (full-width → ASCII, ligatures, etc.)
    2. Removal of control/artifact characters (form-feed, null bytes, etc.)
    3. Whitespace compression (collapse multiple spaces/newlines)
    4. Strip leading/trailing whitespace
    """

    # Control characters to remove (form-feed, null byte, vertical-tab, etc.)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    # Multiple horizontal whitespace → single space
    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    # More than two consecutive newlines → two newlines
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

    def process(self, text: str) -> str:
        """Apply all preprocessing steps to the input text.

        Args:
            text: The raw text string to process.

        Returns:
            The normalized and cleaned text string.
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
