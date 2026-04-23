"""Tests for TextPreprocessor.

This module validates the text cleaning and normalization logic within
the ingestion pipeline.
"""

import pytest

from app.ingestion.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Provide a TextPreprocessor instance."""
    return TextPreprocessor()


def test_preprocess_empty_string(preprocessor: TextPreprocessor) -> None:
    """Test preprocessing an empty string.

    Given: An empty input string.
    When: The text is processed.
    Then: It should return an empty string.
    """
    # Arrange
    text = ""
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == ""


def test_preprocess_unicode_normalization(preprocessor: TextPreprocessor) -> None:
    """Test Unicode NFKC normalization.

    Given: A string containing full-width or non-standard Unicode characters.
    When: The text is processed.
    Then: It should be normalized to standard ASCII/NFKC form.
    """
    # Arrange
    text = "Ｈｅｌｌｏ"  # Full-width Hello
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello"


def test_preprocess_control_character_removal(preprocessor: TextPreprocessor) -> None:
    """Test removal of control characters.

    Given: A string containing hidden control characters like null bytes or form feeds.
    When: The text is processed.
    Then: All non-printable control characters should be stripped.
    """
    # Arrange
    text = "Hello\x00World\x0c!"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "HelloWorld!"


def test_preprocess_whitespace_compression(preprocessor: TextPreprocessor) -> None:
    """Test compression of horizontal whitespace.

    Given: A string with multiple consecutive spaces and tabs.
    When: The text is processed.
    Then: Consecutive whitespace should be collapsed into a single space.
    """
    # Arrange
    text = "Hello    \t  World!"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello World!"


def test_preprocess_newline_compression(preprocessor: TextPreprocessor) -> None:
    """Test compression of multiple newlines.

    Given: A string with many consecutive newline characters.
    When: The text is processed.
    Then: It should allow at most two consecutive newlines (preserving paragraphs).
    """
    # Arrange
    text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Line 1\n\nLine 2\n\nLine 3"


def test_preprocess_strip_whitespace(preprocessor: TextPreprocessor) -> None:
    """Test stripping leading and trailing whitespace.

    Given: A string with whitespace at the boundaries.
    When: The text is processed.
    Then: Leading and trailing spaces and newlines should be removed.
    """
    # Arrange
    text = "   \n  Hello World!  \n  "
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello World!"


def test_preprocess_combined(preprocessor: TextPreprocessor) -> None:
    """Test all preprocessing steps combined.

    Given: A messy string with multiple issues (unicode, control chars, extra whitespace).
    When: The text is processed.
    Then: It should result in a clean, normalized output.
    """
    # Arrange
    text = " \t Ｈｅｌｌｏ\x00 \n\n\n\n Ｗｏｒｌｄ!  \t "
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello \n\n World!"
