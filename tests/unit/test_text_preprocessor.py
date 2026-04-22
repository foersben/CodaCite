"""Tests for TextPreprocessor."""

import pytest

from app.ingestion.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Provide a TextPreprocessor instance."""
    return TextPreprocessor()


def test_preprocess_empty_string(preprocessor: TextPreprocessor) -> None:
    """Test preprocessing an empty string.

    Arrange: Empty string.
    Act: Process string.
    Assert: Result is empty.
    """
    # Arrange
    text = ""
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == ""


def test_preprocess_unicode_normalization(preprocessor: TextPreprocessor) -> None:
    """Test Unicode NFKC normalization.

    Arrange: String with full-width characters.
    Act: Process string.
    Assert: Result is normalized to ASCII.
    """
    # Arrange
    text = "Ｈｅｌｌｏ"  # Full-width Hello
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello"


def test_preprocess_control_character_removal(preprocessor: TextPreprocessor) -> None:
    """Test removal of control characters.

    Arrange: String with form-feed, null byte, etc.
    Act: Process string.
    Assert: Control characters are removed.
    """
    # Arrange
    text = "Hello\x00World\x0c!"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "HelloWorld!"


def test_preprocess_whitespace_compression(preprocessor: TextPreprocessor) -> None:
    """Test compression of horizontal whitespace.

    Arrange: String with multiple spaces and tabs.
    Act: Process string.
    Assert: Multiple spaces are compressed to a single space.
    """
    # Arrange
    text = "Hello    \t  World!"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello World!"


def test_preprocess_newline_compression(preprocessor: TextPreprocessor) -> None:
    """Test compression of multiple newlines.

    Arrange: String with more than two consecutive newlines.
    Act: Process string.
    Assert: Newlines are compressed to at most two.
    """
    # Arrange
    text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Line 1\n\nLine 2\n\nLine 3"


def test_preprocess_strip_whitespace(preprocessor: TextPreprocessor) -> None:
    """Test stripping leading and trailing whitespace.

    Arrange: String with leading and trailing spaces/newlines.
    Act: Process string.
    Assert: Outer whitespace is removed.
    """
    # Arrange
    text = "   \n  Hello World!  \n  "
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello World!"


def test_preprocess_combined(preprocessor: TextPreprocessor) -> None:
    """Test all preprocessing steps combined.

    Arrange: A messy string with unicode, control chars, spaces, and newlines.
    Act: Process string.
    Assert: Cleaned string is correct.
    """
    # Arrange
    text = " \t Ｈｅｌｌｏ\x00 \n\n\n\n Ｗｏｒｌｄ!  \t "
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello \n\n World!"
