"""Unit tests for the TextPreprocessor.

Validates the text cleaning and normalization logic, ensuring consistent
formatting for downstream NLP tasks in the ingestion pipeline.
"""

import pytest

from app.ingestion.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Provides a TextPreprocessor instance for testing.

    Returns:
        An initialized TextPreprocessor.
    """
    return TextPreprocessor()


def test_preprocess_empty_string(preprocessor: TextPreprocessor) -> None:
    """Tests preprocessing an empty string returns an empty string.

    Given:
        An empty input string.
    When:
        The process method is called.
    Then:
        It should return an empty string.
    """
    # Arrange
    text = ""
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == ""


def test_preprocess_unicode_normalization(preprocessor: TextPreprocessor) -> None:
    """Tests Unicode NFKC normalization to standard forms.

    Given:
        A string containing full-width or non-standard Unicode characters.
    When:
        The text is processed.
    Then:
        It should be normalized to standard ASCII/NFKC form.
    """
    # Arrange
    text = "Ｈｅｌｌｏ"  # Full-width Hello
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello"


def test_preprocess_control_character_removal(preprocessor: TextPreprocessor) -> None:
    """Tests removal of non-printable control characters.

    Given:
        A string containing hidden control characters like null bytes or form feeds.
    When:
        The text is processed.
    Then:
        All non-printable control characters should be stripped.
    """
    # Arrange
    text = "Hello\x00World\x0c!"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "HelloWorld!"


def test_preprocess_whitespace_compression(preprocessor: TextPreprocessor) -> None:
    """Tests compression of multiple horizontal whitespace characters.

    Given:
        A string with multiple consecutive spaces and tabs.
    When:
        The text is processed.
    Then:
        Consecutive whitespace should be collapsed into a single space.
    """
    # Arrange
    text = "Hello    \t  World!"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello World!"


def test_preprocess_newline_compression(preprocessor: TextPreprocessor) -> None:
    """Tests compression of excessive consecutive newlines.

    Given:
        A string with many consecutive newline characters.
    When:
        The text is processed.
    Then:
        It should allow at most two consecutive newlines (preserving paragraph structure).
    """
    # Arrange
    text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Line 1\n\nLine 2\n\nLine 3"


def test_preprocess_strip_whitespace(preprocessor: TextPreprocessor) -> None:
    """Tests stripping of leading and trailing whitespace.

    Given:
        A string with whitespace at the start and end.
    When:
        The text is processed.
    Then:
        Leading and trailing spaces and newlines should be removed.
    """
    # Arrange
    text = "   \n  Hello World!  \n  "
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello World!"


def test_preprocess_combined(preprocessor: TextPreprocessor) -> None:
    """Tests all preprocessing steps in combination.

    Given:
        A messy string with Unicode, control characters, and irregular whitespace.
    When:
        The text is processed.
    Then:
        It should result in a clean, normalized output.
    """
    # Arrange
    text = " \t Ｈｅｌｌｏ\x00 \n\n\n\n Ｗｏｒｌｄ!  \t "
    # Act
    result = preprocessor.process(text)
    # Assert
    assert result == "Hello \n\n World!"
