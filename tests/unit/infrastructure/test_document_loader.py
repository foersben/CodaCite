"""Unit tests for the DocumentLoader infrastructure adapter.

Validates the loading of various document formats (PDF, Text, Markdown)
and ensures robust handling of file encoding and unsupported types.
"""

from pathlib import Path
from typing import Any

import pytest

from app.ingestion.loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    """Provides a fresh DocumentLoader instance for each test.

    Returns:
        A DocumentLoader instance.
    """
    return DocumentLoader()


def test_load_text_success(loader: DocumentLoader, tmp_path: Path) -> None:
    """Tests successful loading of a standard text file.

    Given:
        A valid plaintext file on disk.
    When:
        The DocumentLoader loads the file.
    Then:
        It should return exactly one document with 'text' format.

    Args:
        loader: The DocumentLoader fixture.
        tmp_path: Pytest temporary path fixture.
    """
    # Arrange
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World!")

    # Act
    docs = loader.load(test_file)

    # Assert
    assert len(docs) == 1
    assert docs[0].text == "Hello World!"
    assert docs[0].format == "text"


def test_load_markdown_success(loader: DocumentLoader, tmp_path: Path) -> None:
    """Tests successful loading of a markdown file.

    Given:
        A valid markdown file on disk.
    When:
        The DocumentLoader loads the file.
    Then:
        It should return exactly one document with 'markdown' format.

    Args:
        loader: The DocumentLoader fixture.
        tmp_path: Pytest temporary path fixture.
    """
    # Arrange
    test_file = tmp_path / "test.md"
    test_file.write_text("# Markdown\n\nContent")

    # Act
    docs = loader.load(test_file)

    # Assert
    assert len(docs) == 1
    assert docs[0].text == "# Markdown\n\nContent"
    assert docs[0].format == "markdown"


def test_load_unsupported_format(loader: DocumentLoader, tmp_path: Path) -> None:
    """Tests that loading an unsupported file format raises ValueError.

    Given:
        A binary file with an unknown extension.
    When:
        The DocumentLoader attempts to load it.
    Then:
        It should raise a ValueError with a descriptive message.

    Args:
        loader: The DocumentLoader fixture.
        tmp_path: Pytest temporary path fixture.
    """
    # Arrange
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"\x00\x01\x02")

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load(test_file)


def test_load_text_with_weird_encoding(loader: DocumentLoader, tmp_path: Path) -> None:
    """Tests loading text with problematic encodings.

    Given:
        A text file containing invalid UTF-8 byte sequences.
    When:
        The DocumentLoader attempts to load it.
    Then:
        It should fall back gracefully, replacing invalid characters.

    Args:
        loader: The DocumentLoader fixture.
        tmp_path: Pytest temporary path fixture.
    """
    # Arrange
    test_file = tmp_path / "weird.txt"
    test_file.write_bytes(b"Good text \xff\xfe bad text")

    # Act
    docs = loader.load(test_file)

    # Assert
    assert len(docs) == 1
    assert "Good text" in docs[0].text
    assert "bad text" in docs[0].text
    assert "\ufffd" in docs[0].text


def test_load_pdf(mocker: Any, loader: DocumentLoader, tmp_path: Path) -> None:
    """Tests loading a PDF file using a mocked PDF reader.

    Given:
        A PDF file and a mocked PdfReader.
    When:
        The DocumentLoader loads the PDF.
    Then:
        It should concatenate extracted text from all pages and set 'pdf' format.

    Args:
        mocker: The pytest-mock fixture.
        loader: The DocumentLoader fixture.
        tmp_path: Pytest temporary path fixture.
    """
    # Arrange
    test_file = tmp_path / "test.pdf"
    test_file.touch()

    mock_reader = mocker.MagicMock()
    mock_page1 = mocker.MagicMock()
    mock_page1.extract_text.return_value = "Page 1 Content"
    mock_page2 = mocker.MagicMock()
    mock_page2.extract_text.return_value = "Page 2 Content"
    mock_reader.pages = [mock_page1, mock_page2]

    mocker.patch("app.ingestion.loader.PdfReader", return_value=mock_reader)

    # Act
    docs = loader.load(test_file)

    # Assert
    assert len(docs) == 1
    assert docs[0].text == "Page 1 Content\nPage 2 Content"
    assert docs[0].format == "pdf"
