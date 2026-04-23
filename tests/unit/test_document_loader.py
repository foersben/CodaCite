"""Tests for DocumentLoader."""

from pathlib import Path

import pytest

from app.ingestion.loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    """Provide a DocumentLoader instance."""
    return DocumentLoader()


def test_load_text_success(loader: DocumentLoader, tmp_path: Path) -> None:
    """Test loading a standard text file.

    Arrange: Create a text file.
    Act: Load the file using DocumentLoader.
    Assert: The loaded text matches the file content.
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
    """Test loading a markdown file.

    Arrange: Create a markdown file.
    Act: Load the file using DocumentLoader.
    Assert: The loaded text matches the file content.
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
    """Test loading an unsupported file format raises ValueError.

    Arrange: Create a file with an unsupported extension.
    Act & Assert: Loading the file raises a ValueError.
    """
    # Arrange
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"\x00\x01\x02")

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load(test_file)


def test_load_text_with_weird_encoding(loader: DocumentLoader, tmp_path: Path) -> None:
    """Test loading text with weird encodings fallback nicely.

    Arrange: Create a text file containing invalid utf-8 bytes.
    Act: Load the file.
    Assert: The loader falls back using errors='replace' without crashing.
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


def test_load_pdf(mocker, loader: DocumentLoader, tmp_path: Path) -> None:
    """Test loading a basic PDF file.

    Arrange: Create a fake PDF path and mock PdfReader.
    Act: Load the PDF file.
    Assert: The loaded text matches the mocked extracted text.
    """
    # Arrange
    test_file = tmp_path / "test.pdf"
    test_file.touch()  # Create empty file so path exists

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
