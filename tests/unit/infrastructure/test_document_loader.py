"""Unit tests for the DocumentLoader infrastructure adapter.

Validates the loading of various document formats (PDF, Text, Markdown)
and ensures robust handling of file encoding and unsupported types.
Now updated to verify Docling and VLM integration.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.ingestion.loader import DocumentLoader


@pytest.fixture
def loader(mocker: Any) -> DocumentLoader:
    """Provides a fresh DocumentLoader instance with mocked VLM for each test.

    Returns:
        A DocumentLoader instance.
    """
    mocker.patch("app.ingestion.loader.LocalVLM")
    return DocumentLoader()


def test_load_text_success(loader: DocumentLoader, tmp_path: Path) -> None:
    """Tests successful loading of a standard text file."""
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
    """Tests successful loading of a markdown file."""
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
    """Tests that loading an unsupported file format raises ValueError."""
    # Arrange
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"\x00\x01\x02")

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load(test_file)


def test_load_pdf_with_docling_and_vlm(mocker: Any, tmp_path: Path) -> None:
    """Tests loading a PDF using Docling with VLM injection for images.

    Given:
        A PDF file and mocked Docling components.
    When:
        The DocumentLoader loads the PDF.
    Then:
        It should use Docling to extract text and tables, and call VLM for images.
    """
    # Arrange
    test_file = tmp_path / "test.pdf"
    test_file.touch()

    # Mock Docling classes
    mock_converter_cls = mocker.patch("app.ingestion.loader.DocumentConverter")
    mocker.patch("app.ingestion.loader.InputFormat")
    mocker.patch("app.ingestion.loader.PdfPipelineOptions")
    mocker.patch("app.ingestion.loader.PdfFormatOption")

    mock_vlm_cls = mocker.patch("app.ingestion.loader.LocalVLM")
    mock_vlm = mock_vlm_cls.return_value
    mock_vlm.describe_image.return_value = "A beautiful technical drawing"

    mock_converter = mock_converter_cls.return_value
    mock_result = MagicMock()
    mock_doc = MagicMock()
    mock_converter.convert.return_value = mock_result
    mock_result.document = mock_doc

    # Create mock items for Docling document
    from docling_core.types.doc.document import PictureItem, TableItem, TextItem

    mock_text = MagicMock(spec=TextItem)
    mock_text.text = "Introduction to the system."

    mock_table = MagicMock(spec=TableItem)

    mock_picture = MagicMock(spec=PictureItem)
    mock_picture.prov = [MagicMock(page_no=1)]
    mock_picture.image = MagicMock()
    mock_picture.image.pil_image = MagicMock()

    # Configure document to return these items
    mock_doc.iterate_items.return_value = [
        (mock_text, 0),
        (mock_table, 0),
        (mock_picture, 0),
    ]

    def mock_export(item_set=None):
        if item_set:
            item = list(item_set)[0]
            if item == mock_table:
                return "| Col 1 | Col 2 |\n|---|---|"
            if hasattr(item, "text"):
                return item.text
        return "Full MD"

    mock_doc.export_to_markdown.side_effect = mock_export

    mock_doc.get_image.return_value = MagicMock()  # Mock PIL Image

    loader = DocumentLoader()

    # Act
    docs = loader.load(test_file)

    # Assert
    assert len(docs) == 1
    text = docs[0].text
    assert "Introduction to the system." in text
    assert "| Col 1 | Col 2 |" in text
    assert "Technical Drawing Description: A beautiful technical drawing" in text
    assert docs[0].format == "pdf"

    # Verify VLM was called
    mock_vlm.describe_image.assert_called_once()
