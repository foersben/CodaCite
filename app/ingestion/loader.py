"""Document loading utilities for various file formats.

This module provides a unified interface for loading text content from PDF,
DOCX, Markdown, and plain text files.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Represents a document loaded from disk.

    Attributes:
        text: The raw text content of the document.
        source: The file path or source identifier.
        format: The file format (e.g., 'pdf', 'docx').
    """

    text: str
    source: str
    format: str


class DocumentLoader:
    """Standardized entry point for extracting text from diverse file formats.

    This class serves as the first stage of the ingestion pipeline. It maps file
    extensions to specific extraction engines (pypdf, python-docx, etc.) and
    normalizes the output into `LoadedDocument` objects.

    Supported Formats:
        - PDF: Extracted using `pypdf.PdfReader` (page-by-page concatenation).
        - DOCX: Extracted using `python-docx` (paragraph iteration).
        - Markdown/Text: Raw UTF-8 reading with error replacement.

    Pipeline Role:
        `UploadFile (Bytes)` -> `NamedTemporaryFile (Disk)` -> `DocumentLoader.load()` -> `list[LoadedDocument]`
    """

    _SUPPORTED_FORMATS: dict[str, str] = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
    }

    def load(self, path: Path) -> list[LoadedDocument]:
        """Load a document from the specified path.

        Args:
            path: Path to the document file.

        Returns:
            A list containing a single LoadedDocument instance.

        Raises:
            ValueError: If the file format is not supported.
        """
        logger.info("[LOADER] Loading document from: %s", path)
        suffix = path.suffix.lower()
        fmt = self._SUPPORTED_FORMATS.get(suffix)

        if fmt is None:
            logger.error("[LOADER] Unsupported file format: %s", suffix)
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                f"Supported formats: {list(self._SUPPORTED_FORMATS)}"
            )

        if fmt == "pdf":
            text = self._load_pdf(path)
        elif fmt == "docx":
            text = self._load_docx(path)
        elif fmt == "text":
            text = self._load_text(path)
        else:
            text = self._load_markdown(path)

        logger.info("[LOADER] Successfully loaded %s document (%d characters)", fmt, len(text))
        return [LoadedDocument(text=text, source=str(path), format=fmt)]

    @staticmethod
    def _load_pdf(path: Path) -> str:
        """Extract text from a PDF file.

        Args:
            path: Path to the PDF file.

        Returns:
            Extracted text content.
        """
        logger.debug("[LOADER] Extracting text from PDF: %s", path)
        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        logger.debug("[LOADER] PDF has %d pages", num_pages)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    @staticmethod
    def _load_docx(path: Path) -> str:
        """Extract text from a DOCX file.

        Args:
            path: Path to the DOCX file.

        Returns:
            Extracted text content.
        """
        doc = DocxDocument(str(path))
        paragraphs = [para.text for para in doc.paragraphs if para.text]
        return "\n".join(paragraphs)

    @staticmethod
    def _load_markdown(path: Path) -> str:
        """Read content from a Markdown file.

        Args:
            path: Path to the Markdown file.

        Returns:
            File content as text.
        """
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _load_text(path: Path) -> str:
        """Read content from a plain text file.

        Args:
            path: Path to the text file.

        Returns:
            File content as text.
        """
        return path.read_text(encoding="utf-8", errors="replace")
