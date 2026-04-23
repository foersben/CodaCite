import logging
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Represents a document loaded from disk."""

    text: str
    source: str
    format: str


class DocumentLoader:
    """Loads documents in PDF, DOCX, Markdown, and plain text formats."""

    _SUPPORTED_FORMATS: dict[str, str] = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
    }

    def load(self, path: Path) -> list[LoadedDocument]:
        """Load a document from *path* and return a list of :class:`LoadedDocument`."""
        logger.info("Loading document from: %s", path)
        suffix = path.suffix.lower()
        fmt = self._SUPPORTED_FORMATS.get(suffix)
        if fmt is None:
            logger.error("Unsupported file format: %s", suffix)
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

        logger.info("Successfully loaded %s document (%d characters)", fmt, len(text))
        return [LoadedDocument(text=text, source=str(path), format=fmt)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pdf(path: Path) -> str:
        logger.debug("Extracting text from PDF: %s", path)
        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        logger.debug("PDF has %d pages", num_pages)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    @staticmethod
    def _load_docx(path: Path) -> str:
        doc = Document(str(path))
        paragraphs = [para.text for para in doc.paragraphs if para.text]
        return "\n".join(paragraphs)

    @staticmethod
    def _load_markdown(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _load_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")
