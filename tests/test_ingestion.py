"""
Tests for the ingestion module.

Covers:
- DocumentLoader: loads PDF, DOCX, and Markdown files
- TextPreprocessor: Unicode normalization, whitespace compression, artifact removal
- TextChunker: RecursiveCharacterTextSplitter with semantic fallback
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader, LoadedDocument
from app.ingestion.preprocessor import TextPreprocessor

# ---------------------------------------------------------------------------
# DocumentLoader tests
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_markdown_file(self, tmp_path: Path) -> None:
        """DocumentLoader should load a .md file and return text content."""
        md_file = tmp_path / "sample.md"
        md_file.write_text("# Hello\n\nThis is a **test** document.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(md_file)

        assert len(docs) == 1
        assert "Hello" in docs[0].text
        assert docs[0].source == str(md_file)
        assert docs[0].format == "markdown"

    def test_load_pdf_file(self, tmp_path: Path) -> None:
        """DocumentLoader should load a .pdf file via pypdf."""
        pdf_path = tmp_path / "sample.pdf"

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF page content here."

        with patch("app.ingestion.loader.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            loader = DocumentLoader()
            docs = loader.load(pdf_path)

        assert len(docs) == 1
        assert "PDF page content" in docs[0].text
        assert docs[0].format == "pdf"

    def test_load_docx_file(self, tmp_path: Path) -> None:
        """DocumentLoader should load a .docx file via python-docx."""
        docx_path = tmp_path / "sample.docx"

        mock_para = MagicMock()
        mock_para.text = "DOCX paragraph content."

        with patch("app.ingestion.loader.Document") as mock_doc_cls:
            mock_doc = MagicMock()
            mock_doc.paragraphs = [mock_para]
            mock_doc_cls.return_value = mock_doc

            loader = DocumentLoader()
            docs = loader.load(docx_path)

        assert len(docs) == 1
        assert "DOCX paragraph content" in docs[0].text
        assert docs[0].format == "docx"

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """DocumentLoader should raise ValueError for unsupported file types."""
        txt_path = tmp_path / "sample.xyz"
        txt_path.write_text("some content", encoding="utf-8")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(txt_path)

    def test_loaded_document_is_dataclass(self) -> None:
        """LoadedDocument should expose text, source, and format attributes."""
        doc = LoadedDocument(text="hello", source="/path/to/file.md", format="markdown")
        assert doc.text == "hello"
        assert doc.source == "/path/to/file.md"
        assert doc.format == "markdown"


# ---------------------------------------------------------------------------
# TextPreprocessor tests
# ---------------------------------------------------------------------------


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_unicode_nfkc_normalization(self) -> None:
        """Preprocessor should apply NFKC Unicode normalization."""
        preprocessor = TextPreprocessor()
        # Full-width characters -> ASCII
        text = "\uff48\uff45\uff4c\uff4c\uff4f"  # ｈｅｌｌｏ
        result = preprocessor.process(text)
        assert result == "hello"

    def test_whitespace_compression(self) -> None:
        """Preprocessor should compress multiple spaces and newlines."""
        preprocessor = TextPreprocessor()
        text = "hello    world\n\n\n  foo   bar"
        result = preprocessor.process(text)
        assert "  " not in result
        assert "\n\n\n" not in result

    def test_removes_common_artifacts(self) -> None:
        """Preprocessor should remove PDF/OCR artifacts like form-feed and null bytes."""
        preprocessor = TextPreprocessor()
        text = "hello\x0cworld\x00end"
        result = preprocessor.process(text)
        assert "\x0c" not in result
        assert "\x00" not in result

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Preprocessor should strip leading and trailing whitespace."""
        preprocessor = TextPreprocessor()
        text = "   hello world   "
        result = preprocessor.process(text)
        assert result == result.strip()

    def test_empty_string_returns_empty(self) -> None:
        """Preprocessor should handle empty strings gracefully."""
        preprocessor = TextPreprocessor()
        assert preprocessor.process("") == ""


# ---------------------------------------------------------------------------
# TextChunker tests
# ---------------------------------------------------------------------------


class TestTextChunker:
    """Tests for TextChunker."""

    def test_chunks_long_text(self) -> None:
        """TextChunker should split long text into multiple chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        long_text = "word " * 500  # ~2500 chars
        chunks = chunker.chunk(long_text)
        assert len(chunks) > 1

    def test_chunk_size_respected(self) -> None:
        """Each chunk should not exceed chunk_size by more than allowed overlap."""
        chunk_size = 100
        chunk_overlap = 10
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        long_text = "a" * 1000
        chunks = chunker.chunk(long_text)
        for chunk in chunks:
            assert len(chunk) <= chunk_size + chunk_overlap

    def test_short_text_returns_single_chunk(self) -> None:
        """Short text that fits in a single chunk should return one chunk."""
        chunker = TextChunker(chunk_size=1024, chunk_overlap=128)
        short_text = "This is a short document."
        chunks = chunker.chunk(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_empty_string_returns_empty_list(self) -> None:
        """Empty input should return an empty list."""
        chunker = TextChunker(chunk_size=1024, chunk_overlap=128)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_overlap_creates_shared_content(self) -> None:
        """Consecutive chunks should share content due to overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=20)
        text = "abcdefghij" * 20  # 200 chars
        chunks = chunker.chunk(text)
        if len(chunks) > 1:
            # The end of chunk[0] should overlap with the start of chunk[1]
            end_of_first = chunks[0][-20:]
            start_of_second = chunks[1][:20]
            assert end_of_first == start_of_second

    def test_default_parameters_match_config(self) -> None:
        """Default TextChunker parameters should match project config (1024/128)."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 128
