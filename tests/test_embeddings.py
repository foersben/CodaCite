"""
Tests for the embeddings module.

Covers:
- LocalEmbedder: generates embeddings using sentence-transformers (BAAI/bge-large-en-v1.5)
- All LLM/model calls are mocked to avoid actual inference during testing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.embeddings.embedder import LocalEmbedder


class TestLocalEmbedder:
    """Tests for LocalEmbedder."""

    @patch("app.embeddings.embedder.SentenceTransformer")
    def test_embed_single_text(self, mock_st_cls: MagicMock) -> None:
        """embed() should return a float list for a single string."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st_cls.return_value = mock_model

        embedder = LocalEmbedder(model_path="fake/path")
        result = embedder.embed(["hello world"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    @patch("app.embeddings.embedder.SentenceTransformer")
    def test_embed_multiple_texts(self, mock_st_cls: MagicMock) -> None:
        """embed() should return one embedding vector per input text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_st_cls.return_value = mock_model

        embedder = LocalEmbedder(model_path="fake/path")
        result = embedder.embed(["text1", "text2", "text3"])

        assert len(result) == 3

    @patch("app.embeddings.embedder.SentenceTransformer")
    def test_embed_normalizes_output(self, mock_st_cls: MagicMock) -> None:
        """embed() should return standard Python float lists (not numpy arrays)."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_cls.return_value = mock_model

        embedder = LocalEmbedder(model_path="fake/path")
        result = embedder.embed(["test"])

        assert isinstance(result[0][0], float)

    @patch("app.embeddings.embedder.SentenceTransformer")
    def test_embed_empty_list_returns_empty(self, mock_st_cls: MagicMock) -> None:
        """embed() should return an empty list for empty input."""
        mock_model = MagicMock()
        mock_model.encode.return_value = []
        mock_st_cls.return_value = mock_model

        embedder = LocalEmbedder(model_path="fake/path")
        result = embedder.embed([])

        assert result == []

    @patch("app.embeddings.embedder.SentenceTransformer")
    def test_embedder_uses_local_model_path(self, mock_st_cls: MagicMock) -> None:
        """LocalEmbedder should load the model from the given local path."""
        mock_st_cls.return_value = MagicMock()

        LocalEmbedder(model_path="/local/models/BAAI/bge-large-en-v1.5")

        mock_st_cls.assert_called_once_with("/local/models/BAAI/bge-large-en-v1.5")

    @patch("app.embeddings.embedder.SentenceTransformer")
    def test_embedding_dimension(self, mock_st_cls: MagicMock) -> None:
        """embed() should return vectors of the model's dimension."""
        dim = 1024  # bge-large-en-v1.5 dimension
        mock_model = MagicMock()
        mock_model.encode.return_value = [[float(i) for i in range(dim)]]
        mock_st_cls.return_value = mock_model

        embedder = LocalEmbedder(model_path="fake/path")
        result = embedder.embed(["test sentence"])

        assert len(result[0]) == dim
