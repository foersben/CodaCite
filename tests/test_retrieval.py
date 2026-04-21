"""
Tests for the retrieval module.

Covers:
- HybridRetriever: orchestrates vector search + graph traversal + reranking
- CrossEncoderReranker: reranks a list of candidates (mocked)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import HybridRetriever, RetrievalResult

# ---------------------------------------------------------------------------
# CrossEncoderReranker tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    @patch("app.retrieval.reranker.CrossEncoder")
    def test_rerank_returns_sorted_results(self, mock_ce_cls: MagicMock) -> None:
        """rerank() should return candidates sorted by score descending."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9, 0.1]
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker(model_path="fake/path")
        candidates = [
            {"text": "low relevance"},
            {"text": "high relevance"},
            {"text": "very low"},
        ]
        query = "test query"
        result = reranker.rerank(query=query, candidates=candidates, top_k=2)

        assert len(result) == 2
        assert result[0]["text"] == "high relevance"

    @patch("app.retrieval.reranker.CrossEncoder")
    def test_rerank_respects_top_k(self, mock_ce_cls: MagicMock) -> None:
        """rerank() should return at most top_k results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.5, 0.9, 0.2]
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker(model_path="fake/path")
        candidates = [{"text": f"doc{i}"} for i in range(4)]
        result = reranker.rerank(query="query", candidates=candidates, top_k=2)

        assert len(result) == 2

    @patch("app.retrieval.reranker.CrossEncoder")
    def test_rerank_empty_candidates(self, mock_ce_cls: MagicMock) -> None:
        """rerank() on an empty candidates list should return empty list."""
        mock_model = MagicMock()
        mock_model.predict.return_value = []
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker(model_path="fake/path")
        result = reranker.rerank(query="query", candidates=[], top_k=5)

        assert result == []


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock()
        store.vector_search = AsyncMock(
            return_value=[{"id": "chunk:1", "text": "vector result", "score": 0.9}]
        )
        store.traverse_graph = AsyncMock(
            return_value=[{"id": "entity:Alice", "name": "Alice", "entity_type": "PERSON"}]
        )
        return store

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    @pytest.fixture
    def mock_reranker(self) -> MagicMock:
        reranker = MagicMock()
        reranker.rerank.return_value = [{"text": "reranked result", "score": 0.95}]
        return reranker

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(
        self,
        mock_store: MagicMock,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """retrieve() should return a list of RetrievalResult objects."""
        retriever = HybridRetriever(
            store=mock_store,
            embedder=mock_embedder,
            reranker=mock_reranker,
        )
        results = await retriever.retrieve(query="Who is Alice?", top_k=3)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_calls_vector_search(
        self,
        mock_store: MagicMock,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """retrieve() should invoke vector_search on the store."""
        retriever = HybridRetriever(
            store=mock_store,
            embedder=mock_embedder,
            reranker=mock_reranker,
        )
        await retriever.retrieve(query="Who is Alice?", top_k=3)

        mock_store.vector_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_calls_graph_traversal(
        self,
        mock_store: MagicMock,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """retrieve() should invoke traverse_graph for graph-based context enrichment."""
        retriever = HybridRetriever(
            store=mock_store,
            embedder=mock_embedder,
            reranker=mock_reranker,
        )
        await retriever.retrieve(query="Who is Alice?", top_k=3)

        mock_store.traverse_graph.assert_called()

    @pytest.mark.asyncio
    async def test_retrieve_calls_reranker(
        self,
        mock_store: MagicMock,
        mock_embedder: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """retrieve() should pass combined candidates through the reranker."""
        retriever = HybridRetriever(
            store=mock_store,
            embedder=mock_embedder,
            reranker=mock_reranker,
        )
        await retriever.retrieve(query="Who is Alice?", top_k=3)

        mock_reranker.rerank.assert_called_once()

    def test_retrieval_result_dataclass(self) -> None:
        """RetrievalResult should expose text and score attributes."""
        result = RetrievalResult(text="some text", score=0.9, source="doc.pdf")
        assert result.text == "some text"
        assert result.score == 0.9
        assert result.source == "doc.pdf"
