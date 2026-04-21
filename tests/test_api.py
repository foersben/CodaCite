"""
Tests for the FastAPI API endpoints.

Covers:
- POST /api/v1/ingest  – document ingestion pipeline
- POST /api/v1/query   – hybrid GraphRAG query
- GET  /health         – health check
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """GET /health should return HTTP 200."""
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_ok_status(self, client: AsyncClient) -> None:
        """GET /health should return JSON with status='ok'."""
        response = await client.get("/health")
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    """Tests for POST /api/v1/ingest."""

    @pytest.mark.asyncio
    async def test_ingest_accepts_markdown(self, client: AsyncClient) -> None:
        """POST /api/v1/ingest with a markdown file should return 200."""
        with (
            patch("app.main.DocumentLoader") as mock_loader_cls,
            patch("app.main.TextPreprocessor") as mock_prep_cls,
            patch("app.main.TextChunker") as mock_chunker_cls,
            patch("app.main.LocalEmbedder") as mock_embedder_cls,
            patch("app.main.EntityExtractor") as mock_extractor_cls,
            patch("app.main.GraphStore") as mock_store_cls,
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = [
                MagicMock(text="hello world", source="test.md", format="markdown")
            ]
            mock_loader_cls.return_value = mock_loader

            mock_prep = MagicMock()
            mock_prep.process.return_value = "hello world"
            mock_prep_cls.return_value = mock_prep

            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = ["hello world"]
            mock_chunker_cls.return_value = mock_chunker

            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = [[0.1, 0.2]]
            mock_embedder_cls.return_value = mock_embedder

            mock_extractor = MagicMock()
            mock_extractor.extract.return_value = ([], [])
            mock_extractor_cls.return_value = mock_extractor

            mock_store = MagicMock()
            mock_store.store_chunk = AsyncMock()
            mock_store_cls.return_value = mock_store

            files = {"file": ("test.md", b"# Hello\n\nThis is a test.", "text/markdown")}
            response = await client.post("/api/v1/ingest", files=files)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_ingest_returns_chunk_count(self, client: AsyncClient) -> None:
        """POST /api/v1/ingest should return the number of chunks processed."""
        with (
            patch("app.main.DocumentLoader") as mock_loader_cls,
            patch("app.main.TextPreprocessor") as mock_prep_cls,
            patch("app.main.TextChunker") as mock_chunker_cls,
            patch("app.main.LocalEmbedder") as mock_embedder_cls,
            patch("app.main.EntityExtractor") as mock_extractor_cls,
            patch("app.main.GraphStore") as mock_store_cls,
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = [
                MagicMock(text="chunk text", source="doc.md", format="markdown")
            ]
            mock_loader_cls.return_value = mock_loader

            mock_prep = MagicMock()
            mock_prep.process.return_value = "chunk text"
            mock_prep_cls.return_value = mock_prep

            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = ["chunk one", "chunk two"]
            mock_chunker_cls.return_value = mock_chunker

            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = [[0.1], [0.2]]
            mock_embedder_cls.return_value = mock_embedder

            mock_extractor = MagicMock()
            mock_extractor.extract.return_value = ([], [])
            mock_extractor_cls.return_value = mock_extractor

            mock_store = MagicMock()
            mock_store.store_chunk = AsyncMock()
            mock_store_cls.return_value = mock_store

            files = {"file": ("doc.md", b"# Doc\n\nContent here.", "text/markdown")}
            response = await client.post("/api/v1/ingest", files=files)

        data = response.json()
        assert "chunks_processed" in data
        assert data["chunks_processed"] == 2

    @pytest.mark.asyncio
    async def test_ingest_unsupported_format_returns_422(self, client: AsyncClient) -> None:
        """POST /api/v1/ingest with unsupported file type should return 422."""
        files = {"file": ("test.xyz", b"random content", "application/octet-stream")}
        response = await client.post("/api/v1/ingest", files=files)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """Tests for POST /api/v1/query."""

    @pytest.mark.asyncio
    async def test_query_returns_200(self, client: AsyncClient) -> None:
        """POST /api/v1/query should return HTTP 200 for a valid query."""
        with (
            patch("app.main.LocalEmbedder") as mock_embedder_cls,
            patch("app.main.GraphStore") as mock_store_cls,
            patch("app.main.CrossEncoderReranker") as mock_reranker_cls,
            patch("app.main.HybridRetriever") as mock_retriever_cls,
            patch("app.main.IntentRouter") as mock_router_cls,
        ):
            mock_embedder_cls.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mock_reranker_cls.return_value = MagicMock()

            mock_retriever = MagicMock()
            mock_retriever.retrieve = AsyncMock(
                return_value=[MagicMock(text="result", score=0.9, source="doc.pdf")]
            )
            mock_retriever_cls.return_value = mock_retriever

            mock_router = MagicMock()
            mock_router.route.return_value = "knowledge_retrieval"
            mock_router_cls.return_value = mock_router

            response = await client.post(
                "/api/v1/query",
                json={"query": "What is GraphRAG?", "top_k": 3},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_query_returns_results_list(self, client: AsyncClient) -> None:
        """POST /api/v1/query should return a 'results' list in the response body."""
        with (
            patch("app.main.LocalEmbedder") as mock_embedder_cls,
            patch("app.main.GraphStore") as mock_store_cls,
            patch("app.main.CrossEncoderReranker") as mock_reranker_cls,
            patch("app.main.HybridRetriever") as mock_retriever_cls,
            patch("app.main.IntentRouter") as mock_router_cls,
        ):
            mock_embedder_cls.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mock_reranker_cls.return_value = MagicMock()

            from app.retrieval.retriever import RetrievalResult

            mock_retriever = MagicMock()
            mock_retriever.retrieve = AsyncMock(
                return_value=[RetrievalResult(text="GraphRAG combines graph and vector search.", score=0.95)]
            )
            mock_retriever_cls.return_value = mock_retriever

            mock_router = MagicMock()
            mock_router.route.return_value = "knowledge_retrieval"
            mock_router_cls.return_value = mock_router

            response = await client.post(
                "/api/v1/query",
                json={"query": "What is GraphRAG?", "top_k": 3},
            )

        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_query_missing_body_returns_422(self, client: AsyncClient) -> None:
        """POST /api/v1/query with missing body should return 422."""
        response = await client.post("/api/v1/query", json={})
        assert response.status_code == 422
