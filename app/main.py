"""FastAPI application entry point for the Enterprise Omni-Copilot."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.agents.router import IntentRouter
from app.embeddings.embedder import LocalEmbedder
from app.graph.extractor import EntityExtractor
from app.graph.store import GraphStore
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader
from app.ingestion.preprocessor import TextPreprocessor
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown"}

_TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")


class QueryRequest(BaseModel):
    """Request body for the /api/v1/query endpoint."""

    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    """A single result item returned by the query endpoint."""

    text: str
    score: float
    source: str = ""


class QueryResponse(BaseModel):
    """Response body for the /api/v1/query endpoint."""

    query: str
    intent: str
    results: list[QueryResult]


class IngestResponse(BaseModel):
    """Response body for the /api/v1/ingest endpoint."""

    filename: str
    chunks_processed: int
    entities_extracted: int


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Using a factory function enables easy test isolation (each test can get
    a fresh instance without shared state).
    """
    app = FastAPI(
        title="Enterprise Omni-Copilot",
        description="GraphRAG-based Document Intelligence and Workflow Automation",
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    @app.post("/api/v1/ingest", response_model=IngestResponse, tags=["ingestion"])
    async def ingest(file: Annotated[UploadFile, File()]) -> IngestResponse:
        """Upload and ingest a document (PDF, DOCX, or Markdown)."""
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file format '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
            )

        # Persist upload to a temp file so loaders can read it
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            loader = DocumentLoader()
            preprocessor = TextPreprocessor()
            chunker = TextChunker()
            embedder = LocalEmbedder()
            extractor = EntityExtractor(llm=None)  # LLM injected via DI in production
            store = GraphStore(db=None)  # DB connection injected via DI in production

            docs = loader.load(tmp_path)
            all_chunks: list[str] = []
            for doc in docs:
                clean_text = preprocessor.process(doc.text)
                all_chunks.extend(chunker.chunk(clean_text))

            embeddings = embedder.embed(all_chunks)
            total_entities = 0

            for chunk_text, embedding in zip(all_chunks, embeddings, strict=True):
                chunk_id = f"chunk:{hashlib.md5(chunk_text.encode()).hexdigest()[:12]}"
                await store.store_chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    embedding=embedding,
                    source=filename,
                )
                entities, relationships = extractor.extract(chunk_text)
                total_entities += len(entities)
                for entity in entities:
                    await store.store_entity(entity)
                for rel in relationships:
                    await store.store_relationship(rel)

        finally:
            tmp_path.unlink(missing_ok=True)

        return IngestResponse(
            filename=filename,
            chunks_processed=len(all_chunks),
            entities_extracted=total_entities,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @app.post("/api/v1/query", response_model=QueryResponse, tags=["retrieval"])
    async def query(request: QueryRequest) -> QueryResponse:
        """Run a hybrid GraphRAG query against the knowledge base."""
        embedder = LocalEmbedder()
        store = GraphStore(db=None)
        reranker = CrossEncoderReranker(model_path="")
        retriever = HybridRetriever(store=store, embedder=embedder, reranker=reranker)
        router = IntentRouter(llm=None)

        intent = router.route(request.query)
        results = await retriever.retrieve(query=request.query, top_k=request.top_k)

        return QueryResponse(
            query=request.query,
            intent=str(intent),
            results=[
                QueryResult(text=r.text, score=r.score, source=r.source) for r in results
            ],
        )

    # ------------------------------------------------------------------
    # Web UI (server-side rendered, zero client-side JavaScript)
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def ui_index(request: Request, message: str = "", error: bool = False) -> HTMLResponse:
        """Render the main web UI."""
        return _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {"message": message, "error": error},
        )

    @app.post("/ui/ingest", include_in_schema=False)
    async def ui_ingest(
        request: Request, file: Annotated[UploadFile, File()]
    ) -> RedirectResponse:
        """Handle document upload from the web UI and redirect with a status message."""
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            return RedirectResponse(
                url=f"/?error=1&message={quote(f'Unsupported file type {suffix!r}')}",
                status_code=303,
            )

        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            loader = DocumentLoader()
            preprocessor = TextPreprocessor()
            chunker = TextChunker()
            embedder = LocalEmbedder()
            extractor = EntityExtractor(llm=None)
            store = GraphStore(db=None)

            docs = loader.load(tmp_path)
            all_chunks: list[str] = []
            for doc in docs:
                clean_text = preprocessor.process(doc.text)
                all_chunks.extend(chunker.chunk(clean_text))

            embeddings = embedder.embed(all_chunks)
            total_entities = 0

            for chunk_text, embedding in zip(all_chunks, embeddings, strict=True):
                chunk_id = f"chunk:{hashlib.md5(chunk_text.encode()).hexdigest()[:12]}"
                await store.store_chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    embedding=embedding,
                    source=filename,
                )
                entities, relationships = extractor.extract(chunk_text)
                total_entities += len(entities)
                for entity in entities:
                    await store.store_entity(entity)
                for rel in relationships:
                    await store.store_relationship(rel)
        finally:
            tmp_path.unlink(missing_ok=True)

        msg = quote(f"Ingested '{filename}' — {len(all_chunks)} chunks, {total_entities} entities")
        return RedirectResponse(url=f"/?message={msg}", status_code=303)

    @app.post("/ui/query", response_class=HTMLResponse, include_in_schema=False)
    async def ui_query(
        request: Request,
        query_text: Annotated[str, Form(alias="query")] = "",
        top_k: Annotated[int, Form()] = 5,
    ) -> HTMLResponse:
        """Handle a query from the web UI and render results inline."""
        embedder = LocalEmbedder()
        store = GraphStore(db=None)
        reranker = CrossEncoderReranker(model_path="")
        retriever = HybridRetriever(store=store, embedder=embedder, reranker=reranker)
        router = IntentRouter(llm=None)

        intent = router.route(query_text)
        results = await retriever.retrieve(query=query_text, top_k=top_k)

        return _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "query": query_text,
                "top_k": top_k,
                "intent": str(intent),
                "results": [
                    QueryResult(text=r.text, score=r.score, source=r.source) for r in results
                ],
            },
        )

    return app


# ---------------------------------------------------------------------------
# ASGI entry point
# ---------------------------------------------------------------------------

app = create_app()
