"""FastAPI routers."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from pydantic import BaseModel

from app.application.extraction import GraphExtractionUseCase
from app.application.ingestion import DocumentIngestionUseCase
from app.application.retrieval import GraphRAGRetrievalUseCase
from app.interfaces.dependencies import (
    get_extraction_use_case,
    get_ingestion_use_case,
    get_retrieval_use_case,
)

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api/v1")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    filename: str
    chunks_processed: int
    entities_extracted: int


class QueryRequest(BaseModel):
    """Request model for knowledge base query."""

    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for knowledge base query."""

    query: str
    intent: str
    results: list[dict[str, str | float]]


@api_router.post("/ingest", response_model=IngestResponse)
async def api_ingest(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    ingestion_use_case: DocumentIngestionUseCase = Depends(get_ingestion_use_case),
    extraction_use_case: GraphExtractionUseCase = Depends(get_extraction_use_case),
) -> IngestResponse:
    """Ingest a document, chunk it, extract entities/relations, and update GraphRAG."""
    logger.info(f"Starting ingestion for file: {file.filename}")

    content_bytes = await file.read()
    text = content_bytes.decode("utf-8", errors="ignore")

    chunks = await ingestion_use_case.execute(text, filename=file.filename or "unknown")

    logger.info(f"Document chunked into {len(chunks)} chunks")

    nodes, _ = await extraction_use_case.execute(chunks)

    logger.info(f"Extraction complete with {len(nodes)} entities")

    return IngestResponse(
        filename=file.filename or "unknown",
        chunks_processed=len(chunks),
        entities_extracted=len(nodes),
    )


@api_router.post("/query", response_model=QueryResponse)
async def api_query(
    request: QueryRequest,
    retrieval_use_case: GraphRAGRetrievalUseCase = Depends(get_retrieval_use_case),
) -> QueryResponse:
    """Query the GraphRAG knowledge base."""
    logger.info(f"Processing query '{request.query}' with top_k={request.top_k}")

    results = await retrieval_use_case.execute(request.query, top_k=request.top_k)

    return QueryResponse(query=request.query, intent="knowledge_retrieval", results=results)


@api_router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
