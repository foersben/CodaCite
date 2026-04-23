"""FastAPI routers."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.application.enhancement import GraphEnhancementUseCase
from app.application.extraction import GraphExtractionUseCase
from app.application.ingestion import DocumentIngestionUseCase
from app.application.retrieval import GraphRAGRetrievalUseCase
from app.ingestion.loader import DocumentLoader
from app.interfaces.dependencies import (
    get_enhancement_use_case,
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
    ingestion_use_case: DocumentIngestionUseCase = Depends(get_ingestion_use_case),
    extraction_use_case: GraphExtractionUseCase = Depends(get_extraction_use_case),
) -> IngestResponse:
    """Ingest a document, chunk it, extract entities/relations, and update GraphRAG."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a filename.",
        )

    logger.info("Starting ingestion for file: %s", file.filename)

    suffix = Path(file.filename).suffix.lower()
    content_bytes = await file.read()
    loader = DocumentLoader()

    temp_file_path: str | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content_bytes)
            temp_file_path = temp_file.name

        loaded_documents = loader.load(Path(temp_file_path))
    except ValueError as exc:
        logger.warning("Invalid file format or content for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format or content.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive parse guard
        logger.warning("Failed to parse uploaded file '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse uploaded file '{file.filename}'.",
        ) from exc
    finally:
        if temp_file_path:
            Path(temp_file_path).unlink(missing_ok=True)

    text = "\n".join(document.text for document in loaded_documents)

    chunks = await ingestion_use_case.execute(text, filename=file.filename or "unknown")

    logger.info("Document chunked into %d chunks", len(chunks))

    nodes, _ = await extraction_use_case.execute(chunks)

    logger.info("Extraction complete with %d entities", len(nodes))

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


@api_router.post("/enhance")
async def api_enhance(
    enhancement_use_case: GraphEnhancementUseCase = Depends(get_enhancement_use_case),
) -> dict[str, str]:
    """Enhance the graph by running Louvain community detection."""
    logger.info("Starting graph enhancement pipeline")
    await enhancement_use_case.execute()
    logger.info("Graph enhancement pipeline complete")
    return {"message": "Graph communities generated successfully."}


@api_router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
