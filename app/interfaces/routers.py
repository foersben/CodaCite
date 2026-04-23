"""FastAPI routers for the application.

This module defines the RESTful API endpoints for document ingestion,
knowledge retrieval, graph enhancement, and conversational chat.
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.application.chat import ChatUseCase
from app.application.enhancement import GraphEnhancementUseCase
from app.application.extraction import GraphExtractionUseCase
from app.application.ingestion import DocumentIngestionUseCase
from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.models import Document
from app.domain.ports import DocumentStore
from app.ingestion.loader import DocumentLoader
from app.interfaces.dependencies import (
    get_chat_use_case,
    get_document_store,
    get_enhancement_use_case,
    get_extraction_use_case,
    get_ingestion_use_case,
    get_retrieval_use_case,
)

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api/v1")
templates = Jinja2Templates(directory="app/templates")


class IngestResponse(BaseModel):
    """Response model for document ingestion.

    Attributes:
        filename: Name of the processed file.
        chunks_processed: Number of text chunks created.
        entities_extracted: Number of unique entities found.
    """

    filename: str
    chunks_processed: int
    entities_extracted: int


class QueryRequest(BaseModel):
    """Request model for knowledge base query.

    Attributes:
        query: The search string.
        top_k: Number of chunks to retrieve.
    """

    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for knowledge base query.

    Attributes:
        query: Original user query.
        intent: Classified intent (default: knowledge_retrieval).
        results: List of retrieved context chunks with scores.
    """

    query: str
    intent: str
    results: list[dict[str, str | float]]


class ChatRequest(BaseModel):
    """Request model for chat conversations.

    Attributes:
        message: The user's new message.
        history: Optional list of previous message dicts.
    """

    message: str
    history: list[dict[str, str]] | None = None


class ChatResponse(BaseModel):
    """Response model for chat conversations.

    Attributes:
        response: The assistant's grounded response.
    """

    response: str


@api_router.post("/ingest", response_model=IngestResponse)
async def api_ingest(
    file: UploadFile,
    ingestion_use_case: DocumentIngestionUseCase = Depends(get_ingestion_use_case),
    extraction_use_case: GraphExtractionUseCase = Depends(get_extraction_use_case),
) -> IngestResponse:
    """Ingest a document, chunk it, and extract graph knowledge.

    Args:
        file: The uploaded document file (PDF/Text).
        ingestion_use_case: Use case for processing and embedding text.
        extraction_use_case: Use case for mapping chunks to graph entities.

    Returns:
        Summary of the ingestion process.

    Raises:
        HTTPException: If file parsing fails or filename is missing.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a filename.",
        )

    logger.info("[API] Starting ingestion for file: %s", file.filename)

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
        logger.warning("[API] Invalid file format for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format or content.",
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.error("[API] Failed to parse file '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse uploaded file '{file.filename}'.",
        ) from exc
    finally:
        if temp_file_path:
            Path(temp_file_path).unlink(missing_ok=True)

    text = "\n".join(document.text for document in loaded_documents)

    chunks = await ingestion_use_case.execute(text, filename=file.filename or "unknown")
    logger.info("[API] Document chunked into %d chunks", len(chunks))

    nodes, _ = await extraction_use_case.execute(chunks)
    logger.info("[API] Extraction complete with %d entities", len(nodes))

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
    """Query the GraphRAG knowledge base.

    Args:
        request: Query parameters.
        retrieval_use_case: Knowledge retrieval coordinator.

    Returns:
        List of relevant context fragments and associated metadata.
    """
    logger.info("[API] Processing query: '%s' (top_k=%d)", request.query, request.top_k)

    results = await retrieval_use_case.execute(request.query, top_k=request.top_k)

    return QueryResponse(query=request.query, intent="knowledge_retrieval", results=results)


@api_router.post("/enhance")
async def api_enhance(
    enhancement_use_case: GraphEnhancementUseCase = Depends(get_enhancement_use_case),
) -> dict[str, str]:
    """Enhance the graph by running Louvain community detection.

    Args:
        enhancement_use_case: Logic for analyzing graph structure and summarizing clusters.

    Returns:
        Success message.
    """
    logger.info("[API] Starting graph enhancement pipeline")
    await enhancement_use_case.execute()
    logger.info("[API] Graph enhancement pipeline complete")
    return {"message": "Graph communities generated successfully."}


@api_router.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint.

    Returns:
        A status dictionary.
    """
    return {"status": "ok"}


@api_router.get("/documents", response_model=list[Document])
async def list_documents(
    doc_store: DocumentStore = Depends(get_document_store),
) -> list[Document]:
    """List all ingested documents.

    Args:
        doc_store: Document storage implementation.

    Returns:
        List of Document domain models.
    """
    return await doc_store.get_all_documents()


@api_router.post("/chat", response_model=ChatResponse)
async def api_chat(
    request: ChatRequest,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case),
) -> ChatResponse:
    """Chat with the knowledge base using conversation history.

    Args:
        request: Chat message and optional history.
        chat_use_case: Coordinator for RAG-based chat.

    Returns:
        The generated response string.
    """
    logger.info("[API] Processing chat message")
    response = await chat_use_case.execute(request.message, history=request.history)
    return ChatResponse(response=response)


@api_router.get("/notebook")
async def notebook_ui(
    request: Request,
) -> Any:
    """Serve the NotebookLM-style UI.

    Args:
        request: FastAPI request object.

    Returns:
        The rendered HTML template.
    """
    return templates.TemplateResponse("notebook.html", {"request": request})

