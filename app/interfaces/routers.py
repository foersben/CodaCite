"""FastAPI routers for the application.

This module defines the RESTful API endpoints for document ingestion,
knowledge retrieval, graph enhancement, and conversational chat.
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Response

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, status
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.application.chat import ChatUseCase
from app.application.enhancement import GraphEnhancementUseCase
from app.application.ingestion import DocumentIngestionUseCase
from app.application.notebook import NotebookUseCase
from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.models import Document
from app.domain.ports import DocumentStore
from app.ingestion.loader import DocumentLoader
from app.interfaces.dependencies import (
    get_chat_use_case,
    get_document_store,
    get_enhancement_use_case,
    get_ingestion_use_case,
    get_notebook_use_case,
    get_retrieval_use_case,
)

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api/v1")
templates = Jinja2Templates(directory="app/templates")


class IngestResponse(BaseModel):
    """Response model for document ingestion.

    Attributes:
        document_id: Unique identifier for the document.
        filename: Name of the processed file.
        status: Current status of the ingestion.
    """

    document_id: str
    filename: str
    status: str


class QueryRequest(BaseModel):
    """Request model for knowledge base query.

    Attributes:
        query: The search string.
        top_k: Number of chunks to retrieve.
        notebook_ids: Optional list of notebook IDs to filter context.
    """

    query: str
    top_k: int = 5
    notebook_ids: list[str] | None = None


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
    """Request model for conversational chat.

    Attributes:
        query: The user's message.
        history: Previous messages in the conversation.
        notebook_ids: Optional list of notebook IDs to filter context.
    """

    query: str
    history: list[dict[str, str]] | None = None
    notebook_ids: list[str] | None = None


class ChatResponse(BaseModel):
    """Response model for chat conversations.

    Attributes:
        response: The assistant's grounded response.
    """

    response: str


class NotebookRequest(BaseModel):
    """Request model for creating a notebook.

    Attributes:
        title: The name of the notebook.
        description: Optional description.
    """

    title: str
    description: str | None = None


class NotebookResponse(BaseModel):
    """Response model for notebook operations.

    Attributes:
        id: Unique identifier.
        title: Notebook name.
    """

    id: str
    title: str


@api_router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def api_ingest(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    notebook_id: str | None = None,
    ingestion_use_case: DocumentIngestionUseCase = Depends(get_ingestion_use_case),
) -> IngestResponse:
    """Ingest a document and queue it for background graph extraction.

    Args:
        file: The uploaded document file (PDF/Text).
        background_tasks: FastAPI background tasks handler.
        notebook_id: Optional ID of the notebook to attach this document to.
        ingestion_use_case: Use case for document ingestion.

    Returns:
        Immediate response with document ID and 'processing' status.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a filename.",
        )

    logger.info("[API] Starting ingestion for file: %s (Notebook: %s)", file.filename, notebook_id)

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
    except Exception as exc:
        logger.exception("[API] Unexpected error during ingestion of '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse uploaded file: {str(exc)}",
        ) from exc
    finally:
        if temp_file_path:
            Path(temp_file_path).unlink(missing_ok=True)

    text = "\n".join(document.text for document in loaded_documents)

    # Phase 1: Create record and relate to notebook
    document_id = await ingestion_use_case.ingest_and_queue(
        text=text, filename=file.filename, notebook_id=notebook_id
    )

    # Phase 2: Background processing
    background_tasks.add_task(ingestion_use_case.process_background, document_id, text)

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
    )


@api_router.post("/query", response_model=QueryResponse)
async def api_query(
    request: QueryRequest,
    retrieval_use_case: GraphRAGRetrievalUseCase = Depends(get_retrieval_use_case),
) -> QueryResponse:
    """Perform semantic search on the knowledge base.

    Args:
        request: Query parameters.
        retrieval_use_case: Knowledge retrieval coordinator.

    Returns:
        List of relevant context fragments and associated metadata.
    """
    logger.info(
        "[API] Processing query: '%s' (top_k=%d, Notebooks=%s)",
        request.query,
        request.top_k,
        request.notebook_ids,
    )

    results = await retrieval_use_case.execute(
        request.query, top_k=request.top_k, notebook_ids=request.notebook_ids
    )

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


@api_router.post("/chat")
async def api_chat(
    request: ChatRequest,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case),
) -> dict[str, str]:
    """Conversational endpoint with GraphRAG grounding.

    Args:
        request: Query and history.
        chat_use_case: Conversational logic coordinator.

    Returns:
        The generated response.
    """
    logger.info("[API] Received chat request: '%s'", request.query)
    response = await chat_use_case.execute(
        request.query, history=request.history, notebook_ids=request.notebook_ids
    )
    return {"response": response}


@api_router.get("/notebook")
async def notebook_ui(
    request: Request,
) -> "Response":
    """Serve the NotebookLM-style UI.

    Args:
        request: FastAPI request object.

    Returns:
        The rendered HTML template.
    """
    return templates.TemplateResponse(request, "notebook.html")


@api_router.get("/notebooks", response_model=list[NotebookResponse])
async def list_notebooks(
    notebook_use_case: NotebookUseCase = Depends(get_notebook_use_case),
) -> list[NotebookResponse]:
    """List all available notebooks.

    Args:
        notebook_use_case: Notebook management use case.

    Returns:
        A list of notebook summaries.
    """
    notebooks = await notebook_use_case.list_notebooks()
    return [NotebookResponse(id=n.id, title=n.title) for n in notebooks]


@api_router.post("/notebooks", response_model=NotebookResponse, status_code=status.HTTP_201_CREATED)
async def create_notebook(
    request: NotebookRequest,
    notebook_use_case: NotebookUseCase = Depends(get_notebook_use_case),
) -> NotebookResponse:
    """Create a new notebook.

    Args:
        request: Notebook details.
        notebook_use_case: Notebook management use case.

    Returns:
        The created notebook summary.
    """
    notebook = await notebook_use_case.create_notebook(request.title, request.description)
    return NotebookResponse(id=notebook.id, title=notebook.title)


@api_router.delete("/notebooks/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notebook(
    notebook_id: str,
    notebook_use_case: NotebookUseCase = Depends(get_notebook_use_case),
) -> None:
    """Delete a notebook.

    Args:
        notebook_id: The ID of the notebook to remove.
        notebook_use_case: Notebook management use case.
    """
    await notebook_use_case.delete_notebook(notebook_id)


@api_router.post("/notebooks/{notebook_id}/documents/{document_id}", status_code=status.HTTP_200_OK)
async def add_document_to_notebook(
    notebook_id: str,
    document_id: str,
    notebook_use_case: NotebookUseCase = Depends(get_notebook_use_case),
) -> dict[str, str]:
    """Add an existing document to a notebook.

    Args:
        notebook_id: The notebook ID.
        document_id: The document ID.
        notebook_use_case: Notebook management use case.

    Returns:
        Success message.
    """
    await notebook_use_case.add_document(notebook_id, document_id)
    return {"message": f"Document {document_id} added to notebook {notebook_id}"}


@api_router.get("/notebooks/{notebook_id}/documents", response_model=list[Document])
async def get_notebook_documents(
    notebook_id: str,
    notebook_use_case: NotebookUseCase = Depends(get_notebook_use_case),
) -> list[Document]:
    """List all documents in a notebook.

    Args:
        notebook_id: The notebook ID.
        notebook_use_case: Notebook management use case.

    Returns:
        List of documents.
    """
    return await notebook_use_case.get_documents(notebook_id)


@api_router.delete(
    "/notebooks/{notebook_id}/documents/{document_id}", status_code=status.HTTP_200_OK
)
async def remove_document_from_notebook(
    notebook_id: str,
    document_id: str,
    notebook_use_case: NotebookUseCase = Depends(get_notebook_use_case),
) -> dict[str, str]:
    """Remove a document from a notebook.

    Args:
        notebook_id: The notebook ID.
        document_id: The document ID.
        notebook_use_case: Notebook management use case.

    Returns:
        Success message.
    """
    await notebook_use_case.remove_document(notebook_id, document_id)
    return {"message": f"Document {document_id} removed from notebook {notebook_id}"}
