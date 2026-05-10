"""FastAPI routers for the CodaCite application.

This module exposes the external REST API, mapping HTTP requests to internal
Application Use Cases. It defines the schemas and endpoints for the core
GraphRAG functionality.

Endpoints:
    -   `POST /ingest`: Uploads documents and triggers the 8-phase ingestion
        pipeline in the background.
    -   `POST /query`: Executes hybrid vector+graph search for context snippets.
    -   `POST /chat`: Grounded conversational interface with conversation history.
    -   `POST /enhance`: Triggers global graph analysis (Louvain communities).
    -   `GET /notebooks`: Manages logical workspace collections.
"""

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.api.dependencies import (
    get_chat_use_case,
    get_document_store,
    get_enhancement_use_case,
    get_ingestion_use_case,
    get_notebook_use_case,
    get_retrieval_use_case,
    get_vlm,
)
from app.core.bootstrap import get_bootstrap_status
from app.core.interfaces import DocumentStore
from app.models.models import Document
from app.pipelines.extraction.enhancement import GraphEnhancementUseCase
from app.pipelines.generation.chat import ChatUseCase
from app.pipelines.ingestion.ingestion import DocumentIngestionUseCase
from app.pipelines.ingestion.loader import DocumentLoader
from app.pipelines.notebooks.notebook_manager import NotebookUseCase
from app.pipelines.retrieval.retrieval import GraphRAGRetrievalUseCase

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
        answer: Optional generated answer from the RAG pipeline.
    """

    query: str
    intent: str
    results: list[dict[str, Any]]
    answer: str | None = None


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
    vlm: Any = Depends(get_vlm),
) -> IngestResponse:
    """Ingest a document and queue it for background graph extraction.

    Args:
        file: The uploaded document file (PDF/Text).
        background_tasks: FastAPI background tasks handler.
        notebook_id: Optional ID of the notebook to attach this document to.
        ingestion_use_case: Use case for document ingestion.
        vlm: Shared/Singleton LocalVLM instance.

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
    loader = DocumentLoader(vlm=vlm)

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    temp_file_path = uploads_dir / f"{uuid.uuid4()}{suffix}"

    try:
        with open(temp_file_path, "wb") as f:
            f.write(content_bytes)

        loaded_documents = loader.load(temp_file_path)
    except ValueError as exc:
        logger.warning("[API] Invalid file format for '%s': %s", file.filename, exc)
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format or content.",
        ) from exc
    except Exception as exc:
        logger.exception("[API] Unexpected error during ingestion of '%s'", file.filename)
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse uploaded file: {str(exc)}",
        ) from exc

    text = "\n".join(document.text for document in loaded_documents)

    # Phase 1: Create record and relate to notebook
    document_id = await ingestion_use_case.ingest_and_queue(
        text=text, filename=file.filename, file_path=str(temp_file_path), notebook_id=notebook_id
    )

    # Phase 2: Background processing
    background_tasks.add_task(
        ingestion_use_case.process_background, document_id, text, file.filename
    )

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
    )


@api_router.get("/documents/{document_id}/status", response_model=IngestResponse)
async def get_document_status(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store),
) -> IngestResponse:
    """Check the processing status of a document.

    Args:
        document_id: The ID of the document.
        document_store: The document storage port.

    Returns:
        The current status of the document.
    """
    document = await document_store.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    return IngestResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
    )


@api_router.get("/documents/{document_id}/view")
async def view_document(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store),
) -> FileResponse:
    """View a raw document file.

    Args:
        document_id: The ID of the document.
        document_store: The document storage port.

    Returns:
        A FileResponse streaming the document.
    """
    document = await document_store.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    file_path = document.file_path
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raw file for document {document_id} not found.",
        )

    suffix = Path(file_path).suffix.lower()
    media_type = "application/pdf" if suffix == ".pdf" else "text/plain"
    return FileResponse(file_path, media_type=media_type)


@api_router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store),
) -> Response:
    """Cascading delete of a document, its chunks, and physical file.

    Args:
        document_id: The ID of the document to remove.
        document_store: The document storage port.

    Returns:
        HTTP 204 No Content on success.
    """
    logger.info("[API] Request to delete document: %s", document_id)
    success = await document_store.delete_document(document_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found or already deleted.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


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

    return QueryResponse(
        query=request.query,
        intent="knowledge_retrieval",
        results=results.get("documents", []),
    )


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
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint.

    Returns:
        A status dictionary including bootstrap health.
    """
    bootstrap = get_bootstrap_status()
    return {
        "status": "ok" if bootstrap["status"] != "failed" else "degraded",
        "bootstrap": bootstrap,
    }


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
) -> StreamingResponse:
    """Conversational endpoint with GraphRAG grounding and streaming.

    Args:
        request: Query and history.
        chat_use_case: Conversational logic coordinator.

    Returns:
        StreamingResponse yielding Server-Sent Events (SSE).
    """
    logger.info("[API] Received chat request: '%s'", request.query)

    async def generate_response():
        async for chunk in chat_use_case.execute(
            request.query, history=request.history, notebook_ids=request.notebook_ids
        ):
            # The chunk already includes the token payload formatting or event: citations
            # We just need to prepend `data: ` to plain token payloads.
            # But wait, in chat.py, we yield `json.dumps({"token": chunk}) + "\n"`.
            # To be a valid SSE stream event, it should look like:
            # data: {"token": "..."}\n\n
            if chunk.startswith("data: ") or chunk.startswith("event: "):
                yield chunk + "\n\n"
            else:
                yield f"data: {chunk}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@api_router.get("/notebook")
async def notebook_ui(
    request: Request,
) -> Response:
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
