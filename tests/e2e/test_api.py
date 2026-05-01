"""End-to-end (Functional) tests for the FastAPI application.

Validates that the API layer correctly routes requests to application use cases,
handles file uploads, manages dependency overrides, and returns appropriate
HTTP status codes and responses.
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.ingestion.loader import LoadedDocument
from app.interfaces.dependencies import (
    get_db,
    get_extraction_use_case,
    get_ingestion_use_case,
    get_retrieval_use_case,
)
from app.main import app


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient]:
    """Provides an asynchronous HTTP client for testing the FastAPI app.

    Yields:
        An AsyncClient instance configured with ASGITransport.
    """
    # Reset dependency overrides for safety
    app.dependency_overrides.clear()
    # Mock DB dependency globally for these tests
    app.dependency_overrides[get_db] = lambda: None

    transport = ASGITransport(app=app)  # type: ignore
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health(async_client: AsyncClient) -> None:
    """Tests the health check endpoint.

    Given:
        A running FastAPI application.
    When:
        A GET request is sent to /api/v1/health.
    Then:
        It should return 200 OK with status "ok".

    Args:
        async_client: The async HTTP client fixture.
    """
    # Act
    response = await async_client.get("/api/v1/health")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ["ok", "degraded"]
    assert "bootstrap" in body


@pytest.mark.asyncio
async def test_ingest_no_file(async_client: AsyncClient) -> None:
    """Tests that ingestion fails if no file is provided.

    Given:
        A POST request to /api/v1/ingest without multipart data.
    When:
        The request is processed.
    Then:
        It should return 422 Unprocessable Entity.

    Args:
        async_client: The async HTTP client fixture.
    """
    # Act
    response = await async_client.post("/api/v1/ingest")

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_markdown_success(
    async_client: AsyncClient, mock_ingestion_use_case: Any, mock_extraction_use_case: Any
) -> None:
    """Tests successful markdown ingestion.

    Given:
        A valid markdown file and mocked ingestion/extraction services.
    When:
        A POST request is sent to /api/v1/ingest.
    Then:
        It should return 202 Accepted with document metadata.

    Args:
        async_client: The async HTTP client fixture.
        mock_ingestion_use_case: Mocked ingestion use case from conftest.
        mock_extraction_use_case: Mocked extraction use case from conftest.
    """
    # Arrange
    mock_ingestion_use_case.ingest_and_queue.return_value = "doc:123"
    app.dependency_overrides[get_ingestion_use_case] = lambda: mock_ingestion_use_case
    app.dependency_overrides[get_extraction_use_case] = lambda: mock_extraction_use_case

    # Act
    response = await async_client.post(
        "/api/v1/ingest",
        files={"file": ("note.md", b"# Title\n\nBody text", "text/markdown")},
    )

    # Assert
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "note.md"
    assert body["status"] == "processing"


@pytest.mark.asyncio
async def test_ingest_pdf_success(
    async_client: AsyncClient,
    mock_ingestion_use_case: Any,
    mock_extraction_use_case: Any,
    mocker: Any,
) -> None:
    """Tests successful PDF ingestion with mocked loader.

    Given:
        A PDF file and a mocked DocumentLoader.
    When:
        A POST request is sent to /api/v1/ingest.
    Then:
        It should return 202 Accepted.

    Args:
        async_client: The async HTTP client fixture.
        mock_ingestion_use_case: Mocked ingestion use case.
        mock_extraction_use_case: Mocked extraction use case.
        mocker: Pytest-mock fixture.
    """
    # Arrange
    mock_ingestion_use_case.ingest_and_queue.return_value = "doc:789"
    app.dependency_overrides[get_ingestion_use_case] = lambda: mock_ingestion_use_case
    app.dependency_overrides[get_extraction_use_case] = lambda: mock_extraction_use_case

    load_mock = mocker.patch("app.interfaces.routers.DocumentLoader.load")
    load_mock.return_value = [
        LoadedDocument(text="Extracted PDF text", source="/tmp/test.pdf", format="pdf")
    ]

    # Act
    response = await async_client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )

    # Assert
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "doc.pdf"
    assert body["status"] == "processing"
    load_mock.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_unsupported_format_returns_400(async_client: AsyncClient) -> None:
    """Tests that unsupported file formats are rejected.

    Given:
        An executable file (.exe).
    When:
        A POST request is sent to /api/v1/ingest.
    Then:
        It should return 400 Bad Request.

    Args:
        async_client: The async HTTP client fixture.
    """
    # Act
    response = await async_client.post(
        "/api/v1/ingest",
        files={"file": ("archive.exe", b"MZ", "application/octet-stream")},
    )

    # Assert
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_notebook_ui(async_client: AsyncClient) -> None:
    """Tests the notebook UI route.

    Given:
        A request for the notebook UI.
    When:
        A GET request is sent to /api/v1/notebook.
    Then:
        It should return 200 OK with HTML content.

    Args:
        async_client: The async HTTP client fixture.
    """
    # Act
    response = await async_client.get("/api/v1/notebook")

    # Assert
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_get_notebook_documents(
    async_client: AsyncClient, mock_notebook_use_case: Any
) -> None:
    """Tests listing documents in a notebook.

    Given:
        A mocked notebook use case.
    When:
        A GET request is sent to /api/v1/notebooks/{id}/documents.
    Then:
        It should return 200 OK with the document list.

    Args:
        async_client: The async HTTP client fixture.
        mock_notebook_use_case: Mocked notebook use case.
    """
    # Arrange
    from app.interfaces.dependencies import get_notebook_use_case

    mock_notebook_use_case.get_documents.return_value = []
    app.dependency_overrides[get_notebook_use_case] = lambda: mock_notebook_use_case

    # Act
    response = await async_client.get("/api/v1/notebooks/nb123/documents")

    # Assert
    assert response.status_code == 200
    assert response.json() == []
    mock_notebook_use_case.get_documents.assert_called_once_with("nb123")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_status",
    [
        ("What are the key findings?", 200),
        ("", 422),
    ],
)
async def test_query_endpoint(
    async_client: AsyncClient, query: str, expected_status: int, mock_retrieval_use_case: Any
) -> None:
    """Tests the query endpoint with various inputs.

    Given:
        A query string and a mocked retrieval service.
    When:
        A POST request is sent to /api/v1/query.
    Then:
        It should return the expected status code.

    Args:
        async_client: The async HTTP client fixture.
        query: The input query string.
        expected_status: Expected HTTP status code.
        mock_retrieval_use_case: Mocked retrieval use case.
    """
    # Arrange
    mock_retrieval_use_case.execute.return_value = [{"text": "Sample result", "score": 0.9}]
    app.dependency_overrides[get_retrieval_use_case] = lambda: mock_retrieval_use_case

    payload = {}
    if query:
        payload["query"] = query

    # Act
    response = await async_client.post("/api/v1/query", json=payload)

    # Assert
    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_enhance_endpoint(async_client: AsyncClient, mocker: Any) -> None:
    """Tests the graph enhancement endpoint.

    Given:
        A request to trigger graph enhancement.
    When:
        The /api/v1/enhance endpoint is called.
    Then:
        It should return 200 OK.

    Args:
        async_client: The async HTTP client fixture.
        mocker: Pytest-mock fixture.
    """
    # Arrange
    from app.interfaces.dependencies import get_enhancement_use_case

    mock_use_case = mocker.AsyncMock()
    app.dependency_overrides[get_enhancement_use_case] = lambda: mock_use_case

    # Act
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ["ok", "degraded"]
    assert "bootstrap" in body
    response = await async_client.post("/api/v1/enhance")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"message": "Graph communities generated successfully."}
    mock_use_case.execute.assert_called_once()


@pytest.mark.asyncio
async def test_chat_endpoint(async_client: AsyncClient, mocker: Any) -> None:
    """Tests the chat endpoint.

    Given:
        A chat query and history.
    When:
        A POST request is sent to /api/v1/chat.
    Then:
        It should return 200 OK with the assistant response.

    Args:
        async_client: The async HTTP client fixture.
        mocker: Pytest-mock fixture.
    """
    # Arrange
    from app.interfaces.dependencies import get_chat_use_case

    mock_use_case = mocker.AsyncMock()
    mock_use_case.execute.return_value = "This is a grounded response."
    app.dependency_overrides[get_chat_use_case] = lambda: mock_use_case

    # Act
    payload = {
        "query": "What is semantic blocking?",
        "history": [{"role": "user", "content": "Hello"}],
    }
    response = await async_client.post("/api/v1/chat", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.json() == {"response": "This is a grounded response."}
    mock_use_case.execute.assert_called_once_with(
        "What is semantic blocking?",
        history=[{"role": "user", "content": "Hello"}],
        notebook_ids=None,
    )
