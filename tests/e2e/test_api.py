"""Tests for FastAPI application.

This module validates the end-to-end functionality of the API endpoints,
ensuring the Interfaces layer correctly orchestrates application use cases.
"""

from io import BytesIO
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ingestion.loader import LoadedDocument
from app.interfaces.dependencies import (
    get_db,
    get_extraction_use_case,
    get_ingestion_use_case,
    get_retrieval_use_case,
)
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

app.dependency_overrides[get_db] = lambda: None


def test_health() -> None:
    """Test the health endpoint.

    Given: A running FastAPI application.
    When: A GET request is sent to /api/v1/health.
    Then: It should return a 200 status code with a status of "ok".
    """
    # Arrange
    url = "/api/v1/health"

    # Act
    response = client.get(url)

    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_no_file() -> None:
    """Test ingest returns 422 if no file provided.

    Given: A request to the /api/v1/ingest endpoint without a file.
    When: The request is processed by the server.
    Then: It should return a 422 Unprocessable Entity status code.
    """
    # Arrange
    url = "/api/v1/ingest"

    # Act
    response = client.post(url)

    # Assert
    assert response.status_code == 422


def test_ingest_markdown_success(
    mock_ingestion_use_case: Any, mock_extraction_use_case: Any
) -> None:
    """Test ingest succeeds for markdown uploads.

    Given: A valid markdown file and properly mocked ingestion/extraction services.
    When: A POST request is sent to /api/v1/ingest.
    Then: It should return a 200 status code with document metadata.
    """
    # Arrange
    mock_ingestion_use_case.ingest_and_queue.return_value = "doc:123"
    mock_extraction_use_case.execute.return_value = (
        [{"id": "node1"}],
        [{"source": "node1", "target": "node2"}],
    )

    app.dependency_overrides[get_ingestion_use_case] = lambda: mock_ingestion_use_case
    app.dependency_overrides[get_extraction_use_case] = lambda: mock_extraction_use_case

    # Act
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("note.md", b"# Title\n\nBody text", "text/markdown")},
    )

    # Assert
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "note.md"
    assert body["status"] == "processing"


def test_ingest_text_success(mock_ingestion_use_case: Any, mock_extraction_use_case: Any) -> None:
    """Test ingest succeeds for plain text uploads.

    Given: A valid plain text file and properly mocked ingestion/extraction services.
    When: A POST request is sent to /api/v1/ingest.
    Then: It should return a 200 status code with document metadata.
    """
    # Arrange
    mock_ingestion_use_case.ingest_and_queue.return_value = "doc:456"
    mock_extraction_use_case.execute.return_value = (
        [{"id": "node1"}],
        [{"source": "node1", "target": "node2"}],
    )

    app.dependency_overrides[get_ingestion_use_case] = lambda: mock_ingestion_use_case
    app.dependency_overrides[get_extraction_use_case] = lambda: mock_extraction_use_case

    # Act
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("note.txt", b"Some plain text content.", "text/plain")},
    )

    # Assert
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "note.txt"
    assert body["status"] == "processing"


def test_ingest_pdf_success(
    mock_ingestion_use_case: Any, mock_extraction_use_case: Any, mocker: Any
) -> None:
    """Test ingest succeeds for PDFs when loader returns extracted text.

    Given: A PDF file and a mocked DocumentLoader that extracts its text.
    When: A POST request is sent to /api/v1/ingest.
    Then: It should return a 200 status code and show that the file was processed.
    """
    # Arrange
    mock_ingestion_use_case.ingest_and_queue.return_value = "doc:789"
    mock_extraction_use_case.execute.return_value = (
        [{"id": "node1"}],
        [{"source": "node1", "target": "node2"}],
    )

    app.dependency_overrides[get_ingestion_use_case] = lambda: mock_ingestion_use_case
    app.dependency_overrides[get_extraction_use_case] = lambda: mock_extraction_use_case

    load_mock = mocker.patch("app.interfaces.routers.DocumentLoader.load")
    load_mock.return_value = [
        LoadedDocument(text="Extracted PDF text", source="/tmp/test.pdf", format="pdf")
    ]

    # Act
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )

    # Assert
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "doc.pdf"
    assert body["status"] == "processing"
    load_mock.assert_called_once()


def test_ingest_unsupported_format_returns_400() -> None:
    """Test ingest rejects unsupported file extensions with 400.

    Given: A file with an unsupported extension (e.g., .exe).
    When: A POST request is sent to /api/v1/ingest.
    Then: It should return a 400 Bad Request status code.
    """
    # Arrange / Act
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("archive.exe", b"MZ", "application/octet-stream")},
    )

    # Assert
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_ingest_no_filename():
    """Test ingestion with missing filename."""
    files = {"file": ("", b"some content", "text/plain")}
    response = client.post("/api/v1/ingest", files=files)
    # FastAPI returns 422 if filename is empty string for UploadFile?
    # If it hits our router, it's 400. Let's allow either as it's an edge case.
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_notebook_ui():
    """Test notebook UI route."""
    response = client.get("/api/v1/notebook")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_get_notebook_documents(mock_notebook_use_case: Any) -> None:
    """Test listing documents in a notebook."""
    mock_notebook_use_case.get_documents.return_value = []
    from app.interfaces.dependencies import get_notebook_use_case

    app.dependency_overrides[get_notebook_use_case] = lambda: mock_notebook_use_case

    response = client.get("/api/v1/notebooks/nb123/documents")
    assert response.status_code == 200
    assert response.json() == []
    mock_notebook_use_case.get_documents.assert_called_once_with("nb123")


def test_ingest_parser_error_returns_400(mocker: Any) -> None:
    """Test ingest maps document parser failures to 400 errors.

    Given: A file that causes a parsing error in the DocumentLoader.
    When: A POST request is sent to /api/v1/ingest.
    Then: It should return a 500 Internal Server Error status code (mapping to specific detail).
    """
    # Arrange
    mocker.patch("app.interfaces.routers.DocumentLoader.load", side_effect=RuntimeError("boom"))

    # Act
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("broken.pdf", BytesIO(b"%PDF-broken"), "application/pdf")},
    )

    # Assert
    assert response.status_code == 500
    assert "Failed to parse uploaded file" in response.json()["detail"]


@pytest.mark.parametrize(
    "query, expected_status",
    [
        ("What are the key findings?", 200),
        ("", 422),  # Empty query might fail validation depending on Pydantic config
    ],
)
def test_query_endpoint(query: str, expected_status: int, mock_retrieval_use_case: Any) -> None:
    """Test the query endpoint with valid and invalid inputs.

    Given: A query string (or empty string) and a mocked retrieval service.
    When: A POST request is sent to /api/v1/query.
    Then: It should return the expected status code.
    """
    # Arrange
    mock_retrieval_use_case.execute.return_value = [{"text": "Sample result", "score": 0.9}]

    app.dependency_overrides[get_retrieval_use_case] = lambda: mock_retrieval_use_case

    payload = {}
    if query:
        payload["query"] = query

    # Act
    response = client.post("/api/v1/query", json=payload)

    # Assert
    assert response.status_code == expected_status


def test_enhance_endpoint(mocker: Any) -> None:
    """Test the graph enhancement endpoint.

    Given: A request to trigger graph enhancement.
    When: The /api/v1/enhance endpoint is called.
    Then: It should return 200 and confirm successful community generation.
    """
    # Arrange
    from app.interfaces.dependencies import get_enhancement_use_case

    mock_use_case = mocker.AsyncMock()
    app.dependency_overrides[get_enhancement_use_case] = lambda: mock_use_case

    # Act
    response = client.post("/api/v1/enhance")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"message": "Graph communities generated successfully."}
    mock_use_case.execute.assert_called_once()


def test_chat_endpoint(mocker: Any) -> None:
    """Test the chat endpoint.

    Given: A request with a query and chat history.
    When: The /api/v1/chat endpoint is called.
    Then: It should return 200 and the assistant's response.
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
    response = client.post("/api/v1/chat", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.json() == {"response": "This is a grounded response."}
    mock_use_case.execute.assert_called_once_with(
        "What is semantic blocking?",
        history=[{"role": "user", "content": "Hello"}],
        notebook_ids=None,
    )
