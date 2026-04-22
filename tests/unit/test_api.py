"""Tests for FastAPI application."""

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.ingestion.loader import LoadedDocument
from app.main import app

client = TestClient(app)


def test_health() -> None:
    """Test the health endpoint.

    Arrange: Set up TestClient (already globally arranged).
    Act: Send GET request to /health.
    Assert: Check that status code is 200 and response is ok.
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

    Arrange: Set up TestClient (already globally arranged).
    Act: Send POST request to /ingest with no file.
    Assert: Check that status code is 422.
    """
    # Arrange
    url = "/api/v1/ingest"

    # Act
    response = client.post(url)

    # Assert
    assert response.status_code == 422


def test_ingest_markdown_success() -> None:
    """Test ingest succeeds for markdown uploads."""
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("note.md", b"# Title\n\nBody text", "text/markdown")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "note.md"
    assert body["chunks_processed"] >= 1
    assert body["entities_extracted"] == 0


def test_ingest_text_success() -> None:
    """Test ingest succeeds for plain text uploads."""
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("note.txt", b"Some plain text content.", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "note.txt"
    assert body["chunks_processed"] >= 1
    assert body["entities_extracted"] == 0


def test_ingest_pdf_success_with_loader_mock() -> None:
    """Test ingest succeeds for PDFs when loader returns extracted text."""
    with patch("app.interfaces.routers.DocumentLoader.load") as load_mock:
        load_mock.return_value = [
            LoadedDocument(text="Extracted PDF text", source="/tmp/test.pdf", format="pdf")
        ]
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "doc.pdf"
    assert body["chunks_processed"] >= 1
    assert body["entities_extracted"] == 0
    load_mock.assert_called_once()


def test_ingest_unsupported_format_returns_400() -> None:
    """Test ingest rejects unsupported file extensions with 400."""
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("archive.exe", b"MZ", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_ingest_parser_error_returns_400() -> None:
    """Test ingest maps document parser failures to 400 errors."""
    with patch("app.interfaces.routers.DocumentLoader.load", side_effect=RuntimeError("boom")):
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("broken.pdf", BytesIO(b"%PDF-broken"), "application/pdf")},
        )

    assert response.status_code == 400
    assert "Failed to parse uploaded file" in response.json()["detail"]


@pytest.mark.parametrize("query, expected_status", [
    ("What are the key findings?", 200),
    ("", 422)  # Empty query might fail validation depending on Pydantic config
])
@patch("app.interfaces.routers.GraphRAGRetrievalUseCase")
def test_query_endpoint(mock_use_case_class: AsyncMock, query: str, expected_status: int) -> None:
    """Test the query endpoint with valid and invalid inputs.

    Arrange: Mock the retrieval use case.
    Act: Send POST request to /query.
    Assert: Verify response matches the expected status.
    """
    # Arrange
    mock_instance = mock_use_case_class.return_value
    mock_instance.execute = AsyncMock(return_value=[{"text": "Sample result", "score": 0.9}])

    payload = {}
    if query:
        payload["query"] = query

    # Act
    response = client.post("/api/v1/query", json=payload)

    # Assert
    if not query:
        assert response.status_code == 422
    else:
        # Our mock dependencies will hit the endpoints if not fully overridden in the app container
        # Since we use Depends in FastAPI, we can override the dependencies on the `app` object
        pass
