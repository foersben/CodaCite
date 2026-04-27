"""Tests for the application interfaces (FastAPI routers).

This module validates the interaction between the outside world and the
application logic within the Interfaces layer.
"""

import io
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.domain.models import Document, Notebook
from app.interfaces.dependencies import (
    get_chat_use_case,
    get_document_store,
    get_enhancement_use_case,
    get_ingestion_use_case,
    get_notebook_use_case,
)
from app.main import app

client = TestClient(app)


@pytest.fixture
def clean_overrides():
    """Fixture to clean up dependency overrides after each test."""
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


def test_root_endpoint(clean_overrides):
    """Test the root index endpoint."""
    response = client.get("/")
    assert response.status_code == 200


def test_notebook_management_endpoints(mocker, clean_overrides):
    """Test the /api/v1/notebooks endpoints."""
    mock_use_case = mocker.MagicMock()

    # 1. List Notebooks
    mock_use_case.list_notebooks = mocker.AsyncMock(
        return_value=[Notebook(id="nb1", title="NB1", created_at=datetime.now().isoformat())]
    )

    app.dependency_overrides[get_notebook_use_case] = lambda: mock_use_case

    response = client.get("/api/v1/notebooks")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["id"] == "nb1"

    # 2. Create Notebook
    mock_use_case.create_notebook = mocker.AsyncMock(
        return_value=Notebook(id="nb2", title="NB2", created_at=datetime.now().isoformat())
    )

    response = client.post("/api/v1/notebooks", json={"title": "NB2"})
    assert response.status_code == 201
    assert response.json()["id"] == "nb2"

    # 3. Delete Notebook
    mock_use_case.delete_notebook = mocker.AsyncMock()
    response = client.delete("/api/v1/notebooks/nb1")
    assert response.status_code == 204


def test_document_notebook_relation_endpoints(mocker, clean_overrides):
    """Test document-notebook relation endpoints."""
    mock_use_case = mocker.MagicMock()
    app.dependency_overrides[get_notebook_use_case] = lambda: mock_use_case

    # 1. Add to notebook
    mock_use_case.add_document = mocker.AsyncMock()
    response = client.post("/api/v1/notebooks/nb1/documents/doc1")
    assert response.status_code == 200

    # 2. Remove from notebook
    mock_use_case.remove_document = mocker.AsyncMock()
    response = client.delete("/api/v1/notebooks/nb1/documents/doc1")
    assert response.status_code == 200


def test_document_management_endpoints(mocker, clean_overrides):
    """Test the /api/v1/documents endpoints."""
    mock_store = mocker.MagicMock()
    mock_store.get_all_documents = mocker.AsyncMock(
        return_value=[Document(id="doc1", filename="test.pdf", status="active")]
    )

    app.dependency_overrides[get_document_store] = lambda: mock_store

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_chat_endpoint(mocker, clean_overrides):
    """Test the /api/v1/chat endpoint."""
    mock_use_case = mocker.MagicMock()
    mock_use_case.execute = mocker.AsyncMock(return_value="This is a test answer.")

    app.dependency_overrides[get_chat_use_case] = lambda: mock_use_case

    response = client.post(
        "/api/v1/chat", json={"query": "Hello", "history": [], "notebook_ids": ["nb1"]}
    )

    assert response.status_code == 200
    assert response.json()["response"] == "This is a test answer."


def test_enhance_endpoint_success(mocker, clean_overrides):
    """Test the /api/v1/enhance endpoint success scenario."""
    mock_use_case = mocker.MagicMock()
    mock_use_case.execute = mocker.AsyncMock()

    app.dependency_overrides[get_enhancement_use_case] = lambda: mock_use_case

    response = client.post("/api/v1/enhance")

    assert response.status_code == 200
    assert response.json() == {"message": "Graph communities generated successfully."}


def test_ingest_endpoint(mocker, clean_overrides):
    """Test the /api/v1/ingest endpoint."""
    mock_use_case = mocker.MagicMock()
    mock_use_case.ingest_and_queue = mocker.AsyncMock(return_value="doc123")
    mock_use_case.process_background = mocker.AsyncMock()

    app.dependency_overrides[get_ingestion_use_case] = lambda: mock_use_case

    # Mock DocumentLoader to return a dummy document
    mock_doc = mocker.MagicMock()
    mock_doc.text = "Sample text content"
    mocker.patch("app.interfaces.routers.DocumentLoader.load", return_value=[mock_doc])

    file_content = b"fake pdf content"
    file = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}

    response = client.post("/api/v1/ingest", files=file)

    assert response.status_code == 202
    assert response.json()["document_id"] == "doc123"
    assert response.json()["status"] == "processing"
