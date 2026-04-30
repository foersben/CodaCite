"""Unit tests for the application interfaces (FastAPI routers).

Validates the interaction between the outside world and the
application logic, ensuring API endpoints behave as expected.
"""

import io
from collections.abc import Generator
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response

from app.domain.models import Document, Notebook
from app.interfaces.dependencies import (
    get_chat_use_case,
    get_document_store,
    get_enhancement_use_case,
    get_ingestion_use_case,
    get_notebook_use_case,
    get_retrieval_use_case,
)
from app.main import app


@pytest.fixture
def client(mocker: Any) -> Generator[TestClient, None, None]:
    """Provides a TestClient that triggers lifespan events.

    Args:
        mocker: Mock fixture.
    """
    mocker.patch("app.main.init_db", new_callable=mocker.AsyncMock)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def clean_overrides(client: TestClient) -> Generator[None, None, None]:
    """Fixture to clean up dependency overrides after each test.

    Args:
        client: Test client.
    """
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


def test_root_endpoint(mocker: Any, clean_overrides: None, client: TestClient) -> None:
    """Tests the root index endpoint returns a success status.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mocker.patch(
        "app.main.templates.TemplateResponse", return_value=Response(content="ok", status_code=200)
    )
    response = client.get("/")
    assert response.status_code == 200


def test_notebook_management_endpoints(
    mocker: Any, clean_overrides: None, client: TestClient
) -> None:
    """Tests the notebook management API endpoints.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
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


def test_document_notebook_relation_endpoints(
    mocker: Any, clean_overrides: None, client: TestClient
) -> None:
    """Tests document-notebook relation API endpoints.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
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


def test_document_management_endpoints(
    mocker: Any, clean_overrides: None, client: TestClient
) -> None:
    """Tests the document management API endpoints.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mock_store = mocker.MagicMock()
    mock_store.get_all_documents = mocker.AsyncMock(
        return_value=[Document(id="doc1", filename="test.pdf", status="active")]
    )

    app.dependency_overrides[get_document_store] = lambda: mock_store

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_query_endpoint(mocker: Any, clean_overrides: None, client: TestClient) -> None:
    """Tests the semantic search query endpoint.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mock_retrieval = mocker.MagicMock()
    mock_retrieval.execute = mocker.AsyncMock(return_value=[{"text": "Result"}])
    app.dependency_overrides[get_retrieval_use_case] = lambda: mock_retrieval

    response = client.post("/api/v1/query", json={"query": "test query", "top_k": 5})
    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "Result"


def test_chat_endpoint(mocker: Any, clean_overrides: None, client: TestClient) -> None:
    """Tests the chat API endpoint.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mock_use_case = mocker.MagicMock()
    mock_use_case.execute = mocker.AsyncMock(return_value="This is a test answer.")

    app.dependency_overrides[get_chat_use_case] = lambda: mock_use_case

    response = client.post(
        "/api/v1/chat", json={"query": "Hello", "history": [], "notebook_ids": ["nb1"]}
    )

    assert response.status_code == 200
    assert response.json()["response"] == "This is a test answer."


def test_enhance_endpoint_success(mocker: Any, clean_overrides: None, client: TestClient) -> None:
    """Tests the graph enhancement API endpoint success scenario.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mock_use_case = mocker.MagicMock()
    mock_use_case.execute = mocker.AsyncMock()

    app.dependency_overrides[get_enhancement_use_case] = lambda: mock_use_case

    response = client.post("/api/v1/enhance")

    assert response.status_code == 200
    assert response.json() == {"message": "Graph communities generated successfully."}


def test_ingest_endpoint(mocker: Any, clean_overrides: None, client: TestClient) -> None:
    """Tests the document ingestion API endpoint.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
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


def test_ingest_no_filename(clean_overrides: None, client: TestClient) -> None:
    """Tests ingestion rejection when filename is missing.

    Args:
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    file = {"file": ("", io.BytesIO(b"data"))}
    response = client.post("/api/v1/ingest", files=file)
    assert response.status_code == 400


def test_ingest_unsupported_format(mocker: Any, clean_overrides: None, client: TestClient) -> None:
    """Tests ingestion rejection for unsupported formats.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mocker.patch(
        "app.interfaces.routers.DocumentLoader.load", side_effect=ValueError("Format error")
    )
    file = {"file": ("test.exe", io.BytesIO(b"data"), "application/octet-stream")}
    response = client.post("/api/v1/ingest", files=file)
    assert response.status_code == 400


def test_ingest_unexpected_error(mocker: Any, clean_overrides: None, client: TestClient) -> None:
    """Tests ingestion rejection for unexpected internal errors.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mocker.patch("app.interfaces.routers.DocumentLoader.load", side_effect=Exception("Disk full"))
    file = {"file": ("test.txt", io.BytesIO(b"data"), "text/plain")}
    response = client.post("/api/v1/ingest", files=file)
    assert response.status_code == 500


def test_get_document_status_endpoints(
    mocker: Any, clean_overrides: None, client: TestClient
) -> None:
    """Tests the document status retrieval endpoint.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mock_store = mocker.MagicMock()
    app.dependency_overrides[get_document_store] = lambda: mock_store

    # 1. Success
    mock_store.get_document = mocker.AsyncMock(
        return_value=Document(id="doc1", filename="f.pdf", status="active")
    )
    response = client.get("/api/v1/documents/doc1/status")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

    # 2. Not found
    mock_store.get_document = mocker.AsyncMock(return_value=None)
    response = client.get("/api/v1/documents/missing/status")
    assert response.status_code == 404


def test_health_check(client: TestClient) -> None:
    """Tests the health check endpoint.

    Args:
        client: Test client.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_notebook_ui(mocker: Any, client: TestClient) -> None:
    """Tests that the notebook UI template is served.

    Args:
        mocker: Mock fixture.
        client: Test client.
    """
    # Mock templates to avoid looking for actual file
    mocker.patch(
        "app.interfaces.routers.templates.TemplateResponse",
        return_value=Response(content="ok", status_code=200),
    )
    response = client.get("/api/v1/notebook")
    assert response.status_code == 200


def test_get_notebook_documents_endpoint(
    mocker: Any, clean_overrides: None, client: TestClient
) -> None:
    """Tests retrieval of documents for a specific notebook.

    Args:
        mocker: Mock fixture.
        clean_overrides: Clean overrides fixture.
        client: Test client.
    """
    mock_use_case = mocker.MagicMock()
    mock_use_case.get_documents = mocker.AsyncMock(return_value=[])
    app.dependency_overrides[get_notebook_use_case] = lambda: mock_use_case

    response = client.get("/api/v1/notebooks/nb1/documents")
    assert response.status_code == 200
    assert response.json() == []
