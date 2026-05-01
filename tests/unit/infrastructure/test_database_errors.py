from unittest.mock import AsyncMock

import pytest

try:
    from surrealdb.errors import NotFoundError, SurrealDBError  # type: ignore
except ImportError:

    class SurrealDBError(Exception):  # type: ignore[no-redef]
        """Fallback for SurrealDBError."""

    class NotFoundError(SurrealDBError):  # type: ignore[no-redef]
        """Fallback for NotFoundError."""


from app.domain.models import Document
from app.infrastructure.database.store import SurrealDocumentStore


@pytest.fixture
def mock_db():
    """Create a mock database instance."""
    return AsyncMock()


@pytest.mark.asyncio
async def test_get_all_notebooks_table_not_found(mock_db):
    """Test behavior when the notebook table does not exist."""
    store = SurrealDocumentStore(mock_db)

    # Mock SurrealDB raising a NotFoundError (simulating 'table notebook does not exist')
    mock_db.query.side_effect = NotFoundError("The table 'notebook' does not exist")

    with pytest.raises(NotFoundError) as excinfo:
        await store.get_all_notebooks()

    assert "notebook" in str(excinfo.value)


@pytest.mark.asyncio
async def test_initialize_schema_permission_denied(mock_db):
    """Test behavior when SurrealDB has no permission to create indices/tables."""
    store = SurrealDocumentStore(mock_db)

    # Mock a generic SurrealDB error with permission message
    mock_db.query.side_effect = SurrealDBError("IO error: Permission denied (os error 13)")

    with pytest.raises(SurrealDBError) as excinfo:
        await store.initialize_schema()

    assert "Permission denied" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_all_documents_empty_result(mock_db):
    """Test that empty results are handled without error."""
    store = SurrealDocumentStore(mock_db)

    # Mock empty result envelope
    mock_db.query.return_value = [{"result": []}]

    docs = await store.get_all_documents()
    assert docs == []


@pytest.mark.asyncio
async def test_save_document_connection_error(mock_db):
    """Test behavior during a network/connection failure."""
    store = SurrealDocumentStore(mock_db)

    # Mock a connection error (e.g. Broken Pipe or Timeout)
    mock_db.query.side_effect = ConnectionError(
        "Failed to connect to SurrealDB at ws://surrealdb:8000"
    )

    doc = Document(id="doc1", filename="test.pdf")

    with pytest.raises(ConnectionError):
        await store.save_document(doc)
