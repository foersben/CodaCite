"""Unit tests for the NotebookUseCase.

Validates notebook management and document relationship coordination.
"""

from unittest.mock import AsyncMock

import pytest

from app.application.notebook import NotebookUseCase
from app.domain.models import Notebook


@pytest.fixture
def mock_doc_store():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock()


@pytest.fixture
def notebook_use_case(mock_doc_store):
    """Docstring generated to satisfy ruff D103."""
    return NotebookUseCase(store=mock_doc_store)


@pytest.mark.asyncio
async def test_create_notebook(notebook_use_case, mock_doc_store):
    """Test creating a new notebook."""
    title = "Research Lab"
    description = "Academic notes"

    notebook = await notebook_use_case.create_notebook(title, description)

    assert notebook.title == title
    assert notebook.description == description
    mock_doc_store.save_notebook.assert_called_once()


@pytest.mark.asyncio
async def test_list_notebooks(notebook_use_case, mock_doc_store):
    """Test listing all notebooks."""
    mock_doc_store.get_all_notebooks.return_value = [
        Notebook(id="1", title="N1", created_at="2024-01-01")
    ]

    notebooks = await notebook_use_case.list_notebooks()

    assert len(notebooks) == 1
    assert notebooks[0].title == "N1"
    mock_doc_store.get_all_notebooks.assert_called_once()


@pytest.mark.asyncio
async def test_manage_notebook_sources(notebook_use_case, mock_doc_store):
    """Test adding and removing documents from a notebook."""
    nb_id = "nb:123"
    doc_id = "doc:456"

    await notebook_use_case.add_document(nb_id, doc_id)
    mock_doc_store.add_document_to_notebook.assert_called_once_with(doc_id, nb_id)

    await notebook_use_case.remove_document(nb_id, doc_id)
    mock_doc_store.remove_document_from_notebook.assert_called_once_with(doc_id, nb_id)


@pytest.mark.asyncio
async def test_delete_notebook(notebook_use_case, mock_doc_store):
    """Test deleting a notebook."""
    nb_id = "nb:123"
    await notebook_use_case.delete_notebook(nb_id)
    mock_doc_store.delete_notebook.assert_called_once_with(nb_id)
