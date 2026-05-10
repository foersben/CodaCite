"""Unit tests for the NotebookUseCase.

Validates notebook lifecycle management and document association logic.
"""

from typing import Any

import pytest

from app.application.notebook import NotebookUseCase
from app.domain.models import Notebook


@pytest.fixture
def use_case(mock_document_store: Any) -> NotebookUseCase:
    """Provides a NotebookUseCase instance.

    Args:
        mock_document_store: Global mock DocumentStore fixture.

    Returns:
        A NotebookUseCase instance.
    """
    return NotebookUseCase(store=mock_document_store)


@pytest.mark.asyncio
async def test_create_notebook(
    use_case: NotebookUseCase,
    mock_document_store: Any,
) -> None:
    """Tests notebook creation."""
    # Act
    notebook = await use_case.create_notebook(title="Test Notebook", description="Test Desc")

    # Assert
    assert isinstance(notebook, Notebook)
    assert notebook.title == "Test Notebook"
    mock_document_store.save_notebook.assert_called_once_with(notebook)


@pytest.mark.asyncio
async def test_list_notebooks(
    use_case: NotebookUseCase,
    mock_document_store: Any,
) -> None:
    """Tests listing notebooks."""
    # Arrange
    expected = [Notebook(id="1", title="N1")]
    mock_document_store.get_all_notebooks.return_value = expected

    # Act
    results = await use_case.list_notebooks()

    # Assert
    assert results == expected


@pytest.mark.asyncio
async def test_delete_notebook(
    use_case: NotebookUseCase,
    mock_document_store: Any,
) -> None:
    """Tests notebook deletion."""
    # Act
    await use_case.delete_notebook("notebook:123")

    # Assert
    mock_document_store.delete_notebook.assert_called_once_with("notebook:123")


@pytest.mark.asyncio
async def test_add_document_to_notebook(
    use_case: NotebookUseCase,
    mock_document_store: Any,
) -> None:
    """Tests adding a document to a notebook."""
    # Act
    await use_case.add_document("notebook:1", "doc:1")

    # Assert
    mock_document_store.add_document_to_notebook.assert_called_once_with("doc:1", "notebook:1")
