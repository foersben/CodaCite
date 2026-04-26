"""Application use cases for notebook management.

This module coordinates the creation, retrieval, and deletion of notebooks
and their relationships with documents.
"""

import logging
import uuid
from datetime import datetime

from app.domain.models import Notebook
from app.domain.ports import DocumentStore

logger = logging.getLogger(__name__)


class NotebookUseCase:
    """Use case for managing notebooks and document relationships."""

    def __init__(self, store: DocumentStore) -> None:
        """Initialize the notebook use case.

        Args:
            store: The document store dependency.
        """
        self.store = store

    async def create_notebook(self, title: str, description: str | None = None) -> Notebook:
        """Create a new notebook.

        Args:
            title: The name of the notebook.
            description: Optional description.

        Returns:
            The created Notebook object.
        """
        notebook = Notebook(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            created_at=datetime.utcnow().isoformat(),
        )
        await self.store.save_notebook(notebook)
        return notebook

    async def list_notebooks(self) -> list[Notebook]:
        """List all notebooks in the system.

        Returns:
            A list of all notebooks.
        """
        return await self.store.get_all_notebooks()

    async def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook.

        Args:
            notebook_id: The ID of the notebook to remove.
        """
        await self.store.delete_notebook(notebook_id)

    async def add_document(self, notebook_id: str, document_id: str) -> None:
        """Add a document to a notebook.

        Args:
            notebook_id: The target notebook ID.
            document_id: The document ID to add.
        """
        await self.store.add_document_to_notebook(document_id, notebook_id)
