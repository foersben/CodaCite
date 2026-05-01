"""Application use cases for notebook management.

This module coordinates the creation, retrieval, and deletion of notebooks
and their relationships with documents.
"""

import logging
import uuid
from datetime import UTC, datetime

from app.domain.models import Document, Notebook
from app.domain.ports import DocumentStore

logger = logging.getLogger(__name__)


class NotebookUseCase:
    """Manages the lifecycle and organization of Notebooks.

    Notebooks serve as logical containers for Documents, enabling users to
    partition their Knowledge Graph into distinct workspaces. This use case
    coordinates the creation, deletion, and cross-referencing of notebooks
    and their member documents.

    Functional Scope:
        -   **Workspace Creation**: Initializes new notebook nodes in SurrealDB.
        -   **Document Affiliation**: Manages many-to-many relationships
            between documents and notebooks.
        -   **Collection Management**: Provides listing and filtering logic
            for multi-tenant data access.
    """

    def __init__(self, store: DocumentStore) -> None:
        """Initialize the notebook use case.

        Args:
            store: The persistent store for notebook and document metadata.
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
            created_at=datetime.now(UTC).isoformat(),
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

    async def remove_document(self, notebook_id: str, document_id: str) -> None:
        """Remove a document from a notebook.

        Args:
            notebook_id: The notebook ID.
            document_id: The document ID to remove.
        """
        await self.store.remove_document_from_notebook(document_id, notebook_id)

    async def get_documents(self, notebook_id: str) -> list[Document]:
        """Get all documents in a notebook.

        Args:
            notebook_id: The notebook ID.

        Returns:
            List of documents.
        """
        return await self.store.get_notebook_documents(notebook_id)
