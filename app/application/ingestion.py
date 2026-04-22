"""Use Case for Document Ingestion."""

import uuid

from app.domain.models import Chunk, Document
from app.domain.ports import CoreferenceResolver, DocumentStore


# Basic text chunker. You might want to use langchain's RecursiveCharacterTextSplitter.
def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> list[str]:
    """Split text into chunks.
    
    Args:
        text: The input text to split.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The number of characters to overlap between chunks. 

    Returns:
        A list of text chunks.

    Raises:
        ImportError: If langchain is not installed.
    """
    if not text:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return list(splitter.split_text(text))
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return list(splitter.split_text(text))
        except ImportError:
            step = max(1, chunk_size - chunk_overlap)
            return [text[i : i + chunk_size] for i in range(0, len(text), step)]


class DocumentIngestionUseCase:
    """Use case for processing a document and chunking it."""

    def __init__(self, coref_resolver: CoreferenceResolver, document_store: DocumentStore) -> None:
        """Initialize the document ingestion use case."""
        self.coref_resolver = coref_resolver
        self.document_store = document_store

    async def execute(
        self, text: str, filename: str, metadata: dict[str, str | int | float | bool] | None = None
    ) -> list[Chunk]:
        """Execute the document ingestion pipeline."""
        if metadata is None:
            metadata = {}

        document_id = str(uuid.uuid4())
        doc = Document(id=document_id, filename=filename, metadata=metadata)

        await self.document_store.save_document(doc)

        # 1. Coreference Resolution (must happen before chunking!)
        resolved_text = await self.coref_resolver.resolve(text)

        # 2. Chunking
        text_chunks = chunk_text(resolved_text, chunk_size=1024, chunk_overlap=128)

        # 3. Create Chunk models
        chunks = []
        for i, chunk_text_str in enumerate(text_chunks):
            chunk = Chunk(
                id=f"{document_id}_{i}", document_id=document_id, text=chunk_text_str, index=i, embedding=None
            )
            chunks.append(chunk)

        # 4. Save chunks
        await self.document_store.save_chunks(chunks)

        return chunks
