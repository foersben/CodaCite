import logging
import uuid

from app.domain.models import Chunk, Document
from app.domain.ports import CoreferenceResolver, DocumentStore, Embedder

logger = logging.getLogger(__name__)


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
            from langchain.text_splitter import (
                RecursiveCharacterTextSplitter as LegacyRecursiveCharacterTextSplitter,
            )

            splitter = LegacyRecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return list(splitter.split_text(text))
        except ImportError:
            step = max(1, chunk_size - chunk_overlap)
            return [text[i : i + chunk_size] for i in range(0, len(text), step)]


class DocumentIngestionUseCase:
    """Use case for processing a document and chunking it."""

    def __init__(
        self,
        coref_resolver: CoreferenceResolver,
        document_store: DocumentStore,
        embedder: Embedder,
    ) -> None:
        """Initialize the document ingestion use case."""
        self.coref_resolver = coref_resolver
        self.document_store = document_store
        self.embedder = embedder

    async def execute(
        self, text: str, filename: str, metadata: dict[str, str | int | float | bool] | None = None
    ) -> list[Chunk]:
        """Execute the document ingestion pipeline."""
        import time

        start_time = time.time()
        logger.info("[INGEST] Starting ingestion pipeline for: %s", filename)
        if metadata is None:
            metadata = {}

        document_id = str(uuid.uuid4())
        doc = Document(id=document_id, filename=filename, metadata=metadata)

        logger.debug("[INGEST] Document record created with ID: %s", document_id)
        await self.document_store.save_document(doc)

        # 1. Coreference Resolution
        logger.info("[INGEST] Phase 1: Coreference Resolution starting for: %s", filename)
        try:
            resolved_text = await self.coref_resolver.resolve(text)
            logger.info("[INGEST] Phase 1 complete: Coreference resolution finished")
        except Exception as e:
            logger.error("[INGEST] Phase 1 FAILED for %s: %s", filename, str(e))
            # Fallback to original text if coref fails
            resolved_text = text

        # 2. Chunking
        logger.debug("[INGEST] Phase 2: Splitting text into chunks...")
        text_chunks = chunk_text(resolved_text, chunk_size=1024, chunk_overlap=128)
        logger.info("[INGEST] Phase 2 complete: Text split into %d chunks", len(text_chunks))

        if not text_chunks:
            logger.warning("[INGEST] No chunks generated for document: %s", filename)
            return []

        # 3. Generate Embeddings (Batch)
        logger.info("[INGEST] Phase 3: Generating embeddings for %d chunks (batch mode)", len(text_chunks))
        try:
            embeddings = await self.embedder.embed_batch(text_chunks)
            logger.info("[INGEST] Phase 3 complete: All embeddings generated")
        except Exception as e:
            logger.error("[INGEST] Phase 3 FAILED: Could not generate embeddings: %s", str(e))
            raise

        # 4. Create Chunk models
        chunks = []
        for i, (chunk_text_str, embedding) in enumerate(zip(text_chunks, embeddings, strict=True)):
            chunk = Chunk(
                id=f"{document_id}_{i}",
                document_id=document_id,
                text=chunk_text_str,
                index=i,
                embedding=embedding,
            )
            chunks.append(chunk)

        # 5. Save chunks
        logger.info("[INGEST] Phase 4: Saving %d chunks to SurrealDB", len(chunks))
        await self.document_store.save_chunks(chunks)

        duration = time.time() - start_time
        logger.info(
            "[INGEST] Successfully ingested and stored document: %s (%d chunks) in %.2fs",
            filename,
            len(chunks),
            duration,
        )

        return chunks
