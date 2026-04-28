"""Use case for processing and ingesting documents into the system.

This module contains the logic for the document ingestion pipeline, including
coreference resolution, chunking, embedding generation, and persistence.
"""

import logging
import time
import uuid

from app.domain.models import Chunk, Document, Edge, Node
from app.domain.ports import (
    CoreferenceResolver,
    DocumentStore,
    Embedder,
    EntityExtractor,
    EntityResolver,
    GraphStore,
)

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> list[str]:
    """Split text into manageable chunks for processing.

    Uses LangChain's `RecursiveCharacterTextSplitter` to maintain context
    integrity by splitting on paragraph and sentence boundaries before
    falling back to characters.

    Args:
        text: The source text to be fragmented.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        A list of text fragments.
    """
    if not text.strip():
        logger.warning("[CHUNKER] Received empty or whitespace-only text.")
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        logger.info("[CHUNKER] LangChain splitters not found, using manual fallback.")
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]


class DocumentIngestionUseCase:
    """Orchestrates the 8-phase asynchronous ingestion pipeline.

    This use case manages the transition from raw text to a structured
    Knowledge Graph and Vector Store.

    Pipeline Phases:
        1.  **Coreference Resolution**: Resolves pronouns (he, it) using `FastCoref`.
        2.  **Chunking**: Splits text into 1024-char fragments with `LangChain`.
        3.  **Embedding Generation**: Vectorizes chunks using `BGE-M3`.
        4.  **Vector Persistence**: Saves chunks and metadata to `SurrealDB`.
        5.  **KG Extraction**: Identifies entities/relations using `Gemini`.
        6.  **Entity Resolution**: Merges duplicate entities using `Jaro-Winkler`.
        7.  **Graph Persistence**: Saves the local KG subgraph to `SurrealDB`.
        8.  **Status Finalization**: Marks the document as 'active' for retrieval.
    """

    def __init__(
        self,
        coref_resolver: CoreferenceResolver,
        document_store: DocumentStore,
        embedder: Embedder,
        extractor: EntityExtractor,
        resolver: EntityResolver,
        graph_store: GraphStore,
    ) -> None:
        """Initialize the document ingestion use case with required infrastructure.

        Args:
            coref_resolver: Logic for resolving text pronouns.
            document_store: Storage for documents and vector chunks.
            embedder: Transformer model for text vectorization.
            extractor: LLM-based Knowledge Graph extractor.
            resolver: Logic for entity deduplication and merging.
            graph_store: Persistent storage for entity-relationship data.
        """
        self.coref_resolver = coref_resolver
        self.document_store = document_store
        self.embedder = embedder
        self.extractor = extractor
        self.resolver = resolver
        self.graph_store = graph_store

    async def ingest_and_queue(
        self,
        text: str,
        filename: str,
        notebook_id: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> str:
        """Entry point for ingestion: saves metadata and queues background processing.

        This method is non-blocking to the user, immediately returning a
        document ID while the heavy processing happens in a background task.

        Args:
            text: Raw document text.
            filename: Original filename for display.
            notebook_id: Optional ID of the parent notebook.
            metadata: Custom key-value pairs (e.g., author, source_url).

        Returns:
            The generated unique document ID.
        """
        document_id = str(uuid.uuid4())
        doc = Document(
            id=document_id, filename=filename, status="processing", metadata=metadata or {}
        )

        logger.info("[INGEST] Queuing document: %s (ID: %s)", filename, document_id)
        await self.document_store.save_document(doc)

        if notebook_id:
            logger.info("[INGEST] Relating document %s to notebook %s", document_id, notebook_id)
            await self.document_store.add_document_to_notebook(document_id, notebook_id)

        return document_id

    async def process_background(self, document_id: str, text: str, filename: str) -> None:
        """Complete the ingestion pipeline in the background.

        Args:
            document_id: ID of the previously saved document record.
            text: The text content to process.
            filename: Original filename for logging.
        """
        try:
            start_time = time.time()
            logger.info("[INGEST-BG] Starting background processing for: %s", document_id)

            # 1. Coreference Resolution
            logger.info("[INGEST-BG] Phase 1: Coreference Resolution starting")
            try:
                resolved_text = await self.coref_resolver.resolve(text)
            except Exception as e:
                logger.error("[INGEST-BG] Coref failed, using original: %s", str(e))
                resolved_text = text

            # 2. Chunking
            logger.info("[INGEST-BG] Phase 2: Chunking document %s", document_id)
            logger.debug("[INGEST-BG] Input text length: %d chars", len(resolved_text))
            if resolved_text:
                logger.debug(
                    "[INGEST-BG] Text snippet: %s...", resolved_text[:200].replace("\n", " ")
                )
            else:
                logger.warning("[INGEST-BG] Input text for document %s is EMPTY!", document_id)

            try:
                text_chunks = chunk_text(resolved_text, chunk_size=1024, chunk_overlap=128)
                logger.info(
                    "[INGEST-BG] Generated %d chunks for document %s", len(text_chunks), document_id
                )
            except Exception as e:
                logger.error("[INGEST-BG] Chunking failed: %s", str(e))
                text_chunks = []

            if not text_chunks:
                logger.error(
                    "[INGEST-BG] No chunks generated for document %s. Aborting pipeline.",
                    document_id,
                )
                await self.document_store.update_document_status(document_id, "failed")
                return

            # 3. Embeddings (Batch)
            logger.info(
                "[INGEST-BG] Phase 3: Generating embeddings for %d chunks", len(text_chunks)
            )
            embeddings = await self.embedder.embed_batch(text_chunks)

            # 4. Create and Save Chunks
            chunks = []
            for i, (ct, emb) in enumerate(zip(text_chunks, embeddings, strict=True)):
                chunks.append(
                    Chunk(
                        id=f"{document_id}_{i}",
                        document_id=document_id,
                        text=ct,
                        index=i,
                        embedding=emb,
                    )
                )

            logger.info("[INGEST-BG] Phase 4: Saving chunks...")
            await self.document_store.save_chunks(chunks)

            # 5. Graph Extraction (GraphRAG)
            logger.info("[INGEST-BG] Phase 5: Knowledge Graph Extraction...")
            all_nodes: list[Node] = []
            all_edges: list[Edge] = []

            for chunk in chunks:
                nodes, edges = await self.extractor.extract(chunk.text)
                # Tag with source chunk
                for n in nodes:
                    if chunk.id not in n.source_chunk_ids:
                        n.source_chunk_ids.append(chunk.id)
                for edge in edges:
                    if chunk.id not in edge.source_chunk_ids:
                        edge.source_chunk_ids.append(chunk.id)
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            # 6. Entity Resolution
            logger.info("[INGEST-BG] Phase 6: Resolving %d entities...", len(all_nodes))
            existing_nodes = await self.graph_store.get_all_nodes()
            resolved_nodes = await self.resolver.resolve_entities(all_nodes, existing_nodes)

            # Deduplicate by ID
            unique_nodes_dict: dict[str, Node] = {}
            for n in resolved_nodes:
                if n.id not in unique_nodes_dict:
                    unique_nodes_dict[n.id] = n
                else:
                    unique_nodes_dict[n.id].source_chunk_ids.extend(n.source_chunk_ids)

            for n in unique_nodes_dict.values():
                n.source_chunk_ids = list(set(n.source_chunk_ids))
                # Generate description embedding
                text_to_embed = n.description if n.description else n.name
                n.description_embedding = await self.embedder.embed(text_to_embed)

            # 7. Final Persistence
            logger.info("[INGEST-BG] Saving graph data...")
            await self.graph_store.save_nodes(list(unique_nodes_dict.values()))
            await self.graph_store.save_edges(all_edges)

            # 8. Mark Active
            await self.document_store.update_document_status(document_id, "active")
            duration = time.time() - start_time
            logger.info(
                "[INGEST-BG] SUCCESS: Document %s has been ingested. Chunks and Knowledge Graph generated. Total time: %.2fs",
                filename,
                duration,
            )

        except Exception as e:
            logger.error("[INGEST-BG] CRITICAL FAILURE: %s", str(e), exc_info=True)
            await self.document_store.update_document_status(document_id, "failed")

    async def execute(
        self, text: str, filename: str, metadata: dict[str, str | int | float | bool] | None = None
    ) -> list[Chunk]:
        """Legacy synchronous execute method (deprecated)."""
        document_id = await self.ingest_and_queue(text, filename, metadata=metadata)

        # Re-implementing the core logic here to return chunks for legacy compatibility
        try:
            resolved_text = await self.coref_resolver.resolve(text)
        except Exception as e:
            logger.error("[INGEST] Coref failed, using original: %s", str(e))
            resolved_text = text

        text_chunks = chunk_text(resolved_text)
        if not text_chunks:
            return []

        embeddings = await self.embedder.embed_batch(text_chunks)
        chunks = []
        for i, (ct, emb) in enumerate(zip(text_chunks, embeddings, strict=True)):
            chunks.append(
                Chunk(
                    id=f"{document_id}_{i}",
                    document_id=document_id,
                    text=ct,
                    index=i,
                    embedding=emb,
                )
            )
        await self.document_store.save_chunks(chunks)

        return chunks
