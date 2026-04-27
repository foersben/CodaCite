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
    """Split text into manageable chunks for processing."""
    if not text:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return list(splitter.split_text(text))
    except ImportError:
        logger.warning("langchain-text-splitters not found, using manual fallback")
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]


class DocumentIngestionUseCase:
    """Use case for processing a document and chunking it.

    Coordinates the multi-stage ingestion pipeline: resolution -> chunking -> embedding -> storage.
    Now supports background processing and knowledge graph extraction.
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
        """Initialize the document ingestion use case.

        Args:
            coref_resolver: Implementation of the CoreferenceResolver port.
            document_store: Implementation of the DocumentStore port.
            embedder: Implementation of the Embedder port.
            extractor: Implementation of the EntityExtractor port.
            resolver: Implementation of the EntityResolver port.
            graph_store: Implementation of the GraphStore port.
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
        """Prepare document for background processing.

        Args:
            text: Raw document text.
            filename: Original filename.
            notebook_id: Optional ID of the notebook to attach to.
            metadata: Optional additional metadata.

        Returns:
            The generated document_id.
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

    async def process_background(self, document_id: str, text: str) -> None:
        """Complete the ingestion pipeline in the background.

        Args:
            document_id: ID of the previously saved document record.
            text: The text content to process.
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
            logger.debug("[INGEST-BG] Phase 2: Chunking...")
            text_chunks = chunk_text(resolved_text, chunk_size=1024, chunk_overlap=128)
            if not text_chunks:
                logger.warning("[INGEST-BG] No chunks generated")
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
            logger.info("[INGEST-BG] SUCCESS: Processing complete in %.2fs", duration)

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
