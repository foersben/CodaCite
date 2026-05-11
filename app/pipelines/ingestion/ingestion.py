"""Main use case for high-fidelity document ingestion and knowledge synthesis.

This module serves as the central orchestrator for the ingestion slice,
transforming raw unstructured text into a structured, queryable hybrid
knowledge base containing both vector embeddings and a semantic graph.
"""

import logging
import time
import uuid

from app.core.interfaces import (
    Chunker,
    CoreferenceResolver,
    DocumentStore,
    Embedder,
    GraphStore,
    LLMGenerator,
)
from app.models.models import Chunk, Document
from app.pipelines.extraction.extraction_logic import GraphExtractionUseCase
from app.pipelines.ingestion.summarization import DocumentSummarizer

logger = logging.getLogger(__name__)


class DocumentIngestionUseCase:
    """Orchestrator for the asynchronous 9-phase knowledge ingestion pipeline.

    This class manages the lifecycle of a document from the moment it is uploaded
    until it is fully indexed in SurrealDB. It balances low-latency user
    responsiveness with computationally expensive enrichment tasks.
    The pipeline is divided into two operational stages:
    **Stage A: Synchronous Staging (ingest_and_queue)**
    Immediately persists metadata and returns a tracking ID to the caller,
    minimizing UI blocking.
    **Stage B: Asynchronous Synthesis (process_background)**
    Executes a multi-phase enrichment sequence:
        1.  **Normalization**: Deterministic text cleaning (Unicode, whitespace). [PHASE 1]
        2.  **Coreference Resolution**: Anaphora resolution using fastcoref. [PHASE 2]
        3.  **Structural Partitioning**: Chunking with sliding windows. [PHASE 3]
        4.  **Vectorization**: Batch embedding generation via BGE-M3. [PHASE 4]
        5.  **Persistence**: Atomic commits of chunks to SurrealDB. [PHASE 5]
        6.  **Graph Extraction**: NER and Relationship extraction (delegated). [PHASE 6]
        7.  **Entity Resolution**: Node merging in the global KG (delegated). [PHASE 7]
        8.  **Map-Reduce Summarization**: Global document synthesis. [PHASE 8]
        9.  **Finalization**: Mark document as 'active' for retrieval. [PHASE 9]
    """

    def __init__(
        self,
        coref_resolver: CoreferenceResolver,
        document_store: DocumentStore,
        embedder: Embedder,
        chunker: Chunker,
        graph_extraction_use_case: GraphExtractionUseCase,
        graph_store: GraphStore,
        llm_generator: LLMGenerator,
    ) -> None:
        """Initialize the ingestion engine with specialized component ports.

        Args:
            coref_resolver: Interface for resolving entity pronouns (Phase 2).
            document_store: Repository for document and chunk persistence (Phase 5).
            embedder: Implementation for vectorizing text fragments (Phase 4).
            chunker: Strategy for structural document decomposition (Phase 3).
            graph_extraction_use_case: Orchestrator for KG construction (Phase 6-7).
            graph_store: Repository for graph nodes and edges.
            llm_generator: LLM interface for summarization and reasoning (Phase 8).
        """
        self.coref_resolver = coref_resolver
        self.document_store = document_store
        self.embedder = embedder
        self.chunker = chunker
        self.graph_extraction_use_case = graph_extraction_use_case
        self.graph_store = graph_store
        self.llm_generator = llm_generator

    async def ingest_and_queue(
        self,
        text: str,
        filename: str,
        file_path: str | None = None,
        notebook_id: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> str:
        """Staging entry point: initializes document metadata and queues tasks.

        This method corresponds to the transition from Phase 1 (Normalization)
        to the background processing stages.

        Args:
            text: The raw extracted text from the document loader.
            filename: The display name of the document.
            file_path: Absolute path to the source file (if available).
            notebook_id: The UUID of the parent notebook container.
            metadata: Extensible dictionary of domain-specific attributes.

        Returns:
            The unique identifier (UUID string) for the new document.
        """
        document_id = str(uuid.uuid4())
        doc = Document(
            id=document_id,
            filename=filename,
            status="processing",
            file_path=file_path,
            metadata=metadata or {},
        )
        logger.info("[INGEST] Queuing document: %s (ID: %s)", filename, document_id)
        await self.document_store.save_document(doc)
        if notebook_id:
            logger.info("[INGEST] Relating document %s to notebook %s", document_id, notebook_id)
            await self.document_store.add_document_to_notebook(document_id, notebook_id)
        return document_id

    async def process_background(self, document_id: str, text: str, filename: str) -> None:
        """Executes the computationally intensive synthesis pipeline (Phases 2-9).

        This method orchestrates the full suite of NLP and KG enrichment
        operations. It is designed to be resilient; failures in non-critical
        stages (like summarization) are logged but do not abort the core
        vectorization and graph extraction tasks.

        Workflow:
            - **Phase 2**: Coreference resolution for context enrichment.
            - **Phase 3**: Structural chunking with context injection.
            - **Phase 4**: Batch vector embedding.
            - **Phase 5**: Persistence of chunks to SurrealDB.
            - **Phase 6-7**: Relational extraction and entity resolution.
            - **Phase 8**: Map-Reduce global summarization.
            - **Phase 9**: Finalization (Active state).

        Args:
            document_id: The UUID of the document record to update.
            text: The cleaned, normalized text content (Result of Phase 1).
            filename: Original filename for logging and context prefixing.
        """
        try:
            start_time = time.time()
            logger.info("[INGEST-BG] Starting background processing for: %s", document_id)

            # PHASE 2: Coreference Resolution
            logger.info("[INGEST-BG] Phase 2: Coreference Resolution starting")
            try:
                resolved_text = await self.coref_resolver.resolve(text)
            except Exception as e:
                logger.error("[INGEST-BG] Coref failed, using original: %s", str(e))
                resolved_text = text

            # PHASE 3: Structural Chunking
            logger.info("[INGEST-BG] Phase 3: Chunking document %s", document_id)
            try:
                context_prefix = f"Document: {filename}\n"
                chunk_data = await self.chunker.chunk(resolved_text, context_prefix=context_prefix)
                logger.info(
                    "[INGEST-BG] Generated %d structural chunks for document %s",
                    len(chunk_data),
                    document_id,
                )
            except Exception as e:
                logger.error("[INGEST-BG] Structural chunking failed: %s", str(e))
                chunk_data = []

            if not chunk_data:
                logger.error(
                    "[INGEST-BG] No chunks generated for document %s. Aborting pipeline.",
                    document_id,
                )
                await self.document_store.update_document_status(document_id, "failed")
                return

            # PHASE 4: Vectorization (Embeddings)
            logger.info("[INGEST-BG] Phase 4: Generating embeddings for %d chunks", len(chunk_data))
            try:
                chunk_texts = [c["text"] for c in chunk_data]
                embeddings = await self.embedder.embed_batch(chunk_texts)
                logger.info("[INGEST-BG] Successfully generated all %d embeddings", len(embeddings))
            except Exception as e:
                logger.error("[INGEST-BG] Embedding generation failed: %s", str(e))
                await self.document_store.update_document_status(document_id, "failed")
                return

            # PHASE 5: Persistence (Save Chunks)
            chunks = []
            for i, (c_meta, emb) in enumerate(zip(chunk_data, embeddings, strict=True)):
                chunks.append(
                    Chunk(
                        id=f"{document_id}_{i}",
                        document_id=document_id,
                        text=c_meta["text"],
                        index=i,
                        start_char=c_meta["start_char"],
                        end_char=c_meta["end_char"],
                        embedding=emb,
                    )
                )
            logger.info("[INGEST-BG] Phase 5: Saving chunks...")
            await self.document_store.save_chunks(chunks)

            # PHASE 6-7: Graph Extraction & Resolution
            logger.info("[INGEST-BG] Phase 6-7: Graph Extraction and Entity Resolution")
            await self.graph_extraction_use_case.execute(chunks)

            # PHASE 8: Global Document Summarization (Map-Reduce)
            logger.info("[INGEST-BG] Phase 8: Generating Global Document Summary")
            try:
                summarizer = DocumentSummarizer(llm_generator=self.llm_generator)
                document_summary = await summarizer.generate_global_summary(chunk_texts)
                # Save the summary to SurrealDB
                await self.document_store.save_document_with_summary(
                    document_id=document_id, summary=document_summary
                )
                logger.info("[INGEST-BG] Successfully generated and saved global summary.")
            except Exception as e:
                logger.error(
                    "[INGEST-BG] Summarization failed, continuing without summary: %s", str(e)
                )

            # PHASE 9: Finalization (Mark Active)
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
