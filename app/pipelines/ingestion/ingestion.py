"""Use case for processing and ingesting documents into the system.

This module contains the logic for the document ingestion pipeline, including
coreference resolution, chunking, embedding generation, and persistence.
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
    """Orchestrates the 9-phase asynchronous data ingestion pipeline.

    This use case manages the transition from raw text to a hybrid Graph-Vector
    storage structure in SurrealDB.

    Pipeline Phases:
        1.  **Preprocessing**: Text normalization and cleaning (e.g., Unicode fix).
        2.  **Coreference Resolution**: Pronoun resolution (he, it) using `FastCoref`.
        3.  **Semantic Partitioning**: Splitting text into 1024-char overlapping chunks.
        4.  **Embedding Generation**: Vectorizing chunks using `BGE-M3` (1024D).
        5.  **Vector Persistence**: Committing chunks and metadata to `SurrealDB`.
        6.  **Entity Spotting (KG Stage 1)**: Zero-shot NER using `GLiNER`.
        7.  **Relationship Mapping (KG Stage 2)**: Contextual mapping using `DeepSeek-R1`.
        8.  **Entity Resolution**: Merging duplicates via `Jaro-Winkler` distance.
        9.  **Global Summarization**: Map-Reduce synthesis of the entire document.
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
        """Initialize the document ingestion use case with required infrastructure.

        Args:
            coref_resolver: Logic for resolving text pronouns.
            document_store: Storage for documents and vector chunks.
            embedder: Transformer model for text vectorization.
            chunker: Strategy for splitting documents into semantic fragments.
            graph_extraction_use_case: Specialized use case for graph building.
            graph_store: Persistent storage for entity-relationship data.
            llm_generator: Local LLM generator for global summarization.
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
        """Entry point for ingestion: saves metadata and queues background processing.

        This method is non-blocking to the user, immediately returning a
        document ID while the heavy processing happens in a background task.

        Args:
            text: Raw document text.
            filename: Original filename for display.
            file_path: Optional path to the raw file.
            notebook_id: Optional ID of the parent notebook.
            metadata: Custom key-value pairs (e.g., author, source_url).

        Returns:
            The generated unique document ID.
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
                chunk_data = await self.chunker.chunk(resolved_text)
                logger.info(
                    "[INGEST-BG] Generated %d semantic chunks for document %s",
                    len(chunk_data),
                    document_id,
                )
            except Exception as e:
                logger.error("[INGEST-BG] Semantic chunking failed: %s", str(e))
                chunk_data = []

            if not chunk_data:
                logger.error(
                    "[INGEST-BG] No chunks generated for document %s. Aborting pipeline.",
                    document_id,
                )
                await self.document_store.update_document_status(document_id, "failed")
                return

            # 3. Embeddings (Batch)
            logger.info("[INGEST-BG] Phase 3: Generating embeddings for %d chunks", len(chunk_data))
            try:
                chunk_texts = [c["text"] for c in chunk_data]
                embeddings = await self.embedder.embed_batch(chunk_texts)
                logger.info("[INGEST-BG] Successfully generated all %d embeddings", len(embeddings))
            except Exception as e:
                logger.error("[INGEST-BG] Embedding generation failed: %s", str(e))
                await self.document_store.update_document_status(document_id, "failed")
                return

            # 4. Create and Save Chunks
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

            logger.info("[INGEST-BG] Phase 4: Saving chunks...")
            await self.document_store.save_chunks(chunks)

            # 5. Graph Extraction (Delegated to GraphExtractionUseCase)
            logger.info("[INGEST-BG] Phase 5: Delegating Graph Extraction to specialized use case")
            await self.graph_extraction_use_case.execute(chunks)

            # 6. Global Document Summarization (Map-Reduce)
            logger.info("[INGEST-BG] Phase 6: Generating Global Document Summary")
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
                # Note: We don't abort here because the document is still useful for standard RAG even without the global summary.

            # 7. Mark Active
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
