"""Use case for performing document ingestion.

This module orchestrates the end-to-end ingestion pipeline, coordinating
preprocessing, chunking, embedding, and knowledge graph extraction.
"""

import logging
import uuid

from app.domain.models import Chunk, Document
from app.domain.ports import (
    CoreferenceResolver,
    DocumentStore,
    Embedder,
    EntityExtractor,
    EntityResolver,
    GraphStore,
)
from app.ingestion.loader import DocumentLoader
from app.ingestion.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


def recursive_character_chunking(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 100
) -> list[str]:
    """Split text into chunks using LangChain's recursive character splitter.

    This strategy attempts to split on paragraphs, then sentences, then words,
    to keep semantic units together.

    Args:
        text: The resolved text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        logger.info("[CHUNKER] LangChain splitters not found, using manual fallback.")
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]


class DocumentIngestionUseCase:
    """Orchestrates the 8-phase asynchronous ingestion pipeline.

    Pipeline Logic:
        The ingestion process is a **Semantic Decomposition Pipeline**. It
        takes high-entropy raw data and progressively lowers its entropy through
        normalization, coreference resolution, and structured extraction.

        Data Flow & Transformations (8-Phase Lifecycle):
        1. **Phase 1: Loading & Preprocessing**: Bytes -> `LoadedDocument` -> Cleaned `str`.
        2. **Phase 2: Coreference Resolution**: Resolves pronouns (e.g., "he" -> "Einstein").
        3. **Phase 3: Recursive Chunking**: Splits text into overlapping semantic fragments.
        4. **Phase 4: Document Persistence**: Saves raw chunks and notebook relations to SurrealDB.
        5. **Phase 5: Vectorization (Embedding)**: Generates 1024D vectors via BGE-M3.
        6. **Phase 6: Knowledge Extraction**: Discovery of Nodes/Edges via Gemini/GLiNER.
        7. **Phase 7: Entity Resolution**: Deduplication and global graph merging.
        8. **Phase 8: Finalization**: Status updates and vector index maintenance.

        *Note: Tokenization occurs internally during Phases 5 and 6 using model-specific
        SentencePiece/WordPiece tokenizers.*
    """

    def __init__(
        self,
        coref_resolver: CoreferenceResolver,
        document_store: DocumentStore,
        embedder: Embedder,
        extractor: EntityExtractor,
        entity_resolver: EntityResolver,
        graph_store: GraphStore,
        loader: DocumentLoader,
        preprocessor: TextPreprocessor,
    ) -> None:
        """Initialize the ingestion use case with core services.

        Args:
            coref_resolver: Logic for resolving linguistic references.
            document_store: Persistent storage for chunks and metadata.
            embedder: Transformer model for text vectorization.
            extractor: Logic for structured KG extraction.
            entity_resolver: Logic for entity deduplication.
            graph_store: Persistent storage for knowledge graphs.
            loader: Logic for parsing file bytes.
            preprocessor: Logic for text cleaning and normalization.
        """
        self.coref_resolver = coref_resolver
        self.document_store = document_store
        self.embedder = embedder
        self.extractor = extractor
        self.entity_resolver = entity_resolver
        self.graph_store = graph_store
        self.loader = loader
        self.preprocessor = preprocessor

    async def execute(self, file_content: bytes, filename: str, notebook_ids: list[str]) -> str:
        """Execute the end-to-end ingestion pipeline.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: Name of the file.
            notebook_ids: Notebooks to associate this document with.

        Returns:
            The unique ID of the ingested document.
        """
        document_id = str(uuid.uuid4())
        logger.info("[INGEST] Starting ingestion for document: %s (ID: %s)", filename, document_id)

        try:
            # Phase 1: Load and Preprocess
            loaded_doc = await self.loader.load(file_content, filename)
            cleaned_text = self.preprocessor.preprocess(loaded_doc.content)

            # Phase 2: Coreference Resolution
            resolved_text = await self.coref_resolver.resolve(cleaned_text)

            # Phase 3: Chunking
            text_chunks = recursive_character_chunking(resolved_text)

            # Phase 4: Persistence
            # Initial document record
            doc_model = Document(
                id=document_id,
                filename=filename,
                status="processing",
                metadata={"file_path": filename},
            )
            await self.document_store.save_document(doc_model)

            # Associate with notebooks
            for nb_id in notebook_ids:
                await self.document_store.add_document_to_notebook(document_id, nb_id)

            # Phase 5: Embedding
            embeddings = await self.embedder.embed_batch(text_chunks)

            # Phase 6: Extraction
            # Create chunk models and extract KG fragments
            chunks = []
            all_extracted_nodes = []
            all_extracted_edges = []

            for i, (text, vector) in enumerate(zip(text_chunks, embeddings)):
                chunk_id = f"{document_id}_{i}"
                chunk = Chunk(
                    id=chunk_id, document_id=document_id, text=text, index=i, embedding=vector
                )
                chunks.append(chunk)

                # Extract KG fragments from this chunk
                nodes, edges = await self.extractor.extract(text)
                # Link entities back to this source chunk
                for node in nodes:
                    node.source_chunk_ids = [chunk_id]
                for edge in edges:
                    edge.source_chunk_ids = [chunk_id]

                all_extracted_nodes.extend(nodes)
                all_extracted_edges.extend(edges)

            await self.document_store.save_chunks(chunks)

            # Phase 7: Entity Resolution
            existing_nodes = await self.graph_store.get_all_nodes()
            resolved_nodes = await self.entity_resolver.resolve_entities(
                all_extracted_nodes, existing_nodes
            )

            # Phase 8: Final Persistence & Finalization
            await self.graph_store.save_nodes(resolved_nodes)
            await self.graph_store.save_edges(all_extracted_edges)

            # Phase 8: Status Update
            await self.document_store.update_document_status(document_id, "active")
            logger.info("[INGEST] Ingestion completed for document: %s", filename)

            return document_id

        except Exception as e:
            logger.error("[INGEST] Ingestion failed for document %s: %s", filename, str(e))
            await self.document_store.update_document_status(document_id, "failed")
            raise e
