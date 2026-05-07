"""FastAPI dependencies for the application.

This module provides dependency injection providers for use cases, infrastructure
implementations, and database connections.
"""

import threading
from pathlib import Path

from fastapi import Depends
from surrealdb import AsyncSurreal

from app.core.config import settings
from app.core.interfaces import (
    CoreferenceResolver,
    DocumentStore,
    Embedder,
    EntityExtractor,
    EntityResolver,
    GraphStore,
    LLMGenerator,
    Reranker,
)
from app.db.store import SurrealDocumentStore, SurrealGraphStore
from app.pipelines.extraction.enhancement import GraphEnhancementUseCase
from app.pipelines.extraction.extraction_logic import GraphExtractionUseCase
from app.pipelines.extraction.gliner_extractor import GeminiEntityExtractor, GLiNERFallbackExtractor
from app.pipelines.extraction.linker import SimpleEntityLinker
from app.pipelines.extraction.resolution import JaroWinklerResolver
from app.pipelines.generation.chat import ChatUseCase
from app.pipelines.generation.generator import GeminiGenerator
from app.pipelines.generation.local_generator import LocalLlamaGenerator
from app.pipelines.generation.vlm import LocalVLM
from app.pipelines.ingestion.coreference import FastCorefResolver
from app.pipelines.ingestion.ingestion import DocumentIngestionUseCase
from app.pipelines.notebooks.notebook_manager import NotebookUseCase
from app.pipelines.retrieval.embeddings import SentenceTransformerEmbedder
from app.pipelines.retrieval.retrieval import GraphRAGRetrievalUseCase


class MockReranker(Reranker):
    """Mock reranker for development purposes.

    Provides a simple passthrough reranking mechanism.
    """

    async def rerank(
        self, query: str, contexts: list[str], top_k: int = 5
    ) -> list[dict[str, object]]:
        """Rerank mock implementation.

        Args:
            query: The search query.
            contexts: List of context strings to rank.
            top_k: Number of results to return.

        Returns:
            List of dictionaries containing text and a dummy score.
        """
        return [{"text": ctx, "score": 1.0} for ctx in contexts[:top_k]]


# Global SurrealDB connection instance
surreal_db = AsyncSurreal(settings.surrealdb_url)


async def init_db() -> None:
    """Initialize SurrealDB connection with proper async authentication.

    Connects to the database, signs in with configured credentials, and
    initializes the document and graph store schemas.
    """
    await surreal_db.connect(settings.surrealdb_url)
    await surreal_db.signin(
        {
            "username": settings.surrealdb_user,
            "password": settings.surrealdb_pass,
        }
    )

    await surreal_db.use(settings.surrealdb_ns, settings.surrealdb_db)

    # Initialize schema indices
    doc_store = SurrealDocumentStore(surreal_db)
    graph_store = SurrealGraphStore(surreal_db)
    await doc_store.initialize_schema()
    await graph_store.initialize_schema()


def get_db() -> AsyncSurreal:  # type: ignore
    """Get the global SurrealDB connection.

    Returns:
        The active AsyncSurreal database instance.
    """
    return surreal_db


def get_document_store(db: AsyncSurreal = Depends(get_db)) -> DocumentStore:  # type: ignore
    """Get the document store implementation.

    Args:
        db: The database connection dependency.

    Returns:
        An instance of SurrealDocumentStore.
    """
    return SurrealDocumentStore(db)


def get_graph_store(db: AsyncSurreal = Depends(get_db)) -> GraphStore:  # type: ignore
    """Get the graph store implementation.

    Args:
        db: The database connection dependency.

    Returns:
        An instance of SurrealGraphStore.
    """
    return SurrealGraphStore(db)


_coref_lock = threading.Lock()
_coref_resolver: CoreferenceResolver | None = None


def get_coref_resolver() -> CoreferenceResolver:
    """Get the coreference resolver implementation (cached singleton).

    Returns:
        An instance of FastCorefResolver.
    """
    global _coref_resolver
    with _coref_lock:
        if _coref_resolver is None:
            _coref_resolver = FastCorefResolver()
    return _coref_resolver


_embedder_lock = threading.Lock()
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get the text embedder implementation (cached singleton).

    Returns:
        An instance of HuggingFaceEmbedder.
    """
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = SentenceTransformerEmbedder(
                model_name=settings.embedding_model_id, device=settings.device
            )
    return _embedder


_extractor_lock = threading.Lock()
_extractor: EntityExtractor | None = None


def get_extractor() -> EntityExtractor:
    """Get the entity extractor implementation (cached singleton).

    Returns:
        An instance of GeminiEntityExtractor if API key is present AND local models are disabled,
        otherwise falls back to GLiNERFallbackExtractor.
    """
    global _extractor
    with _extractor_lock:
        if _extractor is None:
            # Respect the local NLP toggle before attempting to use the exhausted API
            if settings.gemini_api_key and not settings.use_local_nlp_models:
                _extractor = GeminiEntityExtractor(settings.gemini_api_key, settings.gemini_model)
            else:
                _extractor = GLiNERFallbackExtractor()
    return _extractor


def get_resolver() -> EntityResolver:
    """Get the entity resolver implementation.

    Returns:
        An instance of JaroWinklerResolver.
    """
    return JaroWinklerResolver()


def get_linker(extractor: EntityExtractor = Depends(get_extractor)) -> SimpleEntityLinker:
    """Get the entity linker implementation.

    Args:
        extractor: The entity extractor dependency.

    Returns:
        An instance of SimpleEntityLinker.
    """
    return SimpleEntityLinker(extractor)


def get_reranker() -> MockReranker:
    """Get the reranker implementation.

    Returns:
        An instance of MockReranker.
    """
    return MockReranker()


def get_extraction_use_case(
    extractor: EntityExtractor = Depends(get_extractor),
    resolver: EntityResolver = Depends(get_resolver),
    graph_store: GraphStore = Depends(get_graph_store),
    embedder: Embedder = Depends(get_embedder),
) -> GraphExtractionUseCase:
    """Get the graph extraction use case.

    Args:
        extractor: Entity extraction dependency.
        resolver: Entity resolution dependency.
        graph_store: Graph storage dependency.
        embedder: Text embedding dependency.

    Returns:
        An initialized GraphExtractionUseCase.
    """
    return GraphExtractionUseCase(extractor, resolver, graph_store, embedder)


_generator_lock = threading.Lock()
_generator: LLMGenerator | None = None


def get_generator() -> LLMGenerator:
    """Get the LLM response generator implementation (cached singleton).

    The generator is expensive to initialise — loading a GGUF model from disk
    takes tens of seconds. This singleton ensures it is loaded exactly once
    for the lifetime of the process.

    Returns:
        An instance of LocalLlamaGenerator if local models are enabled,
        otherwise a GeminiGenerator.

    Raises:
        RuntimeError: If local models are enabled but LOCAL_LLM_PATH is unset.
    """
    global _generator
    with _generator_lock:
        if _generator is None:
            if settings.use_local_nlp_models:
                if not settings.local_llm_path:
                    raise RuntimeError("LOCAL_LLM_PATH is not configured in environment.")

                # Resolve relative path from .env to absolute path in models_dir
                llm_path = Path(settings.local_llm_path)
                if not llm_path.is_absolute():
                    # The bootstrap downloads to models_dir / filename
                    llm_path = settings.models_dir / llm_path.name

                if not llm_path.exists():
                    raise RuntimeError(
                        f"Local model not found at {llm_path}. "
                        "Please ensure 'uv run download-models' has completed successfully."
                    )

                _generator = LocalLlamaGenerator(str(llm_path))
            else:
                # Fallback only if local models are explicitly disabled
                _generator = GeminiGenerator(settings.gemini_api_key, settings.gemini_model)
    return _generator


def get_ingestion_use_case(
    coref_resolver: CoreferenceResolver = Depends(get_coref_resolver),
    document_store: DocumentStore = Depends(get_document_store),
    embedder: Embedder = Depends(get_embedder),
    graph_extraction_use_case: GraphExtractionUseCase = Depends(get_extraction_use_case),
    graph_store: GraphStore = Depends(get_graph_store),
    llm_generator: LLMGenerator = Depends(get_generator),
) -> DocumentIngestionUseCase:
    """Get the document ingestion use case.

    Args:
        coref_resolver: Coreference resolution dependency.
        document_store: Document storage dependency.
        embedder: Text embedding dependency.
        graph_extraction_use_case: Graph extraction use case dependency.
        graph_store: Graph storage dependency.
        llm_generator: LLM generator dependency for summarization.

    Returns:
        An initialized DocumentIngestionUseCase.
    """
    return DocumentIngestionUseCase(
        coref_resolver=coref_resolver,
        document_store=document_store,
        embedder=embedder,
        graph_extraction_use_case=graph_extraction_use_case,
        graph_store=graph_store,
        llm_generator=llm_generator,
    )


def get_retrieval_use_case(
    doc_store: DocumentStore = Depends(get_document_store),
    graph_store: GraphStore = Depends(get_graph_store),
    embedder: Embedder = Depends(get_embedder),
    linker: SimpleEntityLinker = Depends(get_linker),
    reranker: MockReranker = Depends(get_reranker),
    generator: LLMGenerator = Depends(get_generator),
) -> GraphRAGRetrievalUseCase:
    """Get the GraphRAG retrieval use case.

    Args:
        doc_store: Document storage dependency.
        graph_store: Graph storage dependency.
        embedder: Text embedding dependency.
        linker: Entity linking dependency.
        reranker: Reranking dependency.
        generator: LLM dependency for document grading and query rewriting.

    Returns:
        An initialized GraphRAGRetrievalUseCase.
    """
    return GraphRAGRetrievalUseCase(doc_store, graph_store, embedder, linker, reranker, generator)


def get_enhancement_use_case(
    graph_store: GraphStore = Depends(get_graph_store),
) -> GraphEnhancementUseCase:
    """Get the graph enhancement use case.

    Args:
        graph_store: Graph storage dependency.

    Returns:
        An initialized GraphEnhancementUseCase.
    """
    return GraphEnhancementUseCase(graph_store)


def get_chat_use_case(
    retrieval_use_case: GraphRAGRetrievalUseCase = Depends(get_retrieval_use_case),
    generator: LLMGenerator = Depends(get_generator),
) -> ChatUseCase:
    """Get the conversational chat use case.

    Args:
        retrieval_use_case: GraphRAG context retrieval dependency.
        generator: LLM response generation dependency.

    Returns:
        An initialized ChatUseCase.
    """
    return ChatUseCase(retrieval_use_case, generator)


def get_notebook_use_case(
    store: DocumentStore = Depends(get_document_store),
) -> NotebookUseCase:
    """Get the notebook management use case.

    Args:
        store: Document storage dependency.

    Returns:
        An initialized NotebookUseCase.
    """
    return NotebookUseCase(store)


_vlm_lock = threading.Lock()
_vlm: LocalVLM | None = None


def get_vlm() -> LocalVLM:
    """Get the local VLM implementation (cached singleton).

    Returns:
        An instance of LocalVLM.
    """
    global _vlm
    with _vlm_lock:
        if _vlm is None:
            from app.pipelines.generation.vlm import LocalVLM as LocalVLMImpl

            _vlm = LocalVLMImpl()
    return _vlm
