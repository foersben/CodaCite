"""FastAPI dependencies for the application.

This module provides dependency injection providers for use cases, infrastructure
implementations, and database connections.
"""

from fastapi import Depends
from surrealdb import AsyncSurreal

from app.application.chat import ChatUseCase
from app.application.enhancement import GraphEnhancementUseCase
from app.application.extraction import GraphExtractionUseCase
from app.application.ingestion import DocumentIngestionUseCase
from app.application.notebook import NotebookUseCase
from app.application.retrieval import GraphRAGRetrievalUseCase
from app.config import settings
from app.domain.ports import (
    CoreferenceResolver,
    DocumentStore,
    Embedder,
    EntityExtractor,
    EntityResolver,
    GraphStore,
    LLMGenerator,
)
from app.infrastructure.coreference import FastCorefResolver
from app.infrastructure.database.store import SurrealDocumentStore, SurrealGraphStore
from app.infrastructure.embeddings import HuggingFaceEmbedder
from app.infrastructure.extraction import GeminiEntityExtractor, GLiNERFallbackExtractor
from app.infrastructure.generator import GeminiGenerator
from app.infrastructure.linker import SimpleEntityLinker
from app.infrastructure.resolution import JaroWinklerResolver


class MockReranker:
    """Mock reranker for development purposes.

    Provides a simple passthrough reranking mechanism.
    """

    async def rerank(
        self, query: str, contexts: list[str], top_k: int = 5
    ) -> list[dict[str, str | float]]:
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


def get_coref_resolver() -> CoreferenceResolver:
    """Get the coreference resolver implementation.

    Returns:
        An instance of FastCorefResolver.
    """
    return FastCorefResolver()


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get the text embedder implementation.

    Returns:
        An instance of HuggingFaceEmbedder (cached as a singleton).
    """
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbedder()
    return _embedder


def get_extractor() -> EntityExtractor:
    """Get the entity extractor implementation.

    Returns:
        An instance of GeminiEntityExtractor if API key is present,
        otherwise falls back to GLiNERFallbackExtractor.
    """
    if settings.gemini_api_key:
        return GeminiEntityExtractor(settings.gemini_api_key, settings.gemini_model)
    return GLiNERFallbackExtractor()


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


def get_ingestion_use_case(
    coref_resolver: CoreferenceResolver = Depends(get_coref_resolver),
    store: DocumentStore = Depends(get_document_store),
    embedder: Embedder = Depends(get_embedder),
    extractor: EntityExtractor = Depends(get_extractor),
    resolver: EntityResolver = Depends(get_resolver),
    graph_store: GraphStore = Depends(get_graph_store),
) -> DocumentIngestionUseCase:
    """Get the document ingestion use case.

    Args:
        coref_resolver: Coreference resolution dependency.
        store: Document storage dependency.
        embedder: Text embedding dependency.
        extractor: Entity extraction dependency.
        resolver: Entity resolution dependency.
        graph_store: Graph storage dependency.

    Returns:
        An initialized DocumentIngestionUseCase.
    """
    return DocumentIngestionUseCase(
        coref_resolver, store, embedder, extractor, resolver, graph_store
    )


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


def get_retrieval_use_case(
    doc_store: DocumentStore = Depends(get_document_store),
    graph_store: GraphStore = Depends(get_graph_store),
    embedder: Embedder = Depends(get_embedder),
    linker: SimpleEntityLinker = Depends(get_linker),
    reranker: MockReranker = Depends(get_reranker),
) -> GraphRAGRetrievalUseCase:
    """Get the GraphRAG retrieval use case.

    Args:
        doc_store: Document storage dependency.
        graph_store: Graph storage dependency.
        embedder: Text embedding dependency.
        linker: Entity linking dependency.
        reranker: Reranking dependency.

    Returns:
        An initialized GraphRAGRetrievalUseCase.
    """
    return GraphRAGRetrievalUseCase(doc_store, graph_store, embedder, linker, reranker)


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


def get_generator() -> LLMGenerator:
    """Get the LLM response generator implementation.

    Returns:
        An instance of GeminiGenerator.
    """
    return GeminiGenerator(settings.gemini_api_key, settings.gemini_model)


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
