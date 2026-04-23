"""FastAPI dependencies for the application."""

from typing import Any

from fastapi import Depends
from surrealdb import AsyncSurreal

from app.application.enhancement import GraphEnhancementUseCase
from app.application.chat import ChatUseCase
from app.application.extraction import GraphExtractionUseCase
from app.application.ingestion import DocumentIngestionUseCase
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
    """Mock reranker."""

    async def rerank(
        self, query: str, contexts: list[str], top_k: int = 5
    ) -> list[dict[str, str | float]]:
        """Rerank mock."""
        return [{"text": ctx, "score": 1.0} for ctx in contexts[:top_k]]


# Global SurrealDB connection
surreal_db = AsyncSurreal(settings.surrealdb_url)


async def init_db() -> None:
    """Initialize SurrealDB connection with proper async authentication."""
    # Now that we use the async driver, .connect() is a valid awaitable
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


def get_db() -> Any:
    """Get SurrealDB connection."""
    return surreal_db


def get_document_store(db: Any = Depends(get_db)) -> DocumentStore:
    """Get document store."""
    return SurrealDocumentStore(db)


def get_graph_store(db: Any = Depends(get_db)) -> GraphStore:
    """Get graph store."""
    return SurrealGraphStore(db)


def get_coref_resolver() -> CoreferenceResolver:
    """Get coref resolver."""
    return FastCorefResolver()


def get_embedder() -> Embedder:
    """Get embedder."""
    return HuggingFaceEmbedder()


def get_extractor() -> EntityExtractor:
    """Get extractor."""
    if settings.gemini_api_key:
        return GeminiEntityExtractor(settings.gemini_api_key, settings.gemini_model)
    return GLiNERFallbackExtractor()


def get_resolver(embedder: Embedder = Depends(get_embedder)) -> EntityResolver:
    """Get resolver."""
    return JaroWinklerResolver(embedder)


def get_linker(extractor: EntityExtractor = Depends(get_extractor)) -> Any:
    """Get linker."""
    return SimpleEntityLinker(extractor)


def get_reranker() -> Any:
    """Get reranker."""
    return MockReranker()


def get_ingestion_use_case(
    resolver: CoreferenceResolver = Depends(get_coref_resolver),
    store: DocumentStore = Depends(get_document_store),
    embedder: Embedder = Depends(get_embedder),
) -> DocumentIngestionUseCase:
    """Get ingestion use case."""
    return DocumentIngestionUseCase(resolver, store, embedder)


def get_extraction_use_case(
    extractor: EntityExtractor = Depends(get_extractor),
    resolver: EntityResolver = Depends(get_resolver),
    graph_store: GraphStore = Depends(get_graph_store),
    embedder: Embedder = Depends(get_embedder),
) -> GraphExtractionUseCase:
    """Get extraction usecase."""
    return GraphExtractionUseCase(extractor, resolver, graph_store, embedder)


def get_retrieval_use_case(
    doc_store: DocumentStore = Depends(get_document_store),
    graph_store: GraphStore = Depends(get_graph_store),
    embedder: Embedder = Depends(get_embedder),
    linker: Any = Depends(get_linker),
    reranker: Any = Depends(get_reranker),
) -> GraphRAGRetrievalUseCase:
    """Get retrieval use case."""
    return GraphRAGRetrievalUseCase(doc_store, graph_store, embedder, linker, reranker)


def get_enhancement_use_case(
    graph_store: GraphStore = Depends(get_graph_store),
) -> GraphEnhancementUseCase:
    """Get enhancement use case."""
    return GraphEnhancementUseCase(graph_store)


def get_generator() -> LLMGenerator:
    """Get LLM generator."""
    return GeminiGenerator(settings.gemini_api_key, settings.gemini_model)


def get_chat_use_case(
    retrieval_use_case: GraphRAGRetrievalUseCase = Depends(get_retrieval_use_case),
    generator: LLMGenerator = Depends(get_generator),
) -> ChatUseCase:
    """Get chat use case."""
    return ChatUseCase(retrieval_use_case, generator)
