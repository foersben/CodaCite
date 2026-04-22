"""FastAPI dependencies for the application."""

from typing import Any

from fastapi import Depends
from surrealdb import AsyncSurreal

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
)
from app.infrastructure.coreference import FastCorefResolver
from app.infrastructure.database.store import SurrealDocumentStore, SurrealGraphStore
from app.infrastructure.extraction import GeminiEntityExtractor, GLiNERFallbackExtractor
from app.infrastructure.linker import SimpleEntityLinker
from app.infrastructure.resolution import JaroWinklerResolver


class MockEmbedder(Embedder):
    """Mock embedder."""

    async def embed(self, text: str) -> list[float]:
        """Embed mock."""
        return [0.1] * 1024

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch mock."""
        return [[0.1] * 1024 for _ in texts]


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
    return MockEmbedder()


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
) -> DocumentIngestionUseCase:
    """Get ingestion use case."""
    return DocumentIngestionUseCase(resolver, store)


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
