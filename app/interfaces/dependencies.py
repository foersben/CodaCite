"""FastAPI dependencies for the application."""

from fastapi import Depends

from app.application.extraction import GraphExtractionUseCase
from app.application.ingestion import DocumentIngestionUseCase
from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.models import Chunk, Edge, Node
from app.domain.ports import (
    CoreferenceResolver,
    DocumentStore,
    Embedder,
    EntityExtractor,
    EntityResolver,
    GraphStore,
)

# Mock implementation of ports to satisfy type checker for now


class MockDocumentStore(DocumentStore):
    """Mock document store."""
    async def save_document(self, doc: object) -> None:
        """Save document mock."""
        pass

    async def save_chunks(self, chunks: object) -> None:
        """Save chunks mock."""
        pass

    async def search_chunks(self, query_emb: list[float], top_k: int = 5) -> list[Chunk]:
        """Search chunks mock."""
        return []


class MockGraphStore(GraphStore):
    """Mock graph store."""
    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save nodes mock."""
        pass

    async def save_edges(self, edges: list[Edge]) -> None:
        """Save edges mock."""
        pass

    async def traverse(self, seed: list[str], depth: int = 2) -> tuple[list[Node], list[Edge]]:
        """Traverse mock."""
        return [], []

    async def get_all_nodes(self) -> list[Node]:
        """Get nodes mock."""
        return []

    async def get_all_edges(self) -> list[Edge]:
        """Get edges mock."""
        return []

    async def save_community(self, comm: object) -> None:
        """Save community mock."""
        pass


class MockCoreferenceResolver(CoreferenceResolver):
    """Mock Coref Resolver."""
    async def resolve(self, text: str) -> str:
        """Resolve mock."""
        return text


class MockExtractor(EntityExtractor):
    """Mock Extractor."""
    async def extract(self, text: str) -> tuple[list[Node], list[Edge]]:
        """Extract mock."""
        return [], []


class MockResolver(EntityResolver):
    """Mock resolver."""
    async def resolve_entities(
        self, new_nodes: list[Node], existing_nodes: list[Node]
    ) -> list[Node]:
        """Resolve mock."""
        return new_nodes


class MockEmbedder(Embedder):
    """Mock embedder."""
    async def embed(self, text: str) -> list[float]:
        """Embed mock."""
        return [0.1] * 1024

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch mock."""
        return [[0.1] * 1024 for _ in texts]


class MockLinker:
    """Mock Linker."""
    async def link_entities(self, query: str, nodes: list[Node]) -> list[Node]:
        """Link entities mock."""
        return []


class MockReranker:
    """Mock reranker."""
    async def rerank(
        self, query: str, contexts: list[str], top_k: int = 5
    ) -> list[dict[str, str | float]]:
        """Rerank mock."""
        return [{"text": ctx, "score": 1.0} for ctx in contexts[:top_k]]


def get_document_store() -> DocumentStore:
    """Get document store."""
    return MockDocumentStore()


def get_graph_store() -> GraphStore:
    """Get graph store."""
    return MockGraphStore()


def get_coref_resolver() -> CoreferenceResolver:
    """Get coref resolver."""
    return MockCoreferenceResolver()


def get_embedder() -> Embedder:
    """Get embedder."""
    return MockEmbedder()


def get_extractor() -> EntityExtractor:
    """Get extractor."""
    return MockExtractor()


def get_resolver() -> EntityResolver:
    """Get resolver."""
    return MockResolver()


def get_linker() -> object:
    """Get linker."""
    return MockLinker()


def get_reranker() -> object:
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
    linker: object = Depends(get_linker),
    reranker: object = Depends(get_reranker),
) -> GraphRAGRetrievalUseCase:
    """Get retrieval use case."""
    return GraphRAGRetrievalUseCase(doc_store, graph_store, embedder, linker, reranker)
