"""Abstract ports (interfaces) for the GraphRAG application."""

from abc import ABC, abstractmethod

from app.domain.models import Chunk, Community, Document, Edge, Node


class CoreferenceResolver(ABC):
    """Port for coreference resolution."""

    @abstractmethod
    async def resolve(self, text: str) -> str:
        """Resolve coreferences in the text (e.g., replacing pronouns with entities)."""
        pass


class EntityExtractor(ABC):
    """Port for extracting entities and relationships."""

    @abstractmethod
    async def extract(self, text: str) -> tuple[list[Node], list[Edge]]:
        """Extract nodes (entities) and edges (relationships) from text."""
        pass


class EntityResolver(ABC):
    """Port for entity resolution and deduplication."""

    @abstractmethod
    async def resolve_entities(
        self, new_nodes: list[Node], existing_nodes: list[Node]
    ) -> list[Node]:
        """Merge synonymous entities."""
        pass


class DocumentStore(ABC):
    """Port for document and chunk storage."""

    @abstractmethod
    async def save_document(self, document: Document) -> None:
        """Save a document record."""
        pass

    @abstractmethod
    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save text chunks."""
        pass

    @abstractmethod
    async def search_chunks(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Search for chunks by vector similarity."""
        pass

    @abstractmethod
    async def get_all_documents(self) -> list[Document]:
        """Retrieve all ingested documents."""
        pass


class GraphStore(ABC):
    """Port for graph storage and traversal."""

    @abstractmethod
    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save entity nodes to the graph."""
        pass

    @abstractmethod
    async def save_edges(self, edges: list[Edge]) -> None:
        """Save relationship edges to the graph."""
        pass

    @abstractmethod
    async def traverse(
        self, seed_node_ids: list[str], depth: int = 2
    ) -> tuple[list[Node], list[Edge]]:
        """Traverse the graph starting from seed nodes up to a given depth."""
        pass

    @abstractmethod
    async def get_all_nodes(self) -> list[Node]:
        """Retrieve all nodes (e.g. for community detection)."""
        pass

    @abstractmethod
    async def get_all_edges(self) -> list[Edge]:
        """Retrieve all edges (e.g. for community detection)."""
        pass

    @abstractmethod
    async def save_community(self, community: Community) -> None:
        """Save a community and its summary."""
        pass


class Embedder(ABC):
    """Port for text embeddings."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text string."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of text strings."""
        pass


class LLMGenerator(ABC):
    """Port for generating text responses using an LLM."""

    @abstractmethod
    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a response for a given prompt and history."""
        pass
