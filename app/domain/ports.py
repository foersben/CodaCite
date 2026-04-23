"""Abstract ports (interfaces) for the GraphRAG application.

This module defines the abstract base classes (ABC) that serve as the
architectural contracts for the domain layer. All infrastructure
implementations must adhere to these interfaces.
"""

from abc import ABC, abstractmethod

from app.domain.models import Chunk, Community, Document, Edge, Node


class CoreferenceResolver(ABC):
    """Port for coreference resolution."""

    @abstractmethod
    async def resolve(self, text: str) -> str:
        """Resolve coreferences in the text.

        Args:
            text: The raw text content to process.

        Returns:
            The text with pronouns and ambiguous mentions resolved to entities.
        """
        pass


class EntityExtractor(ABC):
    """Port for extracting entities and relationships."""

    @abstractmethod
    async def extract(self, text: str) -> tuple[list[Node], list[Edge]]:
        """Extract nodes and edges from text.

        Args:
            text: The semantic chunk of text to analyze.

        Returns:
            A tuple containing a list of entity Nodes and relationship Edges.
        """
        pass


class EntityResolver(ABC):
    """Port for entity resolution and deduplication."""

    @abstractmethod
    async def resolve_entities(
        self, new_nodes: list[Node], existing_nodes: list[Node]
    ) -> list[Node]:
        """Merge synonymous entities using string similarity and embeddings.

        Args:
            new_nodes: Freshly extracted nodes from the pipeline.
            existing_nodes: Nodes already present in the knowledge base.

        Returns:
            A consolidated list of nodes with resolved duplicates.
        """
        pass


class DocumentStore(ABC):
    """Port for document and chunk storage."""

    @abstractmethod
    async def save_document(self, document: Document) -> None:
        """Save a document record.

        Args:
            document: The document metadata to persist.
        """
        pass

    @abstractmethod
    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save text chunks.

        Args:
            chunks: List of chunk objects to persist.
        """
        pass

    @abstractmethod
    async def search_chunks(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Search for chunks by vector similarity.

        Args:
            query_embedding: The vector representation of the search query.
            top_k: Number of most similar results to return.

        Returns:
            A list of matching Chunk objects.
        """
        pass

    @abstractmethod
    async def get_all_documents(self) -> list[Document]:
        """Retrieve all ingested documents.

        Returns:
            A list of all Document records in the store.
        """
        pass


class GraphStore(ABC):
    """Port for graph storage and traversal."""

    @abstractmethod
    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save entity nodes to the graph.

        Args:
            nodes: List of entity nodes to persist.
        """
        pass

    @abstractmethod
    async def save_edges(self, edges: list[Edge]) -> None:
        """Save relationship edges to the graph.

        Args:
            edges: List of relationship edges to persist.
        """
        pass

    @abstractmethod
    async def traverse(
        self, seed_node_ids: list[str], depth: int = 2
    ) -> tuple[list[Node], list[Edge]]:
        """Traverse the graph starting from seed nodes.

        Args:
            seed_node_ids: Initial entity IDs to start the traversal.
            depth: Number of hops to explore from the seeds.

        Returns:
            A tuple of (Nodes, Edges) discovered during traversal.
        """
        pass

    @abstractmethod
    async def get_all_nodes(self) -> list[Node]:
        """Retrieve all nodes for global operations.

        Returns:
            A list of all entity nodes in the graph.
        """
        pass

    @abstractmethod
    async def get_all_edges(self) -> list[Edge]:
        """Retrieve all edges for global operations.

        Returns:
            A list of all relationship edges in the graph.
        """
        pass

    @abstractmethod
    async def save_community(self, community: Community) -> None:
        """Save a community and its summary.

        Args:
            community: The detected community cluster to persist.
        """
        pass


class Embedder(ABC):
    """Port for text embeddings."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text string.

        Args:
            text: The string content to vectorize.

        Returns:
            A list of floats representing the semantic vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for multiple strings.

        Args:
            texts: List of strings to vectorize in a single pass.

        Returns:
            A list of semantic vectors.
        """
        pass


class LLMGenerator(ABC):
    """Port for generating text responses using an LLM."""

    @abstractmethod
    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a response for a given prompt and history.

        Args:
            prompt: The specific question or instruction for the LLM.
            history: Optional conversation history for context.

        Returns:
            The generated text response.
        """
        pass
