"""Abstract ports (interfaces) for the GraphRAG application.

This module defines the abstract base classes (ABC) that serve as the
architectural contracts for the domain layer. All infrastructure
implementations must adhere to these interfaces.
"""

from abc import ABC, abstractmethod

from app.domain.models import Chunk, Community, Document, Edge, Node, Notebook


class CoreferenceResolver(ABC):
    """Port for resolving semantic coreferences in text.

    Implementations (e.g., FastCoref) ensure that mentions like 'he', 'she', or
    'it' are replaced with their respective entity names to improve downstream
    extraction and retrieval accuracy.
    """

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
    """Port for extracting structured Knowledge Graph data from text.

    Implementations (e.g., GeminiExtractor, GlinerExtractor) parse text chunks
    to identify entity Nodes and relationship Edges.
    """

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
    """Port for entity resolution and deduplication across the graph.

    Implementations merge synonymous entities (e.g., 'Apple Inc' and 'Apple')
    to maintain a clean and connected Knowledge Graph.
    """

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
    """Port for persisting raw document metadata and text chunks.

    Handles the storage of Document objects and their associated Chunk vectors.
    Implementations typically use databases with vector search capabilities
    (e.g., SurrealDB).
    """

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
    async def search_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        active_notebook_ids: list[str] | None = None,
    ) -> list[Chunk]:
        """Search for chunks by vector similarity with optional notebook filtering.

        Args:
            query_embedding: The vector representation of the search query.
            top_k: Number of most similar results to return.
            active_notebook_ids: Optional list of notebook IDs to restrict the search.

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

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None:
        """Retrieve a specific document by its ID.

        Args:
            document_id: The unique identifier of the document.

        Returns:
            The Document object if found, otherwise None.
        """
        pass

    @abstractmethod
    async def update_document_status(self, document_id: str, status: str) -> None:
        """Update the processing status of a document.

        Args:
            document_id: The ID of the document to update.
            status: The new status string.
        """
        pass

    @abstractmethod
    async def add_document_to_notebook(self, document_id: str, notebook_id: str) -> None:
        """Relate a document to a notebook using a graph edge.

        Args:
            document_id: The ID of the document.
            notebook_id: The ID of the notebook.
        """
        pass

    @abstractmethod
    async def remove_document_from_notebook(self, document_id: str, notebook_id: str) -> None:
        """Remove a relationship between a document and a notebook.

        Args:
            document_id: The ID of the document.
            notebook_id: The ID of the notebook.
        """
        pass

    @abstractmethod
    async def get_notebook_documents(self, notebook_id: str) -> list[Document]:
        """Retrieve all documents associated with a specific notebook.

        Args:
            notebook_id: The notebook ID.

        Returns:
            A list of Document objects linked to the notebook.
        """
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Delete a document and trigger maintenance if needed.

        Args:
            document_id: The ID of the document to remove.
        """
        pass

    @abstractmethod
    async def save_notebook(self, notebook: Notebook) -> None:
        """Save a notebook record.

        Args:
            notebook: The notebook metadata to persist.
        """
        pass

    @abstractmethod
    async def get_all_notebooks(self) -> list[Notebook]:
        """Retrieve all notebooks.

        Returns:
            A list of all Notebook records in the store.
        """
        pass

    @abstractmethod
    async def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook and its document relations.

        Args:
            notebook_id: The ID of the notebook to remove.
        """
        pass


class GraphStore(ABC):
    """Port for persisting and traversing the Knowledge Graph.

    Handles Node and Edge persistence and provides methods for multi-hop
    traversal used in the GraphRAG retrieval pipeline.
    """

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

        Used in hybrid retrieval to pull in relevant multi-hop context.

        Args:
            seed_node_ids: Initial entity IDs to start the traversal.
            depth: Number of hops to explore from the seeds.

        Returns:
            A tuple of (Nodes, Edges) discovered during traversal.
        """
        pass

    @abstractmethod
    async def get_all_nodes(self) -> list[Node]:
        """Retrieve all nodes for global operations (e.g., entity resolution).

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
        """Save a community cluster and its summary.

        Args:
            community: The detected community cluster to persist.
        """
        pass


class Embedder(ABC):
    """Port for generating semantic vector embeddings.

    Implementations (e.g., HuggingFaceEmbedder, GeminiEmbedder) convert text
    into high-dimensional vectors for similarity search and entity resolution.
    """

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
    """Port for interacting with Large Language Models.

    Contract for text generation and chat features. Implementations typically
    wrap OpenAI, Gemini, or local models via LangChain.
    """

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
