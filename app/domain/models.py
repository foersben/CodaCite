"""Domain models for the GraphRAG application.

This module contains the pure Pydantic models representing the core entities
of the GraphRAG system. These models are strictly decoupled from any
infrastructure-specific details.
"""

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A semantic text chunk extracted from a document.

    Chunks are the fundamental units of retrieval and graph extraction in the
    ingestion pipeline (Phase 2). They are typically generated using the
    RecursiveCharacterTextSplitter.

    Attributes:
        id: Unique identifier for the chunk (e.g., doc_id_index).
        document_id: ID of the source document stored in DocumentStore.
        text: The raw text content of the chunk after coreference resolution.
        index: Sequential index of the chunk within the original document.
        embedding: Vector representation for semantic search (FAISS/SurrealDB).
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the chunk.")
    document_id: str = Field(..., description="ID of the source document.")
    text: str = Field(..., description="The chunk text.")
    index: int = Field(..., description="Sequential index of the chunk in the document.")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding of the chunk text."
    )
    score: float | None = Field(default=None, description="Similarity score from retrieval.")


class Node(BaseModel):
    """An entity node in the Knowledge Graph.

    Nodes represent normalized entities extracted from text chunks during
    Phase 5 of the ingestion pipeline. They are deduplicated via the
    EntityResolver using semantic similarity.

    Attributes:
        id: Unique identifier for the node (normalized and resolved entity name).
        label: The semantic category (e.g., PERSON, ORGANIZATION, GPE).
        name: The human-readable name of the entity.
        description: A summarized context of the entity's role in the dataset.
        description_embedding: Vector representation of the entity description.
        source_chunk_ids: List of source chunks justifying this entity's existence.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(
        ..., description="Unique identifier for the node (usually the entity name normalized)."
    )
    label: str = Field(
        ..., description="The type/label of the entity (e.g., PERSON, ORGANIZATION)."
    )
    name: str = Field(..., description="The extracted name of the entity.")
    description: str | None = Field(default=None, description="Description of the entity.")
    description_embedding: list[float] | None = Field(
        default=None, description="Vector embedding of the description."
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of chunks where this entity was found."
    )


class Edge(BaseModel):
    """A relationship edge in the Knowledge Graph.

    Edges connect two nodes and describe their semantic relationship. They are
    extracted during Phase 5 and normalized during the extraction use case.

    Attributes:
        source_id: ID of the source entity node.
        target_id: ID of the target entity node.
        relation: The normalized relationship type (e.g., WORKS_FOR, LOCATED_IN).
        description: Detailed context or evidence for the relationship.
        source_chunk_ids: List of chunks where this relationship was evidenced.
        weight: Confidence score (0.0 to 1.0) of the extraction.
    """

    model_config = ConfigDict(strict=True)

    source_id: str = Field(..., description="ID of the source node.")
    target_id: str = Field(..., description="ID of the target node.")
    relation: str = Field(..., description="The relationship type (e.g., FOUNDED_BY).")
    description: str | None = Field(default=None, description="Description of the relationship.")
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of chunks where this relationship was found."
    )
    weight: float = Field(default=1.0, description="Weight or confidence of the relationship.")


class Document(BaseModel):
    """A source document ingested into the system.

    Represents the top-level entity for a processed file (PDF, Docx, MD).

    Attributes:
        id: UUID for the document.
        filename: Original filename of the upload.
        status: Current pipeline state ('processing', 'active', 'failed').
        metadata: Dictionary of extra information (e.g., author, date).
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the document.")
    filename: str = Field(..., description="Original filename.")
    status: str = Field(default="active", description="Current processing status.")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Additional document metadata."
    )


class Community(BaseModel):
    """A detected community cluster in the knowledge graph.

    Communities are groups of related nodes detected via algorithms like
    Leiden or Louvain (often used for global RAG summarization).

    Attributes:
        id: Unique identifier for the community.
        summary: AI-generated thematic summary of the entities in this cluster.
        node_ids: List of member node IDs.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the community.")
    summary: str = Field(..., description="Generated summary of the community's theme or focus.")
    node_ids: list[str] = Field(
        default_factory=list, description="IDs of nodes belonging to this community."
    )


class Notebook(BaseModel):
    """A logical collection of documents for scoped retrieval.

    Allows users to organize documents into projects and restrict retrieval
    to specific sub-sets of the knowledge base.

    Attributes:
        id: UUID for the notebook.
        title: User-defined name.
        description: Optional project description.
        created_at: ISO timestamp.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the notebook.")
    title: str = Field(..., description="Title of the notebook.")
    description: str | None = Field(default=None, description="Description of the notebook.")
    created_at: str | None = Field(default=None, description="ISO timestamp of creation.")
