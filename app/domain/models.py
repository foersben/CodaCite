"""Domain models for the GraphRAG application.

This module contains the pure Pydantic models representing the core entities
of the GraphRAG system. These models are strictly decoupled from any
infrastructure-specific details.
"""

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A text chunk extracted from a document.

    Attributes:
        id: Unique identifier for the chunk.
        document_id: ID of the source document.
        text: The raw text content of the chunk.
        index: Sequential index of the chunk within the document.
        embedding: Optional vector embedding for semantic search.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the chunk.")
    document_id: str = Field(..., description="ID of the source document.")
    text: str = Field(..., description="The chunk text.")
    index: int = Field(..., description="Sequential index of the chunk in the document.")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding of the chunk text."
    )


class Node(BaseModel):
    """An entity node in the knowledge graph.

    Attributes:
        id: Unique identifier for the node (normalized entity name).
        label: The semantic category of the entity (e.g., ORGANIZATION).
        name: The human-readable name of the entity.
        description: A textual description of the entity's role or context.
        description_embedding: Vector embedding of the description.
        source_chunk_ids: List of chunk IDs where this entity was discovered.
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
    """A relationship edge in the knowledge graph.

    Attributes:
        source_id: ID of the origin node.
        target_id: ID of the destination node.
        relation: The type of relationship (e.g., WORKS_AT).
        description: Detailed context of the relationship.
        source_chunk_ids: List of chunk IDs justifying this relation.
        weight: Confidence or strength score of the relationship.
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
    """A document processed by the system.

    Attributes:
        id: Unique identifier for the document.
        filename: Original name of the uploaded file.
        status: Current processing status (e.g., 'processing', 'active', 'failed').
        metadata: Key-value store for auxiliary document info.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the document.")
    filename: str = Field(..., description="Original filename.")
    status: str = Field(default="active", description="Current processing status.")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Additional document metadata."
    )


class Community(BaseModel):
    """A detected community of nodes in the graph.

    Attributes:
        id: Unique identifier for the community cluster.
        summary: Generated thematic summary of the community.
        node_ids: List of node IDs belonging to this community.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the community.")
    summary: str = Field(..., description="Generated summary of the community's theme or focus.")
    node_ids: list[str] = Field(
        default_factory=list, description="IDs of nodes belonging to this community."
    )


class Notebook(BaseModel):
    """A collection of documents.

    Attributes:
        id: Unique identifier for the notebook.
        title: Human-readable name of the notebook.
        description: Optional description of the notebook's purpose.
        created_at: ISO timestamp of creation.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the notebook.")
    title: str = Field(..., description="Title of the notebook.")
    description: str | None = Field(default=None, description="Description of the notebook.")
    created_at: str | None = Field(default=None, description="ISO timestamp of creation.")
