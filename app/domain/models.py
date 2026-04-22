"""Domain models for the GraphRAG application."""

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A text chunk extracted from a document."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the chunk.")
    document_id: str = Field(..., description="ID of the source document.")
    text: str = Field(..., description="The chunk text.")
    index: int = Field(..., description="Sequential index of the chunk in the document.")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding of the chunk text."
    )


class Node(BaseModel):
    """An entity node in the knowledge graph."""

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
    """A relationship edge in the knowledge graph."""

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
    """A document processed by the system."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the document.")
    filename: str = Field(..., description="Original filename.")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Additional document metadata."
    )


class Community(BaseModel):
    """A detected community of nodes in the graph."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the community.")
    summary: str = Field(..., description="Generated summary of the community's theme or focus.")
    node_ids: list[str] = Field(
        default_factory=list, description="IDs of nodes belonging to this community."
    )
