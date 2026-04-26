"""SurrealDB implementations for data stores.

This module provides concrete implementations of the DocumentStore and GraphStore
ports using SurrealDB. It includes logic for type harmonization between
SurrealDB RecordIDs and pure Pydantic domain models.
"""

import logging
from typing import Any

from app.domain.models import Chunk, Community, Document, Edge, Node
from app.domain.ports import DocumentStore, GraphStore

logger = logging.getLogger(__name__)


def _extract_rows(result: Any) -> list[dict[str, Any]]:
    """Normalize SurrealDB query results."""
    if not result:
        return []

    # Handle direct list of results
    if isinstance(result, list):
        if not result:
            return []
        first = result[0]
        # Handle envelope format
        if isinstance(first, dict) and "result" in first:
            nested = first["result"]
            return (
                [row for row in nested if isinstance(row, dict)] if isinstance(nested, list) else []
            )
        return [row for row in result if isinstance(row, dict)]

    # Handle single result object
    if isinstance(result, dict):
        nested = result.get("result")
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
        return [result]

    return []


def _clean_id(id_val: Any) -> str:
    """Strip SurrealDB table prefix from RecordID string."""
    id_str = str(id_val)
    return id_str.split(":", 1)[-1] if ":" in id_str else id_str


class SurrealDocumentStore(DocumentStore):
    """SurrealDB implementation of DocumentStore.

    Handles the storage and similarity search of document chunks using
    SurrealDB's native vector indexing (MTREE).
    """

    def __init__(self, db: Any) -> None:
        """Initialize the document store.

        Args:
            db: An active SurrealDB client instance.
        """
        self.db = db

    async def save_document(self, document: Document) -> None:
        """Save a document record to the database.

        Args:
            document: The domain document model to persist.
        """
        logger.info("Saving document to SurrealDB: %s", document.filename)
        await self.db.query(
            "UPDATE type::thing('document', $id) CONTENT { content: $content, metadata: $metadata };",
            {
                "id": document.id,
                "content": document.filename,
                "metadata": document.metadata,
            },
        )

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save text chunks to the database.

        Args:
            chunks: List of chunk domain models to persist.
        """
        logger.info("Saving %d chunks to SurrealDB", len(chunks))
        for chunk in chunks:
            await self.db.query(
                "UPDATE type::thing('chunk', $id) CONTENT { document_id: $document_id, text: $text, index: $index, embedding: $embedding };",
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "index": chunk.index,
                    "embedding": chunk.embedding,
                },
            )

    async def search_chunks(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Search for chunks by vector similarity using cosine distance.

        Args:
            query_embedding: The vector to search for.
            top_k: Maximum number of results to return.

        Returns:
            A list of similar Chunk models.
        """
        result = await self.db.query(
            "SELECT * FROM chunk WHERE embedding <|5|> $embedding;",
            {"embedding": query_embedding},
        )
        rows = _extract_rows(result)
        return [
            Chunk(
                id=_clean_id(item["id"]),
                document_id=_clean_id(item["document_id"]),
                text=item["text"],
                index=item["index"],
                embedding=item.get("embedding"),
            )
            for item in rows
        ][:top_k]

    async def get_all_documents(self) -> list[Document]:
        """Retrieve all ingested documents from the store.

        Returns:
            List of all Document domain models.
        """
        result = await self.db.query("SELECT * FROM document;")
        return [
            Document(
                id=_clean_id(row["id"]),
                filename=row.get("content", "unknown"),
                metadata=row.get("metadata", {}),
            )
            for row in _extract_rows(result)
        ]

    async def initialize_schema(self) -> None:
        """Initialize SurrealDB table indices for document storage.

        Defines the MTREE vector index for chunk embeddings.
        """
        logger.info("Initializing SurrealDocumentStore schema")
        await self.db.query(
            "DEFINE INDEX chunk_embedding_idx ON TABLE chunk FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;"
        )


class SurrealGraphStore(GraphStore):
    """SurrealDB implementation of GraphStore.

    Handles entity and relationship storage, providing graph traversal
    capabilities for context retrieval.
    """

    def __init__(self, db: Any) -> None:
        """Initialize the graph store.

        Args:
            db: An active SurrealDB client instance.
        """
        self.db = db

    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save entity nodes to the knowledge graph.

        Args:
            nodes: List of Node domain models to persist.
        """
        logger.info("Saving %d nodes to SurrealDB Graph", len(nodes))
        for node in nodes:
            await self.db.query(
                "UPDATE type::thing('entity', $id) CONTENT { label: $label, name: $name, description: $description, description_embedding: $description_embedding, source_chunk_ids: $source_chunk_ids };",
                {
                    "id": node.id,
                    "label": node.label,
                    "name": node.name,
                    "description": node.description,
                    "description_embedding": node.description_embedding,
                    "source_chunk_ids": node.source_chunk_ids,
                },
            )

    async def initialize_schema(self) -> None:
        """Initialize SurrealDB table indices for graph storage.

        Defines the MTREE vector index for entity description embeddings.
        """
        logger.info("Initializing SurrealGraphStore schema")
        await self.db.query(
            "DEFINE INDEX entity_embedding_idx ON TABLE entity FIELDS description_embedding MTREE DIMENSION 1024 DIST COSINE;"
        )

    async def save_edges(self, edges: list[Edge]) -> None:
        """Save relationship edges between entities.

        Args:
            edges: List of Edge domain models to persist.
        """
        logger.info("Saving %d edges to SurrealDB Graph", len(edges))
        for edge in edges:
            await self.db.query(
                "RELATE $source->relation->$target CONTENT { relation: $relation, description: $description, source_chunk_ids: $source_chunk_ids, weight: $weight };",
                {
                    "source": f"entity:{edge.source_id}",
                    "target": f"entity:{edge.target_id}",
                    "relation": edge.relation,
                    "description": edge.description,
                    "source_chunk_ids": edge.source_chunk_ids,
                    "weight": edge.weight,
                },
            )

    async def traverse(
        self, seed_node_ids: list[str], depth: int = 2
    ) -> tuple[list[Node], list[Edge]]:
        """Traverse the knowledge graph starting from seed nodes.

        Args:
            seed_node_ids: IDs of nodes to begin the traversal.
            depth: Maximum number of hops to follow.

        Returns:
            A tuple of (reachable_nodes, traversed_edges).
        """
        nodes_dict = {}
        edges_list = []
        visited_edge_ids = set()

        current_level_node_ids = set(seed_node_ids)
        visited_node_ids = set()

        for _ in range(depth):
            if not current_level_node_ids:
                break

            next_level_node_ids = set()

            for node_id in current_level_node_ids:
                visited_node_ids.add(node_id)

                # Query outgoing edges
                out_result = await self.db.query(
                    f"SELECT *, in AS source_id, out AS target_id FROM entity:{node_id}->relation"
                )
                out_rows = _extract_rows(out_result)
                if out_rows:
                    for edge_data in out_rows:
                        edge_id = str(edge_data.get("id"))
                        if edge_id not in visited_edge_ids:
                            visited_edge_ids.add(edge_id)
                            target_id = str(edge_data.get("target_id", "")).replace("entity:", "")
                            source_id = str(edge_data.get("source_id", "")).replace("entity:", "")

                            edges_list.append(
                                Edge(
                                    source_id=source_id,
                                    target_id=target_id,
                                    relation=edge_data.get("relation", ""),
                                    description=edge_data.get("description"),
                                    source_chunk_ids=[
                                        str(cid).replace("chunk:", "")
                                        for cid in edge_data.get("source_chunk_ids", [])
                                    ],
                                    weight=edge_data.get("weight", 1.0),
                                )
                            )
                            if target_id not in visited_node_ids:
                                next_level_node_ids.add(target_id)

                # Query incoming edges
                in_result = await self.db.query(
                    f"SELECT *, in AS source_id, out AS target_id FROM <-relation<-entity WHERE out = entity:{node_id}"
                )
                in_rows = _extract_rows(in_result)
                if in_rows:
                    for edge_data in in_rows:
                        edge_id = str(edge_data.get("id"))
                        if edge_id not in visited_edge_ids:
                            visited_edge_ids.add(edge_id)
                            target_id = str(edge_data.get("target_id", "")).replace("entity:", "")
                            source_id = str(edge_data.get("source_id", "")).replace("entity:", "")

                            edges_list.append(
                                Edge(
                                    source_id=source_id,
                                    target_id=target_id,
                                    relation=edge_data.get("relation", ""),
                                    description=edge_data.get("description"),
                                    source_chunk_ids=[
                                        str(cid).replace("chunk:", "")
                                        for cid in edge_data.get("source_chunk_ids", [])
                                    ],
                                    weight=edge_data.get("weight", 1.0),
                                )
                            )
                            if source_id not in visited_node_ids:
                                next_level_node_ids.add(source_id)

            current_level_node_ids = next_level_node_ids

        # Gather all nodes involved in the traversal
        all_node_ids_to_fetch = set(seed_node_ids)
        for edge in edges_list:
            all_node_ids_to_fetch.add(edge.source_id)
            all_node_ids_to_fetch.add(edge.target_id)

        for n_id in all_node_ids_to_fetch:
            if n_id not in nodes_dict:
                node_result = await self.db.query(f"SELECT * FROM entity:{n_id}")
                node_rows = _extract_rows(node_result)
                if node_rows:
                    n_data = node_rows[0]
                    nodes_dict[n_id] = Node(
                        id=n_id,
                        label=n_data.get("label", ""),
                        name=n_data.get("name", ""),
                        description=n_data.get("description"),
                        description_embedding=n_data.get("description_embedding"),
                        source_chunk_ids=[
                            str(cid).replace("chunk:", "")
                            for cid in n_data.get("source_chunk_ids", [])
                        ],
                    )

        return list(nodes_dict.values()), edges_list

    async def get_all_nodes(self) -> list[Node]:
        """Retrieve all nodes present in the knowledge graph.

        Returns:
            List of all Node domain models.
        """
        result = await self.db.query("SELECT * FROM entity;")
        return [
            Node(
                id=_clean_id(n_data.get("id")),
                label=n_data.get("label", ""),
                name=n_data.get("name", ""),
                description=n_data.get("description"),
                description_embedding=n_data.get("description_embedding"),
                source_chunk_ids=[_clean_id(cid) for cid in n_data.get("source_chunk_ids", [])],
            )
            for n_data in _extract_rows(result)
        ]

    async def get_all_edges(self) -> list[Edge]:
        """Retrieve all relationships present in the knowledge graph.

        Returns:
            List of all Edge domain models.
        """
        result = await self.db.query("SELECT *, in AS source_id, out AS target_id FROM relation;")
        return [
            Edge(
                source_id=_clean_id(edge_data.get("source_id")),
                target_id=_clean_id(edge_data.get("target_id")),
                relation=edge_data.get("relation", ""),
                description=edge_data.get("description"),
                source_chunk_ids=[_clean_id(cid) for cid in edge_data.get("source_chunk_ids", [])],
                weight=edge_data.get("weight", 1.0),
            )
            for edge_data in _extract_rows(result)
        ]

    async def save_community(self, community: Community) -> None:
        """Save a community and its summary to the database.

        Args:
            community: The community cluster domain model to persist.
        """
        await self.db.query(
            "UPDATE type::thing('community', $id) CONTENT { summary: $summary, node_ids: $node_ids };",
            {
                "id": community.id,
                "summary": community.summary,
                "node_ids": community.node_ids,
            },
        )
