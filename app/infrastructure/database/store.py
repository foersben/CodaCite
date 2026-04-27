"""SurrealDB implementations for data stores.

This module provides concrete implementations of the DocumentStore and GraphStore
ports using SurrealDB. It includes logic for type harmonization between
SurrealDB RecordIDs and pure Pydantic domain models.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from surrealdb import RecordID, Value
from surrealdb.connections.async_embedded import AsyncEmbeddedSurrealConnection
from surrealdb.connections.async_http import AsyncHttpSurrealConnection
from surrealdb.connections.async_ws import AsyncWsSurrealConnection

from app.domain.models import Chunk, Community, Document, Edge, Node, Notebook
from app.domain.ports import DocumentStore, GraphStore

logger = logging.getLogger(__name__)

AsyncSurrealType = (
    AsyncWsSurrealConnection | AsyncHttpSurrealConnection | AsyncEmbeddedSurrealConnection
)


def _extract_rows(result: object) -> list[dict[str, object]]:
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


def _clean_id(id_val: object) -> str:
    """Strip SurrealDB table prefix from RecordID string."""
    id_str = str(id_val)
    return id_str.split(":", 1)[-1] if ":" in id_str else id_str


class SurrealDocumentStore(DocumentStore):
    """SurrealDB implementation of DocumentStore.

    Handles the storage and similarity search of document chunks using
    SurrealDB's native vector indexing (HNSW).
    """

    db: AsyncSurrealType

    def __init__(self, db: AsyncSurrealType) -> None:
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
            "UPDATE $id CONTENT { filename: $filename, file_path: $file_path, status: $status, metadata: $metadata };",
            {
                "id": RecordID("document", document.id),
                "filename": document.filename,
                "file_path": document.metadata.get("file_path", ""),
                "status": document.status,
                "metadata": cast("Value", document.metadata),
            },
        )

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save text chunks to the database and relate them to their document.

        Args:
            chunks: List of chunk domain models to persist.
        """
        logger.info("Saving %d chunks to SurrealDB with contains relations", len(chunks))
        for chunk in chunks:
            # 1. Update/Create the chunk
            await self.db.query(
                "UPDATE $chunk_id CONTENT { text: $text, index: $index, embedding: $embedding };",
                {
                    "chunk_id": RecordID("chunk", chunk.id),
                    "text": chunk.text,
                    "index": chunk.index,
                    "embedding": cast("Value", chunk.embedding),
                },
            )
            # 2. Create the relationship
            await self.db.query(
                "RELATE $doc -> contains -> $chunk UNIQUE;",
                {
                    "doc": RecordID("document", chunk.document_id),
                    "chunk": RecordID("chunk", chunk.id),
                },
            )

    async def search_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        active_notebook_ids: list[str] | None = None,
    ) -> list[Chunk]:
        """Search for chunks by vector similarity with optimal pre-filtering.

        Args:
            query_embedding: The vector to search for.
            top_k: Maximum number of results to return.
            active_notebook_ids: Optional list of notebook IDs to filter by.

        Returns:
            A list of similar Chunk models.
        """
        if active_notebook_ids:
            # Optimal Graph Traversal Filter
            query = """
            SELECT *, vector::similarity::cosine(embedding, $embedding) AS score
            FROM chunk
            WHERE <-contains<-document->belongs_to->notebook.id CONTAINSANY $notebook_ids
            ORDER BY score DESC LIMIT $top_k;
            """
            params = {
                "embedding": query_embedding,
                "notebook_ids": [f"notebook:{nid}" for nid in active_notebook_ids],
                "top_k": top_k,
            }
        else:
            # Unfiltered search
            query = "SELECT * FROM chunk WHERE embedding <|5|> $embedding;"
            params = {"embedding": query_embedding}

        result = await self.db.query(query, cast("dict[str, Value]", params))
        rows = _extract_rows(result)
        return [
            Chunk(
                id=_clean_id(item["id"]),
                document_id=_clean_id(item["document_id"]) if "document_id" in item else "",
                text=cast(str, item["text"]),
                index=cast(int, item["index"]),
                embedding=cast("list[float] | None", item.get("embedding")),
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
                filename=cast(str, row.get("filename", "unknown")),
                status=cast(str, row.get("status", "active")),
                metadata=cast(dict[str, str | int | float | bool], row.get("metadata", {})),
            )
            for row in _extract_rows(result)
        ]

    async def update_document_status(self, document_id: str, status: str) -> None:
        """Update the processing status of a document.

        Args:
            document_id: The ID of the document to update.
            status: The new status string.
        """
        await self.db.query(
            "UPDATE type::thing('document', $id) SET status = $status;",
            {"id": document_id, "status": status},
        )

    async def add_document_to_notebook(self, document_id: str, notebook_id: str) -> None:
        """Relate a document to a notebook using a graph edge.

        Args:
            document_id: The ID of the document.
            notebook_id: The ID of the notebook.
        """
        await self.db.query(
            "RELATE $doc -> belongs_to -> $notebook;",
            {
                "doc": RecordID("document", document_id),
                "notebook": RecordID("notebook", notebook_id),
            },
        )

    async def remove_document_from_notebook(self, document_id: str, notebook_id: str) -> None:
        """Remove a relationship between a document and a notebook.

        Args:
            document_id: The ID of the document.
            notebook_id: The ID of the notebook.
        """
        await self.db.query(
            "DELETE belongs_to WHERE in = $doc AND out = $notebook;",
            {
                "doc": RecordID("document", document_id),
                "notebook": RecordID("notebook", notebook_id),
            },
        )

    async def get_notebook_documents(self, notebook_id: str) -> list[Document]:
        """Retrieve all documents associated with a specific notebook.

        Args:
            notebook_id: The notebook ID.

        Returns:
            A list of Document objects linked to the notebook.
        """
        query = "SELECT * FROM document WHERE ->belongs_to->notebook.id CONTAINS $notebook;"
        result = await self.db.query(query, {"notebook": RecordID("notebook", notebook_id)})
        return [
            Document(
                id=_clean_id(row["id"]),
                filename=cast(str, row.get("filename", "unknown")),
                status=cast(str, row.get("status", "active")),
                metadata=cast(dict[str, str | int | float | bool], row.get("metadata", {})),
            )
            for row in _extract_rows(result)
        ]

    async def delete_document(self, document_id: str) -> None:
        """Delete a document and its chunks, then perform maintenance.

        Increments a persistent deletion counter and triggers an HNSW index
        rebuild every 5 deletions to clear tombstones.

        Args:
            document_id: The ID of the document to remove.
        """
        logger.info("Deleting document %s and performing maintenance check", document_id)

        maintenance_query = """
        BEGIN TRANSACTION;
        -- 1. Delete chunks and the document
        DELETE chunk WHERE document_id = $id;
        DELETE type::thing('document', $id);

        -- 2. Update maintenance counter
        LET $current = (SELECT VALUE count FROM maintenance:counts WHERE id = 'deletions')[0] OR 0;
        LET $new_count = $current + 1;
        UPSERT maintenance:counts SET count = $new_count WHERE id = 'deletions';

        -- 3. Check for maintenance threshold (5)
        IF $new_count >= 5 {
            REBUILD INDEX chunk_embedding_idx ON TABLE chunk;
            UPDATE maintenance:counts SET count = 0 WHERE id = 'deletions';
        };
        COMMIT TRANSACTION;
        """

        await self.db.query(maintenance_query, {"id": document_id})

    async def initialize_schema(self) -> None:
        """Initialize SurrealDB table indices for document storage.

        Defines the HNSW vector index for chunk embeddings.
        """
        logger.info("Initializing SurrealDocumentStore schema")
        await self.db.query(
            "DEFINE INDEX chunk_embedding_idx ON TABLE chunk FIELDS embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32;"
        )

    async def save_notebook(self, notebook: Notebook) -> None:
        """Save a notebook record to the database.

        Args:
            notebook: The notebook domain model to persist.
        """
        logger.info("Saving notebook to SurrealDB: %s", notebook.title)
        await self.db.query(
            "UPDATE type::thing('notebook', $id) CONTENT { title: $title, description: $description, created_at: $created_at };",
            {
                "id": notebook.id,
                "title": notebook.title,
                "description": notebook.description,
                "created_at": notebook.created_at,
            },
        )

    async def get_all_notebooks(self) -> list[Notebook]:
        """Retrieve all notebooks from the store.

        Returns:
            List of all Notebook domain models.
        """
        result = await self.db.query("SELECT * FROM notebook;")
        return [
            Notebook(
                id=_clean_id(row["id"]),
                title=cast(str, row.get("title", "Untitled")),
                description=cast(str | None, row.get("description")),
                created_at=cast(str | None, row.get("created_at")),
            )
            for row in _extract_rows(result)
        ]

    async def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook and its document relations.

        Args:
            notebook_id: The ID of the notebook to remove.
        """
        logger.info("Deleting notebook %s and its relations", notebook_id)
        # 1. Delete the notebook record
        # 2. Relations (belongs_to edges) are automatically handled if we use graph deletes
        # but let's be explicit
        query = """
        BEGIN TRANSACTION;
        DELETE belongs_to WHERE out = $id;
        DELETE $id;
        COMMIT TRANSACTION;
        """
        await self.db.query(query, {"id": RecordID("notebook", notebook_id)})


class SurrealGraphStore(GraphStore):
    """SurrealDB implementation of GraphStore.

    Handles entity and relationship storage, providing graph traversal
    capabilities for context retrieval.
    """

    db: AsyncSurrealType

    def __init__(self, db: AsyncSurrealType) -> None:
        """Initialize the graph store.

        Args:
            db: An active SurrealDB client instance.
        """
        self.db = db

    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save entity nodes and relate them to their source chunks.

        Args:
            nodes: List of Node domain models to persist.
        """
        logger.info("Saving %d nodes to SurrealDB Graph with extracted_from relations", len(nodes))
        for node in nodes:
            # 1. Update/Create the entity node
            await self.db.query(
                "UPDATE $id CONTENT { label: $label, name: $name, description: $description, description_embedding: $description_embedding };",
                {
                    "id": RecordID("entity", node.id),
                    "label": node.label,
                    "name": node.name,
                    "description": node.description,
                    "description_embedding": cast("Value", node.description_embedding),
                },
            )
            # 2. Relate to source chunks
            await self.db.query(
                "FOR $cid IN $cids { RELATE $entity -> extracted_from -> $cid UNIQUE; };",
                {
                    "entity": RecordID("entity", node.id),
                    "cids": cast(
                        "Value", [RecordID("chunk", cid) for cid in node.source_chunk_ids]
                    ),
                },
            )

    async def initialize_schema(self) -> None:
        """Initialize SurrealDB table indices for graph storage.

        Defines the HNSW vector index for entity description embeddings.
        """
        logger.info("Initializing SurrealGraphStore schema")
        await self.db.query(
            "DEFINE INDEX entity_embedding_idx ON TABLE entity FIELDS description_embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32;"
        )

    async def save_edges(self, edges: list[Edge]) -> None:
        """Save relationship edges between entities.

        Args:
            edges: List of Edge domain models to persist.
        """
        logger.info("Saving %d edges to SurrealDB Graph", len(edges))
        for edge in edges:
            await self.db.query(
                "RELATE $s->relation->$t CONTENT { relation: $relation, description: $description, source_chunk_ids: $source_chunk_ids, weight: $weight };",
                {
                    "s": RecordID("entity", edge.source_id),
                    "t": RecordID("entity", edge.target_id),
                    "relation": edge.relation,
                    "description": edge.description,
                    "source_chunk_ids": [RecordID("chunk", cid) for cid in edge.source_chunk_ids],
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
                    "SELECT *, in AS source_id, out AS target_id FROM $node->relation",
                    {"node": RecordID("entity", node_id)},
                )
                out_rows = _extract_rows(out_result)
                if out_rows:
                    for edge_data in out_rows:
                        edge_id = str(edge_data.get("id"))
                        if edge_id not in visited_edge_ids:
                            visited_edge_ids.add(edge_id)
                            target_id = str(edge_data.get("target_id", "")).replace("entity:", "")
                            source_id = str(edge_data.get("source_id", "")).replace("entity:", "")

                            raw_chunk_ids = edge_data.get("source_chunk_ids")
                            chunk_ids = []
                            if isinstance(raw_chunk_ids, list):
                                chunk_ids = [_clean_id(cid) for cid in raw_chunk_ids]

                            edges_list.append(
                                Edge(
                                    source_id=source_id,
                                    target_id=target_id,
                                    relation=str(edge_data.get("relation", "")),
                                    description=str(edge_data.get("description"))
                                    if edge_data.get("description")
                                    else None,
                                    source_chunk_ids=chunk_ids,
                                    weight=float(cast(Any, edge_data.get("weight", 1.0)))
                                    if edge_data.get("weight") is not None
                                    else 1.0,
                                )
                            )
                            if target_id not in visited_node_ids:
                                next_level_node_ids.add(target_id)

                # Query incoming edges
                in_result = await self.db.query(
                    "SELECT *, in AS source_id, out AS target_id FROM <-relation<-entity WHERE out = $node",
                    {"node": RecordID("entity", node_id)},
                )
                in_rows = _extract_rows(in_result)
                if in_rows:
                    for edge_data in in_rows:
                        edge_id = str(edge_data.get("id"))
                        if edge_id not in visited_edge_ids:
                            visited_edge_ids.add(edge_id)
                            target_id = str(edge_data.get("target_id", "")).replace("entity:", "")
                            source_id = str(edge_data.get("source_id", "")).replace("entity:", "")

                            raw_chunk_ids = edge_data.get("source_chunk_ids")
                            chunk_ids = []
                            if isinstance(raw_chunk_ids, list):
                                chunk_ids = [_clean_id(cid) for cid in raw_chunk_ids]

                            edges_list.append(
                                Edge(
                                    source_id=source_id,
                                    target_id=target_id,
                                    relation=str(edge_data.get("relation", "")),
                                    description=str(edge_data.get("description"))
                                    if edge_data.get("description")
                                    else None,
                                    source_chunk_ids=chunk_ids,
                                    weight=float(cast(Any, edge_data.get("weight", 1.0)))
                                    if edge_data.get("weight") is not None
                                    else 1.0,
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
                    raw_chunk_ids = n_data.get("source_chunk_ids")
                    chunk_ids = []
                    if isinstance(raw_chunk_ids, list):
                        chunk_ids = [_clean_id(cid) for cid in raw_chunk_ids]

                    nodes_dict[n_id] = Node(
                        id=n_id,
                        label=str(n_data.get("label", "")),
                        name=str(n_data.get("name", "")),
                        description=str(n_data.get("description"))
                        if n_data.get("description")
                        else None,
                        description_embedding=cast(list[float], n_data.get("description_embedding"))
                        if isinstance(n_data.get("description_embedding"), list)
                        else None,
                        source_chunk_ids=chunk_ids,
                    )

        return list(nodes_dict.values()), edges_list

    async def get_all_nodes(self) -> list[Node]:
        """Retrieve all nodes present in the knowledge graph.

        Returns:
            List of all Node domain models.
        """
        result = await self.db.query("SELECT * FROM entity;")
        nodes = []
        for n_data in _extract_rows(result):
            raw_chunk_ids = n_data.get("source_chunk_ids")
            chunk_ids = []
            if isinstance(raw_chunk_ids, list):
                chunk_ids = [_clean_id(cid) for cid in raw_chunk_ids]

            nodes.append(
                Node(
                    id=_clean_id(n_data.get("id")),
                    label=str(n_data.get("label", "")),
                    name=str(n_data.get("name", "")),
                    description=cast(str | None, n_data.get("description"))
                    if n_data.get("description") is None
                    or isinstance(n_data.get("description"), str)
                    else str(n_data.get("description")),
                    description_embedding=cast(list[float], n_data.get("description_embedding"))
                    if isinstance(n_data.get("description_embedding"), list)
                    else None,
                    source_chunk_ids=chunk_ids,
                )
            )
        return nodes

    async def get_all_edges(self) -> list[Edge]:
        """Retrieve all relationships present in the knowledge graph.

        Returns:
            List of all Edge domain models.
        """
        result = await self.db.query("SELECT *, in AS source_id, out AS target_id FROM relation;")
        edges = []
        for edge_data in _extract_rows(result):
            raw_chunk_ids = edge_data.get("source_chunk_ids")
            chunk_ids = []
            if isinstance(raw_chunk_ids, list):
                chunk_ids = [_clean_id(cid) for cid in raw_chunk_ids]

            edges.append(
                Edge(
                    source_id=_clean_id(edge_data.get("source_id")),
                    target_id=_clean_id(edge_data.get("target_id")),
                    relation=str(edge_data.get("relation", "")),
                    description=str(edge_data.get("description"))
                    if edge_data.get("description")
                    else None,
                    source_chunk_ids=chunk_ids,
                    weight=float(cast(Any, edge_data.get("weight", 1.0)))
                    if edge_data.get("weight") is not None
                    else 1.0,
                )
            )
        return edges

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
                "node_ids": cast("Value", community.node_ids),
            },
        )
