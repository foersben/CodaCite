"""SurrealDB implementations for data stores."""

import logging
from typing import Any

from app.domain.models import Chunk, Community, Document, Edge, Node
from app.domain.ports import DocumentStore, GraphStore

logger = logging.getLogger(__name__)


def _extract_rows(result: Any) -> list[dict[str, Any]]:
    """Normalize SurrealDB query results across client response shapes."""
    if isinstance(result, list):
        if not result:
            return []

        first = result[0]
        # Old envelope format: [{"result": [...]}]
        if isinstance(first, dict) and "result" in first and isinstance(first.get("result"), list):
            nested = first.get("result")
            if isinstance(nested, list):
                return [row for row in nested if isinstance(row, dict)]

        # New format from surrealdb-python: directly a list of row dicts.
        return [row for row in result if isinstance(row, dict)]

    if isinstance(result, dict):
        nested = result.get("result")
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
        return [result]

    return []


class SurrealDocumentStore(DocumentStore):
    """SurrealDB implementation of DocumentStore."""

    def __init__(self, db: Any) -> None:
        """Initialize the document store."""
        self.db = db

    async def save_document(self, document: Document) -> None:
        """Save a document record."""
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
        """Save text chunks."""
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
        """Search for chunks by vector similarity."""
        result = await self.db.query(
            "SELECT * FROM chunk WHERE embedding <|5|> $embedding;",
            {"embedding": query_embedding},
        )
        rows = _extract_rows(result)
        if rows:
            chunks = []
            for item in rows:
                chunks.append(
                    Chunk(
                        id=str(item["id"]).replace("chunk:", ""),
                        document_id=str(item["document_id"]).replace("document:", ""),
                        text=item["text"],
                        index=item["index"],
                        embedding=item.get("embedding"),
                    )
                )
            return chunks[:top_k]
        return []

    async def get_all_documents(self) -> list[Document]:
        """Retrieve all ingested documents."""
        result = await self.db.query("SELECT * FROM document;")
        rows = _extract_rows(result)
        documents = []
        for row in rows:
            documents.append(
                Document(
                    id=str(row["id"]).replace("document:", ""),
                    filename=row.get("content", "unknown"),
                    metadata=row.get("metadata", {}),
                )
            )
        return documents

    async def initialize_schema(self) -> None:
        """Initialize SurrealDB schema with vector indices."""
        logger.info("Initializing SurrealDocumentStore schema")
        # Define MTREE index for chunk embeddings (dimension 1024 for bge-large-en-v1.5)
        await self.db.query(
            "DEFINE INDEX chunk_embedding_idx ON TABLE chunk FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;"
        )


class SurrealGraphStore(GraphStore):
    """SurrealDB implementation of GraphStore."""

    def __init__(self, db: Any) -> None:
        """Initialize the graph store."""
        self.db = db

    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save entity nodes to the graph."""
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
        """Initialize SurrealDB schema with vector indices."""
        logger.info("Initializing SurrealGraphStore schema")
        # Define MTREE index for entity description embeddings (dimension 1024)
        await self.db.query(
            "DEFINE INDEX entity_embedding_idx ON TABLE entity FIELDS description_embedding MTREE DIMENSION 1024 DIST COSINE;"
        )

    async def save_edges(self, edges: list[Edge]) -> None:
        """Save relationship edges to the graph."""
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
        """Traverse the graph starting from seed nodes up to a given depth."""
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
                                    source_chunk_ids=[str(cid).replace("chunk:", "") for cid in edge_data.get("source_chunk_ids", [])],
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
                                    source_chunk_ids=[str(cid).replace("chunk:", "") for cid in edge_data.get("source_chunk_ids", [])],
                                    weight=edge_data.get("weight", 1.0),
                                )
                            )
                            if source_id not in visited_node_ids:
                                next_level_node_ids.add(source_id)

            current_level_node_ids = next_level_node_ids

        # Gather all nodes involved
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
                        source_chunk_ids=[str(cid).replace("chunk:", "") for cid in n_data.get("source_chunk_ids", [])],
                    )

        return list(nodes_dict.values()), edges_list

    async def get_all_nodes(self) -> list[Node]:
        """Retrieve all nodes."""
        result = await self.db.query("SELECT * FROM entity;")
        nodes = []
        for n_data in _extract_rows(result):
            nodes.append(
                Node(
                    id=str(n_data.get("id")).replace("entity:", ""),
                    label=n_data.get("label", ""),
                    name=n_data.get("name", ""),
                    description=n_data.get("description"),
                    description_embedding=n_data.get("description_embedding"),
                    source_chunk_ids=[str(cid).replace("chunk:", "") for cid in n_data.get("source_chunk_ids", [])],
                )
            )
        return nodes

    async def get_all_edges(self) -> list[Edge]:
        """Retrieve all edges."""
        result = await self.db.query("SELECT *, in AS source_id, out AS target_id FROM relation;")
        edges = []
        for edge_data in _extract_rows(result):
            edges.append(
                Edge(
                    source_id=str(edge_data.get("source_id", "")).replace("entity:", ""),
                    target_id=str(edge_data.get("target_id", "")).replace("entity:", ""),
                    relation=edge_data.get("relation", ""),
                    description=edge_data.get("description"),
                    source_chunk_ids=[str(cid).replace("chunk:", "") for cid in edge_data.get("source_chunk_ids", [])],
                    weight=edge_data.get("weight", 1.0),
                )
            )
        return edges

    async def save_community(self, community: Community) -> None:
        """Save a community and its summary."""
        await self.db.query(
            "UPDATE type::thing('community', $id) CONTENT { summary: $summary, node_ids: $node_ids };",
            {
                "id": community.id,
                "summary": community.summary,
                "node_ids": community.node_ids,
            },
        )
