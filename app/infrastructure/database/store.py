"""SurrealDB implementations for data stores."""

import logging
from typing import Any

from app.domain.models import Chunk, Community, Document, Edge, Node
from app.domain.ports import DocumentStore, GraphStore

logger = logging.getLogger(__name__)


class SurrealDocumentStore(DocumentStore):
    """SurrealDB implementation of DocumentStore."""

    def __init__(self, db: Any) -> None:
        """Initialize the document store."""
        self.db = db

    async def save_document(self, document: Document) -> None:
        """Save a document record."""
        await self.db.query(
            "CREATE document CONTENT { id: $id, content: $content, metadata: $metadata };",
            {
                "id": document.id,
                # Note: `document` has no `text` attribute, only `filename` and `metadata`
                # Assuming `content` should just be empty for now or we save filename.
                "content": document.filename,
                "metadata": document.metadata,
            },
        )

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save text chunks."""
        for chunk in chunks:
            await self.db.query(
                "CREATE chunk CONTENT { id: $id, document_id: $document_id, text: $text, index: $index, embedding: $embedding };",
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
        if result and len(result) > 0 and len(result[0].get("result", [])) > 0:
            chunks = []
            for item in result[0]["result"]:
                chunks.append(
                    Chunk(
                        id=item["id"],
                        document_id=item["document_id"],
                        text=item["text"],
                        index=item["index"],
                        embedding=item["embedding"],
                    )
                )
            return chunks[:top_k]
        return []


class SurrealGraphStore(GraphStore):
    """SurrealDB implementation of GraphStore."""

    def __init__(self, db: Any) -> None:
        """Initialize the graph store."""
        self.db = db

    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save entity nodes to the graph."""
        for node in nodes:
            await self.db.query(
                "UPSERT entity CONTENT { id: $id, label: $label, name: $name, description: $description, description_embedding: $description_embedding, source_chunk_ids: $source_chunk_ids };",
                {
                    "id": node.id,
                    "label": node.label,
                    "name": node.name,
                    "description": node.description,
                    "description_embedding": node.description_embedding,
                    "source_chunk_ids": node.source_chunk_ids,
                },
            )

    async def save_edges(self, edges: list[Edge]) -> None:
        """Save relationship edges to the graph."""
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
        # Simple implementation for depth 1 for now
        nodes_dict = {}
        edges_list = []

        for seed_id in seed_node_ids:
            # Query outgoing edges
            out_result = await self.db.query(
                f"SELECT *, in AS source_id, out AS target_id FROM entity:{seed_id}->relation"
            )
            if out_result and len(out_result) > 0 and isinstance(out_result[0].get("result"), list):
                for edge_data in out_result[0]["result"]:
                    edges_list.append(
                        Edge(
                            source_id=str(edge_data.get("source_id", "")).replace("entity:", ""),
                            target_id=str(edge_data.get("target_id", "")).replace("entity:", ""),
                            relation=edge_data.get("relation", ""),
                            description=edge_data.get("description"),
                            source_chunk_ids=edge_data.get("source_chunk_ids", []),
                            weight=edge_data.get("weight", 1.0),
                        )
                    )

            # Query incoming edges
            in_result = await self.db.query(
                f"SELECT *, in AS source_id, out AS target_id FROM <-relation<-entity WHERE out = entity:{seed_id}"
            )
            if in_result and len(in_result) > 0 and isinstance(in_result[0].get("result"), list):
                for edge_data in in_result[0]["result"]:
                    edges_list.append(
                        Edge(
                            source_id=str(edge_data.get("source_id", "")).replace("entity:", ""),
                            target_id=str(edge_data.get("target_id", "")).replace("entity:", ""),
                            relation=edge_data.get("relation", ""),
                            description=edge_data.get("description"),
                            source_chunk_ids=edge_data.get("source_chunk_ids", []),
                            weight=edge_data.get("weight", 1.0),
                        )
                    )

            # Query the seed node itself
            node_result = await self.db.query(f"SELECT * FROM entity:{seed_id}")
            if (
                node_result
                and len(node_result) > 0
                and isinstance(node_result[0].get("result"), list)
                and len(node_result[0]["result"]) > 0
            ):
                n_data = node_result[0]["result"][0]
                n_id = str(n_data.get("id")).replace("entity:", "")
                if n_id not in nodes_dict:
                    nodes_dict[n_id] = Node(
                        id=n_id,
                        label=n_data.get("label", ""),
                        name=n_data.get("name", ""),
                        description=n_data.get("description"),
                        description_embedding=n_data.get("description_embedding"),
                        source_chunk_ids=n_data.get("source_chunk_ids", []),
                    )

        # Collect target nodes from edges
        for edge in edges_list:
            for n_id in [edge.source_id, edge.target_id]:
                if n_id not in nodes_dict:
                    node_result = await self.db.query(f"SELECT * FROM entity:{n_id}")
                    if (
                        node_result
                        and len(node_result) > 0
                        and isinstance(node_result[0].get("result"), list)
                        and len(node_result[0]["result"]) > 0
                    ):
                        n_data = node_result[0]["result"][0]
                        nodes_dict[n_id] = Node(
                            id=n_id,
                            label=n_data.get("label", ""),
                            name=n_data.get("name", ""),
                            description=n_data.get("description"),
                            description_embedding=n_data.get("description_embedding"),
                            source_chunk_ids=n_data.get("source_chunk_ids", []),
                        )

        return list(nodes_dict.values()), edges_list

    async def get_all_nodes(self) -> list[Node]:
        """Retrieve all nodes."""
        result = await self.db.query("SELECT * FROM entity;")
        nodes = []
        if result and len(result) > 0 and isinstance(result[0].get("result"), list):
            for n_data in result[0]["result"]:
                nodes.append(
                    Node(
                        id=str(n_data.get("id")).replace("entity:", ""),
                        label=n_data.get("label", ""),
                        name=n_data.get("name", ""),
                        description=n_data.get("description"),
                        description_embedding=n_data.get("description_embedding"),
                        source_chunk_ids=n_data.get("source_chunk_ids", []),
                    )
                )
        return nodes

    async def get_all_edges(self) -> list[Edge]:
        """Retrieve all edges."""
        result = await self.db.query("SELECT *, in AS source_id, out AS target_id FROM relation;")
        edges = []
        if result and len(result) > 0 and isinstance(result[0].get("result"), list):
            for edge_data in result[0]["result"]:
                edges.append(
                    Edge(
                        source_id=str(edge_data.get("source_id", "")).replace("entity:", ""),
                        target_id=str(edge_data.get("target_id", "")).replace("entity:", ""),
                        relation=edge_data.get("relation", ""),
                        description=edge_data.get("description"),
                        source_chunk_ids=edge_data.get("source_chunk_ids", []),
                        weight=edge_data.get("weight", 1.0),
                    )
                )
        return edges

    async def save_community(self, community: Community) -> None:
        """Save a community and its summary."""
        await self.db.query(
            "CREATE community CONTENT { id: $id, summary: $summary, node_ids: $node_ids };",
            {
                "id": community.id,
                "summary": community.summary,
                "node_ids": community.node_ids,
            },
        )
