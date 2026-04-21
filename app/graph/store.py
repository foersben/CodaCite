"""SurrealDB-backed graph and vector store."""

from __future__ import annotations

from typing import Any

from app.graph.extractor import Entity, Relationship


class GraphStore:
    """Persists document chunks, entities, relationships, and vector embeddings in SurrealDB.

    The *db* argument accepts any object with an async ``.query(sql, vars)``
    method (typically a connected :class:`surrealdb.Surreal` instance or a mock).

    Args:
        db: An open SurrealDB connection.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def store_chunk(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        source: str,
    ) -> None:
        """Upsert a document chunk with its vector embedding.

        Args:
            chunk_id: Unique identifier (e.g. ``"chunk:abc123"``).
            text: The raw chunk text.
            embedding: Dense vector representation of the chunk.
            source: Original document path/URI.
        """
        sql = (
            "CREATE type::thing($chunk_id) CONTENT {"
            "  text: $text,"
            "  embedding: $embedding,"
            "  source: $source"
            "};"
        )
        await self._db.query(
            sql,
            {
                "chunk_id": chunk_id,
                "text": text,
                "embedding": embedding,
                "source": source,
            },
        )

    async def store_entity(self, entity: Entity) -> None:
        """Upsert a named entity node.

        Args:
            entity: The :class:`~app.graph.extractor.Entity` to persist.
        """
        sql = (
            "CREATE entity CONTENT {"
            "  name: $name,"
            "  entity_type: $entity_type,"
            "  description: $description"
            "};"
        )
        await self._db.query(
            sql,
            {
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
            },
        )

    async def store_relationship(self, relationship: Relationship) -> None:
        """Upsert a directed relationship edge between two entities.

        Args:
            relationship: The :class:`~app.graph.extractor.Relationship` to persist.
        """
        sql = (
            "CREATE relation CONTENT {"
            "  source: $source,"
            "  target: $target,"
            "  relation: $relation"
            "};"
        )
        await self._db.query(
            sql,
            {
                "source": relationship.source,
                "target": relationship.target,
                "relation": relationship.relation,
            },
        )

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform a cosine similarity vector search over chunk embeddings.

        Args:
            query_embedding: The query vector.
            top_k: Number of top results to return.

        Returns:
            A list of result dicts (may be empty if the table is empty).
        """
        sql = (
            "SELECT id, text, source, "
            "  vector::similarity::cosine(embedding, $query) AS score "
            "FROM chunk "
            "ORDER BY score DESC "
            "LIMIT $top_k;"
        )
        raw = await self._db.query(sql, {"query": query_embedding, "top_k": top_k})
        if raw and isinstance(raw[0], list):
            return raw[0]
        return raw or []

    async def traverse_graph(
        self,
        start_node_id: str,
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Traverse the knowledge graph from *start_node_id* up to *depth* hops.

        Args:
            start_node_id: The SurrealDB record ID to start traversal from.
            depth: Maximum traversal depth (1 or 2 recommended).

        Returns:
            A list of connected node dicts.
        """
        sql = (
            "SELECT * FROM $start_id->relation->(entity WHERE true) "
            "FETCH id, name, entity_type, description "
            "LIMIT 50;"
        )
        raw = await self._db.query(sql, {"start_id": start_node_id})
        if raw and isinstance(raw[0], list):
            return raw[0]
        return raw or []
