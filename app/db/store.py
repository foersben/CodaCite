"""SurrealDB implementations for data stores.

This module provides concrete implementations of the DocumentStore and GraphStore
ports using SurrealDB. It includes logic for type harmonization between
SurrealDB RecordIDs and pure Pydantic domain models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeAlias, cast

from surrealdb import RecordID, Value
from surrealdb.connections.async_embedded import AsyncEmbeddedSurrealConnection
from surrealdb.connections.async_http import AsyncHttpSurrealConnection
from surrealdb.connections.async_ws import AsyncWsSurrealConnection

from app.core.interfaces import DocumentStore, GraphStore
from app.db.schema import get_schema_queries
from app.models.models import Chunk, Community, Document, Edge, Node, Notebook

logger = logging.getLogger(__name__)

AsyncSurrealType: TypeAlias = (  # noqa: UP040
    AsyncWsSurrealConnection | AsyncHttpSurrealConnection | AsyncEmbeddedSurrealConnection
)


def _extract_rows(result: object) -> list[dict[str, object]]:
    """Normalize SurrealDB query results for SurrealDB 3.x driver."""
    if not result:
        return []

    # db.query() in v3 returns a list of result sets (one per statement)
    if isinstance(result, list):
        all_rows: list[dict[str, object]] = []
        for stmt_result in result:
            if stmt_result is None:
                continue

            # If the statement result is a list of rows
            if isinstance(stmt_result, list):
                for row in stmt_result:
                    if isinstance(row, dict):
                        all_rows.append(row)
            # If the statement result is a single dict row
            elif isinstance(stmt_result, dict):
                all_rows.append(stmt_result)
        return all_rows

    # Single statement/direct result case
    if isinstance(result, dict):
        return [result]

    return []


def _clean_id(id_val: object) -> str:
    """Strip SurrealDB table prefix from RecordID string or handle RecordID objects."""
    if id_val is None:
        return ""
    if hasattr(id_val, "id"):
        # Modern RecordID object
        return str(id_val.id)  # type: ignore[attr-defined]
    id_str = str(id_val)
    return id_str.split(":", 1)[-1] if ":" in id_str else id_str


class SurrealDocumentStore(DocumentStore):
    """SurrealDB implementation of DocumentStore.

    Handles storage of files and their semantic chunks. Uses SurrealDB's
    native HNSW indices for high-performance vector search and BM25 indices
    for keyword-based full-text search.

    Pipeline Role:
        Phase 4: Persistence. Persistent storage of document metadata and
        vector chunks.

    Indexing Concept:
        Uses a **Hybrid Indexing** strategy:
        - HNSW vector index (1024D, Cosine) for semantic similarity.
        - BM25 full-text index on the `text` column for keyword matching.
        - Notebook isolation is enforced via Graph Scoping (`belongs_to` edges).
        Hybrid scoring: ``score = (bm25_score * alpha) + (cosine_sim * (1 - alpha))``.

    Implementation Details:
        - Harmonizes SurrealDB's `RecordID` with Pydantic domain models using
          `type::record` casting and manual ID cleaning.
        - Implements graph-aware hybrid search: retrieves chunks where the
          `chunk <- contains <- document -> belongs_to -> notebook` path exists.
        - Includes automatic index maintenance (REBUILD) after deletions to
          handle HNSW tombstones.
    """

    db: AsyncSurrealType

    def __init__(self, db: AsyncSurrealType) -> None:
        """Initialize the document store.

        Args:
            db: An active SurrealDB client instance (WS, HTTP, or Embedded).
        """
        self.db = db

    async def save_document(self, document: Document) -> None:
        """Save a document record to the database.

        Args:
            document: The domain document model to persist.
        """
        logger.info("Saving document to SurrealDB: %s", document.filename)
        await self.db.query(
            "UPSERT $id CONTENT { filename: $filename, file_path: $file_path, status: $status, metadata: $metadata };",
            {
                "id": RecordID("document", document.id),
                "filename": document.filename,
                "file_path": document.metadata.get("file_path", ""),
                "status": document.status,
                "metadata": cast("Value", document.metadata),
            },
        )

    async def save_document_with_summary(self, document_id: str, summary: str) -> None:
        """Updates a document record with its pre-computed global summary."""
        query = """
        UPDATE type::record('document', $doc_id)
        MERGE {
            global_summary: $summary
        };
        """
        await self.db.query(query, {"doc_id": document_id, "summary": summary})

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save text chunks to the database and relate them to their document.

        Establishes `document -> contains -> chunk` edges.

        Args:
            chunks: List of chunk domain models to persist.
        """
        logger.info("Saving %d chunks to SurrealDB with contains relations", len(chunks))
        for chunk in chunks:
            await self.db.query(
                "UPSERT $chunk_id CONTENT { document_id: $document_id, text: $text, index: $index, embedding: $embedding };",
                {
                    "chunk_id": RecordID("chunk", chunk.id),
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "index": chunk.index,
                    "embedding": cast("Value", chunk.embedding),
                },
            )
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
        query_text: str | None = None,
        alpha: float = 0.5,
        top_k: int = 5,
        active_notebook_ids: list[str] | None = None,
    ) -> list[Chunk]:
        """Search for chunks using hybrid BM25 + HNSW vector similarity search.

        When `query_text` is provided, performs a hybrid search that computes a
        combined score::

            final_score = (search::score(1) * alpha)
                        + (vector::similarity::cosine(embedding, $embedding) * (1 - alpha))

        When `query_text` is None or empty, falls back to a pure HNSW vector search.

        If `active_notebook_ids` is provided, performs a multi-hop traversal in the
        WHERE clause to ensure retrieved chunks belong to specific notebooks.

        Args:
            query_embedding: The vector to search for.
            query_text: Optional raw query text for BM25 full-text matching.
            alpha: BM25/vector weighting in [0, 1]. 1.0 = pure BM25; 0.0 = pure vector.
            top_k: Maximum number of results to return.
            active_notebook_ids: Optional list of notebook IDs to filter by.

        Returns:
            A list of similar Chunk models, ordered by combined score descending.
        """
        k = int(top_k)
        use_hybrid = bool(query_text)

        logger.info(
            "[STORE] Searching chunks (Hybrid=%s, k=%d, notebooks=%s)",
            use_hybrid,
            top_k,
            active_notebook_ids,
        )
        # Diagnostic: Check if any chunks exist
        count_res = await self.db.query("SELECT count() FROM chunk GROUP ALL;")
        count_rows = _extract_rows(count_res)
        logger.info(
            "[STORE] Current chunk count: %s", count_rows[0].get("count") if count_rows else 0
        )

        query: str = ""
        params: dict[str, object] = {}

        if use_hybrid:
            if active_notebook_ids and len(active_notebook_ids) > 0:
                query = f"""
                SELECT *,
                       document_id,
                       (vector::similarity::cosine(embedding, $embedding) * (1.0 - $alpha))
                       + ( (search::score(1) OR 0) * $alpha )
                       AS hybrid_score
                FROM chunk
                WHERE (embedding <|{k},150|> $embedding OR text @1@ $query_text)
                AND   (<-contains<-document->belongs_to->notebook.id CONTAINSANY $notebook_ids)
                ORDER BY hybrid_score DESC
                LIMIT {k};
                """
                params = {
                    "embedding": query_embedding,
                    "query_text": query_text,
                    "alpha": alpha,
                    "notebook_ids": [f"notebook:{nid}" for nid in active_notebook_ids],
                }
            else:
                query = f"""
                SELECT *,
                       document_id,
                       (vector::similarity::cosine(embedding, $embedding) * (1.0 - $alpha))
                       + ( (search::score(1) OR 0) * $alpha )
                       AS hybrid_score
                FROM chunk
                WHERE (embedding <|{k},150|> $embedding OR text @1@ $query_text)
                ORDER BY hybrid_score DESC
                LIMIT {k};
                """
                params = {
                    "embedding": query_embedding,
                    "query_text": query_text,
                    "alpha": alpha,
                }
        else:
            if active_notebook_ids and len(active_notebook_ids) > 0:
                query = f"SELECT *, document_id FROM chunk WHERE embedding <|{k},150|> $embedding AND (<-contains<-document->belongs_to->notebook.id CONTAINSANY $notebook_ids) LIMIT {k};"
                params = {
                    "embedding": query_embedding,
                    "notebook_ids": [f"notebook:{nid}" for nid in active_notebook_ids],
                }
            else:
                query = f"SELECT *, document_id FROM chunk WHERE embedding <|{k},150|> $embedding LIMIT {k};"
                params = {"embedding": query_embedding}

        async def run_search(q: str, p: dict[str, object]) -> list[dict[str, object]]:
            try:
                res = await self.db.query(q, cast("dict[str, Value]", p))
                return _extract_rows(res)
            except Exception as e:
                logger.error("[STORE] Query failed: %s\nQuery: %s", e, q)
                return []

        logger.info("[STORE] Attempting Hybrid Search...")
        rows = await run_search(query, params)

        if not rows:
            logger.warning("[STORE] Hybrid search returned 0 rows. Attempting fallbacks...")

            # Fallback 1: Pure Vector Search (KNN)
            vector_query = f"SELECT *, document_id FROM chunk WHERE embedding <|{k},150|> $embedding LIMIT {k};"
            vector_rows = await run_search(vector_query, {"embedding": params["embedding"]})
            logger.info("[STORE] Fallback 1 (Vector) returned %d rows", len(vector_rows))

            # Fallback 2: Pure Text Search (BM25)
            text_rows = []
            if use_hybrid:
                text_query = (
                    f"SELECT *, document_id FROM chunk WHERE text @1@ $query_text LIMIT {k};"
                )
                text_rows = await run_search(text_query, {"query_text": params["query_text"]})
                logger.info("[STORE] Fallback 2 (Text) returned %d rows", len(text_rows))

            # Combine and deduplicate
            seen_ids = set()
            for r in vector_rows + text_rows:
                rid = str(r["id"])
                if rid not in seen_ids:
                    rows.append(r)
                    seen_ids.add(rid)

            if rows:
                logger.info("[STORE] Fallback successful. Total merged rows: %d", len(rows))
            else:
                logger.error(
                    "[STORE] ALL SEARCH FALLBACKS FAILED. Indices returned no results despite data existing."
                )

        # Map back to domain models
        return [
            Chunk(
                id=_clean_id(item["id"]),
                document_id=_clean_id(item["document_id"])
                if "document_id" in item and item["document_id"]
                else "",
                text=cast(str, item["text"]),
                index=cast(int, item["index"]),
                start_char=cast(int, item.get("start_char", 0)),
                end_char=cast(int, item.get("end_char", 0)),
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
                file_path=cast(str | None, row.get("file_path")),
                metadata=cast(dict[str, str | int | float | bool], row.get("metadata", {})),
            )
            for row in _extract_rows(result)
        ]

    async def get_document_summaries(
        self, active_notebook_ids: list[str] | None = None
    ) -> list[dict[str, str]]:
        """Retrieve pre-computed global summaries for documents.

        Args:
            active_notebook_ids: Optional list of notebook IDs to filter by.

        Returns:
            A list of dicts with 'filename' and 'summary'.
        """
        if active_notebook_ids and len(active_notebook_ids) > 0:
            query = """
            SELECT filename, global_summary FROM document
            WHERE (->belongs_to->notebook.id CONTAINSANY $notebook_ids);
            """
            params = {"notebook_ids": [f"notebook:{nid}" for nid in active_notebook_ids]}
        else:
            query = "SELECT filename, global_summary FROM document;"
            params = {}

        result = await self.db.query(query, cast("dict[str, Value]", params))
        rows = _extract_rows(result)

        return [
            {
                "filename": cast(str, row.get("filename", "Unknown")),
                "summary": cast(str, row.get("global_summary", "No summary available.")),
            }
            for row in rows
            if row.get("global_summary")
        ]

    async def get_document(self, document_id: str) -> Document | None:
        """Retrieve a specific document by its ID.

        Args:
            document_id: The unique identifier of the document.

        Returns:
            The Document object if found, otherwise None.
        """
        result = await self.db.query(
            "SELECT * FROM type::record('document', $id);",
            {"id": document_id},
        )
        rows = _extract_rows(result)
        if not rows:
            return None

        row = rows[0]
        return Document(
            id=_clean_id(row["id"]),
            filename=cast(str, row.get("filename", "unknown")),
            status=cast(str, row.get("status", "active")),
            metadata=cast(dict[str, str | int | float | bool], row.get("metadata", {})),
        )

    async def update_document_status(self, document_id: str, status: str) -> None:
        """Update the processing status of a document.

        Args:
            document_id: The ID of the document to update.
            status: The new status string (e.g., 'active', 'failed').
        """
        await self.db.query(
            "UPDATE type::record('document', $id) SET status = $status;",
            {"id": document_id, "status": status},
        )

    async def add_document_to_notebook(self, document_id: str, notebook_id: str) -> None:
        """Relate a document to a notebook using a graph edge.

        Establishes `document -> belongs_to -> notebook` edges.

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

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document, its chunks, and the physical file.

        Maintenance logic:
            Increments a persistent deletion counter and triggers an HNSW index
            rebuild every 5 deletions to clear vector tombstones.

        Args:
            document_id: The ID of the document to remove.

        Returns:
            True if document was successfully deleted, False otherwise.
        """
        logger.info("[STORE] Deleting document %s and performing maintenance check", document_id)

        # 1. Fetch file_path before deletion
        result = await self.db.query(
            "SELECT file_path FROM type::record('document', $id);", {"id": document_id}
        )
        rows = _extract_rows(result)
        if not rows:
            logger.warning("[STORE] Document %s not found for deletion", document_id)
            return False

        file_path_str = cast(str | None, rows[0].get("file_path"))

        # 2. Delete from database
        maintenance_query = """
        BEGIN TRANSACTION;
        -- Delete chunks and then the document itself
        DELETE chunk WHERE document_id = $id;
        DELETE type::record('document', $id);

        -- Update maintenance counter
        LET $current = (SELECT VALUE count FROM maintenance:deletions)[0] OR 0;
        LET $new_count = $current + 1;
        UPSERT maintenance:deletions SET count = $new_count;

        -- Check for maintenance threshold (5)
        IF $new_count >= 5 {
            REBUILD INDEX chunk_embedding_idx ON TABLE chunk;
            UPDATE maintenance:deletions SET count = 0;
        };
        COMMIT TRANSACTION;
        """

        try:
            await self.db.query(maintenance_query, {"id": document_id})
        except Exception as e:
            logger.error("[STORE] Failed to delete document %s from database: %s", document_id, e)
            return False

        # 3. Delete physical file
        if file_path_str:
            try:
                path = Path(file_path_str)
                if path.exists():
                    path.unlink(missing_ok=True)
                    logger.info("[STORE] Deleted physical file: %s", file_path_str)
                else:
                    logger.warning("[STORE] Physical file not found at %s", file_path_str)
            except Exception as e:
                logger.error("[STORE] Failed to delete physical file %s: %s", file_path_str, e)
                # We return True because the DB record was deleted successfully

        return True

    async def initialize_schema(self) -> None:
        """Initialize SurrealDB table indices for document storage.

        Idempotently executes schema queries from schema.py.
        """
        logger.info("Initializing SurrealDocumentStore schema using centralized definitions")
        queries = get_schema_queries()
        for query in queries:
            try:
                await self.db.query(query)
            except Exception as e:
                # Log but continue if indices/tables already exist
                if "already exists" in str(e).lower():
                    logger.debug("Schema component already exists: %s", str(e)[:100])
                else:
                    logger.error("Failed to execute schema query: %s", e)
                    raise e

    async def save_notebook(self, notebook: Notebook) -> None:
        """Save a notebook record to the database.

        Args:
            notebook: The notebook domain model to persist.
        """
        logger.info("Saving notebook to SurrealDB: %s", notebook.title)
        await self.db.query(
            "UPSERT type::record('notebook', $id) CONTENT { title: $title, description: $description, created_at: $created_at };",
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
        query = """
        BEGIN TRANSACTION;
        DELETE belongs_to WHERE out = $id;
        DELETE $id;
        COMMIT TRANSACTION;
        """
        await self.db.query(query, {"id": RecordID("notebook", notebook_id)})


class SurrealGraphStore(GraphStore):
    """SurrealDB implementation of GraphStore.

    Handles persistent storage of extracted Knowledge Graphs and provides
    complex graph traversal capabilities.

    Pipeline Role:
        Phase 7: Graph Persistence. Persistent storage of extracted Knowledge
        Graphs and neighborhood retrieval.

    Graph Topology:
        The `GraphStore` represents the "structural memory" of the system. Unlike
        the DocumentStore's HNSW index which find "similar" things, the GraphStore
        finds "connected" things. It stores nodes (entities) and edges
        (semantic relations) to allow for multi-hop reasoning during retrieval.

    Implementation Details:
        - Harmonizes SurrealDB's `RecordID` with Pydantic domain models.
        - Uses `extracted_from` edges to link entities back to chunks.
        - Uses `relation` edges for semantic entity links.
        - Implements BFS-style traversal with configurable depth for
          neighborhood expansion.
    """

    db: AsyncSurrealType

    def __init__(self, db: AsyncSurrealType) -> None:
        """Initialize the graph store.

        Args:
            db: An active SurrealDB client instance (WS, HTTP, or Embedded).
        """
        self.db = db

    async def save_nodes(self, nodes: list[Node]) -> None:
        """Save entity nodes and relate them to their source chunks.

        Establishes `entity -> extracted_from -> chunk` edges.

        Args:
            nodes: List of Node domain models to persist.
        """
        logger.info("Saving %d nodes to SurrealDB Graph with extracted_from relations", len(nodes))
        for node in nodes:
            await self.db.query(
                "UPSERT $id CONTENT { label: $label, name: $name, description: $description, description_embedding: $description_embedding };",
                {
                    "id": RecordID("entity", node.id),
                    "label": node.label,
                    "name": node.name,
                    "description": node.description,
                    "description_embedding": cast("Value", node.description_embedding),
                },
            )
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
        """Initialize SurrealDB graph storage schema.

        Relies on SurrealDocumentStore.initialize_schema() for shared definitions
        but ensures graph-specific indices exist.
        """
        logger.info("Initializing SurrealGraphStore schema (delegated to DocumentStore)")
        # In this implementation, DocumentStore.initialize_schema() covers all tables.
        # We can call it directly to ensure all defined tables/indices are present.
        doc_store = SurrealDocumentStore(self.db)
        await doc_store.initialize_schema()

    async def save_edges(self, edges: list[Edge]) -> None:
        """Save relationship edges between entities.

        Establishes `entity -> relation -> entity` edges with extracted metadata.

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

        Performs a Breadth-First Search (BFS) up to the specified depth.
        Gathers both incoming and outgoing edges to build a full local
        subgraph for LLM reasoning.

        Args:
            seed_node_ids: IDs of nodes to begin the traversal (usually from linking).
            depth: Maximum number of hops to follow. Defaults to 2.

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
                                    weight=float(cast(float, edge_data.get("weight", 1.0)))
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
                                    weight=float(cast(float, edge_data.get("weight", 1.0)))
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
                    weight=float(cast(float, edge_data.get("weight", 1.0)))
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
            "UPSERT type::record('community', $id) CONTENT { summary: $summary, node_ids: $node_ids };",
            {
                "id": community.id,
                "summary": community.summary,
                "node_ids": cast("Value", community.node_ids),
            },
        )
