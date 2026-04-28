"""Use case for performing hybrid GraphRAG retrieval.

This module coordinates the retrieval process by combining vector-based chunk
search with graph-based traversal and reranking to provide high-fidelity context.
"""

import logging
from typing import Any, cast

from app.domain.models import Edge, Node
from app.domain.ports import DocumentStore, Embedder, GraphStore

logger = logging.getLogger(__name__)


class GraphRAGRetrievalUseCase:
    """Orchestrates the hybrid GraphRAG retrieval pipeline.

    Combines traditional Vector Search with Knowledge Graph traversal to
    provide the LLM with both specific text fragments and broader conceptual
    context.

    Retrieval Stages:
        1.  **Semantic Search**: Finds the top-k chunks from `DocumentStore`
            using vector similarity (BGE-M3).
        2.  **Entity Linking**: Maps terms in the user query to specific nodes
            in the `GraphStore`.
        3.  **Graph Traversal**: Explores the 2-hop neighborhood of linked nodes
            to find related entities and semantic relations.
        4.  **Context Aggregation**: Combines chunks, entity descriptions, and
            relationship triples into a unified context block.
        5.  **Reranking**: (Optional) Uses a Cross-Encoder to prioritize the
            most relevant snippets for the final prompt.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        graph_store: GraphStore,
        embedder: Embedder,
        entity_linker: Any,
        reranker: Any,
    ) -> None:
        """Initialize the retrieval use case with required ports.

        Args:
            document_store: Access to document metadata and vector chunks.
            graph_store: Access to entity-relationship data and traversal logic.
            embedder: Transformer model for query vectorization.
            entity_linker: Logic for mapping query strings to graph nodes.
            reranker: Logic for scoring and sorting context snippets.
        """
        self.document_store = document_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.entity_linker = entity_linker
        self.reranker = reranker

    async def execute(
        self, query: str, top_k: int = 5, notebook_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Execute the hybrid retrieval pipeline.

        Args:
            query: The user's natural language question.
            top_k: Number of context snippets to return after reranking.
            notebook_ids: Optional list of notebook IDs to filter context.

        Returns:
            A list of context dictionaries containing text and relevance scores.
        """
        logger.info(
            "[RETRIEVAL] Starting retrieval for query: %s (Notebooks: %s)", query, notebook_ids
        )

        # 1. Vector Search on Chunks
        query_text = query
        if hasattr(self.embedder, "query_prefix"):
            query_text = f"{self.embedder.query_prefix}{query}"

        query_embedding = await self.embedder.embed(query_text)
        logger.debug("[RETRIEVAL] Query embedding generated (dim: %d)", len(query_embedding))

        retrieved_chunks = await self.document_store.search_chunks(
            query_embedding, top_k=top_k, active_notebook_ids=notebook_ids
        )
        logger.info("[RETRIEVAL] Found %d semantic chunks", len(retrieved_chunks))
        for i, chunk in enumerate(retrieved_chunks):
            logger.debug(
                "[RETRIEVAL] Chunk %d: id=%s doc=%s score=%s text_len=%d",
                i,
                chunk.id,
                chunk.document_id,
                getattr(chunk, "score", "N/A"),
                len(chunk.text),
            )

        # 2. Entity Linking on Query
        all_nodes = await self.graph_store.get_all_nodes()

        # Linker is currently dynamic; ideally it would be a port
        link_entities_func = getattr(self.entity_linker, "link_entities", None)
        linked_nodes: list[Node] = []
        if link_entities_func:
            linked_nodes = await link_entities_func(query, all_nodes)
            logger.debug("[RETRIEVAL] Linked %d entities from query", len(linked_nodes))

        # 3. Multi-hop Traversal
        seed_node_ids = [n.id for n in linked_nodes]
        traversed_nodes: list[Node] = []
        traversed_edges: list[Edge] = []
        if seed_node_ids:
            traversed_nodes, traversed_edges = await self.graph_store.traverse(
                seed_node_ids, depth=2
            )
            logger.debug(
                "[RETRIEVAL] Traversed graph: %d nodes, %d edges",
                len(traversed_nodes),
                len(traversed_edges),
            )

        # 4. Context Combination
        contexts = []
        for chunk in retrieved_chunks:
            contexts.append(chunk.text)

        for node in traversed_nodes:
            desc = f"Entity: {node.name} ({node.label}). {node.description or ''}"
            contexts.append(desc)

        for edge in traversed_edges:
            desc = f"Relationship: {edge.source_id} {edge.relation} {edge.target_id}."
            contexts.append(desc)

        # Deduplicate snippets
        contexts = list(set(contexts))

        if not contexts:
            logger.warning("[RETRIEVAL] No context found for query: %s", query)
            return []

        # 5. Reranking
        rerank_func = getattr(self.reranker, "rerank", None)
        if rerank_func:
            try:
                logger.info("[RETRIEVAL] Reranking %d snippets", len(contexts))
                reranked_results = await rerank_func(query, contexts, top_k=top_k)
                return cast(list[dict[str, Any]], reranked_results)
            except Exception as e:
                logger.error("[RETRIEVAL] Reranking failed: %s", e)
                # Fallback if reranking fails

        return [{"text": ctx, "score": 1.0} for ctx in contexts[:top_k]]
