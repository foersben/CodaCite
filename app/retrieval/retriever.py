"""Hybrid GraphRAG retriever: vector search + graph traversal + reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """A single retrieval result with text, score, and provenance."""

    text: str
    score: float
    source: str = ""


class HybridRetriever:
    """Orchestrates three-stage retrieval: vector search → graph traversal → reranking.

    Stage 1 – Vector Search:
        Embeds the query and performs cosine similarity search in SurrealDB.

    Stage 2 – Graph Traversal:
        For each vector result, traverses connected entity nodes in the knowledge
        graph (depth 1–2) and appends them to the candidate pool.

    Stage 3 – Reranking:
        Applies a local Cross-Encoder to score and sort all candidates.

    Args:
        store: :class:`~app.graph.store.GraphStore` instance.
        embedder: :class:`~app.embeddings.embedder.LocalEmbedder` instance.
        reranker: :class:`~app.retrieval.reranker.CrossEncoderReranker` instance.
        graph_depth: Traversal depth for graph enrichment (default 2).
    """

    def __init__(
        self,
        store: Any,
        embedder: Any,
        reranker: Any,
        graph_depth: int = 2,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._graph_depth = graph_depth

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Run the full hybrid retrieval pipeline.

        Args:
            query: The natural-language query string.
            top_k: Maximum number of final results to return.

        Returns:
            A ranked list of :class:`RetrievalResult` objects.
        """
        # Stage 1: embed query and vector-search
        query_embedding = self._embedder.embed([query])[0]
        vector_hits: list[dict[str, Any]] = await self._store.vector_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
        )

        # Stage 2: graph traversal for each vector hit
        graph_nodes: list[dict[str, Any]] = []
        for hit in vector_hits:
            node_id = hit.get("id", "")
            if node_id:
                neighbours = await self._store.traverse_graph(
                    start_node_id=node_id,
                    depth=self._graph_depth,
                )
                graph_nodes.extend(neighbours)

        # Merge candidates (vector hits + graph nodes) – deduplicate by id
        seen_ids: set[str] = set()
        all_candidates: list[dict[str, Any]] = []
        for item in [*vector_hits, *graph_nodes]:
            item_id = str(item.get("id", id(item)))
            text = item.get("text") or item.get("name") or item.get("description") or ""
            if item_id not in seen_ids and text:
                seen_ids.add(item_id)
                all_candidates.append({**item, "text": text})

        if not all_candidates:
            return []

        # Stage 3: rerank
        reranked = self._reranker.rerank(
            query=query,
            candidates=all_candidates,
            top_k=top_k,
        )

        return [
            RetrievalResult(
                text=r.get("text", ""),
                score=float(r.get("score", 0.0)),
                source=str(r.get("source", "")),
            )
            for r in reranked
        ]
