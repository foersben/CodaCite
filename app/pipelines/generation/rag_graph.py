"""Self-correcting RAG pipeline built with LangGraph.

Implements an agentic, cyclical retrieval loop that re-ranks retrieved documents
for relevance and optionally rewrites the query before returning context.

Graph Topology::

    START → retrieve → rerank ──(all bad + rewrites < max)──→ rewrite ─┐
                           │                                            │
                           └──(some good OR rewrites == max)──────────→ END

This architecture enables the system to autonomously refine its search strategy
if initial retrieval fails to produce high-utility context, ensuring grounding
resilience for underspecified or complex queries.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.interfaces import (
    DocumentStore,
    Embedder,
    EntityLinker,
    GraphStore,
    LLMGenerator,
    Reranker,
)
from app.models.models import Chunk, Node

logger = logging.getLogger(__name__)


def _strip_context_prefix(text: str) -> str:
    """Remove the injected document prefix when it matches the expected format."""
    if text.startswith("Document:") and "\n" in text:
        return text.split("\n", 1)[1]
    return text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_REWRITE_PROMPT = """\
You are a search-query optimizer. Rephrase the following question so that a \
document retrieval system can find better matches. Output only the rephrased question.

Original question: {question}
Rephrased question:"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RAGState(TypedDict, total=False):
    """Shared mutable state threaded through all LangGraph nodes.

    This state object represents the "memory" of the retrieval agent,
    accumulating documents, rewrite attempts, and configuration parameters
    as it traverses the graph.

    Attributes:
        question: The current search query (may be updated by the rewrite node).
        history: Optional list of previous chat messages for context-aware RAG.
        documents: A flat list of retrieved context snippets (chunks, entities,
            or relationships) with metadata.
        generation: The final set of filtered/reranked documents to be returned.
        rewrite_count: Counter for query re-phrasing attempts to prevent
            infinite loops.
        top_k: Target number of context snippets requested by the user.
        notebook_ids: Optional list of notebook UUIDs for partition-based filtering.
    """

    question: str
    history: list[dict[str, str]] | None
    documents: list[dict[str, object]]
    generation: list[dict[str, object]]
    rewrite_count: int
    # Configuration parameters passed per-request
    top_k: int
    notebook_ids: list[str] | None


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def make_retrieve_node(
    store: DocumentStore,
    embedder: Embedder,
    graph_store: GraphStore,
    entity_linker: EntityLinker | None,
) -> Callable[[RAGState], Coroutine[Any, Any, RAGState]]:
    """Build the retrieve node, binding infrastructure dependencies via closure.

    Args:
        store: Document store for hybrid BM25+HNSW chunk search.
        embedder: Embedding model for query vectorization.
        graph_store: Knowledge graph store for entity traversal.
        entity_linker: Duck-typed linker with ``link_entities(query, nodes)`` method.

    Returns:
        An async callable suitable for use as a LangGraph node.
    """

    async def retrieve_node(state: RAGState) -> dict[str, Any]:
        """Executes a hybrid multi-stage retrieval strategy.

        This node combines two distinct search paradigms to build a
        high-fidelity context window:
        1. **Vector/BM25 Search**: Finds semantically similar document chunks.
        2. **Window Expansion**: Fetches immediate neighbors (i-1, i+1) of
           vector hits to provide local narrative continuity.
        3. **Graph Traversal**: Links entities found in the query to the
           Knowledge Graph and fetches related neighbors.

        Args:
            state: Current graph state.

        Returns:
            Partial state update containing the assembled ``documents``.
        """
        question = state["question"]

        # 1. Embed the query
        query_text = question
        if hasattr(embedder, "query_prefix"):
            query_text = f"{embedder.query_prefix}{question}"
        query_embedding = await embedder.embed(query_text)
        logger.debug("[RAG_GRAPH] retrieve: embedding dim=%d", len(query_embedding))

        # 2. Hybrid chunk search (BM25 + HNSW)
        chunks: list[Chunk] = await store.search_chunks(
            query_embedding,
            query_text=question,
            top_k=state["top_k"],
            active_notebook_ids=state["notebook_ids"],
        )

        # 2b. Window Expansion: Fetch neighbors for O(1) context
        neighbor_ids: set[str] = set()
        for c in chunks:
            if c.index > 0:
                neighbor_ids.add(f"{c.document_id}_{c.index - 1}")
            neighbor_ids.add(f"{c.document_id}_{c.index + 1}")

        # Batch fetch all potential neighbors in one O(1) query
        all_related_chunks = await store.get_chunks_by_ids(list(neighbor_ids))
        lookup = {(c.document_id, c.index): c for c in chunks + all_related_chunks}

        documents: list[dict[str, object]] = []
        for hit in chunks:
            # Construct local window [i-1, i, i+1]
            window = []
            for idx in range(hit.index - 1, hit.index + 2):
                if (hit.document_id, idx) in lookup:
                    window.append(lookup[(hit.document_id, idx)])

            # Ensure chronological order
            window.sort(key=lambda x: x.index)

            # Merge with strict prefix cleanup to save tokens
            cleaned_parts = []
            for i, c in enumerate(window):
                text = c.text
                if i > 0:
                    # Strip "Document: [Filename]\n" prefix from subsequent chunks
                    text = _strip_context_prefix(text)
                cleaned_parts.append(text)

            # Join with visible separator for LLM adjacency awareness
            window_text = "\n[...]\n".join(cleaned_parts)

            documents.append(
                {
                    "text": window_text,
                    "type": "chunk",
                    "id": hit.id,
                    "document_id": hit.document_id,
                }
            )

        # 3. Entity linking + graph traversal
        # Optimized: Search for relevant nodes using BM25 instead of fetching the whole graph
        candidate_nodes: list[Node] = await graph_store.search_nodes(question, top_k=20)
        linked_nodes: list[Node] = []
        if entity_linker:
            linked_nodes = await entity_linker.link_entities(question, candidate_nodes)

        if linked_nodes:
            seed_ids = [n.id for n in linked_nodes]
            traversed_nodes, traversed_edges = await graph_store.traverse(seed_ids, depth=2)
            for node in traversed_nodes:
                documents.append(
                    {
                        "text": (f"Entity: {node.name} ({node.label}). {node.description or ''}"),
                        "type": "entity",
                        "id": node.id,
                        "source_chunk_ids": node.source_chunk_ids,
                    }
                )
            for edge in traversed_edges:
                documents.append(
                    {
                        "text": (
                            f"Relationship: {edge.source_id} {edge.relation} {edge.target_id}."
                        ),
                        "type": "relation",
                        "id": edge.id,
                        "source_chunk_ids": edge.source_chunk_ids,
                    }
                )

        # 4. Deduplicate by text
        seen: set[str] = set()
        unique_docs: list[dict[str, object]] = []
        for doc in documents:
            key = str(doc.get("text", ""))
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        logger.info("[RAG_GRAPH] retrieve: %d unique docs", len(unique_docs))
        return {"documents": unique_docs}

    async def retrieve_node_safe(state: RAGState) -> RAGState:
        """Wrapper for retrieve_node with error handling to prevent graph hangs."""
        try:
            return cast(RAGState, await retrieve_node(state))
        except Exception as exc:
            logger.error("[RAG_GRAPH] retrieve_node failed: %s", exc, exc_info=True)
            return cast(RAGState, {"documents": []})

    return retrieve_node_safe


def make_rerank_node(
    reranker: Reranker | None,
) -> Callable[[RAGState], Coroutine[Any, Any, RAGState]]:
    """Build the rerank node using ModernBERT cross-encoders."""

    async def rerank_node(state: RAGState) -> RAGState:
        """Re-ranks and filters retrieved documents using cross-encoders."""
        question = state["question"]
        documents = state["documents"]
        context_texts = [str(doc["text"]) for doc in documents]

        if not reranker or not context_texts:
            logger.debug("[RAG_GRAPH] rerank: skipping (no reranker or no documents)")
            return cast(RAGState, {"documents": documents})

        try:
            # Rerank all candidate documents
            results = await reranker.rerank(question, context_texts, top_k=len(context_texts))

            # Filter by score (e.g., > 0.3 for ModernBERT/GTE)
            threshold = 0.3
            filtered_results = [r for r in results if r["score"] > threshold]

            # Convert back to document objects with scores
            text_to_meta = {str(doc["text"]): doc for doc in documents}
            reranked_docs = []
            for r in filtered_results[: state["top_k"]]:
                text = r["text"]
                if text in text_to_meta:
                    doc = text_to_meta[text].copy()
                    doc["score"] = r["score"]
                    reranked_docs.append(doc)

            if not reranked_docs:
                logger.info("[RAG_GRAPH] rerank: all docs filtered below threshold")
                return cast(RAGState, {"documents": []})

            logger.info(
                "[RAG_GRAPH] rerank: %d/%d docs kept (threshold=%.2f)",
                len(reranked_docs),
                len(documents),
                threshold,
            )
            return cast(RAGState, {"documents": reranked_docs})

        except Exception as exc:
            logger.warning("[RAG_GRAPH] reranking node failed: %s", exc)
            return cast(RAGState, {"documents": documents})

    return rerank_node


def make_rewrite_query_node(
    generator: LLMGenerator,
) -> Callable[[RAGState], Coroutine[Any, Any, RAGState]]:
    """Build the query rewrite node."""

    async def rewrite_query_node(state: RAGState) -> RAGState:
        """Rewrites the query using an LLM to improve retrieval recall."""
        question = state["question"]
        rewrite_count = state["rewrite_count"]

        prompt = _REWRITE_PROMPT.format(question=question)
        new_question = (await generator.agenerate(prompt)).strip()

        # Clean up common model prefixes
        prefixes_to_strip = [
            "Rephrased question:",
            "Optimized query:",
            "New question:",
            "Rewritten question:",
        ]
        for prefix in prefixes_to_strip:
            if new_question.lower().startswith(prefix.lower()):
                new_question = new_question[len(prefix) :].strip()

        # If model returned empty or just noise, fallback to original
        if not new_question or len(new_question) < 2:
            new_question = question

        logger.info(
            "[RAG_GRAPH] rewrite %d: '%s' → '%s'",
            rewrite_count + 1,
            question,
            new_question,
        )
        return cast(RAGState, {"question": new_question, "rewrite_count": rewrite_count + 1})

    async def rewrite_query_node_safe(state: RAGState) -> RAGState:
        """Wrapper for rewrite_query_node with error handling."""
        try:
            return await rewrite_query_node(state)
        except Exception as exc:
            logger.warning("[RAG_GRAPH] rewrite_query_node failed: %s", exc)
            return cast(RAGState, {"rewrite_count": state["rewrite_count"] + 1})

    return rewrite_query_node_safe


def _make_router(max_rewrites: int) -> Callable[[RAGState], str]:
    """Build the conditional edge routing function."""

    def router(state: RAGState) -> str:
        """Route after reranking: rewrite query or proceed to END."""
        if not state["documents"] and state["rewrite_count"] < max_rewrites:
            logger.debug(
                "[RAG_GRAPH] routing → rewrite (attempt %d/%d)",
                state["rewrite_count"] + 1,
                max_rewrites,
            )
            return "rewrite"
        return "__end__"

    return router


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_rag_graph(
    store: DocumentStore,
    graph_store: GraphStore,
    embedder: Embedder,
    entity_linker: EntityLinker | None,
    generator: LLMGenerator,
    reranker: Reranker | None,
    max_rewrites: int = 3,
) -> CompiledStateGraph[RAGState, None, RAGState]:
    """Compile and return the self-correcting retrieval LangGraph."""
    graph = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        cast(Any, make_retrieve_node(store, embedder, graph_store, entity_linker)),
    )

    graph.add_node("rerank", cast(Any, make_rerank_node(reranker)))
    graph.add_node("rewrite", cast(Any, make_rewrite_query_node(generator)))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        _make_router(max_rewrites),
        {"rewrite": "rewrite", "__end__": END},
    )
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()
