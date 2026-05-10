"""Self-correcting RAG pipeline built with LangGraph.

Implements an agentic, cyclical retrieval loop that re-ranks retrieved documents
for relevance and optionally rewrites the query before returning context.

Graph Topology::

    START → retrieve → rerank ──(all bad + rewrites < max)──→ rewrite ─┐
                           │                                            │
                           └──(some good OR rewrites == max)──────────→ END
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import TypedDict

from langgraph.graph import END, StateGraph

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


class RAGState(TypedDict):
    """Shared mutable state threaded through all LangGraph nodes.

    Attributes:
        question: The current (possibly rewritten) user query.
        history: Optional conversation history.
        documents: Retrieved and filtered context snippets.
        generation: The final reranked output (documents) passed back to the caller.
        rewrite_count: How many query rewrites have been attempted so far.
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
) -> Callable[[RAGState], Coroutine[object, object, dict[str, object]]]:
    """Build the retrieve node, binding infrastructure dependencies via closure.

    Args:
        store: Document store for hybrid BM25+HNSW chunk search.
        embedder: Embedding model for query vectorization.
        graph_store: Knowledge graph store for entity traversal.
        entity_linker: Duck-typed linker with ``link_entities(query, nodes)`` method.

    Returns:
        An async callable suitable for use as a LangGraph node.
    """

    async def retrieve_node(state: RAGState) -> dict[str, object]:
        """Retrieve hybrid search results and graph context.

        Args:
            state: Current graph state.

        Returns:
            Partial state update containing ``documents``.
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
        documents: list[dict[str, object]] = [
            {"text": c.text, "type": "chunk", "id": c.id, "document_id": c.document_id}
            for c in chunks
        ]

        # 3. Entity linking + graph traversal
        all_nodes: list[Node] = await graph_store.get_all_nodes()
        linked_nodes: list[Node] = []
        if entity_linker:
            linked_nodes = await entity_linker.link_entities(question, all_nodes)

        if linked_nodes:
            seed_ids = [n.id for n in linked_nodes]
            traversed_nodes, traversed_edges = await graph_store.traverse(seed_ids, depth=2)
            for node in traversed_nodes:
                documents.append(
                    {
                        "text": (f"Entity: {node.name} ({node.label}). {node.description or ''}"),
                        "type": "entity",
                        "id": node.id,
                    }
                )
            for edge in traversed_edges:
                documents.append(
                    {
                        "text": (
                            f"Relationship: {edge.source_id} {edge.relation} {edge.target_id}."
                        ),
                        "type": "relation",
                    }
                )

        # 4. Deduplicate by text
        seen: set[str] = set()
        unique_docs: list[dict[str, object]] = []
        for doc in documents:
            key = str(doc["text"])
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        logger.info("[RAG_GRAPH] retrieve: %d unique docs", len(unique_docs))
        return {"documents": unique_docs}

    return retrieve_node


def make_rerank_node(
    reranker: Reranker | None,
) -> Callable[[RAGState], Coroutine[object, object, dict[str, object]]]:
    """Build the rerank node using ModernBERT cross-encoders.

    Re-scores the retrieved documents and filters those below a quality threshold.

    Args:
        reranker: High-precision re-scoring model.

    Returns:
        An async callable for LangGraph.
    """

    async def rerank_node(state: RAGState) -> dict[str, object]:
        """Re-rank and filter retrieved documents.

        Args:
            state: Current graph state.

        Returns:
            Partial state update with re-scored and filtered ``documents``.
        """
        question = state["question"]
        documents = state["documents"]
        context_texts = [str(doc["text"]) for doc in documents]

        if not reranker or not context_texts:
            logger.debug("[RAG_GRAPH] rerank: skipping (no reranker or no documents)")
            return {"documents": documents}

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
                text = str(r["text"])
                if text in text_to_meta:
                    doc = text_to_meta[text].copy()
                    doc["score"] = r["score"]
                    reranked_docs.append(doc)

            logger.info(
                "[RAG_GRAPH] rerank: %d/%d docs kept (threshold=%.2f)",
                len(reranked_docs),
                len(documents),
                threshold,
            )
            return {"documents": reranked_docs}

        except Exception as exc:
            logger.warning("[RAG_GRAPH] reranking node failed: %s", exc)
            return {"documents": documents}

    return rerank_node


def make_rewrite_query_node(
    generator: LLMGenerator,
) -> Callable[[RAGState], Coroutine[object, object, dict[str, object]]]:
    """Build the query rewrite node.

    Asks the LLM to rephrase the current question to improve retrieval recall,
    then increments the rewrite counter.

    Args:
        generator: LLM interface used to rephrase the query.

    Returns:
        An async callable suitable for use as a LangGraph node.
    """

    async def rewrite_query_node(state: RAGState) -> dict[str, object]:
        """Rewrite the current question for better retrieval.

        Args:
            state: Current graph state.

        Returns:
            Partial state update with new ``question`` and incremented ``rewrite_count``.
        """
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
        return {"question": new_question, "rewrite_count": rewrite_count + 1}

    return rewrite_query_node


def _make_router(max_rewrites: int) -> Callable[[RAGState], str]:
    """Build the conditional edge routing function.

    Routes to ``"rewrite"`` when all documents were filtered and the rewrite
    budget is not exhausted; otherwise routes to ``END``.

    Args:
        max_rewrites: Maximum allowed rewrites before falling through to END.

    Returns:
        A callable ``(state: RAGState) -> str`` for LangGraph conditional edges.
    """

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
) -> object:  # LangGraph CompiledStateGraph has no stable public type export
    """Compile and return the self-correcting retrieval LangGraph.

    The returned compiled graph accepts an initial ``RAGState`` dict via
    ``ainvoke`` and returns the final state after the graph terminates.

    Args:
        store: Document store for hybrid chunk retrieval.
        graph_store: Knowledge graph store for entity traversal.
        embedder: Query embedding model.
        entity_linker: Entity linking duck-typed object.
        generator: LLM for rewriting.
        reranker: Optional reranker duck-typed object.
        max_rewrites: Maximum number of query rewrite cycles (default: 3).

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for ``ainvoke``.
    """
    graph: StateGraph[RAGState] = StateGraph(RAGState)

    graph.add_node(  # type: ignore[call-overload]
        "retrieve",
        make_retrieve_node(store, embedder, graph_store, entity_linker),
    )
    graph.add_node("rerank", make_rerank_node(reranker))  # type: ignore[call-overload]
    graph.add_node("rewrite", make_rewrite_query_node(generator))  # type: ignore[call-overload]

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        _make_router(max_rewrites),
        {"rewrite": "rewrite", "__end__": END},
    )
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()
