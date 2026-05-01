"""Self-correcting RAG pipeline built with LangGraph.

Implements an agentic, cyclical retrieval loop that grades retrieved documents
for relevance and optionally rewrites the query before generating a final answer.

Graph Topology::

    START → retrieve → grade ──(all bad + rewrites < max)──→ rewrite ─┐
                           │                                            │
                           └──(some good OR rewrites == max)──→ generate → END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.domain.models import Chunk, Node
from app.domain.ports import (
    DocumentStore,
    Embedder,
    EntityLinker,
    GraphStore,
    LLMGenerator,
    Reranker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GRADE_PROMPT = """\
You are a relevance grader. Assess whether the document excerpt below contains \
information that could help answer the question.

Question: {question}
Document excerpt: {document}

Answer with a single word — yes or no:"""

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
        documents: Retrieved and filtered context snippets.
        generation: The final reranked output passed back to the caller.
        hallucination_score: Reserved for future faithfulness scoring (0.0–1.0).
        rewrite_count: How many query rewrites have been attempted so far.
    """

    question: str
    documents: list[dict[str, object]]
    generation: list[dict[str, object]]
    hallucination_score: float
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
) -> Any:
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


def make_grade_documents_node(generator: LLMGenerator) -> Any:
    """Build the document grading node.

    Calls the LLM once per document with a binary relevance prompt. Documents
    where the response does not start with ``"yes"`` are discarded.

    Args:
        generator: LLM interface used to judge relevance.

    Returns:
        An async callable suitable for use as a LangGraph node.
    """

    async def grade_documents_node(state: RAGState) -> dict[str, object]:
        """Filter retrieved documents to keep only those relevant to the question.

        Args:
            state: Current graph state.

        Returns:
            Partial state update containing the filtered ``documents`` list.
        """
        question = state["question"]
        documents = state["documents"]

        relevant: list[dict[str, object]] = []
        for doc in documents:
            prompt = _GRADE_PROMPT.format(question=question, document=str(doc["text"]))
            verdict = await generator.agenerate(prompt)
            if verdict.strip().lower().startswith("yes"):
                relevant.append(doc)

        logger.info(
            "[RAG_GRAPH] grade: %d/%d docs kept",
            len(relevant),
            len(documents),
        )
        return {"documents": relevant}

    return grade_documents_node


def make_rewrite_query_node(generator: LLMGenerator) -> Any:
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
        new_question = (await generator.agenerate(prompt)).strip() or question

        logger.info(
            "[RAG_GRAPH] rewrite %d: '%s' → '%s'",
            rewrite_count + 1,
            question,
            new_question,
        )
        return {"question": new_question, "rewrite_count": rewrite_count + 1}

    return rewrite_query_node


def make_generate_node(generator: LLMGenerator, reranker: Reranker | None) -> Any:
    """Build the final answer generation node.

    Runs filtered documents through optional reranking and returns ranked
    context snippets as the ``generation`` result.

    Args:
        generator: Reserved for future faithfulness-scored generation.
        reranker: Reranker interface for scoring context relevance.

    Returns:
        An async callable suitable for use as a LangGraph node.
    """

    async def generate_node(state: RAGState) -> dict[str, object]:
        """Generate the final ranked context list.

        Args:
            state: Current graph state.

        Returns:
            Partial state update with ``generation`` (list of ranked dicts).
        """
        question = state["question"]
        documents = state["documents"]
        context_texts = [str(doc["text"]) for doc in documents]

        if reranker and context_texts:
            try:
                results: list[dict[str, object]] = await reranker.rerank(
                    question, context_texts, top_k=state["top_k"]
                )
                logger.info("[RAG_GRAPH] generate: reranked %d snippets", len(results))
                return {"generation": results}
            except Exception as exc:
                logger.warning("[RAG_GRAPH] reranking failed (%s) — plain fallback", exc)

        fallback: list[dict[str, object]] = [
            {"text": t, "score": 1.0} for t in context_texts[: state["top_k"]]
        ]
        logger.info("[RAG_GRAPH] generate: %d snippets (no reranking)", len(fallback))
        return {"generation": fallback}

    return generate_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _make_router(max_rewrites: int) -> Any:
    """Build the conditional edge routing function.

    Routes to ``"rewrite"`` when all documents were filtered and the rewrite
    budget is not exhausted; otherwise routes to ``"generate"``.

    Args:
        max_rewrites: Maximum allowed rewrites before falling through to generate.

    Returns:
        A callable ``(state: RAGState) -> str`` for LangGraph conditional edges.
    """

    def router(state: RAGState) -> str:
        """Route after grading: rewrite query or proceed to generation."""
        if not state["documents"] and state["rewrite_count"] < max_rewrites:
            logger.debug(
                "[RAG_GRAPH] routing → rewrite (attempt %d/%d)",
                state["rewrite_count"] + 1,
                max_rewrites,
            )
            return "rewrite"
        return "generate"

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
) -> Any:
    """Compile and return the self-correcting RAG LangGraph.

    The returned compiled graph accepts an initial ``RAGState`` dict via
    ``ainvoke`` and returns the final state after the graph terminates.

    Args:
        store: Document store for hybrid chunk retrieval.
        graph_store: Knowledge graph store for entity traversal.
        embedder: Query embedding model.
        entity_linker: Entity linking duck-typed object.
        generator: LLM for grading, rewriting, and generation.
        reranker: Optional reranker duck-typed object.
        max_rewrites: Maximum number of query rewrite cycles (default: 3).

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for ``ainvoke``.
    """
    graph: StateGraph[RAGState] = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        make_retrieve_node(store, embedder, graph_store, entity_linker),
    )
    graph.add_node("grade", make_grade_documents_node(generator))
    graph.add_node("rewrite", make_rewrite_query_node(generator))
    graph.add_node("generate", make_generate_node(generator, reranker))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _make_router(max_rewrites),
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()
