"""Use case for performing hybrid GraphRAG retrieval.

This module coordinates the retrieval process using a self-correcting
LangGraph pipeline: hybrid chunk retrieval → document grading → optional
query rewrite → final context generation.
"""

import logging
from typing import Any

from app.application.rag_graph import RAGState, build_rag_graph
from app.domain.ports import (
    DocumentStore,
    Embedder,
    EntityLinker,
    GraphStore,
    LLMGenerator,
    Reranker,
)

logger = logging.getLogger(__name__)


class GraphRAGRetrievalUseCase:
    """Orchestrates the self-correcting GraphRAG retrieval pipeline.

    Delegates all retrieval logic to a compiled LangGraph that implements a
    cyclical Retrieve → Grade → (Rewrite →)* Generate loop.

    Retrieval Stages:
        1. **Retrieve**: Hybrid BM25+HNSW chunk search + graph traversal.
        2. **Grade**: LLM-based per-document relevance check; irrelevant chunks
           are discarded.
        3. **Rewrite** *(conditional)*: If all chunks are graded irrelevant and
           the rewrite budget is not exhausted, the query is rephrased and
           retrieval repeats.
        4. **Generate**: Remaining relevant chunks are optionally reranked and
           returned to the caller.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        graph_store: GraphStore,
        embedder: Embedder,
        entity_linker: EntityLinker,
        reranker: Reranker,
        generator: LLMGenerator,
    ) -> None:
        """Initialize the retrieval use case with required ports.

        Args:
            document_store: Access to document metadata and vector chunks.
            graph_store: Access to entity-relationship data and traversal logic.
            embedder: Transformer model for query vectorization.
            entity_linker: Logic for mapping query strings to graph nodes.
            reranker: Logic for scoring and sorting context snippets.
            generator: LLM used for document grading and query rewriting.
        """
        self.document_store = document_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.entity_linker = entity_linker
        self.reranker = reranker
        self.generator = generator

    async def execute(
        self, query: str, top_k: int = 5, notebook_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Execute the self-correcting retrieval pipeline.

        Compiles and invokes the LangGraph for the given query. The graph
        handles embedding, hybrid search, grading, optional rewriting, and
        final context assembly.

        Args:
            query: The user's natural language question.
            top_k: Number of context snippets to return.
            notebook_ids: Optional list of notebook IDs to filter context.

        Returns:
            A list of context dictionaries with ``text`` and ``score`` keys,
            ordered by relevance.
        """
        logger.info(
            "[RETRIEVAL] Starting self-correcting RAG for: %s (notebooks: %s)",
            query,
            notebook_ids,
        )

        compiled = build_rag_graph(
            store=self.document_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            entity_linker=self.entity_linker,
            generator=self.generator,
            reranker=self.reranker,
            top_k=top_k,
            notebook_ids=notebook_ids,
        )

        initial_state: RAGState = {
            "question": query,
            "documents": [],
            "generation": [],
            "hallucination_score": 0.0,
            "rewrite_count": 0,
        }

        final_state: dict[str, Any] = await compiled.ainvoke(initial_state)
        generation: list[dict[str, Any]] = final_state.get("generation", [])

        logger.info("[RETRIEVAL] Pipeline complete: %d snippets returned", len(generation))
        return generation
