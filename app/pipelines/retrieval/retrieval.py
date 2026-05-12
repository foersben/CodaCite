"""Use case for performing hybrid GraphRAG retrieval.

This module coordinates the retrieval process using a self-correcting
LangGraph pipeline: hybrid chunk retrieval → document grading → optional
query rewrite → final context generation.
"""

import logging

from app.core.interfaces import (
    DocumentStore,
    Embedder,
    EntityLinker,
    GraphStore,
    LLMGenerator,
    Reranker,
)
from app.pipelines.generation.guardrails import FactualityGuardrail
from app.pipelines.generation.rag_graph import RAGState, build_rag_graph

logger = logging.getLogger(__name__)


class GraphRAGRetrievalUseCase:
    """Orchestrates the self-correcting GraphRAG retrieval pipeline.

    This use case serves as the central coordinator for the retrieval slice,
    delegating logic to a compiled LangGraph state machine. It implements a
    cyclical feedback loop to ensure high-quality context generation.

    Pipeline Role:
        Coordinates the transition between raw vector search and refined context
        assembly. It is responsible for multi-stage retrieval, grading, and
        query rewriting.

    Design Goals:
        - Self-Correction: Automatically detects poor retrieval results and
          attempts to rephrase the query to improve recall.
        - Hybrid Context: Combines linear document chunks with non-linear
          graph-based entity relationships.
        - Accuracy: Grader-based filtering ensures the LLM generator is only
          provided with relevant evidence, reducing hallucinations.

    Retrieval Loop (LangGraph Stages):
        1. **Retrieve**: Executes hybrid BM25 + HNSW chunk search and traverses
           the knowledge graph for relevant entities/relations.
        2. **Grade**: Evaluates retrieved snippets for direct relevance to the
           query; irrelevant chunks are pruned to save context window.
        3. **Rewrite** (Conditional): If no relevant documents remain, the LLM
           rewrites the query for a second retrieval attempt.
        4. **Generate**: Finalizes the context payload for the generation slice.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        graph_store: GraphStore,
        embedder: Embedder,
        entity_linker: EntityLinker,
        reranker: Reranker,
        generator: LLMGenerator,
        guardrail: FactualityGuardrail | None = None,
    ) -> None:
        """Initialize the retrieval use case with required ports.

        Args:
            document_store: Access to document metadata and vector chunks.
            graph_store: Access to entity-relationship data and traversal logic.
            embedder: Transformer model for query vectorization.
            entity_linker: Logic for mapping query strings to graph nodes.
            reranker: Logic for scoring and sorting context snippets.
            generator: LLM used for document grading and query rewriting.
            guardrail: Optional DeBERTa-based factuality checker.
        """
        self.document_store = document_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.entity_linker = entity_linker
        self.reranker = reranker
        self.generator = generator
        self.guardrail = guardrail

        # Compile the graph once and reuse it across requests to avoid overhead
        self._compiled_graph = build_rag_graph(
            store=self.document_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            entity_linker=self.entity_linker,
            generator=self.generator,
            reranker=self.reranker,
        )

    async def execute(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = 4,
        notebook_ids: list[str] | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        """Execute the self-correcting retrieval pipeline.

        Invokes the pre-compiled LangGraph for the given query. The graph
        handles embedding, hybrid search, grading, optional rewriting, and
        final context assembly.

        Args:
            query: The user's natural language question.
            history: Optional conversation history.
            top_k: Number of context snippets to return.
            notebook_ids: Optional list of notebook IDs to filter context.

        Returns:
            A dictionary containing context snippets under the 'documents' key.
        """
        logger.info(
            "[RETRIEVAL] Starting self-correcting RAG for: %s (notebooks: %s)",
            query,
            notebook_ids,
        )

        initial_state: RAGState = {
            "question": query,
            "history": history,
            "documents": [],
            "generation": [],
            "rewrite_count": 0,
            "top_k": top_k,
            "notebook_ids": notebook_ids,
        }

        final_state: RAGState = await self._compiled_graph.ainvoke(initial_state)  # type: ignore[assignment,attr-defined]
        documents: list[dict[str, object]] = final_state.get("documents", [])

        logger.info(
            "[RETRIEVAL] Pipeline complete: %d snippets returned",
            len(documents),
        )
        return {"documents": documents}
