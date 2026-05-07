"""Use case for performing RAG-based chat conversations.

This module coordinates the retrieval of document context and graph knowledge
to generate grounded responses for user queries while maintaining conversation history.
"""

import logging

from app.core.interfaces import DocumentStore, LLMGenerator
from app.pipelines.generation.rag_graph import DEFAULT_SYSTEM_PROMPT
from app.pipelines.generation.router import QueryRouter
from app.pipelines.retrieval.retrieval import GraphRAGRetrievalUseCase

logger = logging.getLogger(__name__)


class ChatUseCase:
    """Orchestrates the Retrieval-Augmented Generation (RAG) chat pipeline.

    This use case acts as the final assembly point for user interactions. It
    combines multi-modal context (vector chunks and graph concepts) with
    conversation history to produce grounded, citeable responses.

    Pipeline:
        1.  **Intent Classification**: Uses `QueryRouter` to detect if the
            query is a broad summarization request.
        2.  **Context Retrieval & Generation**:
            - If "summarize": Fetches pre-computed global summaries from `DocumentStore`
              and generates an answer manually.
            - If "qa": Invokes `GraphRAGRetrievalUseCase` which handles the full
              Retrieve → Rerank → (Rewrite) → Generate → (Verify) cycle.
        3.  **Result Delivery**: Returns the final answer.
    """

    def __init__(
        self,
        retrieval_use_case: GraphRAGRetrievalUseCase,
        generator: LLMGenerator,
        router: QueryRouter,
        document_store: DocumentStore,
    ) -> None:
        """Initialize the chat use case with core services.

        Args:
            retrieval_use_case: The internal pipeline for finding context and answering.
            generator: The LLM interface for generating text.
            router: The intent classifier for routing.
            document_store: Access to global document summaries.
        """
        self.retrieval_use_case = retrieval_use_case
        self.generator = generator
        self.router = router
        self.document_store = document_store

    async def execute(
        self,
        query: str,
        notebook_ids: list[str] | None = None,
        top_k: int = 10,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Execute the chat pipeline to generate a grounded response.

        Args:
            query: The user's current question.
            notebook_ids: Optional list of notebook IDs to restrict retrieval.
            top_k: Number of snippets to retrieve.
            history: Optional list of previous messages in the conversation.

        Returns:
            The LLM-generated response string.
        """
        logger.info(
            "[CHAT] Executing ChatUseCase for query: %s (Notebooks: %s)", query, notebook_ids
        )

        # Normalize history to empty list if None
        safe_history = history or []

        # 1. Classify intent
        intent = self.router.classify_intent(query)
        logger.info("[CHAT] Classified intent: %s", intent)

        if intent == "summarize":
            # 2a. Retrieve global summaries (bypass RAG Graph)
            summaries = await self.document_store.get_document_summaries(
                active_notebook_ids=notebook_ids
            )
            context_snippets = []
            for doc in summaries:
                context_snippets.append(
                    f"[Document: {doc['filename']} - Global Summary]\n{doc['summary']}"
                )

            context_text = (
                "\n\n".join(context_snippets) if context_snippets else "No relevant context found."
            )

            # Manual generation for summarization flow
            system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n### DOCUMENT CONTEXT:\n{context_text}"
            messages = list(safe_history)

            found_system = False
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = system_prompt
                    found_system = True
                    break
            if not found_system:
                messages.insert(0, {"role": "system", "content": system_prompt})

            response = await self.generator.agenerate(query, history=messages)
            return response

        # 2b. Use the full RAG Graph for QA
        logger.info("[CHAT] Invoking GraphRAG pipeline for QA")
        result = await self.retrieval_use_case.execute(
            query, history=safe_history, top_k=top_k, notebook_ids=notebook_ids
        )
        return result["answer"]
