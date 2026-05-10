"""Use case for performing RAG-based chat conversations.

This module coordinates the retrieval of document context and graph knowledge
to generate grounded responses for user queries while maintaining conversation history.
"""

import json
import logging
from collections.abc import AsyncGenerator

from app.core.interfaces import DocumentStore, LLMGenerator
from app.pipelines.generation.guardrails import FactualityGuardrail
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
        guardrail: FactualityGuardrail,
    ) -> None:
        """Initialize the chat use case with core services.

        Args:
            retrieval_use_case: The internal pipeline for finding context and answering.
            generator: The LLM interface for generating text.
            router: The intent classifier for routing.
            document_store: Access to global document summaries.
            guardrail: Factuality check guardrail.
        """
        self.retrieval_use_case = retrieval_use_case
        self.generator = generator
        self.router = router
        self.document_store = document_store
        self.guardrail = guardrail

    async def execute(
        self,
        query: str,
        notebook_ids: list[str] | None = None,
        top_k: int = 10,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str]:
        """Execute the chat pipeline to generate a grounded response.

        Args:
            query: The user's current question.
            notebook_ids: Optional list of notebook IDs to restrict retrieval.
            top_k: Number of snippets to retrieve.
            history: Optional list of previous messages in the conversation.

        Yields:
            Server-Sent Events (SSE) containing tokens and final citations payload.
        """
        logger.info(
            "[CHAT] Executing ChatUseCase for query: %s (Notebooks: %s)", query, notebook_ids
        )

        # Normalize history to empty list if None
        safe_history = history or []

        # 1. Classify intent
        intent = self.router.classify_intent(query)
        logger.info("[CHAT] Classified intent: %s", intent)

        # documents will be populated in QA branch for final citation payload
        documents: list[dict[str, object]] = []
        context_list: list[str] = []

        if intent == "summarize":
            # 2a. Retrieve global summaries (bypass RAG Graph)
            summaries = await self.document_store.get_document_summaries(
                active_notebook_ids=notebook_ids
            )
            for i, doc in enumerate(summaries, 1):
                text = f"[Document: {doc['filename']} - Global Summary]\n{doc['summary']}"
                context_list.append(text)
                # Create a synthetic document for citations payload
                documents.append(
                    {
                        "id": f"summary_{i}",
                        "document_id": "summary",
                        "filename": doc["filename"],
                        "text": doc["summary"],
                    }
                )
        else:
            # 2b. Use the full RAG Graph for QA to get context documents
            logger.info("[CHAT] Invoking GraphRAG pipeline for QA")
            result = await self.retrieval_use_case.execute(
                query, history=safe_history, top_k=top_k, notebook_ids=notebook_ids
            )
            documents = result["documents"]

            # Pre-fetch all document metadata once to build an ID → Document map,
            # avoiding repeated full-table fetches inside the loop.
            all_docs = await self.document_store.get_all_documents()
            doc_map = {str(d.id): d for d in all_docs}

            for chunk_doc in documents:
                # Ensure we have the filename for the citation metadata
                if chunk_doc.get("type") == "chunk" and chunk_doc.get("document_id"):
                    d_id = chunk_doc["document_id"]
                    target_doc = doc_map.get(str(d_id))
                    chunk_doc["filename"] = target_doc.filename if target_doc else "Unknown Source"

                context_list.append(str(chunk_doc.get("text", "")))

        # 3. Stream generation
        # Sentinel strings emitted by LocalLlamaGenerator to signal think-block boundaries.
        _SENTINEL_START = "\x00THINKING_START\x00"
        _SENTINEL_END = "\x00THINKING_END\x00"

        full_response = ""
        async for chunk in self.generator.generate_stream(
            query, context_list, history=safe_history
        ):
            if chunk == _SENTINEL_START:
                yield json.dumps({"thinking": True}) + "\n"
            elif chunk == _SENTINEL_END:
                yield json.dumps({"thinking": False}) + "\n"
            else:
                full_response += chunk
                yield json.dumps({"token": chunk}) + "\n"

        # 4. Run Guardrail post-stream
        context_str = "\n".join(context_list)
        is_verified = self.guardrail.verify(context_str, full_response)

        # 5. Yield final citations payload
        citations_payload = {
            "verified": is_verified,
            "warning": not is_verified,
            "documents": [
                {
                    "id": doc.get("id", doc.get("chunk_id", "")),
                    "document_id": doc.get("document_id", ""),
                    "filename": doc.get("filename", ""),
                    "text": doc.get("text", ""),
                }
                for doc in documents
            ],
        }
        yield f"event: citations\ndata: {json.dumps(citations_payload)}\n\n"
