"""Chat pipeline orchestration using Vertical Slice Architecture.

This module provides the ChatUseCase which coordinates retrieval, generation,
and state management for RAG-based conversations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import cast

from app.core.interfaces import DocumentStore, LLMGenerator
from app.pipelines.generation.guardrails import FactualityGuardrail
from app.pipelines.generation.router import QueryRouter
from app.pipelines.retrieval.retrieval import GraphRAGRetrievalUseCase

logger = logging.getLogger(__name__)


class ChatUseCase:
    """Orchestrates the chat experience by coordinating retrieval and generation slices.

    This class serves as the primary entry point for the chat pipeline, handling
    query routing, multi-stage retrieval (Vector + Graph), and final grounded
    response generation.

    Pipeline Role:
        Final assembly point for the user-facing chat experience. It routes
        intents to the appropriate retrieval path (Global Summary vs. Targeted
        QA) and manages the streaming lifecycle.

    Design Goals:
        - Responsiveness: Uses SSE streaming to provide immediate feedback.
        - Grounding: Enforces strict citation constraints through post-generation
          guardrails and verbatim quote verification.
        - Interpretability: Returns a rich citation payload allowing the
          frontend to highlight the exact source of every claim.
    """

    def __init__(
        self,
        retrieval_use_case: GraphRAGRetrievalUseCase,
        generator: LLMGenerator,
        router: QueryRouter,
        document_store: DocumentStore,
        guardrail: FactualityGuardrail,
    ) -> None:
        """Initialize the ChatUseCase with required store dependencies.

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

    async def chat(self, query: str) -> AsyncGenerator[str]:
        """Process a user query and stream a generated response.

        Args:
            query: The user's input question or command.

        Yields:
            Chunks of the generated response as they become available.
        """
        async for chunk in self.execute(query):
            yield chunk

    async def execute(
        self,
        query: str,
        notebook_ids: list[str] | None = None,
        top_k: int = 4,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str]:
        """Execute the chat pipeline to generate a grounded response.

        This method coordinates the five-step chat lifecycle:
        1. **Intent Routing**: Classifies the query (e.g., summarization vs. QA).
        2. **Context Synthesis**: Retrieves relevant documents or global summaries.
        3. **Token Streaming**: Yields real-time tokens and thinking sentinels.
        4. **Verification**: Runs factuality guardrails on the full response.
        5. **Citation Finalization**: Yields a JSON payload for source mapping.

        Args:
            query: The user's current question.
            notebook_ids: Optional list of notebook IDs to restrict retrieval.
            top_k: Number of snippets to retrieve.
            history: Optional list of previous messages in the conversation.

        Yields:
            Server-Sent Events (SSE) containing tokens, thinking state, and
            the final citations payload.
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

            # Pre-fetch document metadata to build an ID → Document map.
            # We collect document_ids directly, and for entities/relations, we use their source_chunk_ids.
            doc_ids: set[str] = {str(d["document_id"]) for d in documents if d.get("document_id")}

            # Map source_chunk_ids back to document_ids (doc_id is the prefix of chunk_id)
            for d in documents:
                if d.get("source_chunk_ids"):
                    for cid in cast(list[str], d["source_chunk_ids"]):
                        if "_" in cid:
                            doc_ids.add(cid.split("_")[0])

            relevant_docs = await self.document_store.get_documents_by_ids(list(doc_ids))
            doc_map = {str(d.id): d for d in relevant_docs}

            for doc_item in documents:
                # 1. Resolve filename from document_id
                d_id = doc_item.get("document_id")
                if not d_id and doc_item.get("source_chunk_ids"):
                    # Fallback: take first valid chunk's document prefix
                    for cid in cast(list[str], doc_item["source_chunk_ids"]):
                        if "_" in cid:
                            d_id = cid.split("_")[0]
                            break
                if d_id:
                    target_doc = doc_map.get(str(d_id))
                    doc_item["filename"] = target_doc.filename if target_doc else "Unknown Source"
                else:
                    doc_item["filename"] = "Knowledge Graph"

                context_list.append(cast(str, doc_item.get("text", "")))

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
