"""Use case for performing RAG-based chat conversations.

This module coordinates the retrieval of document context and graph knowledge
to generate grounded responses for user queries while maintaining conversation history.
"""

import logging

from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.ports import LLMGenerator

logger = logging.getLogger(__name__)


class ChatUseCase:
    """Orchestrates the Retrieval-Augmented Generation (RAG) chat pipeline.

    This use case acts as the final assembly point for user interactions. It
    combines multi-modal context (vector chunks and graph concepts) with
    conversation history to produce grounded, citeable responses.

    Pipeline:
        1.  **Context Retrieval**: Invokes `GraphRAGRetrievalUseCase` to find
            relevant document fragments and graph nodes.
        2.  **Context Formatting**: Serializes retrieved results into a
            structured prompt block with source attribution.
        3.  **Prompt Engineering**: Constructs a system prompt that enforces
            groundedness and identifies the assistant as "CodaCite".
        4.  **Response Generation**: Calls the `LLMGenerator` (Gemini) to
            produce the final response based on the augmented context.
    """

    def __init__(
        self,
        retrieval_use_case: GraphRAGRetrievalUseCase,
        generator: LLMGenerator,
    ) -> None:
        """Initialize the chat use case with core services.

        Args:
            retrieval_use_case: The internal pipeline for finding context.
            generator: The LLM interface for generating text.
        """
        self.retrieval_use_case = retrieval_use_case
        self.generator = generator

    async def execute(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        notebook_ids: list[str] | None = None,
    ) -> str:
        """Execute the chat pipeline to generate a grounded response.

        Args:
            query: The user's current question.
            history: Optional list of previous messages in the conversation.
            notebook_ids: Optional list of notebook IDs to restrict retrieval.

        Returns:
            The LLM-generated response string.
        """
        logger.info(
            "[CHAT] Executing ChatUseCase for query: %s (Notebooks: %s)", query, notebook_ids
        )

        # 1. Retrieve context using GraphRAG
        # Find relevant chunks and graph elements
        retrieved_results = await self.retrieval_use_case.execute(
            query, top_k=10, notebook_ids=notebook_ids
        )

        context_snippets = []
        for res in retrieved_results:
            text = res.get("text", "")
            source = res.get("source") or res.get("document_id", "Unknown")
            context_snippets.append(f"[Source: {source}]\n{text}")

        context_text = (
            "\n\n".join(context_snippets) if context_snippets else "No relevant context found."
        )

        # 2. Construct System Prompt
        system_prompt = (
            "You are a helpful AI assistant called CodaCite. "
            "You answer questions based on the provided document context and conversation history. "
            "If the answer is not in the context, say you don't know based on the documents. "
            "Always be professional and concise.\n\n"
            f"### DOCUMENT CONTEXT:\n{context_text}"
        )

        # 3. Prepare messages for generation
        full_history = []
        if history:
            full_history = list(history)

        # Insert system message at the beginning if not present
        if not any(msg.get("role") == "system" for msg in full_history):
            full_history.insert(0, {"role": "system", "content": system_prompt})
        else:
            # Update existing system prompt if needed
            for msg in full_history:
                if msg.get("role") == "system":
                    msg["content"] = system_prompt
                    break

        # 4. Generate response
        logger.debug("[CHAT] Generating response from LLM...")
        response = await self.generator.agenerate(query, history=full_history)
        logger.info("[CHAT] Response generated successfully")

        return response
