import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.ports import LLMGenerator

logger = logging.getLogger(__name__)

class ChatUseCase:
    """Use case for performing RAG chat with conversation history."""

    def __init__(
        self,
        retrieval_use_case: GraphRAGRetrievalUseCase,
        generator: LLMGenerator,
    ) -> None:
        """Initialize the chat usecase."""
        self.retrieval_use_case = retrieval_use_case
        self.generator = generator

    async def execute(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        """Execute the chat pipeline."""
        logger.info("Executing ChatUseCase for query: %s", query)

        # 1. Retrieve context using GraphRAG
        # We use the query to find relevant chunks and graph elements
        retrieved_results = await self.retrieval_use_case.execute(query, top_k=10)
        
        context_text = "\n\n".join([str(res["text"]) for res in retrieved_results])

        # 2. Construct System Prompt
        system_prompt = (
            "You are a helpful AI assistant called Enterprise Omni-Copilot. "
            "You answer questions based on the provided document context and conversation history. "
            "If the answer is not in the context, say you don't know based on the documents. "
            "Always be professional and concise.\n\n"
            f"### DOCUMENT CONTEXT:\n{context_text}"
        )

        # 3. Prepare messages for generation
        # We prepend the system prompt to the history or as a system message
        full_history = []
        if history:
            full_history = list(history)
        
        # Insert system message at the beginning if not present
        if not any(msg.get("role") == "system" for msg in full_history):
            full_history.insert(0, {"role": "system", "content": system_prompt})
        else:
            # Update existing system prompt if needed (simplified: just replace first)
            for msg in full_history:
                if msg.get("role") == "system":
                    msg["content"] = system_prompt
                    break

        # 4. Generate response
        response = await self.generator.agenerate(query, history=full_history)

        return response
