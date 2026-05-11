"""LLM generation implementation using Google Gemini.

This module provides the GeminiGenerator class for interfacing with the
Google Generative AI SDK to produce RAG-enriched responses.
"""

import logging
from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.interfaces import LLMGenerator

logger = logging.getLogger(__name__)


def map_history_to_messages(history: list[dict[str, str]] | None) -> list[BaseMessage]:
    """Map raw chat history dictionaries to LangChain message types.

    Args:
        history: List of dictionaries with 'role' and 'content'.

    Returns:
        List of LangChain HumanMessage, AIMessage, or SystemMessage objects.
    """
    messages: list[BaseMessage] = []
    if not history:
        return messages

    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages


class GeminiGenerator(LLMGenerator):
    """Infrastructure implementation for Google Gemini models via Vertex AI / AI Studio.

    This class coordinates the assembly of context-enriched prompts and manages
    streaming interaction with the Gemini API. It adheres to strict grounding
    rules, ensuring the model remains within the provided context window.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview") -> None:
        """Initialize the Gemini generator with API credentials and configuration.

        Args:
            api_key: Google AI Studio API key.
            model_name: Gemini model identifier.
                Defaults to 'gemini-3-flash-preview'.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,  # Lower temperature for higher factuality in RAG
        )

    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a response using Gemini.

        Args:
            prompt: The final formatted prompt (typically including context).
            history: Optional list of previous chat messages with 'role' and 'content'.

        Returns:
            The generated response string, or an error message on failure.
        """
        messages = map_history_to_messages(history)
        messages.append(HumanMessage(content=prompt))

        try:
            response = await self.llm.ainvoke(messages)
            response_content = response.content
            # LangChain's Gemini adapter may return content as either:
            # - A plain string: "Hello world"
            # - A list of content parts: [{"type": "text", "text": "...", ...}]
            if isinstance(response_content, list):
                text_parts = []
                for part in response_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "\n".join(text_parts) if text_parts else str(response_content)

            return response_content
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            return f"I'm sorry, I encountered an error: {e}"

    async def generate_stream(
        self, query: str, context: list[str], history: list[dict[str, str]] | None = None
    ) -> AsyncGenerator[str]:
        """Generate a streaming response based on a query and provided context.

        Args:
            query: The current user input.
            context: The retrieved context (e.g., document chunks or graph nodes).
            history: Optional conversation history for multi-turn chat.

        Yields:
            Chunks of the generated text response.
        """
        indexed_context = [f"[{i + 1}] {text}" for i, text in enumerate(context)]
        system_prompt = (
            "You are CodaCite, a high-precision document-grounded AI.\n"
            "Your task is to answer the user's question using ONLY the provided ### DOCUMENT CONTEXT below.\n\n"
            "STRICT RULES:\n"
            '1. GROUNDING: Use ONLY the provided context. If the answer is not in the context, state: "I am sorry, but the provided documents do not contain information to answer this question."\n'
            "2. CITATIONS: Every factual claim must be followed by a citation like [1], [2], etc., corresponding to the context block index.\n"
            '3. QUOTES: When citing specific evidence, you MUST provide a verbatim quote enclosed in double quotes, followed by the citation. Example: "The sky was a deep shade of indigo." [4]\n'
            "4. NO OUTSIDE KNOWLEDGE: Do not use any information not present in the provided context.\n\n"
            "### DOCUMENT CONTEXT:\n" + "\n\n".join(indexed_context)
        )

        messages = map_history_to_messages(history)
        found_system = False
        for msg in messages:
            if msg.type == "system":
                msg.content = system_prompt
                found_system = True
                break
        if not found_system:
            messages.insert(0, SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=query))

        try:
            async for chunk in self.llm.astream(messages):
                yield str(chunk.content)
        except Exception as e:
            logger.error("Gemini streaming failed: %s", e)
            yield f"I'm sorry, I encountered an error: {e}"
