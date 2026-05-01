"""Infrastructure implementation for LLM Generation.

This module provides an implementation of the LLMGenerator port using
Google Gemini (via LangChain) for chat and retrieval-augmented generation.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.domain.ports import LLMGenerator

logger = logging.getLogger(__name__)


def map_history_to_messages(history: list[dict[str, str]] | None) -> list[Any]:
    """Map raw chat history dictionaries to LangChain message types.

    Args:
        history: List of dictionaries with 'role' and 'content'.

    Returns:
        List of LangChain HumanMessage, AIMessage, or SystemMessage objects.
    """
    messages: list[Any] = []
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
    """Generator using Google GenAI (Gemini) via LangChain.

    Handles conversational history and prompt execution for document-grounded
    question answering (GraphRAG).

    Pipeline Role:
        Final stage of Retrieval. Takes the retrieved multi-hop context and the
        user query to generate a grounded, cited response.

    Implementation Details:
        - Uses 'langchain-google-genai' (ChatGoogleGenerativeAI).
        - Supports conversational state by mapping message roles to LangChain
          Message types (Human, AI, System).
        - Defaults to 'gemini-3-flash-preview' for speed and cost-efficiency.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview") -> None:
        """Initialize the generator.

        Args:
            api_key: Google AI Studio API key.
            model_name: Gemini model identifier.
                Defaults to 'gemini-3-flash-preview'.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
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

            return str(response_content)
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            return f"I'm sorry, I encountered an error: {e}"
