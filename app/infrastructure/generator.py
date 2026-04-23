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


class GeminiGenerator(LLMGenerator):
    """Generator using Google GenAI (Gemini) via LangChain.

    Handles conversational history and prompt execution for document-grounded
    question answering.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview") -> None:
        """Initialize the generator.

        Args:
            api_key: Google AI Studio API key.
            model_name: Gemini model identifier.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )

    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a response using Gemini.

        Args:
            prompt: The formatted prompt to send to the model.
            history: Optional list of previous chat messages.

        Returns:
            The generated response string.
        """
        messages: list[Any] = []

        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))

        messages.append(HumanMessage(content=prompt))

        try:
            response = await self.llm.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            return f"I'm sorry, I encountered an error: {e}"
