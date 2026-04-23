"""Infrastructure implementation for LLM Generation."""

import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.domain.ports import LLMGenerator

logger = logging.getLogger(__name__)

class GeminiGenerator(LLMGenerator):
    """Generator using Google GenAI (Gemini) via LangChain."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview") -> None:
        """Initialize the generator."""
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )

    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a response using Gemini."""
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
