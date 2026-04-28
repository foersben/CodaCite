"""Infrastructure implementation for Local LLM Generation via llama.cpp."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatLlamaCpp

from app.domain.ports import LLMGenerator

logger = logging.getLogger(__name__)


class LocalLlamaGenerator(LLMGenerator):
    """Generator using local GGUF models natively via llama-cpp-python.

    Implementation Details:
        - Uses 'langchain-community' ChatLlamaCpp.
        - Optimized for CPU inference (specifically 6 physical cores).
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the local generator.

        Args:
            model_path: Absolute or relative path to the .gguf model file.
        """
        try:
            self.llm = ChatLlamaCpp(
                model_path=model_path,
                temperature=0.5,
                max_tokens=1024,
                n_ctx=8192,  # RAG requires a large context window
                n_threads=6,  # OPTIMIZATION: Match your i7-10750H physical cores
                n_batch=512,  # Process prompt tokens in chunks
                verbose=False,
            )
        except Exception as e:
            logger.error("Failed to load local model at %s: %s", model_path, e)
            self.llm = None

    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a response using the local model."""
        if not self.llm:
            return "Local model is not initialized."

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
            # Note: ChatLlamaCpp runs locally, so ainvoke operates in a threadpool
            response = await self.llm.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error("Local generation failed: %s", e)
            return f"I'm sorry, I encountered an error: {e}"
