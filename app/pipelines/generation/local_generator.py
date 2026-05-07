"""Infrastructure implementation for Local LLM Generation via llama.cpp."""

import asyncio
import logging
import re

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage

from app.core.interfaces import LLMGenerator
from app.pipelines.generation.generator import map_history_to_messages

logger = logging.getLogger(__name__)


class LocalLlamaGenerator(LLMGenerator):
    """Generator using local GGUF models natively via llama-cpp-python.

    Implementation Details:
        - Uses 'langchain-community' ChatLlamaCpp.
        - Optimized for CPU inference (specifically 6 physical cores).
    """

    llm: ChatLlamaCpp | None = None

    def __init__(self, model_path: str) -> None:
        """Initialize the local generator.

        Args:
            model_path: Absolute or relative path to the .gguf model file.
        """
        self._lock = asyncio.Lock()
        try:
            self.llm = ChatLlamaCpp(
                model_path=model_path,
                temperature=0.5,
                max_tokens=1024,  # Reduced for stability during debug
                n_ctx=4096,  # 4096 is safer for many CPU builds
                n_threads=4,  # More conservative thread count
                n_batch=128,
                n_gpu_layers=0,
                use_mlock=False,
                verbose=True,
            )
        except Exception as e:
            logger.error("Failed to load local model at %s: %s", model_path, e)
            self.llm = None

    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a response using the local model."""
        if not self.llm:
            return "Local model is not initialized."

        async with self._lock:
            messages = map_history_to_messages(history)
            messages.append(HumanMessage(content=prompt))

            try:
                response = await self.llm.ainvoke(messages)
                raw = str(response.content)
                # Strip chain-of-thought blocks emitted by reasoning models
                # (e.g. Qwen3, DeepSeek-R1) before returning to the caller.
                cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return cleaned
            except Exception as e:
                logger.error("Local LLM generation failed: %s", e)
                return f"I'm sorry, I encountered an error: {e}"
