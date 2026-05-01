"""Infrastructure implementation for Local LLM Generation via llama.cpp."""

import logging
import re
from typing import Any

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.domain.ports import LLMGenerator

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
        try:
            self.llm = ChatLlamaCpp(
                model_path=model_path,
                temperature=0.5,
                max_tokens=2048,  # mypy expects int, 2048 is a safe default for RAG responses
                n_ctx=8192,  # RAG requires a large context window
                n_threads=6,  # OPTIMIZATION: Matches i7-10750H physical cores
                n_batch=512,  # Process prompt tokens in chunks
                n_gpu_layers=0,  # CPU only
                use_mlock=True,  # Prevents the OS from swapping the model to the hard drive
                verbose=False,
                # LangChain bypass: pass unsupported llama-cpp args via model_kwargs
                model_kwargs={
                    "type_k": 8,  # 8-bit KV cache
                    "type_v": 8,
                    "flash_attn": True,  # 🚀 The biggest speedup for long context
                },
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
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

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
