"""Infrastructure implementation for Local LLM Generation via llama.cpp."""

import asyncio
import logging
import re
from collections.abc import AsyncGenerator

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage

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

    async def generate_stream(
        self, prompt: str, context: list[str], history: list[dict[str, str]] | None = None
    ) -> AsyncGenerator[str]:
        """Stream a response using the local model with citation formatting."""
        if not self.llm:
            yield "Local model is not initialized."
            return

        async with self._lock:
            system_prompt = (
                "You are a helpful AI assistant called CodaCite. You answer questions based on the provided document context.\n"
                "You must cite the exact source of every factual claim you make. Use the exact Chunk ID provided in the context blocks, enclosed in brackets like this: [chunk_123].\n\n"
                "### DOCUMENT CONTEXT:\n" + "\n\n".join(context)
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

            messages.append(HumanMessage(content=prompt))

            try:
                # State machine to strip <think>…</think> chain-of-thought blocks.
                #
                # Sentinel protocol: instead of silently dropping think tokens (which
                # leaves the user staring at a blank screen for 1-3 minutes), we emit
                # two special sentinel strings that the caller (chat.py) translates into
                # SSE control events so the frontend can show a "Reasoning…" indicator.
                #
                # Sentinels use null-byte delimiters — they can never appear in LLM output.
                _SENTINEL_START = "\x00THINKING_START\x00"
                _SENTINEL_END = "\x00THINKING_END\x00"

                in_think: bool = False
                think_started: bool = False
                buf: str = ""

                _OPEN_TAG = "<think>"
                _CLOSE_TAG = "</think>"
                # Max chars to hold in buffer while scanning for a partial tag
                _MAX_BUF = max(len(_OPEN_TAG), len(_CLOSE_TAG)) + 4

                async for chunk in self.llm.astream(messages):
                    token: str = str(chunk.content)
                    if not token:
                        continue

                    buf += token

                    if in_think:
                        if _CLOSE_TAG in buf:
                            buf = buf.split(_CLOSE_TAG, 1)[1]
                            in_think = False
                            yield _SENTINEL_END  # signal: reasoning finished
                            # fall through so post-think text is flushed below
                        else:
                            if len(buf) > _MAX_BUF:
                                buf = buf[-_MAX_BUF:]
                            continue

                    # Not (or no longer) in a think block
                    if _OPEN_TAG in buf:
                        pre, buf = buf.split(_OPEN_TAG, 1)
                        if pre:
                            yield pre
                        if not think_started:
                            yield _SENTINEL_START  # signal: reasoning begun
                            think_started = True
                        in_think = True
                        if _CLOSE_TAG in buf:
                            buf = buf.split(_CLOSE_TAG, 1)[1]
                            in_think = False
                            yield _SENTINEL_END
                        else:
                            buf = buf[-_MAX_BUF:] if len(buf) > _MAX_BUF else buf
                            continue

                    # Clean prose — yield what we safely can
                    if buf.endswith("<"):
                        pass  # might be start of <think>, hold it
                    elif "<" in buf:
                        safe, tail = buf.rsplit("<", 1)
                        buf = "<" + tail
                        if safe:
                            yield safe
                    else:
                        yield buf
                        buf = ""

                # Final flush
                if buf and not in_think:
                    if _OPEN_TAG in buf:
                        pre = buf.split(_OPEN_TAG, 1)[0]
                        if pre:
                            yield pre
                    else:
                        yield buf

            except Exception as e:
                logger.error("Local LLM streaming failed: %s", e)
                yield f"I'm sorry, I encountered an error: {e}"
