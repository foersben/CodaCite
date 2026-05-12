"""Infrastructure implementation for Local LLM Generation via llama.cpp."""

import asyncio
import logging
import re
from collections.abc import AsyncGenerator

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.interfaces import LLMGenerator
from app.pipelines.generation.generator import map_history_to_messages

logger = logging.getLogger(__name__)


class LocalLlamaGenerator(LLMGenerator):
    """Infrastructure implementation for local Large Language Model (LLM) generation.

    This class interfaces with GGUF-formatted models natively via `llama-cpp-python`
    to provide hardware-accelerated (CPU-first) inference. It incorporates a serial
    locking mechanism to manage thread-safe access to the shared model instance and
    includes specialized post-processing logic to strip chain-of-thought blocks.

    Attributes:
        llm: The underlying LangChain-wrapped ChatLlamaCpp instance.
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
                max_tokens=4096,
                n_ctx=settings.local_llm_n_ctx,
                n_threads=settings.n_threads,
                n_batch=settings.local_llm_n_batch,
                n_gpu_layers=settings.local_llm_gpu_layers,
                use_mlock=False,
                verbose=True,
            )
        except Exception as e:
            logger.error("Failed to load local model at %s: %s", model_path, e)
            self.llm = None

    async def agenerate(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """Generate a complete text response using the local model.

        This method handles non-streaming generation, typically used for short
        synchronous tasks or batch processing. It implements a 60-second timeout
        to prevent deadlocks during heavy inference.

        Args:
            prompt: The text prompt to fulfill.
            history: Optional conversation history to maintain context.

        Returns:
            The generated text response, cleaned of internal reasoning blocks.
        """
        if not self.llm:
            return "Local model is not initialized."

        try:
            # Configurable timeout to prevent deadlocks if llama-cpp-python hangs
            async with asyncio.timeout(settings.local_llm_timeout):
                async with self._lock:
                    messages = map_history_to_messages(history)
                    messages.append(HumanMessage(content=prompt))

                    try:
                        response = await self.llm.ainvoke(messages)
                        raw = str(response.content)
                        # Strip chain-of-thought blocks emitted by reasoning models
                        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                        return cleaned
                    except Exception as e:
                        logger.error("Local LLM generation failed: %s", e)
                        return f"I'm sorry, I encountered an error: {e}"
        except TimeoutError:
            logger.error("Local LLM generation timed out after %ds", settings.local_llm_timeout)
            return "The request timed out while waiting for local inference resources."

    async def generate_stream(
        self, prompt: str, context: list[str], history: list[dict[str, str]] | None = None
    ) -> AsyncGenerator[str]:
        """Generate a streaming response with strict document grounding.

        This method implements the core generation logic for RAG, enforcing a
        verbatim quote policy and citation requirements via a structured
        system prompt. It utilizes a state machine to strip '<think>' blocks
        emitted by reasoning models in real-time.

        Args:
            prompt: The user query.
            context: A list of retrieved document snippets to ground the answer.
            history: Optional conversation history.

        Yields:
            Chunks of the generated text as they are produced by the model.
        """
        if not self.llm:
            yield "Local model is not initialized."
            return

        try:
            # Wait at most settings.local_llm_timeout for the inference lock. Local models are serial.
            async with asyncio.timeout(settings.local_llm_timeout):
                async with self._lock:
                    # Format context with numeric indices for the LLM to cite
                    context_formatted = [f"[{i + 1}] {text}" for i, text in enumerate(context)]
                    system_prompt = (
                        "You are CodaCite, a high-precision document-grounded AI.\n"
                        "Your task is to answer the user's question using ONLY the provided ### DOCUMENT CONTEXT below.\n\n"
                        "STRICT RULES:\n"
                        '1. GROUNDING: Use ONLY the provided context. If the answer is not in the context, state: "I am sorry, but the provided documents do not contain information to answer this question."\n'
                        "2. CITATIONS: Every factual claim must be followed by a citation like [1], [2], etc., corresponding to the context block index.\n"
                        '3. QUOTES: When citing specific evidence, you MUST provide a verbatim quote enclosed in double quotes, followed by the citation. Example: "The sky was a deep shade of indigo." [4]\n'
                        "4. NO OUTSIDE KNOWLEDGE: Do not use any information not present in the provided context.\n\n"
                        "### DOCUMENT CONTEXT:\n" + "\n\n".join(context_formatted)
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
                        _SENTINEL_START = "\x00THINKING_START\x00"
                        _SENTINEL_END = "\x00THINKING_END\x00"

                        in_think: bool = False
                        think_started: bool = False
                        buf: str = ""

                        _OPEN_TAG = "<think>"
                        _CLOSE_TAG = "</think>"
                        _MAX_BUF = max(len(_OPEN_TAG), len(_CLOSE_TAG)) + 4

                        async for chunk in self.llm.astream(messages):
                            raw_token = chunk.content
                            if not raw_token:
                                continue

                            # Handle list of content parts if emitted by some providers/wrappers
                            if isinstance(raw_token, list):
                                token = ""
                                for part in raw_token:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        token += part["text"]
                                    elif isinstance(part, str):
                                        token += part
                            else:
                                token = raw_token

                            buf += token

                            if in_think:
                                if _CLOSE_TAG in buf:
                                    buf = buf.split(_CLOSE_TAG, 1)[1]
                                    in_think = False
                                    yield _SENTINEL_END
                                else:
                                    if len(buf) > _MAX_BUF:
                                        buf = buf[-_MAX_BUF:]
                                    continue

                            if _OPEN_TAG in buf:
                                pre, buf = buf.split(_OPEN_TAG, 1)
                                if pre:
                                    yield pre
                                if not think_started:
                                    yield _SENTINEL_START
                                    think_started = True
                                in_think = True
                                if _CLOSE_TAG in buf:
                                    buf = buf.split(_CLOSE_TAG, 1)[1]
                                    in_think = False
                                    yield _SENTINEL_END
                                else:
                                    buf = buf[-_MAX_BUF:] if len(buf) > _MAX_BUF else buf
                                    continue

                            if buf.endswith("<"):
                                pass
                            elif "<" in buf:
                                safe, tail = buf.rsplit("<", 1)
                                buf = "<" + tail
                                if safe:
                                    yield safe
                            else:
                                yield buf
                                buf = ""

                        # CLEANUP: Handle dangling buffers or incomplete thinking blocks
                        if in_think:
                            logger.warning(
                                "Local LLM stream ended prematurely inside <think> block"
                            )
                            yield _SENTINEL_END
                        if buf:
                            # Final check for tags in the last buffer
                            if _OPEN_TAG in buf:
                                pre = buf.split(_OPEN_TAG, 1)[0]
                                if pre:
                                    yield pre
                            elif not in_think:
                                yield buf

                    except Exception as e:
                        logger.error("Local LLM streaming failed: %s", e)
                        yield f"I'm sorry, I encountered an error: {e}"
        except TimeoutError:
            logger.error("Local LLM stream timed out after %ds", settings.local_llm_timeout)
            yield "The request timed out while waiting for local inference resources."
