"""LLM-based intent router for classifying user queries."""

from __future__ import annotations

import enum
from typing import Any


class IntentType(enum.StrEnum):
    """Possible intent classifications for incoming user queries."""

    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    ACTION_EXECUTION = "action_execution"


class IntentRouter:
    """Classifies a user query as :attr:`~IntentType.KNOWLEDGE_RETRIEVAL` or
    :attr:`~IntentType.ACTION_EXECUTION` using an LLM-based semantic router.

    Args:
        llm: A LangChain-compatible chat model with a ``.invoke()`` method.
    """

    _PROMPT_TEMPLATE = (
        "You are an intent classification system.\n"
        "Given the user query below, respond with EXACTLY one of these labels:\n"
        "  knowledge_retrieval  – the user wants to find information or facts\n"
        "  action_execution     – the user wants to perform an action (e.g. send email)\n\n"
        "Respond with only the label, nothing else.\n\n"
        "Query: {query}"
    )

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def route(self, query: str) -> IntentType:
        """Classify *query* into an intent type.

        Args:
            query: The raw user query string.

        Returns:
            :class:`IntentType` – defaults to :attr:`~IntentType.KNOWLEDGE_RETRIEVAL`
            if the LLM response cannot be mapped to a known intent.
        """
        prompt = self._PROMPT_TEMPLATE.format(query=query)
        response = self._llm.invoke(prompt)
        raw = (response.content if hasattr(response, "content") else str(response)).strip().lower()

        for intent in IntentType:
            if intent.value in raw:
                return intent

        # Default fallback
        return IntentType.KNOWLEDGE_RETRIEVAL
