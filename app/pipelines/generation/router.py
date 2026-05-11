"""Intent classification for adaptive RAG routing.

This module provides the QueryRouter, which classifies user inputs into specific
intents (e.g., 'summarize' or 'qa'). This enables the chat pipeline to select
the most efficient processing path, bypassing heavy retrieval when global
summaries are requested.
"""

import re


class QueryRouter:
    """Adaptive query router for intent-based pipeline steering.

    The router uses low-latency regex heuristics to distinguish between broad
    informational requests (summarization) and specific targeted questions (QA).
    This early classification is critical for maintaining high responsiveness
    in a CPU-constrained environment.
    """

    # Keywords that strongly suggest a summarization intent
    SUMMARIZE_KEYWORDS = [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\btl;dr\b",
        r"\boverview\b",
        r"\bbrief\b",
        r"\boutline\b",
    ]

    def __init__(self) -> None:
        """Initialize the router with compiled patterns."""
        self._summarize_pattern = re.compile("|".join(self.SUMMARIZE_KEYWORDS), re.IGNORECASE)

    def classify_intent(self, query: str) -> str:
        """Classify a query as either 'summarize' or 'qa'.

        Args:
            query: The user's input text to be classified.

        Returns:
            The intent string: 'summarize' for global requests, otherwise 'qa'.
        """
        if self._summarize_pattern.search(query):
            return "summarize"
        return "qa"
