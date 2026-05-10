"""Intent classification for adaptive RAG routing.

This module provides logic to classify user queries into intents like 'summarize'
or 'qa', enabling the chat pipeline to bypass complex retrieval when a global
summary is requested.
"""

import re


class QueryRouter:
    """Classifies user queries to determine the optimal retrieval strategy.

    Uses fast regex-based heuristics to detect broad summarization requests
    vs specific questions.
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
            query: The user's input text.

        Returns:
            'summarize' if keywords match, otherwise 'qa'.
        """
        if self._summarize_pattern.search(query):
            return "summarize"
        return "qa"
