"""Cross-encoder reranking logic for retrieval refinement.

This module provides implementations for the Reranker port, specifically
using ModernBERT or other cross-encoders to score and sort candidate
context snippets based on their relevance to the query.
"""

import logging
from typing import Any, cast

import anyio
import anyio.to_thread
from sentence_transformers import CrossEncoder

from app.core.interfaces import Reranker, RerankResult

logger = logging.getLogger(__name__)


class ModernBertReranker(Reranker):
    """Reranker using Alibaba's GTE-Reranker (ModernBERT) model.

    Pipeline Role:
        Final stage of the retrieval slice. Takes the top-N candidate snippets
        from hybrid search (Vector + BM25) and re-ranks them using a
        computationally expensive but highly accurate cross-attention mechanism.

    Design Goals:
        - Precision: Cross-encoders provide superior relevance scoring compared
          to bi-encoders by processing the query and document together.
        - Refinement: Acts as a quality filter, discarding snippets that do not
          directly support the user's information need.

    Implementation Details:
        - Model: 'Alibaba-NLP/gte-reranker-modernbert-base'.
        - Strategy: Scores (Query, Context) pairs, returning results sorted by
          relevance score (descending).
        - Optimization: Uses anyio.to_thread to prevent blocking the event loop
          during CPU/GPU inference.
    """

    model: CrossEncoder | None

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Initialize the cross-encoder model.

        Args:
            model_name: HuggingFace model ID or local path.
            device: Hardware device to use ('cpu', 'cuda').
        """
        logger.info("[RERANKER] Loading cross-encoder: %s on %s", model_name, device)
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info("[RERANKER] Model loaded successfully.")
        except Exception as e:
            logger.error("[RERANKER] Failed to load model: %s", e)
            self.model = None

    async def rerank(self, query: str, texts: list[str], top_k: int = 5) -> list[RerankResult]:
        """Re-rank context strings against the query.

        Args:
            query: The user's search query.
            texts: List of candidate context strings.
            top_k: Number of top results to return.

        Returns:
            A list of results with 'text' and 'score', ranked by score.
        """
        model = self.model
        if not model or not texts:
            # Fallback: return first N as-is if model failed or no texts
            return [{"text": t, "score": 1.0} for t in texts[:top_k]]

        logger.debug("[RERANKER] Reranking %d texts for query: %s", len(texts), query)

        # Cross-encoder expects pairs of (query, text)
        pairs = [(query, text) for text in texts]
        # Run the blocking predict() in a worker thread to avoid blocking the event loop
        # Casting pairs to Any to avoid complex sentence-transformers union type mismatches
        scores = await anyio.to_thread.run_sync(model.predict, cast(Any, pairs))

        # Combine, sort, and slice
        results: list[RerankResult] = []
        for text, score in zip(texts, scores, strict=True):
            results.append({"text": text, "score": float(score)})

        # Sort by score descending
        results.sort(key=lambda x: float(x["score"]), reverse=True)

        return results[:top_k]
