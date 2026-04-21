"""Cross-Encoder reranker for retrieval result refinement."""

from __future__ import annotations

from typing import Any

from sentence_transformers.cross_encoder import CrossEncoder


class CrossEncoderReranker:
    """Reranks a set of candidate passages against a query using a Cross-Encoder.

    Args:
        model_path: Local path (or HuggingFace model ID) of the Cross-Encoder model.
    """

    def __init__(self, model_path: str) -> None:
        self._model: CrossEncoder = CrossEncoder(model_path)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score and sort *candidates* by relevance to *query*.

        Args:
            query: The user query string.
            candidates: List of candidate dicts, each with at least a ``"text"`` key.
            top_k: Maximum number of results to return.

        Returns:
            A list of candidate dicts sorted by Cross-Encoder score (descending),
            truncated to *top_k*.  Each dict has an added ``"score"`` key.
        """
        if not candidates:
            return []

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores: list[float] = self._model.predict(pairs)

        scored = [
            {**c, "score": float(s)} for c, s in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
