import logging

from sentence_transformers import CrossEncoder

from app.core.interfaces import Reranker, RerankResult

logger = logging.getLogger(__name__)


class ModernBertReranker(Reranker):
    """Reranker using Alibaba's GTE-Reranker (ModernBERT) model.

    Pipeline Role:
        Final stage of retrieval. Takes the top-N candidate snippets from
        hybrid search and re-ranks them using a computationally expensive but
        highly accurate cross-attention mechanism.

    Implementation Details:
        - Uses 'Alibaba-NLP/gte-reranker-modernbert-base'.
        - Optimized for CPU inference via quantization (if supported by backend).
        - Returns snippets sorted by score descending.
    """

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
        if not self.model or not texts:
            # Fallback: return first N as-is if model failed or no texts
            return [{"text": t, "score": 1.0} for t in texts[:top_k]]

        logger.debug("[RERANKER] Reranking %d texts for query: %s", len(texts), query)

        # Cross-encoder expects pairs of (query, text)
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)

        # Combine, sort, and slice
        results: list[RerankResult] = []
        for text, score in zip(texts, scores, strict=True):
            results.append({"text": text, "score": float(score)})

        # Sort by score descending
        results.sort(key=lambda x: float(x["score"]), reverse=True)

        return results[:top_k]
