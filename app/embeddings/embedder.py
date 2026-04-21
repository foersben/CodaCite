"""Local embedding model using sentence-transformers (BAAI/bge-large-en-v1.5)."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import settings


class LocalEmbedder:
    """Wraps a locally-loaded SentenceTransformer model to produce dense embeddings.

    The model is loaded from a local directory on construction so that no
    external HTTP requests are made at runtime.

    Args:
        model_path: Path to the local model directory (e.g. ``./models/BAAI/bge-large-en-v1.5``).
            Defaults to the path derived from :data:`~app.config.settings`.
    """

    def __init__(self, model_path: str | None = None) -> None:
        if model_path is None:
            model_path = str(settings.embedding_model_path)
        self._model: SentenceTransformer = SentenceTransformer(model_path)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of *texts*.

        Args:
            texts: List of strings to embed.

        Returns:
            A list of float lists, one per input text.  Empty input returns
            an empty list.
        """
        if not texts:
            return []
        vectors = self._model.encode(texts)
        # Ensure we always return plain Python float lists (not numpy arrays)
        return [list(map(float, vec)) for vec in vectors]
