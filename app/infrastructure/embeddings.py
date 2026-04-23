"""HuggingFace implementation for text embeddings."""

import logging

import torch
from transformers import AutoModel, AutoTokenizer

from app.domain.ports import Embedder

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(Embedder):
    """Embedder using HuggingFace transformers models locally."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None) -> None:
        """Initialize the tokenizer and model."""
        if device is None:
            from app.config import settings
            self.device = settings.device
        else:
            self.device = device

        logger.info("Initializing HuggingFaceEmbedder with model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Ensure we move the model to the target device
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning("Failed to move model to %s, falling back to cpu: %s", self.device, e)
            self.device = "cpu"
            self.model.to(self.device)
            
        self.model.eval()

        # BGE specific query prefix
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        """Internal helper to compute embeddings for a list of texts."""
        with torch.no_grad():
            encoded_input = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            model_output = self.model(**encoded_input)

            # Perform CLS pooling (BGE uses the first token)
            sentence_embeddings = model_output[0][:, 0]

            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings.cpu().tolist()

    async def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text string."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of text strings."""
        if not texts:
            return []

        # Batch processing to avoid OOM
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            embeddings = self._get_embedding(batch_texts)
            all_embeddings.extend(embeddings)

        return all_embeddings
