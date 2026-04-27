"""HuggingFace implementation for text embeddings.

This module provides an implementation of the Embedder port using HuggingFace
Transformer models (e.g., BGE) for local, high-quality vector generation.
"""

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModel, AutoTokenizer

from app.domain.ports import Embedder

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(Embedder):
    """Embedder using HuggingFace transformers models locally.

    Generates dense vector representations of text suitable for semantic
    search and node description similarity within the knowledge graph.
    """

    def __init__(
        self, model_name: str = "BAAI/bge-large-en-v1.5", device: str | None = None
    ) -> None:
        """Initialize the tokenizer and model with optional quantization.

        Args:
            model_name: The name or path of the transformer model to use.
            device: Optional torch device (e.g., 'cuda', 'cpu').
        """
        from app.config import settings

        self.device = device or settings.device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if settings.quantization_enabled and self.device == "cpu":
            if settings.quantization_backend == "openvino":
                self._init_openvino(settings)
            else:
                self._init_torch_quantization(settings)
        else:
            self._init_standard_model()

        # BGE specific query prefix for asymmetric retrieval
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def _init_openvino(self, settings: "Settings") -> None:
        """Initialize the model using OpenVINO for high-performance CPU inference."""
        from optimum.intel.openvino import OVModelForFeatureExtraction

        logger.info("Initializing OpenVINO quantization for %s", self.model_name)
        # Use the models_dir/ov as a cache for exported IR models
        ov_path = settings.models_dir / "ov" / self.model_name.replace("/", "_")

        # Check if the model has already been exported to IR
        has_ir = (ov_path / "openvino_model.xml").exists()

        try:
            if has_ir:
                logger.info("Loading pre-exported OpenVINO model from %s", ov_path)
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    str(ov_path),
                    export=False,
                    compile=True,
                    device="CPU",
                    attn_implementation="eager",
                )
            else:
                logger.info("Exporting %s to OpenVINO IR at %s", self.model_name, ov_path)
                ov_path.mkdir(parents=True, exist_ok=True)
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    self.model_name,
                    export=True,
                    compile=True,
                    load_in_8bit=(settings.ov_precision == "int8"),
                    device="CPU",
                    cache_dir=str(ov_path),
                    attn_implementation="eager",
                )
                # Save the model to the ov_path for future use
                self.model.save_pretrained(str(ov_path))
        except Exception as e:
            logger.warning(
                "Failed to initialize OpenVINO for %s: %s. Falling back to standard model.",
                self.model_name,
                e,
            )
            self._init_standard_model()

    def _init_torch_quantization(self, settings: "Settings") -> None:
        """Initialize the model using standard PyTorch dynamic quantization."""
        logger.info("Initializing PyTorch dynamic quantization for %s", self.model_name)
        base_model = AutoModel.from_pretrained(self.model_name)
        self.model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.model.to(self.device)
        self.model.eval()

    def _init_standard_model(self) -> None:
        """Initialize the model using standard PyTorch."""
        logger.info(
            "Initializing standard PyTorch model for %s on %s", self.model_name, self.device
        )
        self.model = AutoModel.from_pretrained(self.model_name)
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning("Failed to move model to %s: %s", self.device, e)
            self.device = "cpu"
            self.model.to(self.device)
        self.model.eval()

        # BGE specific query prefix for asymmetric retrieval
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        """Internal helper to compute embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of vector embeddings (lists of floats).
        """
        with torch.no_grad():
            encoded_input = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            model_output = self.model(**encoded_input)

            # Perform CLS pooling (BGE uses the first token)
            sentence_embeddings = model_output[0][:, 0]

            # Normalize embeddings to unit length (cosine similarity becomes dot product)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings.cpu().tolist()

    async def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text string.

        Args:
            text: Input text string.

        Returns:
            Vector embedding.
        """
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of text strings.

        Args:
            texts: List of input text strings.

        Returns:
            List of vector embeddings.
        """
        if not texts:
            return []

        # Batch processing to avoid GPU/CPU memory issues (OOM)
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            embeddings = self._get_embedding(batch_texts)
            all_embeddings.extend(embeddings)

        return all_embeddings
