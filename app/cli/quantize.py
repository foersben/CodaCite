"""CLI tool to pre-quantize the embedding model to OpenVINO format.

This script initializes the HuggingFaceEmbedder, which automatically
handles the export and quantization of the model to the local models directory.
"""

from __future__ import annotations

import logging
import sys

from app.config import settings
from app.infrastructure.embeddings import HuggingFaceEmbedder

# Configure standalone logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def quantize_model() -> None:
    """Initialize the embedder to trigger OpenVINO quantization."""
    model_id = settings.embedding_model_id
    logger.info("[CLI] Starting optimization for model: %s", model_id)

    # Initializing the embedder will trigger _init_openvino()
    # which performs the export and quantization if not already present.
    try:
        HuggingFaceEmbedder(model_name=model_id)
        logger.info(
            "[CLI] Optimization complete. Model is ready at: %s", settings.models_dir / "ov"
        )
    except Exception as e:
        logger.error("[CLI] Optimization failed: %s", e)
        raise


def main() -> None:
    """CLI entry point for model quantization."""
    try:
        quantize_model()
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
