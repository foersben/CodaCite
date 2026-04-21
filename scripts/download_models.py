"""
Download BAAI/bge-large-en-v1.5 embedding model locally for offline use.

Run this script once during environment setup:
    uv run python scripts/download_models.py

The model files will be saved to ./models/BAAI/bge-large-en-v1.5
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_ID = "BAAI/bge-large-en-v1.5"
MODELS_DIR = Path(__file__).parent.parent / "models"


def download_models() -> None:
    """Download the BAAI/bge-large-en-v1.5 model to the local models directory."""
    target_dir = MODELS_DIR / MODEL_ID
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading model '%s' to '%s'...", MODEL_ID, target_dir)

    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    logger.info("Model downloaded successfully to: %s", local_dir)


if __name__ == "__main__":
    download_models()
