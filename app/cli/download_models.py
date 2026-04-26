"""Download embedding model artifacts into the configured models directory.

This script uses the HuggingFace Hub to download pre-trained model weights
and configurations required for local embedding generation.
"""

from __future__ import annotations

import logging
import sys

from huggingface_hub import snapshot_download

from app.config import settings

# Configure standalone logging for the CLI script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def download_models() -> None:
    """Download the configured embedding model to the local models directory.

    This function creates the target directory if it doesn't exist and
    uses `snapshot_download` to fetch the model artifacts, ignoring
    unnecessary framework-specific files.
    """
    model_id = settings.embedding_model_id
    target_dir = settings.models_dir / model_id
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[CLI] Downloading model '%s' to '%s'...", model_id, target_dir)

    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    logger.info("[CLI] Model downloaded successfully to: %s", local_dir)


def main() -> None:
    """CLI entry point for downloading local model artifacts."""
    try:
        download_models()
    except Exception as e:
        logger.error("[CLI] Failed to download models: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
