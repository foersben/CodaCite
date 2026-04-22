"""Download embedding model artifacts into the configured models directory."""

from __future__ import annotations

import logging
import sys

from huggingface_hub import snapshot_download

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def download_models() -> None:
    """Download the configured embedding model to the configured local models directory."""
    model_id = settings.embedding_model_id
    target_dir = settings.models_dir / model_id
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading model '%s' to '%s'...", model_id, target_dir)

    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    logger.info("Model downloaded successfully to: %s", local_dir)


def main() -> None:
    """CLI entry point for downloading local model artifacts."""
    download_models()


if __name__ == "__main__":
    main()
