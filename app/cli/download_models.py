"""Download embedding and generative model artifacts into the configured models directory.

This script uses the HuggingFace Hub to download pre-trained model weights
and configurations required for local GraphRAG ingestion and generation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add hf_hub_download to grab single GGUF files
from huggingface_hub import hf_hub_download, snapshot_download

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def download_models() -> None:
    """Download the configured models to the local directory."""

    # ---------------------------------------------------------
    # 1. Download Embedding Model (Vector Search)
    # ---------------------------------------------------------
    emb_model_id = settings.embedding_model_id
    emb_target_dir = settings.models_dir / emb_model_id
    emb_target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[CLI] Downloading embedding model '%s' to '%s'...", emb_model_id, emb_target_dir)

    # Use snapshot to get all the required config and token files
    emb_local_dir = snapshot_download(
        repo_id=emb_model_id,
        local_dir=str(emb_target_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    logger.info("[CLI] Embedding model downloaded successfully to: %s", emb_local_dir)

    # ---------------------------------------------------------
    # 2. Download Generative Model (Local Chat)
    # ---------------------------------------------------------
    if getattr(settings, "use_local_nlp_models", False) and getattr(settings, "local_llm_path", ""):
        repo_id = getattr(settings, "local_llm_repo_id", "")

        if not repo_id:
            logger.warning("[CLI] local_llm_repo_id is missing. Skipping LLM download.")
            return

        # Extract just the filename (e.g., 'qwen2.5-7b-instruct-q4_k_m.gguf') from the path
        local_path = Path(settings.local_llm_path)
        filename = local_path.name

        logger.info(
            "[CLI] Downloading generative model file '%s' from repo '%s'...", filename, repo_id
        )

        # Use hf_hub_download to get JUST the specific quantization file, not the whole repo
        llm_downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(settings.models_dir),
            local_dir_use_symlinks=False,  # Forces the actual file into ./models instead of a cache link
        )
        logger.info("[CLI] Generative model downloaded successfully to: %s", llm_downloaded_file)


def main() -> None:
    """CLI entry point for downloading local model artifacts."""
    try:
        download_models()
    except Exception as e:
        logger.error("[CLI] Failed to download models: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
