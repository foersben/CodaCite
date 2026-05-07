"""Download model artifacts into the unified data directory.

Optimized for 16GB RAM constraints by using specific quantization formats
and excluding heavy unneeded tensors.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from app.core.bootstrap import COMMON_IGNORE_PATTERNS, REQUIRED_MODELS, is_model_cached
from app.core.config import settings

MODELS_DIR = settings.models_dir
HF_CACHE_DIR = MODELS_DIR / "hf_cache"

# Force HF to use our local cache directory
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def download_models() -> None:
    """Download the core models for CodaCite."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model download. Target: %s", MODELS_DIR)
    logger.info("HuggingFace Cache: %s", HF_CACHE_DIR)

    for model_type, info in REQUIRED_MODELS.items():
        is_snapshot = info.get("is_snapshot", True)
        repo_id = str(info.get("repo_id"))

        # Allow settings to override repo_id for LLM
        if model_type == "llm" and settings.local_llm_repo_id:
            repo_id = settings.local_llm_repo_id

        if is_snapshot:
            if is_model_cached(repo_id):
                logger.info("[CLI] %s already exists in cache. Skipping download.", repo_id)
                continue

            logger.info("[CLI] Downloading %s model: %s", model_type, repo_id)
            snapshot_download(
                repo_id=repo_id,
                ignore_patterns=COMMON_IGNORE_PATTERNS,
            )
        else:
            filename = str(info.get("filename"))
            # Allow settings to override filename for LLM
            if model_type == "llm" and settings.local_llm_path:
                filename = Path(settings.local_llm_path).name

            target_file = MODELS_DIR / filename
            if target_file.exists():
                logger.info("[CLI] %s already exists locally. Skipping download.", filename)
                continue

            logger.info("[CLI] Downloading %s file: %s from %s", model_type, filename, repo_id)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )

    logger.info("All models downloaded successfully.")


def main() -> None:
    """CLI entry point."""
    try:
        download_models()
    except Exception as e:
        logger.error("Failed to download models: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
