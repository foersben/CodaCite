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

# Project root calculation
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"

# Force HF to use our local cache directory
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODELS_TO_DOWNLOAD = [
    {
        "repo_id": "Alibaba-NLP/gte-reranker-modernbert-base",
        "name": "reranker",
    },
    {
        "repo_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c",
        "name": "nli",
    },
    {
        "repo_id": "urchade/gliner_mediumv2.1",
        "name": "ner",
    },
    {
        "repo_id": "BAAI/bge-m3",
        "name": "embedding",
    },
]

# GGUF Model specific
LLM_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
LLM_FILE = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"


def download_models() -> None:
    """Download the core models for CodaCite."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model download. Target: %s", MODELS_DIR)
    logger.info("HuggingFace Cache: %s", HF_CACHE_DIR)

    # 1. Download the GGUF LLM
    logger.info("Downloading LLM: %s (%s)", LLM_REPO, LLM_FILE)
    hf_hub_download(
        repo_id=LLM_REPO, filename=LLM_FILE, local_dir=str(MODELS_DIR), local_dir_use_symlinks=False
    )

    # 2. Download the other support models
    # We exclude legacy formats to prefer safetensors and save disk/RAM
    ignore_patterns = [
        "*.bin",
        "*.pth",
        "*.pt",
        "*.onnx",
        "*.msgpack",
        "*.h5",
        "*.ot",
        "flax_model*",
        "tf_model*",
    ]

    for m in MODELS_TO_DOWNLOAD:
        repo_id = m["repo_id"]
        target_path = MODELS_DIR / repo_id
        logger.info("Downloading %s model: %s -> %s", m["name"], repo_id, target_path)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_path),
            ignore_patterns=ignore_patterns,
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
