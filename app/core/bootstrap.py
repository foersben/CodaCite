"""Application bootstrap and initialization logic.

This module handles the initial setup of the CodaCite system, including
downloading required NLP models from HuggingFace and tracking the
overall readiness of the infrastructure.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from app.core.config import settings

logger = logging.getLogger(__name__)


class BootstrapStatus(StrEnum):
    """Status of the application bootstrap process."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


# Global state to track bootstrap status
_bootstrap_state: dict[str, Any] = {
    "status": BootstrapStatus.PENDING,
    "error": None,
}


def get_bootstrap_status() -> dict[str, Any]:
    """Retrieve the current bootstrap status.

    Returns:
        A dictionary containing 'status' and 'error' (if any).
    """
    return _bootstrap_state


# Default models to download if local NLP is enabled
REQUIRED_MODELS: dict[str, dict[str, str | bool]] = {
    "embeddings": {
        "repo_id": settings.embedding_model_id,
        "is_snapshot": True,
    },
    "llm": {
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "is_snapshot": False,
    },
    "reranker": {
        "repo_id": settings.reranker_model_id,
        "is_snapshot": True,
    },
    "ner": {
        "repo_id": settings.ner_model_id,
        "is_snapshot": True,
    },
}


def ensure_models_exist() -> None:
    """Ensure all required models exist in the models directory.

    Downloads missing models (embeddings and local LLM) from HuggingFace
    with progress indicators. This function is synchronous and should be
    offloaded to a thread in async contexts.
    """
    if not settings.use_local_nlp_models:
        logger.info("[Bootstrap] Local NLP models disabled. Skipping download.")
        _bootstrap_state["status"] = BootstrapStatus.SUCCESS
        return

    try:
        models_dir = settings.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        for model_type, info in REQUIRED_MODELS.items():
            is_snapshot = info.get("is_snapshot", True)
            repo_id = str(info.get("repo_id"))

            # Allow settings to override repo_id for LLM
            if model_type == "llm" and settings.local_llm_repo_id:
                repo_id = settings.local_llm_repo_id

            if is_snapshot:
                target_dir = models_dir / repo_id
                if not target_dir.exists() or not any(target_dir.iterdir()):
                    logger.info("[Bootstrap] Downloading %s model: %s", model_type, repo_id)
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(target_dir),
                        ignore_patterns=[
                            "*.msgpack",
                            "flax_model*",
                            "tf_model*",
                            "rust_model*",
                            "*.bin",
                            "*.pth",
                        ],
                        local_dir_use_symlinks=False,
                    )
            else:
                filename = str(info.get("filename"))
                # Allow settings to override filename for LLM
                if model_type == "llm" and settings.local_llm_path:
                    filename = Path(settings.local_llm_path).name

                target_file = models_dir / filename
                if not target_file.exists():
                    logger.info(
                        "[Bootstrap] Downloading %s file: %s from %s", model_type, filename, repo_id
                    )
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(models_dir),
                        local_dir_use_symlinks=False,
                    )

        _bootstrap_state["status"] = BootstrapStatus.SUCCESS
        _bootstrap_state["error"] = None
    except Exception as e:
        logger.error("[Bootstrap] Failed to ensure models: %s", e)
        _bootstrap_state["status"] = BootstrapStatus.FAILED
        _bootstrap_state["error"] = str(e)
        raise
