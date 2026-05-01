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

from app.config import settings

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

        # 1. Ensure Embedding Model exists
        emb_dir = models_dir / settings.embedding_model_id
        if not emb_dir.exists() or not any(emb_dir.iterdir()):
            logger.info(
                "[Bootstrap] Initializing download for Embedding Model: %s",
                settings.embedding_model_id,
            )
            try:
                snapshot_download(
                    repo_id=settings.embedding_model_id,
                    local_dir=str(emb_dir),
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
                )
                logger.info("[Bootstrap] Embedding model ready.")
            except Exception as e:
                logger.error("[Bootstrap] Failed to download embedding model: %s", e)
                raise RuntimeError(f"Could not download embedding model: {e}") from e

        # 2. Ensure LLM GGUF exists
        llm_info = REQUIRED_MODELS["llm"]
        llm_filename = str(
            Path(settings.local_llm_path).name if settings.local_llm_path else llm_info["filename"]
        )
        llm_repo_id = str(settings.local_llm_repo_id or llm_info["repo_id"])
        llm_path = models_dir / llm_filename

        if not llm_path.exists():
            logger.info(
                "[Bootstrap] Initializing download for Generative AI Model: %s (~4.8GB)",
                llm_filename,
            )

            try:
                hf_hub_download(
                    repo_id=llm_repo_id,
                    filename=llm_filename,
                    local_dir=str(models_dir),
                    local_dir_use_symlinks=False,
                )
                logger.info("[Bootstrap] Generative model ready: %s", llm_filename)
            except Exception as e:
                logger.error("[Bootstrap] Failed to download LLM: %s", e)
                if llm_path.exists():
                    llm_path.unlink()
                raise RuntimeError(f"Could not download required LLM: {e}") from e
        else:
            logger.debug("[Bootstrap] Generative model already present: %s", llm_filename)

        _bootstrap_state["status"] = BootstrapStatus.SUCCESS
        _bootstrap_state["error"] = None
    except Exception as e:
        _bootstrap_state["status"] = BootstrapStatus.FAILED
        _bootstrap_state["error"] = str(e)
        raise
