"""Application bootstrap and initialization logic.

This module handles the initial setup of the CodaCite system, including
downloading required NLP models from HuggingFace and tracking the
overall readiness of the infrastructure.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from app.core.config import settings

logger = logging.getLogger(__name__)


class BootstrapStatus(StrEnum):
    """Status of the application bootstrap process."""

    PENDING = "pending"
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"


class _BootstrapState(TypedDict):
    """Internal structure tracking bootstrap progress."""

    status: BootstrapStatus
    error: str | None


# Global state to track bootstrap status
_bootstrap_state: _BootstrapState = {
    "status": BootstrapStatus.PENDING,
    "error": None,
}


def get_bootstrap_status() -> _BootstrapState:
    """Retrieve the current bootstrap status.

    Returns:
        A dictionary containing 'status' and 'error' (if any).
    """
    return _bootstrap_state


# Default models to download if local NLP is enabled
REQUIRED_MODELS: dict[str, dict[str, str | bool]] = {
    """Registry of models required for full system functionality."""
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
    "nli": {
        "repo_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c",
        "is_snapshot": True,
    },
}

# Aggressive ignore list to prevent downloading 2GB+ of unneeded weights (onnx, pth, flax, etc.)
# We strictly prefer safetensors for safety and performance.
COMMON_IGNORE_PATTERNS = [
    """File patterns to ignore when downloading models to save bandwidth and disk space."""
    "*.bin",
    "*.pth",
    "*.pt",
    "*.onnx",
    "onnx/*",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "flax_model*",
    "tf_model*",
    "rust_model*",
    "*.mlmodel",
    "*.coreml",
    "openvino/*",
]


def is_model_cached(repo_id: str) -> bool:
    """Check if a model exists in the local HF cache.

    Args:
        repo_id: The HuggingFace repository identifier.

    Returns:
        True if the model is cached, False otherwise.
    """
    try:
        from huggingface_hub import scan_cache_dir

        hf_cache = settings.models_dir / "hf_cache" / "hub"
        if not hf_cache.exists():
            return False

        cache_info = scan_cache_dir(hf_cache)
        repo_cache = next((r for r in cache_info.repos if r.repo_id == repo_id), None)
        return bool(repo_cache and any(repo_cache.revisions))
    except Exception as e:
        logger.debug("[Bootstrap] Cache scan failed for %s: %s", repo_id, e)
        return False


def ensure_models_exist() -> None:
    """Ensure all required models exist in the models directory.

    Verifies that all required models (embeddings, LLM, etc.) are present
    on disk. If any model is missing, it raises a RuntimeError with
    instructions to run the CLI downloader.
    """
    if not settings.use_local_nlp_models:
        logger.info("[Bootstrap] Local NLP models disabled. Skipping verification.")
        _bootstrap_state["status"] = BootstrapStatus.SUCCESS
        return

    try:
        models_dir = settings.models_dir

        for model_type, info in REQUIRED_MODELS.items():
            is_snapshot = info.get("is_snapshot", True)
            repo_id = str(info.get("repo_id"))

            # Allow settings to override repo_id for LLM
            if model_type == "llm" and settings.local_llm_repo_id:
                repo_id = settings.local_llm_repo_id

            if is_snapshot:
                if not is_model_cached(repo_id):
                    logger.error(
                        "[Bootstrap] Missing snapshot model: %s. Entering DEGRADED mode.", repo_id
                    )
                    _bootstrap_state["status"] = BootstrapStatus.DEGRADED
                    _bootstrap_state["error"] = (
                        f"Missing model: {repo_id}. Run 'uv run download-models' to resolve."
                    )
                    return
                logger.info("[Bootstrap] Verified snapshot model: %s", repo_id)
            else:
                filename = str(info.get("filename"))
                # Allow settings to override filename for LLM
                if model_type == "llm" and settings.local_llm_path:
                    filename = Path(settings.local_llm_path).name

                target_file = models_dir / filename
                if not target_file.exists():
                    logger.error(
                        "[Bootstrap] Missing model file: %s. Entering DEGRADED mode.", filename
                    )
                    _bootstrap_state["status"] = BootstrapStatus.DEGRADED
                    _bootstrap_state["error"] = (
                        f"Missing model: {filename}. Run 'uv run download-models' to resolve."
                    )
                    return
                logger.info("[Bootstrap] Verified model file: %s", filename)

        _bootstrap_state["status"] = BootstrapStatus.SUCCESS
        _bootstrap_state["error"] = None
    except Exception as e:
        logger.error("[Bootstrap] Unexpected error during model verification: %s", e)
        _bootstrap_state["status"] = BootstrapStatus.FAILED
        _bootstrap_state["error"] = str(e)
        # We don't raise here either to keep the server alive,
        # but FAILED implies a more severe infra issue (like permissions)
