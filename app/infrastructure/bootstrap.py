"""Bootstrap logic for downloading required AI models on first run.

This module ensures that the necessary model weights (Embeddings and LLM) are
available in the user's local directory before the application starts.
"""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from app.config import settings

logger = logging.getLogger(__name__)

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


async def ensure_models_exist() -> None:
    """Ensure all required models exist in the models directory.

    Downloads missing models with progress indicators.
    """
    if not settings.use_local_nlp_models:
        logger.info("[Bootstrap] Local NLP models disabled. Skipping download.")
        return

    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure Embedding Model exists
    emb_dir = models_dir / settings.embedding_model_id
    if not emb_dir.exists() or not any(emb_dir.iterdir()):
        print(f"\n[First Run] Downloading Embedding Model: {settings.embedding_model_id}")
        snapshot_download(
            repo_id=settings.embedding_model_id,
            local_dir=str(emb_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print("✅ Embedding model ready.")

    # 2. Ensure LLM GGUF exists
    # We use the filename from settings if available, otherwise default
    llm_info = REQUIRED_MODELS["llm"]
    llm_filename = str(
        Path(settings.local_llm_path).name if settings.local_llm_path else llm_info["filename"]
    )
    llm_repo_id = str(settings.local_llm_repo_id or llm_info["repo_id"])
    llm_path = models_dir / llm_filename

    if not llm_path.exists():
        print(f"\n[First Run] Downloading Generative AI Model: {llm_filename}")
        print("This is a large file (~4.8GB) and requires a stable connection.")

        # Using hf_hub_download as it handles progress bars and resume natively
        try:
            hf_hub_download(
                repo_id=llm_repo_id,
                filename=llm_filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            )
            print(f"✅ Generative model ready: {llm_filename}")
        except Exception as e:
            logger.error("[Bootstrap] Failed to download LLM: %s", e)
            if llm_path.exists():
                llm_path.unlink()
            raise RuntimeError(f"Could not download required LLM: {e}") from e
