"""Application configuration management using Pydantic Settings.

This module serves as the central configuration hub for CodaCite. It handles
the loading, validation, and resolution of all environment variables,
secrets (via KeePassXC), and directory paths required for both local and
cloud-based operations.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.credentials import resolve_secret

logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller.

    Args:
        relative_path: Relative path to the resource (e.g. 'app/static').

    Returns:
        Absolute path to the resource.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)  # type: ignore
    except Exception:
        base_path = Path(os.path.abspath("."))

    return base_path / relative_path


class Settings(BaseSettings):
    """Global application settings and environment resolution.

    This class defines the configuration schema for the entire system. It
    utilizes Pydantic's validation to ensure that all required services
    (SurrealDB, LLM providers, etc.) have valid connection strings and
    credentials before the application starts.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SurrealDB
    surrealdb_url: str = "ws://localhost:8000"
    surrealdb_user: str = "root"
    surrealdb_pass: str = "root"
    surrealdb_ns: str = "codacite"
    surrealdb_db: str = "production"

    # Files and Storage
    app_dir: Path = Path("data")
    models_dir: Path = Path("data/models")
    upload_dir: Path = Path("data/blobs")
    logs_dir: Path = Path("data/logs")
    db_dir: Path = Path("data/db")
    embedding_model_id: str = "BAAI/bge-m3"
    reranker_model_id: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    ner_model_id: str = "knowledgator/gliner-bi-base-v2.0"

    # Device Mapping (CPU/CUDA/MPS)
    device: str = "cpu"

    # NLP Toggles
    use_local_nlp_models: bool = True
    fail_fast_on_bootstrap: bool = False
    quantization_enabled: bool = True
    quantization_backend: str = "openvino"  # openvino, torch
    ov_precision: str = "int8"  # int8, fp16, fp32

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # LLM (Google GenAI)
    local_llm_repo_id: str = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
    local_llm_path: str = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    local_vlm_repo_id: str = ""
    local_vlm_path: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    @model_validator(mode="after")
    def _initialize_environment(self) -> Settings:
        """Initialize environment variables and ensure directories exist."""
        # Force HuggingFace to use our local cache directory
        hf_cache = self.models_dir / "hf_cache"
        os.environ["HF_HOME"] = str(hf_cache)

        if not self.gemini_api_key:
            # Retrieve from Secret Service (KeePassXC)
            key = resolve_secret("Gemini_API")
            if key:
                self.gemini_api_key = key

        # Ensure directories exist
        for d in [self.app_dir, self.models_dir, self.upload_dir, self.logs_dir, self.db_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return self

    @property
    def embedding_model_path(self) -> str:
        """Resolved model identifier for HuggingFace.

        Returns:
            The model ID string.
        """
        return self.embedding_model_id


settings = Settings()
