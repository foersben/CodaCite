"""Application configuration using pydantic-settings.

This module defines the global settings for the application, loaded from
environment variables or a .env file.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.infrastructure.credentials import resolve_secret

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
    """Application settings loaded from environment variables.

    Attributes:
        surrealdb_url: URL for the SurrealDB instance.
        surrealdb_user: Username for SurrealDB authentication.
        surrealdb_pass: Password for SurrealDB authentication.
        surrealdb_ns: SurrealDB namespace.
        surrealdb_db: SurrealDB database name.
        models_dir: Base directory for local model artifacts.
        embedding_model_id: HuggingFace model ID for embeddings.
        device: Hardware device to use (cpu, cuda, mps).
        use_local_nlp_models: Whether to prefer local models over cloud APIs.
        chunk_size: Maximum character length for document chunks.
        chunk_overlap: Overlap between consecutive chunks.
        gemini_api_key: API key for Google Gemini services.
        gemini_model: Target Gemini model identifier.
        openai_api_key: API key for OpenAI services.
        openai_model: Target OpenAI model identifier.
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
    app_dir: Path = Path.home() / ".codacite"
    models_dir: Path = Path.home() / ".codacite" / "models"
    upload_dir: Path = Path.home() / ".codacite" / "uploads"
    embedding_model_id: str = "BAAI/bge-large-en-v1.5"

    # Device Mapping (CPU/CUDA/MPS)
    device: str = "cpu"

    # NLP Toggles
    use_local_nlp_models: bool = True
    quantization_enabled: bool = True
    quantization_backend: str = "openvino"  # openvino, torch
    ov_precision: str = "int8"  # int8, fp16, fp32

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # LLM (Google GenAI)
    local_llm_repo_id: str = ""
    local_llm_path: str = ""
    local_vlm_repo_id: str = ""
    local_vlm_path: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    @model_validator(mode="after")
    def _retrieve_gemini_key(self) -> Settings:
        """Attempt to retrieve Gemini API key from Secret Service if not provided."""
        if not self.gemini_api_key:
            # Retrieve from Secret Service (KeePassXC)
            # Entry title: Gemini_API
            key = resolve_secret("Gemini_API")
            if key:
                self.gemini_api_key = key

        # Ensure directories exist (wrapped in try/except for read-only environments like CI)
        try:
            self.app_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.upload_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create application directories (likely read-only FS): %s", e)

        return self

    @property
    def embedding_model_path(self) -> Path:
        """Resolved local path for the embedding model.

        Returns:
            A Path object pointing to the specific model directory.
        """
        return self.models_dir / self.embedding_model_id


settings = Settings()
