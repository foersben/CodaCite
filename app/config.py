"""Application configuration using pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SurrealDB
    surrealdb_url: str = "ws://localhost:8000/rpc"
    surrealdb_user: str = "root"
    surrealdb_pass: str = "root"
    surrealdb_ns: str = "omni"
    surrealdb_db: str = "copilot"

    # Models
    models_dir: Path = Path("./models")
    embedding_model_id: str = "BAAI/bge-large-en-v1.5"

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # LLM (OpenAI-compatible; leave blank to use mock in tests)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    @property
    def embedding_model_path(self) -> Path:
        """Resolved local path for the embedding model."""
        return self.models_dir / self.embedding_model_id


settings = Settings()
