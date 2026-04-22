"""
Backwards-compatible entry point for model downloading.

Use the package entry point instead:
    uv run download-models

"""

from __future__ import annotations

from app.cli.download_models import main


def download_models() -> None:
    """Backwards-compatible wrapper around the package CLI implementation."""
    main()


if __name__ == "__main__":
    main()
