"""Unit tests for quantize CLI.

Ensures the quantization command calls the embedder correctly.
"""

from unittest.mock import patch

import pytest

from app.cli.quantize import main


def test_quantize_cli_success():
    """Test successful quantization CLI execution."""
    with (
        patch("app.cli.quantize.HuggingFaceEmbedder") as mock_embedder_class,
        patch("app.cli.quantize.settings") as mock_settings,
    ):
        mock_settings.embedding_model_id = "test-model"

        # Should not raise
        main()

        assert mock_embedder_class.called
        args, kwargs = mock_embedder_class.call_args
        assert kwargs["model_name"] == "test-model"


def test_quantize_cli_failure():
    """Test quantization CLI failure handling."""
    with (
        patch("app.cli.quantize.HuggingFaceEmbedder", side_effect=RuntimeError("optim failed")),
        patch("app.cli.quantize.settings") as mock_settings,
    ):
        mock_settings.embedding_model_id = "test-model"

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
