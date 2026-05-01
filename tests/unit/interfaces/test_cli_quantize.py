"""Unit tests for the quantization CLI.

Ensures the quantization command calls the embedder correctly and handles
failures gracefully.
"""

from typing import Any

import pytest

from app.cli.quantize import main


def test_quantize_cli_success(mocker: Any) -> None:
    """Tests successful quantization CLI execution.

    Given:
        A functional embedder and valid settings.
    When:
        The main CLI entry point is called.
    Then:
        The embedder should be initialized with the correct model name.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mock_embedder_class = mocker.patch("app.cli.quantize.HuggingFaceEmbedder")
    mock_settings = mocker.patch("app.cli.quantize.settings")
    mock_settings.embedding_model_id = "test-model"

    # Act
    main()

    # Assert
    assert mock_embedder_class.called
    args, kwargs = mock_embedder_class.call_args
    assert kwargs["model_name"] == "test-model"


def test_quantize_cli_failure(mocker: Any) -> None:
    """Tests quantization CLI failure handling.

    Given:
        An embedder initialization that fails.
    When:
        The main CLI entry point is called.
    Then:
        The process should exit with a non-zero status code.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mocker.patch("app.cli.quantize.HuggingFaceEmbedder", side_effect=RuntimeError("optim failed"))
    mock_settings = mocker.patch("app.cli.quantize.settings")
    mock_settings.embedding_model_id = "test-model"

    # Act & Assert
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
