"""Unit tests for the LocalVLM infrastructure adapter."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.infrastructure.vlm import LocalVLM


@pytest.fixture
def mock_llama(mocker: Any) -> MagicMock:
    """Mocks the Llama class from llama_cpp."""
    mock_class = mocker.patch("app.infrastructure.vlm.Llama")
    return mock_class


def test_vlm_init_no_path(mocker: Any) -> None:
    """Tests VLM initialization when no model path is configured."""
    mocker.patch("app.infrastructure.vlm.settings.local_vlm_path", "")
    vlm = LocalVLM()
    assert vlm.llm is None


def test_vlm_init_path_not_exists(mocker: Any) -> None:
    """Tests VLM initialization when model path does not exist."""
    mocker.patch("app.infrastructure.vlm.settings.local_vlm_path", "/non/existent/path")
    vlm = LocalVLM()
    assert vlm.llm is None


def test_vlm_init_success(mocker: Any, tmp_path: Path, mock_llama: MagicMock) -> None:
    """Tests successful VLM initialization."""
    model_file = tmp_path / "model.gguf"
    model_file.touch()

    mocker.patch("app.infrastructure.vlm.settings.local_vlm_path", str(model_file))
    vlm = LocalVLM()

    assert vlm.llm is not None
    mock_llama.assert_called_once()


def test_vlm_init_failure(mocker: Any, tmp_path: Path, mock_llama: MagicMock) -> None:
    """Tests VLM initialization failure handled gracefully."""
    model_file = tmp_path / "model.gguf"
    model_file.touch()

    mocker.patch("app.infrastructure.vlm.settings.local_vlm_path", str(model_file))
    mock_llama.side_effect = Exception("Load failed")

    vlm = LocalVLM()
    assert vlm.llm is None


def test_describe_image_no_model() -> None:
    """Tests describe_image when model is not initialized."""
    vlm = LocalVLM()
    vlm.llm = None

    mock_image = MagicMock(spec=Image.Image)
    result = vlm.describe_image(mock_image)
    assert "Error" in result


def test_describe_image_success(mocker: Any) -> None:
    """Tests successful image description generation."""
    vlm = LocalVLM()
    mock_llm = MagicMock()
    vlm.llm = mock_llm

    # Mock LLM response
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "This is a blueprint of a house."}}]
    }

    # Create a real small image to test base64 conversion
    img = Image.new("RGB", (10, 10), color="red")

    result = vlm.describe_image(img)

    assert result == "This is a blueprint of a house."
    mock_llm.create_chat_completion.assert_called_once()


def test_describe_image_failure() -> None:
    """Tests describe_image failure handled gracefully."""
    vlm = LocalVLM()
    mock_llm = MagicMock()
    vlm.llm = mock_llm

    mock_llm.create_chat_completion.side_effect = Exception("Generation failed")

    img = Image.new("RGB", (10, 10), color="red")
    result = vlm.describe_image(img)

    assert "[VLM Error: Generation failed]" == result
