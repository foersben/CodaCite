"""Unit tests for the HuggingFaceEmbedder infrastructure adapter.

Validates model initialization, device handling, quantization fallback logic,
and embedding generation (both single and batch processing).
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from app.infrastructure.embeddings import HuggingFaceEmbedder


@pytest.fixture
def mock_tokenizer(mocker: Any) -> Any:
    """Provides a mocked tokenizer fixture.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked tokenizer object.
    """
    mock = mocker.MagicMock()
    # Mock return value needs a .to() method to simulate tensor movement
    mock_input = mocker.MagicMock()
    mock_input.to.return_value = mock_input
    mock.return_value = mock_input
    return mock


@pytest.fixture
def mock_model(mocker: Any) -> Any:
    """Provides a mocked model fixture.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked model object returning simulated tensor outputs.
    """
    mock = mocker.MagicMock()
    # Mock model output: (batch_size, sequence_length, hidden_size)
    mock_output = [torch.randn(1, 1, 1024)]
    mock.return_value = mock_output
    return mock


@pytest.fixture
def embedder(mocker: Any, mock_tokenizer: Any, mock_model: Any) -> HuggingFaceEmbedder:
    """Provides a HuggingFaceEmbedder instance with a mocked backend.

    Args:
        mocker: The pytest-mock fixture.
        mock_tokenizer: The mocked tokenizer fixture.
        mock_model: The mocked model fixture.

    Returns:
        An initialized HuggingFaceEmbedder using mocked AutoTokenizer and AutoModel.
    """
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch(
        "app.infrastructure.embeddings.AutoModel.from_pretrained",
        return_value=mock_model,
    )

    # Force standard model initialization via settings mocks
    # Patch app.config.settings because it is imported inside __init__
    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = False
    mock_settings.device = "cpu"

    return HuggingFaceEmbedder()


@pytest.mark.asyncio
async def test_embed_single_text(
    embedder: HuggingFaceEmbedder, mock_tokenizer: Any, mock_model: Any
) -> None:
    """Tests embedding a single string.

    Given:
        A valid input string.
    When:
        The embedder's embed method is called.
    Then:
        It should return a list of floats (embedding) with the expected dimension.

    Args:
        embedder: The HuggingFaceEmbedder fixture.
        mock_tokenizer: The mocked tokenizer fixture.
        mock_model: The mocked model fixture.
    """
    text = "Hello world"
    embedding = await embedder.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 1024
    assert mock_tokenizer.called
    assert mock_model.called


@pytest.mark.asyncio
async def test_embed_batch(
    embedder: HuggingFaceEmbedder, mock_tokenizer: Any, mock_model: Any
) -> None:
    """Tests embedding a batch of strings.

    Given:
        A list of strings.
    When:
        The embedder's embed_batch method is called.
    Then:
        It should return a list of embeddings with the correct batch size and dimension.

    Args:
        embedder: The HuggingFaceEmbedder fixture.
        mock_tokenizer: The mocked tokenizer fixture.
        mock_model: The mocked model fixture.
    """
    texts = ["Hello", "World", "Test"]
    # Adjust mock model for batch size 3
    mock_output = [torch.randn(3, 1, 1024)]
    mock_model.return_value = mock_output

    embeddings = await embedder.embed_batch(texts)

    assert len(embeddings) == 3
    assert all(len(e) == 1024 for e in embeddings)


@pytest.mark.asyncio
async def test_embed_empty_list(embedder: HuggingFaceEmbedder) -> None:
    """Tests embedding an empty list.

    Given:
        An empty list of strings.
    When:
        The embedder's embed_batch method is called.
    Then:
        It should return an empty list.

    Args:
        embedder: The HuggingFaceEmbedder fixture.
    """
    assert await embedder.embed_batch([]) == []


@pytest.mark.asyncio
async def test_torch_quantization_init(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests initialization with PyTorch dynamic quantization.

    Given:
        Quantization enabled in settings.
    When:
        HuggingFaceEmbedder is initialized.
    Then:
        It should call torch.quantization.quantize_dynamic.

    Args:
        mocker: The pytest-mock fixture.
        mock_tokenizer: The mocked tokenizer fixture.
    """
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch("app.infrastructure.embeddings.AutoModel.from_pretrained")
    mock_quant = mocker.patch("torch.quantization.quantize_dynamic")

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = True
    mock_settings.quantization_backend = "torch"
    mock_settings.device = "cpu"
    mock_settings.models_dir = Path("./models")

    _ = HuggingFaceEmbedder()
    assert mock_quant.called


@pytest.mark.asyncio
async def test_init_standard_fallback_on_device_error(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests fallback to CPU when target device (e.g., CUDA) fails.

    Given:
        A target device that raises an exception during model movement.
    When:
        HuggingFaceEmbedder is initialized.
    Then:
        It should fall back to CPU and succeed.

    Args:
        mocker: The pytest-mock fixture.
        mock_tokenizer: The mocked tokenizer fixture.
    """
    # Mock model's .to() to fail once then succeed
    mock_model_obj = mocker.MagicMock()
    mock_model_obj.to.side_effect = [Exception("CUDA error"), mock_model_obj]

    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch(
        "app.infrastructure.embeddings.AutoModel.from_pretrained", return_value=mock_model_obj
    )

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = False
    mock_settings.device = "cuda"

    embedder_inst = HuggingFaceEmbedder()
    # It should have fallen back to cpu
    assert embedder_inst.device == "cpu"


@pytest.mark.asyncio
async def test_quantization_caching(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests that a quantized model is loaded from disk if available.

    Given:
        A quantized model file exists on disk.
    When:
        HuggingFaceEmbedder is initialized with quantization enabled.
    Then:
        It should load the model from the cached file instead of re-quantizing.

    Args:
        mocker: The pytest-mock fixture.
        mock_tokenizer: The mocked tokenizer fixture.
    """
    mock_model_obj = mocker.MagicMock()
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch(
        "app.infrastructure.embeddings.AutoModel.from_pretrained", return_value=mock_model_obj
    )
    mock_load = mocker.patch(
        "app.infrastructure.embeddings.torch.load", return_value=mock_model_obj
    )

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = True
    mock_settings.device = "cpu"
    mock_settings.quantization_backend = "torch"
    mock_settings.models_dir = Path("/tmp/models")

    # Mock the cache_file.exists() call via patching the class method
    mocker.patch("pathlib.Path.exists", return_value=True)

    embedder_inst = HuggingFaceEmbedder(model_name="test-model")
    assert embedder_inst.model == mock_model_obj
    assert mock_load.called


@pytest.mark.asyncio
async def test_openvino_init_loading(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests loading an existing OpenVINO IR model."""
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mock_ov_model = mocker.patch("optimum.intel.openvino.OVModelForFeatureExtraction")

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = True
    mock_settings.quantization_backend = "openvino"
    mock_settings.device = "cpu"
    mock_settings.models_dir = Path("/tmp/models")

    # Mock existence of openvino_model.xml
    mocker.patch("pathlib.Path.exists", return_value=True)

    _ = HuggingFaceEmbedder(model_name="test-model")
    mock_ov_model.from_pretrained.assert_called_once()
    assert "export=False" in str(mock_ov_model.from_pretrained.call_args)


@pytest.mark.asyncio
async def test_openvino_init_export(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests exporting a model to OpenVINO IR."""
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mock_ov_model = mocker.patch("optimum.intel.openvino.OVModelForFeatureExtraction")

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = True
    mock_settings.quantization_backend = "openvino"
    mock_settings.device = "cpu"
    mock_settings.models_dir = Path("/tmp/models")
    mock_settings.ov_precision = "int8"

    # Mock non-existence of IR model
    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")

    _ = HuggingFaceEmbedder(model_name="test-model")
    mock_ov_model.from_pretrained.assert_called_once()
    assert "export=True" in str(mock_ov_model.from_pretrained.call_args)


@pytest.mark.asyncio
async def test_openvino_init_failure(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests fallback to standard model when OpenVINO initialization fails."""
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mock_ov_model = mocker.patch("optimum.intel.openvino.OVModelForFeatureExtraction")
    mock_ov_model.from_pretrained.side_effect = Exception("OV Error")

    mock_standard_init = mocker.patch(
        "app.infrastructure.embeddings.HuggingFaceEmbedder._init_standard_model"
    )

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = True
    mock_settings.quantization_backend = "openvino"
    mock_settings.device = "cpu"
    mock_settings.models_dir = Path("/tmp/models")

    mocker.patch("pathlib.Path.exists", return_value=True)

    _ = HuggingFaceEmbedder(model_name="test-model")
    assert mock_standard_init.called


@pytest.mark.asyncio
async def test_torch_quantization_save_failure(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests handling of failures during quantized model caching."""
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch("app.infrastructure.embeddings.AutoModel.from_pretrained")
    mocker.patch("torch.quantization.quantize_dynamic")

    # Mock torch.save to fail
    mocker.patch("torch.save", side_effect=Exception("Save fail"))

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = True
    mock_settings.quantization_backend = "torch"
    mock_settings.device = "cpu"
    mock_settings.models_dir = Path("/tmp/models")

    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")

    # Should not crash
    _ = HuggingFaceEmbedder(model_name="test-model")


@pytest.mark.asyncio
async def test_torch_quantization_save_success(mocker: Any, mock_tokenizer: Any) -> None:
    """Tests successful caching of a quantized PyTorch model."""
    mocker.patch(
        "app.infrastructure.embeddings.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch("app.infrastructure.embeddings.AutoModel.from_pretrained")
    mocker.patch("torch.quantization.quantize_dynamic")
    mock_save = mocker.patch("torch.save")

    mock_settings = mocker.patch("app.config.settings")
    mock_settings.quantization_enabled = True
    mock_settings.quantization_backend = "torch"
    mock_settings.device = "cpu"
    mock_settings.models_dir = Path("/tmp/models")

    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")

    _ = HuggingFaceEmbedder(model_name="test-model")
    assert mock_save.called
