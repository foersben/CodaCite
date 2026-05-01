"""Unit tests for the GLiNERFallbackExtractor infrastructure adapter.

Validates the local named entity recognition (NER) fallback using the GLiNER
library, ensuring correct entity mapping and robust handling of missing dependencies.
"""

from typing import Any

import pytest

from app.infrastructure.extraction import GLiNERFallbackExtractor


@pytest.fixture
def mock_gliner(mocker: Any) -> Any:
    """Mock the GLiNER library and its main class.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked GLiNER class.
    """
    mock_module = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"gliner": mock_module})
    return mock_module.GLiNER


@pytest.mark.asyncio
async def test_gliner_extractor_init_success(mocker: Any, mock_gliner: Any) -> None:
    """Tests successful initialization of GLiNERFallbackExtractor.

    Given:
        The gliner library is installed and available.
    When:
        A GLiNERFallbackExtractor is initialized.
    Then:
        It should load the pretrained model correctly.

    Args:
        mocker: The pytest-mock fixture.
        mock_gliner: The mocked GLiNER class.
    """
    # Arrange
    mock_gliner.from_pretrained.return_value = mocker.MagicMock()

    # Act
    extractor = GLiNERFallbackExtractor()

    # Assert
    assert extractor.model is not None
    mock_gliner.from_pretrained.assert_called_once()


@pytest.mark.asyncio
async def test_gliner_extractor_init_failure(mocker: Any) -> None:
    """Tests that GLiNERFallbackExtractor handles missing package gracefully.

    Given:
        The gliner library is not installed in the environment.
    When:
        A GLiNERFallbackExtractor is initialized.
    Then:
        It should set the model to None without raising an ImportError.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mocker.patch.dict("sys.modules", {"gliner": None})

    # Act
    extractor = GLiNERFallbackExtractor()

    # Assert
    assert extractor.model is None


@pytest.mark.asyncio
async def test_gliner_extractor_extract_success(mocker: Any, mock_gliner: Any) -> None:
    """Tests entity extraction using the GLiNER model.

    Given:
        A functioning GLiNER model instance.
    When:
        Extract is called with sample text.
    Then:
        It should return correctly mapped Node objects and an empty list of edges.

    Args:
        mocker: The pytest-mock fixture.
        mock_gliner: The mocked GLiNER class.
    """
    # Arrange
    mock_model = mocker.MagicMock()
    mock_model.predict_entities.return_value = [
        {"text": "Apple", "label": "organization"},
        {"text": "Steve Jobs", "label": "person"},
    ]
    mock_gliner.from_pretrained.return_value = mock_model

    extractor = GLiNERFallbackExtractor()

    # Act
    nodes, edges = await extractor.extract("Steve Jobs founded Apple.")

    # Assert
    assert len(nodes) == 2
    assert nodes[0].name == "Apple"
    assert nodes[0].label == "ORGANIZATION"
    assert nodes[1].name == "Steve Jobs"
    assert nodes[1].label == "PERSON"
    assert edges == []


@pytest.mark.asyncio
async def test_gliner_extractor_no_model(mocker: Any) -> None:
    """Tests extract behavior when the model failed to load.

    Given:
        A GLiNERFallbackExtractor that failed to initialize its model.
    When:
        Extract is called.
    Then:
        It should immediately return empty lists.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mocker.patch.dict("sys.modules", {"gliner": None})
    extractor = GLiNERFallbackExtractor()

    # Act
    nodes, edges = await extractor.extract("Some text")

    # Assert
    assert nodes == []
    assert edges == []
