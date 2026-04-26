"""Hardened unit tests for entity extractors.

This module provides coverage for the GLiNER fallback extractor and its
error handling.
"""

from unittest.mock import MagicMock

import pytest

from app.infrastructure.extraction import GLiNERFallbackExtractor


@pytest.fixture
def mock_gliner(mocker):
    """Mock GLiNER class."""
    mock_module = MagicMock()
    mocker.patch.dict("sys.modules", {"gliner": mock_module})
    return mock_module.GLiNER


@pytest.mark.asyncio
async def test_gliner_extractor_init_success(mocker, mock_gliner) -> None:
    """Test GLiNERFallbackExtractor initializes properly.

    Given: The gliner library is installed.
    When: A GLiNERFallbackExtractor is initialized.
    Then: It should load the model from pretrained.
    """
    # Arrange
    mock_gliner.from_pretrained.return_value = MagicMock()

    # Act
    extractor = GLiNERFallbackExtractor()

    # Assert
    assert extractor.model is not None
    mock_gliner.from_pretrained.assert_called_once()


@pytest.mark.asyncio
async def test_gliner_extractor_init_failure(mocker) -> None:
    """Test GLiNERFallbackExtractor handles missing package.

    Given: The gliner library is not installed.
    When: A GLiNERFallbackExtractor is initialized.
    Then: It should set model to None and not raise an error.
    """
    # Arrange
    mocker.patch.dict("sys.modules", {"gliner": None})

    # Act
    extractor = GLiNERFallbackExtractor()

    # Assert
    assert extractor.model is None


@pytest.mark.asyncio
async def test_gliner_extractor_extract_success(mocker, mock_gliner) -> None:
    """Test extraction with GLiNER.

    Given: A working GLiNER model.
    When: Extract is called with text.
    Then: It should return correctly mapped nodes and no edges.
    """
    # Arrange
    mock_model = MagicMock()
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
async def test_gliner_extractor_no_model(mocker) -> None:
    """Test extract when model is not loaded.

    Given: A GLiNERFallbackExtractor with no model.
    When: Extract is called.
    Then: It should return empty lists.
    """
    # Arrange
    mocker.patch.dict("sys.modules", {"gliner": None})
    extractor = GLiNERFallbackExtractor()

    # Act
    nodes, edges = await extractor.extract("Some text")

    # Assert
    assert nodes == []
    assert edges == []
