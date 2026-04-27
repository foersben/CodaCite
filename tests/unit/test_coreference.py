"""Unit tests for the FastCorefResolver.

Validates coreference resolution logic, including mention replacement
and fallback behavior.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_fcoref():
    """Docstring generated to satisfy ruff D103."""
    mock = MagicMock()
    # Mock return value of predict
    mock_result = MagicMock()
    mock_result.get_clusters.return_value = []
    mock.predict.return_value = [mock_result]
    return mock


@pytest.fixture
def resolver(mock_fcoref):
    """Docstring generated to satisfy ruff D103."""
    with (
        patch("spacy.blank"),
        patch("fastcoref.FCoref", return_value=mock_fcoref),
        patch("fastcoref.coref_models.modeling_fcoref.FCorefModel"),
    ):
        from app.infrastructure.coreference import FastCorefResolver

        return FastCorefResolver()


@pytest.mark.asyncio
async def test_resolve_empty_text(resolver):
    """Docstring generated to satisfy ruff D103."""
    assert await resolver.resolve("") == ""
    assert await resolver.resolve("   ") == "   "


@pytest.mark.asyncio
async def test_resolve_no_clusters(resolver, mock_fcoref):
    """Docstring generated to satisfy ruff D103."""
    text = "Alice went to the store. She bought some milk."
    mock_result = MagicMock()
    mock_result.get_clusters.return_value = []
    mock_fcoref.predict.return_value = [mock_result]

    result = await resolver.resolve(text)
    assert result == text


@pytest.mark.asyncio
async def test_resolve_with_clusters(resolver, mock_fcoref):
    """Docstring generated to satisfy ruff D103."""
    text = "Alice went to the store. She bought some milk."
    # Cluster: Alice (0, 5), She (25, 28)
    mock_result = MagicMock()
    mock_result.get_clusters.return_value = [[(0, 5), (25, 28)]]
    mock_fcoref.predict.return_value = [mock_result]

    result = await resolver.resolve(text)
    # "Alice went to the store. Alice bought some milk."
    assert "Alice bought some milk" in result
    assert result == "Alice went to the store. Alice bought some milk."


@pytest.mark.asyncio
async def test_resolve_multiple_clusters(resolver, mock_fcoref):
    """Docstring generated to satisfy ruff D103."""
    text = "Alice saw Bob. She waved at him."
    # Clusters: [Alice (0,5), She (15,18)], [Bob (10,13), him (28,31)]
    mock_result = MagicMock()
    mock_result.get_clusters.return_value = [[(0, 5), (15, 18)], [(10, 13), (28, 31)]]
    mock_fcoref.predict.return_value = [mock_result]

    result = await resolver.resolve(text)
    assert result == "Alice saw Bob. Alice waved at Bob."


@pytest.mark.asyncio
async def test_resolve_fallback_on_error(resolver, mock_fcoref):
    """Docstring generated to satisfy ruff D103."""
    text = "Error prone text"
    mock_fcoref.predict.side_effect = Exception("Model failed")

    result = await resolver.resolve(text)
    assert result == text
