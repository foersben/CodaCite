"""Unit tests for the FastCorefResolver.

This module validates coreference resolution logic, including mention replacement
and fallback behavior within the Infrastructure layer.
"""

import sys
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def mock_dependencies(mocker: Any) -> None:
    """Auto-use fixture to mock heavy dependencies.

    Args:
        mocker: The pytest-mock fixture.
    """
    mocker.patch.dict(
        sys.modules,
        {
            "spacy": mocker.MagicMock(),
            "fastcoref": mocker.MagicMock(),
            "fastcoref.coref_models": mocker.MagicMock(),
            "fastcoref.coref_models.modeling_fcoref": mocker.MagicMock(),
        },
    )


@pytest.fixture
def mock_fcoref(mocker: Any) -> Any:
    """Fixture providing a mocked FCoref model.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked FCoref model instance.
    """
    mock = mocker.MagicMock()
    # Mock return value of predict
    mock_result = mocker.MagicMock()
    mock_result.get_clusters.return_value = []
    mock.predict.return_value = [mock_result]
    return mock


@pytest.fixture
def resolver(mocker: Any, mock_fcoref: Any) -> Any:
    """Fixture providing a FastCorefResolver with mocked dependencies.

    Args:
        mocker: The pytest-mock fixture.
        mock_fcoref: The mocked FCoref model instance.

    Returns:
        A FastCorefResolver instance.
    """
    mocker.patch("spacy.blank")
    mocker.patch("fastcoref.FCoref", return_value=mock_fcoref)
    mocker.patch("fastcoref.coref_models.modeling_fcoref.FCorefModel")

    from app.infrastructure.coreference import FastCorefResolver

    return FastCorefResolver()


@pytest.mark.asyncio
async def test_resolve_empty_text(resolver: Any) -> None:
    """Tests that resolving empty or whitespace-only text returns the original text.

    Given:
        Empty or whitespace string.
    When:
        The resolve method is called.
    Then:
        The original string should be returned.
    """
    assert await resolver.resolve("") == ""
    assert await resolver.resolve("   ") == "   "


@pytest.mark.asyncio
async def test_resolve_no_clusters(resolver: Any, mock_fcoref: Any, mocker: Any) -> None:
    """Tests that text is returned as-is if no coreference clusters are found.

    Given:
        Text with no detectable coreferences.
    When:
        The resolve method is called.
    Then:
        The original text should be returned.
    """
    text = "Alice went to the store. She bought some milk."
    mock_result = mocker.MagicMock()
    mock_result.get_clusters.return_value = []
    mock_fcoref.predict.return_value = [mock_result]

    result = await resolver.resolve(text)
    assert result == text


@pytest.mark.asyncio
async def test_resolve_with_clusters(resolver: Any, mock_fcoref: Any, mocker: Any) -> None:
    """Tests that coreferences are correctly resolved using found clusters.

    Given:
        Text where "She" refers to "Alice".
    When:
        The resolve method is called.
    Then:
        The pronoun "She" should be replaced by "Alice".
    """
    text = "Alice went to the store. She bought some milk."
    # Cluster: Alice (0, 5), She (25, 28)
    mock_result = mocker.MagicMock()
    mock_result.get_clusters.return_value = [["Alice", "She"]]
    mock_fcoref.predict.return_value = [mock_result]

    result = await resolver.resolve(text)
    # "Alice went to the store. Alice bought some milk."
    assert "Alice bought some milk" in result
    assert result == "Alice went to the store. Alice bought some milk."


@pytest.mark.asyncio
async def test_resolve_multiple_clusters(resolver: Any, mock_fcoref: Any, mocker: Any) -> None:
    """Tests that multiple independent coreference clusters are resolved correctly.

    Given:
        Text with multiple entities and pronouns.
    When:
        The resolve method is called.
    Then:
        All entities should be resolved to their primary mentions.
    """
    text = "Alice saw Bob. She waved at him."
    # Clusters: [Alice (0,5), She (15,18)], [Bob (10,13), him (28,31)]
    mock_result = mocker.MagicMock()
    mock_result.get_clusters.return_value = [["Alice", "She"], ["Bob", "him"]]
    mock_fcoref.predict.return_value = [mock_result]

    result = await resolver.resolve(text)
    assert result == "Alice saw Bob. Alice waved at Bob."


@pytest.mark.asyncio
async def test_resolve_repeated_mentions(resolver: Any, mock_fcoref: Any, mocker: Any) -> None:
    """Tests that repeated mentions of the same string are resolved correctly.

    Given:
        Text where "She" appears multiple times and refers to "Alice".
    When:
        The resolve method is called.
    Then:
        All occurrences of "She" in the cluster should be replaced by "Alice".
    """
    text = "Alice went home. She was tired. She slept."
    # Cluster: Alice (0, 5), She (17, 20), She (32, 35)
    mock_result = mocker.MagicMock()
    mock_result.get_clusters.return_value = [["Alice", "She", "She"]]
    mock_fcoref.predict.return_value = [mock_result]

    result = await resolver.resolve(text)
    assert result == "Alice went home. Alice was tired. Alice slept."


@pytest.mark.asyncio
async def test_resolve_fallback_on_error(resolver: Any, mock_fcoref: Any) -> None:
    """Tests that the resolver falls back to the original text if the model fails.

    Given:
        A model that raises an exception.
    When:
        The resolve method is called.
    Then:
        The original text should be returned as a fallback.
    """
    text = "Error prone text"
    mock_fcoref.predict.side_effect = Exception("Model failed")

    result = await resolver.resolve(text)
    assert result == text
