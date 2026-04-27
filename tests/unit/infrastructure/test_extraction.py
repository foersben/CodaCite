"""Unit tests for knowledge graph extraction infrastructure.

Validates the Gemini and GLiNER extraction implementations, ensuring
entities and relations are correctly parsed from text and handled.
"""

import sys
from typing import Any

import pytest

# Mock external dependencies to avoid ModuleNotFoundError
# These are kept at module level because they affect imports
mock_gliner_module = sys.modules.get("gliner") or pytest.importorskip("unittest.mock").MagicMock()
if "gliner" not in sys.modules:
    sys.modules["gliner"] = mock_gliner_module

mock_langchain_module = (
    sys.modules.get("langchain_google_genai") or pytest.importorskip("unittest.mock").MagicMock()
)
if "langchain_google_genai" not in sys.modules:
    sys.modules["langchain_google_genai"] = mock_langchain_module

from app.domain.models import Edge, Node  # noqa: E402
from app.infrastructure.extraction import (  # noqa: E402
    GeminiEntityExtractor,
    GLiNERFallbackExtractor,
)


@pytest.mark.asyncio
async def test_gemini_extraction_success(mocker: Any) -> None:
    """Tests successful entity and relation extraction using Gemini.

    Given:
        A valid input text and a functional Gemini LLM mock.
    When:
        The extract method is called.
    Then:
        It should return the expected Node and Edge objects.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mock_llm = mocker.MagicMock()
    mock_extractor = mocker.MagicMock()
    mock_extractor.ainvoke = mocker.AsyncMock()

    from app.infrastructure.extraction import ExtractedGraph as ExtGraph

    mock_result = ExtGraph(
        nodes=[
            Node(
                id="n1",
                label="person",
                name="Alice",
                description="Person named Alice",
                source_chunk_ids=["c1"],
            )
        ],
        edges=[
            Edge(
                source_id="n1",
                target_id="n2",
                relation="works_at",
                description="Alice works at ACME",
                source_chunk_ids=["c1"],
                weight=1.0,
            )
        ],
    )

    mock_extractor.ainvoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_extractor

    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm)
    extractor = GeminiEntityExtractor(api_key="test-key")

    # Act
    nodes, edges = await extractor.extract("Alice works at ACME")

    # Assert
    assert nodes == [
        Node(
            id="n1",
            label="person",
            name="Alice",
            description="Person named Alice",
            source_chunk_ids=["c1"],
        )
    ]
    assert len(edges) == 1
    assert edges[0].relation == "works_at"


@pytest.mark.asyncio
async def test_gemini_extraction_failure(mocker: Any) -> None:
    """Tests handling of API failures during Gemini extraction.

    Given:
        An input text and an LLM that raises an exception.
    When:
        The extract method is called.
    Then:
        It should return empty lists for nodes and edges.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mock_llm = mocker.MagicMock()
    mock_extractor = mocker.MagicMock()
    mock_extractor.ainvoke.side_effect = Exception("API Error")
    mock_llm.with_structured_output.return_value = mock_extractor

    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm)
    extractor = GeminiEntityExtractor(api_key="test-key")

    # Act
    nodes, edges = await extractor.extract("Some text")

    # Assert
    assert nodes == []
    assert edges == []


@pytest.mark.asyncio
async def test_gemini_init_failure(mocker: Any) -> None:
    """Tests handling of initialization failures for the Gemini extractor.

    Given:
        An API key and a failing LLM constructor.
    When:
        The extractor is initialized and used.
    Then:
        The extractor should gracefully return empty results.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mocker.patch(
        "langchain_google_genai.ChatGoogleGenerativeAI", side_effect=Exception("Init failed")
    )
    extractor = GeminiEntityExtractor(api_key="test-key")

    # Assert
    assert extractor.llm is None
    assert extractor.extractor is None

    # Act
    nodes, edges = await extractor.extract("text")

    # Assert
    assert nodes == []


@pytest.mark.asyncio
async def test_gliner_extraction_success(mocker: Any) -> None:
    """Tests successful entity extraction using GLiNER fallback.

    Given:
        A valid input text and a functional GLiNER model mock.
    When:
        The extract method is called.
    Then:
        It should return the detected entities as Nodes and an empty Edge list.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mock_model = mocker.MagicMock()
    # GLiNER returns a list of dicts with 'text' and 'label'
    mock_model.predict_entities.return_value = [
        {"text": "Alice", "label": "person"},
        {"text": "ACME", "label": "organization"},
    ]

    mocker.patch("gliner.GLiNER.from_pretrained", return_value=mock_model)
    extractor = GLiNERFallbackExtractor()

    # Act
    nodes, edges = await extractor.extract("Alice works at ACME")

    # Assert
    assert len(nodes) == 2
    names = [n.name for n in nodes]
    assert "Alice" in names
    assert "ACME" in names
    assert edges == []  # GLiNER only does nodes


@pytest.mark.asyncio
async def test_gliner_init_failure(mocker: Any) -> None:
    """Tests handling of model loading failures for the GLiNER extractor.

    Given:
        A failing model loading function.
    When:
        The extractor is initialized and used.
    Then:
        It should return empty lists on failure.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mocker.patch("gliner.GLiNER.from_pretrained", side_effect=Exception("Load failed"))
    extractor = GLiNERFallbackExtractor()

    # Act
    nodes, edges = await extractor.extract("text")

    # Assert
    assert nodes == []
    assert edges == []
