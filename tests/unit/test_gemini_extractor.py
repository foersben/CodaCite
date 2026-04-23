"""Tests for GeminiEntityExtractor."""

import pytest

from app.domain.models import Edge, Node
from app.infrastructure.extraction import ExtractedGraph, GeminiEntityExtractor


@pytest.fixture
def mock_chat_google_genai(mocker):
    """Mock ChatGoogleGenerativeAI."""
    return mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI")


def test_gemini_extractor_init_success(mocker, mock_chat_google_genai) -> None:
    """Test GeminiEntityExtractor initializes properly when import succeeds.

    Arrange: Mock ChatGoogleGenerativeAI from langchain.
    Act: Initialize GeminiEntityExtractor.
    Assert: the model and structured output extractor are set up.
    """
    # Arrange
    mock_llm_instance = mocker.MagicMock()
    mock_with_structured_output = mocker.MagicMock()
    mock_llm_instance.with_structured_output = mock_with_structured_output
    mock_chat_google_genai.return_value = mock_llm_instance

    # Act
    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Assert
    assert extractor.llm is not None
    assert extractor.extractor is not None
    mock_with_structured_output.assert_called_once_with(ExtractedGraph)


def test_gemini_extractor_init_failure(mocker) -> None:
    """Test GeminiEntityExtractor handles import failures gracefully.

    Arrange: Patch ChatGoogleGenerativeAI to raise an Exception inside init.
    Act: Initialize GeminiEntityExtractor.
    Assert: llm and extractor attributes are None.
    """
    # Arrange & Act
    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=Exception("Failed"))
    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Assert
    assert extractor.llm is None
    assert extractor.extractor is None


# A simple class to simulate missing properties cleanly compared to MagicMock
class PartialResult:
    """Partial mock result."""

    def __init__(self, nodes):
        """Init partial result."""
        self.nodes = nodes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ainvoke_result, expected_nodes, expected_edges",
    [
        # Happy path (perfect JSON / ExtractedGraph object)
        (
            ExtractedGraph(
                nodes=[Node(id="n1", label="PERSON", name="Alice")],
                edges=[Edge(source_id="n1", target_id="n2", relation="KNOWS")],
            ),
            [Node(id="n1", label="PERSON", name="Alice")],
            [Edge(source_id="n1", target_id="n2", relation="KNOWS")],
        ),
        # Empty response / No nodes or edges
        (
            ExtractedGraph(nodes=[], edges=[]),
            [],
            [],
        ),
        # Partial malformed result (missing edges attribute)
        (
            PartialResult(nodes=[Node(id="n1", label="PERSON", name="Bob")]),
            [Node(id="n1", label="PERSON", name="Bob")],
            [],
        ),
    ],
)
async def test_gemini_extractor_extract_success(
    mocker, mock_chat_google_genai, ainvoke_result, expected_nodes, expected_edges
) -> None:
    """Test extract behaves correctly based on various mock structured outputs.

    Arrange: Set up GeminiEntityExtractor with a mocked ainvoke method.
    Act: Call extract with sample text.
    Assert: The returned nodes and edges match the expected outcomes.
    """
    # Arrange
    mock_llm_instance = mocker.MagicMock()
    mock_extractor_instance = mocker.AsyncMock()

    mock_extractor_instance.ainvoke.return_value = ainvoke_result

    mock_llm_instance.with_structured_output.return_value = mock_extractor_instance
    mock_chat_google_genai.return_value = mock_llm_instance

    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Act
    nodes, edges = await extractor.extract("Sample text")

    # Assert
    assert nodes == expected_nodes
    assert edges == expected_edges


@pytest.mark.asyncio
async def test_gemini_extractor_extract_timeout_or_error(mocker, mock_chat_google_genai) -> None:
    """Test extract handles exceptions (e.g. timeout) gracefully.

    Arrange: Set up GeminiEntityExtractor where ainvoke raises an Exception.
    Act: Call extract.
    Assert: It returns empty lists without crashing.
    """
    # Arrange
    mock_llm_instance = mocker.MagicMock()
    mock_extractor_instance = mocker.AsyncMock()
    mock_extractor_instance.ainvoke.side_effect = Exception("API Timeout")

    mock_llm_instance.with_structured_output.return_value = mock_extractor_instance
    mock_chat_google_genai.return_value = mock_llm_instance

    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Act
    nodes, edges = await extractor.extract("Sample text")

    # Assert
    assert nodes == []
    assert edges == []


@pytest.mark.asyncio
async def test_gemini_extractor_without_extractor(mocker) -> None:
    """Test extract when extractor failed to initialize.

    Arrange: Initialize extractor forcing initialization failure.
    Act: Call extract.
    Assert: Returns empty lists.
    """
    # Arrange
    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=Exception("Failed"))
    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Act
    nodes, edges = await extractor.extract("Sample text")

    # Assert
    assert nodes == []
    assert edges == []
