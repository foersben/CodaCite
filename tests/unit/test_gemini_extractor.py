"""Tests for GeminiEntityExtractor.

This module validates the integration with Google's Gemini API for entity and
relation extraction, specifically within the Infrastructure layer (Extraction).
"""

import pytest

from app.domain.models import Edge, Node
from app.infrastructure.extraction import ExtractedGraph, GeminiEntityExtractor


@pytest.fixture
def mock_chat_google_genai(mocker):
    """Mock ChatGoogleGenerativeAI."""
    return mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI")


def test_gemini_extractor_init_success(mocker, mock_chat_google_genai) -> None:
    """Test GeminiEntityExtractor initializes properly when import succeeds.

    Given: The ChatGoogleGenerativeAI library is installed and available.
    When: A GeminiEntityExtractor is initialized with a valid API key.
    Then: It should correctly configure the LLM and structured output extractor.
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

    Given: The ChatGoogleGenerativeAI library initialization fails (e.g., missing package).
    When: A GeminiEntityExtractor is initialized.
    Then: It should handle the failure gracefully by setting LLM and extractor to None.
    """
    # Arrange & Act
    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=Exception("Failed"))
    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Assert
    assert extractor.llm is None
    assert extractor.extractor is None


# A simple class to simulate missing properties cleanly compared to MagicMock
class PartialResult(ExtractedGraph):
    """Partial mock result for testing."""

    def __init__(self, nodes):
        """Init partial result."""
        extracted_nodes = []
        for n in nodes:
            if isinstance(n, Node):
                extracted_nodes.append(n)
            else:
                extracted_nodes.append(Node(id=n["id"], label=n["label"], name=n["name"]))
        super().__init__(nodes=extracted_nodes, edges=[])


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

    Given: A set of possible LLM responses (successful, empty, or partial).
    When: The extract method is called with sample text.
    Then: It should correctly parse the available nodes and edges.
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

    Given: An API error or timeout occurs during extraction.
    When: The extract method is called.
    Then: It should return empty lists and not propagate the exception.
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

    Given: The extractor was not properly initialized due to a previous failure.
    When: The extract method is called.
    Then: It should immediately return empty results.
    """
    # Arrange
    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=Exception("Failed"))
    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Act
    nodes, edges = await extractor.extract("Sample text")

    # Assert
    assert nodes == []
    assert edges == []
