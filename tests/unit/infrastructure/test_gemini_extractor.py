"""Unit tests for the GeminiEntityExtractor infrastructure adapter.

Validates the integration with Google's Gemini API (via LangChain) for
structured entity and relation extraction, ensuring robust error handling.
"""

from typing import Any

import pytest

from app.domain.models import Edge, Node
from app.infrastructure.extraction import ExtractedGraph, GeminiEntityExtractor


@pytest.fixture
def mock_chat_google_genai(mocker: Any) -> Any:
    """Mock the ChatGoogleGenerativeAI class.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked ChatGoogleGenerativeAI class.
    """
    return mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI")


def test_gemini_extractor_init_success(mocker: Any, mock_chat_google_genai: Any) -> None:
    """Tests successful initialization of GeminiEntityExtractor.

    Given:
        The ChatGoogleGenerativeAI library is installed and available.
    When:
        A GeminiEntityExtractor is initialized with a valid API key.
    Then:
        It should correctly configure the LLM and structured output extractor.

    Args:
        mocker: The pytest-mock fixture.
        mock_chat_google_genai: The mocked ChatGoogleGenerativeAI class.
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


def test_gemini_extractor_init_failure(mocker: Any) -> None:
    """Tests that GeminiEntityExtractor handles initialization failures gracefully.

    Given:
        Library initialization fails (e.g., missing package).
    When:
        A GeminiEntityExtractor is initialized.
    Then:
        It should set LLM and extractor to None without crashing.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=Exception("Failed"))

    # Act
    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Assert
    assert extractor.llm is None
    assert extractor.extractor is None


class PartialResult(ExtractedGraph):
    """A partial ExtractedGraph result used for testing robust parsing."""

    def __init__(self, nodes: list[Node | dict[str, Any]]) -> None:
        """Initializes a partial result with only nodes.

        Args:
            nodes: A list of Node objects or dictionaries representing nodes.
        """
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
        # Happy path (perfect ExtractedGraph object)
        (
            ExtractedGraph(
                nodes=[Node(id="n1", label="PERSON", name="Alice")],
                edges=[Edge(source_id="n1", target_id="n2", relation="KNOWS")],
            ),
            [Node(id="n1", label="PERSON", name="Alice")],
            [Edge(source_id="n1", target_id="n2", relation="KNOWS")],
        ),
        # Empty response
        (
            ExtractedGraph(nodes=[], edges=[]),
            [],
            [],
        ),
        # Partial result (missing edges attribute)
        (
            PartialResult(nodes=[Node(id="n1", label="PERSON", name="Bob")]),
            [Node(id="n1", label="PERSON", name="Bob")],
            [],
        ),
    ],
)
async def test_gemini_extractor_extract_success(
    mocker: Any,
    mock_chat_google_genai: Any,
    ainvoke_result: ExtractedGraph,
    expected_nodes: list[Node],
    expected_edges: list[Edge],
) -> None:
    """Tests that extract parses various structured outputs correctly.

    Given:
        A set of possible LLM responses (successful, empty, or partial).
    When:
        The extract method is called with sample text.
    Then:
        It should correctly parse and return the available nodes and edges.

    Args:
        mocker: The pytest-mock fixture.
        mock_chat_google_genai: The mocked ChatGoogleGenerativeAI class.
        ainvoke_result: The simulated LLM response.
        expected_nodes: The expected list of extracted nodes.
        expected_edges: The expected list of extracted edges.
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
async def test_gemini_extractor_extract_timeout_or_error(
    mocker: Any, mock_chat_google_genai: Any
) -> None:
    """Tests that extract handles API exceptions gracefully.

    Given:
        An API error or timeout occurs during extraction.
    When:
        The extract method is called.
    Then:
        It should return empty lists instead of propagating the exception.

    Args:
        mocker: The pytest-mock fixture.
        mock_chat_google_genai: The mocked ChatGoogleGenerativeAI class.
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
async def test_gemini_extractor_without_extractor(mocker: Any) -> None:
    """Tests extract when the extractor was not properly initialized.

    Given:
        An uninitialized extractor due to a previous library failure.
    When:
        The extract method is called.
    Then:
        It should immediately return empty results without attempting API calls.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=Exception("Failed"))
    extractor = GeminiEntityExtractor(api_key="fake-key")

    # Act
    nodes, edges = await extractor.extract("Sample text")

    # Assert
    assert nodes == []
    assert edges == []
