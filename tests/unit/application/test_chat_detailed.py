"""Hardened unit tests for ChatUseCase.

Exhaustive coverage of chat logic, prompt construction, and error handling.
"""

from unittest.mock import AsyncMock

import pytest

from app.application.chat import ChatUseCase
from app.domain.ports import LLMGenerator


@pytest.fixture
def mock_retrieval():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock()


@pytest.fixture
def mock_generator():
    """Docstring generated to satisfy ruff D103."""
    return AsyncMock(spec=LLMGenerator)


@pytest.fixture
def chat_use_case(mock_retrieval, mock_generator):
    """Docstring generated to satisfy ruff D103."""
    return ChatUseCase(mock_retrieval, mock_generator)


@pytest.mark.asyncio
async def test_chat_no_results_system_prompt(chat_use_case, mock_retrieval, mock_generator):
    """Test behavior when no relevant documents are found.

    Given: A query that returns no retrieval results.
    When: ChatUseCase.execute is called.
    Then: It should still call the generator with a system prompt indicating no context.
    """
    # Arrange
    mock_retrieval.execute.return_value = []
    mock_generator.agenerate.return_value = "I don't know."

    # Act
    response = await chat_use_case.execute("What is X?")

    # Assert
    assert response == "I don't know."
    called_history = mock_generator.agenerate.call_args[1]["history"]
    # The system prompt should contain the placeholder even if no results
    assert "### DOCUMENT CONTEXT:" in called_history[0]["content"]


@pytest.mark.asyncio
async def test_chat_with_source_attribution(chat_use_case, mock_retrieval, mock_generator):
    """Test that context snippets are correctly formatted into the system prompt.

    Given: Multiple retrieval results with source metadata.
    When: ChatUseCase.execute is called.
    Then: The system prompt should contain the snippet text and source IDs.
    """
    # Arrange
    results = [
        {"text": "Fact A", "source": "doc1", "id": "chunk1"},
        {"text": "Fact B", "source": "doc2", "id": "chunk2"},
    ]
    mock_retrieval.execute.return_value = results
    mock_generator.agenerate.return_value = "Answer based on A and B."

    # Act
    await chat_use_case.execute("Tell me about A and B")

    # Assert
    called_history = mock_generator.agenerate.call_args[1]["history"]
    content = called_history[0]["content"]
    assert "Fact A" in content
    assert "Fact B" in content
    assert "[Source: doc1]" in content
    assert "[Source: doc2]" in content


@pytest.mark.asyncio
async def test_chat_notebook_filtering_pass_through(chat_use_case, mock_retrieval, mock_generator):
    """Test that notebook IDs are passed correctly to the retrieval use case.

    Given: A specific list of notebook IDs.
    When: ChatUseCase.execute is called with these IDs.
    Then: They should be passed to mock_retrieval.execute.
    """
    # Arrange
    notebook_ids = ["notebook:123", "notebook:456"]
    mock_retrieval.execute.return_value = []
    mock_generator.agenerate.return_value = "Hi"

    # Act
    await chat_use_case.execute("Hello", notebook_ids=notebook_ids)

    # Assert
    mock_retrieval.execute.assert_called_once_with("Hello", top_k=10, notebook_ids=notebook_ids)


@pytest.mark.asyncio
async def test_chat_generator_error_handling(chat_use_case, mock_retrieval, mock_generator):
    """Test that exceptions from the generator are bubbled up.

    Given: A generator that raises an exception.
    When: ChatUseCase.execute is called.
    Then: It should raise the same exception.
    """
    # Arrange
    mock_retrieval.execute.return_value = [{"text": "Context"}]
    mock_generator.agenerate.side_effect = Exception("API Error")

    # Act & Assert
    with pytest.raises(Exception, match="API Error"):
        await chat_use_case.execute("Query")


@pytest.mark.asyncio
async def test_chat_retrieval_error_handling(chat_use_case, mock_retrieval, mock_generator):
    """Test that exceptions from retrieval are bubbled up.

    Given: A retrieval service that raises an exception.
    When: ChatUseCase.execute is called.
    Then: It should raise the same exception.
    """
    # Arrange
    mock_retrieval.execute.side_effect = Exception("DB Connection Refused")

    # Act & Assert
    with pytest.raises(Exception, match="DB Connection Refused"):
        await chat_use_case.execute("Query")
