"""Unit tests for the ChatUseCase.

Validates the retrieval-augmented generation (RAG) logic, including
context formatting, notebook filtering, and error handling.
"""

from typing import Any

import pytest

from app.application.chat import ChatUseCase


@pytest.fixture
def chat_use_case(
    mock_retrieval_use_case: Any,
    mock_llm_generator: Any,
) -> ChatUseCase:
    """Provides a ChatUseCase instance with mocked dependencies.

    Args:
        mock_retrieval_use_case: Mock retrieval use case fixture.
        mock_llm_generator: Mock LLM generator fixture.

    Returns:
        A ChatUseCase instance.
    """
    return ChatUseCase(mock_retrieval_use_case, mock_llm_generator)


@pytest.mark.asyncio
async def test_chat_basic_flow(
    chat_use_case: ChatUseCase,
    mock_retrieval_use_case: Any,
    mock_llm_generator: Any,
) -> None:
    """Tests the basic chat flow with context.

    Given:
        A user query and relevant document results.
    When:
        The ChatUseCase is executed.
    Then:
        It should retrieve context and call the LLM generator.
    """
    # Arrange
    results = [
        {"text": "Fact A", "source": "doc1"},
        {"text": "Fact B", "source": "doc2"},
    ]
    mock_retrieval_use_case.execute.return_value = results
    mock_llm_generator.agenerate.return_value = "Answer based on A and B."

    # Act
    response = await chat_use_case.execute("Tell me about A and B")

    # Assert
    assert response == "Answer based on A and B."
    mock_retrieval_use_case.execute.assert_called_once()

    # Verify context injection in system prompt
    called_history = mock_llm_generator.agenerate.call_args[1]["history"]
    system_prompt = called_history[0]["content"]
    assert "Fact A" in system_prompt
    assert "[Source: doc1]" in system_prompt


@pytest.mark.asyncio
async def test_chat_no_results_placeholder(
    chat_use_case: ChatUseCase,
    mock_retrieval_use_case: Any,
    mock_llm_generator: Any,
) -> None:
    """Tests chat behavior when no relevant documents are found.

    Given:
        A query that returns no retrieval results.
    When:
        ChatUseCase is executed.
    Then:
        It should still call the generator with a system prompt indicating no context.
    """
    # Arrange
    mock_retrieval_use_case.execute.return_value = []
    mock_llm_generator.agenerate.return_value = "I don't know."

    # Act
    await chat_use_case.execute("What is X?")

    # Assert
    called_history = mock_llm_generator.agenerate.call_args[1]["history"]
    system_prompt = called_history[0]["content"]
    assert "### DOCUMENT CONTEXT:" in system_prompt
    assert "No relevant context found." in system_prompt


@pytest.mark.asyncio
async def test_chat_notebook_filtering(
    chat_use_case: ChatUseCase,
    mock_retrieval_use_case: Any,
) -> None:
    """Tests that notebook IDs are passed correctly to the retrieval use case."""
    # Arrange
    notebook_ids = ["notebook:123"]
    mock_retrieval_use_case.execute.return_value = []

    # Act
    await chat_use_case.execute("Hello", notebook_ids=notebook_ids)

    # Assert
    mock_retrieval_use_case.execute.assert_called_once_with(
        "Hello", top_k=10, notebook_ids=notebook_ids
    )


@pytest.mark.asyncio
async def test_chat_error_propagation(
    chat_use_case: ChatUseCase,
    mock_llm_generator: Any,
) -> None:
    """Tests that exceptions from the LLM generator are propagated."""
    # Arrange
    mock_llm_generator.agenerate.side_effect = Exception("LLM failure")

    # Act & Assert
    with pytest.raises(Exception, match="LLM failure"):
        await chat_use_case.execute("Query")
