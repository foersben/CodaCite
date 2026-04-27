from unittest.mock import AsyncMock

import pytest

from app.application.chat import ChatUseCase


@pytest.mark.asyncio
async def test_chat_execute_success():
    """Docstring generated to satisfy ruff D103."""
    # Arrange
    mock_retrieval = AsyncMock()
    mock_retrieval.execute.return_value = [
        {"text": "Python is a programming language.", "source": "doc1.txt"},
        {"text": "FastAPI is for building APIs.", "document_id": "doc2.pdf"},
    ]

    mock_generator = AsyncMock()
    mock_generator.agenerate.return_value = "Python and FastAPI are great."

    use_case = ChatUseCase(mock_retrieval, mock_generator)

    # Act
    response = await use_case.execute("Tell me about Python", history=[])

    # Assert
    assert response == "Python and FastAPI are great."
    mock_retrieval.execute.assert_called_once_with(
        "Tell me about Python", top_k=10, notebook_ids=None
    )

    # Verify generator call - check if system prompt contains context
    args, kwargs = mock_generator.agenerate.call_args
    assert args[0] == "Tell me about Python"
    history = kwargs["history"]
    assert history[0]["role"] == "system"
    assert "Python is a programming language." in history[0]["content"]
    assert "[Source: doc1.txt]" in history[0]["content"]
    assert "[Source: doc2.pdf]" in history[0]["content"]


@pytest.mark.asyncio
async def test_chat_execute_with_history():
    """Docstring generated to satisfy ruff D103."""
    # Arrange
    mock_retrieval = AsyncMock()
    mock_retrieval.execute.return_value = []

    mock_generator = AsyncMock()
    mock_generator.agenerate.return_value = "Answer based on history."

    use_case = ChatUseCase(mock_retrieval, mock_generator)
    existing_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # Act
    await use_case.execute("How are you?", history=existing_history)

    # Assert
    args, kwargs = mock_generator.agenerate.call_args
    history = kwargs["history"]
    assert len(history) == 3  # system + 2 original messages
    assert history[0]["role"] == "system"
    assert history[1]["content"] == "Hello"
    assert history[2]["content"] == "Hi there!"
