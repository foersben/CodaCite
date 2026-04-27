"""Unit tests for the ChatUseCase.

Validates the coordination between retrieval and generation, including
proper history handling and prompt construction.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.chat import ChatUseCase
from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.ports import LLMGenerator


@pytest.fixture
def mock_retrieval():
    """Docstring generated to satisfy ruff D103."""
    return MagicMock(spec=GraphRAGRetrievalUseCase)


@pytest.fixture
def mock_generator():
    """Docstring generated to satisfy ruff D103."""
    return MagicMock(spec=LLMGenerator)


@pytest.fixture
def chat_use_case(mock_retrieval, mock_generator):
    """Docstring generated to satisfy ruff D103."""
    return ChatUseCase(mock_retrieval, mock_generator)


@pytest.mark.asyncio
async def test_chat_execute_no_history(chat_use_case, mock_retrieval, mock_generator):
    """Docstring generated to satisfy ruff D103."""
    # Arrange
    query = "What is the capital of France?"
    mock_retrieval.execute = AsyncMock(return_value=[{"text": "Paris is the capital."}])
    mock_generator.agenerate = AsyncMock(return_value="Paris.")

    # Act
    response = await chat_use_case.execute(query)

    # Assert
    assert response == "Paris."
    mock_retrieval.execute.assert_called_once_with(query, top_k=10, notebook_ids=None)

    # Verify system prompt construction
    called_history = mock_generator.agenerate.call_args[1]["history"]
    assert len(called_history) == 1
    assert called_history[0]["role"] == "system"
    assert "Paris is the capital." in called_history[0]["content"]


@pytest.mark.asyncio
async def test_chat_execute_with_existing_history(chat_use_case, mock_retrieval, mock_generator):
    """Docstring generated to satisfy ruff D103."""
    # Arrange
    query = "And its population?"
    history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    mock_retrieval.execute = AsyncMock(return_value=[{"text": "Population of Paris is 2.1M."}])
    mock_generator.agenerate = AsyncMock(return_value="2.1 million.")

    # Act
    response = await chat_use_case.execute(query, history=history)

    # Assert
    assert response == "2.1 million."
    called_history = mock_generator.agenerate.call_args[1]["history"]
    assert len(called_history) == 3
    assert called_history[0]["role"] == "system"
    assert called_history[1]["content"] == "What is the capital of France?"
    assert "Population of Paris is 2.1M." in called_history[0]["content"]


@pytest.mark.asyncio
async def test_chat_execute_updates_existing_system_prompt(
    chat_use_case, mock_retrieval, mock_generator
):
    """Docstring generated to satisfy ruff D103."""
    # Arrange
    query = "Hello"
    history = [
        {"role": "system", "content": "Old system prompt"},
        {"role": "user", "content": "Hi"},
    ]
    mock_retrieval.execute = AsyncMock(return_value=[{"text": "New context"}])
    mock_generator.agenerate = AsyncMock(return_value="Greeting.")

    # Act
    await chat_use_case.execute(query, history=history)

    # Assert
    called_history = mock_generator.agenerate.call_args[1]["history"]
    assert len(called_history) == 2
    assert called_history[0]["role"] == "system"
    assert "New context" in called_history[0]["content"]
    assert "Old system prompt" not in called_history[0]["content"]
