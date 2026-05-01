"""Unit tests for the ChatUseCase.

Validates the RAG orchestration logic, prompt construction, and history handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.chat import ChatUseCase


@pytest.fixture
def mock_retrieval():
    """Provides a mocked GraphRAGRetrievalUseCase."""
    return MagicMock()


@pytest.fixture
def mock_generator():
    """Provides a mocked LLMGenerator."""
    return MagicMock()


@pytest.fixture
def chat_use_case(mock_retrieval, mock_generator):
    """Provides a ChatUseCase instance with mocked dependencies."""
    return ChatUseCase(retrieval_use_case=mock_retrieval, generator=mock_generator)


@pytest.mark.asyncio
async def test_execute_success(chat_use_case, mock_retrieval, mock_generator):
    """Tests successful chat execution with retrieved context."""
    # Mock retrieval results
    mock_retrieval.execute = AsyncMock(
        return_value=[
            {"text": "Chunk 1 content", "source": "doc1.pdf"},
            {"text": "Chunk 2 content", "document_id": "doc2.pdf"},
        ]
    )
    mock_generator.agenerate = AsyncMock(return_value="AI Response")

    query = "Tell me about X"
    response = await chat_use_case.execute(query)

    assert response == "AI Response"
    mock_retrieval.execute.assert_called_once_with(query, top_k=10, notebook_ids=None)

    # Verify generator call and system prompt
    call_args = mock_generator.agenerate.call_args
    history = call_args[1]["history"]
    assert len(history) == 1
    assert history[0]["role"] == "system"
    assert "Chunk 1 content" in history[0]["content"]
    assert "doc1.pdf" in history[0]["content"]
    assert "doc2.pdf" in history[0]["content"]


@pytest.mark.asyncio
async def test_execute_no_context(chat_use_case, mock_retrieval, mock_generator):
    """Tests chat execution when no context is found."""
    mock_retrieval.execute = AsyncMock(return_value=[])
    mock_generator.agenerate = AsyncMock(return_value="I don't know.")

    await chat_use_case.execute("Where is Y?")

    history = mock_generator.agenerate.call_args[1]["history"]
    assert "No relevant context found." in history[0]["content"]


@pytest.mark.asyncio
async def test_execute_with_existing_history(chat_use_case, mock_retrieval, mock_generator):
    """Tests that history is preserved and system prompt is updated or inserted."""
    mock_retrieval.execute = AsyncMock(return_value=[])
    mock_generator.agenerate = AsyncMock(return_value="Response")

    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    await chat_use_case.execute("How are you?", history=history)

    full_history = mock_generator.agenerate.call_args[1]["history"]
    assert len(full_history) == 3
    assert full_history[0]["role"] == "system"
    assert full_history[1] == history[0]
    assert full_history[2] == history[1]


@pytest.mark.asyncio
async def test_execute_with_existing_system_prompt(chat_use_case, mock_retrieval, mock_generator):
    """Tests that an existing system prompt in history is updated."""
    mock_retrieval.execute = AsyncMock(return_value=[])
    mock_generator.agenerate = AsyncMock(return_value="Response")

    history = [{"role": "system", "content": "Old prompt"}, {"role": "user", "content": "Hi"}]
    await chat_use_case.execute("Query", history=history)

    full_history = mock_generator.agenerate.call_args[1]["history"]
    assert len(full_history) == 2
    assert full_history[0]["role"] == "system"
    assert "Old prompt" not in full_history[0]["content"]
    assert "CodaCite" in full_history[0]["content"]
