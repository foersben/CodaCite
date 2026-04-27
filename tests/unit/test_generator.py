"""Unit tests for the GeminiGenerator.

Validates text generation and structured response handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.generator import GeminiGenerator


@pytest.fixture
def mock_chat_gemini():
    """Docstring generated to satisfy ruff D103."""
    with patch("app.infrastructure.generator.ChatGoogleGenerativeAI") as mock:
        yield mock


@pytest.mark.asyncio
async def test_agenerate_success(mock_chat_gemini):
    """Test successful generation."""
    mock_llm = MagicMock()
    # ainvoke returns a response object with a content attribute
    mock_response = MagicMock()
    mock_response.content = "Generated answer"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_chat_gemini.return_value = mock_llm

    generator = GeminiGenerator(api_key="fake-key")
    response = await generator.agenerate("Prompt")

    assert response == "Generated answer"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_agenerate_failure(mock_chat_gemini):
    """Test generation failure handling."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API error"))
    mock_chat_gemini.return_value = mock_llm

    generator = GeminiGenerator(api_key="fake-key")
    response = await generator.agenerate("Prompt")

    assert "I'm sorry, I encountered an error" in response


@pytest.mark.asyncio
async def test_agenerate_with_history(mock_chat_gemini):
    """Test generation with conversation history."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Follow-up answer"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_chat_gemini.return_value = mock_llm

    generator = GeminiGenerator(api_key="fake-key")
    history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    await generator.agenerate("How are you?", history=history)

    # Check that it was called with 3 messages (User, Assistant, User)
    messages = mock_llm.ainvoke.call_args[0][0]
    assert len(messages) == 3
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there!"
    assert messages[2].content == "How are you?"
