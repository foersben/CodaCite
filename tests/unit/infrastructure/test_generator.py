"""Unit tests for the GeminiGenerator infrastructure adapter.

Validates text generation, conversation history handling, and error resilience
when interacting with the Gemini API via LangChain.
"""

from typing import Any

import pytest

from app.infrastructure.generator import GeminiGenerator


@pytest.fixture
def mock_chat_gemini(mocker: Any) -> Any:
    """Provides a mocked ChatGoogleGenerativeAI class.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked ChatGoogleGenerativeAI class.
    """
    return mocker.patch("app.infrastructure.generator.ChatGoogleGenerativeAI")


@pytest.mark.asyncio
async def test_agenerate_success(mocker: Any, mock_chat_gemini: Any) -> None:
    """Tests successful text generation.

    Given:
        A valid prompt and a functioning LLM.
    When:
        agenerate is called.
    Then:
        It should return the generated text content.

    Args:
        mocker: The pytest-mock fixture.
        mock_chat_gemini: The mocked ChatGoogleGenerativeAI class.
    """
    mock_llm = mocker.MagicMock()
    # ainvoke returns a response object with a content attribute
    mock_response = mocker.MagicMock()
    mock_response.content = "Generated answer"
    mock_llm.ainvoke = mocker.AsyncMock(return_value=mock_response)
    mock_chat_gemini.return_value = mock_llm

    generator = GeminiGenerator(api_key="fake-key")
    response = await generator.agenerate("Prompt")

    assert response == "Generated answer"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_agenerate_failure(mocker: Any, mock_chat_gemini: Any) -> None:
    """Tests generation failure handling.

    Given:
        An LLM that raises an exception during generation.
    When:
        agenerate is called.
    Then:
        It should return a polite error message instead of crashing.

    Args:
        mocker: The pytest-mock fixture.
        mock_chat_gemini: The mocked ChatGoogleGenerativeAI class.
    """
    mock_llm = mocker.MagicMock()
    mock_llm.ainvoke = mocker.AsyncMock(side_effect=Exception("API error"))
    mock_chat_gemini.return_value = mock_llm

    generator = GeminiGenerator(api_key="fake-key")
    response = await generator.agenerate("Prompt")

    assert "I'm sorry, I encountered an error" in response


@pytest.mark.asyncio
async def test_agenerate_with_history(mocker: Any, mock_chat_gemini: Any) -> None:
    """Tests text generation with preserved conversation history.

    Given:
        A list of past message dictionaries and a new prompt.
    When:
        agenerate is called with the history.
    Then:
        The underlying LLM should be called with the full sequence of messages.

    Args:
        mocker: The pytest-mock fixture.
        mock_chat_gemini: The mocked ChatGoogleGenerativeAI class.
    """
    mock_llm = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.content = "Follow-up answer"
    mock_llm.ainvoke = mocker.AsyncMock(return_value=mock_response)
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
