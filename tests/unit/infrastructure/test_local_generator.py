"""Unit tests for the LocalLlamaGenerator.

Validates that the local LLM generator correctly wraps the ChatLlamaCpp
implementation and handles history mapping and error conditions.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.infrastructure.local_generator import LocalLlamaGenerator


@pytest.fixture
def mock_chat_llama(mocker):
    """Fixture to mock ChatLlamaCpp."""
    return mocker.patch("app.infrastructure.local_generator.ChatLlamaCpp")


@pytest.mark.asyncio
async def test_init_success(mock_chat_llama):
    """Tests successful initialization of the local generator."""
    generator = LocalLlamaGenerator("test/model.gguf")
    assert generator.llm is not None
    mock_chat_llama.assert_called_once()


@pytest.mark.asyncio
async def test_init_failure(mock_chat_llama):
    """Tests handling of initialization failures."""
    mock_chat_llama.side_effect = Exception("Load error")
    generator = LocalLlamaGenerator("test/model.gguf")
    assert generator.llm is None


@pytest.mark.asyncio
async def test_agenerate_success(mock_chat_llama, mocker):
    """Tests successful text generation with history mapping."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = mocker.AsyncMock(return_value=AIMessage(content="Generated response"))
    mock_chat_llama.return_value = mock_llm

    generator = LocalLlamaGenerator("test/model.gguf")
    history = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]

    response = await generator.agenerate("Tell me a story", history=history)

    assert response == "Generated response"
    mock_llm.ainvoke.assert_called_once()
    # Verify messages list length (system + user + assistant + current prompt)
    messages = mock_llm.ainvoke.call_args[0][0]
    assert len(messages) == 4
    assert isinstance(messages[0].content, str)
    assert messages[0].content == "You are helpful"


@pytest.mark.asyncio
async def test_agenerate_no_llm():
    """Tests agenerate when the LLM was not properly initialized."""
    # We bypass the constructor's successful init by making it fail
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            "app.infrastructure.local_generator.ChatLlamaCpp",
            lambda **kwargs: exec('raise Exception("fail")'),
        )
        generator = LocalLlamaGenerator("bad/path")

    response = await generator.agenerate("prompt")
    assert response == "Local model is not initialized."


@pytest.mark.asyncio
async def test_agenerate_error(mock_chat_llama, mocker):
    """Tests handling of errors during generation."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = mocker.AsyncMock(side_effect=Exception("Runtime error"))
    mock_chat_llama.return_value = mock_llm

    generator = LocalLlamaGenerator("test/model.gguf")
    response = await generator.agenerate("prompt")

    assert "encountered an error" in response
    assert "Runtime error" in response
