"""Unit tests for the ChatUseCase.

Validates the streaming orchestration logic, prompt construction, and history handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.pipelines.generation.chat import ChatUseCase


@pytest.fixture
def mock_retrieval():
    """Provides a mocked GraphRAGRetrievalUseCase."""
    return MagicMock()


@pytest.fixture
def mock_generator():
    """Provides a mocked LLMGenerator."""
    return MagicMock()


@pytest.fixture
def mock_router():
    """Provides a mocked QueryRouter."""
    router = MagicMock()
    router.classify_intent.return_value = "qa"
    return router


@pytest.fixture
def mock_document_store():
    """Provides a mocked DocumentStore."""
    store = MagicMock()
    store.get_all_documents = AsyncMock(return_value=[])
    store.get_document_summaries = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_guardrail():
    """Provides a mocked FactualityGuardrail."""
    return MagicMock()


@pytest.fixture
def chat_use_case(mock_retrieval, mock_generator, mock_router, mock_document_store, mock_guardrail):
    """Provides a ChatUseCase instance with mocked dependencies."""
    return ChatUseCase(
        retrieval_use_case=mock_retrieval,
        generator=mock_generator,
        router=mock_router,
        document_store=mock_document_store,
        guardrail=mock_guardrail,
    )


@pytest.mark.asyncio
async def test_execute_success(chat_use_case, mock_retrieval, mock_generator, mock_guardrail):
    """Tests successful chat execution with streaming and guardrail verification."""
    # Mock retrieval results
    mock_retrieval.execute = AsyncMock(
        return_value={
            "documents": [
                {"text": "Chunk 1 content", "chunk_id": "c1", "document_id": "doc1"},
                {"text": "Chunk 2 content", "chunk_id": "c2", "document_id": "doc2"},
            ]
        }
    )

    async def _mock_stream(*args, **kwargs):
        yield "Hello"
        yield " World"

    mock_generator.generate_stream = _mock_stream
    mock_guardrail.verify.return_value = True

    query = "Tell me about X"
    chunks = [chunk async for chunk in chat_use_case.execute(query)]

    # We expect 3 chunks: 2 text tokens + 1 final citation payload
    assert len(chunks) == 3
    assert json.loads(chunks[0]) == {"token": "Hello"}
    assert json.loads(chunks[1]) == {"token": " World"}

    # Final chunk is SSE formatted
    last_chunk = chunks[2]
    assert last_chunk.startswith("event: citations")
    # Extract JSON from data: { ... }
    json_str = last_chunk.split("data: ")[1].strip()
    citations = json.loads(json_str)
    assert citations["verified"] is True
    assert citations["warning"] is False
    assert len(citations["documents"]) == 2

    mock_retrieval.execute.assert_called_once_with(query, history=[], top_k=10, notebook_ids=None)
    mock_guardrail.verify.assert_called_once_with("Chunk 1 content\nChunk 2 content", "Hello World")


@pytest.mark.asyncio
async def test_execute_summarize_intent(
    chat_use_case, mock_router, mock_document_store, mock_generator, mock_guardrail
):
    """Tests that a summarization query bypasses retrieval and uses global summaries."""
    mock_router.classify_intent.return_value = "summarize"
    mock_document_store.get_document_summaries = AsyncMock(
        return_value=[
            {"filename": "doc1.pdf", "summary": "Global summary of doc 1"},
            {"filename": "doc2.pdf", "summary": "Global summary of doc 2"},
        ]
    )

    async def _mock_stream(*args, **kwargs):
        yield "Summary"
        yield " Response"

    mock_generator.generate_stream = _mock_stream
    mock_guardrail.verify.return_value = True

    query = "Summarize the documents"
    chunks = [chunk async for chunk in chat_use_case.execute(query, notebook_ids=["nb1"])]

    # 2 text tokens + 1 final citation payload
    assert len(chunks) == 3
    assert json.loads(chunks[0]) == {"token": "Summary"}
    assert json.loads(chunks[1]) == {"token": " Response"}

    # Final chunk is SSE formatted
    last_chunk = chunks[2]
    json_str = last_chunk.split("data: ")[1].strip()
    citations = json.loads(json_str)
    assert citations["verified"] is True
    # For summaries, we expect no exact source documents passed back for citation
    assert len(citations["documents"]) == 2  # Actually summaries create synthetic docs now

    mock_document_store.get_document_summaries.assert_called_once_with(active_notebook_ids=["nb1"])


@pytest.mark.asyncio
async def test_execute_with_existing_history(
    chat_use_case, mock_retrieval, mock_generator, mock_guardrail
):
    """Tests that history is passed to the generation stream."""
    mock_retrieval.execute = AsyncMock(return_value={"documents": []})

    async def _mock_stream(*args, **kwargs):
        yield "Response"

    mock_generator.generate_stream = _mock_stream
    mock_guardrail.verify.return_value = False

    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    chunks = [chunk async for chunk in chat_use_case.execute("How are you?", history=history)]

    # 1 text token + 1 final citation payload
    assert len(chunks) == 2
    assert json.loads(chunks[0]) == {"token": "Response"}

    last_chunk = chunks[1]
    json_str = last_chunk.split("data: ")[1].strip()
    citations = json.loads(json_str)
    assert citations["verified"] is False
    assert citations["warning"] is True

    # Verify history was passed to retrieval
    mock_retrieval.execute.assert_called_once()
    passed_history = mock_retrieval.execute.call_args[1]["history"]
    assert passed_history == history
