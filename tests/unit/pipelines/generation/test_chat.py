"""Unit tests for the ChatUseCase.

Validates the RAG orchestration logic, prompt construction, and history handling.
"""

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
    return MagicMock()


@pytest.fixture
def chat_use_case(mock_retrieval, mock_generator, mock_router, mock_document_store):
    """Provides a ChatUseCase instance with mocked dependencies."""
    return ChatUseCase(
        retrieval_use_case=mock_retrieval,
        generator=mock_generator,
        router=mock_router,
        document_store=mock_document_store,
    )


@pytest.mark.asyncio
async def test_execute_success(chat_use_case, mock_retrieval, mock_generator):
    """Tests successful chat execution with retrieved context."""
    # Mock retrieval results
    mock_retrieval.execute = AsyncMock(
        return_value={
            "generation": [
                {"text": "Chunk 1 content", "source": "doc1.pdf"},
                {"text": "Chunk 2 content", "document_id": "doc2.pdf"},
            ],
            "answer": "Graph Answer",
        }
    )

    query = "Tell me about X"
    response = await chat_use_case.execute(query)

    assert response == "Graph Answer"
    mock_retrieval.execute.assert_called_once_with(query, history=[], top_k=10, notebook_ids=None)
    # For QA, ChatUseCase doesn't call generator.agenerate directly anymore
    assert mock_generator.agenerate.call_count == 0


@pytest.mark.asyncio
async def test_execute_summarize_intent(
    chat_use_case, mock_router, mock_document_store, mock_generator
):
    """Tests that a summarization query bypasses retrieval and uses global summaries."""
    mock_router.classify_intent.return_value = "summarize"
    mock_document_store.get_document_summaries = AsyncMock(
        return_value=[
            {"filename": "doc1.pdf", "summary": "Global summary of doc 1"},
            {"filename": "doc2.pdf", "summary": "Global summary of doc 2"},
        ]
    )
    mock_generator.agenerate = AsyncMock(return_value="Summary Response")

    query = "Summarize the documents"
    response = await chat_use_case.execute(query, notebook_ids=["nb1"])

    assert response == "Summary Response"
    mock_document_store.get_document_summaries.assert_called_once_with(active_notebook_ids=["nb1"])

    # Verify context construction
    history = mock_generator.agenerate.call_args[1]["history"]
    assert "Global summary of doc 1" in history[0]["content"]
    assert "Global summary of doc 2" in history[0]["content"]


@pytest.mark.asyncio
async def test_execute_no_context(chat_use_case, mock_retrieval, mock_generator):
    """Tests chat execution when no context is found (QA intent)."""
    mock_retrieval.execute = AsyncMock(
        return_value={
            "generation": [],
            "answer": "I don't know.",
        }
    )

    response = await chat_use_case.execute("Where is Y?")

    assert response == "I don't know."
    assert mock_generator.agenerate.call_count == 0


@pytest.mark.asyncio
async def test_execute_with_existing_history(chat_use_case, mock_retrieval, mock_generator):
    """Tests that history is passed to the retrieval use case for QA."""
    mock_retrieval.execute = AsyncMock(return_value={"generation": [], "answer": "Response"})

    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    await chat_use_case.execute("How are you?", history=history)

    # Verify history was passed to retrieval
    mock_retrieval.execute.assert_called_once()
    passed_history = mock_retrieval.execute.call_args[1]["history"]
    assert passed_history == history
