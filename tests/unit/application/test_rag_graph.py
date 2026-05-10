"""Unit tests for the LangGraph RAG graph nodes and routing logic.

Each node factory is tested in isolation using AsyncMock dependencies,
following the Arrange-Act-Assert pattern.
"""

from typing import Any

import pytest

from app.application.rag_graph import (
    RAGState,
    _make_router,
    make_generate_node,
    make_grade_documents_node,
    make_retrieve_node,
    make_rewrite_query_node,
)
from app.domain.models import Chunk, Edge, Node


def _make_state(**overrides: Any) -> RAGState:
    """Build a minimal RAGState with sensible defaults.

    Args:
        **overrides: Fields to override on the default state.

    Returns:
        A fully populated RAGState dict.
    """
    base: RAGState = {
        "question": "What is machine learning?",
        "documents": [],
        "generation": [],
        "hallucination_score": 0.0,
        "rewrite_count": 0,
        "top_k": 5,
        "notebook_ids": None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_returns_chunks(mocker: Any) -> None:
    """Tests that retrieve_node correctly populates documents from search_chunks.

    Given:
        A document store returning one chunk and no graph entities.
    When:
        retrieve_node is invoked.
    Then:
        The state documents list contains one entry with type 'chunk'.
    """
    from app.domain.ports import DocumentStore, Embedder, GraphStore

    mock_store = mocker.AsyncMock(spec=DocumentStore)
    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_graph_store = mocker.AsyncMock(spec=GraphStore)
    mock_linker = mocker.AsyncMock()

    mock_embedder.embed.return_value = [0.1, 0.2]
    mock_store.search_chunks.return_value = [
        Chunk(id="c1", text="Relevant text.", document_id="d1", index=0)
    ]
    mock_graph_store.get_all_nodes.return_value = []
    mock_linker.link_entities.return_value = []

    node = make_retrieve_node(mock_store, mock_embedder, mock_graph_store, mock_linker)
    result = await node(_make_state())

    docs = result["documents"]
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0]["text"] == "Relevant text."
    assert docs[0]["type"] == "chunk"


@pytest.mark.asyncio
async def test_retrieve_node_includes_graph_context(mocker: Any) -> None:
    """Tests that entity and relation snippets are appended to documents.

    Given:
        A linker that returns one linked entity and traversal returns node+edge.
    When:
        retrieve_node is invoked.
    Then:
        The documents list contains chunk, entity, and relation entries.
    """
    from app.domain.ports import DocumentStore, Embedder, GraphStore

    mock_store = mocker.AsyncMock(spec=DocumentStore)
    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_graph_store = mocker.AsyncMock(spec=GraphStore)
    mock_linker = mocker.AsyncMock()

    mock_embedder.embed.return_value = [0.1]
    mock_store.search_chunks.return_value = [
        Chunk(id="c1", text="chunk text", document_id="d1", index=0)
    ]
    mock_graph_store.get_all_nodes.return_value = [Node(id="n1", name="A", label="T")]
    mock_linker.link_entities.return_value = [Node(id="n1", name="A", label="T")]
    mock_graph_store.traverse.return_value = (
        [Node(id="n1", name="A", label="T", description="desc")],
        [Edge(source_id="n1", target_id="n2", relation="relates_to")],
    )

    node = make_retrieve_node(mock_store, mock_embedder, mock_graph_store, mock_linker)
    result = await node(_make_state())

    types = {str(d["type"]) for d in result["documents"]}  # type: ignore[arg-type]
    assert "chunk" in types
    assert "entity" in types
    assert "relation" in types


@pytest.mark.asyncio
async def test_retrieve_node_deduplicates(mocker: Any) -> None:
    """Tests that duplicate text snippets are deduplicated.

    Given:
        Two chunks with identical text.
    When:
        retrieve_node is invoked.
    Then:
        Only one entry appears in documents.
    """
    from app.domain.ports import DocumentStore, Embedder, GraphStore

    mock_store = mocker.AsyncMock(spec=DocumentStore)
    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_graph_store = mocker.AsyncMock(spec=GraphStore)
    mock_linker = mocker.AsyncMock()

    mock_embedder.embed.return_value = [0.1]
    mock_store.search_chunks.return_value = [
        Chunk(id="c1", text="same text", document_id="d1", index=0),
        Chunk(id="c2", text="same text", document_id="d1", index=1),
    ]
    mock_graph_store.get_all_nodes.return_value = []
    mock_linker.link_entities.return_value = []

    node = make_retrieve_node(mock_store, mock_embedder, mock_graph_store, mock_linker)
    result = await node(_make_state())

    assert len(result["documents"]) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# grade_documents_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_node_keeps_relevant_docs(mocker: Any) -> None:
    """Tests that grade_node retains documents where the LLM says 'yes'.

    Given:
        Two documents and an LLM that says 'yes' then 'no'.
    When:
        grade_documents_node is invoked.
    Then:
        Only the first document is kept.
    """
    from app.domain.ports import LLMGenerator

    mock_gen = mocker.AsyncMock(spec=LLMGenerator)
    mock_gen.agenerate.side_effect = ["yes, it is relevant", "No, unrelated."]

    state = _make_state(
        documents=[
            {"text": "Relevant doc", "type": "chunk"},
            {"text": "Irrelevant doc", "type": "chunk"},
        ]
    )

    node = make_grade_documents_node(mock_gen)
    result = await node(state)

    assert len(result["documents"]) == 1  # type: ignore[arg-type]
    assert result["documents"][0]["text"] == "Relevant doc"  # type: ignore[index]


@pytest.mark.asyncio
async def test_grade_node_empty_when_all_irrelevant(mocker: Any) -> None:
    """Tests that grade_node returns an empty list when all docs are irrelevant.

    Given:
        One document and an LLM that says 'no'.
    When:
        grade_documents_node is invoked.
    Then:
        The documents list is empty.
    """
    from app.domain.ports import LLMGenerator

    mock_gen = mocker.AsyncMock(spec=LLMGenerator)
    mock_gen.agenerate.return_value = "no"

    state = _make_state(documents=[{"text": "Bad doc", "type": "chunk"}])
    node = make_grade_documents_node(mock_gen)
    result = await node(state)

    assert result["documents"] == []


# ---------------------------------------------------------------------------
# rewrite_query_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewrite_node_updates_question(mocker: Any) -> None:
    """Tests that rewrite_node replaces the question with the LLM's output.

    Given:
        A state with an original question and a generator returning a new question.
    When:
        rewrite_query_node is invoked.
    Then:
        The question is updated and rewrite_count is incremented.
    """
    from app.domain.ports import LLMGenerator

    mock_gen = mocker.AsyncMock(spec=LLMGenerator)
    mock_gen.agenerate.return_value = "  What is deep learning?  "

    state = _make_state(question="What is ML?", rewrite_count=0)
    node = make_rewrite_query_node(mock_gen)
    result = await node(state)

    assert result["question"] == "What is deep learning?"
    assert result["rewrite_count"] == 1


@pytest.mark.asyncio
async def test_rewrite_node_keeps_original_on_empty_response(mocker: Any) -> None:
    """Tests that an empty LLM response falls back to the original question.

    Given:
        A generator returning an empty string.
    When:
        rewrite_query_node is invoked.
    Then:
        The original question is preserved.
    """
    from app.domain.ports import LLMGenerator

    mock_gen = mocker.AsyncMock(spec=LLMGenerator)
    mock_gen.agenerate.return_value = "   "

    state = _make_state(question="original?", rewrite_count=1)
    node = make_rewrite_query_node(mock_gen)
    result = await node(state)

    assert result["question"] == "original?"
    assert result["rewrite_count"] == 2


# ---------------------------------------------------------------------------
# generate_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_node_uses_reranker(mocker: Any) -> None:
    """Tests that generate_node delegates to the reranker when available.

    Given:
        Two documents and a functional reranker.
    When:
        generate_node is invoked.
    Then:
        The reranker output is returned as the generation.
    """
    from app.domain.ports import LLMGenerator

    mock_gen = mocker.AsyncMock(spec=LLMGenerator)
    mock_reranker = mocker.AsyncMock()
    mock_reranker.rerank.return_value = [{"text": "reranked", "score": 0.99}]

    state = _make_state(
        documents=[{"text": "doc A", "type": "chunk"}, {"text": "doc B", "type": "chunk"}]
    )
    node = make_generate_node(mock_gen, mock_reranker)
    result = await node(state)

    assert result["generation"] == [{"text": "reranked", "score": 0.99}]
    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_generate_node_fallback_without_reranker(mocker: Any) -> None:
    """Tests that generate_node returns plain dicts when reranker is unavailable.

    Given:
        Two documents and an object with no rerank method.
    When:
        generate_node is invoked.
    Then:
        Plain fallback dicts with score=1.0 are returned.
    """
    from app.domain.ports import LLMGenerator

    mock_gen = mocker.AsyncMock(spec=LLMGenerator)

    state = _make_state(
        documents=[{"text": "doc A", "type": "chunk"}, {"text": "doc B", "type": "chunk"}],
        top_k=1,
    )
    node = make_generate_node(mock_gen, None)
    result = await node(state)

    assert len(result["generation"]) == 1  # type: ignore[arg-type]
    assert result["generation"][0]["score"] == 1.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# _make_router
# ---------------------------------------------------------------------------


def test_router_routes_to_rewrite_when_no_docs() -> None:
    """Tests router returns 'rewrite' when documents is empty and budget remains."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[], rewrite_count=0)
    assert router(state) == "rewrite"


def test_router_routes_to_generate_when_docs_present() -> None:
    """Tests router returns 'generate' when relevant documents are available."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[{"text": "something", "type": "chunk"}], rewrite_count=0)
    assert router(state) == "generate"


def test_router_routes_to_generate_at_max_rewrites() -> None:
    """Tests router falls through to 'generate' when rewrite budget is exhausted."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[], rewrite_count=3)
    assert router(state) == "generate"


def test_router_routes_to_rewrite_just_below_limit() -> None:
    """Tests router still rewrites at rewrite_count = max_rewrites - 1."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[], rewrite_count=2)
    assert router(state) == "rewrite"
