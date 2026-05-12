"""Unit tests for the LangGraph RAG graph nodes and routing logic.

Each node factory is tested in isolation using AsyncMock dependencies,
following the Arrange-Act-Assert pattern.
"""

from typing import Any, cast

import pytest

from app.models.models import Chunk, Edge, Node
from app.pipelines.generation.rag_graph import (
    RAGState,
    _make_router,
    make_rerank_node,
    make_retrieve_node,
    make_rewrite_query_node,
)


def _make_state(**overrides: Any) -> RAGState:
    """Build a minimal RAGState with sensible defaults.

    Args:
        **overrides: Fields to override on the default state.

    Returns:
        A fully populated RAGState dict.
    """
    base: RAGState = {
        "question": "What is machine learning?",
        "history": None,
        "documents": [],
        "generation": [],
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
    from app.core.interfaces import DocumentStore, Embedder, GraphStore

    mock_store = mocker.AsyncMock(spec=DocumentStore)
    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_graph_store = mocker.AsyncMock(spec=GraphStore)
    mock_linker = mocker.AsyncMock()

    mock_embedder.embed.return_value = [0.1, 0.2]
    mock_store.search_chunks.return_value = [
        Chunk(id="c1", text="Relevant text.", document_id="d1", index=0, start_char=0, end_char=14)
    ]
    mock_store.get_chunks_by_ids.return_value = []
    mock_graph_store.search_nodes.return_value = []
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
    from app.core.interfaces import DocumentStore, Embedder, GraphStore

    mock_store = mocker.AsyncMock(spec=DocumentStore)
    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_graph_store = mocker.AsyncMock(spec=GraphStore)
    mock_linker = mocker.AsyncMock()

    mock_embedder.embed.return_value = [0.1]
    mock_store.search_chunks.return_value = [
        Chunk(id="c1", text="chunk text", document_id="d1", index=0, start_char=0, end_char=10)
    ]
    mock_store.get_chunks_by_ids.return_value = []
    mock_graph_store.search_nodes.return_value = [Node(id="n1", name="A", label="T")]
    mock_linker.link_entities.return_value = [Node(id="n1", name="A", label="T")]
    mock_graph_store.traverse.return_value = (
        [Node(id="n1", name="A", label="T", description="desc")],
        [Edge(id="rel1", source_id="n1", target_id="n2", relation="relates_to")],
    )

    node = make_retrieve_node(mock_store, mock_embedder, mock_graph_store, mock_linker)
    result = await node(_make_state())

    docs = cast(list[dict[str, object]], result["documents"])
    types = {str(d["type"]) for d in docs}
    assert "chunk" in types
    assert "entity" in types
    assert "relation" in types
    relation_doc = next(doc for doc in docs if doc["type"] == "relation")
    assert relation_doc["id"] == "rel1"


@pytest.mark.asyncio
async def test_retrieve_node_keeps_non_prefixed_neighbor_content(mocker: Any) -> None:
    """Tests that neighbor merging preserves chunks without an injected document prefix."""
    from app.core.interfaces import DocumentStore, Embedder, GraphStore

    mock_store = mocker.AsyncMock(spec=DocumentStore)
    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_graph_store = mocker.AsyncMock(spec=GraphStore)
    mock_linker = mocker.AsyncMock()

    mock_embedder.embed.return_value = [0.1]
    mock_store.search_chunks.return_value = [
        Chunk(id="c2", text="Current chunk", document_id="d1", index=1, start_char=10, end_char=23)
    ]
    mock_store.get_chunks_by_ids.return_value = [
        Chunk(id="c1", text="\nLeading newline content", document_id="d1", index=0, start_char=0, end_char=23),
        Chunk(id="c3", text="Legacy chunk without prefix", document_id="d1", index=2, start_char=24, end_char=51),
    ]
    mock_graph_store.search_nodes.return_value = []
    mock_linker.link_entities.return_value = []

    node = make_retrieve_node(mock_store, mock_embedder, mock_graph_store, mock_linker)
    result = await node(_make_state())

    docs = cast(list[dict[str, object]], result["documents"])
    assert (
        docs[0]["text"]
        == "\nLeading newline content\n[...]\nCurrent chunk\n[...]\nLegacy chunk without prefix"
    )


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
    from app.core.interfaces import DocumentStore, Embedder, GraphStore

    mock_store = mocker.AsyncMock(spec=DocumentStore)
    mock_embedder = mocker.AsyncMock(spec=Embedder)
    mock_graph_store = mocker.AsyncMock(spec=GraphStore)
    mock_linker = mocker.AsyncMock()

    mock_embedder.embed.return_value = [0.1]
    mock_store.search_chunks.return_value = [
        Chunk(id="c1", text="same text", document_id="d1", index=0, start_char=0, end_char=9),
        Chunk(id="c2", text="same text", document_id="d1", index=1, start_char=10, end_char=19),
    ]
    mock_store.get_chunks_by_ids.return_value = []
    mock_graph_store.search_nodes.return_value = []
    mock_linker.link_entities.return_value = []

    node = make_retrieve_node(mock_store, mock_embedder, mock_graph_store, mock_linker)
    result = await node(_make_state())

    docs = cast(list[dict[str, object]], result["documents"])
    assert len(docs) == 1


# ---------------------------------------------------------------------------
# rerank_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_node_filters_and_scores(mocker: Any) -> None:
    """Tests that rerank_node filters out low-scoring documents.

    Given:
        Two documents and a reranker returning scores 0.9 and 0.1.
    When:
        rerank_node is invoked.
    Then:
        Only the 0.9 document is kept.
    """
    from app.core.interfaces import Reranker

    mock_reranker = mocker.AsyncMock(spec=Reranker)
    mock_reranker.rerank.return_value = [
        {"text": "Relevant doc", "score": 0.9},
        {"text": "Unrelated doc", "score": 0.1},
    ]

    state = _make_state(
        documents=[
            {"text": "Relevant doc", "type": "chunk"},
            {"text": "Unrelated doc", "type": "chunk"},
        ]
    )

    node = make_rerank_node(mock_reranker)
    result = await node(state)

    docs = result["documents"]
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0]["text"] == "Relevant doc"
    assert docs[0]["score"] == 0.9


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
    from app.core.interfaces import LLMGenerator

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
    from app.core.interfaces import LLMGenerator

    mock_gen = mocker.AsyncMock(spec=LLMGenerator)
    mock_gen.agenerate.return_value = "   "

    state = _make_state(question="original?", rewrite_count=1)
    node = make_rewrite_query_node(mock_gen)
    result = await node(state)

    assert result["question"] == "original?"
    assert result["rewrite_count"] == 2


# ---------------------------------------------------------------------------
# _make_router
# ---------------------------------------------------------------------------


def test_router_routes_to_rewrite_when_no_docs() -> None:
    """Tests router returns 'rewrite' when documents is empty and budget remains."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[], rewrite_count=0)
    assert router(state) == "rewrite"


def test_router_routes_to_end_when_docs_present() -> None:
    """Tests router returns '__end__' when relevant documents are available."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[{"text": "something", "type": "chunk"}], rewrite_count=0)
    assert router(state) == "__end__"


def test_router_routes_to_end_at_max_rewrites() -> None:
    """Tests router falls through to '__end__' when rewrite budget is exhausted."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[], rewrite_count=3)
    assert router(state) == "__end__"


def test_router_routes_to_rewrite_just_below_limit() -> None:
    """Tests router still rewrites at rewrite_count = max_rewrites - 1."""
    router = _make_router(max_rewrites=3)
    state = _make_state(documents=[], rewrite_count=2)
    assert router(state) == "rewrite"
