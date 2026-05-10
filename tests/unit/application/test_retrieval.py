"""Unit tests for the GraphRAGRetrievalUseCase.

This module validates the self-correcting LangGraph retrieval orchestration,
ensuring that the graph correctly coordinates hybrid search, document grading,
query rewriting, and context assembly.
"""

from typing import Any

import pytest

from app.application.retrieval import GraphRAGRetrievalUseCase
from app.domain.models import Chunk, Edge, Node


@pytest.fixture
def use_case(
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
    mock_llm_generator: Any,
) -> GraphRAGRetrievalUseCase:
    """Initialize the GraphRAGRetrievalUseCase with mocked dependencies.

    Args:
        mock_document_store: Mock document store fixture.
        mock_graph_store: Mock graph store fixture.
        mock_embedder: Mock embedder fixture.
        mock_entity_linker: Mock entity linker fixture.
        mock_reranker: Mock reranker fixture.
        mock_llm_generator: Mock LLM generator fixture for grading/rewriting.

    Returns:
        An instance of GraphRAGRetrievalUseCase.
    """
    return GraphRAGRetrievalUseCase(
        document_store=mock_document_store,
        graph_store=mock_graph_store,
        embedder=mock_embedder,
        entity_linker=mock_entity_linker,
        reranker=mock_reranker,
        generator=mock_llm_generator,
    )


@pytest.mark.asyncio
async def test_retrieval_happy_path(
    use_case: GraphRAGRetrievalUseCase,
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
    mock_llm_generator: Any,
) -> None:
    """Tests the full happy-path: retrieved docs are graded relevant and returned.

    Given:
        A query, chunks that are graded relevant, and no graph entities.
    When:
        execute() is called.
    Then:
        The pipeline runs retrieve → grade(pass) → generate.
        search_chunks and the reranker are each called once.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", text="Neural networks are relevant.", document_id="doc1", index=0)
    ]
    mock_graph_store.get_all_nodes.return_value = []
    mock_entity_linker.link_entities.return_value = []
    # Grader says "yes" → chunk is kept
    mock_llm_generator.agenerate.return_value = "yes"
    mock_reranker.rerank.return_value = [{"text": "Neural networks are relevant.", "score": 0.9}]

    # Act
    results = await use_case.execute("What are neural networks?", top_k=3)

    # Assert
    assert len(results) == 1
    assert results[0]["text"] == "Neural networks are relevant."
    mock_document_store.search_chunks.assert_called_once()
    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_rewrite_then_generate(
    use_case: GraphRAGRetrievalUseCase,
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
    mock_llm_generator: Any,
) -> None:
    """Tests that a failed grade triggers a rewrite and second retrieval pass.

    Given:
        First retrieval returns a chunk graded irrelevant.
        Second retrieval (after rewrite) returns a chunk graded relevant.
    When:
        execute() is called.
    Then:
        search_chunks is called twice; the second pass result is returned.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1]
    mock_graph_store.get_all_nodes.return_value = []
    mock_entity_linker.link_entities.return_value = []

    # First retrieval: irrelevant chunk; second retrieval: relevant chunk
    mock_document_store.search_chunks.side_effect = [
        [Chunk(id="c1", text="Unrelated content.", document_id="d1", index=0)],
        [Chunk(id="c2", text="Directly relevant answer.", document_id="d1", index=1)],
    ]

    # Call sequence: grade(c1)="no", rewrite="rephrased question", grade(c2)="yes"
    mock_llm_generator.agenerate.side_effect = [
        "no",  # grade: first chunk irrelevant
        "rephrased question",  # rewrite node
        "yes",  # grade: second chunk relevant
    ]
    mock_reranker.rerank.return_value = [{"text": "Directly relevant answer.", "score": 0.95}]

    # Act
    results = await use_case.execute("original question", top_k=2)

    # Assert
    assert len(results) == 1
    assert results[0]["text"] == "Directly relevant answer."
    assert mock_document_store.search_chunks.call_count == 2


@pytest.mark.asyncio
async def test_retrieval_max_rewrites_safety_valve(
    use_case: GraphRAGRetrievalUseCase,
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
    mock_llm_generator: Any,
) -> None:
    """Tests that the pipeline terminates after max_rewrites cycles.

    Given:
        All retrieved chunks are consistently graded irrelevant across all
        retrieval attempts (default max_rewrites=3, so 4 retrieve calls max).
    When:
        execute() is called.
    Then:
        The graph terminates (does not loop forever) and returns a list.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1]
    mock_graph_store.get_all_nodes.return_value = []
    mock_entity_linker.link_entities.return_value = []
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", text="Always irrelevant.", document_id="d1", index=0)
    ]
    # Alternate: grade="no" then rewrite="new q" repeated for 3 cycles
    # Pattern: (grade no, rewrite) x3 then grade no → generate with empty docs
    mock_llm_generator.agenerate.side_effect = [
        "no",
        "rewrite 1",  # cycle 1: grade → no, rewrite
        "no",
        "rewrite 2",  # cycle 2: grade → no, rewrite
        "no",
        "rewrite 3",  # cycle 3: grade → no, rewrite
        "no",  # cycle 4 (budget exhausted): grade → no, fall to generate
    ]
    mock_reranker.rerank.return_value = []

    # Act — should terminate, not loop forever
    results = await use_case.execute("hopeless query", top_k=2)

    # Assert: pipeline terminates and returns a list
    assert isinstance(results, list)
    # 4 retrieve passes: initial + 3 rewrites
    assert mock_document_store.search_chunks.call_count == 4


@pytest.mark.asyncio
async def test_retrieval_with_graph_context(
    use_case: GraphRAGRetrievalUseCase,
    mock_document_store: Any,
    mock_graph_store: Any,
    mock_embedder: Any,
    mock_entity_linker: Any,
    mock_reranker: Any,
    mock_llm_generator: Any,
) -> None:
    """Tests that graph traversal results are included in the document pool.

    Given:
        A query where entity linking and graph traversal return nodes/edges.
    When:
        execute() is called.
    Then:
        The grader sees both chunk and entity/relation snippets.
    """
    # Arrange
    mock_embedder.embed.return_value = [0.1]
    mock_document_store.search_chunks.return_value = [
        Chunk(id="c1", text="Chunk text.", document_id="d1", index=0)
    ]
    mock_graph_store.get_all_nodes.return_value = [
        Node(id="n1", name="Entity A", label="Concept", description="desc A")
    ]
    mock_entity_linker.link_entities.return_value = [
        Node(id="n1", name="Entity A", label="Concept")
    ]
    mock_graph_store.traverse.return_value = (
        [Node(id="n1", name="Entity A", label="Concept", description="desc A")],
        [Edge(source_id="n1", target_id="n2", relation="related_to")],
    )
    # All docs graded relevant
    mock_llm_generator.agenerate.return_value = "yes"
    mock_reranker.rerank.side_effect = lambda q, ctx, top_k: [
        {"text": c, "score": 1.0} for c in ctx[:top_k]
    ]

    # Act
    results = await use_case.execute("Entity A query", top_k=10)

    # Assert — chunk + entity + relation all graded
    assert len(results) >= 1
    mock_graph_store.traverse.assert_called_once()
