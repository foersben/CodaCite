from unittest.mock import AsyncMock

import pytest

from app.models.models import Chunk
from app.pipelines.generation.rag_graph import make_retrieve_node


@pytest.mark.asyncio
async def test_window_merging_logic():
    """Verify that retrieval results are correctly merged into context windows."""
    # Arrange
    store = AsyncMock()
    embedder = AsyncMock()
    graph_store = AsyncMock()
    entity_linker = AsyncMock()

    # Mock search_chunks to return one hit (the middle chunk)
    hit = Chunk(
        id="doc1_1",
        document_id="doc1",
        index=1,
        text="Document: test.txt\nChunk 1 content",
        start_char=100,
        end_char=200,
    )
    store.search_chunks.return_value = [hit]

    # Mock neighbors
    prev_chunk = Chunk(
        id="doc1_0",
        document_id="doc1",
        index=0,
        text="Document: test.txt\nChunk 0 content",
        start_char=0,
        end_char=100,
    )
    next_chunk = Chunk(
        id="doc1_2",
        document_id="doc1",
        index=2,
        text="Document: test.txt\nChunk 2 content",
        start_char=200,
        end_char=300,
    )
    # Note: retrieval logic calls get_chunks_by_ids with list of neighbor IDs
    store.get_chunks_by_ids.return_value = [prev_chunk, next_chunk]

    # Graph store and entity linker return empty lists
    graph_store.get_all_nodes.return_value = []
    graph_store.traverse.return_value = ([], [])
    entity_linker.link_entities.return_value = []

    # Build the node
    node = make_retrieve_node(store, embedder, graph_store, entity_linker)

    state = {"question": "test question", "top_k": 1, "notebook_ids": None}

    # Act
    result = await node(state)

    # Assert
    docs = result["documents"]
    assert len(docs) == 1
    merged_text = docs[0]["text"]

    # Expected: "Document: test.txt\nChunk 0 content\n[...]\nChunk 1 content\n[...]\nChunk 2 content"
    # Prefix is only on the first chunk in chronological order
    expected = "Document: test.txt\nChunk 0 content\n[...]\nChunk 1 content\n[...]\nChunk 2 content"
    assert merged_text == expected

    # Verify the store was called with correct neighbor IDs
    # prev_id = doc1_0, next_id = doc1_2
    called_args = store.get_chunks_by_ids.call_args[0][0]
    assert "doc1_0" in called_args
    assert "doc1_2" in called_args
    assert len(called_args) == 2
