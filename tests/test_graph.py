"""
Tests for the graph module.

Covers:
- EntityExtractor: LLM-based extraction of entities (nodes) and relationships (edges)
- GraphStore: SurrealDB-backed storage of nodes, edges, and chunk embeddings
  (All LLM/DB calls are mocked)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph.extractor import Entity, EntityExtractor, Relationship
from app.graph.store import GraphStore

# ---------------------------------------------------------------------------
# EntityExtractor tests
# ---------------------------------------------------------------------------


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    def test_entity_dataclass(self) -> None:
        """Entity dataclass should expose name, type, and description fields."""
        entity = Entity(name="Alice", entity_type="PERSON", description="A developer")
        assert entity.name == "Alice"
        assert entity.entity_type == "PERSON"

    def test_relationship_dataclass(self) -> None:
        """Relationship dataclass should expose source, target, and relation fields."""
        rel = Relationship(source="Alice", target="BobCorp", relation="WORKS_FOR")
        assert rel.source == "Alice"
        assert rel.target == "BobCorp"
        assert rel.relation == "WORKS_FOR"

    def test_extract_returns_entities_and_relationships(self) -> None:
        """extract() should return a tuple of (entities, relationships) lists."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=(
                "ENTITIES\nAlice | PERSON | A software developer\n"
                "BobCorp | ORG | A technology company\n"
                "RELATIONSHIPS\nAlice | WORKS_FOR | BobCorp"
            )
        )

        extractor = EntityExtractor(llm=mock_llm)
        entities, relationships = extractor.extract("Alice works for BobCorp.")

        assert len(entities) >= 1
        assert len(relationships) >= 1
        assert any(e.name == "Alice" for e in entities)

    def test_extract_empty_text_returns_empty(self) -> None:
        """extract() on empty text should return empty lists."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ENTITIES\nRELATIONSHIPS\n")

        extractor = EntityExtractor(llm=mock_llm)
        entities, relationships = extractor.extract("")

        assert entities == []
        assert relationships == []

    def test_extract_parses_entities_correctly(self) -> None:
        """extract() should parse entity name, type, and description from LLM output."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=(
                "ENTITIES\nOpenAI | ORG | AI research company\n"
                "GPT-4 | PRODUCT | Large language model\n"
                "RELATIONSHIPS\nOpenAI | CREATED | GPT-4"
            )
        )

        extractor = EntityExtractor(llm=mock_llm)
        entities, relationships = extractor.extract("OpenAI created GPT-4.")

        names = [e.name for e in entities]
        assert "OpenAI" in names
        assert "GPT-4" in names

    def test_extract_parses_relationships_correctly(self) -> None:
        """extract() should parse source, relation, and target from LLM output."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=(
                "ENTITIES\nOpenAI | ORG | AI company\n"
                "GPT-4 | PRODUCT | LLM\n"
                "RELATIONSHIPS\nOpenAI | CREATED | GPT-4"
            )
        )

        extractor = EntityExtractor(llm=mock_llm)
        _, relationships = extractor.extract("OpenAI created GPT-4.")

        assert len(relationships) == 1
        assert relationships[0].source == "OpenAI"
        assert relationships[0].relation == "CREATED"
        assert relationships[0].target == "GPT-4"


# ---------------------------------------------------------------------------
# GraphStore tests
# ---------------------------------------------------------------------------


class TestGraphStore:
    """Tests for GraphStore (SurrealDB-backed, using async mock)."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        db = MagicMock()
        db.query = AsyncMock(return_value=[{}])
        return db

    @pytest.mark.asyncio
    async def test_store_chunk_calls_db_query(self, mock_db: MagicMock) -> None:
        """store_chunk() should execute a SurrealDB query."""
        store = GraphStore(db=mock_db)
        await store.store_chunk(
            chunk_id="chunk:1",
            text="some text",
            embedding=[0.1, 0.2, 0.3],
            source="doc.pdf",
        )
        mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_entity_calls_db_query(self, mock_db: MagicMock) -> None:
        """store_entity() should execute a SurrealDB query."""
        store = GraphStore(db=mock_db)
        entity = Entity(name="Alice", entity_type="PERSON", description="A developer")
        await store.store_entity(entity=entity)
        mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_relationship_calls_db_query(self, mock_db: MagicMock) -> None:
        """store_relationship() should execute a SurrealDB query."""
        store = GraphStore(db=mock_db)
        rel = Relationship(source="Alice", target="BobCorp", relation="WORKS_FOR")
        await store.store_relationship(relationship=rel)
        mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_search_returns_list(self, mock_db: MagicMock) -> None:
        """vector_search() should return a list of result dicts."""
        mock_db.query = AsyncMock(
            return_value=[
                [{"id": "chunk:1", "text": "hello", "score": 0.95}]
            ]
        )
        store = GraphStore(db=mock_db)
        results = await store.vector_search(query_embedding=[0.1, 0.2, 0.3], top_k=5)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_traverse_graph_returns_list(self, mock_db: MagicMock) -> None:
        """traverse_graph() should return a list of connected node dicts."""
        mock_db.query = AsyncMock(
            return_value=[
                [{"id": "entity:Alice", "name": "Alice"}]
            ]
        )
        store = GraphStore(db=mock_db)
        results = await store.traverse_graph(start_node_id="chunk:1", depth=2)

        assert isinstance(results, list)
