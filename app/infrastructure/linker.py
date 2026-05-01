"""Infrastructure implementation for Entity Linking.

This module provides logic for mapping query terms back to existing nodes
in the knowledge graph using exact and fuzzy matching.
"""

from typing import Any

from app.domain.models import Node
from app.domain.ports import EntityLinker


class SimpleEntityLinker(EntityLinker):
    """A simple entity linker using string matching.

    Maps unstructured query terms to existing Knowledge Graph nodes to provide
    the initial 'seed' entities for graph traversal.

    Pipeline Role:
        Initial stage of Retrieval. Converts user intent into graph coordinates
        by identifying entities mentioned in the query.

    Implementation Details:
        - Uses a provided `EntityExtractor` (typically GLiNER) to identify
          potential entities in the query.
        - Performs case-insensitive matching against Node names and IDs.
        - Provides a naive word-based fallback if no entities are extracted.
    """

    def __init__(self, gliner_extractor: Any = None) -> None:
        """Initialize the linker.

        Args:
            gliner_extractor: Optional extractor instance for query parsing.
        """
        self.extractor = gliner_extractor

    async def link_entities(self, query: str, existing_nodes: list[Node]) -> list[Node]:
        """Link entities in the user query to existing graph nodes.

        Args:
            query: The user's unstructured search query.
            existing_nodes: The full list of entity nodes available in the graph.

        Returns:
            A list of matched Node objects to be used as traversal seeds.
        """
        linked_nodes = []
        query_entities = []

        # Use extractor to find entities in the query itself
        if self.extractor:
            extract_func = getattr(self.extractor, "extract", None)
            if extract_func:
                nodes, _ = await extract_func(query)
                query_entities = [n.name.lower() for n in nodes]

        if not query_entities:
            # Naive fallback: split query into words
            query_entities = [word.lower() for word in query.split() if len(word) > 4]

        for node in existing_nodes:
            if node.name.lower() in query_entities or node.name.lower() in query.lower():
                linked_nodes.append(node)

        return linked_nodes
