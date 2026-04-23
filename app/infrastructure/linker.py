"""Infrastructure implementation for Entity Linking.

This module provides logic for mapping query terms back to existing nodes
in the knowledge graph using exact and fuzzy matching.
"""

from typing import Any

from app.domain.models import Node


class SimpleEntityLinker:
    """A simple entity linker using string matching.

    Maps unstructured query terms to extracted entities within the graph
    to provide context for retrieval-augmented generation.
    """

    def __init__(self, gliner_extractor: Any = None) -> None:
        """Initialize linker.

        Args:
            gliner_extractor: Optional extractor instance for query parsing.
        """
        self.extractor = gliner_extractor

    async def link_entities(self, query: str, existing_nodes: list[Node]) -> list[Node]:
        """Link entities in the query to existing graph nodes.

        Args:
            query: The user's query string.
            existing_nodes: List of all nodes currently in the graph.

        Returns:
            A list of nodes mentioned or relevant to the query.
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
