"""Infrastructure implementation for Entity Linking."""

from app.domain.models import Node


class SimpleEntityLinker:
    """A simple entity linker using string matching."""

    def __init__(self, gliner_extractor: object | None = None) -> None:
        """Initialize linker."""
        self.extractor = gliner_extractor

    async def link_entities(self, query: str, existing_nodes: list[Node]) -> list[Node]:
        """Link entities in the query to existing graph nodes."""
        linked_nodes = []

        query_entities = []

        # Simple extraction for now
        if self.extractor:
            extract_func = getattr(self.extractor, "extract", None)
            if extract_func:
                nodes, _ = await extract_func(query)
                query_entities = [n.name.lower() for n in nodes]

        if not query_entities:
            # Very naive fallback
            query_entities = [word.lower() for word in query.split() if len(word) > 4]

        for node in existing_nodes:
            if node.name.lower() in query_entities or node.name.lower() in query.lower():
                linked_nodes.append(node)

        return linked_nodes
