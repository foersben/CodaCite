"""Infrastructure implementation for Entity Resolution."""

import jellyfish

from app.domain.models import Node
from app.domain.ports import Embedder, EntityResolver


class JaroWinklerResolver(EntityResolver):
    """Entity resolution using Jaro-Winkler string similarity and Embeddings."""

    def __init__(self, embedder: Embedder, threshold: float = 0.85) -> None:
        """Initialize the resolver."""
        self.embedder = embedder
        self.threshold = threshold

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity."""
        return float(jellyfish.jaro_winkler_similarity(s1.lower(), s2.lower()))

    async def resolve_entities(
        self, new_nodes: list[Node], existing_nodes: list[Node]
    ) -> list[Node]:
        """Merge new nodes into existing nodes if they are highly similar."""
        resolved = []
        for new_node in new_nodes:
            best_match = None
            best_score = 0.0

            for existing_node in existing_nodes:
                score = self._string_similarity(new_node.name, existing_node.name)
                if score > best_score:
                    best_score = score
                    best_match = existing_node

            if best_match and best_score >= self.threshold:
                # Merge: we keep the existing node ID but might aggregate properties
                # For simplicity, we yield the existing node with new sources appended
                merged_node = Node(
                    id=best_match.id,
                    label=best_match.label,  # Or resolve conflicting labels
                    name=best_match.name,
                    description=best_match.description or new_node.description,
                    source_chunk_ids=list(
                        set(best_match.source_chunk_ids + new_node.source_chunk_ids)
                    ),
                )
                resolved.append(merged_node)
            else:
                resolved.append(new_node)

        return resolved
