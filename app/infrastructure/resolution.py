"""Infrastructure implementation for Entity Resolution.

This module provides implementations for resolving and merging entity nodes
that represent the same real-world entity, typically due to minor variations
in naming or extraction noise.
"""

import jellyfish

from app.domain.models import Node
from app.domain.ports import EntityResolver


class JaroWinklerResolver(EntityResolver):
    """Entity resolution using Jaro-Winkler string similarity.

    Identifies and merges synonymous entities (e.g., 'Google' and 'Google Inc.')
    to maintain graph connectivity and reduce redundancy.

    Pipeline Role:
        Phase 6 of Ingestion. Occurs after extraction but before graph persistence
        to ensure new entities are mapped to existing canonical nodes.

    Implementation Details:
        - Uses the 'jellyfish' library for Jaro-Winkler distance.
        - Defaults to a similarity threshold of 0.85.
        - Merges metadata (descriptions, source IDs) while preserving the ID
          of the existing canonical node.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        """Initialize the resolver.

        Args:
            threshold: Similarity score threshold [0, 1] for merging nodes.
                Defaults to 0.85.
        """
        self.threshold = threshold

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity between two strings.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Similarity score where 1.0 is an exact match.
        """
        return float(jellyfish.jaro_winkler_similarity(s1.lower(), s2.lower()))

    async def resolve_entities(
        self, new_nodes: list[Node], existing_nodes: list[Node]
    ) -> list[Node]:
        """Merge new nodes into existing nodes if they are highly similar.

        Args:
            new_nodes: List of freshly extracted nodes.
            existing_nodes: List of nodes already present in the knowledge graph.

        Returns:
            A list containing either the original new nodes or merged versions.
        """
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
                # Merge: we keep the existing node ID but aggregate properties
                # For simplicity, we yield the existing node with new sources appended
                merged_node = Node(
                    id=best_match.id,
                    label=best_match.label,
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
