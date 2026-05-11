"""Relationship mapping implementation using LLMs.

This module provides the Stage 2 of the two-stage KG extraction pipeline,
where an LLM identifies logical connections between pre-spotted entities.
"""

import json
import logging

from app.core.interfaces import LLMGenerator
from app.models.models import Edge

logger = logging.getLogger(__name__)


class LLMRelator:
    """Stage 2 KG Extractor: Logical Relationship Mapping.

    This component takes a text chunk and a list of verified entities (from Stage 1)
    and uses a high-reasoning LLM to identify the predicates (edges) connecting them.
    """

    def __init__(self, llm: LLMGenerator) -> None:
        """Initialize the relator.

        Args:
            llm: The LLM generator implementation (e.g., DeepSeek).
        """
        self.llm = llm

    async def extract_relationships(
        self, chunk_text: str, spotted_entities: list[str]
    ) -> list[Edge]:
        """Extract logical connections between verified entities.

        Args:
            chunk_text: The raw text context.
            spotted_entities: List of canonical entity names found in the text.

        Returns:
            A list of Edge objects representing the identified relationships.
        """
        if not spotted_entities or len(spotted_entities) < 2:
            return []

        entities_str = ", ".join(spotted_entities)
        prompt = f"""Given the following text and this predefined list of verified entities, extract the logical connections between them.

VERIFIED ENTITIES:
{entities_str}

TEXT CHUNK:
{chunk_text}

INSTRUCTIONS:
1. Extract relationships ONLY between the entities provided in the VERIFIED ENTITIES list.
2. For each relationship, identify the Subject, Predicate (Relationship Type), and Object.
3. Output the results as a JSON list of objects with the following keys: "source", "target", "relation", "description".
4. The "relation" should be a short, uppercase string (e.g., "WORKS_FOR", "LOCATED_IN").
5. Output ONLY the JSON. No preamble or explanation.

Example Output:
[
  {{"source": "Alice", "target": "Acme Corp", "relation": "WORKS_FOR", "description": "Alice is a lead engineer at Acme Corp."}}
]
"""
        try:
            response = await self.llm.agenerate(prompt)
            # Locate the JSON array in the response (it might have preamble or be an error msg)
            start_idx = response.find("[")
            end_idx = response.rfind("]")

            if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
                logger.warning(
                    "[LLMRelator] No valid JSON array found in response. Raw response: %r",
                    response[:100] + "..." if len(response) > 100 else response,
                )
                return []

            clean_response = response[start_idx : end_idx + 1]
            data = json.loads(clean_response)
            edges = []
            # Build a set of canonical entity names in lowercase for validation
            spotted_lower = {e.lower() for e in spotted_entities}
            for item in data:
                # Ensure source and target are non-None strings
                source = item.get("source")
                target = item.get("target")
                if not isinstance(source, str) or not isinstance(target, str):
                    logger.warning(
                        "[LLMRelator] Skipping edge with invalid source/target: %r -> %r",
                        source,
                        target,
                    )
                    continue

                # Enforce that both endpoints refer to one of the verified entities
                if source.lower() not in spotted_lower or target.lower() not in spotted_lower:
                    logger.warning(
                        "[LLMRelator] Skipping edge with unrecognized entities: %r -> %r",
                        source,
                        target,
                    )
                    continue

                # Normalize IDs to lowercase with underscores to match Node ID convention
                edges.append(
                    Edge(
                        source_id=source.lower().replace(" ", "_"),
                        target_id=target.lower().replace(" ", "_"),
                        relation=item.get("relation", "RELATED_TO").upper(),
                        description=item.get("description"),
                    )
                )
            return edges
        except Exception as e:
            logger.error("[LLMRelator] Relationship extraction failed: %s", e)
            return []
