"""LLM-based entity and relationship extractor for knowledge graph construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Entity:
    """A knowledge graph node representing a named entity."""

    name: str
    entity_type: str
    description: str = ""


@dataclass
class Relationship:
    """A knowledge graph edge representing a relation between two entities."""

    source: str
    target: str
    relation: str


_ENTITY_LINE_RE = re.compile(r"^(.+?)\s*\|\s*(.+?)\s*\|\s*(.*)$")
_REL_LINE_RE = re.compile(r"^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$")


class EntityExtractor:
    """Extracts entities and relationships from text using an LLM.

    The LLM is expected to return a structured response in the format::

        ENTITIES
        <name> | <TYPE> | <description>
        ...
        RELATIONSHIPS
        <source> | <RELATION> | <target>
        ...

    Args:
        llm: A LangChain-compatible chat model (or any object with ``.invoke()``).
    """

    _PROMPT_TEMPLATE = (
        "Extract all named entities and relationships from the following text.\n\n"
        "Return your answer in EXACTLY this format (no extra text):\n\n"
        "ENTITIES\n"
        "<name> | <TYPE> | <description>\n\n"
        "RELATIONSHIPS\n"
        "<source_name> | <RELATION> | <target_name>\n\n"
        "Text:\n{text}"
    )

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def extract(self, text: str) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from *text*.

        Args:
            text: The document chunk to analyse.

        Returns:
            A tuple ``(entities, relationships)`` where each element is a list.
        """
        if not text or not text.strip():
            return [], []

        prompt = self._PROMPT_TEMPLATE.format(text=text)
        response = self._llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        return self._parse(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> tuple[list[Entity], list[Relationship]]:
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        section = None
        for line in raw.splitlines():
            line = line.strip()
            if line.upper() == "ENTITIES":
                section = "entities"
                continue
            if line.upper() == "RELATIONSHIPS":
                section = "relationships"
                continue
            if not line:
                continue

            if section == "entities":
                m = _ENTITY_LINE_RE.match(line)
                if m:
                    entities.append(
                        Entity(
                            name=m.group(1).strip(),
                            entity_type=m.group(2).strip(),
                            description=m.group(3).strip(),
                        )
                    )
            elif section == "relationships":
                m = _REL_LINE_RE.match(line)
                if m:
                    relationships.append(
                        Relationship(
                            source=m.group(1).strip(),
                            relation=m.group(2).strip(),
                            target=m.group(3).strip(),
                        )
                    )

        return entities, relationships
