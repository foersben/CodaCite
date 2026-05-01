"""Infrastructure implementation for Graph Extraction.

This module provides entity and relationship extraction implementations using
Gemini (via LangChain) and local fallbacks like GLiNER.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.domain.models import Edge, Node
from app.domain.ports import EntityExtractor

logger = logging.getLogger(__name__)


class ExtractedGraph(BaseModel):
    """Schema for LLM structured output extraction.

    Defines the Pydantic format for Gemini's structured response, containing
    lists of domain-layer Node and Edge objects.
    """

    nodes: list[Node] = Field(description="Extracted entities")
    edges: list[Edge] = Field(description="Extracted relationships")


class GeminiEntityExtractor(EntityExtractor):
    """Extractor using Google GenAI (Gemini) with structured output.

    Pipeline Role:
        Phase 5: Knowledge Extraction. Converts semantic text chunks into
        structured knowledge graph fragments (nodes and edges).

    Extraction Pipeline & Pydantic Rationale:
        This component implements a **Structured Extraction Pipeline**. Unlike
        open-ended generation, this pipeline uses **Pydantic** to define a
        rigorous schema (`ExtractedGraph`) that the LLM must adhere to.

        **Why Pydantic?**
        - **Validation**: Ensures the LLM returns valid JSON that maps perfectly
          to domain `Node` and `Edge` models.
        - **Type Safety**: Guarantees that IDs, labels, and weights are correctly
          typed before they reach the database.
        - **Pydantic vs. PydanticAI**: While PydanticAI is a framework for
          agentic control loops (LLM-as-a-service), CodaCite uses standard
          **Pydantic** for core data modeling and schema enforcement, ensuring
          zero-dependency domain logic.

    Implementation Details:
        - Uses 'langchain-google-genai' (ChatGoogleGenerativeAI).
        - Leverages `with_structured_output(ExtractedGraph)` to enforce the schema.
        - Defaults to 'gemini-pro'.
        - Sets temperature=0.0 for deterministic extraction.
    """

    llm: Any = None
    extractor: Any = None

    def __init__(self, api_key: str, model_name: str = "gemini-pro") -> None:
        """Initialize the extractor.

        Args:
            api_key: Google AI Studio API key.
            model_name: Gemini model identifier (e.g., 'gemini-pro').
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, google_api_key=api_key, temperature=0.0
            )
            self.extractor = self.llm.with_structured_output(ExtractedGraph)
        except Exception as e:
            logger.error("Failed to initialize Gemini extractor: %s", e)
            self.llm = None
            self.extractor = None

    async def extract(self, text: str) -> tuple[list[Node], list[Edge]]:
        """Extract nodes and edges from text using Gemini.

        Args:
            text: Input text chunk.

        Returns:
            A tuple of (extracted_nodes, extracted_edges).
        """
        if not self.extractor:
            return [], []

        try:
            result = await self.extractor.ainvoke(
                f"Extract the knowledge graph (entities and relationships) from the following text:\n\n{text}"
            )
            if isinstance(result, ExtractedGraph):
                return result.nodes, result.edges
            return [], []
        except Exception as e:
            logger.error("Gemini extraction failed: %s", e)
            return [], []


class GLiNERFallbackExtractor(EntityExtractor):
    """Fallback extractor using GLiNER for entities.

    Pipeline Role:
        Phase 5: Knowledge Extraction (Fallback). Provides a local, CPU-friendly
        alternative for entity extraction.

    Hybrid Extraction Theory:
        When API access is restricted, the system switches to a **Named Entity
        Recognition (NER)** approach using GLiNER. This represents a "coarse"
        extraction pipeline that focuses on identifying nodes without the
        semantic relationship modeling (edges) provided by the LLM pipeline.
    """

    def __init__(self) -> None:
        """Initialize the GLiNER model."""
        try:
            from gliner import GLiNER

            from app.config import settings

            self.model: Any = GLiNER.from_pretrained(
                "urchade/gliner_mediumv2.1", device=settings.device
            )
        except Exception:
            self.model = None

    async def extract(self, text: str) -> tuple[list[Node], list[Edge]]:
        """Extract nodes from text using GLiNER.

        Args:
            text: Input text chunk.

        Returns:
            A tuple containing extracted nodes and an empty list for edges.
        """
        if not self.model:
            return [], []

        labels = ["person", "organization", "location", "event", "concept"]
        predict_entities_func = getattr(self.model, "predict_entities", None)
        entities = []
        if predict_entities_func:
            entities = predict_entities_func(text, labels)

        nodes = []
        for ent in entities:
            nodes.append(
                Node(
                    id=ent["text"].lower().replace(" ", "_"),
                    label=ent["label"].upper(),
                    name=ent["text"],
                )
            )
        return nodes, []
