"""Infrastructure implementation for Graph Extraction."""

from pydantic import BaseModel, Field

from app.domain.models import Edge, Node
from app.domain.ports import EntityExtractor


class ExtractedGraph(BaseModel):
    """Schema for LLM structured output extraction."""

    nodes: list[Node] = Field(description="List of entities extracted from the text")
    edges: list[Edge] = Field(description="List of relationships extracted from the text")


class GeminiEntityExtractor(EntityExtractor):
    """Extractor using Google GenAI (Gemini) with structured output."""

    def __init__(self, api_key: str, model_name: str = "gemini-pro") -> None:
        """Initialize the extractor."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm: object | None = ChatGoogleGenerativeAI(
                model=model_name, google_api_key=api_key, temperature=0.0
            )
            with_structured_output_func = getattr(self.llm, "with_structured_output", None)
            self.extractor: object | None = None
            if with_structured_output_func:
                self.extractor = with_structured_output_func(ExtractedGraph)
        except Exception:
            self.llm = None
            self.extractor = None

    async def extract(self, text: str) -> tuple[list[Node], list[Edge]]:
        """Extract nodes and edges from text."""
        if not self.extractor:
            return [], []
        try:
            ainvoke_func = getattr(self.extractor, "ainvoke", None)
            if ainvoke_func:
                # Type system cannot infer dynamic method signature
                result = await ainvoke_func(
                    f"Extract the knowledge graph (entities and relationships) from the following text:\n\n{text}"
                )
                nodes = getattr(result, "nodes", [])
                edges = getattr(result, "edges", [])
                return nodes, edges
            return [], []
        except Exception as e:
            print(f"Extraction failed: {e}")
            return [], []


class GLiNERFallbackExtractor(EntityExtractor):
    """Fallback extractor using GLiNER for entities."""

    def __init__(self) -> None:
        """Initialize the extractor."""
        try:
            from gliner import GLiNER

            self.model: object | None = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
        except Exception:
            self.model = None

    async def extract(self, text: str) -> tuple[list[Node], list[Edge]]:
        """Extract nodes and edges from text."""
        if not self.model:
            return [], []
        # Fallback to local GLiNER for entities
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
        # Note: GLiNER does not extract relations out of the box, return empty edges
        return nodes, []
