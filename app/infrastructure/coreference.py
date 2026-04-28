"""Infrastructure implementation of CoreferenceResolver using fastcoref.

This module provides an implementation of the CoreferenceResolver port using
the fastcoref library for efficient, local coreference resolution.
"""

import asyncio
from typing import Any

from app.domain.ports import CoreferenceResolver


class FastCorefResolver(CoreferenceResolver):
    """Coreference resolution using the fastcoref library.

    This resolver replaces pronouns and other referring expressions (e.g., 'he',
    'it', 'the company') with their primary mention (antecedent) within a
    document chunk.

    Pipeline Role:
        Phase 1 of Ingestion. Pre-processing text before chunking and extraction
        to ensure that entity extraction (Phase 5) captures the correct context
        for every mention.

    Implementation Details:
        - Uses the 'biu-nlp/f-coref' model by default.
        - Offloads CPU-bound prediction to a thread pool via `asyncio.to_thread`.
        - Performs in-place text replacement from end-to-start to maintain offsets.
    """

    def __init__(self, model_name_or_path: str = "biu-nlp/f-coref", nlp: Any = None) -> None:
        """Initialize the fastcoref model.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
                Defaults to "biu-nlp/f-coref" (F-Coref).
            nlp: Optional pre-loaded spaCy model. If None, a blank 'en' model is used.
        """
        import spacy
        from fastcoref import FCoref
        from fastcoref.coref_models.modeling_fcoref import FCorefModel

        from app.config import settings

        # Compatibility fix for transformers 5.x
        if not hasattr(FCorefModel, "all_tied_weights_keys"):
            FCorefModel.all_tied_weights_keys = property(lambda self: {})

        # Use blank model if no model provided to avoid downloading en_core_web_sm
        if nlp is None:
            nlp = spacy.blank("en")

        self.model = FCoref(model_name_or_path=model_name_or_path, device=settings.device, nlp=nlp)

    def _resolve_sync(self, text: str) -> str:
        """Synchronous coreference resolution logic.

        Processes text to identify clusters of mentions and replaces non-primary
        mentions with the cluster head.

        Args:
            text: Raw input text.

        Returns:
            Resolved text with coreferences replaced by their primary mentions.
        """
        if not text.strip():
            return text

        try:
            preds = self.model.predict(texts=[text])
            if not preds:
                return text

            result = preds[0]
            clusters = result.get_clusters(as_strings=False)
            if not clusters:
                return text

            # Simple resolution: replace all mentions with the first mention in the cluster
            # To avoid offset shifts, we process replacements from end to start
            replacements = []
            for cluster in clusters:
                main_mention_indices = cluster[0]
                main_mention_text = text[main_mention_indices[0] : main_mention_indices[1]]
                for mention_indices in cluster[1:]:
                    replacements.append((mention_indices[0], mention_indices[1], main_mention_text))

            # Sort replacements by start offset descending to avoid index shifting
            replacements.sort(key=lambda x: x[0], reverse=True)

            resolved_text = text
            for start, end, replacement in replacements:
                resolved_text = resolved_text[:start] + replacement + resolved_text[end:]

            return resolved_text
        except Exception:
            # Fallback to original text on any internal error
            return text

    async def resolve(self, text: str) -> str:
        """Asynchronously resolve coreferences in text.

        Args:
            text: Raw input text.

        Returns:
            Resolved text.
        """
        return await asyncio.to_thread(self._resolve_sync, text)
