"""Infrastructure implementation of CoreferenceResolver using fastcoref.

This module provides an implementation of the CoreferenceResolver port using
the fastcoref library for efficient, local coreference resolution.
"""

import asyncio
import re
from typing import Any

from app.domain.ports import CoreferenceResolver


def safe_get_clusters(model: Any, text: str) -> list[list[tuple[int, int]]]:
    """Safely extracts cluster indices using string matching.

    Bypasses the fastcoref bug where as_strings=False crashes on token alignment.
    This implementation tracks offsets within clusters to correctly handle
    repeated mentions (e.g., multiple occurrences of 'She').
    """
    preds = model.predict(texts=[text])
    if not preds:
        return []

    # Get the string clusters (which doesn't crash)
    string_clusters = preds[0].get_clusters(as_strings=True)

    index_clusters = []
    for cluster in string_clusters:
        current_cluster_indices = []
        last_end = 0
        for entity in cluster:
            # Find the next occurrence of the string after last_end
            # We use re.escape to handle special characters in mentions
            match = re.search(re.escape(entity), text[last_end:])
            if match:
                start = last_end + match.start()
                end = last_end + match.end()
                current_cluster_indices.append((start, end))
                last_end = end
            else:
                # Fallback: if not found after last_end, try from the beginning.
                # This handles cases where clusters might not be strictly ordered.
                match_from_start = re.search(re.escape(entity), text)
                if match_from_start:
                    span = match_from_start.span()
                    current_cluster_indices.append(span)
                    # Update last_end to ensure subsequent matches continue forward if possible
                    last_end = max(last_end, span[1])

        if current_cluster_indices:
            index_clusters.append(current_cluster_indices)

    return index_clusters


class FastCorefResolver(CoreferenceResolver):
    """Coreference resolution using the fastcoref library.

    This resolver replaces pronouns and other referring expressions (e.g., 'he',
    'it', 'the company') with their primary mention (antecedent) within a
    document chunk.

    Pipeline Role:
        Phase 1: Coreference Resolution. Pre-processing text before chunking
        and extraction to ensure that entity extraction (Phase 5) captures
        the correct context for every mention.

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
            clusters = safe_get_clusters(self.model, text)
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
