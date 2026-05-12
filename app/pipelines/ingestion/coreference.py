"""Infrastructure implementation of CoreferenceResolver using fastcoref.

This module provides an implementation of the CoreferenceResolver port using
the fastcoref library for efficient, local coreference resolution. It handles
the crucial task of resolving entity pronouns and references, which significantly
improves the precision of downstream entity extraction and relationship mapping.
"""

import asyncio
import re
from typing import Any

from app.core.interfaces import CoreferenceResolver


def safe_get_clusters(model: Any, text: str) -> list[list[tuple[int, int]]]:
    """Safely extracts cluster indices using string matching.

    This function serves as an architectural guardrail against a known bug in
    the `fastcoref` library where the `as_strings=False` parameter causes
    segmentation faults or index errors during token-to-character alignment.

    Instead of relying on the library's internal mapping, we retrieve string-based
    clusters and perform our own deterministic string matching. We use a
    stateful search (`last_end`) to correctly resolve multiple identical
    mentions (e.g., 'He said he saw him').

    Args:
        model: The initialized fastcoref model instance.
        text: The raw input text to analyze.

    Returns:
        A list of entity clusters. Each cluster contains character-span tuples
        `(start, end)` identifying every mention of that specific entity.
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
        Phase 2: Coreference Resolution. Pre-processing text before chunking
        (Phase 3) and extraction (Phase 6) to ensure that entity spotting
        captures the correct context for every mention.

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

        from app.core.config import settings

        # Compatibility fix for transformers 5.x
        if not hasattr(FCorefModel, "all_tied_weights_keys"):
            FCorefModel.all_tied_weights_keys = property(lambda self: {})

        # Use blank model if no model provided to avoid downloading en_core_web_sm
        if nlp is None:
            nlp = spacy.blank("en")

        self.model = FCoref(model_name_or_path=model_name_or_path, device=settings.device, nlp=nlp)

    def _resolve_sync(self, text: str) -> str:
        """Synchronous coreference resolution logic.

        Identifies clusters of mentions and replaces every secondary mention
        (usually a pronoun or descriptive noun phrase) with the 'head' mention
        (the most descriptive antecedent).

        Algorithm:
            1. Extract mention clusters via `safe_get_clusters`.
            2. For each cluster, treat the first mention as the primary key.
            3. Generate a list of all required replacements.
            4. **Reverse-Sort Replacement**: Sort replacements by start offset
               descending. This is a critical pattern that allows in-place
               string mutation without invalidating subsequent offsets.

        Args:
            text: The normalized document text to resolve.

        Returns:
            A string where coreferences have been replaced by their primary
            antecedents. Returns the original text if no clusters are found.
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

        Offloads the computationally intensive coreference resolution logic to
        a separate thread to maintain the responsiveness of the async pipeline.

        Args:
            text: The normalized document text.

        Returns:
            The resolved text string.
        """
        return await asyncio.to_thread(self._resolve_sync, text)
