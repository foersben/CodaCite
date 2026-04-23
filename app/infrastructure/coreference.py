"""Infrastructure implementation of CoreferenceResolver using fastcoref."""

import asyncio

from app.domain.ports import CoreferenceResolver


class FastCorefResolver(CoreferenceResolver):
    """Coreference resolution using fastcoref."""

    def __init__(self, model_name_or_path: str = "biu-nlp/f-coref", nlp=None) -> None:
        """Initialize the fastcoref model."""
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
        """Synchronous resolution logic."""
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
                main_mention_text = text[main_mention_indices[0]:main_mention_indices[1]]
                for mention_indices in cluster[1:]:
                    replacements.append((mention_indices[0], mention_indices[1], main_mention_text))

            # Sort replacements by start offset descending
            replacements.sort(key=lambda x: x[0], reverse=True)

            resolved_text = text
            for start, end, replacement in replacements:
                resolved_text = resolved_text[:start] + replacement + resolved_text[end:]

            return resolved_text
        except Exception:
            return text

    async def resolve(self, text: str) -> str:
        """Resolve coreferences in a separate thread to avoid blocking event loop."""
        return await asyncio.to_thread(self._resolve_sync, text)
