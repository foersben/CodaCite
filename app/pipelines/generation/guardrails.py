"""Factuality guardrails for catching LLM hallucinations using NLI."""

import logging
import os
from typing import Any

from transformers import pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


class FactualityGuardrail:
    """Natural Language Inference (NLI) guardrail to verify answer factuality.

    Uses a DeBERTa-v3 model to check if the generated answer is entailed by
    or contradicts the provided context snippets.
    """

    classifier: Any | None

    def __init__(self) -> None:
        """Initialize the NLI classifier.

        Forces transformers to use the unified local cache and run on CPU
        to preserve GPU/VRAM for the main LLM/VLM.
        """
        # Force transformers to use the local cache
        hf_cache = settings.models_dir / "hf_cache"
        os.environ["HF_HOME"] = str(hf_cache)

        model_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
        logger.info("[GUARDRAIL] Loading NLI model: %s", model_id)

        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_id,
                device=-1,  # -1 forces CPU
            )
        except Exception as e:
            logger.error("[GUARDRAIL] Failed to load NLI model: %s", e)
            self.classifier = None

    def verify(self, context: str, generated_answer: str) -> bool:
        """Verify if the generated answer contradicts the context.

        Args:
            context: Combined string of all retrieved context snippets.
            generated_answer: The text produced by the LLM.

        Returns:
            False ONLY if the model explicitly detects a 'contradiction'.
            Returns True for 'entailment' or 'neutral' (or on error).
        """
        if not self.classifier:
            logger.warning("[GUARDRAIL] Classifier not loaded, skipping verification.")
            return True

        if not context.strip() or not generated_answer.strip():
            return True

        try:
            # NLI models take a premise (context) and hypothesis (answer)
            # The model returns labels like 'entailment', 'neutral', 'contradiction'
            # Note: This specific model is trained on 2-class (entailment/contradiction)
            # but sometimes behaves as 3-class depending on the pipeline config.
            result = self.classifier({"text": context, "text_pair": generated_answer})

            if not result:
                return True

            label = str(result[0]["label"]).lower()
            score = float(result[0]["score"])

            logger.debug("[GUARDRAIL] NLI result: label=%s, score=%.4f", label, score)

            if label == "contradiction":
                logger.warning(
                    "[GUARDRAIL] Hallucination detected! Contradiction score: %.4f", score
                )
                return False

            return True

        except Exception as e:
            logger.error("[GUARDRAIL] Verification failed: %s", e)
            return True
