"""Factuality guardrails for catching LLM hallucinations using NLI."""

import logging
import os

from transformers import Pipeline, pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


class FactualityGuardrail:
    """Natural Language Inference (NLI) guardrail to verify answer factuality.

    Uses a DeBERTa-v3 model to check if the generated answer is entailed by
    or contradicts the provided context snippets.
    """

    classifier: Pipeline | None

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
                truncation=True,
                max_length=512,
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
            # DeBERTa has a hard 512-token limit shared between premise and hypothesis.
            # To stay within budget, cap the premise (context) at 1 500 chars; the
            # classifier's own truncation handles the rest.
            premise = context.strip()[:1500]

            # Split the answer into individual sentences.  Guard against empty strings
            # and very-short fragments that would confuse the NLI model.
            raw_sentences = generated_answer.replace("\n", " ").split(".")
            sentences = [s.strip() + "." for s in raw_sentences if len(s.strip()) > 10]

            if not sentences:
                # Very short answer — check the whole thing at once
                sentences = [generated_answer.strip()]

            for sentence in sentences:
                raw_result = self.classifier(
                    {"text": premise, "text_pair": sentence},
                    truncation=True,
                    max_length=512,
                )

                # The HuggingFace pipeline can return either a list[dict] or a bare
                # dict depending on version and input type.  Normalise to list.
                if isinstance(raw_result, dict):
                    result_list: list[dict[str, object]] = [raw_result]
                elif isinstance(raw_result, list):
                    result_list = [r for r in raw_result if isinstance(r, dict)]
                else:
                    # Unexpected return type — skip this sentence
                    logger.warning(
                        "[GUARDRAIL] Unexpected classifier output type: %s", type(raw_result)
                    )
                    continue

                if not result_list:
                    continue

                entry = result_list[0]
                label = str(entry.get("label", "")).lower()
                raw_score = entry.get("score", 0.0)
                score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0

                logger.debug(
                    "[GUARDRAIL] Sentence result: label=%s, score=%.4f, text=%s",
                    label,
                    score,
                    sentence[:50] + "...",
                )

                if label == "contradiction":
                    logger.warning(
                        "[GUARDRAIL] Hallucination detected! Contradiction score: %.4f",
                        score,
                    )
                    return False

            return True

        except Exception as e:
            logger.error("[GUARDRAIL] Verification failed: %s", e)
            return True
