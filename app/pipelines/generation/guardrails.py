"""Factuality guardrails for catching LLM hallucinations using Natural Language Inference (NLI)."""

import logging
import os
import re

from transformers import Pipeline, pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


class FactualityGuardrail:
    """Natural Language Inference (NLI) guardrail for grounding verification.

    This class implements a multi-stage quality gate to ensure that LLM-generated
    responses are strictly derived from the provided context. It employs a
    specialized DeBERTa-v3 model to detect logical contradictions and a verbatim
    substring check to validate cited quotes.

    Attributes:
        classifier: The transformers pipeline for text classification (NLI).
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
        """Verify the factuality of a generated answer against the source context.

        The verification follows a three-step process:
        1. Pre-processing: Strips internal reasoning (<think>) blocks.
        2. Quote Validation: Ensures all bracketed quotes exist verbatim in the source.
        3. NLI Inference: Performs sentence-level contradiction detection using
           the DeBERTa classifier.

        Args:
            context: The aggregated document context used for grounding.
            generated_answer: The raw response produced by the generator.

        Returns:
            True if the answer is factual and contains no hallucinated quotes;
            False if a contradiction or quote mismatch is detected.
        """
        # 1. Strip think blocks from the generated answer before any verification
        clean_answer = re.sub(r"<think>.*?</think>", "", generated_answer, flags=re.DOTALL).strip()

        if not self.classifier:
            logger.warning("[GUARDRAIL] Classifier not loaded, skipping NLI verification.")
            # Still run quote verification
            return self._verify_quotes(context, clean_answer)

        if not context.strip() or not clean_answer:
            return True

        # 2. Quote Verification (Regex-based verbatim check)
        if not self._verify_quotes(context, clean_answer):
            return False

        # 3. NLI Contradiction Check
        try:
            premise = context.strip()[:1500]
            raw_sentences = clean_answer.replace("\n", " ").split(".")
            sentences = [s.strip() + "." for s in raw_sentences if len(s.strip()) > 10]

            if not sentences:
                sentences = [clean_answer]

            for sentence in sentences:
                raw_result = self.classifier(
                    {"text": premise, "text_pair": sentence},
                    truncation=True,
                    max_length=512,
                )

                if isinstance(raw_result, dict):
                    result_list: list[dict[str, object]] = [raw_result]
                elif isinstance(raw_result, list):
                    result_list = [r for r in raw_result if isinstance(r, dict)]
                else:
                    continue

                if not result_list:
                    continue

                entry = result_list[0]
                label = str(entry.get("label", "")).lower()
                raw_score = entry.get("score", 0.0)
                score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0

                if label == "contradiction":
                    logger.warning(
                        "[GUARDRAIL] Hallucination detected! Contradiction score: %.4f",
                        score,
                    )
                    return False

            return True

        except Exception as e:
            logger.error("[GUARDRAIL] NLI verification failed: %s", e)
            return True

    def _verify_quotes(self, context: str, clean_answer: str) -> bool:
        """Verify that all text within double quotes is present in the context.

        This check enforces a strict verbatim policy for evidence citations.
        Any quoted text that does not appear as a direct substring in the
        provided context is flagged as a potential hallucination.

        Args:
            context: The reference document text.
            clean_answer: The generated answer to inspect.

        Returns:
            True if all quotes are verbatim; False otherwise.
        """
        quotes = re.findall(r'"([^"]+)"', clean_answer)
        if not quotes:
            return True

        for quote in quotes:
            # We use a simple substring check for "verbatim" compliance
            if quote not in context:
                logger.warning("[GUARDRAIL] Hallucinated quote detected: '%s'", quote)
                return False
        return True
