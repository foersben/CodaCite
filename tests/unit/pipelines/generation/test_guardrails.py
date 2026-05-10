"""Unit tests for DeBERTa-based factuality guardrails."""

from unittest.mock import MagicMock, patch

import pytest

from app.pipelines.generation.guardrails import FactualityGuardrail


@pytest.fixture
def mock_pipeline():
    """Provides a mocked transformers pipeline."""
    with patch("app.pipelines.generation.guardrails.pipeline") as mock:
        yield mock


def test_guardrail_init(mock_pipeline):
    """Test that the guardrail initializes the pipeline on CPU."""
    guardrail = FactualityGuardrail()
    mock_pipeline.assert_called_once_with(
        "text-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c",
        device=-1,
        truncation=True,
        max_length=512,
    )
    assert guardrail.classifier is not None


def test_guardrail_verify_entailment(mock_pipeline):
    """Test verification when the answer is entailed by the context."""
    mock_clf = MagicMock()
    mock_clf.return_value = [{"label": "entailment", "score": 0.99}]
    mock_pipeline.return_value = mock_clf

    guardrail = FactualityGuardrail()
    is_valid = guardrail.verify("The sky is blue.", "It is a blue sky.")

    assert is_valid is True
    # "It is a blue sky." is > 10 chars, so it should be processed as one sentence
    mock_clf.assert_called_once_with(
        {"text": "The sky is blue.", "text_pair": "It is a blue sky."},
        truncation=True,
        max_length=512,
    )


def test_guardrail_verify_contradiction(mock_pipeline):
    """Test verification when the answer contradicts the context."""
    mock_clf = MagicMock()
    mock_clf.return_value = [{"label": "contradiction", "score": 0.95}]
    mock_pipeline.return_value = mock_clf

    guardrail = FactualityGuardrail()
    is_valid = guardrail.verify("The sky is blue.", "The sky is green.")

    assert is_valid is False
    mock_clf.assert_called_once_with(
        {"text": "The sky is blue.", "text_pair": "The sky is green."},
        truncation=True,
        max_length=512,
    )


def test_guardrail_sentence_splitting(mock_pipeline):
    """Test that the answer is split into sentences for verification."""
    mock_clf = MagicMock()
    mock_clf.return_value = [{"label": "entailment", "score": 0.99}]
    mock_pipeline.return_value = mock_clf

    guardrail = FactualityGuardrail()
    context = "The sky is blue. The grass is green."
    answer = "The sky is blue. The grass is green. This is extra."

    is_valid = guardrail.verify(context, answer)

    assert is_valid is True
    # Should be called 3 times (3 sentences)
    assert mock_clf.call_count == 3
    # Check one of the calls
    mock_clf.assert_any_call(
        {"text": context, "text_pair": "The sky is blue."}, truncation=True, max_length=512
    )
    mock_clf.assert_any_call(
        {"text": context, "text_pair": "The grass is green."}, truncation=True, max_length=512
    )
    mock_clf.assert_any_call(
        {"text": context, "text_pair": "This is extra."}, truncation=True, max_length=512
    )


def test_guardrail_partial_contradiction(mock_pipeline):
    """Test that a single contradicting sentence flags the whole answer."""
    mock_clf = MagicMock()
    # First sentence is fine, second is contradiction
    mock_clf.side_effect = [
        [{"label": "entailment", "score": 0.99}],
        [{"label": "contradiction", "score": 0.95}],
    ]
    mock_pipeline.return_value = mock_clf

    guardrail = FactualityGuardrail()
    context = "The sky is blue."
    answer = "The sky is blue. The sky is red."

    is_valid = guardrail.verify(context, answer)

    assert is_valid is False
    assert mock_clf.call_count == 2


def test_guardrail_empty_inputs(mock_pipeline):
    """Test that empty inputs return True (no contradiction)."""
    guardrail = FactualityGuardrail()
    assert guardrail.verify("", "some answer") is True
    assert guardrail.verify("some context", "") is True


def test_guardrail_load_failure():
    """Test handling of pipeline loading failure."""
    with patch("app.pipelines.generation.guardrails.pipeline", side_effect=RuntimeError("GPU OOM")):
        guardrail = FactualityGuardrail()
        assert guardrail.classifier is None
        assert guardrail.verify("context", "answer") is True  # Should fail open
