"""
Tests for the agents module.

Covers:
- IntentRouter: LLM-based classification of 'knowledge_retrieval' vs 'action_execution'
- ActionExecutor: mock tool bindings (e.g., Draft Email)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agents.router import IntentRouter, IntentType
from app.agents.tools import ActionExecutor, DraftEmailInput

# ---------------------------------------------------------------------------
# IntentRouter tests
# ---------------------------------------------------------------------------


class TestIntentRouter:
    """Tests for IntentRouter."""

    def test_routes_retrieval_intent(self) -> None:
        """route() should return KNOWLEDGE_RETRIEVAL for informational queries."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="knowledge_retrieval")

        router = IntentRouter(llm=mock_llm)
        intent = router.route("What is the capital of France?")

        assert intent == IntentType.KNOWLEDGE_RETRIEVAL

    def test_routes_action_intent(self) -> None:
        """route() should return ACTION_EXECUTION for action-oriented queries."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="action_execution")

        router = IntentRouter(llm=mock_llm)
        intent = router.route("Draft an email to the team about the quarterly review.")

        assert intent == IntentType.ACTION_EXECUTION

    def test_defaults_to_retrieval_on_unknown_response(self) -> None:
        """route() should fall back to KNOWLEDGE_RETRIEVAL for unrecognised LLM output."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="I don't know what to do.")

        router = IntentRouter(llm=mock_llm)
        intent = router.route("Hm?")

        assert intent == IntentType.KNOWLEDGE_RETRIEVAL

    def test_intent_type_enum_values(self) -> None:
        """IntentType should have KNOWLEDGE_RETRIEVAL and ACTION_EXECUTION members."""
        assert IntentType.KNOWLEDGE_RETRIEVAL
        assert IntentType.ACTION_EXECUTION


# ---------------------------------------------------------------------------
# ActionExecutor (tool bindings) tests
# ---------------------------------------------------------------------------


class TestActionExecutor:
    """Tests for ActionExecutor mock tools."""

    def test_draft_email_returns_draft(self) -> None:
        """draft_email() should return a non-empty email draft string."""
        executor = ActionExecutor()
        payload = DraftEmailInput(
            recipient="team@example.com",
            subject="Q3 Review",
            body="Please review the attached report.",
        )
        result = executor.draft_email(payload)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "team@example.com" in result or "Q3 Review" in result

    def test_draft_email_input_dataclass(self) -> None:
        """DraftEmailInput should expose recipient, subject, and body."""
        inp = DraftEmailInput(
            recipient="a@b.com",
            subject="Hello",
            body="World",
        )
        assert inp.recipient == "a@b.com"
        assert inp.subject == "Hello"
        assert inp.body == "World"

    def test_execute_dispatches_draft_email(self) -> None:
        """execute() with 'draft_email' action should call draft_email()."""
        executor = ActionExecutor()
        result = executor.execute(
            action="draft_email",
            params={
                "recipient": "boss@company.com",
                "subject": "Update",
                "body": "Here is the latest update.",
            },
        )
        assert isinstance(result, str)

    def test_execute_unknown_action_raises(self) -> None:
        """execute() with an unknown action name should raise ValueError."""
        executor = ActionExecutor()
        with pytest.raises(ValueError, match="Unknown action"):
            executor.execute(action="send_fax", params={})
