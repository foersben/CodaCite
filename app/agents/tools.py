"""Mock action tool bindings for the agent workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DraftEmailInput:
    """Input schema for the Draft Email tool."""

    recipient: str
    subject: str
    body: str


class ActionExecutor:
    """Dispatches action execution requests to the appropriate tool.

    Currently provides mock implementations suitable for testing and
    demonstration purposes.  Real integrations should replace the mock
    implementations with actual API calls.
    """

    def execute(self, action: str, params: dict[str, Any]) -> str:
        """Dispatch *action* with *params* and return the result string.

        Args:
            action: The action name (e.g. ``"draft_email"``).
            params: A dict of parameters for the action.

        Returns:
            A string describing the result of the action.

        Raises:
            ValueError: If *action* is not a recognised tool name.
        """
        if action == "draft_email":
            inp = DraftEmailInput(
                recipient=params.get("recipient", ""),
                subject=params.get("subject", ""),
                body=params.get("body", ""),
            )
            return self.draft_email(inp)

        raise ValueError(f"Unknown action: '{action}'")

    @staticmethod
    def draft_email(payload: DraftEmailInput) -> str:
        """Generate a mock email draft from *payload*.

        Args:
            payload: The :class:`DraftEmailInput` describing the email.

        Returns:
            A formatted string representing the drafted email.
        """
        return (
            f"To: {payload.recipient}\n"
            f"Subject: {payload.subject}\n\n"
            f"{payload.body}"
        )
