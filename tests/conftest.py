"""Global pytest fixtures."""

import logging

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    """Setup basic logging for tests to suppress noise or capture it."""
    logging.getLogger().setLevel(logging.CRITICAL)
