"""Unit tests for the credentials infrastructure module.

This module validates the secret service lookup logic using mocks for secretstorage.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.infrastructure.credentials import resolve_secret


@pytest.fixture
def mock_secretstorage(mocker: Any) -> Any:
    """Mock the secretstorage library."""
    mock = mocker.patch("app.infrastructure.credentials.secretstorage", create=True)
    return mock


def test_resolve_secret_success(mock_secretstorage: Any) -> None:
    """Test successful secret resolution from the secret service.

    Given: A secret service item exists with the matching label and is unlocked.
    When: resolve_secret is called with that label.
    Then: It should return the decoded secret string.
    """
    # Arrange
    mock_bus = MagicMock()
    mock_secretstorage.dbus_init.return_value = mock_bus

    mock_collection = MagicMock()
    mock_collection.is_locked.return_value = False
    mock_secretstorage.get_default_collection.return_value = mock_collection

    mock_item = MagicMock()
    mock_item.get_label.return_value = "Gemini_API"
    mock_item.is_locked.return_value = False
    mock_item.get_secret.return_value = b"my-secret-key"

    mock_collection.get_all_items.return_value = [mock_item]

    # Act
    result = resolve_secret("Gemini_API")

    # Assert
    assert result == "my-secret-key"
    mock_secretstorage.dbus_init.assert_called_once()
    mock_secretstorage.get_default_collection.assert_called_once_with(mock_bus)


def test_resolve_secret_item_not_found(mock_secretstorage: Any) -> None:
    """Test behavior when no matching item is found.

    Given: A secret service collection exists but contains no matching label.
    When: resolve_secret is called.
    Then: It should return None.
    """
    # Arrange
    mock_collection = MagicMock()
    mock_collection.is_locked.return_value = False
    mock_secretstorage.get_default_collection.return_value = mock_collection
    mock_collection.get_all_items.return_value = []

    # Act
    result = resolve_secret("NonExistent")

    # Assert
    assert result is None


def test_resolve_secret_locked_collection_unlocks(mock_secretstorage: Any) -> None:
    """Test that a locked collection is unlocked before item retrieval.

    Given: A locked secret service collection.
    When: resolve_secret is called.
    Then: It should attempt to unlock the collection and retrieve the item.
    """
    # Arrange
    mock_collection = MagicMock()
    mock_collection.is_locked.return_value = True
    mock_secretstorage.get_default_collection.return_value = mock_collection

    mock_item = MagicMock()
    mock_item.get_label.return_value = "Gemini_API"
    mock_item.is_locked.return_value = False
    mock_item.get_secret.return_value = b"secret"
    mock_collection.get_all_items.return_value = [mock_item]

    # Act
    result = resolve_secret("Gemini_API")

    # Assert
    assert result == "secret"
    mock_collection.unlock.assert_called_once()


def test_resolve_secret_library_missing(mocker: Any) -> None:
    """Test behavior when secretstorage is not installed.

    Given: The secretstorage library cannot be imported.
    When: resolve_secret is called.
    Then: It should return None gracefully.
    """
    # Arrange
    mocker.patch("builtins.__import__", side_effect=ImportError)

    # Act
    # We need to ensure we don't accidentally use the already imported mock
    # In reality, the 'import secretstorage' inside the function handles this
    result = resolve_secret("Gemini_API")

    # Assert
    assert result is None
