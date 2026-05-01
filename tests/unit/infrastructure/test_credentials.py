"""Unit tests for the secret service credential resolution.

Validates the lookup logic against mocked secret storage backends.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.credentials import resolve_secret


@pytest.fixture
def mock_secretstorage():
    """Provides a mocked secretstorage module."""
    with patch("secretstorage.dbus_init") as mock_init:
        mock_bus = MagicMock()
        mock_init.return_value = mock_bus

        mock_collection = MagicMock()
        mock_collection.is_locked.return_value = False

        with patch("secretstorage.get_default_collection", return_value=mock_collection):
            yield {"bus": mock_bus, "collection": mock_collection}


def test_resolve_secret_not_installed():
    """Tests resolution when secretstorage is not installed."""
    with patch("builtins.__import__", side_effect=ImportError):
        assert resolve_secret("KEY") is None


def test_resolve_secret_success(mock_secretstorage):
    """Tests successful secret resolution."""
    mock_item = MagicMock()
    mock_item.get_label.return_value = "MY_KEY"
    mock_item.is_locked.return_value = False
    mock_item.get_secret.return_value = b"top-secret"

    mock_secretstorage["collection"].get_all_items.return_value = [mock_item]

    assert resolve_secret("MY_KEY") == "top-secret"


def test_resolve_secret_locked_collection(mock_secretstorage):
    """Tests resolution when the collection needs unlocking."""
    mock_collection = mock_secretstorage["collection"]
    mock_collection.is_locked.return_value = True

    mock_item = MagicMock()
    mock_item.get_label.return_value = "MY_KEY"
    mock_item.get_secret.return_value = b"secret"
    mock_collection.get_all_items.return_value = [mock_item]

    assert resolve_secret("MY_KEY") == "secret"
    mock_collection.unlock.assert_called_once()


def test_resolve_secret_item_locked(mock_secretstorage):
    """Tests resolution when an individual item needs unlocking."""
    mock_item = MagicMock()
    mock_item.get_label.return_value = "MY_KEY"
    mock_item.is_locked.return_value = True
    mock_item.get_secret.return_value = b"secret"

    mock_secretstorage["collection"].get_all_items.return_value = [mock_item]

    assert resolve_secret("MY_KEY") == "secret"
    mock_item.unlock.assert_called_once()


def test_resolve_secret_item_error(mock_secretstorage):
    """Tests handling of errors when reading an item's secret."""
    mock_item = MagicMock()
    mock_item.get_label.return_value = "FAIL_KEY"
    mock_item.get_secret.side_effect = Exception("Read failure")

    mock_secretstorage["collection"].get_all_items.return_value = [mock_item]

    assert resolve_secret("FAIL_KEY") is None


def test_resolve_secret_not_found(mock_secretstorage):
    """Tests resolution when no item matches the label."""
    mock_secretstorage["collection"].get_all_items.return_value = []
    assert resolve_secret("MISSING") is None


def test_resolve_secret_dbus_error():
    """Tests handling of overall D-Bus or initialization errors."""
    with patch("secretstorage.dbus_init", side_effect=Exception("D-Bus down")):
        assert resolve_secret("KEY") is None


def test_resolve_secret_unlock_failure(mock_secretstorage):
    """Tests resolution when the collection unlock fails."""
    mock_collection = mock_secretstorage["collection"]
    mock_collection.is_locked.return_value = True
    mock_collection.unlock.side_effect = Exception("Unlock fail")

    mock_item = MagicMock()
    mock_item.get_label.return_value = "MY_KEY"
    mock_item.get_secret.return_value = b"secret"
    mock_collection.get_all_items.return_value = [mock_item]

    # Should continue and try to read anyway
    assert resolve_secret("MY_KEY") == "secret"
