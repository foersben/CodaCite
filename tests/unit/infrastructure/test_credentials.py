"""Unit tests for the credentials infrastructure module.

Validates the secret service lookup logic using mocks for secretstorage,
ensuring secure retrieval of API keys and handling of locked collections.
"""

from typing import Any

import pytest

from app.infrastructure.credentials import resolve_secret


@pytest.fixture
def mock_secretstorage(mocker: Any) -> Any:
    """Mock the secretstorage library.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        A mocked secretstorage module.
    """
    mock = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"secretstorage": mock})
    return mock


def test_resolve_secret_success(mocker: Any, mock_secretstorage: Any) -> None:
    """Test successful secret resolution from the secret service.

    Given:
        A secret service item exists with the matching label and is unlocked.
    When:
        resolve_secret is called with that label.
    Then:
        It should return the decoded secret string.

    Args:
        mocker: The pytest-mock fixture.
        mock_secretstorage: The mocked secretstorage library.
    """
    # Arrange
    mock_bus = mocker.MagicMock()
    mock_secretstorage.dbus_init.return_value = mock_bus

    mock_collection = mocker.MagicMock()
    mock_collection.is_locked.return_value = False
    mock_secretstorage.get_default_collection.return_value = mock_collection

    mock_item = mocker.MagicMock()
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


def test_resolve_secret_item_not_found(mocker: Any, mock_secretstorage: Any) -> None:
    """Test behavior when no matching item is found.

    Given:
        A secret service collection exists but contains no matching label.
    When:
        resolve_secret is called.
    Then:
        It should return None.

    Args:
        mocker: The pytest-mock fixture.
        mock_secretstorage: The mocked secretstorage library.
    """
    # Arrange
    mock_collection = mocker.MagicMock()
    mock_collection.is_locked.return_value = False
    mock_secretstorage.get_default_collection.return_value = mock_collection
    mock_collection.get_all_items.return_value = []

    # Act
    result = resolve_secret("NonExistent")

    # Assert
    assert result is None


def test_resolve_secret_locked_collection_unlocks(mocker: Any, mock_secretstorage: Any) -> None:
    """Test that a locked collection is unlocked before item retrieval.

    Given:
        A locked secret service collection.
    When:
        resolve_secret is called.
    Then:
        It should attempt to unlock the collection and retrieve the item.

    Args:
        mocker: The pytest-mock fixture.
        mock_secretstorage: The mocked secretstorage library.
    """
    # Arrange
    mock_collection = mocker.MagicMock()
    mock_collection.is_locked.return_value = True
    mock_secretstorage.get_default_collection.return_value = mock_collection

    mock_item = mocker.MagicMock()
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

    Given:
        The secretstorage library cannot be imported.
    When:
        resolve_secret is called.
    Then:
        It should return None gracefully.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    # Ensure it's not in sys.modules first
    mocker.patch.dict("sys.modules", {"secretstorage": mocker.PropertyMock()})
    # Patch __import__ is risky, let's just make sure it's not in sys.modules and raises ImportError
    import builtins

    mocker.patch.object(builtins, "__import__", side_effect=ImportError)

    # Act
    result = resolve_secret("Gemini_API")

    # Assert
    assert result is None


def test_resolve_secret_unlock_failure(mocker: Any, mock_secretstorage: Any) -> None:
    """Test behavior when unlocking fails.

    Given:
        A locked collection that fails to unlock.
    When:
        resolve_secret is called.
    Then:
        It should log the error and continue (or return None if item not found).

    Args:
        mocker: The pytest-mock fixture.
        mock_secretstorage: The mocked secretstorage library.
    """
    # Arrange
    mock_coll = mocker.MagicMock()
    mock_coll.is_locked.return_value = True
    mock_secretstorage.get_default_collection.return_value = mock_coll
    mock_coll.unlock.side_effect = Exception("Failed to unlock")
    mock_coll.get_all_items.return_value = []

    # Act
    val = resolve_secret("Gemini_API")

    # Assert
    assert val is None


def test_resolve_secret_locked_item_unlocks(mocker: Any, mock_secretstorage: Any) -> None:
    """Test that a locked item within an unlocked collection is unlocked.

    Given:
        An unlocked collection containing a locked item.
    When:
        resolve_secret is called.
    Then:
        It should attempt to unlock the item before retrieving the secret.

    Args:
        mocker: The pytest-mock fixture.
        mock_secretstorage: The mocked secretstorage library.
    """
    # Arrange
    mock_collection = mocker.MagicMock()
    mock_collection.is_locked.return_value = False
    mock_secretstorage.get_default_collection.return_value = mock_collection

    mock_item = mocker.MagicMock()
    mock_item.get_label.return_value = "Gemini_API"
    mock_item.is_locked.return_value = True
    mock_item.get_secret.return_value = b"secret"
    mock_collection.get_all_items.return_value = [mock_item]

    # Act
    result = resolve_secret("Gemini_API")

    # Assert
    assert result == "secret"
    mock_item.unlock.assert_called_once()
