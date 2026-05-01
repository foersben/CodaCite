"""Unit tests for the application configuration layer.

Validates the settings initialization and secret retrieval logic, ensuring
environment variables and secret service integration work correctly.
"""

from typing import Any

from app.config import Settings


def test_gemini_key_retrieval_from_secret_service(mocker: Any) -> None:
    """Tests that Settings retrieves the key from secret service if not provided in env.

    Given:
        A system where GEMINI_API_KEY is not set in the environment.
    When:
        Settings is initialized with an empty gemini_api_key.
    Then:
        It should fall back to retrieving the key from the Secret Service.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mock_resolve = mocker.patch("app.config.resolve_secret")
    mock_resolve.return_value = "secret-key-from-service"

    # Act
    # We need to bypass the initial load from environment for this test
    # or explicitly set it to empty
    settings = Settings(gemini_api_key="")

    # Assert
    assert settings.gemini_api_key == "secret-key-from-service"
    mock_resolve.assert_called_once_with("Gemini_API")


def test_gemini_key_env_takes_precedence(mocker: Any) -> None:
    """Tests that environment variables take precedence over secret service.

    Given:
        A system where GEMINI_API_KEY is provided via environment.
    When:
        Settings is initialized.
    Then:
        It should use the environment key and not call the Secret Service.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mock_resolve = mocker.patch("app.config.resolve_secret")
    mock_resolve.return_value = "secret-key-from-service"

    # Act
    settings = Settings(gemini_api_key="env-key")

    # Assert
    assert settings.gemini_api_key == "env-key"
    mock_resolve.assert_not_called()


def test_gemini_key_lookup_fails_gracefully(mocker: Any) -> None:
    """Tests behavior when secret service lookup fails.

    Given:
        A system where the secret service lookup returns None.
    When:
        Settings is initialized without an environment key.
    Then:
        It should gracefully handle the failure and return an empty key.

    Args:
        mocker: The pytest-mock fixture.
    """
    # Arrange
    mock_resolve = mocker.patch("app.config.resolve_secret")
    mock_resolve.return_value = None

    # Act
    settings = Settings(gemini_api_key="")

    # Assert
    assert settings.gemini_api_key == ""
