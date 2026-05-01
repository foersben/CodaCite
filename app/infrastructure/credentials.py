"""Secret-service key resolution for API credentials.

This module provides logic to look up named entries in the system secret service
(KeePassXC, GNOME Keyring, etc.) via the D-Bus org.freedesktop.secrets protocol.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_secret(label: str) -> str | None:
    """Look up *label* in the system secret service and return its secret.

    Uses the `secretstorage` library to interface with D-Bus secret providers
    (like KeePassXC or GNOME Keyring). This ensures API keys (e.g., Gemini)
    are not stored in plain text or environment variables on the local machine.

    Pipeline Role:
        - Infrastructure Bootstrapping: Securely fetching API credentials for
          Gemini models and other external services.

    Args:
        label: The display name/title of the entry in the secret service
               (e.g., "Gemini_API").

    Returns:
        The decoded secret string if successful, or None if unavailable.
    """
    try:
        import secretstorage  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("secretstorage not installed — skipping secret service lookup")
        return None

    try:
        bus = secretstorage.dbus_init()
        collection = secretstorage.get_default_collection(bus)

        if collection.is_locked():
            # The collection requires user interaction to unlock.
            # We try via the service; if no D-Bus prompt responds we bail.
            try:
                collection.unlock()
            except Exception as e:
                logger.debug("Failed to unlock secret collection: %s", e)

        for item in collection.get_all_items():
            if item.get_label() == label:
                try:
                    if item.is_locked():
                        item.unlock()
                    secret: str = item.get_secret().decode()
                    logger.debug("Resolved secret for label %r from secret service", label)
                    return secret
                except Exception as e:
                    logger.debug(
                        "Secret service item %r found but could not be unlocked/read: %s",
                        label,
                        e,
                    )
                    return None

        logger.debug("No secret service entry with label %r found", label)
        return None

    except Exception as exc:
        # Covers: D-Bus not running, KeePassXC not open, dbus-python unavailable
        logger.debug("Secret service lookup failed for %r: %s", label, exc)
        return None
