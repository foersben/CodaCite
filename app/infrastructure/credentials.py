"""Secret-service key resolution for API credentials.

This module provides logic to look up named entries in the system secret service
(KeePassXC, GNOME Keyring, etc.) via the D-Bus org.freedesktop.secrets protocol.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_secret(label: str) -> str | None:
    """Look up *label* in the system secret service and return its secret.

    The secret is returned as a plain string. The caller must treat it as
    sensitive and must not log, print, or persist it.

    Args:
        label: The display name / title of the entry as shown in the secret
               service provider (e.g. "Gemini_API").

    Returns:
        The secret string if found and successfully unlocked, or None if the
        entry is absent, the service is unavailable, or secretstorage is not
        installed.
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
