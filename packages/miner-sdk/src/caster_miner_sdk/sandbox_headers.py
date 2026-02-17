"""HTTP header names used for host↔sandbox requests."""

from __future__ import annotations

CASTER_SESSION_ID_HEADER = "x-caster-session-id"
CASTER_HOST_CONTAINER_URL_HEADER = "x-caster-host-container-url"

# Backward-compatible aliases.
SANDBOX_SESSION_ID_HEADER = CASTER_SESSION_ID_HEADER
SANDBOX_HOST_CONTAINER_URL_HEADER = CASTER_HOST_CONTAINER_URL_HEADER

__all__ = [
    "CASTER_HOST_CONTAINER_URL_HEADER",
    "CASTER_SESSION_ID_HEADER",
    "SANDBOX_HOST_CONTAINER_URL_HEADER",
    "SANDBOX_SESSION_ID_HEADER",
]
