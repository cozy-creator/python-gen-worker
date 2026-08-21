"""ONE authoritative worker credential, for every path that dials the hub."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TOKEN: str = ""
_EXPIRES_AT: float = 0.0
_BOOTSTRAP: str = ""


def install(token: str, expires_at_unix: float = 0.0) -> None:
    """Record the freshest credential this worker holds."""
    token = str(token or "").strip()
    if not token:
        return
    global _TOKEN, _EXPIRES_AT
    with _LOCK:
        _TOKEN = token
        _EXPIRES_AT = float(expires_at_unix or 0.0)


def install_bootstrap(settings: "Settings") -> None:
    """Hand this module the boot token, from the process entry's `Settings`."""
    global _BOOTSTRAP
    with _LOCK:
        _BOOTSTRAP = str(settings.bootstrap_worker_jwt or "").strip()


def current() -> str:
    """The credential every hub dial must present."""
    with _LOCK:
        if _TOKEN:
            return _TOKEN
        return _BOOTSTRAP


def expires_at() -> float:
    """Unix expiry of the installed credential; 0.0 when unknown."""
    with _LOCK:
        return _EXPIRES_AT


def reset() -> None:
    """Tests only: forget the rotation and fall back to settings."""
    global _TOKEN, _EXPIRES_AT
    with _LOCK:
        _TOKEN, _EXPIRES_AT = "", 0.0


__all__ = ["current", "expires_at", "install", "reset"]
