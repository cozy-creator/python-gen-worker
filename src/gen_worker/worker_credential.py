"""ONE authoritative worker credential, for every path that dials the hub.

The defect this exists to delete is a SPLIT, not a value: a stream-local cache
rotated by the hub at ~80 % of TTL, beside ``Settings.bootstrap_worker_jwt`` —
the boot token, frozen at pod create and never updated, reachable from a code
path that runs forever. The transport reads and writes THIS module only, so
there is no second home left to diverge.

Measured consequence of the split: the scheduler stream stays healthy while the
attestation carrier — which opens a BRAND-NEW gRPC Connect every
``hardware_report._ATTESTATION_REPORT_MIN_INTERVAL_S`` — authenticates with the
frozen token. Past the TTL every one of those dials is a fresh
``worker_token_expired`` -> strikes -> ``worker_auth_wedge`` -> pod terminated.

So the credential is not a transport detail. **Anything that dials the hub must
read from one refreshable place**, or a long-lived pod eventually authenticates
some of its calls with a dead token while the rest work fine — the worst
diagnostic shape there is, because the healthy paths mask it. This is NOT a
mint-only concern: any worker that emits an attestation report past its TTL
burns strikes toward a wedge; a mint is simply the first workload that reliably
lives past the TTL.
"""

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
    """Record the freshest credential this worker holds.

    Called by the transport on every hub rotation. Idempotent, and safe from
    any thread: the attestation carrier runs on its own.
    """
    token = str(token or "").strip()
    if not token:
        return
    global _TOKEN, _EXPIRES_AT
    with _LOCK:
        _TOKEN = token
        _EXPIRES_AT = float(expires_at_unix or 0.0)


def install_bootstrap(settings: "Settings") -> None:
    """Hand this module the boot token, from the process entry's `Settings`.

    Deliberately NOT `getattr(get_settings(), "bootstrap_worker_jwt", "")`
    inside a swallowing `except`: a getattr default plus a swallowed exception
    makes a stale reader silent. Direct attribute access here means a rename
    fails loudly at the one site that feeds it.
    """
    global _BOOTSTRAP
    with _LOCK:
        _BOOTSTRAP = str(settings.bootstrap_worker_jwt or "").strip()


def current() -> str:
    """The credential every hub dial must present.

    Falls back to the boot token when no rotation has arrived yet, which is
    the correct answer for the first ~24 minutes of a pod's life and the only
    answer available before the stream is up. The boot token is HANDED to this
    module by the process entry (`install_bootstrap`), never fetched by it: a
    module that can go and find its own credential is a module that can find a
    different one than the rest of the process is using.
    """
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
