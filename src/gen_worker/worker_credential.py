"""pgw#848: ONE authoritative worker credential, for every path that dials the hub.

The defect this exists to delete is a SPLIT, not a value. The worker held two
credentials with different lifetimes:

* ``transport._worker_jwt`` — rotated by the hub at ~80 % of TTL over the
  scheduler stream, and therefore current (that stream-local cache is DELETED
  as of pgw#893 §2: the transport now reads and writes this module only, so
  there is no second home left to diverge);
* ``Settings.bootstrap_worker_jwt`` — the boot token, **frozen at pod create and never
  updated by anything**, and reachable from a code path that runs forever.

MEASURED (hub ``pod_events``, pgw#846 attempts 16 and 17). The scheduler stream
never dropped — the lease and the beats were healthy the whole way. What killed
both pods was the *diagnostic* carrier: the post-mortem/attestation reporter
opens a BRAND-NEW gRPC Connect every
``hardware_report._ATTESTATION_REPORT_MIN_INTERVAL_S`` (300 s, a throttle, not
a reconnect cadence) and authenticated it with the FROZEN token. Past T+30 min
every one of those dials is a fresh ``worker_token_expired`` -> three strikes ->
``worker_auth_wedge`` -> pod terminated. One dial per five minutes puts death
~15 minutes after first expiry; attempt sixteen's 00:30:05 -> 00:40:39 fits
exactly.

So the credential is not a transport detail. **Anything that dials the hub must
read from one refreshable place**, or a long-lived pod eventually authenticates
some of its calls with a dead token while the rest work fine — which is the
worst diagnostic shape there is, because the healthy paths mask it.

NOT a mint-only concern, and it should not be filed as one: *any* worker that
emits an attestation report past its TTL burns strikes toward a wedge. A mint
is simply the first workload that reliably lives past 30 minutes.
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

    pgw#931 (§1.18): `current()` used to reach for `get_settings()` itself,
    inside `getattr(..., "bootstrap_worker_jwt", "")` inside a bare
    `except Exception: return ""`. That is the DEFECT-TAXONOMY C8 shape on the
    field pgw#848 renamed precisely so a stale reader would raise: a getattr
    default plus a swallowed exception restores the silence the rename existed
    to break. Direct attribute access here means a rename fails loudly at the
    one site that feeds it.
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
