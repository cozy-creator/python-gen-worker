"""Process-wide HTTP timeout floor for huggingface_hub (gw#456).

huggingface_hub's default HTTP client has NO timeout (httpx.Client(timeout=None)
on 1.x; requests never times out by default on 0.x), and several HfApi methods
pass an explicit ``timeout=None``. One stalled connection then blocks a clone
forever: sockets sit in CLOSE-WAIT, the job never fails, and tensorhub's mirror
demand rows dedup-join the hung job (observed live on cozy-r2).

:func:`install_hf_http_timeouts` installs a client factory whose requests can
never wait forever: caller-provided numeric timeouts are kept, infinite ones
are floored to env-tunable defaults. The read timeout applies per socket read,
so it doubles as a stall detector for streaming bodies.
"""

from __future__ import annotations

import os
import threading
from typing import Any

CONNECT_TIMEOUT_ENV = "COZY_HTTP_CONNECT_TIMEOUT_S"
READ_TIMEOUT_ENV = "COZY_HTTP_READ_TIMEOUT_S"
DEFAULT_CONNECT_TIMEOUT_S = 15.0
DEFAULT_READ_TIMEOUT_S = 60.0

_lock = threading.Lock()
_installed = False


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(0.1, float(raw))
        except ValueError:
            pass
    return default


def http_timeouts() -> tuple[float, float]:
    """(connect, read) floor, read from env on every call (test-tunable)."""
    return (
        _env_seconds(CONNECT_TIMEOUT_ENV, DEFAULT_CONNECT_TIMEOUT_S),
        _env_seconds(READ_TIMEOUT_ENV, DEFAULT_READ_TIMEOUT_S),
    )


def _floor_timeout_hook(request: Any) -> None:
    """httpx request event hook: entries in request.extensions['timeout'] are
    None when infinite — replace only those; explicit numbers win."""
    connect, read = http_timeouts()
    defaults = {"connect": connect, "read": read, "write": read, "pool": connect}
    current = dict(request.extensions.get("timeout") or {})
    request.extensions["timeout"] = {
        k: (current[k] if current.get(k) is not None else v)
        for k, v in defaults.items()
    }


def install_hf_http_timeouts() -> None:
    """Idempotent; call before any huggingface_hub network use."""
    global _installed
    with _lock:
        if _installed:
            return
        try:
            from huggingface_hub.utils import _http as hf_http

            default_factory = hf_http.default_client_factory
            set_factory = hf_http.set_client_factory
        except (ImportError, AttributeError):
            _install_requests_backend()  # huggingface_hub 0.x
            _installed = True
            return

        def _factory() -> Any:
            client = default_factory()
            client.event_hooks["request"] = [_floor_timeout_hook] + list(
                client.event_hooks.get("request") or []
            )
            return client

        set_factory(_factory)
        _installed = True


def _install_requests_backend() -> None:
    """huggingface_hub 0.x (requests): default a (connect, read) timeout on
    every Session request that would otherwise wait forever."""
    import requests
    from huggingface_hub import configure_http_backend

    class _TimeoutSession(requests.Session):
        def request(self, method: str, url: str, **kwargs: Any) -> Any:  # type: ignore[override]
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = http_timeouts()
            return super().request(method, url, **kwargs)

    configure_http_backend(backend_factory=_TimeoutSession)


def hf() -> Any:
    """THE sanctioned huggingface_hub accessor (gw#467): installs the timeout
    floor, then returns the module. Every network entry point (HfApi,
    snapshot_download, hf_hub_download, ...) must be reached through here —
    the CI guard (scripts/lint_http_timeouts.py) rejects direct imports
    anywhere else in src/, so the floor is structurally unskippable.
    Non-network imports (huggingface_hub.errors / .constants) stay direct."""
    install_hf_http_timeouts()
    import huggingface_hub

    return huggingface_hub


__all__ = [
    "CONNECT_TIMEOUT_ENV",
    "READ_TIMEOUT_ENV",
    "hf",
    "http_timeouts",
    "install_hf_http_timeouts",
]
