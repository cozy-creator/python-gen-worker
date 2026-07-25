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
import requests

CONNECT_TIMEOUT_ENV = "COZY_HTTP_CONNECT_TIMEOUT_S"
READ_TIMEOUT_ENV = "COZY_HTTP_READ_TIMEOUT_S"
DEFAULT_CONNECT_TIMEOUT_S = 15.0
DEFAULT_READ_TIMEOUT_S = 60.0

_lock = threading.Lock()
_installed = False


class HfHttpFloorError(RuntimeError):
    """The huggingface_hub timeout floor could not be installed or PROVEN.

    pgw#657: this patch is load-bearing and reaches into ``huggingface_hub``'s
    backend, which has already been reshaped once (requests -> httpx). A silent
    revert puts the whole fleet back on infinite timeouts — the gw#456 hang —
    and nothing would say so. So installation ends in a behavioural assertion
    on the session huggingface_hub will actually use, and a failure is loud at
    boot instead of a wedge weeks later.
    """


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


def _hf_version() -> str:
    try:
        import huggingface_hub

        return str(getattr(huggingface_hub, "__version__", "") or "unknown")
    except Exception:  # noqa: BLE001 — only used to name a version in an error
        return "unknown"


def _verify_httpx_floor() -> None:
    """Prove the floor on the session huggingface_hub will actually use."""
    from huggingface_hub.utils import _http as hf_http

    # Drop any client built before the factory was registered, so this checks
    # the client huggingface_hub will BUILD, not one that happens to be cached
    # (set_client_factory does this today; do not depend on it).
    close = getattr(hf_http, "close_session", None)
    if callable(close):
        close()
    client = hf_http.get_session()
    hooks = list(getattr(client, "event_hooks", {}).get("request") or [])
    if _floor_timeout_hook not in hooks:
        raise HfHttpFloorError(
            "huggingface_hub timeout floor did not take: get_session() returned "
            f"a client without the floor hook (huggingface_hub {_hf_version()}). "
            "The backend was reshaped — re-derive the patch in gen_worker/net.py; "
            "running without it means infinite HTTP timeouts fleet-wide (gw#456)."
        )


def _verify_requests_floor() -> None:
    from huggingface_hub.utils import get_session

    session = get_session()
    if not isinstance(session, _TimeoutSession):
        raise HfHttpFloorError(
            "huggingface_hub timeout floor did not take: get_session() returned "
            f"{type(session).__name__} (huggingface_hub {_hf_version()}). "
            "See gen_worker/net.py — infinite HTTP timeouts fleet-wide (gw#456)."
        )


def install_hf_http_timeouts() -> None:
    """Idempotent; call before any huggingface_hub network use. Raises
    :class:`HfHttpFloorError` if the floor cannot be proven (pgw#657)."""
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
            _verify_requests_floor()
            _installed = True
            return

        def _factory() -> Any:
            client = default_factory()
            client.event_hooks["request"] = [_floor_timeout_hook] + list(
                client.event_hooks.get("request") or []
            )
            return client

        set_factory(_factory)
        _verify_httpx_floor()
        _installed = True


class _TimeoutSession(requests.Session):
    """huggingface_hub 0.x backend: floor every otherwise-infinite timeout."""

    def request(self, method: str, url: str, **kwargs: Any) -> Any:  # type: ignore[override]
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = http_timeouts()
        return super().request(method, url, **kwargs)


def _install_requests_backend() -> None:
    """huggingface_hub 0.x (requests): default a (connect, read) timeout on
    every Session request that would otherwise wait forever."""
    from huggingface_hub import configure_http_backend

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
