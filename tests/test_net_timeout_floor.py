"""gw#467: the wrapper's client can never carry an infinite timeout.

Asserts the EFFECTIVE per-request timeouts on the sanctioned hf client are
numeric — if an upstream huggingface_hub change ever bypasses our floor
(different session plumbing, new client factory contract), this fails in
tests instead of hanging in production. There is no hf_hub 2.x coming to fix
the timeout=None default; we own the floor.
"""

from __future__ import annotations

import pytest

from gen_worker import net


def _effective_timeout(client, **request_kwargs) -> dict:
    """Build a request and run the client's request hooks — exactly what
    httpx does in send() before the transport reads the timeout."""
    req = client.build_request("GET", "http://127.0.0.1:9/never", **request_kwargs)
    for hook in client.event_hooks.get("request", []):
        hook(req)
    return dict(req.extensions.get("timeout") or {})


@pytest.fixture()
def hf_client(monkeypatch):
    monkeypatch.setenv(net.CONNECT_TIMEOUT_ENV, "7")
    monkeypatch.setenv(net.READ_TIMEOUT_ENV, "11")
    hub = net.hf()  # the sanctioned accessor installs the floor
    assert hasattr(hub, "HfApi") and hasattr(hub, "snapshot_download")
    from huggingface_hub.utils._http import get_session

    return get_session()


def test_default_timeouts_are_numeric_never_none(hf_client):
    t = _effective_timeout(hf_client)
    for key in ("connect", "read", "write", "pool"):
        assert isinstance(t.get(key), (int, float)) and t[key] > 0, \
            f"effective {key} timeout must be numeric, got {t!r}"
    assert t["connect"] == 7.0 and t["read"] == 11.0


def test_explicit_none_is_floored(hf_client):
    # The HfApi.repo_info shape: an explicit timeout=None must not mean
    # "wait forever" — it gets the env floor.
    t = _effective_timeout(hf_client, timeout=None)
    assert t.get("connect") == 7.0 and t.get("read") == 11.0, t


def test_explicit_numeric_timeout_wins(hf_client):
    t = _effective_timeout(hf_client, timeout=3)
    for key in ("connect", "read", "write", "pool"):
        assert t.get(key) == 3.0, t
