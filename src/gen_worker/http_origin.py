"""Who actually answered — the hub, or a proxy standing in front of it? A status code is not one condition."""

from __future__ import annotations

from typing import Any

__all__ = [
    "response_is_from_hub",
    "is_proxy_outage",
    "is_definite_hub_answer",
    "REQUEST_DETERMINED_STATUSES",
]

REQUEST_DETERMINED_STATUSES = frozenset({413})


def _content_type(resp: Any) -> str:
    try:
        return str((resp.headers or {}).get("Content-Type", "")).lower()
    except Exception:  # noqa: BLE001 - header access must never mask the real error
        return ""


def response_is_from_hub(resp: Any) -> bool:
    """True when the body looks like tensorhub's own JSON error envelope."""
    if "json" not in _content_type(resp):
        return False
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - a JSON label with an unparseable body is not ours
        return False
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if isinstance(err, dict) and "code" in err:
        return True
    return isinstance(err, str) and bool(err.strip()) and "message" in body


def is_proxy_outage(resp: Any) -> bool:
    """True when a 404 came from something in front of the hub, not the hub."""
    return not response_is_from_hub(resp)


def is_definite_hub_answer(resp: Any) -> bool:
    """True when this response is the hub's own verdict and may end a loop."""
    try:
        code = int(getattr(resp, "status_code", 0))
    except Exception:  # noqa: BLE001 - an unreadable status is not a verdict
        return False
    if 200 <= code < 400:
        return True
    if code in REQUEST_DETERMINED_STATUSES:
        return True
    return response_is_from_hub(resp)
