"""Who actually answered — the hub, or a proxy standing in front of it?

te#125/th#1238: a 404 is not one condition. It is two, with opposite handling:

  * The HUB answering 404 means "this route/resource does not exist". Fatal.
  * A PROXY answering 404 (ngrok, an ingress, a load balancer with no healthy
    backend) means "the hub is not reachable right now". Transient.

Conflating them cost a 116-minute H100 producer run: the hub restarted for a
few seconds mid-job, the in-flight upload got ngrok's 404 page, and the worker
classified it `retryable=False` and killed the job at the last mile. The gRPC
worker stream reconnects across a hub restart; HTTP calls in flight did not.

The two are trivially separable in practice, and we verify it rather than
assume it: the hub answers `application/json` carrying its `{"error": {...}}`
envelope, while ngrok's offline page is `text/html`. Measured 2026-07-27:

    hub 404   -> content-type: application/json; {"error":{"code":"not_found",...}}
    ngrok 404 -> content-type: text/html

DELIBERATELY CONSERVATIVE: we only report "proxy" when the response clearly is
NOT the hub's envelope. An unrecognised body is treated as proxy-origin, which
biases toward RETRYING. That is the safe direction here — retrying a genuinely
missing route costs a bounded backoff and then fails anyway, whereas treating
an outage as fatal destroys hours of GPU work with no recourse.

Scope note: this is for calls to OUR hub. It must NOT be applied to third-party
hosts (e.g. civitai), where a 404 really does mean "not found" and there is no
proxy of ours in the path.
"""

from __future__ import annotations

from typing import Any

__all__ = ["response_is_from_hub", "is_proxy_outage"]


def _content_type(resp: Any) -> str:
    try:
        return str((resp.headers or {}).get("Content-Type", "")).lower()
    except Exception:  # noqa: BLE001 - header access must never mask the real error
        return ""


def response_is_from_hub(resp: Any) -> bool:
    """True when the body looks like tensorhub's own JSON error envelope.

    Checks the parsed body rather than trusting Content-Type alone: a proxy is
    free to label an HTML page as JSON, but it will not synthesise our
    ``{"error": {...}}`` shape.
    """
    if "json" not in _content_type(resp):
        return False
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - a JSON label with an unparseable body is not ours
        return False
    if not isinstance(body, dict):
        return False
    # The hub always wraps failures as {"error": {"code": ..., ...}}.
    err = body.get("error")
    return isinstance(err, dict) and "code" in err


def is_proxy_outage(resp: Any) -> bool:
    """True when a 404 came from something in front of the hub, not the hub.

    Callers use this to decide RETRYABLE (proxy/outage) vs FATAL (hub said the
    route is missing).
    """
    return not response_is_from_hub(resp)
