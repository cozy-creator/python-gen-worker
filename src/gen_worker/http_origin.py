"""Who actually answered — the hub, or a proxy standing in front of it?

A status code is not one condition. It is two, with opposite handling:

  * The HUB answering 404 means "this route/resource does not exist". Fatal.
  * A PROXY answering 404 (ngrok, an ingress, a load balancer with no healthy
    backend) means "the hub is not reachable right now". Transient.

This holds for EVERY status, not just 404 — a proxy 503 carrying an HTML page
is the same condition wearing a different number. The question a caller
actually needs answered is never "was it a 404?" but "did the hub ITSELF
answer?", so that is the question this module exposes
(:func:`is_definite_hub_answer`). Conflating them costs whole GPU runs: a hub
restarting for a few seconds mid-job hands the in-flight upload a proxy page,
and a worker classifying that `retryable=False` kills the job at the last mile.

The two are trivially separable in practice, and we verify it rather than
assume it: the hub answers `application/json` carrying its `{"error": {...}}`
envelope, while ngrok's offline page is `text/html`:

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

__all__ = [
    "response_is_from_hub",
    "is_proxy_outage",
    "is_definite_hub_answer",
    "REQUEST_DETERMINED_STATUSES",
]

# Statuses whose cause is a property of the REQUEST WE SENT, not of who
# answered. The same bytes earn the same refusal from the hub, from a proxy and
# from anything else in the path, forever — so "did the hub itself answer?" is
# not the question, and the retry-biased default below does not hold: retrying
# is pure cost with a guaranteed identical outcome.
#
# 413 only, deliberately. A 404 can be a proxy with no healthy backend, a 403
# can be a token that will be rotated, a 409 can be a race — none of those are
# determined by our own bytes. A Content-Length refusal is: `Content-Length` is
# byte-identical on every retry.
REQUEST_DETERMINED_STATUSES = frozenset({413})


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
    err = body.get("error")
    # Shape 1, the common envelope: {"error": {"code": ..., ...}}.
    if isinstance(err, dict) and "code" in err:
        return True
    # Shape 2, the PUBLISH envelope: {"error": "<code>", "message": ...}.
    # `publishError.body()` (tensorhub `internal/api/repo_publish.go`) emits the
    # code as a STRING, not an object — so every publish refusal was read as
    # proxy-shaped and RETRIED under the silence window. Measured live on a v2
    # publish: a 422 from `/publishes/{id}/complete` was retried, the
    # retry hit the now-terminal session and returned 409 `publish_repudiated`,
    # and the ORIGINAL 422 — the only response that said WHY — was discarded.
    # A retry loop that converts a diagnosable refusal into a different, later
    # error is worse than one that does not retry at all.
    # A proxy does not synthesise a string `error` alongside a `message` either,
    # so this stays a hub-only signature. The hub-side fix is to emit the object
    # envelope everywhere; this must keep working regardless, because old hubs
    # exist and the client cannot require a deploy to classify an answer.
    return isinstance(err, str) and bool(err.strip()) and "message" in body


def is_proxy_outage(resp: Any) -> bool:
    """True when a 404 came from something in front of the hub, not the hub.

    Callers use this to decide RETRYABLE (proxy/outage) vs FATAL (hub said the
    route is missing).
    """
    return not response_is_from_hub(resp)


def is_definite_hub_answer(resp: Any) -> bool:
    """True when this response is the hub's own verdict and may end a loop.

    A retry loop needs exactly one bit: did we HEAR from the hub? Two answers
    are definite:

      * 2xx/3xx — nothing in front of us fabricates a success for a route it
        cannot reach, so the body does not need inspecting.
      * a 4xx carrying the hub's error envelope — the hub refusing.

    Everything else is the peer failing to answer, whatever number rode on
    it: 5xx (an upstream with no healthy backend, an edge timing out over a
    synchronous verify), and any status whose body is proxy-shaped — HTML,
    empty, or JSON without our envelope. Those are NOT verdicts and must
    never be terminal; the caller retries them under a silence window.

    The asymmetry that justifies the conservative default (an unrecognised
    body counts as proxy-shaped): mis-retrying a real refusal costs a bounded
    backoff and then fails anyway, while mis-terminating an outage throws away
    work that has already been paid for.
    """
    try:
        code = int(getattr(resp, "status_code", 0))
    except Exception:  # noqa: BLE001 - an unreadable status is not a verdict
        return False
    if 200 <= code < 400:
        return True
    if code in REQUEST_DETERMINED_STATUSES:
        # A 413 is definite even when the body does not match either envelope
        # shape below: gin's `AbortWithStatusJSON` carries a STRING `error`
        # with no `message`, so without this a paid pod re-sends a body that
        # can never be accepted and reports it as "network".
        return True
    return response_is_from_hub(resp)
