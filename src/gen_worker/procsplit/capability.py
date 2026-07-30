"""Parent-side policy on the per-job capability token (pgw#763 delta 4).

A per-job, short-TTL, least-authority grant is the ONE credential shape the
compute child is allowed to hold — the enumeration in pgw#763 says so
explicitly, and the child genuinely needs it (inputs it fetches and outputs it
uploads go child -> object store directly, which is what keeps the data plane
out of the parent's interpreter).

That makes this not a leak. It makes it a DECISION, and the parent was not
making it: ``RunJob.worker_capability_token`` was relayed verbatim, so whatever
arrived on the stream was handed to tenant code unexamined.

The parent cannot mint or re-mint — the hub holds the key — so "downscope"
here means the narrowest thing a relay CAN do: forward, or withhold. What it
checks is that the grant it is about to hand to tenant code is scoped to THIS
job on THIS worker and is still alive:

* ``cap_kind`` is a worker capability at all;
* ``request_id`` / ``attempt`` / ``function_name`` name this dispatch;
* ``worker_id`` names this worker (the parent's JWT identity);
* ``exp`` is in the future.

A token that fails any of these is WITHHELD — stripped from the relay and the
job refused typed — rather than forwarded for the child to discover mid-upload.
A confused-deputy grant (one job's token delivered with another job's work) is
the case that matters: it is exactly the shape a hub-side mix-up would take,
and forwarding it would have tenant code acting under authority derived from
somebody else's request.

TTL longer than policy is REPORTED, not refused: only the hub can shorten it,
and refusing legitimate work over it would trade a real outage for a
theoretical exposure. The observation is what lets the hub-side TTL be tuned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# A per-job grant living longer than this is worth naming. Not a refusal: the
# minter chooses the TTL (runtimestore.WorkerCapabilityTokenTTL) and only it
# can shorten one.
MAX_EXPECTED_TTL_S = 6 * 3600
# Clock skew between the hub that minted and the pod that received.
EXPIRY_SKEW_S = 30.0


@dataclass(frozen=True)
class Decision:
    """``forward`` False means: strip the token and refuse the job."""

    forward: bool
    reason: str = ""
    retryable: bool = False
    note: str = ""          # forwarded, but worth recording


FORWARD = Decision(forward=True)


def _claims(token: str) -> Dict[str, Any]:
    from ..request_context import _decode_unverified_jwt_claims

    try:
        claims = _decode_unverified_jwt_claims(token)
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def decide(
    token: str,
    *,
    request_id: str,
    attempt: int,
    function_name: str = "",
    worker_id: str = "",
    now: Optional[float] = None,
) -> Decision:
    """Forward, or withhold, one per-job grant.

    Claims are read UNVERIFIED on purpose: the parent has no signing key and
    does not need one. It is not deciding whether the hub minted the token —
    the hub re-checks its own signature on every use. It is deciding whether
    the grant in front of it belongs to the job it is about to attach it to,
    which is a question the signature cannot answer.
    """
    token = (token or "").strip()
    if not token:
        return FORWARD          # jobs with no file authority legitimately exist
    claims = _claims(token)
    if not claims:
        return Decision(
            forward=False,
            reason="capability token is not a decodable JWT",
        )

    kind = str(claims.get("cap_kind") or "")
    if kind and kind != "worker_capability":
        return Decision(
            forward=False,
            reason=f"capability token is a {kind!r}, not a worker capability",
        )

    claimed_request = str(claims.get("request_id") or "")
    if claimed_request and claimed_request != request_id:
        return Decision(
            forward=False,
            reason=(
                f"capability token is scoped to request {claimed_request} but "
                f"arrived with {request_id} — a grant derived from another "
                "caller's request must never reach handler code"
            ),
        )

    claimed_attempt = _as_int(claims.get("attempt"))
    if claimed_attempt is not None and claimed_attempt != int(attempt):
        return Decision(
            forward=False,
            reason=(
                f"capability token is scoped to attempt {claimed_attempt} but "
                f"arrived with attempt {attempt}"
            ),
        )

    claimed_worker = str(claims.get("worker_id") or "")
    if claimed_worker and worker_id and claimed_worker != worker_id:
        return Decision(
            forward=False,
            reason=(
                f"capability token names worker {claimed_worker}, not this "
                f"worker ({worker_id})"
            ),
        )

    claimed_fn = str(claims.get("function_name") or "")
    if claimed_fn and function_name and claimed_fn != function_name:
        return Decision(
            forward=False,
            reason=(
                f"capability token is scoped to function {claimed_fn}, not "
                f"{function_name}"
            ),
        )

    clock = time.time() if now is None else now
    exp = _as_int(claims.get("exp"))
    if exp is not None and exp > 0:
        if exp <= clock - EXPIRY_SKEW_S:
            return Decision(
                forward=False,
                retryable=True,
                reason=(
                    f"capability token expired {int(clock - exp)}s ago; the job "
                    "could not have uploaded its output"
                ),
            )
        iat = _as_int(claims.get("iat")) or 0
        start = iat if 0 < iat < exp else clock
        ttl = exp - start
        if ttl > MAX_EXPECTED_TTL_S:
            return Decision(
                forward=True,
                note=(
                    f"capability token TTL is {int(ttl)}s (policy expects "
                    f"<= {MAX_EXPECTED_TTL_S}s); only the hub can shorten it"
                ),
            )
    return FORWARD


def scope_of(token: str) -> Tuple[str, int]:
    """(request_id, attempt) a token claims, for renewal cross-checks."""
    claims = _claims(token)
    return str(claims.get("request_id") or ""), int(_as_int(claims.get("attempt")) or 0)


__all__ = [
    "EXPIRY_SKEW_S",
    "FORWARD",
    "MAX_EXPECTED_TTL_S",
    "Decision",
    "decide",
    "scope_of",
]
