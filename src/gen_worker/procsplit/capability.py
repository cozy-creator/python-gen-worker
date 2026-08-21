"""Parent-side policy on the per-job capability token."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from ..request_context._helpers import _decode_unverified_jwt_claims

MAX_EXPECTED_TTL_S = 6 * 3600
EXPIRY_SKEW_S = 30.0


@dataclass(frozen=True)
class Decision:
    """``forward`` False means: strip the token and refuse the job."""

    forward: bool
    reason: str = ""
    retryable: bool = False
    note: str = ""


FORWARD = Decision(forward=True)


def _claims(token: str) -> Dict[str, Any]:

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
    """Forward, or withhold, one per-job grant."""
    token = (token or "").strip()
    if not token:
        return FORWARD
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
