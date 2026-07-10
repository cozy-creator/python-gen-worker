"""Per-job capability-token renewal (client half of tensorhub #561 / th#639).

While a job is in flight, a background task renews its capability token at
~80% of the token's TTL via ``POST {file_base_url}/v1/worker/capability/renew``
(worker-JWT auth; body ``{request_id, attempt, capability_token}``). The hub
re-mints the SAME grants iff the job is still RUNNING on this worker and the
attempt matches; response is ``{capability_token, expires_at_unix}``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional, Tuple

from .request_context import _decode_unverified_jwt_claims

logger = logging.getLogger(__name__)

RENEW_FRACTION = 0.8
_MIN_SLEEP_S = 1.0
_TRANSIENT_RETRIES = 3
_TRANSIENT_BACKOFF_S = 5.0


class RenewDenied(Exception):
    """Terminal renewal refusal (fencing, job no longer running, auth)."""


def renew_once(
    *,
    file_base_url: str,
    worker_jwt: str,
    request_id: str,
    attempt: int,
    capability_token: str,
) -> Tuple[str, int]:
    """One renewal POST. Returns (new_token, expires_at_unix).

    Raises ``RenewDenied`` on terminal refusals (401/403/404/409) and
    ``RuntimeError`` on transient failures (5xx / malformed response).
    """
    import requests

    url = f"{file_base_url.rstrip('/')}/v1/worker/capability/renew"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {worker_jwt}"},
        json={
            "request_id": request_id,
            "attempt": int(attempt),
            "capability_token": capability_token,
        },
        timeout=30,
    )
    if resp.status_code in (401, 403, 404, 409):
        raise RenewDenied(f"capability renew denied ({resp.status_code}): {resp.text[:256]}")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"capability renew failed ({resp.status_code}): {resp.text[:256]}")
    data = resp.json() if resp.text else {}
    token = str(data.get("capability_token") or "")
    if not token:
        raise RuntimeError("capability renew returned no capability_token")
    return token, int(data.get("expires_at_unix") or 0)


def _renew_at(token: str, *, now: float) -> Optional[float]:
    """Absolute time to renew ``token`` (iat + RENEW_FRACTION·TTL), or None
    when the token carries no usable exp."""
    claims = _decode_unverified_jwt_claims(token)
    try:
        exp = float(claims.get("exp") or 0)
    except (TypeError, ValueError):
        return None
    if exp <= 0:
        return None
    try:
        iat = float(claims.get("iat") or 0)
    except (TypeError, ValueError):
        iat = 0.0
    start = iat if 0 < iat < exp else now
    return start + (exp - start) * RENEW_FRACTION


async def renew_capability_while_running(
    *,
    file_base_url: str,
    request_id: str,
    attempt: int,
    get_worker_jwt: Callable[[], str],
    get_token: Callable[[], str],
    set_token: Callable[[str], None],
) -> None:
    """Background renewal loop for one in-flight job. Cancel when it ends.

    Sleeps until ~80% of the current token's TTL, renews, swaps the stored
    token via ``set_token``, and repeats with the fresh token. Transient
    failures retry a few times with backoff; a terminal denial stops the
    loop loudly (the job keeps its current token and fails on next use).
    """
    while True:
        token = get_token()
        if not token:
            return
        due = _renew_at(token, now=time.time())
        if due is None:
            return  # non-expiring / opaque token: nothing to renew
        await asyncio.sleep(max(_MIN_SLEEP_S, due - time.time()))

        renewed = False
        for i in range(_TRANSIENT_RETRIES):
            try:
                new_token, exp = await asyncio.to_thread(
                    renew_once,
                    file_base_url=file_base_url,
                    worker_jwt=get_worker_jwt(),
                    request_id=request_id,
                    attempt=attempt,
                    capability_token=get_token(),
                )
            except RenewDenied as exc:
                logger.warning("capability renewal for %s stopped: %s", request_id, exc)
                return
            except Exception as exc:
                logger.warning(
                    "capability renewal for %s attempt %d/%d failed: %s",
                    request_id, i + 1, _TRANSIENT_RETRIES, exc,
                )
                await asyncio.sleep(_TRANSIENT_BACKOFF_S * (i + 1))
                continue
            set_token(new_token)
            logger.info("capability token renewed for %s (exp=%d)", request_id, exp)
            renewed = True
            break
        if not renewed:
            logger.error(
                "capability renewal for %s exhausted retries; token will expire",
                request_id,
            )
            return
