"""Per-job capability-token renewal (worker half)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional, Tuple

from . import activity as activity_mod
from .procsplit import broker
from .request_context._helpers import _decode_unverified_jwt_claims

logger = logging.getLogger(__name__)

RENEW_PATH = "/v1/worker/capability/renew"
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
    """One renewal POST."""

    # Under the process split this runs in the compute child, which holds no worker JWT — the parent performs the renewal and returns only the new token. Off the split it is the same POST.
    resp = broker.request(
        "POST",
        RENEW_PATH,
        base_url=file_base_url,
        bearer=worker_jwt,
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
    """Background renewal loop for one in-flight job."""
    while True:
        token = get_token()
        if not token:
            return
        due = _renew_at(token, now=time.time())
        if due is None:
            return
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
                activity_mod.emit_event(
                    activity_mod.KIND_CAPABILITY_RENEWAL,
                    f"request={request_id} attempt={attempt}: terminal "
                    f"renewal denial; the job keeps its current token and "
                    f"fails on next use: {exc}",
                    phase="denied",
                )
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
            activity_mod.emit_event(
                activity_mod.KIND_CAPABILITY_RENEWAL,
                f"request={request_id} attempt={attempt}: "
                f"{_TRANSIENT_RETRIES} transient renewal attempts exhausted; "
                "token will expire",
                phase="exhausted",
            )
            return
