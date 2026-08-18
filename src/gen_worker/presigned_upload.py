"""TensorHub upload client.

Upload flow (one file):
  1. Client computes BLAKE3 hash of the file (and declares sha256 at create —
     so the hub can presign into the final content-addressed key).
  2. POST {base_url}{endpoint_path} with {path, blake3, size_bytes, sha256}.
  3. TensorHub answers dedup, a store-enforced single PUT, or
     presigned multipart part URLs uploaded part-by-part and completed with
     part ETags. The worker NEVER holds store credentials; an expired presign
     is a RE-PLAN, never terminal.

Used by worker callers via ctx.save_file / ctx.save_checkpoint. Tensorhub also
exposes the same upload protocol to other authenticated clients; the caller
authenticates with either a worker capability token or a user JWT. The
orchestrator is NOT in the upload path: clients talk directly to tensorhub, and
bytes go straight to R2/S3.

This is the standard tensorhub upload client. The same control-plane shape is
used at different route prefixes for datasets
(/api/v1/datasets/:dataset_id/upload-sessions/:session_id/uploads),
endpoint source (/api/v1/endpoints/:owner/:endpoint/releases/uploads),
and user media (/api/v1/media/uploads — org-less, the hub derives the org
from the credential). Repo checkpoints do NOT use this client — they publish
via the /commits API (gen_worker.hubio.client).

# HTTP stack

The two planes have DIFFERENT connection scopes, and the boundary is a
ratified safety property — do not blur it:

  * control plane (create / complete / abort, worker -> tensorhub) — one
    PROCESS-scoped ``requests.Session`` per hub origin, reused across saves.
    One fresh TCP+TLS handshake through the tunnel is 109-155 ms, and without
    reuse it is paid on every single save. Auth headers are passed
    per-request, so worker JWT rotation never forces a new connection, and the
    session carries SOCKETS ONLY: cookies are refused so no server state
    crosses saves. Reuse buys a new failure mode — a socket the peer closed
    while the pod was denoising — so a control-plane POST that fails to
    CONNECT on a REUSED session evicts it and retries once on a fresh one. On
    a session this call just built there is no retry: that error is the hub
    being down, not staleness.
  * data plane (part PUTs, worker -> R2) — one ``hubio.transport.PutPool``
    per save, torn down with it, NEVER process-scoped. Retry attempts always
    get a fresh ``urllib3.PoolManager`` — the structural guard against the
    stale-socket ``SSLV3_ALERT_BAD_RECORD_MAC`` R2 edge behaviour (see
    ``hubio.transport``), which is why per-save scoping exists at all. The
    control plane is a different peer with a different failure history.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import os
import threading
import time
import urllib.parse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

from .hubio.transport import (
    STREAM_CHUNK_BYTES,
    PutPool,
    TransportError,
    optimal_part_concurrency,
    upload_part_to_presigned_url,
)
from . import activity as _activity
from . import progress as _progress
from .api.errors import ArtifactTransferError, AuthError, CanceledError
from .stall import SilenceWindow

logger = logging.getLogger(__name__)

# Per-CALL socket budgets on the two hub round trips, not give-up decisions:
# the give-up is `_COMPLETE_SILENCE_WINDOW_S` below, over the hub's own answers.
# Without them a hub that accepts a connection and never replies wedges a publish
# forever. `/complete` gets ten times `/create`'s budget because it VERIFIES the
# object synchronously (streams it back from R2 and hashes it) while `/create`
# only mints presigns — and it is the DERIVATION BASIS for the silence window,
# which is two full finalize-length attempts. Deleting it deletes that basis.
_FINALIZE_TIMEOUT_S = 600
_CREATE_TIMEOUT_S = 60
#: Attempts at `/complete` for a NON-definite answer only (a definite hub
#: refusal is terminal on the first). Bounded because every attempt makes the
#: hub re-verify a multi-GB object; the 409 in-progress case is polled, not
#: retried, so this budget is not spent on the common slow path.
_FINALIZE_RETRY_ATTEMPTS = 5
_FINALIZE_RETRY_BACKOFF_S = 0.5

# tensorhub's /complete verifies the whole object (streams it back from R2 and
# hashes it) synchronously and holds a per-upload lock for the duration; for
# large single files this can run past whatever timeout an intermediary in
# front of tensorhub enforces (~100-120s observed live), so the CLIENT sees a
# transient 5xx/timeout on an attempt that is still running server-side. Our
# own retry then races the first attempt and gets 409
# upload_complete_in_progress — a false negative. Poll on that specific 409
# rather than treating it as fatal: once
# the in-flight attempt finishes, /complete's `sess.Finalized` fast path
# returns the same 200 success payload to the next poll, no data lost.
_COMPLETE_IN_PROGRESS_POLL_S = 5.0

# Deliberately a SILENCE window, not a wall-clock cap: a clock cannot
# distinguish "assembly is taking a while" from "nothing is happening", and
# failing after N minutes discards every byte the worker already uploaded while
# the hub is still stitching a large multipart object.
#
# Each `409 upload_complete_in_progress` is a DEFINITE answer: the hub is up
# and holds the completion lock. The hub sets that lock NX with a TTL and
# never renews it, so a dead holder's lock expires and the next poll takes
# over the verify — a 409 therefore cannot persist without live work behind
# it. Only silence (no HTTP answer at all) accumulates the window, which is
# derived from the call cadence: two full finalize-length attempts.
_COMPLETE_SILENCE_WINDOW_S = 2.0 * _FINALIZE_TIMEOUT_S

# Default part size sent by server, but we read it from the response.
_FALLBACK_PART_SIZE = 64 * 1024 * 1024  # 64 MiB

# THE THREAT THIS COVERS, which nothing else does: an endpoint author calling
# ``ctx.save()`` from their own threads. That is the one axis
# ``optimal_part_concurrency`` cannot see, because it bounds ONE file's parts and
# knows nothing about how many files are in flight beside it. Without this,
# N author threads x 4 parts is unbounded and rebuilds the 100+ PUT retry storm
# that broke R2 mirrors. In-repo callers are sequential, so this is never the
# binding constraint on any in-repo path.
#
# NOT DERIVED. 8 is round. What would change it: one measured author workload
# whose concurrent saves exceed 2 files.
_PRESIGNED_PUT_BUDGET = 8
_presigned_put_slots = threading.BoundedSemaphore(_PRESIGNED_PUT_BUDGET)

__all__ = [
    "STREAM_CHUNK_BYTES",
    "PresignedUploadResult",
    "control_plane_session",
    "presigned_upload_file",
    "reset_control_plane_sessions",
]


# --------------------------------------------------------------------------
# Control-plane keepalive — see the module docstring for the boundary this must
# not cross. One session per HUB ORIGIN, not one global: an eviction then drops
# the poisoned peer's pool and leaves every other peer's connections alone.
# --------------------------------------------------------------------------
#: Matches ``_PRESIGNED_PUT_BUDGET``: an endpoint author saving from N threads
#: can have at most that many control-plane POSTs in flight beside each other.
_CONTROL_POOL_MAXSIZE = _PRESIGNED_PUT_BUDGET
#: A worker talks to exactly one hub. The bound only exists so a caller that
#: rotates base URLs cannot grow this map without limit.
_MAX_CONTROL_ORIGINS = 8

_control_sessions: Dict[str, requests.Session] = {}
_control_sessions_lock = threading.Lock()


def _control_origin(base_url: str) -> str:
    parts = urllib.parse.urlsplit(str(base_url or ""))
    return f"{parts.scheme}://{parts.netloc}"


def _new_control_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=2,
        pool_maxsize=_CONTROL_POOL_MAXSIZE,
        # This module owns retry classification (create: once, on a reused
        # socket only; complete: `_FINALIZE_RETRY_ATTEMPTS`). A second,
        # invisible budget inside urllib3 would replay a POST we decided was
        # terminal.
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # SOCKETS ONLY, never state. A process-scoped session that accumulated
    # cookies would carry one save's server state into the next one — auth is
    # per-request headers and nothing else may persist.
    session.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    return session


def control_plane_session(base_url: str) -> Tuple[requests.Session, bool]:
    """The process-scoped hub session for ``base_url``.

    Returns ``(session, fresh)``. ``fresh`` is True when this call built the
    session — the caller uses it to tell "the hub refused the connection"
    (fresh) apart from "the pooled socket was dead" (reused), which is the
    only difference that justifies a retry.
    """
    origin = _control_origin(base_url)
    with _control_sessions_lock:
        existing = _control_sessions.get(origin)
        if existing is not None:
            return existing, False
        if len(_control_sessions) >= _MAX_CONTROL_ORIGINS:
            _control_sessions.clear()
        session = _new_control_session()
        _control_sessions[origin] = session
        return session, True


def _evict_control_session(base_url: str, session: requests.Session) -> None:
    """Drop a session whose pooled socket proved dead, by IDENTITY.

    Compare-and-swap: a sibling thread may already have replaced it, and
    dropping the replacement would make the next save pay a handshake for
    nothing. The evicted session is NOT closed — another thread may be
    mid-request on it, and its own connections close when the last reference
    goes.
    """
    origin = _control_origin(base_url)
    with _control_sessions_lock:
        if _control_sessions.get(origin) is session:
            _control_sessions.pop(origin, None)


def reset_control_plane_sessions() -> None:
    """Close and forget every control-plane session (tests, teardown)."""
    with _control_sessions_lock:
        sessions = list(_control_sessions.values())
        _control_sessions.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            logger.debug("control-plane session close failed", exc_info=True)


def _is_connection_error(exc: BaseException) -> bool:
    """True for "the socket was dead", false for "the hub answered slowly".

    A timeout is deliberately NOT in here: it means the request may be live on
    the server, and the stale-socket case this covers cannot present as one.
    """
    return isinstance(exc, requests.ConnectionError) and not isinstance(exc, requests.Timeout)


def _response_body_sample(resp: requests.Response, limit: int = 300) -> str:
    try:
        text = str(resp.text or "")
    except Exception:
        text = ""
    text = text.strip()
    return text[:limit]


def _parse_json_response(resp: requests.Response, *, phase: str) -> Dict[str, Any]:
    if not resp.text:
        return {}
    try:
        parsed = resp.json()
    except ValueError as exc:
        raise ArtifactTransferError(
            "tensorhub upload response was not valid JSON",
            provider="tensorhub",
            phase=phase,
            retryable=False,
            status_code=int(resp.status_code),
            cause_type=type(exc).__name__,
        ) from exc
    if not isinstance(parsed, dict):
        raise ArtifactTransferError(
            "tensorhub upload response was not a JSON object",
            provider="tensorhub",
            phase=phase,
            retryable=False,
            status_code=int(resp.status_code),
        )
    return parsed


@contextmanager
def _presigned_put_slot() -> Iterator[None]:
    _presigned_put_slots.acquire()
    try:
        yield
    finally:
        _presigned_put_slots.release()




@contextmanager
def _phase(on_phase: Optional[Any], name: str) -> Iterator[None]:
    """Time one leg of the three-leg protocol and hand it to ``on_phase``.

    ``stage_ms.upload`` is ONE bracket around create -> PUT -> complete, and it
    is 98.6% of a fast request's finalize tail; the split across the three legs
    has to be measured rather than budgeted. The callback fires on the way out of
    the leg, failure included — a leg that raised still cost its time.
    """
    started = time.monotonic()
    try:
        yield
    finally:
        if on_phase is not None:
            try:
                on_phase(name, max(0.0, time.monotonic() - started))
            except Exception:  # a metric may never break an upload
                logger.debug("upload phase callback failed for %s", name, exc_info=True)


class PresignedUploadResult:
    """Result of a presigned upload."""

    __slots__ = ("meta", "dedup")

    def __init__(self, meta: Dict[str, Any], dedup: bool = False):
        self.meta = meta
        self.dedup = dedup


def presigned_upload_file(
    *,
    file_path: str | Path,
    base_url: str,
    endpoint_path: str,
    headers: Dict[str, str],
    create_payload: Dict[str, Any],
    blake3_hex: str,
    size_bytes: int,
    on_progress: Optional[Any] = None,
    cancel_check: Optional[Any] = None,
    complete_extra: Optional[Dict[str, Any]] = None,
    on_phase: Optional[Any] = None,
) -> PresignedUploadResult:
    """Upload a file to TensorHub.

    Args:
        file_path: Local path to the file.
        base_url: TensorHub base URL.
        endpoint_path: e.g. "/api/v1/media/uploads" or "/api/v1/repos/.../uploads".
        headers: Auth headers (Authorization).
        create_payload: Additional fields for the create POST (ref, path, request_id, etc.).
        blake3_hex: Pre-computed BLAKE3 hash of the file.
        size_bytes: File size in bytes.
        on_progress: Optional callback(parts_done, total_parts, bytes_uploaded).
        cancel_check: Optional callable that returns True if canceled.
        on_phase: Optional callback(phase_name, seconds) invoked as each leg
            of the protocol finishes — "create", "put", "complete". Without it
            ``stage_ms.upload`` is one opaque number.
        complete_extra: Optional extra fields merged into the /complete POST
            body (after the `parts` array). NOTE: tensorhub's
            per-file /complete is parts-only and does NOT persist lineage
            metadata; step/epoch/quant identity reach the catalog via the
            commit body's `provenance` object (worker-addable stamp fields),
            not through this seam.
    """
    # two scopes, deliberately different (see the module docstring): the R2
    # PUT pool lives exactly as long as this save and then closes, while the
    # hub control-plane session is process-scoped and survives it.
    #
    # ONE grant-expiry behavior: an expired presign is a RE-PLAN, never
    # terminal. The media create response carries no expires_at, so the
    # discrimination
    # here is behavioral: on a terminal 403 from the PUT leg, re-create the
    # session ONCE (fresh presigns) and re-drive; a substituted-claim 403
    # recurs on the fresh presigns and stays terminal. A duplicate create is
    # bounded and safe for the same reasons _post_create's retry is.
    session, session_is_fresh = control_plane_session(base_url)
    for grant_attempt in (1, 2):
        with PutPool() as put_pool:
            try:
                return _presigned_upload_file_scoped(
                    file_path=file_path,
                    base_url=base_url,
                    endpoint_path=endpoint_path,
                    headers=headers,
                    create_payload=create_payload,
                    blake3_hex=blake3_hex,
                    size_bytes=size_bytes,
                    on_progress=on_progress,
                    cancel_check=cancel_check,
                    complete_extra=complete_extra,
                    on_phase=on_phase,
                    session=session,
                    session_is_fresh=session_is_fresh,
                    put_pool=put_pool,
                )
            except ArtifactTransferError as exc:
                if (grant_attempt == 1 and exc.phase == "put"
                        and getattr(exc, "status_code", None) == 403):
                    logger.warning(
                        "presigned PUT answered 403; re-planning the upload "
                        "session once (expired presign is a re-plan, not a "
                        "failure of the bytes): %s", exc)
                    continue
                raise
    raise AssertionError("unreachable")  # the loop returns or raises


def _post_create(
    *,
    session: requests.Session,
    session_is_fresh: bool,
    base_url: str,
    url: str,
    headers: Dict[str, str],
    body: str,
) -> Tuple[requests.Response, requests.Session]:
    """POST the create leg, retrying ONCE if a REUSED socket was already dead.

    Returns the response and the session it came from — the caller rebinds to
    it so this save's ``/complete`` reuses the connection create just opened
    rather than paying a second handshake behind the eviction.

    **Why this retry is safe, i.e. what a duplicate create costs.** The create
    leg opens an upload session and mints presigns; it writes no media row and
    publishes nothing — the object only exists once ``/complete`` runs, and
    this save completes exactly one of the two sessions. A duplicate is an
    unfinished session the hub GCs, and the direct-final path makes
    it emptier still: the bytes go straight to the content-addressed key the
    digest names, so two sessions for the same save even name the same key.
    The one real charge is the capability grant's budget, which tensorhub
    debits AT CREATE: a duplicate spends 1 of the request's `upload_media`
    `max_count`, minted at 64 against the 1-4 assets a request actually saves.
    That is what bounds this to ONE retry.
    ``/complete`` is a different question and is handled where it lives —
    ``_complete_upload_session`` already retries it, on the hub contract that
    a finalized session answers the same payload again.

    **Why only on a REUSED session.** Process-scoped keepalive is what
    introduced the dead-socket case; a session this call just built has no
    pooled socket to be stale, so a connection error on it is the hub being
    unreachable, and create deliberately has no retry for that case.
    """
    try:
        return session.post(url, headers=headers, data=body, timeout=_CREATE_TIMEOUT_S), session
    except requests.RequestException as exc:
        if session_is_fresh or not _is_connection_error(exc):
            raise
        _evict_control_session(base_url, session)
        logger.info(
            "control-plane keepalive socket was dead on upload create (%s) — "
            "retrying once on a fresh connection",
            type(exc).__name__,
        )
    retry_session, _ = control_plane_session(base_url)
    return retry_session.post(url, headers=headers, data=body, timeout=_CREATE_TIMEOUT_S), retry_session


def _presigned_upload_file_scoped(
    *,
    file_path: str | Path,
    base_url: str,
    endpoint_path: str,
    headers: Dict[str, str],
    create_payload: Dict[str, Any],
    blake3_hex: str,
    size_bytes: int,
    on_progress: Optional[Any],
    cancel_check: Optional[Any],
    complete_extra: Optional[Dict[str, Any]],
    session: requests.Session,
    put_pool: PutPool,
    on_phase: Optional[Any] = None,
    session_is_fresh: bool = True,
) -> PresignedUploadResult:
    url = f"{base_url}{endpoint_path}"

    # --- Step 1: Create presigned upload session ---
    payload = dict(create_payload)
    payload["blake3"] = blake3_hex
    payload["size_bytes"] = size_bytes

    create_headers = dict(headers)
    create_headers["Content-Type"] = "application/json"

    try:
        with _phase(on_phase, "create"):
            resp, session = _post_create(
                session=session,
                session_is_fresh=session_is_fresh,
                base_url=base_url,
                url=url,
                headers=create_headers,
                body=json.dumps(payload),
            )
    except requests.RequestException as e:
        raise ArtifactTransferError(
            f"tensorhub upload create request failed: {e}",
            provider="tensorhub",
            phase="create",
            retryable=True,
            cause_type=type(e).__name__,
        ) from e
    code = resp.status_code
    if code in (401, 403):
        raise AuthError(f"file save unauthorized ({code})")
    if code == 404:
        # A 404 here has two very different causes. The HUB saying 404 means the
        # route does not exist -> fatal. A PROXY saying 404 (ngrok with no
        # healthy backend, i.e. the hub is restarting) means "try again
        # shortly" -> retryable.
        from .http_origin import is_proxy_outage

        if is_proxy_outage(resp):
            raise ArtifactTransferError(
                "tensorhub unreachable during upload create — a proxy answered "
                "404 (backend offline, e.g. hub restarting), not the hub itself",
                provider="tensorhub",
                phase="create",
                retryable=True,
                status_code=code,
            )
        raise ArtifactTransferError(
            "tensorhub upload endpoint is not supported",
            provider="tensorhub",
            phase="create",
            retryable=False,
            status_code=code,
        )
    if code < 200 or code >= 300:
        raise ArtifactTransferError(
            f"tensorhub upload create failed: {_response_body_sample(resp)}",
            provider="tensorhub",
            phase="create",
            retryable=code >= 500 or code == 429,
            status_code=code,
        )

    parsed = _parse_json_response(resp, phase="create")

    # Handle dedup response.
    if parsed.get("dedup"):
        return PresignedUploadResult(meta=parsed, dedup=True)

    upload_id = str(parsed.get("upload_id") or "").strip()
    if not upload_id:
        raise ArtifactTransferError(
            "tensorhub upload create response missing upload_id",
            provider="tensorhub",
            phase="create",
            retryable=False,
        )

    # A store-enforced single PUT straight into the FINAL content-addressed key.
    # The hub can only mint this when the create declared `sha256` — given the
    # digest it knows where the bytes belong, so there is nothing to assemble and
    # nothing to promote, dropping five serialized object-store round trips. The
    # headers carry `x-amz-checksum-sha256` INSIDE the signature: sent verbatim
    # or the store answers 403.
    put_url = str(parsed.get("put_url") or "").strip()
    if put_url:
        put_headers = parsed.get("put_headers") or {}
        with _phase(on_phase, "put"):
            _put_whole_object(
                url=put_url,
                file_path=str(file_path),
                size_bytes=size_bytes,
                extra_headers={str(k): str(v) for k, v in dict(put_headers).items()},
                on_progress=on_progress,
                cancel_check=cancel_check,
                put_pool=put_pool,
            )
        complete_payload = dict(complete_extra or {})
        complete_payload.pop("parts", None)
        with _phase(on_phase, "complete"):
            result_meta = _complete_upload_session(
                complete_url=f"{url}/{upload_id}/complete",
                headers=headers,
                payload=complete_payload,
                cancel_check=cancel_check,
                session=session,
            )
        return PresignedUploadResult(meta=result_meta, dedup=False)

    part_urls: List[str] = parsed.get("part_urls") or []
    part_size: int = int(parsed.get("part_size") or _FALLBACK_PART_SIZE)
    total_parts: int = int(parsed.get("total_parts") or len(part_urls))

    if not part_urls or total_parts == 0:
        raise ArtifactTransferError(
            "tensorhub upload create response missing part URLs",
            provider="tensorhub",
            phase="create",
            retryable=False,
        )

    # --- Step 2: Upload parts to S3 ---
    session_id = upload_id
    abort_url = f"{url}/{session_id}"

    try:
        with _phase(on_phase, "put"):
            etags = _upload_parts_to_s3(
                file_path=str(file_path),
                part_urls=part_urls,
                part_size=part_size,
                total_parts=total_parts,
                on_progress=on_progress,
                cancel_check=cancel_check,
                put_pool=put_pool,
            )
    except BaseException:
        # Abort the multipart upload on failure.
        try:
            abort_headers = dict(headers)
            session.delete(abort_url, headers=abort_headers, timeout=15)
        except Exception:
            pass
        raise

    # --- Step 3: Complete ---
    complete_url = f"{url}/{session_id}/complete"
    complete_payload = {
        "parts": [{"part_number": pn, "etag": et} for pn, et in etags],
    }
    if complete_extra:
        for k, v in complete_extra.items():
            if v is None:
                continue
            # Reserved name — never let caller smuggle in a fake parts list.
            if k == "parts":
                continue
            complete_payload[k] = v
    with _phase(on_phase, "complete"):
        result_meta = _complete_upload_session(
            complete_url=complete_url,
            headers=headers,
            payload=complete_payload,
            cancel_check=cancel_check,
            session=session,
        )
    return PresignedUploadResult(meta=result_meta, dedup=False)


def _error_code_of(resp: requests.Response) -> str:
    """The hub's `error.code`, or "" when the body isn't an envelope.
    pgw#1229: one parser, in ``hub_error``."""
    from .hub_error import hub_error_of

    return hub_error_of(resp).code


def _poll_until_finalized(
    *,
    complete_url: str,
    complete_headers: Dict[str, str],
    payload: Dict[str, Any],
    cancel_check: Optional[Any],
    session: requests.Session,
) -> Dict[str, Any]:
    """A prior /complete attempt is still finalizing server-side (409
    upload_complete_in_progress) — tensorhub verifies large objects
    synchronously and can outlast whatever timeout sits in front of it, so the
    CLIENT'S view (5xx/timeout) can lag the server's. /complete is idempotent
    once finalized (`sess.Finalized` fast path returns the same success
    payload), so re-POST it instead of treating the race as fatal.

    The wait is bounded by SILENCE, never by a clock: a hub that keeps answering
    409 is actively assembling the object we just uploaded, and failing the job
    at that moment throws the whole upload away for nothing."""
    contact = SilenceWindow(_COMPLETE_SILENCE_WINDOW_S)
    while True:
        if cancel_check and cancel_check():
            raise CanceledError("canceled")
        if contact.stalled():
            raise ArtifactTransferError(
                "tensorhub upload finalize: the hub stopped answering while a "
                "completion was in progress (no response for "
                f"{contact.silent_for():.0f}s, window {contact.window_s:.0f}s)",
                provider="tensorhub",
                phase="complete",
                retryable=True,
                status_code=409,
            )
        time.sleep(_COMPLETE_IN_PROGRESS_POLL_S)
        try:
            resp = session.post(
                complete_url, headers=complete_headers, data=json.dumps(payload),
                timeout=_FINALIZE_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            # No answer: the silence window is the only give-up. Drop the
            # keepalive session first if the socket itself died, so the next
            # poll does not replay a dead connection.
            if _is_connection_error(exc):
                _evict_control_session(complete_url, session)
                session, _ = control_plane_session(complete_url)
            continue
        code = resp.status_code
        if code in (401, 403):
            raise AuthError(f"file save unauthorized ({code})")
        if 200 <= code < 300:
            return _parse_json_response(resp, phase="complete")
        if code == 409 and _error_code_of(resp) == "upload_complete_in_progress":
            contact.touch()  # definite answer: assembly is live, keep waiting
            continue
        # Any other terminal error: stop polling, surface it normally.
        raise ArtifactTransferError(
            f"tensorhub upload finalize failed: {_response_body_sample(resp)}",
            provider="tensorhub",
            phase="complete",
            retryable=code >= 500 or code == 429,
            status_code=code,
        )


def _complete_upload_session(
    *,
    complete_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    cancel_check: Optional[Any],
    session: requests.Session,
) -> Dict[str, Any]:
    complete_headers = dict(headers)
    complete_headers["Content-Type"] = "application/json"
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _FINALIZE_RETRY_ATTEMPTS + 1):
        if cancel_check and cancel_check():
            raise CanceledError("canceled")
        try:
            resp = session.post(
                complete_url,
                headers=complete_headers,
                data=json.dumps(payload),
                timeout=_FINALIZE_TIMEOUT_S,
            )
        except requests.RequestException as e:
            last_exc = ArtifactTransferError(
                f"tensorhub upload finalize request failed: {e}",
                provider="tensorhub",
                phase="complete",
                retryable=True,
                cause_type=type(e).__name__,
            )
            # Retrying `/complete` on a connection failure is safe by the hub
            # contract: a finalized session answers the same success payload
            # again. Keepalive adds a pool that can hand the retry the same dead
            # socket, so evict it by identity and take a fresh session.
            if _is_connection_error(e):
                _evict_control_session(complete_url, session)
                session, _ = control_plane_session(complete_url)
        else:
            code = resp.status_code
            if code in (401, 403):
                raise AuthError(f"file save unauthorized ({code})")
            if code == 409 and _error_code_of(resp) == "upload_complete_in_progress":
                return _poll_until_finalized(
                    complete_url=complete_url,
                    complete_headers=complete_headers,
                    payload=payload,
                    cancel_check=cancel_check,
                    session=session,
                )
            if code >= 500:
                last_exc = ArtifactTransferError(
                    f"tensorhub upload finalize failed: {_response_body_sample(resp)}",
                    provider="tensorhub",
                    phase="complete",
                    retryable=True,
                    status_code=code,
                )
            elif code < 200 or code >= 300:
                raise ArtifactTransferError(
                    f"tensorhub upload finalize failed: {_response_body_sample(resp)}",
                    provider="tensorhub",
                    phase="complete",
                    retryable=code == 429,
                    status_code=code,
                )
            else:
                return _parse_json_response(resp, phase="complete")
        if attempt < _FINALIZE_RETRY_ATTEMPTS:
            time.sleep(_FINALIZE_RETRY_BACKOFF_S)

    if last_exc:
        raise last_exc
    raise ArtifactTransferError("tensorhub upload failed", provider="tensorhub", retryable=False)


def _typed_presigned_put(
    *,
    url: str,
    file_path: str,
    offset: int,
    length: int,
    cancel_check: Optional[Any],
    put_pool: Optional[PutPool],
    extra_headers: Optional[Dict[str, str]] = None,
    what: str,
) -> str:
    """THE presigned PUT. One leg, one typed vocabulary — for BOTH legs.

    ``hubio.transport`` speaks ``TransportError``/``InterruptedError``; every
    caller above this module speaks ``ArtifactTransferError``/``CanceledError``,
    and the ``phase == "put"`` + ``status_code == 403`` pair is what
    :func:`presigned_upload_file` re-plans an expired presign on.

    BOTH legs convert, not just the multipart one. The direct-final PUT is the
    leg production always takes (``_stream`` always sends ``sha256``, so the
    hub always mints ``put_url``), and a raw ``TransportError`` there never
    matches the re-plan's ``except ArtifactTransferError`` — an expired presign
    then loses a whole completed render as an untyped ``RuntimeError``.
    """
    try:
        with _presigned_put_slot():
            return upload_part_to_presigned_url(
                url=url,
                file_path=file_path,
                offset=int(offset),
                length=int(length),
                cancel_check=cancel_check,
                pool=put_pool,
                extra_headers=extra_headers,
            )
    except InterruptedError as ie:
        raise CanceledError("canceled") from ie
    except TransportError as te:
        raise ArtifactTransferError(
            f"tensorhub R2 {what} PUT failed: {str(te) or type(te).__name__}",
            provider="tensorhub",
            phase="put",
            retryable=bool(getattr(te, "retryable", False)),
            status_code=getattr(te, "status_code", None),
            cause_type=type(te).__name__,
        ) from te


def _put_whole_object(
    *,
    url: str,
    file_path: str,
    size_bytes: int,
    extra_headers: Dict[str, str],
    on_progress: Optional[Any],
    cancel_check: Optional[Any],
    put_pool: Optional[PutPool],
) -> None:
    """PUT the whole object to one presigned URL (direct-to-final).

    Shares the part transport — the same retry classification, the same
    stale-socket isolation on retry — because a single-shot PUT is a part
    upload with one part and no ETag ceremony, not a new transport.
    """
    _typed_presigned_put(
        url=url,
        file_path=file_path,
        offset=0,
        length=int(size_bytes),
        cancel_check=cancel_check,
        put_pool=put_pool,
        extra_headers=extra_headers,
        what="direct-final",
    )
    if on_progress is not None:
        try:
            on_progress(1, 1, int(size_bytes))
        except Exception:
            logger.debug("upload progress callback failed", exc_info=True)


def _upload_parts_to_s3(
    *,
    file_path: str,
    part_urls: List[str],
    part_size: int,
    total_parts: int,
    on_progress: Optional[Any],
    cancel_check: Optional[Any],
    put_pool: Optional[PutPool] = None,
) -> List[Tuple[int, str]]:
    """Upload file parts to S3 using presigned URLs. Returns list of (part_number, etag).

    Each part PUT is dispatched through ``hubio.transport``, which owns the pool
    lifecycle (save-scoped keepalive pool for first attempts, fresh pool per
    retry), exponential-backoff retry classification, and TLS-pool isolation.
    This function just fans out across parts and aggregates ETags.
    """
    etags: List[Tuple[int, str]] = []
    file_size = os.path.getsize(file_path)

    def _feed(n: int) -> None:
        # Uploaded bytes are visible on the beat while a seal_publish-class
        # activity is open, and proof-of-life either way.
        act = _activity.current()
        if act is not None:
            act.counter("upload:bytes", _progress.UNIT_BYTES).add(n)
        _activity.note_progress()

    def _upload_one_part(part_index: int) -> Tuple[int, str, int]:
        part_number = part_index + 1
        presigned_url = part_urls[part_index]
        offset = part_index * part_size
        length = min(part_size, file_size - offset)
        etag = _typed_presigned_put(
            url=presigned_url,
            file_path=file_path,
            offset=offset,
            length=length,
            cancel_check=cancel_check,
            put_pool=put_pool,
            what="multipart",
        )
        return (part_number, etag, length)

    # Upload parts in parallel.
    workers = min(optimal_part_concurrency(total_parts), total_parts)
    parts_done = 0
    if workers <= 1:
        bytes_uploaded = 0
        for i in range(total_parts):
            pn, et, n = _upload_one_part(i)
            etags.append((pn, et))
            parts_done += 1
            bytes_uploaded += n
            _feed(n)
            if on_progress:
                on_progress(parts_done, total_parts, bytes_uploaded)
    else:
        bytes_uploaded = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gw-part") as pool:
            futures = {pool.submit(_upload_one_part, i): i for i in range(total_parts)}
            for future in as_completed(futures):
                pn, et, n = future.result()
                etags.append((pn, et))
                parts_done += 1
                bytes_uploaded += n
                _feed(n)
                if on_progress:
                    on_progress(parts_done, total_parts, min(bytes_uploaded, file_size))

    # Sort by part number for S3 CompleteMultipartUpload.
    etags.sort(key=lambda x: x[0])
    return etags
