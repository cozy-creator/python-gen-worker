"""TensorHub upload client: create (declares blake3 + sha256) -> dedup, store-enforced single PUT, or presigned multipart parts -> complete. The worker never holds store credentials; an expired presign is a RE-PLAN, never terminal. Repo checkpoints do NOT use this client (they publish via gen_worker.hubio.client). The two HTTP planes have DIFFERENT connection scopes and the boundary is a ratified safety property — do not blur it: control plane (create/complete/abort -> hub) rides one PROCESS-scoped requests.Session per hub origin, SOCKETS ONLY (cookies refused; auth is per-request headers), evict-and-retry-once when a REUSED socket proves dead; data plane (part PUTs -> R2) rides one per-save PutPool torn down with the save, and retry attempts always get a fresh PoolManager — the structural guard against R2's stale-socket SSLV3_ALERT_BAD_RECORD_MAC edge behaviour."""

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

_FINALIZE_TIMEOUT_S = 600
_CREATE_TIMEOUT_S = 60
_FINALIZE_RETRY_ATTEMPTS = 5
_FINALIZE_RETRY_BACKOFF_S = 0.5

_COMPLETE_IN_PROGRESS_POLL_S = 5.0

_COMPLETE_SILENCE_WINDOW_S = 2.0 * _FINALIZE_TIMEOUT_S

_FALLBACK_PART_SIZE = 64 * 1024 * 1024

_PRESIGNED_PUT_BUDGET = 8
_presigned_put_slots = threading.BoundedSemaphore(_PRESIGNED_PUT_BUDGET)

__all__ = [
    "STREAM_CHUNK_BYTES",
    "PresignedUploadResult",
    "control_plane_session",
    "presigned_upload_file",
    "reset_control_plane_sessions",
]


_CONTROL_POOL_MAXSIZE = _PRESIGNED_PUT_BUDGET
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
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    return session


def control_plane_session(base_url: str) -> Tuple[requests.Session, bool]:
    """The process-scoped hub session for ``base_url``."""
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
    started = time.monotonic()
    try:
        yield
    finally:
        if on_phase is not None:
            try:
                on_phase(name, max(0.0, time.monotonic() - started))
            except Exception:
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
    """Upload a file to TensorHub."""
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
    raise AssertionError("unreachable")


def _post_create(
    *,
    session: requests.Session,
    session_is_fresh: bool,
    base_url: str,
    url: str,
    headers: Dict[str, str],
    body: str,
) -> Tuple[requests.Response, requests.Session]:
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
        try:
            abort_headers = dict(headers)
            session.delete(abort_url, headers=abort_headers, timeout=15)
        except Exception:
            pass
        raise

    complete_url = f"{url}/{session_id}/complete"
    complete_payload = {
        "parts": [{"part_number": pn, "etag": et} for pn, et in etags],
    }
    if complete_extra:
        for k, v in complete_extra.items():
            if v is None:
                continue
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
            contact.touch()
            continue
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
    etags: List[Tuple[int, str]] = []
    file_size = os.path.getsize(file_path)

    def _feed(n: int) -> None:
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

    etags.sort(key=lambda x: x[0])
    return etags
