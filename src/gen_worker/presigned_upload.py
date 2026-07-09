"""TensorHub upload client.

Upload flow (one file):
  1. Client computes BLAKE3 hash of the file.
  2. POST {base_url}{endpoint_path} with {path, blake3, size_bytes}.
  3. For repo/model uploads, TensorHub returns a scoped R2/S3 transfer grant;
     the worker uploads through boto3/s3transfer and completes with transfer
     metadata.
  4. Older non-model platform uploads may still return presigned multipart
     URLs; those are uploaded part-by-part and completed with part ETags.

Used by worker callers via ctx.save_file / ctx.save_checkpoint /
ctx.open_output_stream. Tensorhub also exposes the same upload protocol to
other authenticated clients; the caller authenticates with either a worker
capability token or a user JWT. The orchestrator is NOT in the upload path:
clients talk directly to tensorhub, and bytes go straight to R2/S3.

This is the standard tensorhub upload client. The same control-plane shape is
used at different route prefixes for repo checkpoints
(/api/v1/repos/:owner/:repo/revisions/:revision_id/uploads), datasets
(/api/v1/datasets/:dataset_id/upload-sessions/:session_id/uploads),
endpoint source (/api/v1/endpoints/:owner/:endpoint/releases/uploads),
and user media (/api/v1/media/:owner/uploads).

# HTTP stack (issues #13 / #385)

Connections are scoped to ONE save (``presigned_upload_file`` call) and
torn down with it — never shared across saves:

  * control plane (create / complete / abort) — one ``requests.Session``
    per save, so create -> complete reuses the tensorhub connection
    instead of paying a fresh TCP+TLS handshake per POST (bare
    ``requests.post`` builds a new Session per call). Auth headers are
    passed per-request, so worker JWT rotation never forces a new
    connection.
  * data plane (part PUTs) — one ``_upload_transport.PutPool`` per save,
    shared by first attempts so consecutive parts reuse the R2
    connection. Retry attempts always get a fresh ``urllib3.PoolManager``
    — the structural guard against the stale-socket
    ``SSLV3_ALERT_BAD_RECORD_MAC`` R2 incident (see ``_upload_transport``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from blake3 import blake3

from ._upload_transport import (
    STREAM_CHUNK_BYTES,
    PutPool,
    TransportError,
    optimal_part_concurrency,
    upload_part_to_presigned_url,
)
from .api.errors import ArtifactTransferError, AuthError, CanceledError

logger = logging.getLogger(__name__)

_FINALIZE_TIMEOUT_S = 600
_CREATE_TIMEOUT_S = 60
_FINALIZE_RETRY_ATTEMPTS = 5
_FINALIZE_RETRY_BACKOFF_S = 0.5

# tensorhub's /complete verifies the whole object (streams it back from R2 and
# hashes it) synchronously and holds a per-upload lock for the duration; for
# large single files this can run past whatever timeout an intermediary in
# front of tensorhub enforces (~100-120s observed live), so the CLIENT sees a
# transient 5xx/timeout on an attempt that is still running server-side. Our
# own retry then races the first attempt and gets 409 upload_complete_in_progress
# — a false negative (found live: e2e tracker #110, a ~6.94GB singlefile
# mirror). Poll on that specific 409 rather than treating it as fatal: once
# the in-flight attempt finishes, /complete's `sess.Finalized` fast path
# returns the same 200 success payload to the next poll, no data lost.
_COMPLETE_IN_PROGRESS_POLL_S = 5.0
_COMPLETE_IN_PROGRESS_MAX_WAIT_S = 600.0

# Default part size sent by server, but we read it from the response.
_FALLBACK_PART_SIZE = 64 * 1024 * 1024  # 64 MiB

# Hard-coded internal safety budget for the current Tensorhub presigned
# upload path. File-level fan-out is fixed at 4 and per-file part fan-out is
# fixed at 4, so this semaphore is the authoritative cap that keeps the two
# axes from multiplying. Eight concurrent PUTs preserves useful parallelism
# while avoiding the 100+ PUT retry storm that broke R2 mirrors.
_PRESIGNED_PUT_BUDGET = 8
_presigned_put_slots = threading.BoundedSemaphore(_PRESIGNED_PUT_BUDGET)

__all__ = [
    "STREAM_CHUNK_BYTES",
    "PresignedUploadResult",
    "blake3_hash_file",
    "presigned_upload_file",
]


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


def _is_tensorhub_model_weight_upload(endpoint_path: str) -> bool:
    path = str(endpoint_path or "")
    return "/api/v1/repos/" in path and "/uploads" in path


@contextmanager
def _presigned_put_slot() -> Iterator[None]:
    _presigned_put_slots.acquire()
    try:
        yield
    finally:
        _presigned_put_slots.release()


def blake3_hash_file(path: str | Path, chunk_size: int = STREAM_CHUNK_BYTES) -> str:
    """Compute BLAKE3 hash of a file without loading it into memory.

    Fans BLAKE3 internals across available CPU cores via
    ``max_threads=blake3.AUTO`` — on a 16-core host this is ~5-8× the
    single-threaded throughput. (Issue #269.)
    """
    h = blake3(max_threads=blake3.AUTO)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
) -> PresignedUploadResult:
    """Upload a file to TensorHub.

    Args:
        file_path: Local path to the file.
        base_url: TensorHub base URL.
        endpoint_path: e.g. "/api/v1/media/:owner/uploads" or "/api/v1/repos/.../uploads".
        headers: Auth headers (Authorization, X-Cozy-Owner).
        create_payload: Additional fields for the create POST (ref, path, request_id, etc.).
        blake3_hex: Pre-computed BLAKE3 hash of the file.
        size_bytes: File size in bytes.
        on_progress: Optional callback(parts_done, total_parts, bytes_uploaded).
        cancel_check: Optional callable that returns True if canceled.
        complete_extra: Optional extra fields merged into the /complete POST
            body (after the `parts` array). NOTE (gw#401/th#606): tensorhub's
            per-file /complete is parts-only and does NOT persist lineage
            metadata; step/epoch/quant identity reach the catalog via the
            commit body's `provenance` object (worker-addable stamp fields),
            not through this seam.
    """
    # Per-save connection scope (issue #385): one hub Session + one R2 PUT
    # pool live exactly as long as this save, then close. Never cross-request.
    with requests.Session() as session, PutPool() as put_pool:
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
            session=session,
            put_pool=put_pool,
        )


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
) -> PresignedUploadResult:
    url = f"{base_url}{endpoint_path}"

    # --- Step 1: Create presigned upload session ---
    payload = dict(create_payload)
    payload["blake3"] = blake3_hex
    payload["size_bytes"] = size_bytes

    create_headers = dict(headers)
    create_headers["Content-Type"] = "application/json"

    try:
        resp = session.post(url, headers=create_headers, data=json.dumps(payload), timeout=_CREATE_TIMEOUT_S)
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

    transfer_grant = parsed.get("transfer_grant") or parsed.get("s3_transfer_grant")
    if isinstance(transfer_grant, dict):
        from .s3_transfer import S3TransferGrant, upload_file_with_grant

        grant = S3TransferGrant.from_mapping(transfer_grant)
        sdk_result = upload_file_with_grant(
            file_path=file_path,
            grant=grant,
            blake3_hex=blake3_hex,
            size_bytes=size_bytes,
            on_progress=on_progress,
        )
        complete_payload: Dict[str, Any] = {
            "transfer": {
                "mode": "s3_sdk",
                "bucket": sdk_result.bucket,
                "key": sdk_result.key,
                "size_bytes": sdk_result.size_bytes,
                "blake3": sdk_result.blake3,
                "etag": sdk_result.etag,
            }
        }
        if complete_extra:
            for k, v in complete_extra.items():
                if v is not None and k != "transfer":
                    complete_payload[k] = v
        result_meta = _complete_upload_session(
            complete_url=f"{url}/{upload_id}/complete",
            headers=headers,
            payload=complete_payload,
            cancel_check=cancel_check,
            session=session,
        )
        return PresignedUploadResult(meta=result_meta, dedup=False)

    if _is_tensorhub_model_weight_upload(endpoint_path):
        raise ArtifactTransferError(
            "tensorhub model upload response missing transfer_grant",
            provider="tensorhub",
            phase="create",
            retryable=False,
        )

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
    complete_payload: Dict[str, Any] = {
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
    result_meta = _complete_upload_session(
        complete_url=complete_url,
        headers=headers,
        payload=complete_payload,
        cancel_check=cancel_check,
        session=session,
    )
    return PresignedUploadResult(meta=result_meta, dedup=False)


def _error_code_of(resp: requests.Response) -> str:
    """Best-effort extraction of the structured `error.code` field
    (docs/api-conventions.md: `{"error": {"code": ..., ...}}`); "" if the
    body isn't that shape."""
    try:
        body = resp.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if not isinstance(err, dict):
        return ""
    return str(err.get("code") or "")


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
    payload), so re-POST it instead of treating the race as fatal."""
    deadline = time.monotonic() + _COMPLETE_IN_PROGRESS_MAX_WAIT_S
    while True:
        if cancel_check and cancel_check():
            raise CanceledError("canceled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ArtifactTransferError(
                "tensorhub upload finalize: gave up waiting for a concurrent "
                "completion to finish (upload_complete_in_progress persisted "
                f"past {_COMPLETE_IN_PROGRESS_MAX_WAIT_S:.0f}s)",
                provider="tensorhub",
                phase="complete",
                retryable=True,
                status_code=409,
            )
        time.sleep(min(_COMPLETE_IN_PROGRESS_POLL_S, remaining))
        try:
            resp = session.post(
                complete_url, headers=complete_headers, data=json.dumps(payload),
                timeout=_FINALIZE_TIMEOUT_S,
            )
        except requests.RequestException:
            continue  # transient — keep polling within the deadline
        code = resp.status_code
        if code in (401, 403):
            raise AuthError(f"file save unauthorized ({code})")
        if 200 <= code < 300:
            return _parse_json_response(resp, phase="complete")
        if code == 409 and _error_code_of(resp) == "upload_complete_in_progress":
            continue  # still racing the first attempt
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

    Each part PUT is dispatched through ``_upload_transport`` which
    owns the pool lifecycle (save-scoped keepalive pool for first
    attempts, fresh pool per retry), exponential-backoff retry
    classification, and TLS-pool isolation. This function just fans
    out across parts and aggregates ETags.
    """
    etags: List[Tuple[int, str]] = []
    file_size = os.path.getsize(file_path)

    def _upload_one_part(part_index: int) -> Tuple[int, str, int]:
        part_number = part_index + 1
        presigned_url = part_urls[part_index]
        offset = part_index * part_size
        length = min(part_size, file_size - offset)
        try:
            with _presigned_put_slot():
                etag = upload_part_to_presigned_url(
                    url=presigned_url,
                    file_path=file_path,
                    offset=offset,
                    length=length,
                    cancel_check=cancel_check,
                    pool=put_pool,
                )
        except InterruptedError as ie:
            raise CanceledError("canceled") from ie
        except TransportError as te:
            raise ArtifactTransferError(
                f"tensorhub R2 multipart PUT failed: {str(te) or type(te).__name__}",
                provider="tensorhub",
                phase="put",
                retryable=bool(getattr(te, "retryable", False)),
                status_code=getattr(te, "status_code", None),
                cause_type=type(te).__name__,
            ) from te
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
                if on_progress:
                    on_progress(parts_done, total_parts, min(bytes_uploaded, file_size))

    # Sort by part number for S3 CompleteMultipartUpload.
    etags.sort(key=lambda x: x[0])
    return etags
