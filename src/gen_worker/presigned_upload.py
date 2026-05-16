"""Presigned S3 multipart upload client for TensorHub.

Upload flow (one file):
  1. Client computes BLAKE3 hash of the file.
  2. POST {base_url}{endpoint_path} with {path, blake3, size_bytes}
     → either `{dedup: true, ...}` (CAS hit) or `{upload_id, part_urls[],
     part_size, total_parts}` (multipart session opened).
  3. PUT parts directly to S3 via the presigned URLs (tensorhub not in
     the data path).
  4. POST {base_url}{endpoint_path}/{upload_id}/complete with part ETags
     → tensorhub calls S3 CompleteMultipartUpload, runs commit-time
     validation, and returns the finalized blob/snapshot metadata.

Used by worker callers via ctx.save_file / ctx.save_checkpoint /
ctx.open_output_stream. Tensorhub also exposes the same upload protocol to
other authenticated clients; the caller authenticates with either a worker
capability token or a user JWT.
The orchestrator is NOT in the upload path: clients talk directly to
tensorhub, and tensorhub's presigned URLs let bytes go straight to S3.

This is the standard tensorhub multipart-upload client. The same shape
is used at different route prefixes for repo checkpoints
(/api/v1/repos/:owner/:repo/revisions/:revision_id/uploads), datasets
(/api/v1/datasets/:dataset_id/upload-sessions/:session_id/uploads),
endpoint source (/api/v1/endpoints/:owner/:endpoint/releases/uploads),
and user media (/api/v1/media/:owner/uploads).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from blake3 import blake3

from .api.errors import AuthError, CanceledError

logger = logging.getLogger(__name__)

_UPLOAD_TIMEOUT_S = 120
_FINALIZE_TIMEOUT_S = 600
_CREATE_TIMEOUT_S = 60

# Default part size sent by server, but we read it from the response.
_FALLBACK_PART_SIZE = 64 * 1024 * 1024  # 64 MiB
_MAX_PARALLEL_PARTS = 4

# Streaming-from-disk read chunk size used by the bounded part-body reader
# below and by ``blake3_hash_file``. 16 MiB is comfortably above S3's 5 MiB
# minimum multipart part size and gives the kernel a fat-enough syscall to
# amortize read overhead on NVMe / page-cache hits. (Issue #269.)
STREAM_CHUNK_BYTES = 16 * 1024 * 1024


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


class _BoundedFileReader:
    """File-like view of ``[offset, offset+length)`` for a path.

    Passed to ``requests.put(data=...)`` so the HTTP client streams the
    part body from disk in small reads instead of loading the entire
    part into memory. Peak resident bytes per in-flight part is one
    ``STREAM_CHUNK_BYTES`` read buffer plus whatever requests/urllib3
    buffer internally (~32 KiB).

    Not thread-safe — each part-uploader owns its own instance.
    """

    __slots__ = ("_fh", "_remaining", "_length")

    def __init__(self, path: str, offset: int, length: int) -> None:
        # buffering=0 forces unbuffered binary I/O; we manage chunking
        # explicitly so the kernel doesn't double-buffer behind us.
        self._fh = open(path, "rb", buffering=0)
        try:
            self._fh.seek(offset)
        except BaseException:
            self._fh.close()
            raise
        self._length = int(length)
        self._remaining = int(length)

    def __len__(self) -> int:
        # urllib3 looks for __len__ when computing Content-Length on a
        # file-like body. Returning total length keeps the framing
        # consistent across retries.
        return int(self._length)

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size is None or size < 0 or size > self._remaining:
            size = self._remaining
        # Cap each read at STREAM_CHUNK_BYTES so peak per-read RSS stays
        # bounded even if requests/urllib3 asks for a huge slab.
        if size > STREAM_CHUNK_BYTES:
            size = STREAM_CHUNK_BYTES
        data = self._fh.read(size)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "_BoundedFileReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False


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
    retry_attempts: int = 5,
    retry_backoff_ms: int = 500,
    max_parallel: int = _MAX_PARALLEL_PARTS,
    on_progress: Optional[Any] = None,
    cancel_check: Optional[Any] = None,
    complete_extra: Optional[Dict[str, Any]] = None,
) -> PresignedUploadResult:
    """Upload a file to TensorHub via presigned S3 multipart.

    Args:
        file_path: Local path to the file.
        base_url: TensorHub base URL.
        endpoint_path: e.g. "/api/v1/media/:owner/uploads" or "/api/v1/repos/.../uploads".
        headers: Auth headers (Authorization, X-Cozy-Owner).
        create_payload: Additional fields for the create POST (ref, path, request_id, etc.).
        blake3_hex: Pre-computed BLAKE3 hash of the file.
        size_bytes: File size in bytes.
        retry_attempts: Number of retries per part.
        retry_backoff_ms: Backoff between retries.
        max_parallel: Max concurrent part uploads.
        on_progress: Optional callback(parts_done, total_parts, bytes_uploaded).
        cancel_check: Optional callable that returns True if canceled.
        complete_extra: Optional extra fields merged into the /complete POST
            body (after the `parts` array). Used by repo-cas uploads to carry
            lineage metadata — step_number, epoch_number, output_kind,
            target_dtype, flavor, produced_by_kind. Tensorhub persists
            these into checkpoint_lineage.
    """
    url = f"{base_url}{endpoint_path}"

    # --- Step 1: Create presigned upload session ---
    payload = dict(create_payload)
    payload["blake3"] = blake3_hex
    payload["size_bytes"] = size_bytes

    create_headers = dict(headers)
    create_headers["Content-Type"] = "application/json"

    try:
        resp = requests.post(url, headers=create_headers, data=json.dumps(payload), timeout=_CREATE_TIMEOUT_S)
    except requests.RequestException as e:
        raise RuntimeError(f"file save failed (network_error): {e}") from e
    code = resp.status_code
    if code in (401, 403):
        raise AuthError(f"file save unauthorized ({code})")
    if code == 404:
        raise RuntimeError("file save failed (upload_not_supported)")
    if code < 200 or code >= 300:
        raise RuntimeError(f"file save failed ({code}): {resp.text[:200]}")

    parsed = resp.json() if resp.text else {}

    # Handle dedup response.
    if parsed.get("dedup"):
        return PresignedUploadResult(meta=parsed, dedup=True)

    upload_id = str(parsed.get("upload_id") or "").strip()
    if not upload_id:
        raise RuntimeError("file save failed (missing upload_id)")

    part_urls: List[str] = parsed.get("part_urls") or []
    part_size: int = int(parsed.get("part_size") or _FALLBACK_PART_SIZE)
    total_parts: int = int(parsed.get("total_parts") or len(part_urls))

    if not part_urls or total_parts == 0:
        raise RuntimeError("file save failed (no part_urls in response)")

    # --- Step 2: Upload parts to S3 ---
    session_id = upload_id
    abort_url = f"{url}/{session_id}"

    try:
        etags = _upload_parts_to_s3(
            file_path=str(file_path),
            part_urls=part_urls,
            part_size=part_size,
            total_parts=total_parts,
            retry_attempts=retry_attempts,
            retry_backoff_ms=retry_backoff_ms,
            max_parallel=max_parallel,
            on_progress=on_progress,
            cancel_check=cancel_check,
        )
    except BaseException:
        # Abort the multipart upload on failure.
        try:
            abort_headers = dict(headers)
            requests.delete(abort_url, headers=abort_headers, timeout=15)
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
    complete_headers = dict(headers)
    complete_headers["Content-Type"] = "application/json"

    last_exc: Optional[BaseException] = None
    for attempt in range(1, retry_attempts + 1):
        if cancel_check and cancel_check():
            raise CanceledError("canceled")
        try:
            resp = requests.post(
                complete_url,
                headers=complete_headers,
                data=json.dumps(complete_payload),
                timeout=_FINALIZE_TIMEOUT_S,
            )
        except requests.RequestException as e:
            last_exc = RuntimeError(f"file save failed (network_error): {e}")
        else:
            code = resp.status_code
            if code in (401, 403):
                raise AuthError(f"file save unauthorized ({code})")
            if code >= 500:
                last_exc = RuntimeError(f"file save failed ({code})")
            elif code < 200 or code >= 300:
                raise RuntimeError(f"file save failed ({code}): {resp.text[:200]}")
            else:
                result_meta = resp.json() if resp.text else {}
                return PresignedUploadResult(meta=result_meta, dedup=False)
        if attempt < retry_attempts and retry_backoff_ms > 0:
            time.sleep(retry_backoff_ms / 1000.0)

    if last_exc:
        raise last_exc
    raise RuntimeError("file save failed (unknown_error)")


def _upload_parts_to_s3(
    *,
    file_path: str,
    part_urls: List[str],
    part_size: int,
    total_parts: int,
    retry_attempts: int,
    retry_backoff_ms: int,
    max_parallel: int,
    on_progress: Optional[Any],
    cancel_check: Optional[Any],
) -> List[Tuple[int, str]]:
    """Upload file parts to S3 using presigned URLs. Returns list of (part_number, etag)."""
    etags: List[Tuple[int, str]] = []
    file_size = os.path.getsize(file_path)

    def _upload_one_part(part_index: int) -> Tuple[int, str]:
        part_number = part_index + 1
        presigned_url = part_urls[part_index]
        offset = part_index * part_size
        length = min(part_size, file_size - offset)

        last_exc: Optional[BaseException] = None
        for attempt in range(1, retry_attempts + 1):
            if cancel_check and cancel_check():
                raise CanceledError("canceled")
            # Re-open the bounded reader on every attempt so a retry
            # after a partial-PUT failure starts from the part's true
            # offset rather than wherever the prior generator left off.
            #
            # Use a fresh `requests.Session()` per attempt and immediately
            # close it. The module-global session that `requests.put()`
            # uses by default keeps a TLS connection pool alive; under
            # concurrent multipart uploads to R2 (16 parallel parts × 4
            # parallel files = 64 in-flight streams) that pool will hand
            # out a connection R2 has already half-closed, and the next
            # send aborts with `SSLV3_ALERT_BAD_RECORD_MAC` — a TLS-state-
            # corruption symptom that no number of retries can recover
            # from while the same pooled connection is in play. A scoped
            # Session forces a fresh TCP+TLS handshake on each attempt,
            # so a retry actually re-tries instead of replaying the same
            # broken socket. Also disable urllib3's internal retry adapter
            # (max_retries=0) so the outer retry_attempts loop is the
            # only retry surface.
            try:
                with _BoundedFileReader(file_path, offset, length) as body:
                    with requests.Session() as session:
                        adapter = requests.adapters.HTTPAdapter(max_retries=0)
                        session.mount("https://", adapter)
                        session.mount("http://", adapter)
                        resp = session.put(
                            presigned_url,
                            data=body,
                            headers={"Content-Length": str(length), "Connection": "close"},
                            timeout=_UPLOAD_TIMEOUT_S,
                        )
                if resp.status_code < 200 or resp.status_code >= 300:
                    last_exc = RuntimeError(f"S3 part upload failed ({resp.status_code}): {resp.text[:200]}")
                else:
                    etag = resp.headers.get("ETag", "").strip()
                    if not etag:
                        raise RuntimeError("S3 part upload returned no ETag")
                    return (part_number, etag)
            except requests.RequestException as e:
                last_exc = RuntimeError(f"S3 part upload network error: {e}")
            if attempt < retry_attempts and retry_backoff_ms > 0:
                time.sleep(retry_backoff_ms / 1000.0)

        if last_exc:
            raise last_exc
        raise RuntimeError("S3 part upload failed (unknown_error)")

    # Upload parts in parallel.
    workers = min(max_parallel, total_parts)
    parts_done = 0
    if workers <= 1:
        for i in range(total_parts):
            pn, et = _upload_one_part(i)
            etags.append((pn, et))
            parts_done += 1
            if on_progress:
                on_progress(parts_done, total_parts, min((i + 1) * part_size, file_size))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_upload_one_part, i): i for i in range(total_parts)}
            for future in as_completed(futures):
                pn, et = future.result()
                etags.append((pn, et))
                parts_done += 1
                if on_progress:
                    on_progress(parts_done, total_parts, parts_done * part_size)

    # Sort by part number for S3 CompleteMultipartUpload.
    etags.sort(key=lambda x: x[0])
    return etags
