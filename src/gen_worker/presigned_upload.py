"""Presigned S3 multipart upload client for TensorHub.

New upload flow:
1. Client computes BLAKE3 hash of file.
2. POST to TensorHub with blake3 + size_bytes -> get presigned S3 part URLs.
3. PUT parts directly to S3 using presigned URLs.
4. POST to TensorHub /complete with part ETags.

This replaces the old PATCH-based streaming upload flow.
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

from .api.errors import AuthError

logger = logging.getLogger(__name__)

_UPLOAD_TIMEOUT_S = 120
_FINALIZE_TIMEOUT_S = 600
_CREATE_TIMEOUT_S = 60

# Default part size sent by server, but we read it from the response.
_FALLBACK_PART_SIZE = 64 * 1024 * 1024  # 64 MiB
_MAX_PARALLEL_PARTS = 4


def blake3_hash_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute BLAKE3 hash of a file without loading it into memory."""
    h = blake3()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def blake3_hash_bytes(data: bytes) -> str:
    """Compute BLAKE3 hash of bytes."""
    h = blake3()
    h.update(data)
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
    retry_attempts: int = 5,
    retry_backoff_ms: int = 500,
    max_parallel: int = _MAX_PARALLEL_PARTS,
    on_progress: Optional[Any] = None,
    cancel_check: Optional[Any] = None,
) -> PresignedUploadResult:
    """Upload a file to TensorHub via presigned S3 multipart.

    Args:
        file_path: Local path to the file.
        base_url: TensorHub base URL.
        endpoint_path: e.g. "/api/v1/media/uploads" or "/api/v1/repos/.../uploads".
        headers: Auth headers (Authorization, X-Cozy-Owner).
        create_payload: Additional fields for the create POST (ref, path, request_id, etc.).
        blake3_hex: Pre-computed BLAKE3 hash of the file.
        size_bytes: File size in bytes.
        retry_attempts: Number of retries per part.
        retry_backoff_ms: Backoff between retries.
        max_parallel: Max concurrent part uploads.
        on_progress: Optional callback(parts_done, total_parts, bytes_uploaded).
        cancel_check: Optional callable that returns True if canceled.
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
        err = RuntimeError(f"file save failed (network_error): {e}")
        err.__cause__ = e
        raise err
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
    complete_payload = {
        "parts": [{"part_number": pn, "etag": et} for pn, et in etags],
    }
    complete_headers = dict(headers)
    complete_headers["Content-Type"] = "application/json"

    last_exc: Optional[BaseException] = None
    for attempt in range(1, retry_attempts + 1):
        if cancel_check and cancel_check():
            raise InterruptedError("canceled")
        try:
            resp = requests.post(
                complete_url,
                headers=complete_headers,
                data=json.dumps(complete_payload),
                timeout=_FINALIZE_TIMEOUT_S,
            )
        except requests.RequestException as e:
            last_exc = RuntimeError(f"file save failed (network_error): {e}")
            last_exc.__cause__ = e
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

        with open(file_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)

        last_exc: Optional[BaseException] = None
        for attempt in range(1, retry_attempts + 1):
            if cancel_check and cancel_check():
                raise InterruptedError("canceled")
            try:
                resp = requests.put(
                    presigned_url,
                    data=data,
                    headers={"Content-Length": str(len(data))},
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
                last_exc.__cause__ = e
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
