"""Production-grade HTTP transport for presigned S3 multipart PUTs (issue #13).

The hand-rolled ``requests.put`` loop this replaces failed in production
against Cloudflare R2 with ``SSLV3_ALERT_BAD_RECORD_MAC`` during the
FLUX.2-klein-4B clone. Root cause: ``requests``'s default
``HTTPAdapter`` keeps a process-global ``urllib3.PoolManager`` whose
connection pool happily hands out TLS sockets that R2's edge has
half-closed. The next write on a stale socket fails the MAC check on
the receiver side — fatal, unrecoverable on the same connection — and
``requests`` cannot tell that error apart from a fresh-handshake
failure, so its retry just replays the broken socket.

This module fixes that with three mechanical changes that boto3's
``s3transfer`` already does by default:

  1. **Per-part pool isolation.** Each multipart PUT gets its own
     ``urllib3.PoolManager(maxsize=1, block=False)``; the pool is
     destroyed after the part completes. A retry on a failed part
     allocates a fresh pool — guaranteed-new TCP+TLS handshake — so a
     stale half-closed socket never propagates to the retry.
  2. **Explicit retry classification.** TLS/connection/timeout errors,
     429, and 5xx are retried with exponential backoff + decorrelated
     jitter. 4xx (other than 429) is terminal. The classifier matches
     ``botocore``'s ``StandardRetryHandler`` shape.
  3. **Bounded streaming body.** A ``_BoundedFileReader`` re-opens the
     source file on each attempt so a partial-PUT retry starts from
     the part's true offset, not wherever the previous generator
     stalled.

The control plane (create / complete / abort POSTs to tensorhub) stays
on ``requests`` because those are small, idempotent JSON calls — the
TLS-pool pathology is specific to the long-lived multi-GiB S3 PUT
path.

Public API: ``upload_part_to_presigned_url(url, file_path, offset,
length)`` -> ``etag``. Caller owns the part-level fan-out
(``presigned_upload.py``) and the file-level fan-out
(``_concurrent_upload.py``). This module is a pure transport leaf.

Phase 1 of issue #13. Phase 2 (boto3 with directly-minted R2
credentials) requires tensorhub-side STS credential issuance and is
tracked separately.
"""

from __future__ import annotations

import logging
import random
import socket
import ssl
import time
from typing import Any, Optional

import urllib3
from urllib3.exceptions import HTTPError, MaxRetryError, ProtocolError, SSLError, TimeoutError as Urllib3TimeoutError

logger = logging.getLogger(__name__)

# Per-part read chunk. 16 MiB matches the value used by the upload-stream
# writer; comfortably above S3's 5 MiB minimum-part-size and gives the
# kernel a fat syscall to amortize read overhead on NVMe / page cache hits.
STREAM_CHUNK_BYTES = 16 * 1024 * 1024

# Socket-level timeouts. ``connect`` covers TCP+TLS handshake.
# ``read`` is per-recv — a 60 s gap with the peer silent fails the part
# and triggers a retry rather than blocking forever.
_CONNECT_TIMEOUT_S = 30.0
_READ_TIMEOUT_S = 120.0

# Retry budget. Five attempts at exponential backoff with full jitter
# tops out around 30 s of total backoff sleep — matches botocore's
# "standard" mode (max_attempts=5, equal_jitter_base=0.5s).
_DEFAULT_MAX_ATTEMPTS = 5
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 20.0


class _BoundedFileReader:
    """File-like view of ``[offset, offset+length)`` for a path.

    Re-opened on every retry attempt — never reused across attempts.
    A retry after a partial-PUT failure starts from the part's true
    offset rather than wherever the prior generator left off.

    Not thread-safe — each part-upload attempt owns its own instance.
    """

    __slots__ = ("_fh", "_remaining", "_length")

    def __init__(self, path: str, offset: int, length: int) -> None:
        # buffering=0 forces unbuffered binary I/O; chunking is managed
        # here so the kernel doesn't double-buffer behind us.
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
        # consistent across reads.
        return int(self._length)

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size is None or size < 0 or size > self._remaining:
            size = self._remaining
        # Cap each read at STREAM_CHUNK_BYTES so peak per-read RSS stays
        # bounded even if urllib3 asks for a huge slab.
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


class TransportError(RuntimeError):
    """Raised when a part upload fails after exhausting retries."""

    def __init__(self, message: str, *, retryable: bool, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status_code = status_code


def _classify_response_status(status: int, body_sample: str) -> Optional[TransportError]:
    """Decide whether an HTTP status from S3 is success / retry / terminal.

    Mirrors botocore's StandardRetryHandler classification: 2xx is
    success, 429 + 5xx are retryable, 4xx (other than 429) are terminal.
    """
    if 200 <= status < 300:
        return None
    if status == 429 or status >= 500:
        return TransportError(
            f"S3 part upload retryable status ({status}): {body_sample[:200]}",
            retryable=True,
            status_code=status,
        )
    return TransportError(
        f"S3 part upload terminal status ({status}): {body_sample[:200]}",
        retryable=False,
        status_code=status,
    )


def _classify_transport_exception(exc: BaseException) -> TransportError:
    """Map a urllib3 / socket / ssl exception to a TransportError.

    All connection-level errors (TLS handshake failure, broken socket,
    truncated response, timeout) are retryable. The SSLV3_ALERT_BAD_RECORD_MAC
    R2 failure lands here as a ``ssl.SSLError`` wrapped in urllib3's
    ``SSLError`` / ``ProtocolError``.
    """
    if isinstance(exc, (Urllib3TimeoutError, socket.timeout, TimeoutError)):
        return TransportError(f"S3 part upload timeout: {exc!r}", retryable=True)
    if isinstance(exc, (SSLError, ssl.SSLError)):
        return TransportError(f"S3 part upload tls error: {exc!r}", retryable=True)
    if isinstance(exc, (ProtocolError, ConnectionError, OSError)):
        return TransportError(f"S3 part upload connection error: {exc!r}", retryable=True)
    if isinstance(exc, MaxRetryError):
        # urllib3 MaxRetryError wraps the underlying cause; unwrap and
        # classify. We disable urllib3's internal retries so this is
        # rare, but defensive in case the default ever changes.
        return _classify_transport_exception(exc.reason or exc)
    if isinstance(exc, HTTPError):
        return TransportError(f"S3 part upload http error: {exc!r}", retryable=True)
    # Unknown exception — treat as non-retryable so we surface it rather
    # than spinning. Matches botocore's "exception not in retry list"
    # default.
    return TransportError(f"S3 part upload unexpected error: {exc!r}", retryable=False)


def _backoff_sleep_s(attempt: int, base_s: float = _BACKOFF_BASE_S, cap_s: float = _BACKOFF_CAP_S) -> float:
    """Decorrelated-jitter backoff (AWS Architecture Blog: 'Exponential Backoff And Jitter').

    Attempt is 1-indexed; attempt=1 returns U[base, base*3], attempt=2
    returns U[base, base*9], etc., capped at cap_s. This is the
    'Decorrelated Jitter' algorithm — better steady-state distribution
    than 'Full Jitter' for many parallel clients hitting the same
    rate-limited endpoint.
    """
    if attempt <= 0:
        return 0.0
    max_window = min(cap_s, base_s * (3 ** attempt))
    return random.uniform(base_s, max_window)


def upload_part_to_presigned_url(
    *,
    url: str,
    file_path: str,
    offset: int,
    length: int,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    cancel_check: Optional[Any] = None,
) -> str:
    """PUT one multipart part to a presigned S3 URL, return the ETag.

    Each attempt allocates its own ``urllib3.PoolManager(maxsize=1)``,
    issues the PUT, and tears the pool down. Stale-socket reuse — the
    root cause of the ``SSLV3_ALERT_BAD_RECORD_MAC`` against R2 — is
    structurally impossible because no connection ever outlives its
    attempt.

    Retries on transport-level failures (TLS, connection reset,
    timeout), 429, and 5xx. Terminal on 4xx (other than 429) and on
    unexpected exception types.

    Raises:
        TransportError: on terminal failure or after exhausting retries.
        InterruptedError: if ``cancel_check()`` returns true mid-attempt.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be >= 1")

    last_err: Optional[TransportError] = None

    for attempt in range(1, max_attempts + 1):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("canceled")

        # Fresh pool per attempt. ``maxsize=1`` keeps the pool's
        # connection set tiny — there's only one in-flight request, and
        # ``block=False`` means urllib3 won't queue on it. On context-
        # manager exit the pool's idle connections are closed, so the
        # TCP socket is unambiguously released before the next attempt.
        try:
            with _BoundedFileReader(file_path, offset, length) as body:
                with urllib3.PoolManager(
                    num_pools=1,
                    maxsize=1,
                    block=False,
                    # ssl_context=None  -> use urllib3's default (which uses
                    # the system's default OpenSSL + CA bundle). No custom
                    # context: the R2 cert chain is publicly trusted.
                    retries=False,  # we own the retry loop
                    timeout=urllib3.Timeout(connect=_CONNECT_TIMEOUT_S, read=_READ_TIMEOUT_S),
                ) as http:
                    resp = http.request(
                        "PUT",
                        url,
                        body=body,
                        headers={
                            "Content-Length": str(length),
                            # Force connection teardown after this single
                            # PUT. Defense in depth — the pool tears down
                            # anyway, but ``Connection: close`` also tells
                            # the server-side TLS terminator to not hold
                            # the socket in its keep-alive table waiting
                            # for a follow-up request.
                            "Connection": "close",
                        },
                        preload_content=True,
                        decode_content=False,
                    )
        except InterruptedError:
            raise
        except BaseException as exc:
            err = _classify_transport_exception(exc)
            last_err = err
            if not err.retryable or attempt >= max_attempts:
                raise err
            sleep_s = _backoff_sleep_s(attempt)
            logger.info(
                "presigned_part_retry attempt=%d/%d sleep_s=%.2f err=%s",
                attempt, max_attempts, sleep_s, err,
            )
            time.sleep(sleep_s)
            continue

        # Got a complete HTTP response — classify status.
        body_sample = ""
        try:
            body_sample = resp.data.decode("utf-8", errors="replace")
        except Exception:
            body_sample = ""
        err = _classify_response_status(resp.status, body_sample)
        if err is None:
            etag = ""
            try:
                # urllib3 normalizes header names case-insensitively.
                etag = (resp.headers.get("ETag") or resp.headers.get("etag") or "").strip()
            except Exception:
                etag = ""
            if not etag:
                # An S3-compatible server that returns 2xx without an
                # ETag is malformed. Treat as terminal — retrying a
                # successful PUT against the same presigned URL is not
                # safe (S3 allows multiple PUTs to the same part, but
                # we'd still get no ETag back).
                raise TransportError(
                    "S3 part upload succeeded but returned no ETag header",
                    retryable=False,
                    status_code=resp.status,
                )
            return etag

        last_err = err
        if not err.retryable or attempt >= max_attempts:
            raise err
        sleep_s = _backoff_sleep_s(attempt)
        logger.info(
            "presigned_part_retry attempt=%d/%d sleep_s=%.2f status=%d",
            attempt, max_attempts, sleep_s, err.status_code or 0,
        )
        time.sleep(sleep_s)

    # Exhausted retries without a successful return.
    raise last_err or TransportError("S3 part upload failed (unknown)", retryable=True)


def optimal_part_concurrency(total_parts: int) -> int:
    """Adaptive part-level concurrency for one file's multipart upload.

    Replaces the old hardcoded ``_MAX_PARALLEL_PARTS = 4``. Rationale:
    a single file with N parts can saturate the wire with a handful of
    in-flight PUTs; pushing higher just adds TLS-pool churn (the same
    pool churn that triggered the R2 SSLV3_ALERT_BAD_RECORD_MAC in the
    first place). The right bound depends on:

      - total_parts itself (don't spin up 16 workers for a 4-part file)
      - the file-level fan-out above (handled by the caller — file-level
        × part-level should not exceed ~16 in-flight PUTs total).

    Sized at ``min(total_parts, 8)``. Above 8 is empirically diminishing
    returns on a single NIC, and the per-part PoolManager allocation
    becomes the bottleneck.
    """
    if total_parts <= 1:
        return 1
    return min(int(total_parts), 8)
