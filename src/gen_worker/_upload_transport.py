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

This module fixes that with three mechanical changes while preserving the
Tensorhub upload contract: Tensorhub creates a multipart session and returns
presigned part URLs; the worker PUTs bytes directly to those URLs.

  1. **Pool isolation.** Every RETRY attempt allocates a fresh
     ``urllib3.PoolManager(maxsize=1, block=False)`` — guaranteed-new
     TCP+TLS handshake — so a stale half-closed socket never propagates
     to the retry. First attempts within ONE save may share a
     save-scoped ``PutPool`` (issue #385): keepalive across that save's
     parts, torn down with the save, never cross-request. See
     ``PutPool`` for why this scope cannot reproduce the R2 incident.
  2. **Explicit retry classification.** TLS/connection/timeout errors,
     429, and 5xx are retried with exponential backoff + decorrelated
     jitter. 4xx (other than 429) is terminal. The classifier matches
     ``botocore``'s ``StandardRetryHandler`` shape.
  3. **Bounded streaming body.** A ``_BoundedFileReader`` re-opens the
     source file on each attempt so a partial-PUT retry starts from
     the part's true offset, not wherever the previous generator
     stalled.

The control plane (create / complete / abort POSTs to tensorhub) stays
on ``requests``, on a per-save ``requests.Session`` owned by
``presigned_upload.py``.

Public API: ``upload_part_to_presigned_url(url, file_path, offset,
length, pool=None)`` -> ``etag``, plus ``PutPool``. Caller owns the
part-level fan-out (``presigned_upload.py``) and the file-level fan-out
(``_concurrent_upload.py``). This module is a pure transport leaf.
"""

from __future__ import annotations

import logging
import random
import socket
import ssl
import time
from typing import IO, Any, Optional, cast

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

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class PutPool:
    """Save-scoped keepalive pool for presigned part PUTs (issue #385).

    Shared by the FIRST attempt of each part PUT within one save, so
    consecutive parts reuse the R2 TCP+TLS connection instead of paying a
    fresh handshake each (measured from a pod: fresh-conn PUTs 0.8-5.0s,
    keepalive 0.6-0.8s). Retry attempts never touch this pool — they get
    the fresh-per-attempt PoolManager that fixed the R2
    ``SSLV3_ALERT_BAD_RECORD_MAC`` incident.

    Why this scope cannot reproduce that incident: the 2026-05-16 failure
    came from a process-global pool whose sockets sat idle across a
    minutes-long, 64-PUT-in-flight clone until R2's edge half-closed them,
    and whose retries were handed the same poisoned socket. This pool
    (a) lives only for the seconds-scale span of one save and is closed
    with it — never cross-request, (b) idles only for the ms-scale gap
    between one save's parts, and (c) is cleared on any transport failure,
    whose retry then runs on a guaranteed-fresh pool. Worst case a stale
    socket costs one retried attempt — the same recovery path as today.

    Thread-safe (urllib3.PoolManager is); sized for the fixed part fan-out.
    """

    def __init__(self, maxsize: int = 4) -> None:
        self._pool = urllib3.PoolManager(
            num_pools=1,
            maxsize=maxsize,
            block=False,
            retries=False,  # callers own the retry loop
            timeout=urllib3.Timeout(connect=_CONNECT_TIMEOUT_S, read=_READ_TIMEOUT_S),
        )

    def put(self, url: str, *, body: Any, length: int) -> Any:
        # No ``Connection: close`` — keepalive within the save is the point.
        return self._pool.request(
            "PUT",
            url,
            body=body,
            headers={"Content-Length": str(length)},
            preload_content=True,
            decode_content=False,
        )

    def discard_connections(self) -> None:
        """Drop all pooled sockets. Called after any transport failure so a
        possibly-poisoned connection can never serve another part."""
        try:
            self._pool.clear()
        except Exception:
            pass

    def close(self) -> None:
        self.discard_connections()

    def __enter__(self) -> "PutPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


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
    pool: Optional[PutPool] = None,
) -> str:
    """PUT one multipart part to a presigned S3 URL, return the ETag.

    With ``pool`` given, the FIRST attempt goes through that save-scoped
    keepalive pool (issue #385). Every retry attempt — and every attempt
    when ``pool`` is None — allocates its own ``urllib3.PoolManager
    (maxsize=1)``, issues the PUT, and tears the pool down, so a stale
    socket (root cause of the ``SSLV3_ALERT_BAD_RECORD_MAC`` against R2)
    can never propagate into a retry.

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

        use_shared_pool = pool is not None and attempt == 1
        try:
            with _BoundedFileReader(file_path, offset, length) as body:
                if pool is not None and use_shared_pool:
                    resp = pool.put(url, body=body, length=length)
                else:
                    # Fresh pool per attempt. ``maxsize=1`` keeps the pool's
                    # connection set tiny — there's only one in-flight request,
                    # and ``block=False`` means urllib3 won't queue on it. On
                    # context-manager exit the pool's idle connections are
                    # closed, so the TCP socket is unambiguously released
                    # before the next attempt.
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
                            # urllib3 accepts any object with read() (duck
                            # file); its 2.7+ stubs only name IO[Any].
                            body=cast("IO[bytes]", body),
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
            if pool is not None and use_shared_pool:
                # Never let a possibly-poisoned socket serve another part.
                pool.discard_connections()
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
        status_err = _classify_response_status(resp.status, body_sample)
        if status_err is None:
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

        last_err = status_err
        if not status_err.retryable or attempt >= max_attempts:
            raise status_err
        sleep_s = _backoff_sleep_s(attempt)
        logger.info(
            "presigned_part_retry attempt=%d/%d sleep_s=%.2f status=%d",
            attempt, max_attempts, sleep_s, status_err.status_code or 0,
        )
        time.sleep(sleep_s)

    # Exhausted retries without a successful return.
    raise last_err or TransportError("S3 part upload failed (unknown)", retryable=True)


def optimal_part_concurrency(total_parts: int) -> int:
    """Fixed part-level concurrency for one file's multipart upload.

    A single file can saturate R2 with a small number of in-flight PUTs.
    Keep this fixed so it cannot multiply with file-level fan-out into an
    uncontrolled retry storm. The current Tensorhub presigned path also
    has a process-wide PUT budget in ``presigned_upload.py``.
    """
    if total_parts <= 1:
        return 1
    return min(int(total_parts), 4)
