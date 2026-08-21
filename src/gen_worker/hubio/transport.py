"""Production-grade HTTP transport for presigned S3 multipart PUTs."""

from __future__ import annotations

import logging
import random
import socket
import ssl
import time
from typing import IO, Any, Mapping, Optional, cast

import urllib3
from urllib3.exceptions import HTTPError, MaxRetryError, ProtocolError, SSLError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

logger = logging.getLogger(__name__)

STREAM_CHUNK_BYTES = 16 * 1024 * 1024

_CONNECT_TIMEOUT_S = 30.0
_READ_TIMEOUT_S = 120.0

_DEFAULT_MAX_ATTEMPTS = 5
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 20.0


class _BoundedFileReader:

    __slots__ = ("_fh", "_remaining", "_length")

    def __init__(self, path: str, offset: int, length: int) -> None:
        self._fh = open(path, "rb", buffering=0)
        try:
            self._fh.seek(offset)
        except BaseException:
            self._fh.close()
            raise
        self._length = int(length)
        self._remaining = int(length)

    def __len__(self) -> int:
        return int(self._length)

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size is None or size < 0 or size > self._remaining:
            size = self._remaining
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
    """Save-scoped keepalive pool for presigned part PUTs (issue #385)."""

    def __init__(self, maxsize: int = 4) -> None:
        self._pool = urllib3.PoolManager(
            num_pools=1,
            maxsize=maxsize,
            block=False,
            retries=False,
            timeout=urllib3.Timeout(connect=_CONNECT_TIMEOUT_S, read=_READ_TIMEOUT_S),
        )

    def put(self, url: str, *, body: Any, length: int,
            extra_headers: Optional[Mapping[str, str]] = None) -> Any:
        headers = {"Content-Length": str(length)}
        if extra_headers:
            headers.update(extra_headers)
        return self._pool.request(
            "PUT",
            url,
            body=body,
            headers=headers,
            preload_content=True,
            decode_content=False,
        )

    def discard_connections(self) -> None:
        """Drop all pooled sockets."""
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


PUT_OK = "ok"
PUT_EXPIRED = "expired"
PUT_TRANSIENT = "transient"
PUT_TERMINAL = "terminal"


def put_verdict(status: int, *, presign_expired: bool = False) -> str:
    """THE status classification for presigned PUTs."""
    if 200 <= status < 300:
        return PUT_OK
    if status == 403 and presign_expired:
        return PUT_EXPIRED
    if status in (408, 429) or status >= 500:
        return PUT_TRANSIENT
    return PUT_TERMINAL


def _classify_response_status(status: int, body_sample: str) -> Optional[TransportError]:
    verdict = put_verdict(status)
    if verdict == PUT_OK:
        return None
    return TransportError(
        f"S3 part upload {'retryable' if verdict == PUT_TRANSIENT else 'terminal'} "
        f"status ({status}): {body_sample[:200]}",
        retryable=verdict == PUT_TRANSIENT,
        status_code=status,
    )


def _classify_transport_exception(exc: BaseException) -> TransportError:
    if isinstance(exc, (Urllib3TimeoutError, socket.timeout, TimeoutError)):
        return TransportError(f"S3 part upload timeout: {exc!r}", retryable=True)
    if isinstance(exc, (SSLError, ssl.SSLError)):
        return TransportError(f"S3 part upload tls error: {exc!r}", retryable=True)
    if isinstance(exc, (ProtocolError, ConnectionError, OSError)):
        return TransportError(f"S3 part upload connection error: {exc!r}", retryable=True)
    if isinstance(exc, MaxRetryError):
        return _classify_transport_exception(exc.reason or exc)
    if isinstance(exc, HTTPError):
        return TransportError(f"S3 part upload http error: {exc!r}", retryable=True)
    return TransportError(f"S3 part upload unexpected error: {exc!r}", retryable=False)


def backoff_sleep_s(attempt: int, base_s: float = _BACKOFF_BASE_S, cap_s: float = _BACKOFF_CAP_S) -> float:
    """Decorrelated-jitter backoff (AWS Architecture Blog: 'Exponential Backoff And Jitter')."""
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
    extra_headers: Optional[Mapping[str, str]] = None,
) -> str:
    """PUT one presigned object (or multipart part) to S3, return the ETag."""
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
                    resp = pool.put(url, body=body, length=length,
                                    extra_headers=extra_headers)
                else:
                    with urllib3.PoolManager(
                        num_pools=1,
                        maxsize=1,
                        block=False,
                        retries=False,
                        timeout=urllib3.Timeout(connect=_CONNECT_TIMEOUT_S, read=_READ_TIMEOUT_S),
                    ) as http:
                        resp = http.request(
                            "PUT",
                            url,
                            body=cast("IO[bytes]", body),
                            headers={
                                "Content-Length": str(length),
                                "Connection": "close",
                                **(dict(extra_headers) if extra_headers else {}),
                            },
                            preload_content=True,
                            decode_content=False,
                        )
        except InterruptedError:
            raise
        except BaseException as exc:
            if pool is not None and use_shared_pool:
                pool.discard_connections()
            err = _classify_transport_exception(exc)
            last_err = err
            if not err.retryable or attempt >= max_attempts:
                raise err
            sleep_s = backoff_sleep_s(attempt)
            logger.info(
                "presigned_part_retry attempt=%d/%d sleep_s=%.2f err=%s",
                attempt, max_attempts, sleep_s, err,
            )
            time.sleep(sleep_s)
            continue

        body_sample = ""
        try:
            body_sample = resp.data.decode("utf-8", errors="replace")
        except Exception:
            body_sample = ""
        status_err = _classify_response_status(resp.status, body_sample)
        if status_err is None:
            etag = ""
            try:
                etag = (resp.headers.get("ETag") or resp.headers.get("etag") or "").strip()
            except Exception:
                etag = ""
            if not etag:
                raise TransportError(
                    "S3 part upload succeeded but returned no ETag header",
                    retryable=False,
                    status_code=resp.status,
                )
            return etag

        last_err = status_err
        if not status_err.retryable or attempt >= max_attempts:
            raise status_err
        sleep_s = backoff_sleep_s(attempt)
        logger.info(
            "presigned_part_retry attempt=%d/%d sleep_s=%.2f status=%d",
            attempt, max_attempts, sleep_s, status_err.status_code or 0,
        )
        time.sleep(sleep_s)

    raise last_err or TransportError("S3 part upload failed (unknown)", retryable=True)


def optimal_part_concurrency(total_parts: int) -> int:
    """Fixed part-level concurrency for one file's multipart upload."""
    if total_parts <= 1:
        return 1
    return min(int(total_parts), 4)
