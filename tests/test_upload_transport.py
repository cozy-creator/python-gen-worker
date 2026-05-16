"""Tests for the production-grade upload transport (issue #13).

Focused on the retry-classification + backoff-loop logic, NOT on
actually-PUT-against-a-real-S3 (that's covered by the live FLUX clone
verification). The transport-mechanics tests we want to be green here:

  - 5xx retries, eventually fails after max_attempts
  - 4xx (other than 429) fails immediately
  - 429 retries
  - Transport exceptions (TLS, socket) retry
  - 2xx returns the ETag
  - 2xx without ETag header is a hard error (not silently swallowed)
  - cancel_check short-circuits
  - _BoundedFileReader yields exactly [offset, offset+length)
"""

from __future__ import annotations

import socket
import ssl
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from gen_worker._upload_transport import (
    TransportError,
    _BoundedFileReader,
    _backoff_sleep_s,
    _classify_response_status,
    _classify_transport_exception,
    optimal_part_concurrency,
    upload_part_to_presigned_url,
)


def test_bounded_file_reader_reads_exact_range(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"ABCDEFGHIJKLMNOP")  # 16 bytes
    with _BoundedFileReader(str(src), offset=4, length=8) as r:
        assert len(r) == 8
        data = r.read(100)  # ask for more than available
        assert data == b"EFGHIJKL"
        # second read returns empty
        assert r.read(10) == b""


def test_bounded_file_reader_chunked_reads(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"X" * 1024)
    chunks: list[bytes] = []
    with _BoundedFileReader(str(src), offset=0, length=300) as r:
        while True:
            c = r.read(100)
            if not c:
                break
            chunks.append(c)
    assert b"".join(chunks) == b"X" * 300


def test_classify_response_2xx_is_success() -> None:
    assert _classify_response_status(200, "") is None
    assert _classify_response_status(204, "") is None


def test_classify_response_5xx_is_retryable() -> None:
    err = _classify_response_status(500, "internal")
    assert err is not None
    assert err.retryable is True
    assert err.status_code == 500

    err = _classify_response_status(502, "bad gateway")
    assert err is not None and err.retryable is True


def test_classify_response_429_is_retryable() -> None:
    err = _classify_response_status(429, "slow down")
    assert err is not None
    assert err.retryable is True
    assert err.status_code == 429


def test_classify_response_4xx_is_terminal() -> None:
    err = _classify_response_status(403, "forbidden")
    assert err is not None
    assert err.retryable is False
    assert err.status_code == 403

    err = _classify_response_status(404, "not found")
    assert err is not None and err.retryable is False


def test_classify_transport_tls_error_is_retryable() -> None:
    # This is the exact error class that hit production against R2.
    err = _classify_transport_exception(ssl.SSLError("SSLV3_ALERT_BAD_RECORD_MAC"))
    assert err.retryable is True
    assert "tls" in str(err).lower()


def test_classify_transport_timeout_is_retryable() -> None:
    err = _classify_transport_exception(socket.timeout("read timed out"))
    assert err.retryable is True


def test_classify_transport_connection_reset_is_retryable() -> None:
    err = _classify_transport_exception(ConnectionResetError("peer reset"))
    assert err.retryable is True


def test_classify_transport_unknown_is_terminal() -> None:
    # An unexpected exception type should NOT spin forever; surface it.
    class _Weird(Exception):
        pass

    err = _classify_transport_exception(_Weird("???"))
    assert err.retryable is False


def test_backoff_sleep_grows_then_caps() -> None:
    # Lower bound on each attempt is base_s; upper bound grows but is
    # capped at cap_s.
    base, cap = 0.5, 20.0
    for attempt in range(1, 6):
        s = _backoff_sleep_s(attempt, base_s=base, cap_s=cap)
        assert s >= base
        assert s <= cap


def test_backoff_attempt_zero_is_no_sleep() -> None:
    assert _backoff_sleep_s(0) == 0.0


def test_optimal_part_concurrency_single_part_is_one() -> None:
    assert optimal_part_concurrency(1) == 1
    assert optimal_part_concurrency(0) == 1


def test_optimal_part_concurrency_caps_at_four() -> None:
    assert optimal_part_concurrency(2) == 2
    assert optimal_part_concurrency(4) == 4
    assert optimal_part_concurrency(8) == 4
    assert optimal_part_concurrency(16) == 4
    assert optimal_part_concurrency(128) == 4


# --- upload_part_to_presigned_url integration with mocked urllib3 ---


class _FakeResp:
    def __init__(self, status: int, etag: Optional[str] = None, body: bytes = b"") -> None:
        self.status = status
        self.headers = {"ETag": etag} if etag is not None else {}
        self.data = body


class _FakePool:
    """Mimics urllib3.PoolManager's context-manager + .request() interface."""

    def __init__(self, *, behavior: list[Any]) -> None:
        # behavior is a list of either _FakeResp instances or exceptions to raise.
        self._behavior = list(behavior)

    def __enter__(self) -> "_FakePool":
        return self

    def __exit__(self, *a: Any) -> bool:
        return False

    def request(self, *a: Any, **kw: Any) -> _FakeResp:
        step = self._behavior.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _patch_pool_manager(behaviors: list[list[Any]]) -> Any:
    """Return a PoolManager-factory mock that yields one _FakePool per call.

    Each call to ``urllib3.PoolManager(...)`` consumes the next behavior
    list and constructs a fresh _FakePool from it. This simulates the
    'fresh pool per attempt' contract — and lets us assert that the
    transport allocates a new pool on every attempt.
    """
    consumed = {"count": 0}

    def _factory(*args: Any, **kwargs: Any) -> _FakePool:
        idx = consumed["count"]
        consumed["count"] = idx + 1
        if idx >= len(behaviors):
            raise AssertionError(
                f"PoolManager called more times ({idx + 1}) than mocked behaviors ({len(behaviors)})"
            )
        return _FakePool(behavior=behaviors[idx])

    _factory.consumed = consumed  # type: ignore[attr-defined]
    return _factory


def test_upload_part_success_returns_etag(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")

    pool_factory = _patch_pool_manager([[_FakeResp(200, etag='"abc123"')]])
    with patch("gen_worker._upload_transport.urllib3.PoolManager", pool_factory):
        etag = upload_part_to_presigned_url(
            url="https://example.com/part-1",
            file_path=str(src),
            offset=0,
            length=5,
            max_attempts=3,
        )
    assert etag == '"abc123"'
    assert pool_factory.consumed["count"] == 1  # type: ignore[attr-defined]


def test_upload_part_retries_on_5xx(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")

    # Two 503s, then a 200. Each attempt gets its own fresh pool.
    pool_factory = _patch_pool_manager([
        [_FakeResp(503, body=b"unavail")],
        [_FakeResp(503, body=b"unavail")],
        [_FakeResp(200, etag='"ok"')],
    ])
    with patch("gen_worker._upload_transport.urllib3.PoolManager", pool_factory), \
         patch("gen_worker._upload_transport.time.sleep"):  # skip real backoff sleeps
        etag = upload_part_to_presigned_url(
            url="https://example.com/p",
            file_path=str(src),
            offset=0,
            length=5,
            max_attempts=5,
        )
    assert etag == '"ok"'
    assert pool_factory.consumed["count"] == 3  # type: ignore[attr-defined]


def test_upload_part_fails_fast_on_403(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")

    pool_factory = _patch_pool_manager([[_FakeResp(403, body=b"forbidden")]])
    with patch("gen_worker._upload_transport.urllib3.PoolManager", pool_factory):
        with pytest.raises(TransportError) as ei:
            upload_part_to_presigned_url(
                url="https://example.com/p",
                file_path=str(src),
                offset=0,
                length=5,
                max_attempts=5,
            )
    assert ei.value.status_code == 403
    assert ei.value.retryable is False
    # Only one attempt: 4xx is terminal, no retry.
    assert pool_factory.consumed["count"] == 1  # type: ignore[attr-defined]


def test_upload_part_retries_on_tls_error_then_succeeds(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 1024)

    # Simulate the exact R2 production failure on attempt 1.
    pool_factory = _patch_pool_manager([
        [ssl.SSLError("SSLV3_ALERT_BAD_RECORD_MAC")],
        [_FakeResp(200, etag='"recovered"')],
    ])
    with patch("gen_worker._upload_transport.urllib3.PoolManager", pool_factory), \
         patch("gen_worker._upload_transport.time.sleep"):
        etag = upload_part_to_presigned_url(
            url="https://example.com/p",
            file_path=str(src),
            offset=0,
            length=1024,
            max_attempts=5,
        )
    assert etag == '"recovered"'
    # CRITICAL: the second attempt must have used a FRESH PoolManager,
    # not reused the one whose TLS state was poisoned.
    assert pool_factory.consumed["count"] == 2  # type: ignore[attr-defined]


def test_upload_part_exhausts_retries_then_raises(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")

    pool_factory = _patch_pool_manager([
        [_FakeResp(500)],
        [_FakeResp(500)],
        [_FakeResp(500)],
    ])
    with patch("gen_worker._upload_transport.urllib3.PoolManager", pool_factory), \
         patch("gen_worker._upload_transport.time.sleep"):
        with pytest.raises(TransportError) as ei:
            upload_part_to_presigned_url(
                url="https://example.com/p",
                file_path=str(src),
                offset=0,
                length=5,
                max_attempts=3,
            )
    assert ei.value.retryable is True
    assert ei.value.status_code == 500
    assert pool_factory.consumed["count"] == 3  # type: ignore[attr-defined]


def test_upload_part_no_etag_in_200_is_terminal_error(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")

    pool_factory = _patch_pool_manager([[_FakeResp(200, etag=None)]])
    with patch("gen_worker._upload_transport.urllib3.PoolManager", pool_factory):
        with pytest.raises(TransportError) as ei:
            upload_part_to_presigned_url(
                url="https://example.com/p",
                file_path=str(src),
                offset=0,
                length=5,
                max_attempts=3,
            )
    assert "ETag" in str(ei.value)
    assert ei.value.retryable is False


def test_upload_part_cancel_check_short_circuits(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")

    # No PoolManager call should ever happen — cancel_check fires first.
    pool_factory = _patch_pool_manager([])
    with patch("gen_worker._upload_transport.urllib3.PoolManager", pool_factory):
        with pytest.raises(InterruptedError):
            upload_part_to_presigned_url(
                url="https://example.com/p",
                file_path=str(src),
                offset=0,
                length=5,
                max_attempts=3,
                cancel_check=lambda: True,
            )
    assert pool_factory.consumed["count"] == 0  # type: ignore[attr-defined]


def test_upload_part_invalid_max_attempts_raises() -> None:
    with pytest.raises(ValueError):
        upload_part_to_presigned_url(
            url="https://example.com/p",
            file_path="/dev/null",
            offset=0,
            length=0,
            max_attempts=0,
        )
