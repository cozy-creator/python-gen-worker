from __future__ import annotations

import http.server
import socket
import ssl
import threading
from typing import List

import pytest
import urllib3
from urllib3.exceptions import MaxRetryError, ProtocolError, SSLError

from gen_worker.hubio import transport as tr
from gen_worker.hubio.transport import (
    PutPool,
    TransportError,
    _BoundedFileReader,
    backoff_sleep_s,
    upload_part_to_presigned_url,
)


def blob(n: int, seed: int = 5) -> bytes:
    out = bytearray(n)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


class _S3(http.server.BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D102
        pass

    def do_PUT(self):  # noqa: N802
        srv = self.server
        with srv.lock:
            srv.attempts += 1
            attempt = srv.attempts
            plan = srv.plan.pop(0) if srv.plan else 200
        n = int(self.headers.get("Content-Length") or 0)
        if plan == "reset":
            try:
                self.rfile.read(min(n, 16))
                self.connection.close()
            except Exception:
                pass
            return
        body = self.rfile.read(n)
        with srv.lock:
            srv.bodies.append(body)
        if plan == "no_etag":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if isinstance(plan, int) and plan != 200:
            self.send_response(plan)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("ETag", f'"etag-{attempt}"')
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture()
def s3():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _S3)
    srv.attempts, srv.bodies, srv.plan = 0, [], []
    srv.lock = threading.Lock()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


@pytest.fixture()
def no_sleep(monkeypatch):
    calls: List[float] = []
    monkeypatch.setattr(tr.time, "sleep", lambda s: calls.append(float(s)))
    return calls


def test_bounded_reader_serves_exactly_its_span_and_caps_each_read(tmp_path):
    data = blob(1000)
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    with _BoundedFileReader(str(p), 100, 250) as r:
        assert len(r) == 250
        first = r.read(64)
        assert first == data[100:164]
        rest = r.read(-1)
        assert rest == data[164:350]
        assert r.read(10) == b""
    with _BoundedFileReader(str(p), 0, 1000) as r:
        assert len(r.read(1 << 30)) == 1000


def test_a_retry_re_reads_the_part_FROM_ITS_TRUE_OFFSET(s3, tmp_path, no_sleep):
    """The entire reason the reader class exists (`hubio/transport.py:29-32`)."""
    data = blob(4096, seed=7)
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    s3.plan = ["reset"]

    etag = upload_part_to_presigned_url(
        url=f"{s3.base}/part/2", file_path=str(p), offset=1024, length=2048)

    assert etag == '"etag-2"'
    assert s3.bodies == [data[1024:3072]]
    assert len(no_sleep) == 1 and no_sleep[0] > 0


def test_a_5xx_then_success_retries_with_backoff_and_returns_the_etag(
    s3, tmp_path, no_sleep,
):
    data = blob(512, seed=11)
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    s3.plan = [503, 429]

    etag = upload_part_to_presigned_url(
        url=f"{s3.base}/part/1", file_path=str(p), offset=0, length=len(data))

    assert etag == '"etag-3"'
    assert s3.bodies == [data, data, data], "every attempt sends the whole part"
    assert len(no_sleep) == 2 and all(s > 0 for s in no_sleep)


def test_a_4xx_is_terminal_on_the_first_attempt(s3, tmp_path, no_sleep):
    p = tmp_path / "f.bin"
    p.write_bytes(blob(64))
    s3.plan = [403, 200]

    with pytest.raises(TransportError) as err:
        upload_part_to_presigned_url(
            url=f"{s3.base}/part/1", file_path=str(p), offset=0, length=64)

    assert err.value.retryable is False and err.value.status_code == 403
    assert s3.attempts == 1 and no_sleep == []


def test_a_2xx_WITHOUT_an_etag_is_refused_rather_than_retried(s3, tmp_path, no_sleep):
    """An S3-compatible server that answers 200 with no ETag is malformed, and re-PUTting a part that already succeeded is not a fix."""
    p = tmp_path / "f.bin"
    p.write_bytes(blob(64, seed=13))
    s3.plan = ["no_etag", 200]

    with pytest.raises(TransportError) as err:
        upload_part_to_presigned_url(
            url=f"{s3.base}/part/1", file_path=str(p), offset=0, length=64)

    assert "no ETag" in str(err.value)
    assert err.value.retryable is False
    assert s3.attempts == 1 and no_sleep == []


def test_the_retry_budget_is_exhausted_and_the_last_cause_is_reported(
    s3, tmp_path, no_sleep,
):
    p = tmp_path / "f.bin"
    p.write_bytes(blob(64, seed=17))
    s3.plan = [500] * 20

    with pytest.raises(TransportError) as err:
        upload_part_to_presigned_url(
            url=f"{s3.base}/part/1", file_path=str(p), offset=0, length=64,
            max_attempts=3)

    assert err.value.retryable is True and err.value.status_code == 500
    assert s3.attempts == 3
    assert len(no_sleep) == 2


def test_cancel_check_interrupts_before_an_attempt(s3, tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(blob(64))
    with pytest.raises(InterruptedError):
        upload_part_to_presigned_url(
            url=f"{s3.base}/part/1", file_path=str(p), offset=0, length=64,
            cancel_check=lambda: True)
    assert s3.attempts == 0


def test_a_failed_first_attempt_DISCARDS_the_shared_pools_connections(
    s3, tmp_path, no_sleep,
):
    data = blob(256, seed=19)
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    s3.plan = ["reset"]

    pool = PutPool(maxsize=2)
    discards: List[int] = []
    real = pool.discard_connections
    pool.discard_connections = lambda: (discards.append(1), real())[1]  # type: ignore[method-assign]
    used: List[bool] = []
    real_put = pool.put

    def _put(*a, **kw):
        used.append(True)
        return real_put(*a, **kw)

    pool.put = _put  # type: ignore[method-assign]
    try:
        etag = upload_part_to_presigned_url(
            url=f"{s3.base}/part/1", file_path=str(p), offset=0, length=len(data),
            pool=pool)
    finally:
        pool.close()

    assert etag == '"etag-2"'
    assert used == [True], "retries must never go through the shared pool"
    assert discards, "a transport failure must clear the possibly-poisoned socket"
    assert s3.bodies == [data]


def test_every_retry_attempt_allocates_its_own_pool_manager(s3, tmp_path, no_sleep, monkeypatch):
    made: List[int] = []
    real = urllib3.PoolManager

    def _counted(*a, **kw):
        made.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(tr.urllib3, "PoolManager", _counted)
    p = tmp_path / "f.bin"
    p.write_bytes(blob(64, seed=23))
    s3.plan = [503, 503]

    upload_part_to_presigned_url(
        url=f"{s3.base}/part/1", file_path=str(p), offset=0, length=64)

    assert len(made) == 3, "one fresh PoolManager per attempt"


@pytest.mark.parametrize("status,retryable", [
    (200, None), (204, None),
    (429, True), (500, True), (502, True), (503, True),
    (400, False), (403, False), (404, False), (412, False),
])
def test_response_status_classification(status, retryable):
    out = tr._classify_response_status(status, "body")
    if retryable is None:
        assert out is None
    else:
        assert out is not None and out.retryable is retryable
        assert out.status_code == status


@pytest.mark.parametrize("exc,retryable", [
    (socket.timeout("t"), True),
    (TimeoutError("t"), True),
    (ssl.SSLError("bad record mac"), True),
    (SSLError("bad record mac"), True),
    (ProtocolError("closed"), True),
    (ConnectionResetError("reset"), True),
    (OSError("broken pipe"), True),
    (ValueError("nonsense"), False),
])
def test_transport_exception_classification(exc, retryable):
    assert tr._classify_transport_exception(exc).retryable is retryable


def test_max_retry_error_is_unwrapped_to_its_cause():
    wrapped = MaxRetryError(pool=None, url="u", reason=ssl.SSLError("bad record mac"))
    assert tr._classify_transport_exception(wrapped).retryable is True


def test_backoff_is_decorrelated_jitter_bounded_by_the_cap():
    assert backoff_sleep_s(0) == 0.0
    for attempt in range(1, 8):
        samples = [backoff_sleep_s(attempt) for _ in range(200)]
        assert all(tr._BACKOFF_BASE_S <= s <= tr._BACKOFF_CAP_S for s in samples)
    early = [backoff_sleep_s(1) for _ in range(200)]
    late = [backoff_sleep_s(6) for _ in range(200)]
    assert max(early) < tr._BACKOFF_CAP_S
    assert max(late) > max(early)


def test_max_attempts_must_be_positive(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        upload_part_to_presigned_url(
            url="http://127.0.0.1:1/x", file_path=str(p), offset=0, length=1,
            max_attempts=0)
