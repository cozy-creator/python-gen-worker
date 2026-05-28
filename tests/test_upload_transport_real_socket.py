"""Real-socket integration for upload_part_to_presigned_url — replaces the
deleted mock-HTTP test_presigned_upload_errors.py (which mocked
requests.post / upload_part and asserted call args).

This drives the REAL transport (real urllib3 PoolManager, real TCP+TLS-less
HTTP, real bounded file read at an offset) against a local ThreadingHTTPServer
that returns scripted statuses. No urllib3 mock, no requests mock — the only
thing patched is time.sleep so the backoff doesn't slow the suite.

Guards the production R2 path end to end:
  * a part PUT lands the exact byte range at the part's offset on the server,
  * 503 -> 200 actually retries against a FRESH connection and returns the ETag,
  * a 403 is terminal (no retry, raises a non-retryable TransportError),
  * a 2xx with no ETag header is a terminal malformed-response error.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import pytest

from gen_worker._upload_transport import TransportError, upload_part_to_presigned_url


class _ScriptedHandler(BaseHTTPRequestHandler):
    # Class-level script: list of (status, etag) consumed in order across PUTs.
    script: List[Tuple[int, Optional[str]]] = []
    received: List[bytes] = []

    def log_message(self, *args: Any) -> None:  # silence the server's stderr noise
        pass

    def do_PUT(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        type(self).received.append(body)
        status, etag = type(self).script.pop(0)
        self.send_response(status)
        if etag is not None:
            self.send_header("ETag", etag)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _serve(script: List[Tuple[int, Optional[str]]]) -> Tuple[ThreadingHTTPServer, str]:
    _ScriptedHandler.script = list(script)
    _ScriptedHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    addr = server.server_address
    host, port = str(addr[0]), int(addr[1])
    return server, f"http://{host}:{port}/part"


def test_part_upload_lands_exact_offset_bytes(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"AAAA" + b"BCDE" + b"FFFF")  # the middle 4 bytes are the part

    server, url = _serve([(200, '"etag-123"')])
    try:
        etag = upload_part_to_presigned_url(
            url=url, file_path=str(src), offset=4, length=4, max_attempts=2,
        )
    finally:
        server.shutdown()

    assert etag == '"etag-123"'
    # The server received exactly the [offset, offset+length) slice.
    assert _ScriptedHandler.received == [b"BCDE"]


def test_part_upload_retries_503_then_succeeds(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"payload-bytes")

    server, url = _serve([(503, None), (200, '"recovered"')])
    try:
        with patch("gen_worker._upload_transport.time.sleep"):  # skip backoff
            etag = upload_part_to_presigned_url(
                url=url, file_path=str(src), offset=0, length=13, max_attempts=5,
            )
    finally:
        server.shutdown()

    assert etag == '"recovered"'
    # The retry re-sent the full part body (fresh BoundedFileReader per attempt).
    assert _ScriptedHandler.received == [b"payload-bytes", b"payload-bytes"]


def test_part_upload_403_is_terminal(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 8)

    server, url = _serve([(403, None), (200, '"never-reached"')])
    try:
        with pytest.raises(TransportError) as ei:
            upload_part_to_presigned_url(
                url=url, file_path=str(src), offset=0, length=8, max_attempts=5,
            )
    finally:
        server.shutdown()

    assert ei.value.status_code == 403
    assert ei.value.retryable is False
    # Exactly one PUT — 4xx is terminal, the second script entry is untouched.
    assert len(_ScriptedHandler.received) == 1


def test_part_upload_200_without_etag_is_terminal(tmp_path: Path) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 8)

    # A real S3-compatible server returning 200 but NO ETag is malformed —
    # the transport treats it as terminal (a successful PUT can't be safely
    # retried for an ETag).
    server, url = _serve([(200, None)])
    try:
        with pytest.raises(TransportError) as ei:
            upload_part_to_presigned_url(
                url=url, file_path=str(src), offset=0, length=8, max_attempts=3,
            )
    finally:
        server.shutdown()

    assert ei.value.retryable is False
    assert "ETag" in str(ei.value)
    assert len(_ScriptedHandler.received) == 1  # no retry on terminal
