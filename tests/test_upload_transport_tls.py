"""Real-TLS integration for the #385 save-scoped connection reuse.

Drives the REAL transport (urllib3 PoolManager / PutPool, requests.Session,
real TCP+TLS handshakes against a local ssl-wrapped ThreadingHTTPServer with a
self-signed cert). The server tags every request with a per-connection serial,
so the tests assert connection reuse/isolation directly:

  * sequential part PUTs sharing one PutPool ride ONE TLS connection,
  * without a pool every PUT pays a fresh handshake (the pre-#385 behavior),
  * a retry NEVER reuses the shared pool's connection (the R2
    SSLV3_ALERT_BAD_RECORD_MAC guard: fresh pool per retry attempt),
  * a full presigned_upload_file save reuses the hub connection across
    create -> complete, and NOTHING is reused across two saves.
"""

from __future__ import annotations

import json
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from gen_worker._upload_transport import PutPool, upload_part_to_presigned_url
from gen_worker.presigned_upload import presigned_upload_file, blake3_hash_file


# --------------------------------------------------------------------------
# Local TLS server with per-connection accounting
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tls_cert(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Path]:
    d = tmp_path_factory.mktemp("tls")
    cert, key = d / "cert.pem", d / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert),
            "-days", "1", "-nodes", "-subj", "/CN=localhost",
            "-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture()
def trust_local_cert(tls_cert: Tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    cert, _ = tls_cert
    # urllib3's default ssl context loads system verify paths (SSL_CERT_FILE);
    # requests reads REQUESTS_CA_BUNDLE per request.
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))


class _CountingHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 (keep-alive capable) handler that tags each request with a
    per-TLS-connection serial in ``server.requests``."""

    protocol_version = "HTTP/1.1"
    conn_id: int = -1

    def setup(self) -> None:  # one call per accepted (TLS) connection
        super().setup()
        srv: Any = self.server
        with srv.lock:
            srv.conn_serial += 1
            self.conn_id = srv.conn_serial

    def log_message(self, *args: Any) -> None:
        pass

    def _record(self) -> None:
        srv: Any = self.server
        with srv.lock:
            srv.requests.append((self.command, self.path, self.conn_id))

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def _respond(self, status: int, body: bytes = b"", headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


class _ScriptedPutHandler(_CountingHandler):
    """PUT handler driven by ``server.script``: entries are (status, etag) or
    the string "drop" (close the connection without responding — simulates the
    half-closed-socket failure mode behind the R2 bad-record-MAC incident)."""

    def do_PUT(self) -> None:
        self._read_body()
        self._record()
        srv: Any = self.server
        with srv.lock:
            entry = srv.script.pop(0) if srv.script else (200, '"etag"')
        if entry == "drop":
            self.close_connection = True
            self.connection.close()
            return
        status, etag = entry
        self._respond(status, headers={"ETag": etag} if etag else None)


class _ProtocolHandler(_CountingHandler):
    """Speaks just enough of the tensorhub presigned-upload protocol for a
    full presigned_upload_file save: create -> part PUT(s) -> complete."""

    def do_POST(self) -> None:
        self._read_body()
        self._record()
        srv: Any = self.server
        if self.path.endswith("/complete"):
            self._respond(200, json.dumps({"published": []}).encode())
            return
        with srv.lock:
            srv.save_serial += 1
            n = srv.save_serial
        body = json.dumps({
            "upload_id": f"u{n}",
            "part_urls": [f"{srv.base_url}/part/{n}"],
            "part_size": srv.part_size,
            "total_parts": 1,
        }).encode()
        self._respond(200, body)

    def do_PUT(self) -> None:
        self._read_body()
        self._record()
        self._respond(200, headers={"ETag": '"etag-put"'})


def _serve_tls(handler: type, tls_cert: Tuple[Path, Path], **attrs: Any) -> Tuple[ThreadingHTTPServer, str]:
    cert, key = tls_cert
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    server.lock = threading.Lock()
    server.conn_serial = 0
    server.requests = []
    for k, v in attrs.items():
        setattr(server, k, v)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[0], server.server_address[1]
    base_url = f"https://{host}:{port}"
    server.base_url = base_url
    return server, base_url


def _put_conn_ids(server: Any) -> List[int]:
    with server.lock:
        return [cid for (method, _p, cid) in server.requests if method == "PUT"]


# --------------------------------------------------------------------------
# Transport-level: PutPool reuse scoping
# --------------------------------------------------------------------------


def test_shared_pool_reuses_one_tls_connection_across_parts(
    tmp_path: Path, tls_cert: Tuple[Path, Path], trust_local_cert: None
) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"0123456789ab")

    server, base = _serve_tls(
        _ScriptedPutHandler, tls_cert, script=[(200, f'"e{i}"') for i in range(3)]
    )
    try:
        with PutPool() as pool:
            etags = [
                upload_part_to_presigned_url(
                    url=f"{base}/part/{i}", file_path=str(src),
                    offset=4 * i, length=4, pool=pool,
                )
                for i in range(3)
            ]
    finally:
        server.shutdown()

    assert etags == ['"e0"', '"e1"', '"e2"']
    conn_ids = _put_conn_ids(server)
    assert len(conn_ids) == 3
    assert len(set(conn_ids)) == 1  # ONE TLS handshake served all three parts


def test_without_pool_every_put_pays_a_fresh_handshake(
    tmp_path: Path, tls_cert: Tuple[Path, Path], trust_local_cert: None
) -> None:
    src = tmp_path / "f.bin"
    src.write_bytes(b"0123456789ab")

    server, base = _serve_tls(
        _ScriptedPutHandler, tls_cert, script=[(200, '"e"')] * 3
    )
    try:
        for i in range(3):
            upload_part_to_presigned_url(
                url=f"{base}/part/{i}", file_path=str(src), offset=4 * i, length=4,
            )
    finally:
        server.shutdown()

    conn_ids = _put_conn_ids(server)
    assert len(conn_ids) == 3
    assert len(set(conn_ids)) == 3  # fresh connection per PUT (pre-#385 behavior)


def test_retry_never_rides_the_shared_pool_connection(
    tmp_path: Path, tls_cert: Tuple[Path, Path], trust_local_cert: None
) -> None:
    """503 on the pooled first attempt -> the retry must run on a fresh pool
    (new TLS connection): the bad-record-MAC guard, unchanged by #385."""
    src = tmp_path / "f.bin"
    src.write_bytes(b"payload!")

    server, base = _serve_tls(
        _ScriptedPutHandler, tls_cert, script=[(503, None), (200, '"recovered"')]
    )
    try:
        with PutPool() as pool, patch("gen_worker._upload_transport.time.sleep"):
            etag = upload_part_to_presigned_url(
                url=f"{base}/part/0", file_path=str(src), offset=0, length=8, pool=pool,
            )
    finally:
        server.shutdown()

    assert etag == '"recovered"'
    conn_ids = _put_conn_ids(server)
    assert len(conn_ids) == 2
    assert conn_ids[0] != conn_ids[1]


def test_dropped_connection_discards_pool_and_retry_recovers(
    tmp_path: Path, tls_cert: Tuple[Path, Path], trust_local_cert: None
) -> None:
    """Server kills the socket mid-request (the stale/half-closed-socket
    failure shape). The pooled attempt fails, the pool discards its
    connections, and the fresh-pool retry succeeds."""
    src = tmp_path / "f.bin"
    src.write_bytes(b"payload!")

    server, base = _serve_tls(
        _ScriptedPutHandler, tls_cert, script=["drop", (200, '"recovered"')]
    )
    try:
        with PutPool() as pool, patch("gen_worker._upload_transport.time.sleep"):
            etag = upload_part_to_presigned_url(
                url=f"{base}/part/0", file_path=str(src), offset=0, length=8, pool=pool,
            )
            # After the failure the shared pool holds no poisoned socket: a
            # subsequent first-attempt PUT opens a NEW connection.
            etag2 = upload_part_to_presigned_url(
                url=f"{base}/part/1", file_path=str(src), offset=0, length=8, pool=pool,
            )
    finally:
        server.shutdown()

    assert etag == '"recovered"'
    assert etag2 == '"etag"'
    conn_ids = _put_conn_ids(server)
    assert len(conn_ids) == 3
    assert conn_ids[0] != conn_ids[1]  # retry on a fresh connection


# --------------------------------------------------------------------------
# Save-level: presigned_upload_file scoping (integration)
# --------------------------------------------------------------------------


def test_save_reuses_hub_connection_and_never_crosses_saves(
    tmp_path: Path, tls_cert: Tuple[Path, Path], trust_local_cert: None
) -> None:
    src = tmp_path / "img.webp"
    src.write_bytes(b"w" * 4096)

    server, base = _serve_tls(
        _ProtocolHandler, tls_cert, save_serial=0, part_size=4096
    )
    try:
        for _ in range(2):
            result = presigned_upload_file(
                file_path=src,
                base_url=base,
                endpoint_path="/api/v1/media/o/uploads",
                headers={"Authorization": "Bearer t"},
                create_payload={"ref": "img.webp"},
                blake3_hex=blake3_hash_file(src),
                size_bytes=4096,
            )
            assert result.meta == {"published": []}
    finally:
        server.shutdown()

    with server.lock:
        reqs = list(server.requests)

    def save_conns(n: int) -> Tuple[int, int, int]:
        # create for save n is the n-th plain /uploads POST
        creates = [c for (m, p, c) in reqs if m == "POST" and not p.endswith("/complete")]
        puts = [c for (m, p, c) in reqs if m == "PUT" and p == f"/part/{n}"]
        completes = [c for (m, p, c) in reqs if m == "POST" and p == f"/api/v1/media/o/uploads/u{n}/complete"]
        assert len(puts) == 1 and len(completes) == 1
        return creates[n - 1], puts[0], completes[0]

    c1, p1, f1 = save_conns(1)
    c2, p2, f2 = save_conns(2)

    # Within a save the hub control plane (create -> complete) rides ONE
    # requests.Session connection.
    assert c1 == f1
    assert c2 == f2
    # Nothing — hub or R2 — is ever reused across saves.
    assert {c1, p1} & {c2, p2} == set()
