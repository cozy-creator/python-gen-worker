from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path

import blake3
import pytest

from gen_worker.request_context._datasets import download_entries

GOOD = b"pgw882 dataset shard bytes" * 64
EVIL = GOOD[:-1] + b"X"
assert len(GOOD) == len(EVIL) and GOOD != EVIL, "the size check must not be what fails these tests"

SHA_GOOD = hashlib.sha256(GOOD).hexdigest()
B3_GOOD = blake3.blake3(GOOD).hexdigest()


class _Server(http.server.BaseHTTPRequestHandler):
    body = GOOD

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *_):
        pass


@pytest.fixture
def serve():
    def _serve(body: bytes) -> str:
        _Server.body = body
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Server)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{srv.server_port}/shard.bin"

    return _serve


def _entry(url: str, checksum: str) -> dict:
    return {"path": "shard.bin", "url": url, "checksum": checksum, "size_bytes": len(GOOD)}


def test_sha256_checksum_rejects_wrong_bytes(serve, tmp_path: Path):
    url = serve(EVIL)
    with pytest.raises(RuntimeError, match="bytes do not match sha256:"):
        download_entries([_entry(url, f"sha256:{SHA_GOOD}")], tmp_path)
    assert not (tmp_path / "shard.bin").exists(), "unverified bytes must not survive"


def test_sha256_checksum_accepts_right_bytes(serve, tmp_path: Path):
    url = serve(GOOD)
    download_entries([_entry(url, f"sha256:{SHA_GOOD}")], tmp_path)
    assert (tmp_path / "shard.bin").read_bytes() == GOOD


def test_blake3_checksum_is_now_refused(serve, tmp_path: Path):
    url = serve(GOOD)
    with pytest.raises(RuntimeError, match="unsupported digest algorithm"):
        download_entries([_entry(url, f"blake3:{B3_GOOD}")], tmp_path)
    assert not (tmp_path / "shard.bin").exists()


def test_untagged_checksum_is_refused_not_guessed(serve, tmp_path: Path):
    """A bare 64-hex names no algorithm — blake3 and sha256 are both 32 bytes."""
    url = serve(GOOD)
    with pytest.raises(RuntimeError, match="untagged"):
        download_entries([_entry(url, SHA_GOOD)], tmp_path)


def test_unknown_algorithm_is_refused(serve, tmp_path: Path):
    url = serve(GOOD)
    with pytest.raises(RuntimeError, match="unsupported digest algorithm"):
        download_entries([_entry(url, f"md5:{'0' * 32}")], tmp_path)


def test_missing_checksum_is_refused(serve, tmp_path: Path):
    url = serve(GOOD)
    entry = _entry(url, "")
    del entry["checksum"]
    with pytest.raises(RuntimeError, match="no checksum"):
        download_entries([entry], tmp_path)
