"""pgw#882: a dataset shard download is never accepted on size alone.

Real path, no mocks: a stdlib HTTP server stands in for the presigned URL and
``download_entries`` is driven exactly as ``resolve_dataset`` drives it. What
is under test is the digest decision, so the bytes are real and the server is
real; only the hub is absent (it contributes the manifest, which is the test
input).

The hub has emitted ``sha256:`` checksums since th#1303's dataset write flip
(``datasets_files.go`` — the rows index outright, every post-flip blob by
resolution). Before this fix the reader selected a hasher with
``if expected_digest.startswith("blake3:")`` and left it ``None`` otherwise, so
those downloads were checked on size and nothing else.
"""
from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path

import blake3
import pytest

from gen_worker.request_context._datasets import download_entries

GOOD = b"pgw882 dataset shard bytes" * 64
EVIL = GOOD[:-1] + b"X"  # SAME LENGTH, one byte different
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
    # size_bytes is DELIBERATELY correct in every case: if a test goes green
    # because the size disagreed, it proved nothing about the digest.
    return {"path": "shard.bin", "url": url, "checksum": checksum, "size_bytes": len(GOOD)}


def test_sha256_checksum_rejects_wrong_bytes(serve, tmp_path: Path):
    """THE defect: a sha256 entry served the wrong bytes used to succeed."""
    url = serve(EVIL)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        download_entries([_entry(url, f"sha256:{SHA_GOOD}")], tmp_path)
    assert not (tmp_path / "shard.bin").exists(), "unverified bytes must not survive"


def test_sha256_checksum_accepts_right_bytes(serve, tmp_path: Path):
    url = serve(GOOD)
    download_entries([_entry(url, f"sha256:{SHA_GOOD}")], tmp_path)
    assert (tmp_path / "shard.bin").read_bytes() == GOOD


def test_blake3_checksum_still_rejects_wrong_bytes(serve, tmp_path: Path):
    """The legacy corpus keeps its guard while it exists."""
    url = serve(EVIL)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        download_entries([_entry(url, f"blake3:{B3_GOOD}")], tmp_path)


def test_blake3_checksum_accepts_right_bytes(serve, tmp_path: Path):
    url = serve(GOOD)
    download_entries([_entry(url, f"blake3:{B3_GOOD}")], tmp_path)
    assert (tmp_path / "shard.bin").read_bytes() == GOOD


def test_untagged_checksum_is_refused_not_guessed(serve, tmp_path: Path):
    """A bare 64-hex names no algorithm — blake3 and sha256 are both 32 bytes.

    Guessing ``blake3:`` is what produced the original bug's blind spot, so the
    reader must refuse rather than pick.
    """
    url = serve(GOOD)
    with pytest.raises(RuntimeError, match="untagged"):
        download_entries([_entry(url, SHA_GOOD)], tmp_path)


def test_unknown_algorithm_is_refused(serve, tmp_path: Path):
    url = serve(GOOD)
    with pytest.raises(RuntimeError, match="unsupported digest algorithm"):
        download_entries([_entry(url, f"md5:{'0' * 32}")], tmp_path)


def test_missing_checksum_is_refused(serve, tmp_path: Path):
    """An absent checksum used to mean 'no verification'. It now means refuse."""
    url = serve(GOOD)
    entry = _entry(url, "")
    del entry["checksum"]
    with pytest.raises(RuntimeError, match="no checksum"):
        download_entries([entry], tmp_path)
