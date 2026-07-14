"""Download failure paths (#373): expired presigned URLs fail in seconds with
ModelEvent{FAILED, url_expired} (the hub re-mints fresh URLs), verify
mismatches get a bounded retry budget, a full disk surfaces as
insufficient_disk immediately, and CAS downloads report byte progress."""

from __future__ import annotations

import asyncio
import errno
import http.server
import threading
from pathlib import Path

import pytest
import requests

import gen_worker.models.cozy_cas as cas_mod
import gen_worker.models.cozy_snapshot as snap_mod
from blake3 import blake3
from gen_worker.capability import InsufficientDiskError
from gen_worker.executor import ModelStore, _is_terminal_download_error
from gen_worker.models.cozy_cas import _download_one_file
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.errors import UrlExpiredError
from gen_worker.models.refs import TensorhubRef
from gen_worker.pb import worker_scheduler_pb2 as pb


def _serve(handler_cls) -> tuple[http.server.ThreadingHTTPServer, str]:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    hits = 0
    payload = b""
    status = 200

    def do_GET(self):  # noqa: N802
        type(self).hits += 1
        self.send_response(self.status)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        if self.payload:
            self.wfile.write(self.payload)

    def log_message(self, *a):
        pass


def test_expired_url_fails_immediately_without_retries(tmp_path: Path) -> None:
    class _Expired(_CountingHandler):
        hits = 0
        status = 403

    httpd, base = _serve(_Expired)
    try:
        with pytest.raises(UrlExpiredError) as ei:
            asyncio.run(_download_one_file(
                f"{base}/blob", tmp_path / "blob", expected_size=4, expected_blake3=""))
        assert ei.value.status_code == 403
        assert _Expired.hits == 1  # dead URL: zero retries
    finally:
        httpd.shutdown()


def test_verify_mismatch_retries_are_bounded(tmp_path: Path) -> None:
    class _Corrupt(_CountingHandler):
        hits = 0
        payload = b"garbage!"

    httpd, base = _serve(_Corrupt)
    try:
        with pytest.raises(ValueError, match="blake3 mismatch"):
            asyncio.run(_download_one_file(
                f"{base}/blob", tmp_path / "blob",
                expected_size=len(_Corrupt.payload),
                expected_blake3=blake3(b"expected-content").hexdigest()))
        assert _Corrupt.hits == cas_mod._VERIFY_MAX_FAILURES  # initial + 2 retries
    finally:
        httpd.shutdown()


def test_enospc_raises_insufficient_disk_immediately(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}

    def _full_disk(*a, **kw):
        calls["n"] += 1
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cas_mod, "_download_one_file_sync", _full_disk)
    with pytest.raises(InsufficientDiskError):
        asyncio.run(_download_one_file(
            "http://example.invalid/blob", tmp_path / "blob",
            expected_size=4, expected_blake3=""))
    assert calls["n"] == 1


def test_terminal_download_error_classification() -> None:
    def _http_error(status: int) -> requests.HTTPError:
        resp = requests.Response()
        resp.status_code = status
        return requests.HTTPError(f"HTTP {status}", response=resp)

    assert _is_terminal_download_error(UrlExpiredError("gone", status_code=403))
    assert _is_terminal_download_error(InsufficientDiskError("full"))
    assert _is_terminal_download_error(_http_error(403))   # exc.response.status_code (#373)
    assert _is_terminal_download_error(_http_error(404))
    assert not _is_terminal_download_error(_http_error(429))
    assert not _is_terminal_download_error(_http_error(408))
    assert not _is_terminal_download_error(_http_error(500))
    assert not _is_terminal_download_error(requests.ConnectionError("reset"))


def _resolved(payload: bytes, url: str) -> dict:
    return {
        "snapshot_digest": "ab" * 32,
        "files": [{
            "path": "model.safetensors",
            "size_bytes": len(payload),
            "blake3": blake3(payload).hexdigest(),
            "url": url,
        }],
    }


def test_snapshot_disk_headroom_check(tmp_path: Path, monkeypatch) -> None:
    class _Usage:
        free = 1  # byte

    monkeypatch.setattr(snap_mod.shutil, "disk_usage", lambda p: _Usage)
    with pytest.raises(InsufficientDiskError):
        asyncio.run(ensure_snapshot_async(
            base_dir=tmp_path, ref=TensorhubRef(owner="e2e", repo="tiny"),
            resolved=_resolved(b"12345", "http://example.invalid/blob")))


def test_snapshot_download_reports_progress(tmp_path: Path) -> None:
    payload = b"x" * 5000

    class _Blob(_CountingHandler):
        hits = 0

    _Blob.payload = payload
    httpd, base = _serve(_Blob)
    seen: list[tuple[int, int | None]] = []
    try:
        out = asyncio.run(ensure_snapshot_async(
            base_dir=tmp_path, ref=TensorhubRef(owner="e2e", repo="tiny"),
            resolved=_resolved(payload, f"{base}/blob"),
            progress=lambda done, total: seen.append((done, total))))
        assert (out / "model.safetensors").read_bytes() == payload
        assert seen and seen[-1] == (len(payload), len(payload))
    finally:
        httpd.shutdown()


def test_model_store_emits_url_expired_within_one_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path))

    class _Expired(_CountingHandler):
        hits = 0
        status = 403

    httpd, base = _serve(_Expired)
    sent: list[pb.WorkerMessage] = []

    async def _emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    snapshot = pb.Snapshot(digest="snap-1", files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=4, blake3="cd" * 32, url=f"{base}/blob")])
    try:
        store = ModelStore(_emit, cache_dir=tmp_path)
        with pytest.raises(UrlExpiredError):
            asyncio.run(store.ensure_local("e2e/tiny", snapshot))
        assert _Expired.hits == 1  # no outer executor retries either
        failed = [m.model_event for m in sent
                  if m.WhichOneof("msg") == "model_event"
                  and m.model_event.state == pb.MODEL_STATE_FAILED]
        assert failed and failed[-1].error == "url_expired"
    finally:
        httpd.shutdown()
