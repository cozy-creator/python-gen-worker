"""P4 (th#960/pgw#609 design table): CAS multi-writer integrity — N real OS
processes racing one shared CAS root. th#850: the worker's CAS root can be a
RunPod network volume shared by several same-endpoint pods; the ordinary
write path must be safe when several independent processes race it. Real
multiprocessing throughout (the bug class — a shared, non-writer-unique temp
filename — only manifests across process boundaries, gw#597/598).

Absorbed from tests/test_shared_cas_root_multiwriter.py (PR #339), plus one
NEW row closing the design's "fresh-materialization verifies before trust"
half: a download whose bytes don't match the declared blake3 must never be
silently trusted onto the CAS path.
"""

from __future__ import annotations

import asyncio
import http.server
import multiprocessing
import threading
from pathlib import Path
from typing import Any

import pytest
from blake3 import blake3

import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker.models.cozy_cas import _download_one_file
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef

_PAYLOAD = b"endpoint-volume-cas-root-payload" * 4096  # ~128KB
_BLAKE3 = blake3(_PAYLOAD).hexdigest()
_SNAPSHOT = "b6" * 32
_N_WRITERS = 4


def _resolved() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest=_SNAPSHOT,
        files=[WorkerResolvedRepoFile(
            path="model.safetensors", size_bytes=len(_PAYLOAD),
            blake3=_BLAKE3, url="https://tensorhub.invalid/authorized-blob",
        )],
    )


def _blob(base: Path) -> Path:
    return base / "blobs" / "blake3" / _BLAKE3[:2] / _BLAKE3[2:4] / _BLAKE3


def test_second_pod_on_shared_cas_root_makes_no_network_call(tmp_path: Path, monkeypatch) -> None:
    """Design proof: no separate shared/read-through tier is needed — a
    second "pod" pointed at the SAME base_dir just finds the blob already
    warm (th#850 collapse of the earlier gw#277 shared-tier design)."""
    calls = 0

    async def _public_get(
        _url: str, dst: Path, expected_size: int, expected_blake3: str, on_bytes=None,
    ) -> None:
        del expected_size, expected_blake3
        nonlocal calls
        calls += 1
        dst.write_bytes(_PAYLOAD)
        if on_bytes is not None:
            on_bytes(len(_PAYLOAD))

    monkeypatch.setattr(snap_mod, "_download_one_file", _public_get)
    shared_root = tmp_path / "volume"

    first = asyncio.run(ensure_snapshot_async(
        base_dir=shared_root, ref=TensorhubRef(owner="org", repo="model"), resolved=_resolved(),
    ))
    assert (first / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == 1

    second = asyncio.run(ensure_snapshot_async(
        base_dir=shared_root, ref=TensorhubRef(owner="org", repo="model2"), resolved=_resolved(),
    ))
    assert (second / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == 1  # no second network fetch — the blob was already warm


class _PayloadHandler(http.server.BaseHTTPRequestHandler):
    payload = _PAYLOAD

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_a: Any) -> None:
        pass


def _serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PayloadHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _download_worker(start: Any, results: Any, url: str, dst: str) -> None:
    if not start.wait(10):
        results.put((False, "start barrier timed out"))
        return
    try:
        asyncio.run(_download_one_file(
            url, Path(dst), expected_size=len(_PAYLOAD), expected_blake3=_BLAKE3,
        ))
        results.put((True, ""))
    except BaseException as exc:  # pragma: no cover - surfaced to parent
        results.put((False, repr(exc)))


def test_concurrent_processes_racing_same_missing_blob_do_not_corrupt(tmp_path: Path) -> None:
    httpd, base_url = _serve()
    try:
        dst = tmp_path / "blobs" / "blake3" / _BLAKE3[:2] / _BLAKE3[2:4] / _BLAKE3
        dst.parent.mkdir(parents=True)
        url = f"{base_url}/blob"

        ctx = multiprocessing.get_context("spawn")
        start = ctx.Event()
        results = ctx.Queue()
        processes = [
            ctx.Process(target=_download_worker, args=(start, results, url, str(dst)))
            for _ in range(_N_WRITERS)
        ]
        try:
            for p in processes:
                p.start()
            start.set()
            for p in processes:
                p.join(30)
            assert [p.exitcode for p in processes] == [0] * _N_WRITERS
            outcomes = [results.get(timeout=5) for _ in processes]
        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(5)
            results.close()

        assert outcomes == [(True, "")] * _N_WRITERS
        assert dst.read_bytes() == _PAYLOAD
        assert not list(dst.parent.glob(f".{dst.name}.part-*"))
    finally:
        httpd.shutdown()


def _materialize_worker(start: Any, results: Any, base_dir: str) -> None:
    if not start.wait(10):
        results.put((False, "start barrier timed out"))
        return
    try:
        snap_dir = asyncio.run(ensure_snapshot_async(
            base_dir=Path(base_dir), ref=TensorhubRef(owner="org", repo="model"),
            resolved=_resolved(),
        ))
        content = (snap_dir / "model.safetensors").read_bytes()
        results.put((content == _PAYLOAD, ""))
    except BaseException as exc:  # pragma: no cover - surfaced to parent
        results.put((False, repr(exc)))


def test_concurrent_processes_racing_same_snapshot_materialization(tmp_path: Path) -> None:
    base_dir = tmp_path / "volume"
    blob = _blob(base_dir)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(_PAYLOAD)  # pre-warmed: this race is materialization-only

    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_materialize_worker, args=(start, results, str(base_dir)))
        for _ in range(_N_WRITERS)
    ]
    try:
        for p in processes:
            p.start()
        start.set()
        for p in processes:
            p.join(30)
        assert [p.exitcode for p in processes] == [0] * _N_WRITERS
        outcomes = [results.get(timeout=5) for _ in processes]
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(5)
        results.close()

    assert outcomes == [(True, "")] * _N_WRITERS
    snaps_root = base_dir / "snapshots"
    assert not list(snaps_root.glob(f"{_SNAPSHOT}.building-*"))


def test_fresh_materialization_verifies_blake3_before_trusting_bytes(tmp_path: Path) -> None:
    """A server that answers with WRONG bytes for the declared blake3 must
    fail loud — never leave mismatched content trusted at the CAS path.
    Real HTTP server, real download+verify code path, no mocking."""

    class _WrongBytesHandler(http.server.BaseHTTPRequestHandler):
        wrong_payload = b"\xffcorrupted-on-the-wire" * 100

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.wrong_payload)))
            self.end_headers()
            self.wfile.write(self.wrong_payload)

        def log_message(self, *_a: Any) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _WrongBytesHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        dst = tmp_path / "blobs" / "blake3" / _BLAKE3[:2] / _BLAKE3[2:4] / _BLAKE3
        dst.parent.mkdir(parents=True)
        url = f"http://127.0.0.1:{httpd.server_address[1]}/blob"

        with pytest.raises((ValueError, OSError)):
            asyncio.run(_download_one_file(
                url, dst, expected_size=len(_PAYLOAD), expected_blake3=_BLAKE3,
            ))
        assert not dst.exists(), "mismatched bytes must never land at the trusted CAS path"
    finally:
        httpd.shutdown()
