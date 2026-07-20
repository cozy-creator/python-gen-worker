"""th#850: the worker's CAS root can now be a RunPod network volume shared by
several same-endpoint pods concurrently, instead of always being pod-local
disk. There is no separate shared/read-through tier (gw PR #277's design was
superseded) — pointing ``TENSORHUB_CACHE_DIR`` at the mount IS the whole
mechanism, so the ordinary CAS write path must itself be safe when several
independent OS processes race on it. These tests prove that with REAL
multiprocessing (spawn), not mocked concurrency, because the bug class this
guards against (a shared, non-writer-unique temp filename) only manifests
across process boundaries.
"""

from __future__ import annotations

import asyncio
import http.server
import multiprocessing
import threading
from pathlib import Path
from typing import Any

from blake3 import blake3

import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker.models.cozy_cas import _download_one_file
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef

_PAYLOAD = b"endpoint-volume-cas-root-payload" * 4096  # ~128KB
_BLAKE3 = blake3(_PAYLOAD).hexdigest()
_SNAPSHOT = "b6" * 32


def _resolved() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest=_SNAPSHOT,
        files=[
            WorkerResolvedRepoFile(
                path="model.safetensors",
                size_bytes=len(_PAYLOAD),
                blake3=_BLAKE3,
                url="https://tensorhub.invalid/authorized-blob",
            )
        ],
    )


def _blob(base: Path) -> Path:
    return base / "blobs" / "blake3" / _BLAKE3[:2] / _BLAKE3[2:4] / _BLAKE3


# ---------------------------------------------------------------------------
# Design proof: CAS-root-on-volume needs no separate tier — a second "pod"
# pointed at the SAME base_dir just finds the blob already there.
# ---------------------------------------------------------------------------

def test_second_pod_on_shared_cas_root_makes_no_network_call(
    tmp_path: Path, monkeypatch,
) -> None:
    calls = 0

    async def _public_get(
        _url: str, dst: Path, expected_size: int, expected_blake3: str,
        on_bytes=None,
    ) -> None:
        del expected_size, expected_blake3
        nonlocal calls
        calls += 1
        dst.write_bytes(_PAYLOAD)
        if on_bytes is not None:
            on_bytes(len(_PAYLOAD))

    monkeypatch.setattr(snap_mod, "_download_one_file", _public_get)
    ref = TensorhubRef(owner="org", repo="model")
    shared_root = tmp_path / "volume"  # what TENSORHUB_CACHE_DIR/cas resolves to

    first = asyncio.run(ensure_snapshot_async(
        base_dir=shared_root, ref=ref, resolved=_resolved(),
    ))
    assert (first / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == 1

    # Second "pod" boots against the exact same CAS root (the volume): a
    # fresh ref with a different snapshot key still hits the warmed blob.
    ref2 = TensorhubRef(owner="org", repo="model2")
    second = asyncio.run(ensure_snapshot_async(
        base_dir=shared_root, ref=ref2, resolved=_resolved(),
    ))
    assert (second / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == 1  # no second network fetch — the blob was already warm


# ---------------------------------------------------------------------------
# Multi-writer safety: real OS processes racing the PRIMARY download path
# against one shared destination (simulating several pods on one volume).
# ---------------------------------------------------------------------------

class _PayloadHandler(http.server.BaseHTTPRequestHandler):
    payload = _PAYLOAD

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_a: Any) -> None:  # silence
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


def test_concurrent_processes_racing_same_missing_blob_do_not_corrupt(
    tmp_path: Path,
) -> None:
    httpd, base_url = _serve()
    try:
        dst = tmp_path / "blobs" / "blake3" / _BLAKE3[:2] / _BLAKE3[2:4] / _BLAKE3
        dst.parent.mkdir(parents=True)
        url = f"{base_url}/blob"

        ctx = multiprocessing.get_context("spawn")
        start = ctx.Event()
        results = ctx.Queue()
        n = 4
        processes = [
            ctx.Process(target=_download_worker, args=(start, results, url, str(dst)))
            for _ in range(n)
        ]
        try:
            for p in processes:
                p.start()
            start.set()
            for p in processes:
                p.join(30)
            assert [p.exitcode for p in processes] == [0] * n
            outcomes = [results.get(timeout=5) for _ in processes]
        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(5)
            results.close()

        assert outcomes == [(True, "")] * n
        assert dst.read_bytes() == _PAYLOAD
        # No writer-unique temp artifacts left behind by any racing writer.
        assert not list(dst.parent.glob(f".{dst.name}.part-*"))
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Multi-writer safety: real OS processes racing snapshot MATERIALIZATION
# (blob already present; the race is purely on the .building tmp dir).
# ---------------------------------------------------------------------------

def _materialize_worker(start: Any, results: Any, base_dir: str) -> None:
    if not start.wait(10):
        results.put((False, "start barrier timed out"))
        return
    try:
        snap_dir = asyncio.run(ensure_snapshot_async(
            base_dir=Path(base_dir),
            ref=TensorhubRef(owner="org", repo="model"),
            resolved=_resolved(),
        ))
        content = (snap_dir / "model.safetensors").read_bytes()
        results.put((content == _PAYLOAD, ""))
    except BaseException as exc:  # pragma: no cover - surfaced to parent
        results.put((False, repr(exc)))


def test_concurrent_processes_racing_same_snapshot_materialization(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "volume"
    blob = _blob(base_dir)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(_PAYLOAD)  # pre-warmed: this race is materialization-only

    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    n = 4
    processes = [
        ctx.Process(target=_materialize_worker, args=(start, results, str(base_dir)))
        for _ in range(n)
    ]
    try:
        for p in processes:
            p.start()
        start.set()
        for p in processes:
            p.join(30)
        assert [p.exitcode for p in processes] == [0] * n
        outcomes = [results.get(timeout=5) for _ in processes]
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(5)
        results.close()

    assert outcomes == [(True, "")] * n
    snaps_root = base_dir / "snapshots"
    assert not list(snaps_root.glob(f"{_SNAPSHOT}.building-*"))
