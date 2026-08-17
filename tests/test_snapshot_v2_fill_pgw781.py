"""Tensorhub's resolved-manifest adapter over the public tensorfs API."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import http.server
import multiprocessing
import threading
from pathlib import Path
from typing import Any

import pytest
from gen_worker._vendor.tensorfs import CASRef, FileEntry, LocalCAS, RepositoryManifest

import gen_worker.models.cozy_snapshot as snapshot_mod
from gen_worker.models.cozy_snapshot import NetworkBytesScope, ensure_snapshot_async
from gen_worker.models.hub_client import (
    WorkerResolvedChunk,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)
from gen_worker.models.refs import TensorhubRef
from gen_worker.models.store import _snapshot_to_resolved
from gen_worker.pb import worker_scheduler_pb2 as pb


#: pgw#1316: SPAWN, never fork. These tests need two real OS processes racing
#: on one CAS tree; they do not need the parent's address space. Under
#: `-n 4 --dist loadfile` an xdist worker that already ran a grpc.aio file
#: still has gRPC's event-engine threads live, and gRPC skips its
#: `pthread_atfork` handlers whenever another thread is inside gRPC
#: (`fork_posix.cc:71`) — the child then inherits a poller in an inconsistent
#: state and dies on `ev_epoll1_linux.cc: Check failed: next_worker->state ==
#: KICKED`. Spawn shares no such state. Everything crossing the boundary must
#: stay picklable.
_MP = multiprocessing.get_context("spawn")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        key = self.path.rsplit("/", 1)[-1]
        with server.lock:  # type: ignore[attr-defined]
            server.hits[key] = server.hits.get(key, 0) + 1  # type: ignore[attr-defined]
            body = server.blobs.get(key)  # type: ignore[attr-defined]
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BlobServer:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.blobs = blobs  # type: ignore[attr-defined]
        self.server.hits = {}  # type: ignore[attr-defined]
        self.server.lock = threading.Lock()  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, digest: str) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/{digest}"

    def hits(self, digest: str) -> int:
        return int(self.server.hits.get(digest, 0))  # type: ignore[attr-defined]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def _ref() -> TensorhubRef:
    return TensorhubRef(owner="acme", repo="model", release="latest")


def test_grpc_adapter_keeps_ordered_lengths_and_drops_fixed_layout_scalar() -> None:
    snapshot = pb.Snapshot(
        digest="sha256:" + "ff" * 32,
        files=[
            pb.SnapshotFile(
                path="weights.safetensors",
                size_bytes=60,
                digest="sha256:" + "ee" * 32,
                chunk_size_bytes=64 * 1024 * 1024,
                chunks=[
                    pb.ChunkRef(sha256="aa" * 32, url="https://cas/0", len=3),
                    pb.ChunkRef(sha256="bb" * 32, url="https://cas/1", len=56),
                    pb.ChunkRef(sha256="cc" * 32, url="https://cas/2", len=1),
                ],
            )
        ],
    )

    file = _snapshot_to_resolved(snapshot).files[0]

    assert [chunk.length for chunk in file.chunks] == [3, 56, 1]
    assert not hasattr(file, "chunk_size_bytes")


class _BarrierCAS(LocalCAS):
    def __init__(self, root: Path, barrier: Any) -> None:
        super().__init__(root)
        self._barrier = barrier

    def materialize_repository(
        self, manifest: RepositoryManifest, destination: str | Path
    ) -> Path:
        if Path(destination).exists():
            raise FileExistsError(destination)
        self._barrier.wait(30)
        return super().materialize_repository(manifest, destination)


def _materialize_process(
    root: str, target: str, digest: str, size: int, barrier: Any, results: Any
) -> None:
    cas = _BarrierCAS(Path(root), barrier)
    manifest = RepositoryManifest(
        (FileEntry("config.json", size, CASRef(digest)),)
    )
    try:
        path = snapshot_mod._materialize_repository(cas, manifest, Path(target))
        results.put(("ok", (path / "config.json").read_bytes()))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def _resident_resolved(digest: str, size: int) -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest="sha256:" + "8" * 64,
        files=[
            WorkerResolvedRepoFile(
                "config.json",
                size,
                "http://127.0.0.1:1/must-not-fetch",
                digest="sha256:" + digest,
            )
        ],
    )


def _stale_invalid_process(
    root: str,
    digest: str,
    size: int,
    stale_checked: Any,
    rebuilt: Any,
    ensure_entered: Any,
    allow_ensure: Any,
    results: Any,
) -> None:
    real_tree_matches = snapshot_mod._tree_matches
    real_ensure = snapshot_mod.CozySnapshotDownloader._ensure_objects
    first_match = True

    def delayed_first_match(path: Path, manifest: RepositoryManifest) -> bool:
        nonlocal first_match
        matches = real_tree_matches(path, manifest)
        if first_match:
            first_match = False
            stale_checked.set()
            if not rebuilt.wait(30):
                raise TimeoutError("rebuild did not finish")
        return matches

    async def observed_ensure(self: Any, *args: Any, **kwargs: Any) -> None:
        ensure_entered.set()
        if not allow_ensure.wait(30):
            raise TimeoutError("rebuild observer did not release ensure")
        await real_ensure(self, *args, **kwargs)

    setattr(snapshot_mod, "_tree_matches", delayed_first_match)
    setattr(snapshot_mod.CozySnapshotDownloader, "_ensure_objects", observed_ensure)
    try:
        path = asyncio.run(
            snapshot_mod.CozySnapshotDownloader().ensure_snapshot(
                Path(root),
                _ref(),
                resolved=_resident_resolved(digest, size),
            )
        )
        results.put(("stale", (path / "config.json").read_bytes()))
    except BaseException as exc:
        results.put(("stale_error", f"{type(exc).__name__}: {exc}"))
        stale_checked.set()
        ensure_entered.set()


def test_whole_file_downloads_into_tensorfs_and_reuses_it(tmp_path: Path) -> None:
    body = b"tensorfs-worker-adapter"
    digest = _sha(body)
    server = BlobServer({digest: body})
    try:
        resolved = WorkerResolvedRepo(
            snapshot_digest="sha256:" + "1" * 64,
            files=[
                WorkerResolvedRepoFile(
                    "config.json",
                    len(body),
                    server.url(digest),
                    digest="sha256:" + digest,
                )
            ],
        )
        with NetworkBytesScope() as scope:
            path = asyncio.run(
                ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
            )
        assert (path / "config.json").read_bytes() == body
        assert scope.network_bytes == len(body)
        assert LocalCAS(tmp_path).contains(CASRef(digest), size=len(body))

        again = asyncio.run(
            ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
        )
        assert again == path
        assert server.hits(digest) == 1
    finally:
        server.close()


def test_two_processes_converge_on_the_same_materialized_tree(tmp_path: Path) -> None:
    body = b"cross-process-winner"
    cas = LocalCAS(tmp_path)
    digest = cas.put_bytes(body)
    target = tmp_path / "snapshots" / "same"
    barrier = _MP.Barrier(2)
    results = _MP.Queue()
    processes = [
        _MP.Process(
            target=_materialize_process,
            args=(str(tmp_path), str(target), digest.digest, len(body), barrier, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert [results.get(timeout=5) for _ in processes] == [
        ("ok", body),
        ("ok", body),
    ]


def test_materialization_collision_refuses_an_invalid_winner(tmp_path: Path) -> None:
    good = b"expected"
    digest = CASRef.digest_bytes(good)
    manifest = RepositoryManifest((FileEntry("config.json", len(good), digest),))
    target = tmp_path / "snapshot"

    class _DivergentCAS(LocalCAS):
        def materialize_repository(
            self, _manifest: RepositoryManifest, destination: str | Path
        ) -> Path:
            winner = Path(destination)
            winner.mkdir(parents=True)
            (winner / "config.json").write_bytes(b"diverged")
            raise OSError(errno.ENOTEMPTY, "another process published")

    with pytest.raises(OSError) as caught:
        snapshot_mod._materialize_repository(
            _DivergentCAS(tmp_path / "cas"), manifest, target
        )
    assert caught.value.errno == errno.ENOTEMPTY
    assert (target / "config.json").read_bytes() == b"diverged"


def test_stale_invalid_validation_cannot_delete_a_rebuilt_tree(
    tmp_path: Path,
) -> None:
    body = b"valid-after-recovery"
    digest = LocalCAS(tmp_path).put_bytes(body)
    target = tmp_path / "snapshots" / snapshot_mod.snapshot_dir_key(
        "sha256:" + "8" * 64
    )
    target.mkdir(parents=True)
    (target / "config.json").write_bytes(b"invalid-old-target")

    stale_checked = _MP.Event()
    rebuilt = _MP.Event()
    ensure_entered = _MP.Event()
    allow_ensure = _MP.Event()
    results = _MP.Queue()
    stale = _MP.Process(
        target=_stale_invalid_process,
        args=(
            str(tmp_path),
            digest.digest,
            len(body),
            stale_checked,
            rebuilt,
            ensure_entered,
            allow_ensure,
            results,
        ),
    )
    stale.start()
    try:
        assert stale_checked.wait(30)
        rebuilt_path = asyncio.run(
            snapshot_mod.CozySnapshotDownloader().ensure_snapshot(
                tmp_path,
                _ref(),
                resolved=_resident_resolved(digest.digest, len(body)),
            )
        )
        rebuilt.set()
        assert ensure_entered.wait(30)
        assert (rebuilt_path / "config.json").read_bytes() == body
    finally:
        rebuilt.set()
        allow_ensure.set()
        stale.join(timeout=40)

    assert stale.exitcode == 0
    assert results.get(timeout=5) == ("stale", body)


def test_same_snapshot_key_in_two_scoped_roots_does_not_crosstalk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"root-scoped-snapshot"
    digest = CASRef.digest_bytes(body)
    roots = (tmp_path / "left", tmp_path / "right")
    for root in roots:
        LocalCAS(root).put_bytes(body, expected=digest)
    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "9" * 64,
        files=[
            WorkerResolvedRepoFile(
                "config.json",
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(digest),
            )
        ],
    )
    real_ensure = snapshot_mod.CozySnapshotDownloader._ensure_objects

    async def scenario() -> tuple[Path, Path]:
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed(self: Any, cas: LocalCAS, *args: Any, **kwargs: Any) -> None:
            if cas.root == roots[0]:
                started.set()
                await release.wait()
            await real_ensure(self, cas, *args, **kwargs)

        monkeypatch.setattr(
            snapshot_mod.CozySnapshotDownloader, "_ensure_objects", delayed
        )
        left = asyncio.create_task(
            ensure_snapshot_async(base_dir=roots[0], ref=_ref(), resolved=resolved)
        )
        await started.wait()
        right = asyncio.create_task(
            ensure_snapshot_async(base_dir=roots[1], ref=_ref(), resolved=resolved)
        )
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(left, right)

    left, right = asyncio.run(scenario())
    assert (left / "config.json").read_bytes() == body
    assert (right / "config.json").read_bytes() == body


def test_chunked_file_uses_manifest_recorded_variable_lengths(tmp_path: Path) -> None:
    first = b"header"
    second = b"tensor-body"
    chunks = (first, second)
    whole = hashlib.sha256(first + second).hexdigest()
    blobs = {_sha(chunk): chunk for chunk in chunks}
    server = BlobServer(blobs)
    try:
        resolved = WorkerResolvedRepo(
            snapshot_digest="sha256:" + "2" * 64,
            files=[
                WorkerResolvedRepoFile(
                    "weights.safetensors",
                    len(first) + len(second),
                    None,
                    digest="sha256:" + whole,
                    chunks=tuple(
                        WorkerResolvedChunk(_sha(chunk), server.url(_sha(chunk)), len(chunk))
                        for chunk in chunks
                    ),
                )
            ],
        )
        path = asyncio.run(
            ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
        )
        output = path / "weights.safetensors"
        assert output.stat().st_size == len(first) + len(second)
        assert hashlib.sha256(output.read_bytes()).hexdigest() == whole
        assert all(server.hits(digest) == 1 for digest in blobs)
    finally:
        server.close()


def test_endpoint_volume_is_a_verified_fill_source(tmp_path: Path) -> None:
    body = b"warm-volume"
    digest = CASRef.digest_bytes(body)
    volume = tmp_path / "volume"
    LocalCAS(volume).put_bytes(body, expected=digest)
    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "3" * 64,
        files=[
            WorkerResolvedRepoFile(
                "config.json",
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(digest),
            )
        ],
    )
    with NetworkBytesScope() as scope:
        path = asyncio.run(
            ensure_snapshot_async(
                base_dir=tmp_path / "local",
                ref=_ref(),
                resolved=resolved,
                fill_source_dir=volume,
            )
        )
    assert (path / "config.json").read_bytes() == body
    assert scope.network_bytes == 0


@pytest.mark.parametrize("legacy", ["weights.parts.json", "weights.part0000"])
def test_legacy_split_snapshots_are_refused(tmp_path: Path, legacy: str) -> None:
    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "4" * 64,
        files=[
            WorkerResolvedRepoFile(
                legacy,
                1,
                "http://127.0.0.1:1/unused",
                digest="sha256:" + "a" * 64,
            )
        ],
    )
    with pytest.raises(ValueError, match="legacy split-file"):
        asyncio.run(
            ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
        )
