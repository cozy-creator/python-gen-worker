from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path
from typing import Any

from blake3 import blake3

import gen_worker.models.cozy_snapshot as snap_mod
import gen_worker.models.cache_paths as cache_paths
from gen_worker.models.cozy_snapshot import CozySnapshotDownloader, ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef


_PAYLOAD = b"endpoint-isolated-model-weights" * 1024
_BLAKE3 = blake3(_PAYLOAD).hexdigest()
_SNAPSHOT = "a5" * 32


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


def _copy_blob_process(
    start: Any,
    results: Any,
    source: str,
    destination: str,
    size_bytes: int,
    digest: str,
) -> None:
    """Run the production CAS publisher in an independently spawned process."""
    if not start.wait(10):
        results.put((False, "start barrier timed out"))
        return
    try:
        copied = CozySnapshotDownloader._copy_verified_blob(
            Path(source),
            Path(destination),
            WorkerResolvedRepoFile(
                path="model.safetensors",
                size_bytes=size_bytes,
                blake3=digest,
                url="https://tensorhub.invalid/authorized-blob",
            ),
        )
        results.put((copied, ""))
    except BaseException as exc:  # pragma: no cover - returned to the parent for diagnosis
        results.put((False, repr(exc)))


def test_shared_cache_is_enabled_only_for_real_provider_mount(monkeypatch) -> None:
    monkeypatch.setattr(cache_paths.os.path, "ismount", lambda _path: False)
    assert cache_paths.endpoint_shared_cas_dir() is None

    monkeypatch.setattr(cache_paths.os.path, "ismount", lambda path: path == cache_paths.ENDPOINT_SHARED_CACHE_MOUNT)
    assert cache_paths.endpoint_shared_cas_dir() == (
        cache_paths.ENDPOINT_SHARED_CACHE_MOUNT / "tensorhub-cas-v1"
    )


def test_second_pod_reuses_verified_endpoint_volume_without_public_get(
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
    shared = tmp_path / "endpoint-volume"

    first = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path / "pod-a", shared_base_dir=shared,
        ref=ref, resolved=_resolved(),
    ))
    assert (first / "model.safetensors").read_bytes() == _PAYLOAD
    assert _blob(shared).read_bytes() == _PAYLOAD

    second = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path / "pod-b", shared_base_dir=shared,
        ref=ref, resolved=_resolved(),
    ))
    assert (second / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == 1


def test_corrupt_endpoint_volume_blob_falls_back_and_atomically_repairs(
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
    shared = tmp_path / "endpoint-volume"
    corrupt = _blob(shared)
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"x" * len(_PAYLOAD))

    out = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path / "pod", shared_base_dir=shared,
        ref=TensorhubRef(owner="org", repo="model"), resolved=_resolved(),
    ))

    assert calls == 1
    assert (out / "model.safetensors").read_bytes() == _PAYLOAD
    assert corrupt.read_bytes() == _PAYLOAD
    assert not list(corrupt.parent.glob(f".{_BLAKE3}.writer-*"))


def test_spawned_process_writers_publish_only_complete_verified_blob(tmp_path: Path) -> None:
    payload = _PAYLOAD * 128
    digest = blake3(payload).hexdigest()
    shared = (
        tmp_path / "endpoint-volume" / "blobs" / "blake3" / digest[:2] / digest[2:4]
        / digest
    )
    sources = [tmp_path / f"pod-{i}.blob" for i in range(4)]
    for source in sources:
        source.write_bytes(payload)

    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_copy_blob_process,
            args=(start, results, str(source), str(shared), len(payload), digest),
        )
        for source in sources
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(20)
        assert [process.exitcode for process in processes] == [0] * len(processes)
        outcomes = [results.get(timeout=5) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        results.close()

    assert outcomes == [(True, "")] * len(processes)
    assert shared.read_bytes() == payload
    assert not list(shared.parent.glob(f".{digest}.writer-*"))
