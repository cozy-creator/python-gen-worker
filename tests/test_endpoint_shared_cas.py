from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def test_racing_writers_publish_only_complete_verified_blob(tmp_path: Path) -> None:
    shared = _blob(tmp_path / "endpoint-volume")
    sources = [tmp_path / "pod-a.blob", tmp_path / "pod-b.blob"]
    for source in sources:
        source.write_bytes(_PAYLOAD)
    file = _resolved().files[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda source: CozySnapshotDownloader._copy_verified_blob(
                source, shared, file
            ),
            sources,
        ))

    assert results == [True, True]
    assert shared.read_bytes() == _PAYLOAD
    assert not list(shared.parent.glob(f".{_BLAKE3}.writer-*"))
