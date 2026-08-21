"""The snapshot headroom gate must size WHAT IT WRITES — no more, no less."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest
from gen_worker._vendor.tensorfs import CASRef, LocalCAS, stub_bytes

from gen_worker.capability import InsufficientDiskError
from gen_worker.models import cozy_snapshot, projection
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import (
    WorkerResolvedChunk,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)
from gen_worker.models.refs import TensorhubRef

_HEADROOM = cozy_snapshot._DISK_HEADROOM_BYTES


def _ref() -> TensorhubRef:
    return TensorhubRef(owner="acme", repo="model", release="latest")


def _pin_free(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    real = shutil.disk_usage

    def fake(path: Any) -> Any:
        usage = real(path)
        return type(usage)(total=usage.total, used=usage.used, free=free)

    monkeypatch.setattr(cozy_snapshot.shutil, "disk_usage", fake)


def _stub_cost(digest: CASRef, size: int) -> int:
    return len(stub_bytes(digest, size))


def test_a_fully_resident_snapshot_is_still_sized_before_it_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to fetch is not nothing to write."""

    bodies = [b"weights-" + bytes([index]) * 64 for index in range(3)]
    store = LocalCAS(tmp_path)
    digests = [store.put_bytes(body) for body in bodies]
    stubs = sum(
        _stub_cost(digest, len(body))
        for body, digest in zip(bodies, digests, strict=True)
    )

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "a" * 64,
        files=[
            WorkerResolvedRepoFile(
                f"shard-{index}.safetensors",
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(digest),
            )
            for index, (body, digest) in enumerate(zip(bodies, digests, strict=True))
        ],
    )

    _pin_free(monkeypatch, _HEADROOM + stubs - 1)

    with pytest.raises(InsufficientDiskError) as refusal:
        asyncio.run(
            ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
        )
    assert refusal.value.required_bytes == _HEADROOM + stubs
    assert "to publish" in str(refusal.value)


def test_the_gate_counts_the_projection_on_top_of_the_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pod with room for the download alone still runs out publishing it."""

    remote = b"remote-" + b"m" * 200
    first, second = b"clip-header" + b"h" * 40, b"clip-body" + b"b" * 90
    store = LocalCAS(tmp_path)
    remote_digest = LocalCAS(tmp_path / "other").put_bytes(remote)
    chunk_digests = [store.put_bytes(part) for part in (first, second)]
    whole = CASRef.digest_bytes(first + second)
    reassembled = len(first) + len(second)

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "b" * 64,
        files=[
            WorkerResolvedRepoFile(
                "model.safetensors",
                len(remote),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(remote_digest),
            ),
            WorkerResolvedRepoFile(
                "dataset/clip.mp4",
                reassembled,
                None,
                digest=str(whole),
                chunks=tuple(
                    WorkerResolvedChunk(
                        digest.digest, "http://127.0.0.1:1/must-not-fetch", len(part)
                    )
                    for digest, part in zip(chunk_digests, (first, second), strict=True)
                ),
            ),
        ],
    )

    _pin_free(monkeypatch, _HEADROOM + len(remote))

    with pytest.raises(InsufficientDiskError) as refusal:
        asyncio.run(
            ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
        )
    stub = _stub_cost(remote_digest, len(remote))
    assert refusal.value.required_bytes == (
        _HEADROOM + len(remote) + stub + reassembled
    )
    assert refusal.value.available_bytes == _HEADROOM + len(remote)


def test_the_gate_does_not_charge_a_whole_second_copy_of_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    body = b"a-model-that-fits-" + b"w" * 4096
    store = LocalCAS(tmp_path)
    digest = store.put_bytes(body)

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "d" * 64,
        files=[
            WorkerResolvedRepoFile(
                "model.safetensors",
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(digest),
            )
        ],
    )

    stub = _stub_cost(digest, len(body))
    assert stub < len(body)
    _pin_free(monkeypatch, _HEADROOM + stub)

    path = asyncio.run(
        ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
    )
    assert projection.logical_size(path / "model.safetensors") == len(body)


def test_a_snapshot_that_fits_still_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate must leave the ordinary path alone."""

    body = b"small-model-" + b"s" * 32
    store = LocalCAS(tmp_path)
    digest = store.put_bytes(body)

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "c" * 64,
        files=[
            WorkerResolvedRepoFile(
                "config.json",
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(digest),
            )
        ],
    )

    _pin_free(monkeypatch, _HEADROOM + len(body))

    path = asyncio.run(
        ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
    )
    assert (path / "config.json").read_bytes() == body
