"""The snapshot headroom gate must size the publish, not only the fetch.

`_publish_snapshot` writes a full second copy of every file in the manifest
next to the CAS objects, so a snapshot costs ~2x its size on disk. The gate
sized only the objects still to be fetched, and skipped itself entirely when
none were missing — which is exactly the case where the publish is the sole
writer. Both arms below pass a fetch-only gate and fail an honest one.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest
from gen_worker._vendor.tensorfs import LocalCAS

from gen_worker.capability import InsufficientDiskError
from gen_worker.models import cozy_snapshot
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
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


def test_a_fully_resident_snapshot_is_still_sized_before_it_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to fetch is not nothing to write.

    Every object is already in the CAS, so `missing` is empty and the old
    `if missing and ...` guard never evaluated the comparison at all. The
    publish still writes the whole tree. With free space below the tree size
    plus the reserve, this must refuse rather than ENOSPC mid-materialize.
    """

    bodies = [b"weights-" + bytes([index]) * 64 for index in range(3)]
    store = LocalCAS(tmp_path)
    digests = [store.put_bytes(body) for body in bodies]
    tree_bytes = sum(len(body) for body in bodies)

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

    # Enough for the reserve and every byte but one of the tree.
    _pin_free(monkeypatch, _HEADROOM + tree_bytes - 1)

    with pytest.raises(InsufficientDiskError) as refusal:
        asyncio.run(
            ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
        )
    assert refusal.value.required_bytes == _HEADROOM + tree_bytes
    assert "to publish" in str(refusal.value)


def test_the_gate_counts_the_published_tree_on_top_of_the_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pod with room for the download alone still runs out publishing it.

    One object is resident and one must be fetched. Free space covers the
    reserve plus the fetch exactly — the old arithmetic's whole budget — and
    is short by the tree. The refusal must name both halves.
    """

    resident = b"resident-" + b"r" * 128
    remote = b"remote-" + b"m" * 200
    store = LocalCAS(tmp_path)
    resident_digest = store.put_bytes(resident)
    remote_digest = LocalCAS(tmp_path / "other").put_bytes(remote)
    tree_bytes = len(resident) + len(remote)

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "b" * 64,
        files=[
            WorkerResolvedRepoFile(
                "config.json",
                len(resident),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(resident_digest),
            ),
            WorkerResolvedRepoFile(
                "model.safetensors",
                len(remote),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(remote_digest),
            ),
        ],
    )

    _pin_free(monkeypatch, _HEADROOM + len(remote))

    with pytest.raises(InsufficientDiskError) as refusal:
        asyncio.run(
            ensure_snapshot_async(base_dir=tmp_path, ref=_ref(), resolved=resolved)
        )
    assert refusal.value.required_bytes == _HEADROOM + len(remote) + tree_bytes
    assert refusal.value.available_bytes == _HEADROOM + len(remote)


def test_a_snapshot_that_fits_both_copies_still_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate must not refuse a snapshot that genuinely fits.

    Sizing the publish is only correct if it leaves the ordinary path alone;
    a gate that always refuses would pass both arms above and ship an outage.
    """

    body = b"small-model-" + b"s" * 32
    store = LocalCAS(tmp_path)
    digest = store.put_bytes(body)

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "c" * 64,
        files=[
            WorkerResolvedRepoFile(
                "model.safetensors",
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
    assert (path / "model.safetensors").read_bytes() == body
