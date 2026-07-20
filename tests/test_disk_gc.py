"""Disk retention (#370) — real files in a budgeted tmp cache dir.

Drives the REAL ModelStore.ensure_local path (download layer stubbed to write
actual bytes) with an injected free-disk probe: LRU non-keep eviction with
EVICTED events, keep-pressure escape hatch, in-use pins, fail-fast
insufficient_disk, and the boot-time rescan baseline.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

import gen_worker.executor as executor_mod
from gen_worker.capability import InsufficientDiskError
from gen_worker.executor import ModelStore
from gen_worker.models.residency import Tier
from gen_worker.pb import worker_scheduler_pb2 as pb

_BUDGET = 10_000


def _snapshot(digest: str, size: int) -> pb.Snapshot:
    return pb.Snapshot(digest=digest, files=[
        pb.SnapshotFile(path="w.bin", size_bytes=size, blake3="ab" * 16,
                        url="http://example.invalid/w.bin"),
    ])


@pytest.fixture(autouse=True)
def _tight_budget(monkeypatch):
    monkeypatch.setattr(executor_mod, "_DISK_GC_MARGIN_BYTES", 0)
    monkeypatch.setattr(executor_mod, "_DISK_GC_GRACE_S", 0.0)


def _store(tmp_path: Path, sent: list) -> ModelStore:
    async def _emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    def _free() -> int:
        from gen_worker.models.disk_gc import tree_bytes

        return max(0, _BUDGET - tree_bytes(tmp_path))

    return ModelStore(_emit, cache_dir=tmp_path, disk_free_bytes_fn=_free)


@pytest.fixture()
def _fake_download(monkeypatch, tmp_path):
    """Stub the provider download layer: write real bytes into the CAS
    snapshots layout so GC deletes real files."""

    async def fake_ensure_local(ref, *, snapshot=None, cache_dir=None, **kw) -> Path:
        size = sum(int(f.size_bytes) for f in (snapshot.files if snapshot else []))
        d = Path(cache_dir) / "snapshots" / ref.replace("/", "--")
        d.mkdir(parents=True, exist_ok=True)
        (d / "w.bin").write_bytes(b"\0" * size)
        return d

    monkeypatch.setattr(executor_mod, "ensure_local", fake_ensure_local)


def _evicted(sent) -> list:
    return [m.model_event.ref for m in sent if m.WhichOneof("msg") == "model_event"
            and m.model_event.state == pb.MODEL_STATE_EVICTED]


def _failed(sent) -> list:
    return [(m.model_event.ref, m.model_event.error) for m in sent
            if m.WhichOneof("msg") == "model_event"
            and m.model_event.state == pb.MODEL_STATE_FAILED]


def test_gc_evicts_lru_nonkeep_and_spares_recent_and_keep(tmp_path, _fake_download) -> None:
    sent: list = []
    store = _store(tmp_path, sent)
    store.keep = ["t/keep"]

    async def _run() -> None:
        await store.ensure_local("t/a", _snapshot("da", 3000))
        await store.ensure_local("t/b", _snapshot("db", 3000))
        await store.ensure_local("t/keep", _snapshot("dk", 1000))
        await store.ensure_local("t/a", _snapshot("da", 3000))  # touch: b is LRU
        # 7000 used; 4000 needed -> GC must evict exactly the LRU non-keep ref.
        await store.ensure_local("t/c", _snapshot("dc", 4000))

    asyncio.run(_run())
    assert _evicted(sent) == ["t/b"]
    assert store.residency.tier("t/b") is None
    assert not (tmp_path / "snapshots" / "t--b").exists()  # bytes actually gone
    for ref in ("t/a", "t/keep", "t/c"):
        assert store.residency.tier(ref) is Tier.DISK, ref


def test_keep_pressure_escape_hatch_evicts_keep_with_event(tmp_path, _fake_download) -> None:
    sent: list = []
    store = _store(tmp_path, sent)
    store.keep = ["t/keep"]

    async def _run() -> None:
        await store.ensure_local("t/keep", _snapshot("dk", 3000))
        await store.ensure_local("t/big", _snapshot("dbig", 8000))

    asyncio.run(_run())
    assert _evicted(sent) == ["t/keep"]  # EVICTED still emitted -> hub re-downloads
    assert store.residency.tier("t/big") is Tier.DISK


def test_keep_pressure_evicts_lowest_controller_priority_first(
    tmp_path, _fake_download
) -> None:
    sent: list = []
    store = _store(tmp_path, sent)
    store.keep = ["t/high", "t/mid", "t/low"]

    async def _run() -> None:
        await store.ensure_local("t/high", _snapshot("dh", 3000))
        await store.ensure_local("t/mid", _snapshot("dm", 3000))
        await store.ensure_local("t/low", _snapshot("dl", 3000))
        await store.ensure_local("t/job", _snapshot("dj", 2000))

    asyncio.run(_run())
    assert _evicted(sent) == ["t/low"]


def test_keep_priority_outranks_recent_use(tmp_path, _fake_download, monkeypatch) -> None:
    sent: list = []
    store = _store(tmp_path, sent)
    store.keep = ["t/high", "t/low"]

    async def _run() -> None:
        await store.ensure_local("t/high", _snapshot("dh", 3000))
        await store.ensure_local("t/low", _snapshot("dl", 3000))
        now = time.time()
        monkeypatch.setattr(
            store._index,
            "last_used",
            lambda ref: now - 7200 if ref == "t/high" else now,
        )
        monkeypatch.setattr(executor_mod, "_DISK_GC_GRACE_S", 3600.0)
        await asyncio.to_thread(store.gc_disk, 6000)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert _evicted(sent) == ["t/low"]


def test_gc_uses_one_keep_snapshot_during_replace(
    tmp_path, _fake_download, monkeypatch
) -> None:
    sent: list = []
    store = _store(tmp_path, sent)
    store.keep = ["t/high", "t/low"]

    async def _run() -> None:
        await store.ensure_local("t/high", _snapshot("dh", 3000))
        await store.ensure_local("t/low", _snapshot("dl", 3000))
        refs_in = store.residency.refs_in
        replaced = False

        def replace_during_scan(tier):
            nonlocal replaced
            refs = refs_in(tier)
            if not replaced:
                replaced = True
                store.keep = ["t/replacement"]
            return refs

        monkeypatch.setattr(store.residency, "refs_in", replace_during_scan)
        await asyncio.to_thread(store.gc_disk, 6000)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert _evicted(sent) == ["t/low"]


def test_insufficient_disk_fails_fast_with_event(tmp_path, _fake_download) -> None:
    sent: list = []
    store = _store(tmp_path, sent)

    async def _run() -> None:
        with pytest.raises(InsufficientDiskError):
            await store.ensure_local("t/huge", _snapshot("dh", 50_000))

    asyncio.run(_run())
    assert ("t/huge", "insufficient_disk") in _failed(sent)


def test_gc_never_evicts_in_use_refs(tmp_path, _fake_download) -> None:
    sent: list = []
    store = _store(tmp_path, sent)

    async def _run() -> None:
        await store.ensure_local("t/a", _snapshot("da", 6000))
        with store.residency.executing("t/a"):
            with pytest.raises(InsufficientDiskError):
                await store.ensure_local("t/b", _snapshot("db", 6000))

    asyncio.run(_run())
    assert _evicted(sent) == []
    assert store.residency.tier("t/a") is Tier.DISK


def test_rescan_restores_disk_baseline_after_restart(tmp_path, _fake_download) -> None:
    sent: list = []
    store = _store(tmp_path, sent)

    async def _run() -> None:
        await store.ensure_local("t/a", _snapshot("da", 2000))
        await store.ensure_local("t/b", _snapshot("db", 2000))

    asyncio.run(_run())
    (tmp_path / "snapshots" / "t--b" / "w.bin").unlink()  # b lost outside our control
    (tmp_path / "snapshots" / "t--b").rmdir()

    restarted = _store(tmp_path, [])  # fresh process: Residency starts empty
    assert restarted.residency.tier("t/a") is None
    restarted.rescan_disk()
    assert restarted.residency.tier("t/a") is Tier.DISK
    assert restarted.residency.local_path("t/a") == tmp_path / "snapshots" / "t--a"
    assert restarted.residency.tier("t/b") is None  # stale entry dropped


def test_rescan_sweeps_stale_writer_temp_artifacts(tmp_path, _fake_download) -> None:
    """th#850: a CAS root on a persistent volume (unlike ephemeral pod-local
    disk) keeps a crashed writer's temp artifacts forever unless boot-time
    rescan sweeps them. Fresh/live artifacts must survive the sweep."""
    from gen_worker.models import disk_gc

    store = _store(tmp_path, [])
    blob_dir = tmp_path / "blobs" / "blake3" / "ab" / "cd"
    blob_dir.mkdir(parents=True)
    stale_blob_tmp = blob_dir / ".deadbeef.part-stale-writer"
    fresh_blob_tmp = blob_dir / ".deadbeef.part-live-writer"
    stale_blob_tmp.write_bytes(b"partial")
    fresh_blob_tmp.write_bytes(b"partial")

    snaps_root = tmp_path / "snapshots"
    stale_building = snaps_root / "abc123.building-stale-writer"
    fresh_building = snaps_root / "abc123.building-live-writer"
    stale_building.mkdir(parents=True)
    fresh_building.mkdir(parents=True)
    (stale_building / "w.bin").write_bytes(b"x")

    old = time.time() - disk_gc._STALE_WRITER_TEMP_AGE_S - 1
    import os
    os.utime(stale_blob_tmp, (old, old))
    os.utime(stale_building, (old, old))

    store.rescan_disk()

    assert not stale_blob_tmp.exists()
    assert fresh_blob_tmp.exists()
    assert not stale_building.exists()
    assert fresh_building.exists()
