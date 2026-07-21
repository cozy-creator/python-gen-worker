"""pgw#610/th#962: measured disk telemetry — honest free/reclaimable.

Real files in a tmp CAS dir, real statvfs on the mount actually holding it,
real ModelStore residency/ref-index state. The report's free bytes come from
the filesystem (quantized, never a declared size); reclaimable equals exactly
the disk-GC LRU's eligible set (inactive AND not in the desired set); the
capacity generation bumps only when the measured shape changes.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

from gen_worker.executor import Executor, ModelStore
from gen_worker.lifecycle import Lifecycle
from gen_worker.models.disk_telemetry import DISK_QUANTUM_BYTES
from gen_worker.pb import worker_scheduler_pb2 as pb


async def _noop_send(msg) -> None:  # pragma: no cover
    pass


def _store(tmp_path: Path) -> ModelStore:
    return ModelStore(_noop_send, cache_dir=tmp_path)


def _track(store: ModelStore, tmp_path: Path, ref: str, size: int) -> Path:
    p = tmp_path / "snapshots" / ref.replace("/", "_") / "w.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)
    store._index.record(ref, p, size)
    store.residency.track_disk(ref, p)
    return p


def _container(report: pb.DiskUsageReport) -> pb.StorageTierUsage:
    tiers = [t for t in report.tiers if t.tier == pb.STORAGE_TIER_CONTAINER]
    assert len(tiers) == 1
    return tiers[0]


def test_report_measures_real_mount_and_reclaimable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _track(store, tmp_path, "acme/idle", 4096)          # inactive + undesired
    _track(store, tmp_path, "acme/kept", 2048)          # in the desired set
    _track(store, tmp_path, "acme/busy", 1024)          # actively executing
    store.keep = ["acme/kept"]

    with store.residency.executing("acme/busy"):
        report = store.disk_usage_report()

    st = os.statvfs(tmp_path)
    frsize = st.f_frsize or st.f_bsize
    tier = _container(report)
    assert tier.mount_path == str(tmp_path)
    assert tier.total_bytes == st.f_blocks * frsize
    # Measured, quantized, and never above what statvfs actually reports.
    assert tier.free_bytes % DISK_QUANTUM_BYTES == 0
    assert tier.free_bytes <= st.f_bavail * frsize
    assert tier.used_bytes == tier.total_bytes - tier.free_bytes
    # Only the inactive AND undesired ref is safely reclaimable.
    assert tier.reclaimable_bytes == 4096
    assert report.capacity_generation == 1

    # Unchanged measurement -> unchanged generation.
    with store.residency.executing("acme/busy"):
        again = store.disk_usage_report()
    assert again.capacity_generation == 1
    assert _container(again).reclaimable_bytes == 4096


def test_desired_set_and_pins_move_reclaimable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _track(store, tmp_path, "acme/a", 4096)
    _track(store, tmp_path, "acme/b", 2048)
    store.keep = ["acme/a", "acme/b"]
    assert _container(store.disk_usage_report()).reclaimable_bytes == 0

    # Dropping a ref from the desired set makes its bytes reclaimable and
    # bumps the generation (a real capacity change the hub may budget).
    store.keep = ["acme/a"]
    report = store.disk_usage_report()
    assert _container(report).reclaimable_bytes == 2048
    assert report.capacity_generation == 2


def test_generation_bumps_on_eviction(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _track(store, tmp_path, "acme/idle", 4096)
    first = store.disk_usage_report()
    assert _container(first).reclaimable_bytes == 4096
    assert first.capacity_generation == 1

    store._evict_disk_ref("acme/idle")
    after = store.disk_usage_report()
    assert _container(after).reclaimable_bytes == 0
    assert after.capacity_generation == 2


def test_volume_tier_reported_when_fill_source_present(tmp_path: Path) -> None:
    cas = tmp_path / "cas"
    volume = tmp_path / "volume"
    volume.mkdir()
    store = ModelStore(_noop_send, cache_dir=cas, fill_source_dir=volume)
    report = store.disk_usage_report()
    tiers = {t.tier: t for t in report.tiers}
    assert pb.STORAGE_TIER_CONTAINER in tiers
    assert pb.STORAGE_TIER_VOLUME in tiers
    assert tiers[pb.STORAGE_TIER_VOLUME].mount_path == str(volume)
    assert tiers[pb.STORAGE_TIER_VOLUME].total_bytes > 0


def test_hello_and_state_delta_carry_measured_disk(tmp_path: Path) -> None:
    async def _go() -> None:
        store = _store(tmp_path)
        _track(store, tmp_path, "acme/idle", 4096)
        ex = Executor([], _noop_send, store=store)
        lc = Lifecycle(
            SimpleNamespace(worker_jwt="", worker_id="w-test",
                            runpod_pod_id="", worker_image_digest=""),
            ex,
        )
        delta = lc._state_delta()
        assert delta.disk_usage.capacity_generation >= 1
        assert _container(delta.disk_usage).reclaimable_bytes == 4096
        hello = lc.build_hello()
        assert hello.state.disk_usage.capacity_generation >= 1
        assert _container(hello.state.disk_usage).total_bytes > 0

    asyncio.run(_go())
