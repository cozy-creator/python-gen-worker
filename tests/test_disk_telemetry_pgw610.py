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
import time
from pathlib import Path
from types import SimpleNamespace

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
        report = asyncio.run(store.refresh_disk_usage_report())

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
    # The cache (the hot state-delta path's ONLY read) matches.
    assert store.disk_usage_report() == report

    # Unchanged measurement -> unchanged generation.
    with store.residency.executing("acme/busy"):
        again = asyncio.run(store.refresh_disk_usage_report())
    assert again.capacity_generation == 1
    assert _container(again).reclaimable_bytes == 4096


def test_desired_set_and_pins_move_reclaimable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _track(store, tmp_path, "acme/a", 4096)
    _track(store, tmp_path, "acme/b", 2048)
    store.keep = ["acme/a", "acme/b"]
    assert _container(
        asyncio.run(store.refresh_disk_usage_report())).reclaimable_bytes == 0

    # Dropping a ref from the desired set makes its bytes reclaimable and
    # bumps the generation (a real capacity change the hub may budget).
    store.keep = ["acme/a"]
    report = asyncio.run(store.refresh_disk_usage_report())
    assert _container(report).reclaimable_bytes == 2048
    assert report.capacity_generation == 2


def test_generation_bumps_on_eviction(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _track(store, tmp_path, "acme/idle", 4096)
    first = asyncio.run(store.refresh_disk_usage_report())
    assert _container(first).reclaimable_bytes == 4096
    assert first.capacity_generation == 1

    store._evict_disk_ref("acme/idle")
    after = asyncio.run(store.refresh_disk_usage_report())
    assert _container(after).reclaimable_bytes == 0
    assert after.capacity_generation == 2


def test_volume_tier_reported_when_fill_source_present(tmp_path: Path) -> None:
    cas = tmp_path / "cas"
    volume = tmp_path / "volume"
    volume.mkdir()
    store = ModelStore(_noop_send, cache_dir=cas, fill_source_dir=volume)
    report = asyncio.run(store.refresh_disk_usage_report())
    tiers = {t.tier: t for t in report.tiers}
    assert pb.STORAGE_TIER_CONTAINER in tiers
    assert pb.STORAGE_TIER_VOLUME in tiers
    assert tiers[pb.STORAGE_TIER_VOLUME].mount_path == str(volume)
    assert tiers[pb.STORAGE_TIER_VOLUME].total_bytes > 0


def test_disk_usage_report_never_measures_only_reads_the_cache(
    tmp_path: Path, monkeypatch,
) -> None:
    """boothang: disk_usage_report() (the method _state_delta() calls
    synchronously, incl. right after seal_publish, with no event loop
    guaranteed) must NEVER touch a filesystem — only refresh_disk_usage_
    report() (awaited, off-loop) may. If disk_usage_report() ever measures
    again, a stalled provider VOLUME mount can freeze the whole worker."""
    from gen_worker.models import disk_telemetry

    store = _store(tmp_path)

    def _boom(*_a, **_k):
        raise AssertionError(
            "disk_usage_report() must not measure — it must only read "
            "the cache refresh_disk_usage_report() last populated")

    monkeypatch.setattr(disk_telemetry, "measure_tiers", _boom)
    # Empty cache before any refresh — reading it must not raise or block.
    assert store.disk_usage_report() == pb.DiskUsageReport()


def test_hello_and_state_delta_carry_measured_disk(tmp_path: Path) -> None:
    async def _go() -> None:
        store = _store(tmp_path)
        _track(store, tmp_path, "acme/idle", 4096)
        await store.refresh_disk_usage_report()
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


def test_state_delta_never_blocks_event_loop_on_a_stalled_mount(
    tmp_path: Path, monkeypatch,
) -> None:
    """boothang revert-turns-red: the real 0.40.7 LTX shape. A provider
    VOLUME mount that stalls under statvfs() (network-backed fill-source
    under load right after a self-mint's weight download + cell pack) must
    NEVER freeze the event loop. Before the fix, _state_delta() measured
    disk usage synchronously and inline — a stalled statvfs() there wedges
    every StateDelta/RunJob/drain signal for as long as the mount hangs.
    This test hangs (via the overall pytest timeout) on the broken shape
    and completes in well under the stall duration once the measurement is
    off-loaded."""
    import gen_worker.models.disk_telemetry as disk_telemetry_mod

    store = _store(tmp_path)
    real_statvfs = os.statvfs

    def _stalled_statvfs(path):
        if str(path) == str(tmp_path):
            time.sleep(5.0)  # simulates a hung provider network mount
        return real_statvfs(path)

    monkeypatch.setattr(disk_telemetry_mod.os, "statvfs", _stalled_statvfs)

    async def _go() -> None:
        ticks = 0

        async def _heartbeat() -> None:
            nonlocal ticks
            for _ in range(20):
                await asyncio.sleep(0.05)
                ticks += 1

        refresh_task = asyncio.create_task(store.refresh_disk_usage_report())
        heartbeat_task = asyncio.create_task(_heartbeat())
        # The event loop's OTHER work (heartbeat) must keep making progress
        # while the stalled statvfs runs in its own thread.
        await asyncio.sleep(0.3)
        assert ticks >= 4, (
            "the event loop froze while disk usage was being measured — "
            "a stalled mount must never block anything but its own "
            "to_thread call")
        # Meanwhile _state_delta() (the hot path) must also stay non-
        # blocking and cheap: it only reads the (still-empty) cache.
        t0 = time.monotonic()
        report = store.disk_usage_report()
        assert time.monotonic() - t0 < 0.1
        assert report == pb.DiskUsageReport()

        await heartbeat_task
        await refresh_task  # the stalled measurement eventually completes

    asyncio.run(asyncio.wait_for(_go(), timeout=10.0))
