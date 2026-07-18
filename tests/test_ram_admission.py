"""Host-RAM admission (gw#407) — CPU-only, deterministic.

J17 livelock: 16 SDXL variants on a 31GB host — every load/demote grew the
warm RAM tier until the host entered reclaim-thrash, which stalled the whole
process (including gRPC keepalive acks), so the hub disconnected, requeued the
same job, and the cycle repeated. The RAM tier is now admission-controlled:
demote is size-aware, loads vacate warm endpoint records through their owner,
and a load that still cannot fit fails RETRYABLE instead of thrashing.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import inspect
import logging
import mmap
import os
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import msgspec
import psutil
import pytest

import gen_worker.executor as executor_mod
from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.api.slot import Slot
from gen_worker.capability import InsufficientHostRamError
from gen_worker.config import Settings
from gen_worker.executor import Executor, ModelStore, _ClassRecord, _Job
from gen_worker.lifecycle import Lifecycle
from gen_worker.models import memory as memory_mod
from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import HostRamHeadroom, LoadedComponentKey, Residency, Tier
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.transport import Transport

_GiB = 1024 ** 3


def test_pressure_release_returns_real_unused_pinned_blocks_to_os() -> None:
    """#579: Python GC/device empty-cache do not flush the host allocator."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real pinned-host allocator test requires CUDA")
    if not callable(getattr(torch.cuda.memory, "host_memory_stats", None)):
        pytest.skip("installed PyTorch does not expose host allocator stats")

    # Start from an empty reusable pool so this allocation cannot hide inside
    # a block left by another CUDA test in the same process.
    memory_mod.release_unused_pinned_host_cache()
    baseline = int(torch.cuda.memory.host_memory_stats().get(
        "allocated_bytes.current", 0))
    block = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    block.fill_(1)  # fault every pinned page into this process's working set
    torch.cuda.synchronize()
    live = int(torch.cuda.memory.host_memory_stats().get(
        "allocated_bytes.current", 0))
    assert live >= baseline + block.numel()

    del block
    gc.collect()
    torch.cuda.synchronize()
    cached = int(torch.cuda.memory.host_memory_stats().get(
        "allocated_bytes.current", 0))
    assert cached >= live  # freed tensor is still owned by the host cache

    released = memory_mod.release_unused_pinned_host_cache()
    after = int(torch.cuda.memory.host_memory_stats().get(
        "allocated_bytes.current", 0))
    assert released >= cached - after > 0
    assert after <= baseline


def _resident_file_bytes(path: Path) -> int:
    """Kernel page-cache residency without faulting file contents."""
    if os.name != "posix" or not hasattr(os, "POSIX_FADV_DONTNEED"):
        pytest.skip("real page-cache test requires POSIX file advice")
    size = path.stat().st_size
    if size <= 0:
        return 0
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    pages = (size + page_size - 1) // page_size
    libc = ctypes.CDLL(None, use_errno=True)
    mincore = getattr(libc, "mincore", None)
    if mincore is None:
        pytest.skip("real page-cache test requires mincore")
    mincore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    mincore.restype = ctypes.c_int
    fd = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
        try:
            states = (ctypes.c_ubyte * pages)()
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            if mincore(address, size, states) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return sum(bool(state & 1) for state in states) * page_size
        finally:
            mapping.close()
    finally:
        os.close(fd)


def _warm_file(path: Path) -> None:
    with path.open("rb", buffering=0) as stream:
        while stream.read(1024 * 1024):
            pass


def _res(events: list | None = None, budget_gb: int = 24) -> Residency:
    ev = events if events is not None else []
    return Residency(
        on_event=lambda ref, state, vb, dur=0: ev.append((ref, state, vb)),
        vram_budget_bytes=budget_gb * _GiB,
    )


class _Pipe:
    def to(self, device: str) -> "_Pipe":
        return self


# --------------------------------------------------------------------------- #
# Size-aware demote floor
# --------------------------------------------------------------------------- #


def test_ram_floor_adapts_to_small_hosts(monkeypatch) -> None:
    """A 16GiB dev box uses a fractional floor (3.2GiB), not the flat 8GiB —
    otherwise small hosts could never hold a warm tier at all."""
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 16.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 10.0)
    res = _res()
    res.track_vram("m/a", _Pipe(), vram_bytes=6 * _GiB)
    # 10 - 6 = 4 >= 3.2 floor -> allowed on the small host…
    assert res.demote("m/a") is True
    # …but the same numbers on a big host (8GiB floor) refuse.
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 64.0)
    res2 = _res()
    res2.track_vram("m/b", _Pipe(), vram_bytes=6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 12.0)
    assert res2.demote("m/b") is False
    assert res2.tier("m/b") is Tier.VRAM
    # More headroom (20 - 6 >= 8 floor) clears the big-host refusal.
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 20.0)
    assert res2.demote("m/b") is True
    assert res2.tier("m/b") is Tier.RAM


# --------------------------------------------------------------------------- #
# Host-RAM headroom and warm-tier victim selection
# --------------------------------------------------------------------------- #


def test_host_ram_headroom_includes_derived_floor(monkeypatch) -> None:
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 7.3)
    res = _res()
    headroom = res.host_ram_headroom(6 * _GiB)
    assert headroom.floor_bytes == pytest.approx(6.2 * _GiB, rel=1e-6)
    assert headroom.required_bytes == 6 * _GiB + headroom.floor_bytes
    assert headroom.available_bytes == pytest.approx(7.3 * _GiB, rel=1e-6)
    assert headroom.sufficient is False


def test_lru_ram_victims_skip_pinned_and_executing(tmp_path: Path) -> None:
    res = _res()
    res.track_ram("m/pinned", _Pipe(), path=tmp_path)
    res._entries["m/pinned"].pinned = True
    res.track_ram("m/busy", _Pipe(), path=tmp_path)
    res.track_ram("m/idle", _Pipe(), path=tmp_path)
    with res.executing("m/busy"):
        assert res.lru_ram_victims() == ["m/idle"]


# --------------------------------------------------------------------------- #
# Executor load gate: refuse RETRYABLE instead of thrashing; never disable fn
# --------------------------------------------------------------------------- #


class _In(msgspec.Struct):
    x: str


class _FakePipe:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def to(self, device: str) -> "_FakePipe":
        return self

    def enable_model_cpu_offload(self) -> None:
        pass


def _spec(
    name: str, cls: type, models: dict, *, vram_gb: float | None = None,
) -> EndpointSpec:
    return EndpointSpec(
        name=name, method=cls.run, kind="inference", payload_type=_In,
        output_mode="single", cls=cls, attr_name="run", models=models,
        is_async=inspect.iscoroutinefunction(cls.run),
        resources=Resources(vram_gb=vram_gb),
    )


def _executor(
    specs, tmp_path: Path, sent: list[pb.WorkerMessage] | None = None,
    monkeypatch=None,
) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        if sent is not None:
            sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=24 * _GiB)

    async def _fake_ensure_local(ref, **kwargs) -> Path:
        store.residency.track_disk(ref, tmp_path)
        return tmp_path

    if monkeypatch is not None:
        monkeypatch.setattr(executor_mod, "ensure_local", _fake_ensure_local)
    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return Executor(specs, _send, store=store)


def _host_ram_error(
    *, evicted_refs: tuple[str, ...] = (),
) -> InsufficientHostRamError:
    return InsufficientHostRamError(
        "generate",
        incoming_bytes=6 * _GiB,
        floor_bytes=6 * _GiB,
        required_bytes=12 * _GiB,
        available_before_bytes=8 * _GiB,
        available_after_bytes=8 * _GiB,
        evicted_refs=evicted_refs,
    )


def test_failed_setup_rolls_back_exact_ownership_before_fresh_reload(
    tmp_path: Path, monkeypatch,
) -> None:
    """A failure after residency registration cannot leak a stale instance.

    This uses the real setup, typed injection, residency registration,
    compile-target publication seam, rollback, and retry path. Only the tiny
    local pipeline and hardware probes are deterministic substitutes.
    """
    shutdown_markers: list[int] = []
    constructed = 0

    class Endpoint:
        def __init__(self) -> None:
            nonlocal constructed
            constructed += 1
            self.marker = constructed

        def setup(self, pipeline: _FakePipe) -> None:
            self.pipeline = pipeline

        def shutdown(self) -> None:
            shutdown_markers.append(self.marker)

        def run(self, ctx, payload: _In) -> _In:  # pragma: no cover
            return payload

    ref = "acme/reload"
    spec = _spec("generate", Endpoint, {"pipeline": HF(ref)})
    ex = _executor([spec], tmp_path, monkeypatch=monkeypatch)
    install = ex._install_compile_targets
    attempts = 0

    def fail_first_publish(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected post-registration failure")
        return install(*args, **kwargs)

    monkeypatch.setattr(ex, "_install_compile_targets", fail_first_publish)

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="post-registration"):
            await ex.ensure_setup(spec)

        rec = ex._classes[spec.instance_key]
        assert not rec.ready
        assert rec.instance is None
        assert rec.server is None
        assert rec.held_refs == []
        assert rec.held_objects == {}
        assert rec.shared_keys == []
        assert rec.compile_targets == {}
        assert ex.store.residency.tier(ref) is Tier.DISK
        assert shutdown_markers == [1]

        fresh = await ex.ensure_setup(spec)
        assert fresh.marker == 2
        assert fresh is rec.instance
        assert rec.ready
        assert rec.held_refs == [ref]
        assert ex.store.residency.obj(ref) is fresh.pipeline
        assert shutdown_markers == [1]

    asyncio.run(_run())


def test_setup_refused_retryable_when_host_ram_insufficient(tmp_path: Path, monkeypatch) -> None:
    from gen_worker.models import disk_gc

    class Endpoint:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"m": HF("acme/big")})
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 8.0)

    async def _run() -> None:
        ex = _executor([spec], tmp_path, monkeypatch=monkeypatch)
        with pytest.raises(InsufficientHostRamError, match="insufficient host RAM") as caught:
            await ex.ensure_setup(spec)
        err = caught.value
        assert err.incoming_bytes == 6 * _GiB
        assert err.floor_bytes == pytest.approx(6.2 * _GiB, rel=1e-6)
        assert err.required_bytes == err.incoming_bytes + err.floor_bytes
        assert err.available_before_bytes == 8 * _GiB
        assert err.available_after_bytes == 8 * _GiB
        assert err.evicted_refs == ()
        assert "6.0GiB incoming + 6.2GiB safety floor = 12.2GiB required" in str(err)
        assert "8.0GiB available before eviction, 8.0GiB after" in str(err)
        # Transient pressure: the function is NOT disabled and not failed.
        assert "ep" not in ex.unavailable
        assert ex._classes[spec.instance_key].failed is None

        # Pressure gone -> the retry loads normally.
        monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 24.0)
        inst = await ex.ensure_setup(spec)
        assert isinstance(inst.m, _FakePipe)

    asyncio.run(_run())


def test_pressure_eviction_reclaims_real_snapshot_cache_and_protects_shared_inode(
    tmp_path: Path, monkeypatch,
) -> None:
    """#579: exercise the real owner-teardown and kernel file-advice path.

    Cgroup-equivalent headroom is derived from real ``mincore`` residency, not
    fake object reachability.  Without ``POSIX_FADV_DONTNEED`` the old snapshot
    stays hot, admission remains below the exact requirement, and this test
    raises ``InsufficientHostRamError``.  A shared VAE hardlink proves that an
    active owner's inode is protected even when reachable through the victim
    tree.
    """
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("real page-cache test requires os.posix_fadvise")

    victim_dir = tmp_path / "victim"
    vae_dir = tmp_path / "shared-vae"
    incoming_dir = tmp_path / "incoming"
    for directory in (victim_dir, vae_dir, incoming_dir):
        directory.mkdir()
    victim_file = victim_dir / "weights.bin"
    vae_file = vae_dir / "vae.bin"
    incoming_file = incoming_dir / "weights.bin"
    victim_file.write_bytes(b"v" * (32 * 1024 * 1024))
    vae_file.write_bytes(b"a" * (8 * 1024 * 1024))
    incoming_file.write_bytes(b"n" * (8 * 1024 * 1024))
    os.link(vae_file, victim_dir / "shared-vae.bin")
    for file_path in (victim_file, vae_file, incoming_file):
        with file_path.open("rb") as stream:
            os.fsync(stream.fileno())
        fd = os.open(file_path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        _warm_file(file_path)

    victim_size = victim_file.stat().st_size
    assert _resident_file_bytes(victim_file) >= victim_size
    assert _resident_file_bytes(vae_file) >= vae_file.stat().st_size
    assert _resident_file_bytes(incoming_file) >= incoming_file.stat().st_size

    class Endpoint:
        def setup(self, pipeline: _FakePipe) -> None:
            self.pipeline = pipeline

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    old_ref = "tensorhub/old"
    vae_ref = "tensorhub/shared-vae"
    old_spec = _spec("old", Endpoint, {"pipeline": HF(old_ref)})
    incoming_spec = _spec(
        "incoming", Endpoint, {"pipeline": HF("tensorhub/incoming")},
    )
    ex = _executor([old_spec, incoming_spec], tmp_path / "cache")
    res = ex.store.residency
    old_pipeline = _FakePipe()
    old_vae = _FakePipe()
    survivor_vae = _FakePipe()
    res.track_ram(old_ref, old_pipeline, path=victim_dir)
    res.track_ram(vae_ref, survivor_vae, path=vae_dir)

    old_record = ex._classes[old_spec.instance_key]
    old_record.instance = SimpleNamespace(pipeline=old_pipeline, vae=old_vae)
    old_record.ready = True
    old_record.held_refs = [old_ref, vae_ref]
    old_record.held_objects = {old_ref: old_pipeline, vae_ref: old_vae}
    ex._classes["shared-vae-survivor"] = _ClassRecord(
        cls=Endpoint,
        instance=SimpleNamespace(vae=survivor_vae),
        ready=True,
        held_refs=[vae_ref],
        held_objects={vae_ref: survivor_vae},
    )

    floor = 8 * 1024 * 1024
    baseline = 4 * 1024 * 1024

    def _headroom(needed: int) -> HostRamHeadroom:
        reclaimed = victim_size - min(victim_size, _resident_file_bytes(victim_file))
        return HostRamHeadroom(
            available_bytes=baseline + reclaimed,
            floor_bytes=floor,
            required_bytes=int(needed) + floor,
        )

    monkeypatch.setattr(res, "host_ram_headroom", _headroom)

    async def _run() -> None:
        async with ex._load_lock:
            await ex._ensure_host_ram_for(
                incoming_spec, {"pipeline": str(incoming_dir)},
            )

    asyncio.run(_run())
    assert old_record.ready is False
    assert res.tier(old_ref) is Tier.DISK
    assert res.tier(vae_ref) is Tier.RAM
    assert _resident_file_bytes(victim_file) < victim_size // 4
    assert _resident_file_bytes(vae_file) >= vae_file.stat().st_size
    assert _resident_file_bytes(incoming_file) >= incoming_file.stat().st_size
    assert victim_file.read_bytes() == b"v" * victim_size
    assert vae_file.read_bytes() == b"a" * vae_file.stat().st_size


@pytest.mark.parametrize("owned_by_record", [True, False])
def test_pressure_pinned_release_runs_after_owner_teardown_and_stops_eviction(
    tmp_path: Path, monkeypatch, owned_by_record: bool,
) -> None:
    """#579: production admission must wire host-cache release in order.

    The controlled release is the only operation that can satisfy measured
    headroom.  Removing it, moving it before owner teardown, skipping the
    immediate re-probe, or falling through to file advice must fail here.
    Record-owned and ownerless residency entries share the same contract.
    """
    from gen_worker.models import disk_gc

    class Endpoint:
        def setup(self, pipeline: _FakePipe) -> None:
            self.pipeline = pipeline

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    victim_ref = "tensorhub/pinned-victim"
    survivor_ref = "tensorhub/warm-survivor"
    victim_spec = _spec("victim", Endpoint, {"pipeline": HF(victim_ref)})
    survivor_spec = _spec(
        "survivor", Endpoint, {"pipeline": HF(survivor_ref)},
    )
    incoming_spec = _spec(
        "incoming", Endpoint, {"pipeline": HF("tensorhub/incoming")},
    )
    sent: list[pb.WorkerMessage] = []
    ex = _executor(
        [victim_spec, survivor_spec, incoming_spec], tmp_path, sent,
    )
    res = ex.store.residency
    victim_dir = tmp_path / "victim"
    incoming_dir = tmp_path / "incoming"
    victim_dir.mkdir()
    incoming_dir.mkdir()

    victim = _FakePipe()
    victim_weak = weakref.ref(victim)
    survivor = _FakePipe()
    res.track_ram(victim_ref, victim, path=victim_dir)
    res.track_ram(survivor_ref, survivor, path=tmp_path / "survivor")
    victim_record = ex._classes[victim_spec.instance_key]
    if owned_by_record:
        victim_record.instance = SimpleNamespace(pipeline=victim)
        victim_record.ready = True
        victim_record.held_refs = [victim_ref]
        victim_record.held_objects = {victim_ref: victim}
    survivor_record = ex._classes[survivor_spec.instance_key]
    survivor_record.instance = SimpleNamespace(pipeline=survivor)
    survivor_record.ready = True
    survivor_record.held_refs = [survivor_ref]
    survivor_record.held_objects = {survivor_ref: survivor}
    del victim

    released = {"host_cache": False}

    def _headroom(needed: int) -> HostRamHeadroom:
        available = 32 * _GiB if released["host_cache"] else 1 * _GiB
        return HostRamHeadroom(
            available_bytes=available,
            floor_bytes=4 * _GiB,
            required_bytes=int(needed) + 4 * _GiB,
        )

    def _release_pinned() -> int:
        assert res.tier(victim_ref) is Tier.DISK
        assert res.obj(victim_ref) is None
        if owned_by_record:
            assert victim_record.ready is False
            assert victim_record.instance is None
            assert victim_record.held_objects == {}
        assert victim_weak() is None
        released["host_cache"] = True
        return 9 * _GiB

    async def _unexpected_file_advice(*_args, **_kwargs) -> int:
        raise AssertionError("sufficient pinned release must skip file advice")

    monkeypatch.setattr(disk_gc, "tree_bytes", lambda _path: 4 * _GiB)
    monkeypatch.setattr(res, "host_ram_headroom", _headroom)
    monkeypatch.setattr(
        executor_mod, "release_unused_pinned_host_cache", _release_pinned,
    )
    ex._reclaim_released_file_cache = _unexpected_file_advice  # type: ignore[method-assign]

    async def _run() -> None:
        await ex._record_host_ram_failure(
            ["tensorhub/blocked"],
            InsufficientHostRamError(
                "incoming",
                incoming_bytes=4 * _GiB,
                floor_bytes=4 * _GiB,
                required_bytes=8 * _GiB,
                available_before_bytes=1 * _GiB,
                available_after_bytes=1 * _GiB,
            ),
        )
        async with ex._load_lock:
            await ex._ensure_host_ram_for(
                incoming_spec, {"pipeline": str(incoming_dir)},
            )

    asyncio.run(_run())
    assert released["host_cache"] is True
    assert res.tier(survivor_ref) is Tier.RAM
    assert survivor_record.ready is True
    progress = [
        message.model_event for message in sent
        if message.WhichOneof("msg") == "model_event"
        and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
    ]
    assert len(progress) == 1
    assert progress[0].host_ram_available_before_bytes == 1 * _GiB
    assert progress[0].host_ram_available_after_bytes == 32 * _GiB
    assert list(progress[0].host_ram_evicted_refs) == [victim_ref]


def test_capacity_progress_requires_measured_owner_release(
    tmp_path: Path, monkeypatch,
) -> None:
    """pgw#548: a release is not progress until measured headroom fits.

    The first idle owner teardown improves RAM but remains below the exact
    failed requirement, so it must not emit. The second teardown crosses the
    requirement and emits one typed, generation-fenced event carrying the ref
    from that exact satisfying transition. The same event is replayed on stream reconnect; a
    fresh executor (process restart) has no stale in-memory progress to emit.
    """
    from gen_worker.models import disk_gc

    class A:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    class B(A):
        pass

    class C(A):
        pass

    spec_a = _spec("a", A, {"m": HF("acme/a")})
    spec_b = _spec("b", B, {"m": HF("acme/b")})
    spec_c = _spec("c", C, {"m": HF("acme/c")})
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    available_gb = {"value": 24.0}
    monkeypatch.setattr(
        residency_mod, "get_available_ram_gb", lambda: available_gb["value"],
    )
    sent: list[pb.WorkerMessage] = []

    async def _run() -> None:
        ex = _executor(
            [spec_a, spec_b, spec_c], tmp_path, sent, monkeypatch=monkeypatch,
        )
        await ex.ensure_setup(spec_a)
        await ex.ensure_setup(spec_b)
        rec_a = ex._classes[spec_a.instance_key]
        rec_b = ex._classes[spec_b.instance_key]

        available_gb["value"] = 8.0
        with ex.store.residency.executing("acme/a", "acme/b"):
            with pytest.raises(InsufficientHostRamError):
                await ex.ensure_setup(spec_c)

        failures = [
            message.model_event for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_FAILED
            and message.model_event.error == "insufficient_host_ram"
        ]
        assert len(failures) == 1
        failure = failures[0]
        assert failure.ref == "acme/c"
        assert failure.host_ram_required_bytes == pytest.approx(12.2 * _GiB, rel=1e-6)
        assert failure.host_ram_available_before_bytes == 8 * _GiB
        assert failure.host_ram_available_after_bytes == 8 * _GiB
        assert list(failure.host_ram_evicted_refs) == []
        assert failure.host_ram_capacity_generation == 1

        # One genuine owner release raises measured headroom, but not enough.
        available_gb["value"] = 10.0
        rec_a.stale = True
        await ex._revalidate_record(rec_a)
        assert not [
            message for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]

        # The next owner release crosses the remembered numeric requirement.
        available_gb["value"] = 16.0
        rec_b.stale = True
        await ex._revalidate_record(rec_b)
        progress = [
            message.model_event for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]
        assert len(progress) == 1
        event = progress[0]
        assert event.ref == "acme/c"
        assert event.host_ram_required_bytes == failure.host_ram_required_bytes
        assert event.host_ram_available_before_bytes == 10 * _GiB
        assert event.host_ram_available_after_bytes == 16 * _GiB
        assert list(event.host_ram_evicted_refs) == ["acme/b"]
        assert event.host_ram_capacity_generation == 2

        # Once progress supersedes the active block, reconnect replay retains
        # only the self-contained progress. Replaying obsolete FAILED would
        # strand an older hub that safely ignores the additive progress enum.
        replay = await ex.host_ram_capacity_replay()
        assert [message.model_event.state for message in replay] == [
            pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
        ]
        assert replay[0].model_event.SerializeToString(deterministic=True) == (
            event.SerializeToString(deterministic=True)
        )

        restarted_sent: list[pb.WorkerMessage] = []
        restarted = _executor([spec_a, spec_b, spec_c], tmp_path, restarted_sent)
        assert await restarted.host_ram_capacity_replay() == []

    asyncio.run(_run())


def test_host_capacity_batches_commit_before_cancelled_send(
    tmp_path: Path, monkeypatch,
) -> None:
    """Multi-ref failure/progress state is atomic and replayable.

    The first send deliberately backpressures and then gets cancelled. Its
    callback must still be able to acquire the capacity lock, proving no send
    is awaited while state is locked. Every causal ref is visible to replay
    before that first send completes.
    """
    async def _run() -> None:
        ex = _executor([], tmp_path)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _backpressured_send(message: pb.WorkerMessage) -> None:
            await asyncio.wait_for(ex._host_ram_lock.acquire(), 0.1)
            ex._host_ram_lock.release()
            entered.set()
            await release.wait()

        ex._send = _backpressured_send
        failure_task = asyncio.create_task(ex._record_host_ram_failure(
            ["acme/b", "acme/a"], _host_ram_error(),
        ))
        await asyncio.wait_for(entered.wait(), 0.2)
        replay = await asyncio.wait_for(ex.host_ram_capacity_replay(), 0.1)
        assert [message.model_event.ref for message in replay] == ["acme/a", "acme/b"]
        assert all(
            message.model_event.state == pb.MODEL_STATE_FAILED for message in replay
        )
        failure_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await failure_task

        entered.clear()
        monkeypatch.setattr(
            ex.store.residency,
            "host_ram_headroom",
            lambda _incoming: SimpleNamespace(available_bytes=16 * _GiB),
        )
        progress_task = asyncio.create_task(
            ex._observe_host_ram_progress(["acme/released"])
        )
        await asyncio.wait_for(entered.wait(), 0.2)
        replay = await asyncio.wait_for(ex.host_ram_capacity_replay(), 0.1)
        assert [message.model_event.state for message in replay] == [
            pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
            pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
        ]
        assert [message.model_event.ref for message in replay] == [
            "acme/a", "acme/b",
        ]
        progress_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await progress_task

        # Cancellation/backpressure never discards the committed progress.
        # A later active block replays before those satisfied refs, while the
        # obsolete failures for a/b stay absent for old-hub compatibility.
        async def _discard(_message: pb.WorkerMessage) -> None:
            pass

        ex._send = _discard
        await ex._record_host_ram_failure(["acme/c"], _host_ram_error())
        replay = await ex.host_ram_capacity_replay()
        assert [message.model_event.state for message in replay] == [
            pb.MODEL_STATE_FAILED,
            pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
            pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
        ]
        assert [message.model_event.ref for message in replay] == [
            "acme/c", "acme/a", "acme/b",
        ]

    asyncio.run(_run())


def test_host_capacity_generations_enqueue_in_commit_order(tmp_path: Path) -> None:
    async def _run() -> None:
        ex = _executor([], tmp_path)
        entered = asyncio.Event()
        release = asyncio.Event()
        sent: list[tuple[int, str]] = []

        async def _ordered_send(message: pb.WorkerMessage) -> None:
            event = message.model_event
            sent.append((int(event.host_ram_capacity_generation), event.ref))
            if len(sent) == 1:
                entered.set()
                await release.wait()

        ex._send = _ordered_send
        first = asyncio.create_task(
            ex._record_host_ram_failure(["acme/a"], _host_ram_error())
        )
        await asyncio.wait_for(entered.wait(), 0.2)
        second = asyncio.create_task(
            ex._record_host_ram_failure(["acme/b"], _host_ram_error())
        )
        await asyncio.sleep(0)
        replay = await ex.host_ram_capacity_replay()
        assert [
            (int(message.model_event.host_ram_capacity_generation),
             message.model_event.ref)
            for message in replay
        ] == [(1, "acme/a"), (2, "acme/b")]

        release.set()
        await asyncio.gather(first, second)
        assert sent == [(1, "acme/a"), (2, "acme/b")]

    asyncio.run(_run())


def test_delivered_progress_retires_distinct_satisfied_refs(
    tmp_path: Path, monkeypatch,
) -> None:
    async def _run() -> None:
        sent: list[pb.WorkerMessage] = []
        ex = _executor([], tmp_path, sent)
        refs = [f"acme/model-{index}" for index in range(5)]
        await ex._record_host_ram_failure(refs, _host_ram_error())
        monkeypatch.setattr(
            ex.store.residency,
            "host_ram_headroom",
            lambda _incoming: SimpleNamespace(available_bytes=16 * _GiB),
        )
        await ex._observe_host_ram_progress(["acme/released"])
        progress = [
            message.model_event for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]
        assert len(progress) == 5
        assert len(ex._host_ram_progress) == 5
        lifecycle = Lifecycle(
            Settings(orchestrator_public_addr="localhost:1"), ex,
        )

        stale = pb.ModelEvent()
        stale.CopyFrom(progress[0])
        stale.host_ram_capacity_generation -= 1
        await lifecycle.on_message_shipped(pb.WorkerMessage(model_event=stale))
        assert progress[0].ref in ex._host_ram_progress
        for event in progress:
            await lifecycle.on_message_shipped(pb.WorkerMessage(model_event=event))

        assert ex._host_ram_progress == {}
        assert await ex.host_ram_capacity_replay() == []

    asyncio.run(_run())


def test_progress_reports_only_the_satisfying_release_transition(
    tmp_path: Path, monkeypatch,
) -> None:
    async def _run() -> None:
        sent: list[pb.WorkerMessage] = []
        ex = _executor([], tmp_path, sent)
        await ex._record_host_ram_failure(["acme/blocked"], _host_ram_error())
        available = {"bytes": 9 * _GiB}
        monkeypatch.setattr(
            ex.store.residency,
            "host_ram_headroom",
            lambda _incoming: SimpleNamespace(available_bytes=available["bytes"]),
        )
        for index in range(5):
            await ex._observe_host_ram_progress([f"acme/unrelated-{index}"])
        available["bytes"] = 16 * _GiB
        await ex._observe_host_ram_progress(["acme/final-release"])

        progress = [
            message.model_event for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]
        assert len(progress) == 1
        assert list(progress[0].host_ram_evicted_refs) == ["acme/final-release"]

    asyncio.run(_run())


def test_hello_ack_replays_failure_before_preserved_results(
    tmp_path: Path,
) -> None:
    """Exercise the real pre-send HelloAck boundary with maxsize=1."""
    async def _run() -> None:
        settings = Settings(orchestrator_public_addr="localhost:1")
        ex = _executor([], tmp_path)
        lifecycle = Lifecycle(settings, ex)
        transport = Transport(settings, lifecycle, queue_maxsize=1)
        lifecycle.transport = transport
        ex._send = transport.send

        # Failure enters the old stream queue, followed by two durable results.
        await ex._record_host_ram_failure(["acme/incoming"], _host_ram_error())
        for request_id in ("r1", "r2"):
            await transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                request_id=request_id,
                attempt=1,
                status=pb.JOB_STATUS_RETRYABLE,
            )))
        await transport.queue.reset_for_reconnect()
        transport._connected.set()

        # on_hello_ack runs before Transport starts its send loop. It must not
        # block on the two results, and it must prepend the lost active failure.
        await asyncio.wait_for(lifecycle.on_hello_ack(pb.HelloAck()), 0.2)
        lifecycle._cancel_residency_reconcile()
        kind, message = await transport.queue.get()
        assert kind == "event"
        assert message.WhichOneof("msg") == "model_event"
        assert message.model_event.state == pb.MODEL_STATE_FAILED
        kind, message = await transport.queue.get()
        assert kind == "event"
        assert message.WhichOneof("msg") == "state_delta"
        kind, message = await transport.queue.get()
        assert kind == "result"
        assert message.job_result.request_id == "r1"

    asyncio.run(_run())


def test_hello_ack_baseline_bypasses_full_disconnected_event_lane(
    tmp_path: Path,
) -> None:
    """A real pre-send HelloAck cannot wait for its own not-yet-started sender."""
    async def _run() -> None:
        settings = Settings(orchestrator_public_addr="localhost:1")
        ex = _executor([], tmp_path)
        lifecycle = Lifecycle(settings, ex)
        transport = Transport(settings, lifecycle, queue_maxsize=1)
        lifecycle.transport = transport
        ex._send = transport.send

        await transport.queue.reset_for_reconnect()
        await transport.send(pb.WorkerMessage(model_event=pb.ModelEvent(
            ref="acme/already-on-disk",
            state=pb.MODEL_STATE_ON_DISK,
        )))
        transport._connected.set()

        await asyncio.wait_for(lifecycle.on_hello_ack(pb.HelloAck()), 0.2)
        lifecycle._cancel_residency_reconcile()
        first = await transport.queue.get()
        second = await transport.queue.get()
        assert first[1].WhichOneof("msg") == "state_delta"
        assert second[1].WhichOneof("msg") == "model_event"
        assert second[1].model_event.ref == "acme/already-on-disk"

    asyncio.run(_run())


def test_disconnected_capacity_failure_moves_ahead_of_results_once(
    tmp_path: Path,
) -> None:
    """Queued capacity evidence is promoted atomically on the real HelloAck path."""
    async def _run() -> None:
        settings = Settings(orchestrator_public_addr="localhost:1")
        ex = _executor([], tmp_path)
        lifecycle = Lifecycle(settings, ex)
        transport = Transport(settings, lifecycle, queue_maxsize=1)
        lifecycle.transport = transport
        ex._send = transport.send

        for request_id in ("r1", "r2"):
            await transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                request_id=request_id,
                attempt=1,
                status=pb.JOB_STATUS_RETRYABLE,
            )))
        await transport.queue.reset_for_reconnect()
        # This is emitted while disconnected, after results were requeued. It
        # begins in the ordinary lane behind them and must be moved, not copied.
        await ex._record_host_ram_failure(["acme/incoming"], _host_ram_error())
        transport._connected.set()

        await asyncio.wait_for(lifecycle.on_hello_ack(pb.HelloAck()), 0.2)
        await asyncio.wait_for(lifecycle.on_hello_ack(pb.HelloAck()), 0.2)
        lifecycle._cancel_residency_reconcile()
        got = [await transport.queue.get() for _ in range(4)]
        messages = [message for _kind, message in got]
        failures = [
            index for index, message in enumerate(messages)
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_FAILED
        ]
        results = [
            index for index, message in enumerate(messages)
            if message.WhichOneof("msg") == "job_result"
        ]
        assert failures == [0]
        assert len(results) == 2
        assert failures[0] < min(results)
        assert len(transport.queue) == 0

    asyncio.run(_run())


def test_undelivered_progress_replays_after_queue_reset(
    tmp_path: Path, monkeypatch,
) -> None:
    async def _run() -> None:
        settings = Settings(orchestrator_public_addr="localhost:1")
        ex = _executor([], tmp_path)
        lifecycle = Lifecycle(settings, ex)
        transport = Transport(settings, lifecycle, queue_maxsize=1)
        lifecycle.transport = transport
        ex._send = transport.send

        await ex._record_host_ram_failure(["acme/incoming"], _host_ram_error())
        monkeypatch.setattr(
            ex.store.residency,
            "host_ram_headroom",
            lambda _incoming: SimpleNamespace(available_bytes=16 * _GiB),
        )
        await ex._observe_host_ram_progress(["acme/released"])
        assert "acme/incoming" in ex._host_ram_progress

        await transport.queue.reset_for_reconnect()
        transport._connected.set()
        await asyncio.wait_for(lifecycle.on_hello_ack(pb.HelloAck()), 0.2)
        lifecycle._cancel_residency_reconcile()
        _kind, message = await transport.queue.get()
        assert message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        assert await transport.queue.should_ship_capacity(message)
        await transport.queue.mark_event_shipped(message)
        await lifecycle.on_message_shipped(message)
        assert ex._host_ram_progress == {}

    asyncio.run(_run())


def test_host_capacity_events_never_expose_shared_cache_ids(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        ex = _executor([], tmp_path)
        key = LoadedComponentKey.for_component(
            content_digest="digest", component="text_encoder",
        )
        cache_id = key.cache_id()
        await ex._record_host_ram_failure(
            ["acme/incoming"],
            _host_ram_error(evicted_refs=(cache_id, "acme/model")),
        )
        replay = await ex.host_ram_capacity_replay()
        assert list(replay[0].model_event.host_ram_evicted_refs) == ["acme/model"]

        # The actual shared-owner teardown path also keeps its opaque key
        # local instead of presenting it as a canonical model ref.
        ex.store.residency.acquire_shared(key, object)
        rec = _ClassRecord(cls=object, ready=True, shared_keys=[key])
        observed: list[str] = []

        async def _capture(refs: list[str], *, collect_host: bool = False) -> None:
            observed.extend(refs)

        ex._observe_host_ram_progress = _capture  # type: ignore[method-assign]
        await ex._vacate_record(rec)
        assert cache_id not in observed

    asyncio.run(_run())


def test_teardown_collects_cyclic_host_owner_without_flush_memory(
    tmp_path: Path, monkeypatch,
) -> None:
    """Actual teardown collects host cycles before its capacity probe only."""
    class CyclicEndpoint:
        def __init__(self) -> None:
            self.cycle = self

    async def _run() -> None:
        sent: list[pb.WorkerMessage] = []
        ex = _executor([], tmp_path, sent)
        endpoint = CyclicEndpoint()
        endpoint_ref = weakref.ref(endpoint)
        rec = _ClassRecord(cls=CyclicEndpoint, instance=endpoint, ready=True)
        endpoint = None  # type: ignore[assignment]

        monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
        monkeypatch.setattr(
            residency_mod,
            "get_available_ram_gb",
            lambda: 8.0 if endpoint_ref() is not None else 16.0,
        )
        await ex._record_host_ram_failure(["acme/incoming"], _host_ram_error())

        def _forbidden_flush() -> None:
            raise AssertionError("teardown capacity probe must not call flush_memory")

        monkeypatch.setattr("gen_worker.executor.flush_memory", _forbidden_flush)
        await ex._vacate_record(rec)

        assert endpoint_ref() is None
        progress = [
            message.model_event for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]
        assert len(progress) == 1
        assert progress[0].host_ram_available_before_bytes == 8 * _GiB
        assert progress[0].host_ram_available_after_bytes == 16 * _GiB

    asyncio.run(_run())


def test_capacity_progress_observes_shutdown_endpoint_collection(
    tmp_path: Path, monkeypatch,
) -> None:
    """The vacate frame must not retain an endpoint through ``shutdown``.

    Endpoint shutdown hooks are bound methods, so keeping the local callable
    alive through the cgroup probe also keeps every pipeline on ``self`` alive.
    This models available RAM from the endpoint's actual reachability and
    proves progress is emitted only after the last departing owner is gone.
    """
    from gen_worker.models import disk_gc

    class Old:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        def shutdown(self) -> None:
            pass

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    class Incoming:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    old_spec = _spec("old", Old, {"m": HF("acme/old")})
    incoming_spec = _spec("incoming", Incoming, {"m": HF("acme/incoming")})
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    pressure = {"active": False}
    endpoint: weakref.ReferenceType[object] | None = None

    def _available() -> float:
        if not pressure["active"]:
            return 24.0
        assert endpoint is not None
        return 8.0 if endpoint() is not None else 24.0

    monkeypatch.setattr(residency_mod, "get_available_ram_gb", _available)
    sent: list[pb.WorkerMessage] = []

    async def _run() -> None:
        nonlocal endpoint
        ex = _executor(
            [old_spec, incoming_spec], tmp_path, sent, monkeypatch=monkeypatch,
        )
        await ex.ensure_setup(old_spec)
        rec = ex._classes[old_spec.instance_key]
        endpoint = weakref.ref(rec.instance)
        pressure["active"] = True

        # Pinning the old owner forces an honest failed admission and records
        # the exact required capacity without evicting it in the same attempt.
        with ex.store.residency.executing("acme/old"):
            with pytest.raises(InsufficientHostRamError):
                await ex.ensure_setup(incoming_spec)

        await ex._vacate_record(rec)
        assert endpoint() is None
        progress = [
            message.model_event for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]
        assert len(progress) == 1
        assert progress[0].ref == "acme/incoming"
        assert progress[0].host_ram_available_before_bytes == 8 * _GiB
        assert progress[0].host_ram_available_after_bytes == 24 * _GiB
        assert list(progress[0].host_ram_evicted_refs) == ["acme/old"]

    asyncio.run(_run())


def test_runjob_pin_release_emits_measured_capacity_progress(
    tmp_path: Path, monkeypatch,
) -> None:
    """Concurrent admission stays blocked until a real RunJob pin releases."""
    from gen_worker.models import disk_gc

    started = asyncio.Event()
    finish = asyncio.Event()
    running = {"value": False}

    class Busy:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        async def run(self, ctx, payload: _In) -> _In:
            running["value"] = True
            started.set()
            await finish.wait()
            running["value"] = False
            return payload

    class Incoming:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    busy_spec = _spec("busy", Busy, {"m": HF("acme/busy")})
    incoming_spec = _spec("incoming", Incoming, {"m": HF("acme/incoming")})
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(
        residency_mod,
        "get_available_ram_gb",
        lambda: 8.0 if running["value"] else 24.0,
    )
    sent: list[pb.WorkerMessage] = []

    async def _run() -> None:
        ex = _executor(
            [busy_spec, incoming_spec], tmp_path, sent, monkeypatch=monkeypatch,
        )
        await ex.ensure_setup(busy_spec)
        run = pb.RunJob(
            request_id="busy-job",
            attempt=1,
            function_name="busy",
            input_payload=msgspec.msgpack.encode(_In(x="work")),
        )
        await ex.handle_run_job(run)
        await asyncio.wait_for(started.wait(), 1)

        # The concurrent load cannot evict the executing record and records a
        # typed block at the observed low headroom.
        with pytest.raises(InsufficientHostRamError):
            await ex.ensure_setup(incoming_spec)
        assert not [
            message for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]

        # A pin release is not model teardown. Capacity observation may yield
        # and probe host cgroup state, but must not run flush_memory (which
        # empties CUDA cache and resets peak-memory accounting).
        flush_calls = 0

        def _unexpected_flush() -> None:
            nonlocal flush_calls
            flush_calls += 1

        monkeypatch.setattr("gen_worker.executor.flush_memory", _unexpected_flush)
        finish.set()
        job = ex.jobs[(run.request_id, run.attempt)]
        assert job.task is not None
        await asyncio.wait_for(job.task, 1)
        assert flush_calls == 0
        progress = [
            message.model_event for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS
        ]
        assert len(progress) == 1
        assert progress[0].ref == "acme/incoming"
        assert progress[0].host_ram_available_before_bytes == 8 * _GiB
        assert progress[0].host_ram_available_after_bytes == 24 * _GiB
        assert list(progress[0].host_ram_evicted_refs) == []

    asyncio.run(_run())


def test_host_capacity_wire_is_ignored_by_pre_548_hub_descriptor() -> None:
    """The additive fields/enum parse under the previous protocol-v3 shape."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    file = descriptor_pb2.FileDescriptorProto(
        name="worker_scheduler_pre_548.proto",
        package="cozy.scheduler.pre548",
        syntax="proto3",
    )
    state = file.enum_type.add(name="ModelState")
    for name, number in (
        ("MODEL_STATE_UNSPECIFIED", 0),
        ("MODEL_STATE_DOWNLOADING", 1),
        ("MODEL_STATE_ON_DISK", 2),
        ("MODEL_STATE_IN_RAM", 3),
        ("MODEL_STATE_IN_VRAM", 4),
        ("MODEL_STATE_EVICTED", 5),
        ("MODEL_STATE_FAILED", 6),
        ("MODEL_STATE_ADOPTED", 7),
    ):
        state.value.add(name=name, number=number)

    model_event = file.message_type.add(name="ModelEvent")
    fields = (
        ("ref", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, ""),
        ("state", 2, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
         ".cozy.scheduler.pre548.ModelState"),
        ("vram_bytes", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, ""),
        ("error", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, ""),
        ("bytes_done", 5, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, ""),
        ("bytes_total", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, ""),
        ("duration_ms", 7, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, ""),
        ("cache_hits", 8, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, ""),
        ("cache_misses", 9, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, ""),
        ("warmup_s", 10, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE, ""),
    )
    for name, number, field_type, type_name in fields:
        field = model_event.field.add(
            name=name,
            number=number,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type=field_type,
        )
        if type_name:
            field.type_name = type_name

    worker_message = file.message_type.add(name="WorkerMessage")
    worker_message.oneof_decl.add(name="msg")
    field = worker_message.field.add(
        name="model_event",
        number=6,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".cozy.scheduler.pre548.ModelEvent",
    )
    field.oneof_index = 0
    descriptor = descriptor_pool.DescriptorPool().Add(file)
    OldWorkerMessage = message_factory.GetMessageClass(
        descriptor.message_types_by_name["WorkerMessage"]
    )

    progress = pb.WorkerMessage(model_event=pb.ModelEvent(
        ref="tensorhub/model:prod",
        state=pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
        host_ram_required_bytes=12 * _GiB,
        host_ram_available_before_bytes=10 * _GiB,
        host_ram_available_after_bytes=16 * _GiB,
        host_ram_evicted_refs=["tensorhub/old:prod"],
        host_ram_capacity_generation=2,
    ))
    old_progress = OldWorkerMessage.FromString(progress.SerializeToString())
    assert old_progress.WhichOneof("msg") == "model_event"
    assert old_progress.model_event.ref == "tensorhub/model:prod"
    assert old_progress.model_event.state == 8
    assert old_progress.model_event.state not in range(0, 8)  # old switch: default/ignore
    assert not hasattr(old_progress.model_event, "host_ram_required_bytes")

    failure = pb.WorkerMessage(model_event=pb.ModelEvent(
        ref="tensorhub/model:prod",
        state=pb.MODEL_STATE_FAILED,
        error="insufficient_host_ram",
        host_ram_required_bytes=12 * _GiB,
        host_ram_capacity_generation=1,
    ))
    old_failure = OldWorkerMessage.FromString(failure.SerializeToString())
    assert old_failure.model_event.state == 6
    assert old_failure.model_event.error == "insufficient_host_ram"
    assert not hasattr(old_failure.model_event, "host_ram_capacity_generation")


def test_setup_vacates_warm_record_after_vram_make_room(
    tmp_path: Path, monkeypatch
) -> None:
    """pgw#541: VRAM make-room can put the old pipeline into host RAM.

    Admission must observe that post-demotion pressure, tear down the owning
    endpoint record, and only then publish ON_DISK. Clearing Residency alone
    leaves ``record.instance`` strongly owning the pipeline and cannot reclaim
    the 6GiB represented by this deterministic probe.
    """
    from gen_worker.models import disk_gc

    class A:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    class B:
        def setup(self, m: _FakePipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec_a = _spec("a", A, {"m": HF("acme/a")})
    spec_b = _spec("b", B, {"m": HF("acme/b")}, vram_gb=18)
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)

    async def _run() -> None:
        ex = _executor([spec_a, spec_b], tmp_path, monkeypatch=monkeypatch)
        res = ex.store.residency
        from gen_worker import executor as executor_mod

        if executor_mod.torch is None:
            pytest.skip("torch is required for the VRAM-admission ordering check")
        cuda_enabled = {"value": False}
        monkeypatch.setattr(executor_mod.torch.cuda, "is_available", lambda: cuda_enabled["value"])

        monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 24.0)
        await ex.ensure_setup(spec_a)
        rec_a = ex._classes[spec_a.instance_key]
        pipeline = weakref.ref(rec_a.instance.m)
        # Model the first successful SDXL request: its ~6GiB pipeline is in
        # VRAM, and a 24GiB card needs to demote it before loading B.
        res.track_vram("acme/a", rec_a.instance.m, vram_bytes=6 * _GiB)

        cuda_enabled["value"] = True
        make_room = ex._make_room_for

        async def _make_room_then_use_cpu(spec, slots):
            await make_room(spec, slots)
            cuda_enabled["value"] = False

        monkeypatch.setattr(ex, "_make_room_for", _make_room_then_use_cpu)

        def _avail() -> float:
            if res.tier("acme/a") is Tier.VRAM:
                return 13.0
            return 13.0 if rec_a.instance is None else 7.0

        monkeypatch.setattr(residency_mod, "get_available_ram_gb", _avail)
        on_disk: list[bool] = []

        def _event(ref: str, state: str, _vram: int, _duration: int = 0) -> None:
            if ref == "acme/a" and state == residency_mod.ON_DISK:
                on_disk.append(rec_a.instance is None)

        res._on_event = _event
        # _run_job pins refs before setup. The pre-existing shared entry is
        # therefore execution-pinned here, but that incidental pin still must
        # not make either older record the incoming job's active instance.
        with res.executing("acme/shared-vae"):
            inst_b = await ex.ensure_setup(spec_b)
        assert isinstance(inst_b.m, _FakePipe)
        assert res.tier("acme/a") is Tier.DISK
        assert rec_a.ready is False
        assert rec_a.instance is None
        await asyncio.sleep(0)
        assert pipeline() is None
        assert on_disk == [True]

    asyncio.run(_run())


def test_shared_ref_does_not_pin_idle_record_or_publish_false_disk(
    tmp_path: Path, monkeypatch
) -> None:
    """pgw#542: a job's shared VAE is not ownership of every old pipeline."""
    from gen_worker.models import disk_gc

    class Endpoint:
        def setup(self, pipeline: _FakePipe, vae: _FakePipe) -> None:
            self.pipeline = pipeline
            self.vae = vae

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    shared = HF("acme/shared-vae")
    spec_a = _spec(
        "a", Endpoint, {"pipeline": HF("acme/a"), "vae": shared})
    spec_b = _spec(
        "b", Endpoint, {"pipeline": HF("acme/b"), "vae": shared})
    spec_c = _spec(
        "c", Endpoint, {"pipeline": HF("acme/c"), "vae": shared})
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)

    async def _run() -> None:
        ex = _executor(
            [spec_a, spec_b, spec_c], tmp_path, monkeypatch=monkeypatch,
        )
        res = ex.store.residency
        monkeypatch.setattr(
            residency_mod, "get_available_ram_gb", lambda: 24.0)
        await ex.ensure_setup(spec_a)
        await ex.ensure_setup(spec_c)
        rec_a = ex._classes[spec_a.instance_key]
        rec_c = ex._classes[spec_c.instance_key]

        async def _cached_local(ref, snapshot=None, *, binding=None) -> Path:
            if res.local_path(ref) is None:
                res.track_disk(ref, tmp_path)
            return tmp_path

        ex.store.ensure_local = _cached_local  # type: ignore[method-assign]

        # This is the production ordering: handle_run_job installs B before
        # its setup begins. B's shared VAE must not make A or C look active.
        ex.jobs[("incoming", 1)] = _Job("incoming", 1, spec_b)
        monkeypatch.setattr(
            residency_mod,
            "get_available_ram_gb",
            lambda: 20.0 if rec_a.instance is None else 8.0,
        )
        events: list[tuple[str, str]] = []
        res._on_event = lambda ref, state, *_: events.append((ref, state))
        sent: list[pb.WorkerMessage] = []

        async def _capture(message: pb.WorkerMessage) -> None:
            sent.append(message)

        ex._send = _capture

        inst_b = await ex.ensure_setup(spec_b)

        assert isinstance(inst_b.pipeline, _FakePipe)
        assert rec_a.ready is False
        assert rec_a.instance is None
        assert rec_c.ready is True
        assert res.tier("acme/a") is Tier.DISK
        assert res.tier("acme/shared-vae") is Tier.RAM
        assert ("acme/a", residency_mod.ON_DISK) in events
        assert ("acme/shared-vae", residency_mod.ON_DISK) not in events
        assert not any(
            message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_FAILED
            for message in sent
        )

        # The latest-loaded B owns Residency.obj(shared). Removing B while C
        # survives must transfer that strong reference to C, not retain B's
        # VAE or falsely publish shared as ON_DISK.
        rec_b = ex._classes[spec_b.instance_key]
        b_vae = weakref.ref(inst_b.vae)
        c_vae = rec_c.instance.vae
        ex.jobs.pop(("incoming", 1))
        events.clear()
        del inst_b
        await ex._vacate_record(rec_b)
        gc.collect()

        assert b_vae() is None
        assert res.obj("acme/shared-vae") is c_vae
        assert res.tier("acme/shared-vae") is Tier.RAM
        assert not [event for event in events if event[0] == "acme/shared-vae"]

    asyncio.run(_run())


def test_declarative_ninth_sdxl_load_uses_reclaimable_cgroup_cache(
    tmp_path: Path, monkeypatch
) -> None:
    """#543: inactive file cache must not look like live SDXL pipeline RAM.

    The injected cgroup observation stays at its byte limit while anonymous
    pipeline memory is replaced by inactive model-file cache.  Raw
    ``memory.current`` therefore never falls, exactly the false admission
    signal seen by e2e #170.  The production DesiredResidency reconciler must
    use reclaimable working-set headroom, vacate the minimum idle records, and
    load model nine without a timer or retry loop.
    """
    from gen_worker.models import disk_gc

    live: weakref.WeakSet[object] = weakref.WeakSet()

    def _load(cls, path, **kwargs):
        obj = cls()
        live.add(obj)
        return obj

    monkeypatch.setattr(_FakePipe, "from_pretrained", classmethod(_load))

    class Endpoint:
        def setup(self, pipeline: _FakePipe, vae: _FakePipe) -> None:
            self.pipeline = pipeline
            self.vae = vae

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    base_pipeline = HF("acme/sdxl-0")
    shared_vae = HF("acme/sdxl-vae")
    spec = EndpointSpec(
        name="generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"pipeline": base_pipeline, "vae": shared_vae},
        slots={
            "pipeline": Slot(_FakePipe, default_checkpoint=base_pipeline),
            "vae": Slot(_FakePipe, default_checkpoint=shared_vae),
        },
        resources=Resources(vram_gb=12),
    )
    ex = _executor([spec], tmp_path, monkeypatch=monkeypatch)
    sent: list[pb.WorkerMessage] = []

    async def _capture(message: pb.WorkerMessage) -> None:
        sent.append(message)

    ex._send = _capture
    lifecycle = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="worker", runpod_pod_id=""),
        ex,
    )
    model_bytes = 6_941_377_969
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda path: model_bytes)

    root = tmp_path / "cgroup"
    root.mkdir(exist_ok=True)
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text("0::/\n")
    limit = 31 * _GiB
    (root / "memory.max").write_text(str(limit))
    (root / "memory.current").write_text(str(limit))
    pressure = False
    object_bytes = 7 * _GiB // 5  # 1.4GiB per pipeline/VAE object

    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: SimpleNamespace(total=125 * _GiB, available=125 * _GiB),
    )

    def _ram() -> memory_mod.HostRam:
        if not pressure:
            inactive = limit
        else:
            working = len(live) * object_bytes
            inactive = max(0, limit - working)
        (root / "memory.stat").write_text(f"inactive_file {inactive}\n")
        return memory_mod.probe_host_ram(root=root, proc_self_cgroup=proc)

    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(
        residency_mod, "get_available_ram_gb",
        lambda: 31.0 if not pressure else _ram().available_gb,
    )

    async def _apply(generation: int, pipeline_ref: str) -> None:
        vae_ref = "tensorhub/sdxl-vae:prod"
        await lifecycle.on_hello_ack(pb.HelloAck(
            desired_residency=pb.DesiredResidency(
                generation=generation,
                snapshots={
                    pipeline_ref: pb.Snapshot(
                        digest="blake3:" + f"{generation:064x}",
                    ),
                    vae_ref: pb.Snapshot(digest="blake3:" + "f" * 64),
                },
                hot=[pb.DesiredInstance(
                    function_name="generate",
                    models=[
                        pb.ModelBinding(slot="pipeline", ref=pipeline_ref),
                        pb.ModelBinding(slot="vae", ref=vae_ref),
                    ],
                )],
            ),
        ))
        task = lifecycle._residency_task
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), 2)

    async def _run() -> None:
        nonlocal pressure
        for i in range(8):
            await _apply(i + 1, f"tensorhub/sdxl-{i}:prod")
        assert len(live) == 16
        old_records = [rec for rec in ex._classes.values() if rec.ready]

        pressure = True
        await _apply(9, "tensorhub/sdxl-8:prod")

        ninth = next(
            rec for rec in ex._classes.values()
            if rec.ready and "tensorhub/sdxl-8:prod" in ex._record_refs(rec)
        )
        assert ninth.instance is not None
        assert sum(not rec.ready for rec in old_records) == 2
        assert len(live) == 14
        assert not [
            msg for msg in sent
            if msg.WhichOneof("msg") == "model_event"
            and msg.model_event.state == pb.MODEL_STATE_FAILED
        ]
        lifecycle._cancel_residency_reconcile()

    asyncio.run(_run())


def test_runjob_sixteen_model_cgroup_swap_stress(
    tmp_path: Path, monkeypatch,
) -> None:
    """Sixteen production RunJob picks converge under a 31GiB cgroup cap.

    The fake model objects supply deterministic anonymous working-set bytes;
    the real cgroup parser, dynamic binding, RunJob, setup, owner-aware LRU,
    pin, result, and cleanup paths remain in use. Time is irrelevant: every
    decision follows the observed state of the disposable cgroup files.
    """
    from gen_worker.models import disk_gc

    live: weakref.WeakSet[object] = weakref.WeakSet()

    def _load(cls, path, **kwargs):
        obj = cls()
        live.add(obj)
        return obj

    monkeypatch.setattr(_FakePipe, "from_pretrained", classmethod(_load))

    class Endpoint:
        def setup(self, pipeline: _FakePipe, vae: _FakePipe) -> None:
            self.pipeline = pipeline
            self.vae = vae

        def run(self, ctx, payload: _In) -> _In:
            return payload

    base_pipeline = HF("acme/sdxl-0")
    shared_vae = HF("acme/sdxl-vae")
    spec = EndpointSpec(
        name="generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"pipeline": base_pipeline, "vae": shared_vae},
        slots={
            "pipeline": Slot(_FakePipe, default_checkpoint=base_pipeline),
            "vae": Slot(_FakePipe, default_checkpoint=shared_vae),
        },
        resources=Resources(vram_gb=12),
    )
    sent: list[pb.WorkerMessage] = []
    ex = _executor([spec], tmp_path, sent, monkeypatch=monkeypatch)
    model_bytes = 6_941_377_969
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda path: model_bytes)

    root = tmp_path / "runjob-cgroup"
    root.mkdir()
    proc = tmp_path / "runjob-proc-self-cgroup"
    proc.write_text("0::/\n")
    limit = 31 * _GiB
    (root / "memory.max").write_text(str(limit))
    (root / "memory.current").write_text(str(limit))
    object_bytes = 7 * _GiB // 5
    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: SimpleNamespace(total=125 * _GiB, available=125 * _GiB),
    )

    def _ram() -> memory_mod.HostRam:
        working = len(live) * object_bytes
        (root / "memory.stat").write_text(
            f"inactive_file {max(0, limit - working)}\n"
        )
        return memory_mod.probe_host_ram(root=root, proc_self_cgroup=proc)

    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: _ram().available_gb)

    async def _run() -> None:
        observed_before: list[int] = []
        ready_after: list[int] = []
        first_record: _ClassRecord | None = None
        first_instance: weakref.ReferenceType[object] | None = None
        for i in range(16):
            observed_before.append(int(_ram().available_gb * _GiB))
            pipeline_ref = f"tensorhub/sdxl-{i}:prod"
            vae_ref = "tensorhub/sdxl-vae:prod"
            run = pb.RunJob(
                request_id=f"swap-{i}",
                attempt=1,
                function_name="generate",
                input_payload=msgspec.msgpack.encode(_In(x=str(i))),
                models=[
                    pb.ModelBinding(slot="pipeline", ref=pipeline_ref),
                    pb.ModelBinding(slot="vae", ref=vae_ref),
                ],
                snapshots={
                    pipeline_ref: pb.Snapshot(
                        digest="blake3:" + f"{i + 1:064x}",
                    ),
                    vae_ref: pb.Snapshot(digest="blake3:" + "f" * 64),
                },
            )
            await ex.handle_run_job(run)
            job = ex.jobs[(run.request_id, run.attempt)]
            assert job.task is not None
            await job.task
            result = next(
                message.job_result for message in reversed(sent)
                if message.WhichOneof("msg") == "job_result"
                and message.job_result.request_id == run.request_id
            )
            assert result.status == pb.JOB_STATUS_OK, result.safe_message
            if i == 0:
                first_record = next(
                    rec for rec in ex._classes.values()
                    if rec.ready and pipeline_ref in ex._record_refs(rec)
                )
                first_instance = weakref.ref(first_record.instance)
            ready_after.append(len({
                id(rec) for rec in ex._classes.values() if rec.ready
            }))

        required = model_bytes + int(31.0 * 0.2 * _GiB)
        assert any(available < required for available in observed_before[8:])
        assert max(ready_after) <= 8
        assert first_record is not None and first_instance is not None
        assert not first_record.ready
        assert first_record.instance is None
        assert first_record.held_refs == []

        # Revisit an actually evicted pick. The same record identity is
        # intentionally reusable, but it must own a newly constructed
        # endpoint/pipeline and rebind Residency to that exact object.
        replay_ref = "tensorhub/sdxl-0:prod"
        vae_ref = "tensorhub/sdxl-vae:prod"
        replay = pb.RunJob(
            request_id="swap-reload-0",
            attempt=1,
            function_name="generate",
            input_payload=msgspec.msgpack.encode(_In(x="replay-0")),
            models=[
                pb.ModelBinding(slot="pipeline", ref=replay_ref),
                pb.ModelBinding(slot="vae", ref=vae_ref),
            ],
            snapshots={
                replay_ref: pb.Snapshot(digest="blake3:" + f"{1:064x}"),
                vae_ref: pb.Snapshot(digest="blake3:" + "f" * 64),
            },
        )
        await ex.handle_run_job(replay)
        replay_job = ex.jobs[(replay.request_id, replay.attempt)]
        assert replay_job.task is not None
        await replay_job.task
        replay_result = next(
            message.job_result for message in reversed(sent)
            if message.WhichOneof("msg") == "job_result"
            and message.job_result.request_id == replay.request_id
        )
        assert replay_result.status == pb.JOB_STATUS_OK, replay_result.safe_message
        reloaded = first_record.instance
        assert first_record.ready and reloaded is not None
        previous = first_instance()
        assert previous is None or reloaded is not previous
        assert set(first_record.held_refs) == {replay_ref, vae_ref}
        assert ex.store.residency.obj(replay_ref) is reloaded.pipeline
        assert len(live) <= 16
        assert len([
            message for message in sent
            if message.WhichOneof("msg") == "job_result"
        ]) == 17
        assert not [
            message for message in sent
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_FAILED
            and message.model_event.error == "insufficient_host_ram"
        ]

    asyncio.run(_run())


def test_gate_ignores_tenant_owned_slots(tmp_path: Path, monkeypatch) -> None:
    """str/Path-typed slots (tenant-owned loads, engine runtimes) must not be
    counted: a 26GiB vllm model on a 32GiB host is NOT a from_pretrained
    host-RAM staging load and must not be refused by the gate."""
    from gen_worker.models import disk_gc

    class Endpoint:
        def setup(self, model: str) -> None:
            self.model = model

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"model": HF("acme/huge-llm")})
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 26 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 32.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 28.0)

    async def _run() -> None:
        ex = _executor([spec], tmp_path, monkeypatch=monkeypatch)
        inst = await ex.ensure_setup(spec)  # gate skipped -> loads fine
        assert inst.model  # slot injected as the local path

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# Loop-stall watchdog: stall episodes are visible in worker logs
# --------------------------------------------------------------------------- #


def test_loop_stall_watchdog_logs_stall(caplog) -> None:
    from gen_worker.worker import _LoopStallWatchdog

    async def _main() -> None:
        wd = _LoopStallWatchdog(
            asyncio.get_running_loop(), interval_s=0.05, warn_after_s=0.1
        )
        wd.start()
        await asyncio.sleep(0.1)  # let the first ping cycle start
        time.sleep(0.5)  # block the loop (simulated reclaim stall)
        await asyncio.sleep(0.05)
        wd.stop()

    with caplog.at_level(logging.WARNING, logger="gen_worker.worker"):
        asyncio.run(_main())
    assert any("event loop stalled" in r.message for r in caplog.records)
