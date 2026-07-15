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
import logging
import time
import weakref
from pathlib import Path

import msgspec
import pytest

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.capability import InsufficientHostRamError
from gen_worker.executor import Executor, ModelStore
from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import Residency, Tier
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

_GiB = 1024 ** 3


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


def test_demote_floor_is_size_aware(monkeypatch) -> None:
    """Demoting a 6GiB pipeline into 12GiB available must be refused on a
    64GiB host (floor 8GiB): 12 - 6 < 8 would land in thrash territory."""
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 64.0)
    res = _res()
    res.track_vram("m/a", _Pipe(), vram_bytes=6 * _GiB)

    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 12.0)
    assert res.demote("m/a") is False
    assert res.tier("m/a") is Tier.VRAM

    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 20.0)
    assert res.demote("m/a") is True
    assert res.tier("m/a") is Tier.RAM


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
    assert res2.demote("m/b") is False


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


def _spec(
    name: str, cls: type, models: dict, *, vram_gb: float | None = None,
) -> EndpointSpec:
    return EndpointSpec(
        name=name, method=cls.run, kind="inference", payload_type=_In,
        output_mode="single", cls=cls, attr_name="run", models=models,
        resources=Resources(vram_gb=vram_gb),
    )


def _executor(specs, tmp_path: Path) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        pass

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=24 * _GiB)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        store.residency.track_disk(ref, tmp_path)
        return tmp_path

    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return Executor(specs, _send, store=store)


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
        ex = _executor([spec], tmp_path)
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
        ex = _executor([spec_a, spec_b], tmp_path)
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
        inst_b = await ex.ensure_setup(spec_b)
        assert isinstance(inst_b.m, _FakePipe)
        assert res.tier("acme/a") is Tier.DISK
        assert rec_a.ready is False
        assert rec_a.instance is None
        await asyncio.sleep(0)
        assert pipeline() is None
        assert on_disk == [True]

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
        ex = _executor([spec], tmp_path)
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
