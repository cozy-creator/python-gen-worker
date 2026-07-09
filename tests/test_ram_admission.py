"""Host-RAM admission (gw#407) — CPU-only, deterministic.

J17 livelock: 16 SDXL variants on a 31GB host — every load/demote grew the
warm RAM tier until the host entered reclaim-thrash, which stalled the whole
process (including gRPC keepalive acks), so the hub disconnected, requeued the
same job, and the cycle repeated. The RAM tier is now admission-controlled:
demote is size-aware, loads free the warm tier first via make_room_ram, and a
load that still cannot fit fails RETRYABLE instead of thrashing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import msgspec
import pytest

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.api.errors import RetryableError
from gen_worker.executor import Executor, ModelStore
from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import Residency, Tier
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

_GiB = 1024 ** 3


def _res(events: list | None = None, budget_gb: int = 24) -> Residency:
    ev = events if events is not None else []
    return Residency(
        on_event=lambda ref, state, vb: ev.append((ref, state, vb)),
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
# make_room_ram: warm-tier LRU release under host-RAM pressure
# --------------------------------------------------------------------------- #


def test_make_room_ram_releases_lru_until_headroom(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    res = _res()
    sizes = {"m/a": 6.0, "m/b": 6.0}

    def _avail() -> float:
        # Warm entries eat their size; releases give it back.
        used = sum(gb for ref, gb in sizes.items() if res.tier(ref) is Tier.RAM)
        return 19.0 - used

    monkeypatch.setattr(residency_mod, "get_available_ram_gb", _avail)
    res.track_ram("m/a", _Pipe(), path=tmp_path)
    res.touch("m/a")
    res.track_ram("m/b", _Pipe(), path=tmp_path)

    # Incoming 6GiB load: available 7, target 6 + 6.2 floor -> release LRU (a).
    assert res.make_room_ram(6 * _GiB) is True
    assert res.tier("m/a") is Tier.DISK
    assert res.tier("m/b") is Tier.RAM  # only as much as needed was released


def test_make_room_ram_skips_pinned_and_executing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 2.0)
    res = _res()
    res.track_ram("m/pinned", _Pipe(), path=tmp_path)
    res.pin("m/pinned")
    res.track_ram("m/busy", _Pipe(), path=tmp_path)
    with res.executing("m/busy"):
        assert res.make_room_ram(6 * _GiB) is False
        assert res.tier("m/pinned") is Tier.RAM
        assert res.tier("m/busy") is Tier.RAM


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


def _spec(name: str, cls: type, models: dict) -> EndpointSpec:
    return EndpointSpec(
        name=name, method=cls.run, kind="inference", payload_type=_In,
        output_mode="single", cls=cls, attr_name="run", models=models,
        resources=Resources(),
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
        with pytest.raises(RetryableError, match="insufficient host RAM"):
            await ex.ensure_setup(spec)
        # Transient pressure: the function is NOT disabled and not failed.
        assert "ep" not in ex.unavailable
        assert ex._classes[spec.instance_key].failed is None

        # Pressure gone -> the retry loads normally.
        monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 24.0)
        inst = await ex.ensure_setup(spec)
        assert isinstance(inst.m, _FakePipe)

    asyncio.run(_run())


def test_setup_frees_warm_tier_before_refusing(tmp_path: Path, monkeypatch) -> None:
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
    spec_b = _spec("b", B, {"m": HF("acme/b")})
    monkeypatch.setattr(disk_gc, "tree_bytes", lambda p: 6 * _GiB)
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)

    async def _run() -> None:
        ex = _executor([spec_a, spec_b], tmp_path)
        res = ex.store.residency

        monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 24.0)
        await ex.ensure_setup(spec_a)
        # CPU-only harness: the loaded pipeline books straight into the warm
        # RAM tier — exactly the tier make_room_ram drains.
        assert res.tier("acme/a") is Tier.RAM

        # B's load: 7GiB available < 6 + 6.2 floor, but releasing warm A
        # (6GiB) makes room -> the load proceeds, A lands on disk.
        def _avail() -> float:
            return 7.0 + (6.0 if res.tier("acme/a") is not Tier.RAM else 0.0)

        monkeypatch.setattr(residency_mod, "get_available_ram_gb", _avail)
        inst_b = await ex.ensure_setup(spec_b)
        assert isinstance(inst_b.m, _FakePipe)
        assert res.tier("acme/a") is Tier.DISK

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
