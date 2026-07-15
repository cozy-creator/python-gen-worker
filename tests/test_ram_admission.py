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
import gc
import logging
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import msgspec
import psutil
import pytest

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.api.slot import Slot
from gen_worker.capability import InsufficientHostRamError
from gen_worker.executor import Executor, ModelStore, _Job
from gen_worker.lifecycle import Lifecycle
from gen_worker.models import memory as memory_mod
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

    def enable_model_cpu_offload(self) -> None:
        pass


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
        ex = _executor([spec_a, spec_b, spec_c], tmp_path)
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
    ex = _executor([spec], tmp_path)
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
        await lifecycle.on_hello_ack(pb.HelloAck(
            desired_residency=pb.DesiredResidency(
                generation=generation,
                hot=[pb.DesiredInstance(
                    function_name="generate",
                    models=[
                        pb.ModelBinding(slot="pipeline", ref=pipeline_ref),
                        pb.ModelBinding(slot="vae", ref="tensorhub/sdxl-vae:prod"),
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
