"""Executor <-> Residency unification (#369/#371) — CPU-only, deterministic.

A fake CUDA allocator (module counter) + fake pipelines with ``from_pretrained``
drive the REAL ensure_setup / handle_model_op paths against a budgeted
Residency: per-ref measured vram_bytes, serialized loads, UNLOAD demoting to
the warm RAM tier, and make_room-before-load demote/promote swaps.
"""

import asyncio
import time
from pathlib import Path

import msgspec
import pytest

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor, ModelStore
from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import Tier
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

_GiB = 1024 ** 3


class _In(msgspec.Struct):
    x: str


class _Alloc:
    bytes = 0


class _FakePipe:
    """Stands in for a diffusers pipeline: from_pretrained allocates, .to()
    moves the same object the endpoint instance holds."""

    size = 1 * _GiB
    load_sleep_s = 0.0

    def __init__(self) -> None:
        self.device = "cuda"

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        _Alloc.bytes += cls.size
        if cls.load_sleep_s:
            time.sleep(cls.load_sleep_s)
        return cls()

    def to(self, device: str) -> "_FakePipe":
        if device == "cpu" and self.device == "cuda":
            _Alloc.bytes -= self.size
        elif device == "cuda" and self.device == "cpu":
            _Alloc.bytes += self.size
        self.device = device
        return self


class _Unet(_FakePipe):
    size = 3 * _GiB


class _Vae(_FakePipe):
    size = 1 * _GiB


@pytest.fixture(autouse=True)
def _fake_gpu(monkeypatch):
    _Alloc.bytes = 0
    monkeypatch.setattr(Executor, "_vram_allocated", staticmethod(lambda: _Alloc.bytes))
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 64.0)


def _spec(name: str, cls: type, models: dict, vram_gb: float | None = None) -> EndpointSpec:
    return EndpointSpec(
        name=name, method=cls.run, kind="inference", payload_type=_In,
        output_mode="single", cls=cls, attr_name="run", models=models,
        resources=Resources(vram_gb=vram_gb) if vram_gb else Resources(),
    )


def _executor(specs, tmp_path: Path, budget_gb: int, sent: list) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=budget_gb * _GiB)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        store.residency.track_disk(ref, tmp_path)
        return tmp_path

    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return Executor(specs, _send, store=store)


def _events(sent, state) -> list:
    return [m.model_event.ref for m in sent
            if m.WhichOneof("msg") == "model_event" and m.model_event.state == state]


# --------------------------------------------------------------------------- #
# #369: per-ref measured vram_bytes; residency owns the pipeline objects
# --------------------------------------------------------------------------- #


def test_two_model_setup_books_per_ref_measured_bytes(tmp_path: Path) -> None:
    class Endpoint:
        def setup(self, unet: _Unet, vae: _Vae) -> None:
            self.unet, self.vae = unet, vae

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"unet": HF("acme/unet"), "vae": HF("acme/vae")})
    sent: list = []

    async def _run() -> None:
        ex = _executor([spec], tmp_path, budget_gb=10, sent=sent)
        inst = await ex.ensure_setup(spec)
        res = ex.store.residency
        assert res.vram_bytes("acme/unet") == 3 * _GiB
        assert res.vram_bytes("acme/vae") == 1 * _GiB
        # Residency owns the very objects the instance uses.
        assert res.obj("acme/unet") is inst.unet
        assert res.obj("acme/vae") is inst.vae

    asyncio.run(_run())
    vram_events = [(m.model_event.ref, m.model_event.vram_bytes) for m in sent
                   if m.WhichOneof("msg") == "model_event"
                   and m.model_event.state == pb.MODEL_STATE_IN_VRAM]
    assert ("acme/unet", 3 * _GiB) in vram_events
    assert ("acme/vae", 1 * _GiB) in vram_events


def test_tenant_loaded_ref_gets_residual_delta(tmp_path: Path) -> None:
    class Endpoint:
        def setup(self, model: str) -> None:
            _Alloc.bytes += 2 * _GiB  # tenant loads weights itself

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"model": HF("acme/opaque")})

    async def _run() -> None:
        ex = _executor([spec], tmp_path, budget_gb=10, sent=[])
        await ex.ensure_setup(spec)
        assert ex.store.residency.vram_bytes("acme/opaque") == 2 * _GiB
        assert ex.store.residency.movable("acme/opaque") is False

    asyncio.run(_run())


def test_concurrent_cold_setups_do_not_cross_contaminate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_FakePipe, "load_sleep_s", 0.05)

    class A:
        def setup(self, m: _Unet) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    class B:
        def setup(self, m: _Vae) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec_a = _spec("a", A, {"m": HF("acme/a")})
    spec_b = _spec("b", B, {"m": HF("acme/b")})

    async def _run() -> None:
        ex = _executor([spec_a, spec_b], tmp_path, budget_gb=20, sent=[])
        await asyncio.gather(ex.ensure_setup(spec_a), ex.ensure_setup(spec_b))
        # Interleaved loads would double-count each other's allocations.
        assert ex.store.residency.vram_bytes("acme/a") == 3 * _GiB
        assert ex.store.residency.vram_bytes("acme/b") == 1 * _GiB

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# #371: UNLOAD demotes to warm RAM; next setup promotes; make_room swaps LRU
# --------------------------------------------------------------------------- #


def test_unload_demotes_to_ram_and_next_setup_promotes(tmp_path: Path) -> None:
    class Endpoint:
        def setup(self, m: _Unet) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"m": HF("acme/unet")})
    sent: list = []

    async def _run() -> None:
        ex = _executor([spec], tmp_path, budget_gb=10, sent=sent)
        inst = await ex.ensure_setup(spec)
        await ex.handle_model_op(pb.ModelOp(op=pb.MODEL_OP_KIND_UNLOAD, ref="acme/unet"))
        res = ex.store.residency
        assert res.tier("acme/unet") is Tier.RAM
        assert inst.m.device == "cpu"          # actually moved, not just booked
        rec = ex._classes[spec.instance_key]
        assert rec.ready and rec.instance is inst  # no teardown

        again = await ex.ensure_setup(spec)     # RunJob path: promote, no reload
        assert again is inst
        assert res.tier("acme/unet") is Tier.VRAM
        assert inst.m.device == "cuda"

    asyncio.run(_run())
    assert _events(sent, pb.MODEL_STATE_IN_RAM) == ["acme/unet"]
    assert _events(sent, pb.MODEL_STATE_IN_VRAM) == ["acme/unet", "acme/unet"]


def test_unload_of_tenant_loaded_ref_tears_down_record(tmp_path: Path) -> None:
    class Endpoint:
        def setup(self, model: str) -> None:
            _Alloc.bytes += 2 * _GiB

        def shutdown(self) -> None:
            _Alloc.bytes -= 2 * _GiB

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"model": HF("acme/opaque")})
    sent: list = []

    async def _run() -> None:
        ex = _executor([spec], tmp_path, budget_gb=10, sent=sent)
        await ex.ensure_setup(spec)
        await ex.handle_model_op(pb.ModelOp(op=pb.MODEL_OP_KIND_UNLOAD, ref="acme/opaque"))
        assert ex.store.residency.tier("acme/opaque") is Tier.DISK
        assert ex._classes[spec.instance_key].ready is False
        assert _Alloc.bytes == 0  # shutdown() actually freed the memory

    asyncio.run(_run())
    assert _events(sent, pb.MODEL_STATE_ON_DISK)[-1] == "acme/opaque"


def test_alternating_endpoints_swap_via_demote_promote(tmp_path: Path) -> None:
    """Budget fits one 3GiB pipeline (+2GiB make_room margin): alternating
    setups must demote/promote, never tear down or degrade to offload."""

    class A:
        def setup(self, m: _Unet) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    class B:
        def setup(self, m: _Unet) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec_a = _spec("a", A, {"m": HF("acme/a")}, vram_gb=3.0)
    spec_b = _spec("b", B, {"m": HF("acme/b")}, vram_gb=3.0)
    sent: list = []

    async def _run() -> None:
        ex = _executor([spec_a, spec_b], tmp_path, budget_gb=6, sent=sent)
        res = ex.store.residency

        inst_a = await ex.ensure_setup(spec_a)
        assert res.tier("acme/a") is Tier.VRAM

        inst_b = await ex.ensure_setup(spec_b)  # make_room demotes idle A first
        assert res.tier("acme/b") is Tier.VRAM
        assert res.tier("acme/a") is Tier.RAM
        assert inst_a.m.device == "cpu"

        again_a = await ex.ensure_setup(spec_a)  # promote A back; B demoted
        assert again_a is inst_a                 # instance survived throughout
        assert res.tier("acme/a") is Tier.VRAM
        assert res.tier("acme/b") is Tier.RAM
        assert inst_b.m.device == "cpu"
        for rec in ex._classes.values():
            assert rec.ready

    asyncio.run(_run())
    assert _events(sent, pb.MODEL_STATE_IN_RAM) == ["acme/a", "acme/b"]
    assert _events(sent, pb.MODEL_STATE_IN_VRAM) == ["acme/a", "acme/b", "acme/a"]
