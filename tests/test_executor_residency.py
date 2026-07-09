"""Executor <-> Residency unification (#369/#371) against REAL tiny pipelines.

No fake pipes (gw#417): the subject is whether memory actually moves between
VRAM/RAM/disk, so these tests build tiny-config real diffusers pipelines
locally (no network) and assert on real allocator deltas and real device
placement. CUDA required; skips cleanly on CPU-only hosts.
"""

import asyncio
from pathlib import Path

import msgspec
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor, ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.models.residency import Tier

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="residency moves real memory between VRAM/RAM (gw#417); needs CUDA",
)

_GiB = 1024 ** 3
_MiB = 1024 ** 2


def _save_tiny_ddpm(path: Path, channels: int) -> None:
    """Real DDPMPipeline with a tiny-config UNet, built locally (no network).
    channels=384 -> ~148MB fp32; 256 -> ~66MB; 128 -> ~16MB."""
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3, layers_per_block=1,
        block_out_channels=(channels, channels), norm_num_groups=8,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10)
                 ).save_pretrained(str(path))


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("tiny-pipes")
    out = {}
    for name, channels in (("big-a", 384), ("big-b", 384),
                           ("mid", 256), ("small", 128)):
        _save_tiny_ddpm(root / name, channels)
        out[name] = root / name
    return out


@pytest.fixture(autouse=True)
def _settle_allocator():
    """Free the previous test's pipelines before measuring this one's."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    yield


class _In(msgspec.Struct):
    x: str


def _spec(name: str, cls: type, models: dict, vram_gb: float | None = None) -> EndpointSpec:
    return EndpointSpec(
        name=name, method=cls.run, kind="inference", payload_type=_In,
        output_mode="single", cls=cls, attr_name="run", models=models,
        resources=Resources(vram_gb=vram_gb) if vram_gb else Resources(),
    )


def _executor(specs, tmp_path: Path, budget_bytes: int, sent: list,
              refs_to_snapshots: dict) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=budget_bytes)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        path = refs_to_snapshots[ref]
        store.residency.track_disk(ref, path)
        return path

    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return Executor(specs, _send, store=store)


def _events(sent, state) -> list:
    return [m.model_event.ref for m in sent
            if m.WhichOneof("msg") == "model_event" and m.model_event.state == state]


def _device(pipe) -> str:
    return next(pipe.unet.parameters()).device.type


# --------------------------------------------------------------------------- #
# #369: per-ref measured vram_bytes; residency owns the pipeline objects
# --------------------------------------------------------------------------- #


def test_two_model_setup_books_per_ref_measured_bytes(tmp_path, snapshots) -> None:
    from diffusers import DDPMPipeline

    class Endpoint:
        def setup(self, mid: DDPMPipeline, small: DDPMPipeline) -> None:
            self.mid, self.small = mid, small

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"mid": HF("acme/mid"), "small": HF("acme/small")})
    sent: list = []
    refs = {"acme/mid": snapshots["mid"], "acme/small": snapshots["small"]}

    async def _run() -> None:
        ex = _executor([spec], tmp_path, 4 * _GiB, sent, refs)
        inst = await ex.ensure_setup(spec)
        res = ex.store.residency
        # Real allocator deltas: ~66MB and ~16MB fp32 parameter sets.
        assert res.vram_bytes("acme/mid") == pytest.approx(66 * _MiB, rel=0.25)
        assert res.vram_bytes("acme/small") == pytest.approx(16 * _MiB, rel=0.25)
        # Residency owns the very objects the instance uses, on cuda for real.
        assert res.obj("acme/mid") is inst.mid
        assert res.obj("acme/small") is inst.small
        assert _device(inst.mid) == "cuda" and _device(inst.small) == "cuda"

    asyncio.run(_run())
    vram_events = {m.model_event.ref for m in sent
                   if m.WhichOneof("msg") == "model_event"
                   and m.model_event.state == pb.MODEL_STATE_IN_VRAM
                   and m.model_event.vram_bytes > 0}
    assert vram_events == {"acme/mid", "acme/small"}


def test_tenant_loaded_ref_gets_residual_delta(tmp_path, snapshots) -> None:
    class Endpoint:
        def setup(self, model: str) -> None:
            # Tenant loads weights itself: a real 64MiB CUDA allocation.
            self.buf = torch.zeros(64 * _MiB, dtype=torch.uint8, device="cuda")

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"model": HF("acme/opaque")})
    refs = {"acme/opaque": snapshots["small"]}

    async def _run() -> None:
        ex = _executor([spec], tmp_path, 4 * _GiB, [], refs)
        await ex.ensure_setup(spec)
        res = ex.store.residency
        assert res.vram_bytes("acme/opaque") == pytest.approx(64 * _MiB, rel=0.1)
        assert res.movable("acme/opaque") is False

    asyncio.run(_run())


def test_concurrent_cold_setups_do_not_cross_contaminate(tmp_path, snapshots) -> None:
    from diffusers import DDPMPipeline

    class A:
        def setup(self, m: DDPMPipeline) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    class B:
        def setup(self, m: DDPMPipeline) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec_a = _spec("a", A, {"m": HF("acme/mid")})
    spec_b = _spec("b", B, {"m": HF("acme/small")})
    refs = {"acme/mid": snapshots["mid"], "acme/small": snapshots["small"]}

    async def _run() -> None:
        ex = _executor([spec_a, spec_b], tmp_path, 4 * _GiB, [], refs)
        await asyncio.gather(ex.ensure_setup(spec_a), ex.ensure_setup(spec_b))
        # Interleaved loads would double-count each other's allocations.
        res = ex.store.residency
        assert res.vram_bytes("acme/mid") == pytest.approx(66 * _MiB, rel=0.25)
        assert res.vram_bytes("acme/small") == pytest.approx(16 * _MiB, rel=0.25)

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# #371: UNLOAD demotes to warm RAM; next setup promotes; make_room swaps LRU
# --------------------------------------------------------------------------- #


def test_unload_demotes_to_ram_and_next_setup_promotes(tmp_path, snapshots) -> None:
    from diffusers import DDPMPipeline

    class Endpoint:
        def setup(self, m: DDPMPipeline) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"m": HF("acme/unet")})
    sent: list = []
    refs = {"acme/unet": snapshots["big-a"]}

    async def _run() -> None:
        ex = _executor([spec], tmp_path, 4 * _GiB, sent, refs)
        inst = await ex.ensure_setup(spec)
        base = torch.cuda.memory_allocated()
        await ex.handle_model_op(pb.ModelOp(op=pb.MODEL_OP_KIND_UNLOAD, ref="acme/unet"))
        res = ex.store.residency
        assert res.tier("acme/unet") is Tier.RAM
        assert _device(inst.m) == "cpu"        # actually moved, not just booked
        assert torch.cuda.memory_allocated() <= base - 100 * _MiB  # VRAM freed
        rec = ex._classes[spec.instance_key]
        assert rec.ready and rec.instance is inst  # no teardown

        again = await ex.ensure_setup(spec)     # RunJob path: promote, no reload
        assert again is inst
        assert res.tier("acme/unet") is Tier.VRAM
        assert _device(inst.m) == "cuda"

    asyncio.run(_run())
    assert _events(sent, pb.MODEL_STATE_IN_RAM) == ["acme/unet"]
    assert _events(sent, pb.MODEL_STATE_IN_VRAM) == ["acme/unet", "acme/unet"]


def test_unload_of_tenant_loaded_ref_tears_down_record(tmp_path, snapshots) -> None:
    class Endpoint:
        def setup(self, model: str) -> None:
            self.buf = torch.zeros(64 * _MiB, dtype=torch.uint8, device="cuda")

        def shutdown(self) -> None:
            self.buf = None

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    spec = _spec("ep", Endpoint, {"model": HF("acme/opaque")})
    sent: list = []
    refs = {"acme/opaque": snapshots["small"]}

    async def _run() -> None:
        base = torch.cuda.memory_allocated()
        ex = _executor([spec], tmp_path, 4 * _GiB, sent, refs)
        await ex.ensure_setup(spec)
        assert torch.cuda.memory_allocated() >= base + 64 * _MiB
        await ex.handle_model_op(pb.ModelOp(op=pb.MODEL_OP_KIND_UNLOAD, ref="acme/opaque"))
        assert ex.store.residency.tier("acme/opaque") is Tier.DISK
        assert ex._classes[spec.instance_key].ready is False
        assert torch.cuda.memory_allocated() == base  # shutdown really freed it

    asyncio.run(_run())
    assert _events(sent, pb.MODEL_STATE_ON_DISK)[-1] == "acme/opaque"


def test_alternating_endpoints_swap_via_demote_promote(tmp_path, snapshots) -> None:
    """Budget fits one ~148MB pipeline (+2GiB make_room margin): alternating
    setups must demote/promote, never tear down or degrade to offload."""
    from diffusers import DDPMPipeline

    class A:
        def setup(self, m: DDPMPipeline) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    class B:
        def setup(self, m: DDPMPipeline) -> None:
            self.m = m

        def run(self, ctx, payload: _In):  # pragma: no cover
            return payload

    # Real pipelines are ~148MB (0.145GiB). make_room targets needed + 2GiB
    # margin; a 2.2GiB budget holds exactly one pipeline: loading the second
    # (declared 0.15GiB) finds 2.2 - 0.145 < 2.15 and must demote the LRU.
    spec_a = _spec("a", A, {"m": HF("acme/a")}, vram_gb=0.15)
    spec_b = _spec("b", B, {"m": HF("acme/b")}, vram_gb=0.15)
    sent: list = []
    refs = {"acme/a": snapshots["big-a"], "acme/b": snapshots["big-b"]}

    async def _run() -> None:
        ex = _executor([spec_a, spec_b], tmp_path, int(2.2 * _GiB), sent, refs)
        res = ex.store.residency

        inst_a = await ex.ensure_setup(spec_a)
        assert res.tier("acme/a") is Tier.VRAM
        assert _device(inst_a.m) == "cuda"

        inst_b = await ex.ensure_setup(spec_b)  # make_room demotes idle A first
        assert res.tier("acme/b") is Tier.VRAM
        assert res.tier("acme/a") is Tier.RAM
        assert _device(inst_a.m) == "cpu"

        again_a = await ex.ensure_setup(spec_a)  # promote A back; B demoted
        assert again_a is inst_a                 # instance survived throughout
        assert res.tier("acme/a") is Tier.VRAM
        assert res.tier("acme/b") is Tier.RAM
        assert _device(inst_a.m) == "cuda"
        assert _device(inst_b.m) == "cpu"
        for rec in ex._classes.values():
            assert rec.ready

    asyncio.run(_run())
    assert _events(sent, pb.MODEL_STATE_IN_RAM) == ["acme/a", "acme/b"]
    assert _events(sent, pb.MODEL_STATE_IN_VRAM) == ["acme/a", "acme/b", "acme/a"]
