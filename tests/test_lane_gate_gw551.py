"""Lane serve gate + pinned host swap (gw#551).

te#79's serve proof: a merged two-lane endpoint whose lanes overcommit VRAM
demotes one lane to host RAM — and the next request on that lane CRASHED
(addmm device mismatch / cuda generator vs cpu latents) because nothing
between "demoted" and "the handler calls the pipeline" re-promoted it, while
the executor's whole-job pin of every declared slot made the idle sibling
un-demotable anyway.

Contract here:
- the gate wraps a pipeline's ``__call__`` preserving identity/isinstance;
- a demoted lane promotes (pinning itself, LRU-swapping the idle sibling)
  before executing — never runs cpu-resident;
- when VRAM truly cannot fit it queues briefly, then fails RETRYABLE (or
  arms the monolithic offload fallback);
- the pinned host cache makes the swap cheap: demote of an unchanged weight
  is a pointer swap, promote a single pinned non_blocking H2D.

CUDA-gated tests move real memory (the gw#414/gw#417 lesson: CPU-tier tests
prove plumbing, only a real card proves the path).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gen_worker.api.errors import RetryableError
from gen_worker.models import lane_gate as lane_gate_mod
from gen_worker.models import residency as residency_mod
from gen_worker.models.lane_gate import LaneGate, arm_lane_gate
from gen_worker.models.pinned_swap import cached_swap_bytes, swap_module
from gen_worker.models.residency import Residency, Tier

_GiB = 1024 ** 3
_MiB = 1024 ** 2

_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="moves real memory between VRAM/RAM; needs CUDA",
)


@pytest.fixture(autouse=True)
def _plenty_of_ram(monkeypatch):
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 64.0)


def _res(budget_gb: float = 24.0, **kw) -> Residency:
    return Residency(vram_budget_bytes=int(budget_gb * _GiB), **kw)


# --------------------------------------------------------------------------- #
# Gate wrapping semantics (no CUDA needed)
# --------------------------------------------------------------------------- #


class _CallablePipe:
    """Minimal pipeline shape: instance __call__, movable, attributes."""

    def __init__(self) -> None:
        self.calls = 0
        self.flavor = "tangerine"

    def to(self, device: str) -> "_CallablePipe":
        return self

    def __call__(self, x: int) -> int:
        self.calls += 1
        return x * 2


class _CallLess:
    def to(self, device: str) -> "_CallLess":
        return self


def test_arm_preserves_identity_and_passthrough() -> None:
    res = _res()
    pipe = _CallablePipe()
    original_cls = type(pipe)
    assert arm_lane_gate(pipe, LaneGate(ref="acme/a", residency=res))

    assert isinstance(pipe, original_cls)            # dynamic subclass
    assert type(pipe).__name__ == "_CallablePipe"    # same readable name
    assert pipe.flavor == "tangerine"                # attributes intact
    assert pipe(21) == 42 and pipe.calls == 1        # call passes through

    # Re-arm is idempotent: fresh gate replaces, no double-wrapping.
    assert arm_lane_gate(pipe, LaneGate(ref="acme/a2", residency=res))
    assert type(type(pipe).__mro__[1]) is type       # still one wrapper deep
    assert pipe(1) == 2


def test_arm_refuses_callless_class() -> None:
    # getattr(cls, "__call__") on a call-less class resolves to the metaclass
    # constructor — arming must refuse, not capture it.
    assert not arm_lane_gate(_CallLess(), LaneGate(ref="x", residency=_res()))


def test_gate_noop_without_cuda(monkeypatch) -> None:
    """CPU-only host: everything already runs on cpu; the gate never moves."""
    res = _res()
    pipe = _CallablePipe()
    res.track_ram("acme/a", pipe)
    gate = LaneGate(ref="acme/a", residency=res)
    monkeypatch.setattr(lane_gate_mod, "_cuda_available", lambda: False)
    arm_lane_gate(pipe, gate)
    assert pipe(3) == 6
    assert res.tier("acme/a") is Tier.RAM


# --------------------------------------------------------------------------- #
# Promote-on-use / queue / fallback (budget-mode residency, forced-cuda path)
# --------------------------------------------------------------------------- #


def test_gate_promotes_demoted_lane_before_call(monkeypatch) -> None:
    monkeypatch.setattr(lane_gate_mod, "_cuda_available", lambda: True)
    moves: list = []
    res = Residency(
        vram_budget_bytes=int(4 * _GiB),
        move_fn=lambda obj, device: moves.append((obj, device)),
    )
    a, b = _CallablePipe(), _CallablePipe()
    res.track_vram("acme/b", b, vram_bytes=3 * _GiB)
    res.track_ram("acme/a", a)
    e = res._entries["acme/a"]
    e.vram_hint = 3 * _GiB  # promote must LRU-demote b first

    swaps: list = []
    gate = LaneGate(
        ref="acme/a", residency=res, retry_exc=RetryableError,
        on_swap=lambda ref, ms: swaps.append((ref, ms)),
    )
    arm_lane_gate(a, gate)
    assert a(2) == 4
    assert res.tier("acme/a") is Tier.VRAM
    assert res.tier("acme/b") is Tier.RAM        # idle sibling swapped out
    assert swaps and swaps[0][0] == "acme/a"
    assert (b, "cpu") in moves and (a, "cuda") in moves


def _oom_when_sibling_resident(res_holder: list, a: object) -> "callable":
    """Simulated allocator: moving ``a`` onto the card OOMs while the
    sibling still occupies it (budget-mode moves never really fail)."""
    def move(obj: object, device: str) -> None:
        res = res_holder[0]
        if obj is a and device == "cuda" and res.tier("acme/b") is Tier.VRAM:
            raise RuntimeError("CUDA out of memory (simulated)")
    return move


def test_gate_queue_then_retryable_when_vram_never_fits(monkeypatch) -> None:
    monkeypatch.setattr(lane_gate_mod, "_cuda_available", lambda: True)
    a, b = _CallablePipe(), _CallablePipe()
    holder: list = []
    res = Residency(
        vram_budget_bytes=int(4 * _GiB),
        move_fn=_oom_when_sibling_resident(holder, a),
    )
    holder.append(res)
    res.track_vram("acme/b", b, vram_bytes=3 * _GiB)
    res.track_ram("acme/a", a)
    res._entries["acme/a"].vram_hint = 3 * _GiB

    gate = LaneGate(ref="acme/a", residency=res,
                    retry_exc=RetryableError, wait_s=0.6)
    arm_lane_gate(a, gate)
    with res.executing("acme/b"):   # a concurrent job pins the sibling
        t0 = time.monotonic()
        with pytest.raises(RetryableError):
            a(1)
        assert time.monotonic() - t0 >= 0.5   # queued before failing
    assert a.calls == 0                        # never executed cpu-resident
    # Sibling pin released: the same call now swaps and serves.
    assert a(2) == 4
    assert res.tier("acme/a") is Tier.VRAM


def test_gate_offload_fallback_engages(monkeypatch) -> None:
    monkeypatch.setattr(lane_gate_mod, "_cuda_available", lambda: True)
    a, b = _CallablePipe(), _CallablePipe()
    holder: list = []
    res = Residency(
        vram_budget_bytes=int(4 * _GiB),
        move_fn=_oom_when_sibling_resident(holder, a),
    )
    holder.append(res)
    res.track_vram("acme/b", b, vram_bytes=3 * _GiB)
    res.track_ram("acme/a", a)
    res._entries["acme/a"].vram_hint = 3 * _GiB

    armed: list = []

    def fallback() -> bool:
        armed.append(True)
        a._cozy_low_vram_mode = "model_offload"  # what rearm_offload does
        res.track_vram("acme/a", a)              # rebooks RAM (offload-hooked)
        return True

    gate = LaneGate(ref="acme/a", residency=res, retry_exc=RetryableError,
                    wait_s=0.3, offload_fallback=fallback)
    arm_lane_gate(a, gate)
    with res.executing("acme/b"):
        assert a(5) == 10                       # served, not raised
    assert armed
    assert res.tier("acme/a") is Tier.RAM       # honest offload booking
    # Next call: offload-hooked objects own their placement — gate no-ops.
    assert a(1) == 2


def test_gate_skips_offload_hooked_pipelines(monkeypatch) -> None:
    monkeypatch.setattr(lane_gate_mod, "_cuda_available", lambda: True)
    res = _res()
    a = _CallablePipe()
    a._cozy_low_vram_mode = "model_offload"
    res.track_vram("acme/a", a)                  # books RAM: offload-hooked
    arm_lane_gate(a, LaneGate(ref="acme/a", residency=res))
    assert a(4) == 8
    assert res.tier("acme/a") is Tier.RAM        # untouched


# --------------------------------------------------------------------------- #
# Pinned host swap cache (CUDA)
# --------------------------------------------------------------------------- #


def _tiny_module(mb: int = 64) -> "torch.nn.Module":
    import torch.nn as nn

    n = int(mb * _MiB / 4 / 1024)  # fp32 rows of 1024
    m = nn.Sequential(nn.Linear(1024, n, bias=True), nn.ReLU())
    m.register_buffer("_marker", torch.arange(8, dtype=torch.float32))
    return m


@_cuda
def test_pinned_swap_roundtrip_preserves_values_and_caches() -> None:
    m = _tiny_module().cuda()
    w = m[0].weight
    before = w.detach().cpu().clone()  # bitwise reference (device-reduction
    #                                    sums are order-dependent -> flaky)

    assert swap_module(m, "cpu")
    assert w.device.type == "cpu"
    if not w.data.is_pinned():
        # Fail-soft path engaged (pinned alloc refused under host pressure —
        # by design the swap continues pageable). Cache semantics identical.
        logging.getLogger(__name__).warning(
            "gw#551: pinned alloc unavailable; validating pageable cache path")
    assert cached_swap_bytes(m) > 0
    host_ptr = w.data.data_ptr()

    assert swap_module(m, "cuda")
    assert w.device.type == "cuda"
    assert torch.equal(w.detach().cpu(), before)
    assert cached_swap_bytes(m) > 0              # cache retained across promote

    # Unchanged weights: second demote is a pointer swap onto the SAME
    # pinned host storage — no fresh allocation. (``p.data`` returns a fresh
    # view per access: compare storage pointers, not identity.)
    assert swap_module(m, "cpu")
    assert w.data.data_ptr() == host_ptr
    assert torch.equal(w.detach(), before)


@_cuda
def test_pinned_swap_detects_out_of_band_weight_replacement() -> None:
    m = _tiny_module(8).cuda()
    assert swap_module(m, "cpu") and swap_module(m, "cuda")
    # Replace the GPU weight out-of-band: the cached host copy is stale.
    with torch.no_grad():
        m[0].weight.data = torch.ones_like(m[0].weight.data)
    assert swap_module(m, "cpu")
    assert m[0].weight.detach().float().mean().item() == pytest.approx(1.0)


@_cuda
def test_pinned_swap_promote_is_faster_than_pageable(caplog) -> None:
    """Measured, logged (informational): pinned H2D vs a pageable .to()."""
    m = _tiny_module(256).cuda()
    swap_module(m, "cpu")            # builds the pinned cache
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    swap_module(m, "cuda")
    pinned_s = time.perf_counter() - t0

    m2 = _tiny_module(256)           # pageable cpu weights
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m2.to("cuda")
    torch.cuda.synchronize()
    pageable_s = time.perf_counter() - t0
    logging.getLogger(__name__).warning(
        "gw#551 promote 256MB: pinned=%.1fms pageable=%.1fms",
        pinned_s * 1e3, pageable_s * 1e3,
    )
    assert pinned_s < max(pageable_s * 2.0, 1.0)  # sanity, not a benchmark


# --------------------------------------------------------------------------- #
# Real two-lane juggling through the executor (CUDA)
# --------------------------------------------------------------------------- #


diffusers = pytest.importorskip("diffusers")


@_cuda
def test_two_lane_juggle_serves_alternating_traffic(tmp_path, lane_repos_gw551) -> None:
    """The te#79 shape, end-to-end on tiny real pipelines: two lanes whose
    exclusive unets overcommit the VRAM budget, alternating calls. Every call
    must execute with its unet on cuda (the gate swaps per alternation);
    warmup() during setup already exercises a demoted lane (the mid-setup
    demote used to crash exactly there)."""
    from tests.test_content_lanes import _executor, _lane_spec
    from gen_worker.api.binding import HF

    repos = {"acme/a": lane_repos_gw551["a"], "acme/b": lane_repos_gw551["b"]}
    spec = _lane_spec(_JuggleLanes, {"a": HF("acme/a"), "b": HF("acme/b")})

    async def _run() -> None:
        ex = _executor([spec], tmp_path, int(2.25 * _GiB), [], repos)
        inst = await ex.ensure_setup(spec)
        res = ex.store.residency

        # Overcommitted: exactly one lane VRAM-resident after setup+warmup.
        tiers = {res.tier("acme/a"), res.tier("acme/b")}
        assert tiers == {Tier.VRAM, Tier.RAM}

        # Alternating traffic: 3 rounds, zero failures, device-coherent.
        for i in range(3):
            for pipe, ref in ((inst.a, "acme/a"), (inst.b, "acme/b")):
                out_device = pipe(steps=1)
                assert out_device == "cuda"
                assert res.tier(ref) is Tier.VRAM

        stats = res.transition_stats()
        assert stats["acme/a"]["promotes"] >= 3
        assert stats["acme/b"]["promotes"] >= 2
        assert stats["acme/a"]["last_promote_ms"] >= 0
        logging.getLogger(__name__).warning(
            "gw#551 juggle transition stats: %s", stats)

        # The executor's job-pin surface excludes the call-time-owned lanes.
        rec = ex._classes[spec.instance_key]
        assert rec.lane_refs == {"acme/a", "acme/b"}
        assert ex._job_pin_refs(spec, list(spec.models)) == []

    asyncio.run(_run())


class JugglePipe(diffusers.DiffusionPipeline):
    """Real DiffusionPipeline with a diffusers-shaped __call__: creates its
    latents on the device it BELIEVES it is on (the te#79 crash recipe when
    that belief is cpu while feeding a cuda encoder)."""

    def __init__(self, unet, vae, scheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, vae=vae, scheduler=scheduler)

    def __call__(self, steps: int = 1) -> str:
        unet_dev = next(self.unet.parameters()).device
        # Mixed-device execution must fail exactly like a real denoise.
        sample = torch.randn(1, 3, 8, 8, device=unet_dev)
        t = torch.tensor(1, device=unet_dev)
        for _ in range(steps):
            sample = self.unet(sample, t).sample
        vae_dev = next(self.vae.parameters()).device
        if vae_dev != unet_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, got "
                f"{unet_dev} and {vae_dev}")
        self.vae.encode(sample)
        return unet_dev.type


class _JuggleLanes:
    def setup(self, a: JugglePipe, b: JugglePipe) -> None:
        self.a, self.b = a, b

    def warmup(self) -> None:
        # Runs inside _setup_locked: lane a is typically demoted by lane b's
        # load at this point — this call crashed pre-gw#551.
        self.a(steps=1)
        self.b(steps=1)

    def run(self, ctx, payload):  # pragma: no cover
        return payload


@pytest.fixture(scope="module")
def lane_repos_gw551(tmp_path_factory) -> dict:
    """Two lane repos with byte-identical vae, ~148MB exclusive unets."""
    import shutil

    from tests.test_content_lanes import _save_lane_repo
    from gen_worker.models.cozy_cas import _blake3_file

    root = tmp_path_factory.mktemp("gw551-repos")
    a, b = root / "repo-a", root / "repo-b"
    _save_lane_repo(a, unet_seed=11)
    _save_lane_repo(b, unet_seed=12)
    shutil.rmtree(b / "vae")
    shutil.copytree(a / "vae", b / "vae")
    assert (
        _blake3_file(next((a / "vae").glob("*.safetensors")))
        == _blake3_file(next((b / "vae").glob("*.safetensors")))
    )
    return {"a": a, "b": b}
