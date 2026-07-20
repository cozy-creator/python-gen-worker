"""gw#494: HelloAck re-resolution is transactional — residency re-keys and
gates re-run.

The th#736 mechanic worker-side: a second HelloAck with a different pick
rebinds ``spec.models`` while loaded pipelines stay booked under the OLD
resolved ref (VRAM orphaned forever, pins/promotes/adapters miss, UNLOAD by
the new ref frees nothing). These tests pin the closure: booking and clearing
derive from ONE keying (the record's load-time ``held_refs``), a pick change
marks the record stale and vacates it, and ``gate_functions`` re-runs
idempotently against the rebound bindings.
"""

from __future__ import annotations

import asyncio
from typing import List

import msgspec
import pytest

from gen_worker.api.binding import HF, Hub, rebind_pick, wire_ref
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline) -> None:  # pragma: no cover - never run here
        self.pipeline = pipeline

    def generate(self, ctx, payload: _In) -> dict:  # pragma: no cover
        return {}


def _spec(**kw) -> EndpointSpec:
    return EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub("acme/z-image")},
        resources=kw.pop("resources", Resources(gpu=True)),
        **kw,
    )


def _executor(spec: EndpointSpec | None = None) -> Executor:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    return Executor([spec or _spec()], _send)


def _simulate_loaded(ex: Executor, spec: EndpointSpec) -> str:
    """Pretend ensure_setup completed for the spec's CURRENT bindings:
    book residency and stamp held_refs the way _setup_locked does."""
    rec = ex._classes[spec.instance_key]
    refs = sorted({wire_ref(b) for b in spec.models.values()})
    for ref in refs:
        ex.store.residency.track_vram(ref, object(), vram_bytes=1024)
    rec.held_refs = list(refs)
    rec.stale = False
    rec.instance = _Fake()
    rec.ready = True
    return refs[0]


def _vram_refs(ex: Executor) -> set:
    from gen_worker.models.residency import Tier

    res = ex.store.residency
    return {
        m.ref for m in ex.store.residency_snapshot()
        if res.tier(m.ref) is Tier.VRAM
    }


def test_repick_rekeys_residency_zero_orphans() -> None:
    """resolve -> book -> re-resolve -> clear leaves ZERO orphans: nothing
    stays VRAM-booked under a key no longer reachable from any binding."""

    async def _run() -> None:
        ex = _executor()
        spec = ex.specs["generate"]

        # HelloAck 1: pick #fp8; instance loads and books under the pick.
        ex.apply_model_resolutions({"acme/z-image": ("acme/z-image#fp8", "", "")})
        assert wire_ref(spec.models["pipeline"]) == "acme/z-image#fp8"
        old_ref = _simulate_loaded(ex, spec)
        assert old_ref == "acme/z-image#fp8"
        assert _vram_refs(ex) == {"acme/z-image#fp8"}
        rec = ex._classes[spec.instance_key]

        # HelloAck 2: different pick. The record is stale and gets vacated —
        # the OLD key's VRAM booking is released (no orphan), ready drops.
        ex.apply_model_resolutions(
            {"acme/z-image": ("acme/z-image#svdq-int4-r128", "", "")})
        assert rec.stale is True
        await asyncio.sleep(0.05)  # let the scheduled revalidate task run
        assert rec.ready is False
        assert rec.held_refs == []
        assert _vram_refs(ex) == set()

        # Instance reloads under the new pick; a revert-to-declared HelloAck
        # (empty map) vacates again — still zero orphans anywhere.
        new_ref = _simulate_loaded(ex, spec)
        assert new_ref == "acme/z-image#svdq-int4-r128"
        rec2 = ex._classes[spec.instance_key]
        ex.apply_model_resolutions({})
        assert wire_ref(spec.models["pipeline"]) == "acme/z-image"
        assert rec2.stale is True
        await asyncio.sleep(0.05)
        assert rec2.ready is False
        assert _vram_refs(ex) == set()

    asyncio.run(_run())


def test_vacate_releases_booked_keys_not_rebound_ones() -> None:
    """_vacate_record must release the LOAD-TIME keys even after the spec
    was rebound (the exact orphan mechanic)."""

    async def _run() -> None:
        ex = _executor()
        spec = ex.specs["generate"]
        old_ref = _simulate_loaded(ex, spec)  # booked under declared ref
        rec = ex._classes[spec.instance_key]

        # Rebind WITHOUT vacating first (simulates the pre-fix window).
        ex.apply_model_resolutions({"acme/z-image": ("acme/z-image#fp8", "", "")})
        # The rehome carried the live record to the new key; vacate it.
        rec = next(r for r in ex._classes.values() if r is rec)
        await ex._vacate_record(rec)
        assert _vram_refs(ex) == set(), f"orphan under {old_ref!r}"
        await asyncio.sleep(0.05)  # settle the scheduled revalidate task

    asyncio.run(_run())


def test_hf_binding_resolution_is_rejected_keeps_declared() -> None:
    ex = _executor(EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": HF("bfl/FLUX.2-klein-4B")},
        resources=Resources(gpu=True),
    ))
    spec = ex.specs["generate"]
    ex.apply_model_resolutions(
        {"bfl/FLUX.2-klein-4B": ("bfl/FLUX.2-klein-4B#fp8", "", "")})
    # HF picks fold flavor -- but HF has no flavor field: rejected, declared kept.
    assert spec.models["pipeline"] == HF("bfl/FLUX.2-klein-4B")


def test_regate_runs_after_resolutions_and_is_idempotent() -> None:
    """apply_model_resolutions re-runs gate_functions against the rebound
    bindings; gate marks are gate-owned (cleared on re-gate), setup failures
    survive."""
    spec = _spec(resources=Resources(gpu=True, compute_capability=8.9))
    ex = _executor(spec)

    small = {"gpu_total_mem": 48 * 1024**3, "gpu_free_mem": 48 * 1024**3,
             "gpu_sm": "75", "installed_libs": []}
    big = dict(small, gpu_sm="90")

    ex.gate_functions(small)
    assert ex.unavailable["generate"][0] == "compute_capability_unmet"

    # Re-gate with capable silicon: the gate-owned mark clears.
    ex.gate_functions(big)
    assert "generate" not in ex.unavailable

    # A setup failure is NOT gate-owned and survives a re-gate.
    ex.unavailable["generate"] = ("setup_failed", "boom", {})
    ex.gate_functions(big)
    assert ex.unavailable["generate"][0] == "setup_failed"

    # Resolutions re-run the gates using the remembered probe.
    ex.unavailable.pop("generate")
    ex.apply_model_resolutions({"acme/z-image": ("acme/z-image#fp8", "", "")})
    assert ex._last_gpu_info is not None
    assert "generate" in ex.serve_plans


def test_rebind_pick_is_the_single_fold() -> None:
    b = Hub("acme/z-image")

    # hub path: resolved_ref authoritative (flavor fold + cast stamp)
    out = rebind_pick(b, resolved_ref="acme/z-image#fp8", cast="")
    assert wire_ref(out) == "acme/z-image#fp8"
    out = rebind_pick(b, resolved_ref="", cast="fp8")
    assert out.storage_dtype == "fp8" and wire_ref(out) == "acme/z-image"
    # non-normal hub spelling still round-trips (':latest' is elided form)
    out = rebind_pick(b, resolved_ref="acme/z-image:latest#fp8")
    assert wire_ref(out) == "acme/z-image#fp8"

    # ladder path: flavor/cast overlay with the same guard
    out = rebind_pick(b, flavor="svdq-int4-r128", cast="")
    assert wire_ref(out) == "acme/z-image#svdq-int4-r128"
    with pytest.raises(ValueError):
        rebind_pick(b, resolved_ref="acme/OTHER#fp8")  # ref mismatch
    with pytest.raises(ValueError):
        rebind_pick(HF("o/r"), resolved_ref="o/r#fp8")  # HF has no flavor
