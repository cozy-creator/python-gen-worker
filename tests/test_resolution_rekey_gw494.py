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

from gen_worker.api.binding import Hub, rebind_pick, wire_ref
from gen_worker.models.refs import FlavorSelectorRemoved
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
    stays VRAM-booked under an instance key no longer reachable from any
    binding.

    pgw#1148 changed WHICH axis moves the key, not the mechanic. A pick can
    no longer change a binding's WIRE REF — §1.32(d) deleted the `#flavor`
    that used to, and th#1803's digest pin is a resolution of the SAME
    address, not a second one — so the axis under test is the CAST, which
    still re-keys the instance."""

    async def _run() -> None:
        ex = _executor()
        spec = ex.specs["generate"]
        declared_key = spec.instance_key

        # HelloAck 1: cast pick; instance loads and books under the ref.
        ex.apply_model_resolutions({"acme/z-image": ("", "fp8", "")})
        assert spec.models["pipeline"].storage_dtype == "fp8"
        old_key = spec.instance_key
        old_ref = _simulate_loaded(ex, spec)
        assert old_ref == "acme/z-image"
        assert _vram_refs(ex) == {"acme/z-image"}
        rec = ex._classes[spec.instance_key]

        # HelloAck 2: a different cast. The instance key MOVES and the live
        # record is carried to it — nothing is left booked under a key no
        # binding reaches, which is the invariant this test is named for.
        ex.apply_model_resolutions({"acme/z-image": ("", "fp8+te", "")})
        assert spec.instance_key != old_key
        assert old_key not in ex._classes
        assert ex._classes[spec.instance_key] is rec

        # A revert-to-declared HelloAck (empty map) moves it back, again with
        # no record stranded on the vacated key.
        mid_key = spec.instance_key
        ex.apply_model_resolutions({})
        assert spec.models["pipeline"].storage_dtype == ""
        assert spec.instance_key == declared_key
        assert mid_key not in ex._classes
        assert _vram_refs(ex) == {"acme/z-image"}

        await ex._vacate_record(ex._classes[spec.instance_key])
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
        ex.apply_model_resolutions({"acme/z-image": ("", "fp8", "")})
        # The rehome carried the live record to the new key; vacate it.
        rec = next(r for r in ex._classes.values() if r is rec)
        await ex._vacate_record(rec)
        assert _vram_refs(ex) == set(), f"orphan under {old_ref!r}"
        await asyncio.sleep(0.05)  # settle the scheduled revalidate task

    asyncio.run(_run())


def test_hf_binding_resolution_is_rejected_keeps_declared() -> None:
    from gen_worker.api.binding import HF

    ex = _executor(EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": HF("bfl/FLUX.2-klein-4B")},
        resources=Resources(gpu=True),
    ))
    spec = ex.specs["generate"]
    # A tensorhub-shaped pick against an HF binding cannot round-trip
    # (pgw#1148 makes the flavored spelling a typed refusal, and both are
    # swallowed by apply_model_resolutions' keep-the-declared arm).
    ex.apply_model_resolutions(
        {"bfl/FLUX.2-klein-4B": ("bfl/FLUX.2-klein-4B#fp8", "", "")})
    assert spec.models["pipeline"] == HF("bfl/FLUX.2-klein-4B")
    ex.apply_model_resolutions(
        {"bfl/FLUX.2-klein-4B": ("acme/other", "", "")})
    assert spec.models["pipeline"] == HF("bfl/FLUX.2-klein-4B")


def test_regate_runs_after_resolutions_and_is_idempotent() -> None:
    """apply_model_resolutions re-runs gate_functions against the rebound
    bindings; gate marks are gate-owned (cleared on re-gate), setup failures
    survive. SDK v2 (pgw#647) deleted the compute-capability gate (the fit
    ladder owns precision), so the regate mechanic is pinned on the
    surviving missing-library gate."""
    spec = _spec(resources=Resources(gpu=True, libraries=("nunchaku",)))
    ex = _executor(spec)

    without_lib = {"gpu_total_mem": 48 * 1024**3, "gpu_free_mem": 48 * 1024**3,
                   "gpu_sm": "90", "installed_libs": []}
    with_lib = dict(without_lib, installed_libs=["nunchaku"])

    ex.gate_functions(without_lib)
    assert ex.unavailable["generate"][0] == "missing_cuda_library"

    # Re-gate with the library present: the gate-owned mark clears.
    ex.gate_functions(with_lib)
    assert "generate" not in ex.unavailable

    # A setup failure is NOT gate-owned and survives a re-gate.
    ex.unavailable["generate"] = ("setup_failed", "boom", {})
    ex.gate_functions(with_lib)
    assert ex.unavailable["generate"][0] == "setup_failed"

    # Resolutions re-run the gates using the remembered probe.
    ex.unavailable.pop("generate")
    ex.apply_model_resolutions({"acme/z-image": (_DIGEST_PICK, "", "")})
    assert ex._last_gpu_info is not None
    assert "generate" in ex.serve_plans


#: th#1803: the hub ladder pins a CHECKPOINT by digest, never a flavor.
_DIGEST_PICK = "acme/z-image@sha256:" + "1a" * 32


def test_rebind_pick_is_the_single_fold() -> None:
    """pgw#1148: the ladder/`flavor=` arm is GONE. A hub pick is a DIGEST
    (th#1803) or a cast, and the round-trip guard is unchanged."""
    b = Hub("acme/z-image")

    # hub path: resolved_ref is checked, cast is stamped
    out = rebind_pick(b, resolved_ref="", cast="fp8")
    assert out.storage_dtype == "fp8" and wire_ref(out) == "acme/z-image"
    # non-normal hub spelling still round-trips (':prod' is the elided form
    # since th#1276)
    out = rebind_pick(b, resolved_ref="acme/z-image:prod")
    assert wire_ref(out) == "acme/z-image"
    # ...and a pick at a genuinely DIFFERENT tag is refused, because the
    # rebound binding could not re-mint it (two residency identities).
    with pytest.raises(ValueError):
        rebind_pick(b, resolved_ref="acme/z-image:latest")

    with pytest.raises(ValueError):
        rebind_pick(b, resolved_ref="acme/OTHER")  # ref mismatch
    # A pick that still carries a `#flavor` is refused outright (§1.32(d)).
    with pytest.raises(FlavorSelectorRemoved):
        rebind_pick(b, resolved_ref="acme/z-image#fp8")
