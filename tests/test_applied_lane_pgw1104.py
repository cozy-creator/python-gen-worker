"""The lane a request REPORTS must be the lane its weights EXECUTE.

An endpoint can bind a bare tag with an EMPTY flavor and then quantize its DiT
to w8a8 fp8 with torchao INSIDE `setup()`. If `_served_execution_lane` is a pure
function of the BINDING, every request reports `bf16-w16a16+compiled` while an
fp8 DiT executes. The lane id is a KEY (quant verdicts, compile cells, floors,
pricing, the executed-lane proof), so the label is then wrong everywhere the key
is joined.

The control matters as much as the positive test:
`test_an_unapplied_recipe_keeps_the_binding_lane` is why a STATIC
`handles=`-style declaration was rejected. A recipe runtime-gated on sm89 and on
the compile preflight would be declared fp8 on the card that skips it — the same
defect in the UNSAFE direction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

import msgspec
import pytest

import gen_worker
import gen_worker.executor as ex_mod
from gen_worker import (
    RequestContext,
    Resources,
    Slot,
    activity,
    endpoint,
    worker_function,
)
from gen_worker.executor import Executor
from gen_worker.models import execution_lanes as lanespec
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from gen_worker.models import store as store_mod

REF = "tensorhub/minimax-h3:serve-narrowed"

#: Flipped by the fixture to stand in for `serve_recipe.w8a8_capable()`.
W8A8_CAPABLE = True


class GenIn(msgspec.Struct):
    prompt: str = "x"
    model: str = ""


class Out(msgspec.Struct):
    y: str = "ok"


class Pipe:
    """A self-hydrating pipeline whose weights setup() converts in place."""

    def __init__(self) -> None:
        self.quantized = False

    @classmethod
    def from_pretrained(cls, path: Any, **_kw: Any) -> "Pipe":
        return cls()

    def __call__(self, *_a: Any, **_kw: Any) -> dict:
        return {"out": 1}


@endpoint(models={"pipeline": Slot(Pipe, selected_by="model")},
          resources=Resources(gpu=True))
class H3:
    def setup(self, pipeline: Pipe) -> None:
        self.pipeline = pipeline
        # The shape of `minimax_h3.serve_recipe.quantize_dit`: a capability
        # gate, an in-place `quantize_()`, then the report — never before.
        if not W8A8_CAPABLE:
            return
        pipeline.quantized = True
        gen_worker.report_applied_lane(
            "transformer", "fp8-w8a8-dynamic", modules=300, kept_bf16=70)

    @worker_function()
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        return Out()


@pytest.fixture
def boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One real `ensure_setup()`. Nothing about the lane derivation is faked:
    the executor's own AppliedLaneScope, `_record_applied_lanes` and
    `_served_execution_lane` all run."""
    events: List[Any] = []
    real_emit = activity.emit_event

    def _spy(kind: str, detail: str = "", **kw: Any) -> Any:
        events.append((kind, kw.get("phase", ""), detail))
        try:
            return real_emit(kind, detail, **kw)
        except Exception:
            return None

    monkeypatch.setattr(activity, "emit_event", _spy)
    monkeypatch.setattr(ex_mod.activity_mod, "emit_event", _spy)

    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(extract_specs(H3), _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _fake_download(ref: str, **_kw: Any) -> Path:
        p = tmp_path / ref.replace("/", "_").replace(":", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(store_mod, "ensure_local", _fake_download)

    from gen_worker import dispatch

    eff = ex._dispatched_spec(
        ex.specs["generate"],
        {"pipeline": dispatch.SlotOrder(ref=REF, components=())},
    )
    snaps = {
        REF: pb.Snapshot(
            digest="blake3:" + "a" * 64,
            files=[pb.SnapshotFile(
                path="model.safetensors", size_bytes=5, blake3="cd" * 32,
                url="http://r2.invalid/p")],
        )
    }
    return ex, eff, snaps, events


def test_a_serve_time_recipe_moves_the_reported_lane(boot) -> None:
    """The red test. The binding is a bare tag (empty flavor, bf16), setup()
    quantizes to w8a8 fp8 and says so — the served lane is the fp8 one."""
    ex, eff, snaps, events = boot

    async def _go() -> None:
        inst = await ex.ensure_setup(eff, snaps)
        assert inst.pipeline.quantized is True
        rec = ex._classes[eff.instance_key]
        assert [a.body for a in rec.applied_lanes] == ["fp8-w8a8-dynamic"]
        # The binding on its own still says bf16 — the divergence is real,
        # not a mis-resolved binding.
        assert ex._bound_execution_body(eff) == "bf16-w16a16"
        # This rig has no compiled cell, so the honest execution axis
        # is `+eager`. Until ie#655 this line read `+compiled` — the lane
        # table's compiled-only PLAN for the w8a8 body coerced an observed
        # eager posture, and the test encoded the over-claim.
        assert ex._served_execution_lane(eff) == "fp8-w8a8-dynamic+eager"

    asyncio.run(_go())


def test_the_divergence_is_a_wire_row(boot) -> None:
    """A lane that stopped following its checkpoint must be visible from the
    events alone — pgw#1093 cost an issue to a fact that was only inferable
    from allocated-bytes rows."""
    ex, eff, snaps, events = boot

    async def _go() -> None:
        await ex.ensure_setup(eff, snaps)
        rows = [(phase, detail) for kind, phase, detail in events
                if kind == activity.KIND_APPLIED_LANE]
        assert len(rows) == 1
        phase, detail = rows[0]
        assert phase == "transformer"
        assert "applied=fp8-w8a8-dynamic" in detail
        assert "modules=300 kept_bf16=70" in detail
        assert "bound=bf16-w16a16" in detail

    asyncio.run(_go())


def test_an_unapplied_recipe_keeps_the_binding_lane(
    boot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the report is the mechanism and a declaration is not: the recipe's
    capability gate refuses, no conversion happens, and the lane stays the
    binding's. A static declaration would over-claim here."""
    ex, eff, snaps, events = boot
    monkeypatch.setattr(__import__(__name__), "W8A8_CAPABLE", False)

    async def _go() -> None:
        inst = await ex.ensure_setup(eff, snaps)
        assert inst.pipeline.quantized is False
        assert ex._classes[eff.instance_key].applied_lanes == []
        assert ex._served_execution_lane(eff).startswith("bf16-w16a16+")
        assert not [e for e in events if e[0] == activity.KIND_APPLIED_LANE]

    asyncio.run(_go())


def test_the_lane_dies_with_the_instance(boot) -> None:
    """An applied lane describes THESE weights. A vacated record must not keep
    reporting fp8 for the bf16 instance that replaces it."""
    ex, eff, snaps, events = boot

    async def _go() -> None:
        await ex.ensure_setup(eff, snaps)
        rec = ex._classes[eff.instance_key]
        assert rec.applied_lanes
        await ex._vacate_record(rec)
        assert rec.applied_lanes == []
        assert ex._served_execution_lane(eff).startswith("bf16-w16a16+")

    asyncio.run(_go())


# --- the vocabulary is platform-wide: an endpoint never extends it ---------


def test_an_unknown_lane_body_is_refused_at_report_time() -> None:
    with pytest.raises(ValueError) as e:
        gen_worker.report_applied_lane("transformer", "fp8-w8a8")  # no scale axis
    assert "not a known lane body" in str(e.value)
    with pytest.raises(ValueError):
        gen_worker.report_applied_lane("transformer", "fp6-w6a6-dynamic")
    # The execution axis is the platform's, never the author's.
    with pytest.raises(ValueError):
        gen_worker.report_applied_lane("transformer", "fp8-w8a8-dynamic+compiled")


def test_every_reportable_body_composes_at_either_observed_posture() -> None:
    """Whatever an endpoint may report must compose into a lane whose BODY is
    the platform vocabulary (`known_execution_lane_bodies()`, the byte-identical
    twin of the hub's `precision.KnownExecutionLaneBodies()` — what verdicts
    key on) at WHICHEVER posture was observed.

    the reported set is deliberately WIDER than
    `known_execution_lanes()`. That list is the lanes the platform CHOOSES —
    an owner ladder, an admin override, a resolution — and `fp8-w8a8-dynamic`
    is compiled-only there because eager w8a8 is unmeasured, not because it
    cannot happen. It happens: a serve-time recipe quantizes and the self-mint
    then declines (wan-2.2, `insufficient_vram` on an H100). Naming that state
    is the whole point; coercing it into the choosable set is the defect."""
    for body in lanespec.known_execution_lane_bodies():
        for compiled in (True, False):
            lane = lanespec.observed_execution_lane(body, compiled)
            assert lane.execution == ("compiled" if compiled else "eager")
            assert lanespec.execution_lane_body_id(lane) == body


def test_a_report_outside_setup_is_not_attributed_and_never_raises() -> None:
    """`arm_compile`'s contract: hub-less runs and unit rigs call it too."""
    assert gen_worker.report_applied_lane("transformer", "fp8-w8a8-dynamic") is False
    with provision.AppliedLaneScope() as scope:
        assert gen_worker.report_applied_lane(
            "transformer", "fp8-w8a8-dynamic", modules=300, kept_bf16=70) is True
    assert [a.body for a in scope.applied] == ["fp8-w8a8-dynamic"]
    # The scope closed: a later report reaches nothing.
    assert gen_worker.report_applied_lane("transformer", "fp8-w8a8-dynamic") is False
