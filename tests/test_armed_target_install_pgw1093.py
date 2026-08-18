"""Armed at setup with no installed target is an IMPOSSIBLE state, and a target
that armed and then DEGRADED must never read as one that was never installed.

A pod that arms a regional JIT-intake target, compiles it whole, and then serves
every request eager reports only `metrics.lane=…+eager`,
`fallback_reason=uncompiled` and `self_mint_skipped/boot_ended_uncompiled`. TWO
different defects produce that exact reading, and they must be distinguishable:

  (A) NEVER INSTALLED — `_install_compile_targets` dropped the candidate on one
      of three exits that emitted nothing at all: `cfg is None`, the bare
      `if not has_compile_target(...): continue`, or an empty candidate list
      (whose only confession is gated on a MANDATORY quant lane, which
      `bf16-w16a16` is not).
  (B) INSTALLED THEN DEGRADED — `_guarded_regional` caught an exception on a
      served call, set `degraded=True` and cleared the region. A graph break
      and a `DeclaredRangeExceeded` each get their own wire row; every OTHER
      exception class took `logger.warning` and reached the wire as nothing,
      and on a hub-spawned pod stdout is unreachable. `is_compile_armed` then
      reads False and every downstream reader falls through to the generic
      `uncompiled`.

REVERT-TURNS-RED: each test below fails without the fix — no posture is
recorded, no `serve_eager_posture` row is emitted, and `_eager_posture()`
answers `uncompiled` for a target that demonstrably armed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

import msgspec
import pytest

from gen_worker.serving_facts import FactsUnavailable as _FU

_NO_FACTS = _FU(owed_by="a test that resolves no catalog")
import torch

import gen_worker
import gen_worker.executor as ex_mod
from gen_worker import (
    Compile,
    DynamicDim,
    RequestContext,
    Resources,
    Slot,
    activity,
    compiled_graph_adopt,
    endpoint,
    worker_function,
)
from gen_worker import compile_cache as cc
# pgw#1331: the marker READERS live beside the marker's definition.
from gen_worker import compile_facts
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from gen_worker.models import store as store_mod

FAMILY = "minimax-h3"
REF = "tensorhub/minimax-h3@serve-narrowed"

_COMPILE = Compile(
    family=FAMILY,
    shapes=((1344, 768, 124),),
    targets=("transformer",),
    regional=True,
    dynamic=(DynamicDim("sequence", min=2, max=116126),),
)


class GenIn(msgspec.Struct):
    prompt: str = "x"
    # minimax-h3's slot is `Slot(..., selected_by="model")`; the payload must
    # carry the field or registration refuses.
    model: str = ""
    num_inference_steps: int = 2


class Out(msgspec.Struct):
    y: str = "ok"


class _Block(torch.nn.Module):
    def forward(self, x: Any, **_kw: Any) -> Any:
        return x


class _DiT(torch.nn.Module):
    """Stands in for the 20.1B denoiser: a real Module (the regional rollback
    walks `.modules()`), owning `compile_repeated_blocks` so the REGIONAL arm
    path is the one exercised."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block(), _Block()])

    def compile_repeated_blocks(self, dynamic: Any = None, fullgraph: bool = False) -> None:
        return None

    def forward(self, hidden_states: Any = None, **_kw: Any) -> Any:
        return hidden_states


class LazyPipe:
    """A `ModularPipeline`-shaped slot: worker-loaded (class-annotated), but
    the weight-bearing component hydrates only INSIDE setup(), so the
    injection-time arm legitimately declines `no_compile_target` first."""

    def __init__(self) -> None:
        self.transformer: Any = None

    @classmethod
    def from_pretrained(cls, path: Any, **_kw: Any) -> "LazyPipe":
        return cls()

    def hydrate(self) -> None:
        self.transformer = _DiT()

    def __call__(self, *_a: Any, **_kw: Any) -> dict:
        self.transformer.forward(hidden_states=None)
        return {"out": 1}


@endpoint(
    models={"pipeline": Slot(LazyPipe, selected_by="model")},
    resources=Resources(gpu=True),
    compile=_COMPILE,
)
class H3:
    def setup(self, pipeline: LazyPipe) -> None:
        self.pipeline = pipeline
        pipeline.hydrate()
        self.armed = gen_worker.arm_compile(pipeline)

    @worker_function()
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        self.pipeline()
        return Out()


# ---------------------------------------------------------------------------


@pytest.fixture
def boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One real `ensure_setup()` on the real arm path.

    TWO environment fakes, both about the CARD and neither about the defect:
    `torch.cuda.is_available` (`arming_block` refuses a cardless process), and
    the `apply_low_vram_config` seam. `arm_jit_intake`, `ArmingScope`, the warm
    plan and the target install are all the production code.

    The applier is stubbed for the reason `test_rung_ladder_pgw1206` stubs it
    (the established th#1043 seam): the diffusers placement hooks are not the
    code under test and a cardless box cannot run them.
    Stubbing at `place_pipeline`'s applier call is the only seam that
    works: pinning free VRAM high instead would select the resident rung and
    route into `pipeline.to("cuda")`, which a cardless box cannot satisfy.
    """
    import torch

    from gen_worker.models import memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _fake_apply(
        pipeline: Any, *, mode: str = "auto", logger: Any = None, **_kw: Any,
    ) -> dict:
        return {"mode": mode}

    monkeypatch.setattr(memory, "apply_low_vram_config", _fake_apply)

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
    monkeypatch.setattr(cc.logger, "warning", cc.logger.warning)

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
        {"pipeline": dispatch.SlotOrder(ref=REF, facts=_NO_FACTS)},
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


def _postures(events: List[Any]) -> List[str]:
    return [phase for kind, phase, _d in events if kind == "serve_eager_posture"]


def test_the_healthy_boot_installs_its_target(boot) -> None:
    """The control. Same shape, nothing broken: the scope-armed, lazily
    hydrated, `selected_by` slot ends the boot OWNING an installed target and
    serving compiled. Without this, the two red tests below could pass for the
    wrong reason."""
    ex, eff, snaps, events = boot

    async def _go() -> None:
        inst = await ex.ensure_setup(eff, snaps)
        rec = ex._classes[eff.instance_key]
        assert inst.armed is True
        assert rec.compile_targets, "the arm resolved; a target must own it"
        assert rec.eager_posture == ""
        assert ex._served_execution_lane(eff).endswith("+compiled")
        assert _postures(events) == []

    asyncio.run(_go())


def test_an_armed_object_that_resolves_no_target_is_typed_not_silent(
    boot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defect class (A). The arm succeeds inside setup(); by install time the
    object no longer resolves the declared target (the pgw#1078 D2 shape one
    layer up — a component replaced under a live arm).

    On 0.98.0 this is `if not has_compile_target(...): continue` — a bare
    `continue`. No posture, no counter, no event, and every request afterwards
    reports the generic `uncompiled`. That reading is identical to a healthy
    pod that simply declares no compile, which is why nobody could see it.
    """
    ex, eff, snaps, events = boot
    real_has = cc.has_compile_target
    seen: dict = {"armed": False}

    def _has(pipeline: Any, cfg: Any) -> bool:
        # Truthful until the object is armed; afterwards the declared target
        # no longer resolves on it.
        if cc.is_compile_armed(pipeline):
            seen["armed"] = True
            return False
        return real_has(pipeline, cfg)

    monkeypatch.setattr(cc, "has_compile_target", _has)
    monkeypatch.setattr(ex_mod.compile_cache, "has_compile_target", _has)

    async def _go() -> None:
        await ex.ensure_setup(eff, snaps)
        rec = ex._classes[eff.instance_key]
        assert seen["armed"], "the arm must have succeeded for this to be the case"
        assert not rec.compile_targets

        # THE INVARIANT: an arm that returned True and owns no installed
        # target is a typed, wire-visible refusal — never a bare `continue`.
        assert rec.eager_posture == (
            compiled_graph_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value), (
            "0.98.0 records NOTHING here: the boot compiles graphs it can "
            "never dispatch to and every request reports `uncompiled`"
        )
        assert compiled_graph_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value in _postures(events)
        assert any(
            phase == compiled_graph_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value
            and kind == activity.KIND_SERVE_DEGRADE
            for kind, phase, _d in events
        ), "the terminus must ALSO ride the degrade stream, not only a posture"
        # And the request path names the real cause instead of the terminal
        # fallback token.
        assert ex._eager_posture(eff, rec) == (
            compiled_graph_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value)

    asyncio.run(_go())


def test_a_boot_warmup_degrade_is_named_not_reported_as_uncompiled(
    boot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defect class (B), and the reason (A) was misdiagnosed.

    `_bind_compile_guard` installs the revocation callback inside
    `_install_compile_targets`, which runs AFTER the boot warmup. So a target
    that armed, compiled, and then broke DURING that warmup has no callback to
    fire and no record to write on — pgw#1082's entire confession path is
    structurally unreachable for it.

    A `RuntimeError` from a kernel is neither a graph break nor a
    `DeclaredRangeExceeded`, so on 0.98.0 it takes `logger.warning` and reaches
    the wire as NOTHING, and `_eager_posture()` answers `uncompiled` — the
    exact token request `55971bce` carries.
    """
    ex, eff, snaps, events = boot

    # The COMPILED call breaks; the eager fallback the guard then runs
    # succeeds — the real degrade shape (pgw#672: a broken optimization never
    # kills a serving worker, it just silently stops being an optimization).
    calls = {"n": 0}

    def _explode(self: Any, hidden_states: Any = None, **_kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("fa3 kernel refuses this packed sequence")
        return hidden_states

    monkeypatch.setattr(_DiT, "forward", _explode)

    async def _go() -> None:
        await ex.ensure_setup(eff, snaps)
        rec = ex._classes[eff.instance_key]

        # The target IS installed — this is NOT the never-installed class, and
        # that is precisely the distinction 0.98.0 could not express.
        assert rec.compile_targets, (
            "the arm resolved and the target installed; the failure is an "
            "EXECUTION failure, not a wiring one"
        )
        pipe = next(iter(rec.compile_targets.values())).pipeline
        assert cc.is_compile_armed(pipe) is False, "the guard degraded it"
        assert "fa3 kernel refuses" in compile_facts.degrade_reason(pipe)

        assert rec.eager_posture == compiled_graph_adopt.EagerPhase.COMPILED_DEGRADED.value, (
            "0.98.0 leaves this EMPTY, so `_eager_posture` falls through to "
            "the generic `uncompiled` and an installed-then-degraded target "
            "reads exactly like one that was never installed"
        )
        assert ex._eager_posture(eff, rec) == (
            compiled_graph_adopt.EagerPhase.COMPILED_DEGRADED.value)
        assert ex._eager_posture(eff, rec) != compiled_graph_adopt.EagerPhase.UNCOMPILED.value

        # And it is a dated, greppable row — not a log line on a pod whose
        # stdout nobody can reach.
        assert any(
            kind == activity.KIND_SERVE_DEGRADE
            and phase == cc.COMPILED_DEGRADE_TOKEN
            for kind, phase, _d in events
        ), "every permanent degrade confesses, whatever raised it"

    asyncio.run(_go())
