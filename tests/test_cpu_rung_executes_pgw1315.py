"""pgw#1315 — the always-runs guarantee holds on a REACTIVE descent, not only at plan time.

The guarantee (Paul, 2026-08-17): *"the answer should be IT ALWAYS RUNS no
matter what, just horribly inefficiently"*; §1.35 second amendment: *"even a pod
without a GPU, heck; we can run it CPU only… 'This model does not run on this
card' is never an acceptable terminal state."*

Two things falsified it, and both were in OUR code, not in any card:

1. **``FLOOR_CPU_RUNG_UNEXECUTABLE``** — ``rung.descend`` deliberately stopped
   one rung ABOVE ``cpu`` and the executor turned that into a ``serve_degrade``
   event plus a retryable request. So the honest answer was *yes at plan time,
   NO on a reactive descent that reaches the bottom*. The refusal was correct in
   FORM (it named our build, never the card), which is exactly why it was
   actionable: make the build execute the rung. That is what this file proves.
2. **``_move_pipeline_to_cpu`` swallowed ``HostRamMoveRefusedError``** into a
   DEBUG line, after which the CPU-rollback check resurfaced it as a generic
   ``RuntimeError("… mixed-device … rollback failed")``. The one case where the
   host-move guard LEGITIMATELY stops a degrade became indistinguishable from a
   bug. The guard is correct and stays on; the TYPE now propagates.

RED against 66706529: (1) fails with ``RuntimeError: CUDA error: out of
memory`` — the walk refuses to hand out ``cpu`` — and (2) fails with the generic
mixed-device ``RuntimeError``.

Deliberately compile-free and inference-free (workspace local-inference rule):
the seam under test is the ladder WALK and the placement APPLIER, so the
diffusers hooks are stubbed exactly as ``test_rung_ladder_pgw1206`` stubs them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pytest

from gen_worker.api.binding import wire_ref
from gen_worker.api.errors import HostRamMoveRefusedError
from gen_worker.models import memory as memory_mod
from gen_worker.models import rung
from gen_worker.models.memory import apply_low_vram_config, place_pipeline

_LOG = logging.getLogger("pgw1315")


# ---------------------------------------------------------------------------
# 0. the ladder hands out the bottom rung
# ---------------------------------------------------------------------------


def test_the_reactive_walk_reaches_the_cpu_rung() -> None:
    """``sequential`` descends to ``cpu``; the unexecutable floor is DELETED.

    ``test_descent_floor_th1867`` wrote the deletion condition itself: *"When
    pgw#1212 makes the rung real the token must be DELETED — a floor that no
    longer exists must not be left pointing somewhere else."*
    """
    nxt = rung.descend("sequential")
    assert nxt is rung.CPU
    assert rung.descent_floor("sequential") is None
    assert not hasattr(rung, "FLOOR_CPU_RUNG_UNEXECUTABLE"), (
        "the rung executes; a floor saying it cannot is a diagnostic that "
        "outlived its cause")
    # The bottom is still a bottom: nothing below it, and it does not climb.
    assert rung.descend("cpu") is None
    assert rung.descent_floor("cpu") == rung.FLOOR_LADDER_EXHAUSTED


def test_the_cpu_rung_is_charged_host_ram() -> None:
    """A CPU-placed pipeline keeps its WHOLE tree in host RAM — that is what
    the rung IS. Declaring otherwise let the loader hand a CPU-rung load the
    per-component staging discount, which is the pgw#1063 admission lie."""
    assert rung.touches_host_ram("cpu")
    assert "cpu" in rung.PLACEMENT_LADDER
    assert rung.floor_of("sequential", "cpu") == "cpu"


# ---------------------------------------------------------------------------
# 1. THE acceptance: a reactive OOM descent reaches the CPU rung and SERVES
# ---------------------------------------------------------------------------


def _arm(monkeypatch: pytest.MonkeyPatch, serves_at: str) -> List[str]:
    """The established th#1043 seam: the ladder WALK is under test, the
    diffusers hooks are not.

    Torch-free on purpose — ``place_pipeline`` asks ``cuda_ready()``, not torch,
    and CI installs no torch extra. A ladder guard that SKIPS on the runner is
    a guard that is not there.
    """
    monkeypatch.setattr(memory_mod, "cuda_ready", lambda: True)
    monkeypatch.setattr(memory_mod, "_move_pipeline_to_cpu", lambda *_: None)
    monkeypatch.setattr(memory_mod, "repair_device_placement", lambda *_: [])
    monkeypatch.setattr(memory_mod, "flush_memory", lambda: None)
    attempted: List[str] = []

    def fake_apply(pipeline: object, *, mode: str, logger: object = None) -> dict:
        attempted.append(mode)
        if mode != serves_at:
            raise RuntimeError("CUDA error: out of memory")
        return {"mode": mode}

    monkeypatch.setattr(memory_mod, "apply_low_vram_config", fake_apply)
    return attempted


def test_a_reactive_descent_that_reaches_the_bottom_SERVES(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE test this issue exists for.

    A pipeline that OOMs at every offload rung lands on ``cpu`` and RUNS. The
    old build stopped at ``sequential`` and returned the request retryable —
    "this model does not run here", said about our own ladder.
    """
    attempted = _arm(monkeypatch, serves_at="cpu")

    applied = place_pipeline(object(), mode="off", logger=_LOG)

    assert attempted == ["off", "model_offload", "group_offload", "sequential", "cpu"]
    assert applied["mode"] == "cpu"
    assert applied["oom_demotions"] == 4
    assert applied["requested_mode"] == "off"


def test_the_cpu_rung_is_APPLIED_not_merely_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EXECUTION, not registration: the applier must actually put the pipeline
    on the host and stamp the rung, or ``mode="cpu"`` is a label on a pipeline
    still sitting on a card that just OOM'd."""
    monkeypatch.setattr(memory_mod, "cuda_ready", lambda: True)

    pipe = _FakePipeline()
    applied = apply_low_vram_config(pipe, mode="cpu", logger=_LOG)

    assert applied["mode"] == "cpu"
    assert pipe.moved_to == ["cpu"], "the weights never left the device"
    assert memory_mod.low_vram_mode(pipe) == "cpu"
    # No CUDA-only hook may be armed on the rung whose whole premise is that
    # there is no usable device to onload to.
    assert not applied["model_offload"]
    assert not applied["group_offload"]
    assert not applied["sequential_offload"]


def test_the_cpu_rung_CONFESSES_like_every_other_host_touching_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1312 made every CPU-touching activation emit ONE typed
    `serve_degrade`. The CPU rung is the LOUDEST of them (~40x), so it may not
    be the one route that reaches a host-resident placement silently.

    Read off the REAL activity sink, so this is the ActivityUpdate a hub would
    actually bank."""
    from test_offload_always_allowed_pgw1312 import _Events

    monkeypatch.setattr(memory_mod, "cuda_ready", lambda: True)

    with _Events() as events:
        apply_low_vram_config(_FakePipeline(), mode="cpu", logger=_LOG)

    engaged = events.offload_engaged()
    assert len(engaged) == 1, "expected ONE confession from the CPU rung"
    assert "`cpu`" in engaged[0].detail


def test_a_cardless_pod_stamps_the_SAME_rung_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan time and the reactive descent must name one rung. A cardless pod
    used to get a bare ``{"mode": "cpu"}`` describing a placement nobody
    applied, so ``low_vram_mode()`` read ``""`` — the ladder's own view of that
    pipeline was "never placed"."""
    monkeypatch.setattr(memory_mod, "cuda_ready", lambda: False)

    pipe = _FakePipeline()
    applied = place_pipeline(pipe, mode="auto", logger=_LOG)

    assert applied["mode"] == "cpu"
    assert memory_mod.low_vram_mode(pipe) == "cpu"


# ---------------------------------------------------------------------------
# 2. the guard's refusal keeps its TYPE across the rollback seam
# ---------------------------------------------------------------------------


def test_a_guard_refusal_during_rollback_surfaces_TYPED(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GEN_WORKER_HOST_MOVE_GUARD`` refusing the rollback move is a DIFFERENT
    fact from "the rollback left the pipeline mixed-device", and it was
    reported as the second. Red-verified by restoring the swallow."""
    monkeypatch.setattr(memory_mod, "cuda_ready", lambda: True)
    monkeypatch.setattr(memory_mod, "flush_memory", lambda: None)
    # The refused move is exactly why components are still on the device: the
    # generic check below then reported THAT, and the guard's verdict was gone.
    monkeypatch.setattr(
        memory_mod, "repair_device_placement", lambda *_: ["transformer"])

    refusal = HostRamMoveRefusedError(
        incoming_bytes=1 << 35, available_bytes=1 << 30,
        floor_bytes=1 << 33, limit_bytes=1 << 34,
    )

    def fake_apply(pipeline: object, *, mode: str, logger: object = None) -> dict:
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(memory_mod, "apply_low_vram_config", fake_apply)
    pipe = _FakePipeline(refuse_move=refusal)

    with pytest.raises(HostRamMoveRefusedError) as caught:
        place_pipeline(pipe, mode="off", logger=_LOG)

    assert caught.value is refusal
    assert "mixed-device" not in str(caught.value)
    # The OOM that provoked the rollback stays attached as the cause.
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "out of memory" in str(caught.value.__cause__)


def test_an_untyped_move_failure_still_falls_back_to_the_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The propagation is TYPE-scoped. A best-effort ``.to('cpu')`` that fails
    for any other reason is still swallowed — ``repair_device_placement`` is the
    check that decides whether the rollback actually worked, and promoting every
    move error to fatal would turn recoverable descents into refusals."""
    monkeypatch.setattr(memory_mod, "cuda_ready", lambda: True)
    monkeypatch.setattr(memory_mod, "flush_memory", lambda: None)
    monkeypatch.setattr(memory_mod, "repair_device_placement", lambda *_: [])
    attempted: List[str] = []

    def fake_apply(pipeline: object, *, mode: str, logger: object = None) -> dict:
        attempted.append(mode)
        if mode != "model_offload":
            raise RuntimeError("CUDA error: out of memory")
        return {"mode": mode}

    monkeypatch.setattr(memory_mod, "apply_low_vram_config", fake_apply)
    # The REAL `_move_pipeline_to_cpu` runs here, against a `.to` that raises
    # something the guard did not produce.
    pipe = _FakePipeline(refuse_move=ValueError("some driver hiccup"))

    applied = place_pipeline(pipe, mode="off", logger=_LOG)

    assert applied["mode"] == "model_offload"
    assert attempted == ["off", "model_offload"]


# ---------------------------------------------------------------------------
# 3. the executor's mid-inference walk learns the bottom rung
# ---------------------------------------------------------------------------


def _executor(monkeypatch: pytest.MonkeyPatch, pipe: Any) -> tuple[Any, Any]:
    from gen_worker.executor import Executor
    from gen_worker.registry import extract_specs

    from harness.toy_endpoints import ModelBoundEndpoint

    spec = extract_specs(ModelBoundEndpoint)[0]

    async def _send(_msg: Any) -> None:
        return None

    ex = Executor([spec], _send)
    monkeypatch.setattr(ex, "_slot_pipeline", lambda _s, _slot: pipe)

    async def _no_refusal(*_a: Any, **_k: Any) -> str:
        return ""

    monkeypatch.setattr(ex, "_refuse_unfittable_offload", _no_refusal)
    return ex, spec


class _Ctx:
    cancelled = False

    def log(self, *_a: Any, **_k: Any) -> None:
        return None


def test_the_executor_walk_learns_cpu_as_the_next_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production seam that makes the descent STICK: an instance that OOMs
    on ``sequential`` is quarantined with ``cpu`` as its learned per-ref floor,
    so the hub's retry reloads onto the CPU rung. Before this it learned
    nothing and emitted ``cpu_rung_unexecutable``."""
    pipe = _FakePipeline(mode="sequential")
    ex, spec = _executor(monkeypatch, pipe)
    slot = next(iter(spec.models))
    ref = wire_ref(spec.models[slot])

    asyncio.run(ex._quarantine_for_oom(
        spec, _Ctx(), RuntimeError("CUDA error: out of memory")))

    assert ex.degraded_floor.get(ref) == "cpu"
    assert ex._placement_mode(spec, ref) == "cpu"


def test_the_degraded_wire_token_for_the_cpu_rung_is_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tensorhub matches ``FnDegraded.ran`` against its RunMode vocabulary
    EXACTLY, and ``cpu`` is a member in its own right. A descent onto the CPU
    rung reported as ``offload`` is a wrong measurement, so the rung's own
    ``run_mode`` travels, not the tail's."""
    pipe = _FakePipeline(mode="sequential")
    ex, spec = _executor(monkeypatch, pipe)

    asyncio.run(ex._quarantine_for_oom(
        spec, _Ctx(), RuntimeError("CUDA error: out of memory")))

    plan = ex.serve_plans.get(spec.name)
    assert plan is not None
    assert plan.ran == rung.RUN_CPU
    assert plan.run_mode == rung.RUN_CPU


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Carries the three offload capabilities the executor's walk looks for,
    plus a ``.to`` that records every move — the observable that separates
    "the rung was applied" from "the rung was named"."""

    def __init__(
        self, mode: str = "", refuse_move: Optional[BaseException] = None,
    ) -> None:
        self.moved_to: List[str] = []
        self._refuse_move = refuse_move
        self.hooks: Dict[str, bool] = {}
        if mode:
            setattr(self, "_cozy_low_vram_mode", mode)

    def to(self, device: Any, *_a: Any, **_k: Any) -> "_FakePipeline":
        # Only the HOST move is refusable — that is the guard's whole subject,
        # and a fake that also refuses `to("cuda")` would pass this file's
        # rollback arm without the rollback ever being reached.
        if self._refuse_move is not None and str(device) == "cpu":
            raise self._refuse_move
        self.moved_to.append(str(device))
        return self

    def enable_model_cpu_offload(self, *_a: Any, **_k: Any) -> None:
        self.hooks["model_offload"] = True

    def enable_group_offload(self, *_a: Any, **_k: Any) -> None:
        self.hooks["group_offload"] = True

    def enable_sequential_cpu_offload(self, *_a: Any, **_k: Any) -> None:
        self.hooks["sequential"] = True
