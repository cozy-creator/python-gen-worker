"""pgw#760: important fail-soft outcomes ride typed wire events.

Doctrine (Paul, verbatim in spirit): errors should be exposed to the
orchestrator so the orchestrator can report on them — an important error
that only reaches a local logger does not exist on hub-spawned workers
(no stdout). These are the red halves for the audit's MUST-REPORT class:
each forced failure must produce an ActivityUpdate whose ``phase`` names
the reason class and whose ``detail`` names the identifiers. Fail-soft
BEHAVIOR is asserted unchanged in every case.

Capture is at ``activity._emit`` — the exact envelope the stream sink
sends (the pgw#733 test convention), so kind/phase/detail are asserted as
wired, not through a test double of the event API.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker import activity, capability_renewal, hot_swap, preload
from gen_worker.models import lane_residency_gate, residency
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.utils import lora



def _gate(kind: str) -> "contextlib.AbstractContextManager[None]":
    """`Executor._wire_turn_gate`'s factory, as a test double.

    pgw#1215 step 4: an ungated router is a typed refusal, not a mode — so a
    test that enables concurrent routing (or hands `_run_warm` a job) wires
    the gate exactly as production does.
    """
    return contextlib.nullcontext()


def _router(*, fail_closed: bool = False) -> "hot_swap.Router":
    router = hot_swap.Router(fail_closed=fail_closed)
    router.set_turn_gate(_gate)
    return router


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


def _by_kind(events: List[Any], kind: str) -> List[Any]:
    return [e for e in events if e.kind == kind]


# pgw#1187 DELETED the two `trt_engine.enable` rows that stood here — the
# pgw#733 pattern's TRT half. TensorRT was removed from the platform outright
# on Paul's 2026-08-12 ruling, so their subject is gone; they die with it and
# are not ported (DESIGN-RULINGS §4.34). The pattern they guarded is still
# guarded by the `aot_serve` and `hot_swap` rows in this file.


# ---------------------------------------------------------------------------
# hot_swap — background warm/heal compile failure
# ---------------------------------------------------------------------------


def test_warm_compile_failure_rides_typed_event(events: List[Any]) -> None:
    router = _router()
    router.enable()
    sig = ("unet", (("T", (1, 4), "torch.bfloat16", "cpu"),))
    with router.lock:
        router.pending.add(sig)

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("inductor exploded")

    job = hot_swap._WarmJob(
        router=router, label="unet", sig=sig, compiled=boom,
        args=(), kwargs={}, device=None, grad_mode="grad",
        autocast_dtype=None, turn=_gate,
    )
    hot_swap._run_warm_compile(job)

    # fail-soft behavior unchanged: sig is failed, routed eager, no raise
    assert sig in router.bg_failed
    got = _by_kind(events, activity.KIND_SERVE_DEGRADE)
    assert [e.phase for e in got] == ["warm_compile_failed"]
    assert "target=unet" in got[0].detail
    assert "inductor exploded" in got[0].detail


def test_sig_vocab_explosion_rides_typed_event(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hot_swap, "_MAX_SIGS", 1)
    router = _router()
    router.enable()
    with router.lock:
        router.warm.add(("unet", ("old",)))
    verdict, _sig = router.route("unet", lambda: None, (1,), {})

    assert verdict == hot_swap.COMPILED  # behavior: inline compile, as before
    assert router.concurrent is False
    got = _by_kind(events, activity.KIND_SERVE_DEGRADE)
    assert [e.phase for e in got] == ["sig_vocab_exceeded"]
    assert "target=unet" in got[0].detail


# ---------------------------------------------------------------------------
# capability_renewal — terminal denial and silent retry exhaustion
# ---------------------------------------------------------------------------


def _renew_loop(monkeypatch: pytest.MonkeyPatch, renew_exc: Exception) -> None:
    monkeypatch.setattr(capability_renewal, "_MIN_SLEEP_S", 0.0)
    monkeypatch.setattr(capability_renewal, "_TRANSIENT_BACKOFF_S", 0.0)
    monkeypatch.setattr(
        capability_renewal, "_renew_at", lambda token, *, now: now)

    def refuse(**k: Any) -> Any:
        raise renew_exc

    monkeypatch.setattr(capability_renewal, "renew_once", refuse)
    asyncio.run(capability_renewal.renew_capability_while_running(
        file_base_url="http://hub.invalid",
        request_id="req-1",
        attempt=2,
        get_worker_jwt=lambda: "jwt",
        get_token=lambda: "tok",
        set_token=lambda t: None,
    ))


def test_renewal_denial_rides_typed_event(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _renew_loop(
        monkeypatch, capability_renewal.RenewDenied("denied (409): fenced"))
    got = _by_kind(events, activity.KIND_CAPABILITY_RENEWAL)
    assert [e.phase for e in got] == ["denied"]
    assert "request=req-1" in got[0].detail
    assert "fenced" in got[0].detail


def test_renewal_exhaustion_rides_typed_event(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _renew_loop(monkeypatch, RuntimeError("hub 503"))
    got = _by_kind(events, activity.KIND_CAPABILITY_RENEWAL)
    assert [e.phase for e in got] == ["exhausted"]
    assert "request=req-1" in got[0].detail


# ---------------------------------------------------------------------------
# lora hygiene — a failed deactivate may bleed adapters into later requests
# ---------------------------------------------------------------------------


class _PeftlessPipe:
    """disable_lora raises: the peft-surface teardown fails."""

    def disable_lora(self) -> None:
        raise RuntimeError("PEFT backend is required")


def test_lora_deactivate_failure_rides_typed_event(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    res = lora.AdapterResidency()
    pipe = _PeftlessPipe()
    st = res._state("ref-a", pipe)
    st.attached["k"] = ("adapter-a", 123)
    st.active = True
    monkeypatch.setattr(
        lora.w8a8_lora, "branch_targets", lambda p: {})
    res.deactivate("ref-a", pipe, request_id="req-9")  # must not raise

    got = _by_kind(events, activity.KIND_LORA_HYGIENE)
    assert [e.phase for e in got] == ["deactivate_failed"]
    assert "ref=ref-a" in got[0].detail
    assert "req-9" in got[0].detail


# ---------------------------------------------------------------------------
# rotation preload — a stage failure abandons the hub's desired plan
# ---------------------------------------------------------------------------


class _ExecutorStub:
    draining = False


def test_preload_stage_failure_rides_typed_event(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = preload.Preloader(_ExecutorStub())  # type: ignore[arg-type]
    loader._generation = 7
    loader._hot = (pb.DesiredInstance(function_name="generate"),)

    async def boom(self: Any, instance: Any) -> bool:
        raise OSError("disk full")

    monkeypatch.setattr(preload.Preloader, "_stage_instance", boom)
    assert asyncio.run(loader._pass()) is False  # fail-soft: pass completes

    got = _by_kind(events, activity.KIND_ROTATION_PRELOAD)
    assert [e.phase for e in got] == ["stage_failed"]
    assert "fn=generate" in got[0].detail
    assert "generation=7" in got[0].detail
    assert "disk full" in got[0].detail


# ---------------------------------------------------------------------------
# residency — mixed-device unusable object must be named on the wire
# ---------------------------------------------------------------------------


class _FailingMoveResidency(residency.Residency):
    def __init__(self) -> None:  # skip the real constructor entirely
        pass

    def _move(self, obj: Any, device: str) -> None:
        raise RuntimeError(f"CUDA error moving to {device}")


def test_mixed_device_rollback_failure_rides_typed_event(
    events: List[Any],
) -> None:
    res = _FailingMoveResidency()
    assert res._move_verified(object(), "cpu", ref="ref-b") is False

    got = _by_kind(events, activity.KIND_RESIDENCY_FAULT)
    assert [e.phase for e in got] == ["mixed_device_unusable"]
    assert "ref=ref-b" in got[0].detail


# ---------------------------------------------------------------------------
# lane gate — silent loss of the te#79 promote-on-use protection
# ---------------------------------------------------------------------------


class _SlotsPipe:
    __slots__ = ()

    def __call__(self) -> None:  # instance __call__ exists in the MRO
        pass


def test_lane_residency_gate_wrap_failure_rides_typed_event(events: List[Any]) -> None:
    gate = lane_residency_gate.LaneResidencyGate(
        ref="ref-c", residency=object(), label="lane-c")  # type: ignore[arg-type]
    pipe = _SlotsPipe()
    # __class__ assignment onto a __dict__-bearing subclass of a __slots__
    # class fails with a layout TypeError — the wrap's failure mode.
    assert lane_residency_gate.arm_lane_residency_gate(pipe, gate) is False

    got = _by_kind(events, activity.KIND_SERVE_DEGRADE)
    assert [e.phase for e in got] == ["lane_gate_unarmed"]
    assert "ref=ref-c" in got[0].detail
