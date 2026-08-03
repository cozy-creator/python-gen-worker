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
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker import activity, capability_renewal, hot_swap, preload, trt_engine
from gen_worker.compile_cache import AdoptError
from gen_worker.models import lane_gate, residency
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.utils import lora


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


def _by_kind(events: List[Any], kind: str) -> List[Any]:
    return [e for e in events if e.kind == kind]


# ---------------------------------------------------------------------------
# trt_engine.enable — the pgw#733 pattern, TRT half
# ---------------------------------------------------------------------------


class _Cfg:
    family = "sdxl"


def test_trt_enable_refusal_names_the_classified_reason(
    events: List[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """pgw#923: the reason is RETURNED, not narrated.

    It used to ride a free-text `trt_adopt` event — a third spelling of "a cell
    adopted and what it cost", next to `aot_adopt` and the measured
    `compile_cache_adopt`. `trt_adopt` is deleted; the classified reason now
    reaches the hub as this adoption's own `adopt_failed:<reason>`, which is
    countable in the same query as every other adoption outcome.
    """
    def refuse(*a: Any, **k: Any) -> Dict[str, Any]:
        raise AdoptError("no_target", "pipeline has no module 'unet'")

    monkeypatch.setattr(trt_engine, "load_and_wrap", refuse)
    artifact = tmp_path / "engine.tar"
    artifact.write_bytes(b"not-a-real-artifact")
    out = trt_engine.enable(object(), _Cfg(), artifact=artifact)
    assert out.armed is False
    assert out.reason == "no_target"  # the CLASSIFIED reason, not a kind
    assert "engine.tar" in out.detail
    assert "no module 'unet'" in out.detail
    # And no second vocabulary was written on the way past.
    assert not [e for e in events if e.kind == "trt_adopt"]


def test_trt_enable_success_names_the_engine(
    events: List[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    meta = {"module": "unet", "sku": "l4", "trt": "10.8", "precision": "fp8"}
    monkeypatch.setattr(trt_engine, "load_and_wrap", lambda *a, **k: meta)
    artifact = tmp_path / "engine.tar"
    artifact.write_bytes(b"x")
    out = trt_engine.enable(object(), _Cfg(), artifact=artifact)
    assert out.armed is True
    assert "module=unet" in out.identity
    assert not [e for e in events if e.kind == "trt_adopt"]


# ---------------------------------------------------------------------------
# hot_swap — background warm/heal compile failure
# ---------------------------------------------------------------------------


def test_warm_compile_failure_rides_typed_event(events: List[Any]) -> None:
    router = hot_swap.Router()
    router.enable()
    sig = ("unet", (("T", (1, 4), "torch.bfloat16", "cpu"),))
    with router.lock:
        router.pending.add(sig)

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("inductor exploded")

    job = hot_swap._WarmJob(
        router=router, label="unet", sig=sig, compiled=boom,
        args=(), kwargs={}, device=None, grad_mode="grad",
        autocast_dtype=None,
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
    router = hot_swap.Router()
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


def test_lane_gate_wrap_failure_rides_typed_event(events: List[Any]) -> None:
    gate = lane_gate.LaneGate(
        ref="ref-c", residency=object(), label="lane-c")  # type: ignore[arg-type]
    pipe = _SlotsPipe()
    # __class__ assignment onto a __dict__-bearing subclass of a __slots__
    # class fails with a layout TypeError — the wrap's failure mode.
    assert lane_gate.arm_lane_gate(pipe, gate) is False

    got = _by_kind(events, activity.KIND_SERVE_DEGRADE)
    assert [e.phase for e in got] == ["lane_gate_unarmed"]
    assert "ref=ref-c" in got[0].detail
