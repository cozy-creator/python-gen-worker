"""pgw#714: background-compile crashes tell the truth and degrade to eager.

A signal death with a `compile` inflight marker attributes the streak to the
compile marker, never to the tenant request in flight (the th#1226
misattribution that refused fn=generate and condemned H200/B200 for a
software race). A prior compile-death disables process compiles (serve
eager, never re-crash). An operator-pinned `+eager` lane suppresses compile
arming entirely — the real kill switch.
"""

from __future__ import annotations

import json
import threading

import pytest

from gen_worker import compile_cache as cc
from gen_worker import hot_swap, postmortem


@pytest.fixture()
def tmp_postmortem(tmp_path, monkeypatch):
    inflight = tmp_path / "inflight.json"
    registry = tmp_path / "crash_registry.json"
    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", inflight)
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)
    postmortem.clear_inflight(path=inflight)
    with postmortem._inflight_lock:
        postmortem._inflight_active.clear()
    yield inflight, registry
    with postmortem._inflight_lock:
        postmortem._inflight_active.clear()


def test_compile_marker_takes_the_blame(tmp_postmortem):
    inflight, registry = tmp_postmortem
    postmortem.note_inflight(
        "request", "generate", request_id="req-1", path=inflight)
    postmortem.note_inflight(
        postmortem.COMPILE_KIND, postmortem.compile_marker("transformer"),
        path=inflight)

    extra = postmortem.attribute_signal_death(
        signal_name="SIGSEGV", inflight_path=inflight,
        registry_path=registry, dump_path=inflight.with_name("no.dump"))

    streaks = extra["native_crash_streaks"]
    assert streaks == {"compile:transformer": 1}
    rows = postmortem.native_crash_streaks(registry)
    assert "generate" not in rows
    assert rows["compile:transformer"]["last_kind"] == "compile"
    # Both executions still appear as evidence.
    fns = {r["function"] for r in extra["inflight"]}
    assert fns == {"generate", "compile:transformer"}


def test_no_compile_marker_keeps_request_attribution(tmp_postmortem):
    inflight, registry = tmp_postmortem
    postmortem.note_inflight(
        "request", "generate", request_id="req-1", path=inflight)
    extra = postmortem.attribute_signal_death(
        signal_name="SIGSEGV", inflight_path=inflight,
        registry_path=registry, dump_path=inflight.with_name("no.dump"))
    assert extra["native_crash_streaks"] == {"generate": 1}


def test_compile_crash_rows_selects_only_compiles(tmp_postmortem):
    _, registry = tmp_postmortem
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV", path=registry)
    postmortem.record_native_crash(
        postmortem.compile_marker("transformer"),
        kind=postmortem.COMPILE_KIND, signal_name="SIGSEGV", path=registry)
    rows = postmortem.compile_crash_rows(registry)
    assert set(rows) == {"compile:transformer"}


def test_warm_thread_marks_and_clears_the_compile_inflight(tmp_postmortem):
    inflight, registry = tmp_postmortem
    seen: dict = {}

    def compiled() -> None:
        seen["active"] = json.loads(inflight.read_text())["active"]
        raise RuntimeError("boom (contained per-signature)")

    router = hot_swap.Router()
    job = hot_swap._WarmJob(
        router=router, label="unet", sig=("s",), compiled=compiled,
        args=(), kwargs={}, device=None, grad_mode="no_grad",
        autocast_dtype=None, turn=None)
    router.pending.add(job.sig)
    hot_swap._run_warm(job)

    assert [r["function"] for r in seen["active"]] == ["compile:unet"]
    assert seen["active"][0]["kind"] == postmortem.COMPILE_KIND
    # Failure contained; marker cleared; signature stays eager.
    assert postmortem.take_inflight(inflight) == []
    assert job.sig in router.bg_failed


@pytest.fixture()
def reset_compile_disable():
    yield
    cc._PROCESS_COMPILES_DISABLED = ""


def test_process_disable_makes_apply_a_noop(reset_compile_disable):
    cc.disable_process_compiles("1 process signal death(s) during compile")
    assert cc._PROCESS_COMPILES_DISABLED

    class Pipe:  # never armed: apply must return before touching torch
        pass

    assert cc.apply(Pipe(), None, cache_ready=True) is False


def test_operator_eager_pin_suppresses_arming(reset_compile_disable):
    class Pipe:
        pass

    pinned = Pipe()
    setattr(pinned, cc.EXECUTION_LANE_ATTR, "bf16-w16a16+eager")
    setattr(pinned, cc.EXECUTION_LANE_PINNED_ATTR, True)
    assert cc.operator_eager_pin(pinned) is True
    assert cc.apply(pinned, None, cache_ready=True) is False

    # The same lane WITHOUT pin provenance is not a kill switch.
    auto = Pipe()
    setattr(auto, cc.EXECUTION_LANE_ATTR, "bf16-w16a16+eager")
    assert cc.operator_eager_pin(auto) is False

    # A pinned COMPILED lane never suppresses.
    comp = Pipe()
    setattr(comp, cc.EXECUTION_LANE_ATTR, "fp8-w8a8-dynamic+compiled")
    setattr(comp, cc.EXECUTION_LANE_PINNED_ATTR, True)
    assert cc.operator_eager_pin(comp) is False


def test_setup_window_carries_pin_provenance(reset_compile_disable):
    class Pipe:
        pass

    execution_lane_tok = cc._SETUP_EXEC_EXECUTION_LANE.set("bf16-w16a16+eager")
    pin_tok = cc._SETUP_EXEC_EXECUTION_LANE_PINNED.set(True)
    try:
        pipe = Pipe()
        assert cc.operator_eager_pin(pipe) is True
        # Stamped through, like the lane itself.
        assert getattr(pipe, cc.EXECUTION_LANE_PINNED_ATTR) is True
    finally:
        cc._SETUP_EXEC_EXECUTION_LANE_PINNED.reset(pin_tok)
        cc._SETUP_EXEC_EXECUTION_LANE.reset(execution_lane_tok)
