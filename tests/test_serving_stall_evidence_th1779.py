"""th#1779 — the in-call stall gate was a wall clock wearing a counter's name.

`_execute`'s docstring promises there is no wall deadline, "deliberately", and
that the abort authority is `progress.self_diagnosis()`. But the ONLY counter a
serving request opened was `infer:steps`, which `_make_ctx_emitter` advances
once per ctx event — so the number the gate read measured how CHATTY the
endpoint is, not whether it was working. An endpoint whose render is one long
silent library call froze the counter at its opening `ctx.log` and the `infer`
family's 300 s window condemned it mid-render.

Measured in production, request `790f6145-5f38-4f1a-b4b7-48836c34f3c4`
(minimax-h3 0.4.10, four attempts on pod `ptndxsdsy5ws1u`): `job.log` at
05:40:41.173 -> `request.requeued reason=worker_retryable` at 05:45:41.112;
05:45:45.161 -> 05:50:45.320; 05:50:49.154 -> 05:55:49.358. Exactly 300 s,
three times, and `timeout_ms: 2400000` on the request moved none of it.
`generate` on the SAME endpoint completed with compute_ms 126926 / 188730 /
207588 / 208320 / 212132 / 219639 / 228918 — the identical silence, just short
enough to fit.

Two defects, one incident:

1. the gate had no evidence but the endpoint's chatter -> `_HandlerEvidence`
2. `_reap_stuck_thread` watched the task the caller had just CANCELLED, so it
   read that cancellation as "thread finished" and never fired once. The
   abandoned handler kept denoising while the hub re-dispatched onto the same
   pod: reserved VRAM 59.0 -> 75.1 -> 75.8 -> 76.7 GiB across those attempts.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict

import pytest

from gen_worker import activity as activity_mod
from gen_worker import executor as executor_mod
from gen_worker import progress as progress_mod


@pytest.fixture(autouse=True)
def _clean() -> Any:
    progress_mod.reset()
    yield
    progress_mod.reset()


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> Dict[str, float]:
    t = {"t": 0.0}
    monkeypatch.setattr(progress_mod, "_now", lambda: t["t"])
    return t


OWNER = "request:790f6145-5f38-4f1a-b4b7-48836c34f3c4"


# ---------------------------------------------------------------------------
# 1. the verdict a silent-but-working render gets
# ---------------------------------------------------------------------------


def test_a_silent_but_working_handler_is_condemned_without_evidence(
    clock: Dict[str, float],
) -> None:
    """The production shape, reproduced: one ctx event, then real work.

    This is the RED assertion — it documents what the gate DID, and the fix
    below is what stops it deciding on this evidence alone."""
    steps = progress_mod.counter(
        "infer:steps", progress_mod.UNIT_STEPS, owner=OWNER)
    steps.add(1)  # the endpoint's opening ctx.log, and the last thing it says

    clock["t"] = 299.0
    assert progress_mod.self_diagnosis() is None
    clock["t"] = 300.1
    verdict = progress_mod.self_diagnosis()  # the call `_execute` makes
    assert verdict is not None and verdict.name == "infer:steps"


def test_handler_evidence_keeps_a_silent_render_alive(clock: Dict[str, float]) -> None:
    """With the request's own evidence counter open and advancing, the same
    silent render is NOT stalled — five minutes in or fifty."""
    steps = progress_mod.counter(
        "infer:steps", progress_mod.UNIT_STEPS, owner=OWNER)
    steps.add(1)

    cpu = {"v": 0.0}
    ev = executor_mod._HandlerEvidence(
        OWNER, interval_s=0.01, evidence=lambda: cpu["v"])
    with ev:
        for elapsed in (300.1, 600.1, 3000.1):
            clock["t"] = elapsed - 1.0
            cpu["v"] += 10.0  # the render burns real CPU issuing kernels
            _wait_until(lambda: _age(OWNER, "evidence:handler") == 0.0)
            clock["t"] = elapsed
            assert progress_mod.self_diagnosis() is None, (
                f"condemned a working handler at t={elapsed}")


def test_handler_evidence_still_confesses_a_wedged_process(
    clock: Dict[str, float],
) -> None:
    """The gate must keep its teeth: neither counter advancing IS a stall."""
    progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS, owner=OWNER).add(1)
    cpu = {"v": 0.0}  # a deadlocked process burns no CPU and moves no bytes
    with executor_mod._HandlerEvidence(
        OWNER, interval_s=0.01, evidence=lambda: cpu["v"]
    ):
        clock["t"] = 301.0
        verdict = progress_mod.self_diagnosis()
    assert verdict is not None
    assert verdict.age_s > verdict.window_s


def test_evidence_is_scoped_to_the_request_and_closed_after(
    clock: Dict[str, float],
) -> None:
    """pgw#894's rule, applied to the in-call loop it left registry-wide: a
    neighbour's counter can neither save nor condemn this request, and the
    counter dies with the handler."""
    other = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS, owner="request:other")
    other.add(1)
    with executor_mod._HandlerEvidence(OWNER, interval_s=0.01, evidence=lambda: 0.0):
        names = {s.name for s in progress_mod.snapshot(OWNER)}
        assert names == {"evidence:handler"}
    assert progress_mod.snapshot(OWNER) == []


def test_the_default_evidence_source_is_the_one_watchdog_already_trusts() -> None:
    ev = executor_mod._HandlerEvidence(OWNER)
    assert ev._evidence is activity_mod.default_evidence


# ---------------------------------------------------------------------------
# 2. the reaper that never fired
# ---------------------------------------------------------------------------


def _job() -> Any:
    return executor_mod._Job(request_id="rid-th1779", attempt=1, spec=None)


def test_reaper_recycles_when_the_handler_thread_outlives_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before the fix: the reaper watched `job.exec_task`, which the caller
    cancels one line before arming it, so `shield()` raised CancelledError
    immediately and the `except BaseException: pass` arm concluded the thread
    had finished. It never once fired, and the abandoned handler kept the
    card while the next attempt landed on the same pod."""
    monkeypatch.setattr(executor_mod, "_STUCK_THREAD_RECYCLE_S", 0.05)
    monkeypatch.setattr(executor_mod, "_STUCK_THREAD_POLL_S", 0.01)
    exits: list[int] = []

    async def _drive() -> None:
        ex = executor_mod.Executor.__new__(executor_mod.Executor)
        monkeypatch.setattr(ex, "_process_exit", exits.append, raising=False)
        job = _job()
        # Exactly the state `_execute` arms the reaper in: the task cancelled,
        # the THREAD still running.
        job.exec_task = asyncio.ensure_future(asyncio.sleep(60))
        job.exec_task.cancel()
        ex._reap_stuck_thread(job)
        await asyncio.sleep(0.3)

    asyncio.run(_drive())
    assert exits == [70], "the abandoned handler thread was never reaped"


def test_reaper_stands_down_when_the_thread_really_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_mod, "_STUCK_THREAD_RECYCLE_S", 0.2)
    monkeypatch.setattr(executor_mod, "_STUCK_THREAD_POLL_S", 0.01)
    exits: list[int] = []

    async def _drive() -> None:
        ex = executor_mod.Executor.__new__(executor_mod.Executor)
        monkeypatch.setattr(ex, "_process_exit", exits.append, raising=False)
        job = _job()
        job.exec_task = asyncio.ensure_future(asyncio.sleep(60))
        job.exec_task.cancel()
        job.handler_thread_done.set()  # what `_call_sync`'s finally does
        ex._reap_stuck_thread(job)
        await asyncio.sleep(0.4)

    asyncio.run(_drive())
    assert exits == []


def test_call_sync_reports_the_thread_ending_on_both_paths() -> None:
    job = _job()
    executor_mod.Executor._call_sync(job, lambda: "out", {}, 0)
    assert job.handler_thread_done.is_set()

    job2 = _job()

    def _boom() -> None:
        raise RuntimeError("handler exploded")

    with pytest.raises(RuntimeError):
        executor_mod.Executor._call_sync(job2, _boom, {}, 0)
    assert job2.handler_thread_done.is_set()


# ---------------------------------------------------------------------------


def _age(owner: str, name: str) -> float:
    for s in progress_mod.snapshot(owner):
        if s.name == name:
            return s.age_s
    return float("inf")


def _wait_until(pred: Any, timeout_s: float = 2.0) -> None:
    deadline = threading.Event()
    waited = 0.0
    while waited < timeout_s:
        if pred():
            return
        deadline.wait(0.01)
        waited += 0.01
    raise AssertionError("condition never held")
