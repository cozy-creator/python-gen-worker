"""A cancel that never unwinds must never absorb the next assignment.

The failure shape is ABSENCE: a worker whose cancelled job never unwound stays
connected, heartbeating and advertising, while the next assignment produces zero
events of any kind.

Mechanism, reproduced here over the real executor: cancelling a SYNC handler
is cooperative (``ctx._cancel()`` sets a flag; ``exec_task.cancel()`` is only
sent for async handlers, and ``asyncio.to_thread`` cannot be cancelled
anyway). A handler that does not poll the flag keeps running, so ``_run_job``
never returns, its GPU permit and instance gate are never released, and the
next job parks in ``_gpu_semaphore.acquire()`` — a wait that emits nothing.

The tests below assert the fix's contract: the next assignment either RUNS
(when the unwind works) or is REFUSED (when it does not) — never silently
absorbed. ``test_wedge_shape_...`` is the red-verification, kept permanently:
with the guard's grace pushed out of reach it reproduces the pre-fix silence.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import msgspec
import pytest

from gen_worker import executor as executor_mod
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness import stubborn_endpoints_pgw687 as wedge
from harness.hub_double import hub_double, is_accept_for, is_ready, is_result_for
from harness.progress_wait import Cadence, await_progress

MODULE = "harness.stubborn_endpoints_pgw687"
CUDA = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)


def _payload() -> bytes:
    return msgspec.msgpack.encode(wedge.WedgeIn(text="marco"))


def _send(conn: Any, rid: str, fn: str) -> None:
    conn.send(run_job=pb.RunJob(
        request_id=rid, attempt=1, function_name=fn,
        input_payload=_payload(), compute=CUDA))


def _result(conn: Any, rid: str, timeout: float = 10.0) -> Optional[Any]:
    try:
        return conn.wait_for(is_result_for(rid), timeout=timeout).job_result
    except TimeoutError:
        return None


@pytest.fixture(autouse=True)
def _wedge_state() -> Any:
    wedge.reset()
    grace = executor_mod._CANCEL_UNWIND_GRACE_S
    recycle = executor_mod._CANCEL_UNWIND_RECYCLE_S
    yield
    # Backstop only. A wedged handler holds the executor, so the worker cannot
    # drain — and ``hub_double``'s exit runs ``harness.stop()``, whose thread
    # join is a 15s bound that is then SILENTLY IGNORED. This fixture's teardown
    # runs after that exit, so releasing here alone bought nothing: pgw#796
    # measured the four wedge tapes at 15.5-19.0s each, ~60s of the suite, all
    # of it that ignored join. Every tape now sets STUBBORN_RELEASE as its last
    # statement INSIDE the ``with``, once its assertions have all run.
    wedge.STUBBORN_RELEASE.set()
    executor_mod._CANCEL_UNWIND_GRACE_S = grace
    executor_mod._CANCEL_UNWIND_RECYCLE_S = recycle


def _fn_unavailable(conn: Any) -> List[Any]:
    with conn._recv_cond:
        return [
            m.fn_unavailable for m in conn.received
            if m.WhichOneof("msg") == "fn_unavailable"
        ]


# ---------------------------------------------------------------------------
# Red verification: the pre-fix behaviour, with the guard put out of reach.
# ---------------------------------------------------------------------------


def test_wedge_shape_without_the_guard_is_silent_absorption() -> None:
    executor_mod._CANCEL_UNWIND_GRACE_S = 1e9  # guard effectively absent
    with hub_double(modules=(MODULE,)) as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _send(conn, "r-wedge-a", "stubborn")
        assert wedge.STUBBORN_RUNNING.wait(timeout=10.0)
        conn.send(cancel_job=pb.CancelJob(request_id="r-wedge-a", attempt=1))
        _send(conn, "r-wedge-b", "polite")
        conn.wait_for(is_accept_for("r-wedge-b"))
        # THE DEFECT: accepted, and then nothing at all.
        assert _result(conn, "r-wedge-a", timeout=2.0) is None, (
            "the cancel cannot have landed — the handler ignores it")
        assert _result(conn, "r-wedge-b", timeout=2.0) is None, (
            "pre-fix shape: job B is absorbed with no result and no event")
        assert "polite" not in wedge.CALLS, "job B never reached its handler"
        wedge.STUBBORN_RELEASE.set()  # see _wedge_state: release INSIDE the block


# ---------------------------------------------------------------------------
# The fix: refuse loudly, recover when the unwind lands late.
# ---------------------------------------------------------------------------


def test_stuck_cancel_refuses_the_parked_next_job() -> None:
    executor_mod._CANCEL_UNWIND_GRACE_S = 0.5
    with hub_double(modules=(MODULE,)) as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _send(conn, "r-park-a", "stubborn")
        assert wedge.STUBBORN_RUNNING.wait(timeout=10.0)
        conn.send(cancel_job=pb.CancelJob(request_id="r-park-a", attempt=1))
        # Assigned while the wedged handler still holds the permit — the
        # incident's 61 s reassignment.
        _send(conn, "r-park-b", "polite")
        res = _result(conn, "r-park-b", timeout=10.0)
        assert res is not None, "the parked job must not be absorbed silently"
        assert res.status == pb.JOB_STATUS_RETRYABLE
        assert "cancel_unwind_stuck" in res.safe_message
        assert "polite" not in wedge.CALLS
        wedge.STUBBORN_RELEASE.set()  # see _wedge_state: release INSIDE the block


def test_quarantine_stops_advertising_and_refuses_new_assignments() -> None:
    executor_mod._CANCEL_UNWIND_GRACE_S = 0.5
    with hub_double(modules=(MODULE,)) as (scheduler, harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _send(conn, "r-quar-a", "stubborn")
        assert wedge.STUBBORN_RUNNING.wait(timeout=10.0)
        conn.send(cancel_job=pb.CancelJob(request_id="r-quar-a", attempt=1))
        # Positive signal on the wire, not merely an empty function set.
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "fn_unavailable"
            and m.fn_unavailable.reason == "cancel_unwind_stuck",
            timeout=10.0,
        )
        assert harness.worker.executor.available_functions() == []
        # An assignment arriving AFTER the quarantine is refused at admission.
        _send(conn, "r-quar-c", "polite")
        res = _result(conn, "r-quar-c", timeout=10.0)
        assert res is not None and res.status == pb.JOB_STATUS_RETRYABLE
        assert "cancel_unwind_stuck" in res.safe_message
        wedge.STUBBORN_RELEASE.set()  # see _wedge_state: release INSIDE the block


def test_late_unwind_re_advertises_and_the_next_job_runs() -> None:
    executor_mod._CANCEL_UNWIND_GRACE_S = 0.5
    with hub_double(modules=(MODULE,)) as (scheduler, harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _send(conn, "r-late-a", "stubborn")
        assert wedge.STUBBORN_RUNNING.wait(timeout=10.0)
        conn.send(cancel_job=pb.CancelJob(request_id="r-late-a", attempt=1))
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "fn_unavailable"
            and m.fn_unavailable.reason == "cancel_unwind_stuck",
            timeout=10.0,
        )
        wedge.STUBBORN_RELEASE.set()  # the unwind lands, late
        res_a = _result(conn, "r-late-a", timeout=10.0)
        assert res_a is not None and res_a.status == pb.JOB_STATUS_CANCELED
        # pgw#1349: a 10 s expiry here only decided WHEN the assert below ran,
        # so a loaded runner turned "the unwind has not landed yet" into "the
        # unwind never re-advertised". Wait on the advertisement.
        await_progress(
            harness.worker.executor.available_functions,
            bool,
            what="the landed unwind to re-advertise the functions we took",
            cadence=Cadence(),
        )
        _send(conn, "r-late-d", "polite")
        res_d = _result(conn, "r-late-d", timeout=10.0)
        assert res_d is not None and res_d.status == pb.JOB_STATUS_OK


def test_thread_that_never_unwinds_recycles_the_process() -> None:
    executor_mod._CANCEL_UNWIND_GRACE_S = 0.3
    executor_mod._CANCEL_UNWIND_RECYCLE_S = 0.5
    exits: List[int] = []
    with hub_double(modules=(MODULE,)) as (scheduler, harness):
        harness.worker.executor._process_exit = exits.append
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _send(conn, "r-recycle-a", "stubborn")
        assert wedge.STUBBORN_RUNNING.wait(timeout=10.0)
        conn.send(cancel_job=pb.CancelJob(request_id="r-recycle-a", attempt=1))
        # pgw#1349, same shape: the process-exit call is what is being waited
        # for, and the assert below still fails on a WRONG exit code.
        await_progress(
            lambda: list(exits), bool,
            what="the recycle exit for a thread that never honours a cancel",
            cadence=Cadence(),
        )
        assert exits == [70], (
            "a handler thread that never honours a cancel cannot be reclaimed "
            "in-process — the pod must be replaced")
        wedge.STUBBORN_RELEASE.set()  # see _wedge_state: release INSIDE the block


# ---------------------------------------------------------------------------
# The unwind-works row: a cancellable handler must NOT be quarantined.
# ---------------------------------------------------------------------------


def test_cancellable_handler_unwinds_and_the_next_job_runs() -> None:
    executor_mod._CANCEL_UNWIND_GRACE_S = 0.5
    with hub_double(modules=(MODULE,)) as (scheduler, harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        _send(conn, "r-ok-a", "patient")
        assert wedge.STUBBORN_RUNNING.wait(timeout=10.0)
        conn.send(cancel_job=pb.CancelJob(request_id="r-ok-a", attempt=1))
        res_a = _result(conn, "r-ok-a", timeout=10.0)
        assert res_a is not None and res_a.status == pb.JOB_STATUS_CANCELED
        _send(conn, "r-ok-b", "polite")
        res_b = _result(conn, "r-ok-b", timeout=10.0)
        assert res_b is not None and res_b.status == pb.JOB_STATUS_OK, (
            "a clean cancel leaves the worker able to run the next job")
        assert "polite" in wedge.CALLS
        time.sleep(1.0)  # past the grace: nothing may quarantine retroactively
        assert not [
            fu for fu in _fn_unavailable(conn)
            if fu.reason == "cancel_unwind_stuck"
        ], "a cancel that unwound must never mark the worker unfit"
        assert harness.worker.executor.available_functions()
