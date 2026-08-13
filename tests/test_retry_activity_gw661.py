"""A will-retry setup condition must not present to the hub as a failure.

The hub's ``lastWorkerProgressLocked`` *excludes* failed activities from
progress evidence, so reporting ACTIVITY_STATE_FAILED for a retryable loss (a
lane that cannot promote to VRAM yet, say) erases exactly the evidence that
keeps a working pod alive — the hub condemns pods whose compiles then complete.

So: retryable losses report RUNNING with a ``retrying`` detail, exhaustion
reports FAILED and disables the function (the hub must still see the terminal
truth — the genuinely-unfittable VRAM lane depends on it), and a non-retryable
failure reports FAILED on the first attempt.

These drive the real ``Executor.ensure_setup`` path with the real activity
sink, not a stub reporter.
"""

from __future__ import annotations

import asyncio
from typing import List

import msgspec
import pytest

from gen_worker import activity
from gen_worker import Resources, endpoint
from gen_worker.api.errors import RetryableError
from gen_worker.executor import MAX_TRANSIENT_SETUP_ATTEMPTS, Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs


class _In(msgspec.Struct):
    prompt: str = "x"


class _Out(msgspec.Struct):
    y: str


@pytest.fixture(autouse=True)
def _reset_activity_sink():
    yield
    with activity._lock:
        activity._sink = None
        activity._current = None
        activity._last_progress_heartbeat = 0.0


def _updates(sent: List[pb.WorkerMessage]) -> List[pb.ActivityUpdate]:
    return [m.activity_update for m in sent if m.WhichOneof("msg") == "activity_update"]


def _executor(setup_fn):
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    @endpoint(resources=Resources(vram_gb_hint=8))
    class Ep:
        def setup(self) -> None:
            setup_fn()

        def generate(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

    specs = extract_specs(Ep)
    return Executor(specs, _send), specs, sent


async def _attempt(ex, spec, expected_exc) -> None:
    if expected_exc is None:
        await ex.ensure_setup(spec)
    else:
        with pytest.raises(expected_exc):
            await ex.ensure_setup(spec)
    for _ in range(10):  # the sink schedules sends onto this loop
        await asyncio.sleep(0)


def test_retryable_setup_loss_reports_running_not_failed():
    """The lane-gate case: the attempt lost, the work will be re-attempted,
    so the hub must still see a RUNNING activity — its only progress
    evidence — and the function must stay servable."""
    outcomes = [RetryableError("cannot promote to VRAM (waited 45s); retrying"), None]

    def _setup() -> None:
        exc = outcomes.pop(0)
        if exc is not None:
            raise exc

    ex, specs, sent = _executor(_setup)

    async def _go() -> None:
        await _attempt(ex, specs[0], RetryableError)

        ups = _updates(sent)
        assert ups, "no activity envelopes emitted"
        last = ups[-1]
        assert last.state == pb.ActivityState.ACTIVITY_STATE_RUNNING, (
            "a will-retry condition reported the FAILED terminal — the hub "
            "drops failed activities from its progress evidence and condemns "
            "the pod (gw#661)"
        )
        assert "retrying (attempt 1/" in last.detail
        assert "RetryableError" in last.detail
        assert not last.error, "a RUNNING update must not carry the error rung"
        # Not disabled: the hub may still dispatch, and the retry may win.
        assert specs[0].name not in ex.unavailable

        # The retry wins, and the record's patience is restored for next time.
        await _attempt(ex, specs[0], None)
        assert _updates(sent)[-1].state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
        assert ex._class_record(specs[0]).transient_setup_failures == 0

    asyncio.run(_go())


def test_retry_exhaustion_reports_failed_and_disables_the_function():
    """Exhaustion is the terminal truth and the hub must see it: FAILED on
    the activity carrier plus a typed per-function unavailability. Without
    this, 'retryable' would mean an infinite entitlement and th#1159's
    genuinely-unfittable VRAM lane could never reach a verdict."""
    def _setup() -> None:
        raise RetryableError("cannot promote to VRAM (waited 45s); retrying")

    ex, specs, sent = _executor(_setup)

    async def _go() -> None:
        for attempt in range(1, MAX_TRANSIENT_SETUP_ATTEMPTS + 1):
            await _attempt(ex, specs[0], RetryableError)
            last = _updates(sent)[-1]
            if attempt < MAX_TRANSIENT_SETUP_ATTEMPTS:
                assert last.state == pb.ActivityState.ACTIVITY_STATE_RUNNING, (
                    f"attempt {attempt} of {MAX_TRANSIENT_SETUP_ATTEMPTS} "
                    "reported terminal before the budget was spent"
                )
                assert specs[0].name not in ex.unavailable
            else:
                assert last.state == pb.ActivityState.ACTIVITY_STATE_FAILED, (
                    "the budget was spent and the worker still told the hub "
                    "it was working — the terminal truth never arrives"
                )
                assert "RetryableError" in last.error

        reason, detail, _axes = ex.unavailable[specs[0].name]
        assert reason == "retry_exhausted"
        assert "RetryableError" in detail

    asyncio.run(_go())


def test_non_retryable_setup_failure_still_reports_failed_immediately():
    """No budget for a failure that carries no retry contract: first
    attempt, FAILED, function disabled."""
    def _setup() -> None:
        raise RuntimeError("induced setup crash")

    ex, specs, sent = _executor(_setup)

    async def _go() -> None:
        await _attempt(ex, specs[0], RuntimeError)

        last = _updates(sent)[-1]
        assert last.state == pb.ActivityState.ACTIVITY_STATE_FAILED
        assert "RuntimeError: induced setup crash" in last.error
        assert not last.detail.startswith("retrying")
        assert ex.unavailable[specs[0].name][0] == "setup_failed"

    asyncio.run(_go())
