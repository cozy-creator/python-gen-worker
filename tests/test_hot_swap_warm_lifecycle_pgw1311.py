"""pgw#1311 — a background warm compile has an OWNER, and outlives nothing.

`hot_swap` runs every warm/heal compile on one process-global daemon thread.
The thread is deliberately immortal; what has a lifetime is the JOB. Until
this issue nothing said so: a mint abandonment deleted the capture directory
its own in-flight warm compile was writing into, and the resulting
`FileNotFoundError` reached the hub as a `serve_degrade` plus a permanent
`shape_gap` — a coverage hole invented by our own cleanup — while the
signature was stranded in `bg_failed` (eager for the life of the process).

In the suite the same leak reads as non-determinism: the envelope lands in
the module-level activity sink DURING a later test (measured:
`test_mint_abort_classification_th1299` failing an assertion inside
`test_retry_activity_gw661`, on another xdist worker, victim varying run to
run).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path
from typing import Any, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker import hot_swap
from gen_worker.pb import worker_scheduler_pb2 as pb

from test_eager_first_boot_pgw671 import _Harness


def _turn(kind: str) -> Any:
    return contextlib.nullcontext()


def _router() -> hot_swap.Router:
    router = hot_swap.Router()
    router.set_turn_gate(_turn)
    router.enable()
    return router


@pytest.fixture
def emitted(monkeypatch: pytest.MonkeyPatch) -> List[pb.ActivityUpdate]:
    """Everything the worker would have told the hub."""
    seen: List[pb.ActivityUpdate] = []
    lock = threading.Lock()

    def sink(update: pb.ActivityUpdate) -> None:
        with lock:
            seen.append(update)

    monkeypatch.setattr(activity_mod, "_sink", sink)
    return seen


class _BlockingCompile:
    """A compiled callable that parks inside the warm thread until released,
    then fails the way a deleted capture directory fails."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.entered.set()
        assert self.release.wait(30), "warm job was never released"
        raise FileNotFoundError(
            "capture/inductor/fxgraph/g0.bin: the owner deleted this")


def test_a_disowned_warm_failure_is_not_a_shape_gap(
    emitted: List[pb.ActivityUpdate],
) -> None:
    """The seam: once the owner cancels, the compile's failure is OUR
    teardown — it must not reach the hub at all, and must not strand the
    signature eager."""
    router = _router()
    compiled = _BlockingCompile()

    verdict, sig = router.route("transformer", compiled, (1,), {})
    assert verdict == hot_swap.EAGER
    assert compiled.entered.wait(30), "the warm job never started"

    router.cancel_warm()
    compiled.release.set()
    assert hot_swap.quiesce(timeout=30.0, cancel=False) == 0, (
        "the compile never ended")

    assert emitted == [], (
        "a cancelled warm job reported its own cancellation to the hub as a "
        f"fleet fact: {[(e.kind, e.phase) for e in emitted]}"
    )
    assert sig not in router.bg_failed, (
        "the signature was condemned to eager for a failure the owner caused"
    )
    assert sig not in router.warm and sig not in router.pending


def test_a_live_warm_failure_is_still_reported(
    emitted: List[pb.ActivityUpdate],
) -> None:
    """The other arm — without a cancellation the SAME failure is a real
    degrade and still goes out. Silence is a decision, not the default."""
    router = _router()
    compiled = _BlockingCompile()

    _verdict, sig = router.route("transformer", compiled, (1,), {})
    assert compiled.entered.wait(30)
    compiled.release.set()
    assert hot_swap.quiesce(timeout=30.0, cancel=False) == 0

    phases = {e.phase for e in emitted}
    assert "warm_compile_failed" in phases, (
        f"a genuine background compile failure went unreported: {phases}")
    assert sig in router.bg_failed


def test_quiesce_is_what_keeps_a_warm_job_inside_its_test() -> None:
    """The leak fence itself, red arm first: while a warm job is compiling it
    IS outstanding, and only the cancel-plus-wait closes it. The autouse
    `_no_warm_job_outlives_its_test` fixture runs exactly this after every
    test in the suite."""
    router = _router()
    compiled = _BlockingCompile()
    router.route("transformer", compiled, (2,), {})
    assert compiled.entered.wait(30)

    # RED: the leak as it stands — a job still running with nobody waiting.
    assert hot_swap.quiesce(timeout=0.0, cancel=False) == 1

    # GREEN: cancel disowns it, the bounded wait observes it end.
    compiled.release.set()
    assert hot_swap.quiesce(timeout=30.0) == 0


def test_abandoning_a_mint_disowns_its_in_flight_warm_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production path, through the real executor: abandonment deletes
    the capture directory, so the warm compile that was writing into it fails.
    Nothing about that is a fleet fact."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=2.0)

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is not None
        await h.ex.abandon_background_mint(
            h.rec, reason="instance vacate", code="vacate")
        # Let the disowned compile run to its own end and try to report,
        # with the transport loop still alive to carry it if it does.
        await asyncio.to_thread(
            lambda: hot_swap.quiesce(timeout=30.0, cancel=False))
        for _ in range(50):
            await asyncio.sleep(0.01)

    asyncio.run(_run())

    leaked = [
        m.activity_update for m in h.sent
        if m.WhichOneof("msg") == "activity_update"
        and (m.activity_update.kind == "shape_gap"
             or m.activity_update.phase == "warm_compile_failed")
    ]
    assert not leaked, (
        "the abandonment's own rmtree came back as a coverage gap: "
        f"{[(e.kind, e.phase, e.detail[:120]) for e in leaked]}"
    )
