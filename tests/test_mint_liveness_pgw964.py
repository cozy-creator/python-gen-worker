"""Finishing an entry is not dying.

The mint stall monitor must not sample its child's process-tree CPU by summing
``user + system`` over the LIVE members of the tree. On Linux a process's CPU
leaves its own ``utime/stime`` and enters its parent's ``cutime/cstime`` the
instant the parent reaps it, so that sum goes DOWN by a whole child's lifetime
every time a child finishes — and ``_observe`` compares against a high-water
mark. An AOT mint whose compile pool completes one ~390-second entry then falls
into a hole one entry deep and is SIGTERMed for making progress, while the pool
ledger it is writing records full concurrency and zero idle slots.

Both tapes below run REAL processes and a REAL grandchild that burns REAL CPU
and then exits, because the property under test is exactly what ``/proc`` says
after a ``wait()`` — which no mock can tell you. No wait here is a clock: they
key on markers the children write and on measured advances.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from gen_worker.mint_process import _evidence, _observe, _tree_cpu_seconds
from gen_worker.stall import SilenceWindow
from harness.progress_wait import Cadence, await_progress

# One grandchild burn, sized in ITERATIONS rather than seconds so a loaded
# runner makes the tape slower, never wrong. ~1-3 CPU-seconds here.
_BURN_ITERS = 24_000_000

# CPU-seconds of grandchild work the tape insists on OBSERVING before it lets
# the reap happen. A threshold on measured work, not on the clock.
_LIVE_CPU_FLOOR_S = 0.4

_WORKER = '''
import os, subprocess, sys

state, iters = sys.argv[1], sys.argv[2]

# The entry compile child: burns hard, then EXITS. Its exit is the event the
# old monitor read as death.
g = subprocess.Popen([
    sys.executable, "-c",
    "import sys\\nx = 0\\nfor _ in range(int(sys.argv[1])): x = (x + 1) & 65535\\n",
    iters])
g.wait()   # reaped -> the grandchild's CPU moves into OUR cutime/cstime
open(os.path.join(state, "reaped"), "w").write("1")

# The replacement entry the pool spawns straight after. Still working — and
# under the old sampler still invisible, because the tree total is sitting in
# a hole as deep as the grandchild that just finished.
x = 0
while True:
    x = (x + 1) & 65535
'''


def _spawn_tree(tmp_path: Path) -> subprocess.Popen:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    return subprocess.Popen(
        [sys.executable, str(script), str(state), str(_BURN_ITERS)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)


def _reap_tree(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(Exception):
        proc.wait()


def _gone(proc: subprocess.Popen) -> Optional[str]:
    return None if proc.poll() is None else f"the child exited {proc.poll()}"


def test_tree_cpu_survives_a_reaped_grandchild(tmp_path: Path) -> None:
    """The measured total must never go backwards when a grandchild finishes.

    RED on the pre-pgw#964 sampler: the post-reap sample loses the whole burn.
    """
    proc = _spawn_tree(tmp_path)
    reaped = tmp_path / "state" / "reaped"
    try:
        # 1. A live grandchild's CPU must be visible. (This half always
        #    worked; asserting it keeps the fix honest — the job is to ADD the
        #    reaped counters, not to trade one blindness for another.)
        live = await_progress(
            lambda: _tree_cpu_seconds(proc.pid) or 0.0,
            lambda cpu: cpu >= _LIVE_CPU_FLOOR_S,
            what="a live grandchild's CPU to reach the floor",
            cadence=Cadence(floor_s=60.0),
            gone=lambda: _gone(proc),
            render=lambda cpu: f"{cpu:.2f} CPU-s of {_LIVE_CPU_FLOOR_S}",
        )

        # 2. Let it finish and be waited for.
        await_progress(
            reaped.exists,
            bool,
            what="the grandchild to be reaped",
            cadence=Cadence(floor_s=60.0),
            gone=lambda: _gone(proc),
        )

        after = _tree_cpu_seconds(proc.pid)
        assert after is not None, "a live tree must be samplable"
        assert after >= live, (
            "process-tree CPU went BACKWARDS when a grandchild was reaped: "
            f"{live:.2f}s while it ran -> {after:.2f}s after wait(). The "
            "reaped child's CPU is in its parent's cutime/cstime and the "
            "sampler is dropping it — which is what SIGTERM'd pgw#868 "
            "attempt eighteen.")
    finally:
        _reap_tree(proc)


class _Ticks:
    """A clock the TAPE drives, so the window is exercised with no wall time.

    Every read costs ``step`` seconds, so ``SilenceWindow(300)`` gives up after
    30 consecutive non-advancing polls — and never after a single slow one.
    """

    def __init__(self, step: float = 10.0) -> None:
        self.step = float(step)
        self.t = 0.0

    def now(self) -> float:
        self.t += self.step
        return self.t


async def _await(
    settled: Callable[[], bool],
    advance: Callable[[], Any],
    watch: "asyncio.Future[str]",
    proc: subprocess.Popen,
    *,
    what: str,
) -> None:
    """Wait for ``settled()``; give up only on a measured stall or a death.

    ``watch`` finishing is a DEFINITIVE end, not a stall: the monitor issued
    its kill verdict, so there is nothing left to wait for and the caller's
    assertion gets to report it verbatim. No duration bounds this.
    """
    cadence = Cadence(floor_s=60.0)
    window = SilenceWindow(cadence.window_s)
    mark, last = advance(), time.monotonic()
    while not settled():
        if watch.done() or proc.poll() is not None:
            return
        await asyncio.sleep(0.02)
        now, fresh = time.monotonic(), advance()
        if fresh != mark:
            cadence.record(now - last)
            mark, last = fresh, now
            window.window_s = cadence.window_s
            window.touch()
            continue
        window.window_s = cadence.window_s
        if window.stalled():
            raise AssertionError(
                f"waiting for {what}: nothing advanced for "
                f"{window.silent_for():.1f}s ({cadence.describe()}); "
                f"last saw {mark!r}")


def test_observe_does_not_kill_a_pool_that_finished_an_entry(
    tmp_path: Path,
) -> None:
    """The real ``_observe`` against a real tree that reaps a real grandchild.

    RED on the pre-pgw#964 monitor: ``_observe`` returns its stall reason and
    the tape reports the false kill verbatim.
    """
    proc = _spawn_tree(tmp_path)
    reaped = tmp_path / "state" / "reaped"
    # An EMPTY capture dir that never grows. The other half of the predicate
    # is pinned at zero deliberately: during a long single-threaded
    # `g++ -O1 -c wrapper.cpp` that is exactly what a healthy mint's capture
    # dir does, so capture silence must not be able to vote for a kill.
    capture = tmp_path / "capture"
    capture.mkdir()

    async def drive() -> None:
        seen: List[float] = []
        window = SilenceWindow(300.0, now=_Ticks().now)
        watch = asyncio.ensure_future(
            _observe(proc.pid, capture, window, seen.append, 0.15))
        try:
            await _await(
                reaped.exists, lambda: len(seen), watch, proc,
                what="the grandchild to be reaped")
            assert not watch.done(), (
                "_observe killed the mint before its entry even finished: "
                f"{watch.result()!r}")

            floor = len(seen) + 5
            await _await(
                lambda: len(seen) >= floor, lambda: len(seen), watch, proc,
                what="evidence to advance again after the entry finished")

            assert not watch.done(), (
                "FALSE KILL: _observe SIGTERM'd a mint whose replacement "
                "entry was burning real CPU the whole time — "
                f"{watch.result()!r}")
        finally:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch

    try:
        asyncio.run(drive())
    finally:
        _reap_tree(proc)


def test_the_two_signals_cannot_cancel_each_other(tmp_path: Path) -> None:
    """``_evidence`` reports a PAIR, never a sum.

    Summing CPU-seconds and capture-MiB is how a falling term vetoed a rising
    one. The shape IS the guarantee, so it is asserted directly.
    """
    sample = _evidence(os.getpid(), tmp_path)
    assert isinstance(sample, tuple) and len(sample) == 2, (
        "the mint's measured evidence must stay two independent signals — "
        "adding them lets a reaped child's CPU drop swallow the capture bytes "
        "its successor is writing")


def test_an_unsamplable_tree_is_not_a_zero() -> None:
    """A ``/proc`` miss reads as UNKNOWN, never as a cliff to zero.

    A sampler that answers 0.0 on a transient error manufactures the same
    hole the reaped-child bug did, from nothing at all.
    """
    assert _tree_cpu_seconds(0) is None
    assert _tree_cpu_seconds(1 << 30) is None
