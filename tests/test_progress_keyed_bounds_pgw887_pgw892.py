"""Three flat wall clocks that ended REAL WORK.

The rule (also `stall.py`'s opening line): *nothing that can END REAL WORK may
key on a wall clock. A fixed budget cannot tell a healthy three-hour transfer
from a wedge, so it is either useless or it kills work that was about to
succeed.* A deadline may bound a COMMAND, never the WORK.

Three sites violated it; each replacement primitive already existed in-repo:

* `lifecycle._finish_drain` — `wait_idle(timeout=deadline)` then
  `abort_all()`. A render at step 30/50 and one at step 1/50 aborted alike; a
  MINT, which has no partial result to requeue, was abandoned outright.
  Replacement: `progress.self_diagnosis()`, already trusted on the 10 s beat.
* `parallel/group.wait_armed` — 1800 s over "a follower's full pipeline
  materialization (a cold model load)", against a first self-mint measured at
  ~28 min and an sdxl projection of 47 min - 6.26 h.
* `parallel/group._await_arrivals` — 180 s over a follower that may still be
  in a cold `import torch`.
  Replacement for both: `stall.SilenceWindow` over
  `proc_evidence.tree_evidence`, plus the death detection `check_alive()`
  already did.

Every test drives an injected clock or an injected evidence source; none
sleeps out a real window, and none contains a fixed-deadline wait of its own.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import pytest

from gen_worker import progress as progress_mod
from gen_worker.parallel import group as group_mod
from gen_worker.stall import SilenceWindow


# ---------------------------------------------------------------------------
# The drain aborts on non-advancement, never on elapsed time
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Just enough of `Executor` for `_await_tenant_idle`."""

    def __init__(self) -> None:
        self._idle = asyncio.Event()

    async def wait_idle(self, timeout: Optional[float] = None) -> bool:
        await self._idle.wait()
        return True

    def finish(self) -> None:
        self._idle.set()


def _lifecycle(executor: Any, *, work_deadline_at: Any = None) -> Any:
    from gen_worker.lifecycle import Lifecycle

    # Stubbed Lifecycles skip __init__ by repo convention.
    obj = Lifecycle.__new__(Lifecycle)
    obj.executor = executor
    obj._drain_work_deadline_at = work_deadline_at
    return obj


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    progress_mod.reset()
    yield
    progress_mod.reset()


def test_work_that_keeps_advancing_is_never_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline. The old code aborted at the deadline no matter what; this
    one runs a job PAST any deadline the fleet ever set, because the counter
    keeps moving."""
    from gen_worker import lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, "_DRAIN_PROGRESS_POLL_S", 0.01)
    clock = {"t": 0.0}
    monkeypatch.setattr(progress_mod, "_now", lambda: clock["t"])
    counter = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS)
    executor = _FakeExecutor()

    async def _go() -> None:
        life = _lifecycle(executor)
        task = asyncio.ensure_future(life._await_tenant_idle())
        # Drive the fake clock far past both the 30 s SIGTERM budget and the
        # hub's 60 s cluster window, advancing the counter as real work would.
        for _ in range(20):
            clock["t"] += 60.0
            counter.add(1)
            await asyncio.sleep(0.02)
        assert not task.done(), (
            "the drain gave up on work that was still advancing — "
            f"{clock['t']:.0f}s of simulated elapsed time")
        executor.finish()
        assert await task is True

    asyncio.run(_go())


def test_work_that_stops_advancing_is_given_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half: this is not "wait forever". A counter frozen past its
    own per-phase window is the typed `self_stalled` confession the hub
    already kills on, and the drain reads the same fact."""
    from gen_worker import lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, "_DRAIN_PROGRESS_POLL_S", 0.01)
    clock = {"t": 0.0}
    monkeypatch.setattr(progress_mod, "_now", lambda: clock["t"])
    progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS).add(1)

    async def _go() -> None:
        life = _lifecycle(_FakeExecutor())
        task = asyncio.ensure_future(life._await_tenant_idle())
        await asyncio.sleep(0.02)
        assert not task.done()
        # Past `STALL_WINDOW_S["infer"]` (300 s) with no advance.
        clock["t"] += 400.0
        assert await task is False

    asyncio.run(_go())


def test_a_job_that_finishes_first_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker import lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, "_DRAIN_PROGRESS_POLL_S", 0.01)

    async def _go() -> None:
        executor = _FakeExecutor()
        life = _lifecycle(executor)
        task = asyncio.ensure_future(life._await_tenant_idle())
        executor.finish()
        assert await task is True

    asyncio.run(_go())


def test_no_counters_at_all_is_not_a_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate, and the same call the in-call stall loop makes: an
    evidence-free job is not a stalled one, and the outer authority (the hub's
    terminate, the runtime's SIGKILL) bounds the pathological case. Throwing
    away work that was about to finish is the failure this issue is about."""
    from gen_worker import lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, "_DRAIN_PROGRESS_POLL_S", 0.01)
    clock = {"t": 0.0}
    monkeypatch.setattr(progress_mod, "_now", lambda: clock["t"])

    async def _go() -> None:
        life = _lifecycle(_FakeExecutor())
        task = asyncio.ensure_future(life._await_tenant_idle())
        clock["t"] += 10_000.0
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()

    asyncio.run(_go())


def test_an_explicit_operator_budget_still_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `Drain` that arrives on the wire CARRYING `deadline_ms` is "an
    explicit operator budget on a specific command", and that is
    kept. What pgw#887 deletes is the worker INVENTING one: the SIGTERM
    handler's fleet default, which is what abandoned a 29-minute mint as
    `abandoned_shutdown`."""
    from gen_worker import lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, "_DRAIN_PROGRESS_POLL_S", 0.01)

    async def _go() -> None:
        loop = asyncio.get_running_loop()
        life = _lifecycle(_FakeExecutor(), work_deadline_at=loop.time() + 0.05)
        assert await life._await_tenant_idle() is False

    asyncio.run(_go())


def test_a_signal_drain_supplies_no_work_budget() -> None:
    """The SIGTERM path calls `start_drain` without `work_deadline=True`, and
    the hub's `Drain` handler calls it WITH — that asymmetry is the fix."""
    import inspect

    from gen_worker import worker as worker_mod
    from gen_worker.lifecycle import Lifecycle

    assert "work_deadline" not in inspect.getsource(worker_mod.Worker.arun)
    assert (
        inspect.signature(Lifecycle.start_drain)
        .parameters["work_deadline"].default is False)
    assert "work_deadline=True" in inspect.getsource(Lifecycle.on_message)


def test_the_drain_no_longer_reads_a_deadline_for_the_tenant_wait() -> None:
    """Structural: the wait must not be reachable from `_drain_deadline_at`
    again. The deadline still bounds the FLUSH, which ships results
    already held — a command, not the work."""
    import inspect

    from gen_worker.lifecycle import Lifecycle

    wait_src = inspect.getsource(Lifecycle._await_tenant_idle)
    assert "self._drain_deadline_at" not in wait_src, (
        "the tenant wait reads the SHUTDOWN deadline again")

    finish_src = inspect.getsource(Lifecycle._finish_drain)
    assert "wait_idle(timeout=" not in finish_src, (
        "the tenant wait is deadline-bounded again")
    assert "self._await_tenant_idle()" in finish_src


# ---------------------------------------------------------------------------
# The two group-formation bounds
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class _FakeStore:
    def __init__(self) -> None:
        self.arrived: set = set()

    def check(self, keys: List[str]) -> bool:
        return all(k in self.arrived for k in keys)


def _group(monkeypatch: pytest.MonkeyPatch, *, degree: int = 2) -> Any:
    g = group_mod.RankGroup.__new__(group_mod.RankGroup)
    g.devices = tuple(range(degree))
    g.backend = "gloo"
    g._procs = [_FakeProc(1000 + i) for i in range(1, degree)]
    g._staging_peaks = {}
    g._store = _FakeStore()
    g._error_q = None
    return g


def _specs(degree: int) -> List[group_mod.RankSpec]:
    return [
        group_mod.RankSpec(
            rank=r, world_size=degree, device=r, master_addr="127.0.0.1",
            master_port=1, backend="gloo", group_name="t")
        for r in range(degree)
    ]


def test_a_follower_that_keeps_working_is_never_condemned_while_arming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_ARM_TIMEOUT_S = 1800.0` raised "followers not armed after 1800s" at
    30:01 for a follower that was still legitimately materializing. The first
    self-mint measured ~28 min; pgw#846 projects 47 min - 6.26 h for sdxl."""
    monkeypatch.setattr(group_mod, "_STAGING_POLL_S", 0.0)
    clock = {"t": 0.0}
    work = {"cpu": 0.0}
    monkeypatch.setattr(
        group_mod.proc_evidence, "tree_evidence", lambda _pid: work["cpu"])

    ready: List[int] = []
    ticks = {"n": 0}

    class _Q:
        @staticmethod
        def get(timeout: float = 0.0) -> int:
            import queue

            ticks["n"] += 1
            clock["t"] += 60.0        # a minute of simulated wall time a tick
            work["cpu"] += 1.0        # the follower is doing real work
            if ticks["n"] > 60:       # ... for a simulated hour
                if ready:
                    return ready.pop()
                ready.append(1)
                return 1
            raise queue.Empty

    g = _group(monkeypatch, degree=2)
    g._ready_q = _Q()
    g.check_alive = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        group_mod, "SilenceWindow",
        lambda window_s: SilenceWindow(window_s, now=lambda: clock["t"]))

    g.wait_armed()  # must not raise: an hour of advancing work
    assert clock["t"] > 1800.0, "the fixture did not outrun the old 1800s bound"


def test_a_follower_that_stops_working_is_condemned_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half. Death is `check_alive()`'s; SILENCE is this."""
    import queue

    monkeypatch.setattr(group_mod, "_STAGING_POLL_S", 0.0)
    clock = {"t": 0.0}
    monkeypatch.setattr(
        group_mod.proc_evidence, "tree_evidence", lambda _pid: 1.0)  # frozen

    class _Q:
        @staticmethod
        def get(timeout: float = 0.0) -> int:
            clock["t"] += 60.0
            raise queue.Empty

    g = _group(monkeypatch, degree=2)
    g._ready_q = _Q()
    g.check_alive = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        group_mod, "SilenceWindow",
        lambda window_s: SilenceWindow(window_s, now=lambda: clock["t"]))

    with pytest.raises(group_mod.RankGroupError) as excinfo:
        g.wait_armed()
    assert "no CPU or I/O" in str(excinfo.value)


def test_a_slow_arrival_is_not_a_failed_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_RENDEZVOUS_TIMEOUT_S = 180.0` condemned a follower wedged in a cold
    `import torch` at 181 s — which is a follower burning CPU, i.e. the one
    state this must not condemn."""
    monkeypatch.setattr(group_mod, "_STAGING_POLL_S", 0.0)
    clock = {"t": 0.0}
    work = {"cpu": 0.0}
    calls = {"n": 0}

    def _evidence(_pid: int) -> float:
        return work["cpu"]

    monkeypatch.setattr(group_mod.proc_evidence, "tree_evidence", _evidence)
    monkeypatch.setattr(
        group_mod, "SilenceWindow",
        lambda window_s: SilenceWindow(window_s, now=lambda: clock["t"]))

    g = _group(monkeypatch, degree=2)
    g.check_alive = lambda: None  # type: ignore[method-assign]
    specs = _specs(2)

    original_check = g._store.check

    def _check(keys: List[str]) -> bool:
        calls["n"] += 1
        clock["t"] += 30.0
        work["cpu"] += 1.0
        if calls["n"] > 40:  # arrives at ~20 simulated minutes
            g._store.arrived.update(group_mod.arrive_key(s) for s in specs[1:])
        return original_check(keys)

    g._store.check = _check  # type: ignore[method-assign]
    g._await_arrivals(specs)
    assert clock["t"] > 180.0, "the fixture did not outrun the old 180s bound"


def test_a_silent_arrival_still_fails_the_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(group_mod, "_STAGING_POLL_S", 0.0)
    clock = {"t": 0.0}
    monkeypatch.setattr(
        group_mod.proc_evidence, "tree_evidence", lambda _pid: 1.0)
    monkeypatch.setattr(
        group_mod, "SilenceWindow",
        lambda window_s: SilenceWindow(window_s, now=lambda: clock["t"]))

    g = _group(monkeypatch, degree=2)
    g.check_alive = lambda: None  # type: ignore[method-assign]
    original = g._store.check
    g._store.check = lambda keys: (  # type: ignore[method-assign]
        clock.__setitem__("t", clock["t"] + 30.0) or original(keys))

    with pytest.raises(group_mod.RankGroupError) as excinfo:
        g._await_arrivals(_specs(2))
    assert "no CPU or I/O" in str(excinfo.value)


def test_evidence_uses_a_high_water_mark_not_a_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A descendant's CPU migrates into its parent's `cutime`/`cstime` on
    reap, so a live-only tree sum FALLS when a subprocess finishes. Comparing
    against the previous sample would read that fall as an advance."""
    readings = iter([10.0, 5.0, 5.0, 11.0])
    monkeypatch.setattr(
        group_mod.proc_evidence, "tree_evidence", lambda _pid: next(readings))
    g = _group(monkeypatch, degree=2)
    assert g._followers_advanced() is True    # 10.0, first mark
    assert g._followers_advanced() is False   # 5.0 < mark: NOT an advance
    assert g._followers_advanced() is False   # 5.0 again
    assert g._followers_advanced() is True    # 11.0 clears the mark


def test_an_unreadable_follower_is_left_to_check_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tree_evidence` returning None is "cannot say", not "stalled". An
    unreadable process is condemned by death detection or not at all."""
    monkeypatch.setattr(
        group_mod.proc_evidence, "tree_evidence", lambda _pid: None)
    g = _group(monkeypatch, degree=2)
    assert g._followers_advanced() is False
    assert g._staging_peaks == {}


def test_the_work_bounding_constants_are_gone() -> None:
    """Structural, and the point of the issue: the two names that bounded
    WORK must not exist to be reached for again. `_COLLECTIVE_TIMEOUT_S`
    (an in-call collective ceiling) and `_STORE_CONNECT_TIMEOUT_S` (a
    localhost socket connect) are commands and stay."""
    assert not hasattr(group_mod, "_ARM_TIMEOUT_S")
    assert not hasattr(group_mod, "_RENDEZVOUS_TIMEOUT_S")
    assert hasattr(group_mod, "_STAGING_SILENCE_WINDOW_S")
    assert hasattr(group_mod, "_COLLECTIVE_TIMEOUT_S")


def test_wait_armed_takes_no_duration() -> None:
    import inspect

    sig = inspect.signature(group_mod.RankGroup.wait_armed)
    assert list(sig.parameters) == ["self"], (
        "a caller can still hand this a work budget")


# ---------------------------------------------------------------------------
# The shared producer
# ---------------------------------------------------------------------------


def test_tree_evidence_has_one_implementation() -> None:
    """pgw#892 needed exactly what `procsplit.parent._child_evidence` already
    was. Two copies of a liveness primitive is how one of them stops matching
    the other's failure modes."""
    import inspect

    from gen_worker import proc_evidence
    from gen_worker.procsplit import parent

    assert "proc_evidence.tree_evidence" in inspect.getsource(
        parent._ChildSlot._child_evidence)
    assert proc_evidence.tree_evidence(-1) is None  # no such pid, no raise


def test_tree_evidence_reads_this_process() -> None:
    from gen_worker import proc_evidence
    import os

    value = proc_evidence.tree_evidence(os.getpid())
    assert value is not None and value > 0.0
