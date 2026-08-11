"""pgw#894 — serving work could refresh a MINT's stall clock.

Two process-globals met, and the meeting had no identity in it.
`activity._current` is a singleton every `begin()` replaces; `progress._counters`
was keyed by NAME alone, so `freshest()` returned whichever counter anywhere in
the process advanced most recently. The 10 s beat then attached THAT counter to
whatever activity happened to be current, and the hub advances an activity's
`UpdatedAt` from a counter-name change or value increase
(`worker_activity.go:323-338`) — which is the timestamp its stall and
condemnation path reads.

So a request's `infer:steps` deferred a background mint's stall verdict. The
standing chaos hub's 8,398-line log carried 28 lines reporting `infer:steps`
under `self_mint_compile`; at line 4534 a request was assigned to a worker, at
4535 that worker reported `self_mint_compile/trace_graph infer:steps 779/?`
two seconds later, and at 4542 the hub declined a condemnation because that
mint activity was "0s ago".

The fix is an OWNER on every counter and an owner-scoped stall question. What
is deliberately NOT changed: registry-wide `freshest()` still exists and still
means "is this process doing anything at all", which is the right answer to
the liveness question the in-call stall loop and the drain ask. And the
delegated child's own measured CPU/file watchdog is untouched — this is a
re-scope, not a watchdog deletion.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gen_worker import activity as activity_mod
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


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_two_scopes_may_hold_the_same_counter_name() -> None:
    """Two concurrent requests both counting `infer:steps` is the ordinary
    case, and a name-keyed registry made them one counter."""
    a = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS, owner="req-a")
    b = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS, owner="req-b")
    assert a is not b
    a.add(5)
    by_owner = {s.owner: s.done for s in progress_mod.snapshot()}
    assert by_owner == {"req-a": 5.0, "req-b": 0.0}


def test_a_scope_sees_only_its_own_counters(clock: Dict[str, float]) -> None:
    mint = progress_mod.counter("compile:evidence", progress_mod.UNIT_EVIDENCE,
                                owner="mint")
    req = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS,
                               owner="request:r1")
    mint.add(1)
    clock["t"] += 100.0
    req.add(1)

    assert progress_mod.freshest("request:r1").name == "infer:steps"
    assert progress_mod.freshest("mint").name == "compile:evidence"
    # Registry-wide is unchanged and still answers the PROCESS question.
    assert progress_mod.freshest().name == "infer:steps"


def test_a_stalled_scope_is_diagnosed_while_the_process_is_busy(
    clock: Dict[str, float],
) -> None:
    """The headline, at the registry layer: the mint is frozen, the request is
    moving, and the mint must still read as stalled."""
    progress_mod.counter("compile:evidence", progress_mod.UNIT_EVIDENCE,
                         owner="mint").add(1)
    req = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS,
                               owner="request:r1")
    # Past `STALL_WINDOW_S["compile"]` (600 s), with the request advancing.
    for _ in range(10):
        clock["t"] += 100.0
        req.add(1)

    assert progress_mod.self_diagnosis("mint") is not None, (
        "a frozen mint reads as healthy because a request is moving beside it")
    assert progress_mod.self_diagnosis("request:r1") is None
    # And the process is, correctly, not stalled.
    assert progress_mod.self_diagnosis() is None


def test_finishing_one_scope_leaves_the_other(clock: Dict[str, float]) -> None:
    a = progress_mod.counter("x", "steps", owner="a")
    progress_mod.counter("x", "steps", owner="b")
    a.finish()
    assert {s.owner for s in progress_mod.snapshot()} == {"b"}


# ---------------------------------------------------------------------------
# The activity beat
# ---------------------------------------------------------------------------


class _Sink:
    def __init__(self) -> None:
        self.updates: List[Any] = []

    def __call__(self, update: Any) -> None:
        self.updates.append(update)


@pytest.fixture()
def sink(monkeypatch: pytest.MonkeyPatch) -> _Sink:
    s = _Sink()
    monkeypatch.setattr(activity_mod, "_sink", s)
    return s


def test_an_activity_has_its_own_scope() -> None:
    a = activity_mod.Activity("self_mint_compile")
    b = activity_mod.Activity("self_mint_compile")
    assert a.id != b.id and a.kind in a.id


def test_the_beat_reports_the_activitys_own_counter(
    clock: Dict[str, float], sink: _Sink, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED: the beat attached `progress.freshest()` — the min-age counter
    ANYWHERE in the process — to whatever activity was current, so a request
    counter was reported under `self_mint_compile` and refreshed the hub's
    stall clock for it."""
    mint = activity_mod.begin("self_mint_compile", "trace_graph")
    mint.counter("compile:evidence", progress_mod.UNIT_EVIDENCE).add(1)

    # Unrelated serving work, advancing, in its own scope.
    req = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS,
                               owner="request:c8100857")
    clock["t"] += 5.0
    req.add(779)

    sink.updates.clear()
    activity_mod.on_beat()
    assert len(sink.updates) == 1
    beat = sink.updates[0]
    assert beat.kind == "self_mint_compile"
    assert beat.counter == "compile:evidence", (
        f"the mint's beat carried {beat.counter!r} — a counter it is not "
        "producing; the hub advances UpdatedAt from exactly this field")
    mint.completed()


def test_a_frozen_mint_confesses_while_a_request_runs(
    clock: Dict[str, float], sink: _Sink,
) -> None:
    """The end-to-end shape of the observable: mint frozen past its window,
    request and download both moving, and the beat must still say
    `self_stalled`. Before this, `self_diagnosis()` was registry-wide and the
    moving counters answered for the mint."""
    mint = activity_mod.begin("self_mint_compile", "trace_graph")
    mint.counter("compile:evidence", progress_mod.UNIT_EVIDENCE).add(1)

    req = progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS,
                               owner="request:r1")
    dl = progress_mod.counter("download:m", progress_mod.UNIT_BYTES,
                              owner="request:r1")
    for _ in range(10):
        clock["t"] += 100.0   # 1000 s total, past the 600 s compile window
        req.add(1)
        dl.add(1 << 20)

    sink.updates.clear()
    activity_mod.on_beat()
    beat = sink.updates[0]
    assert beat.self_stalled is True, (
        "the mint reported healthy because a request was moving beside it")
    assert beat.counter == "compile:evidence"
    mint.completed()


def test_an_activity_that_really_is_advancing_does_not_confess(
    clock: Dict[str, float], sink: _Sink,
) -> None:
    mint = activity_mod.begin("self_mint_compile", "trace_graph")
    c = mint.counter("compile:evidence", progress_mod.UNIT_EVIDENCE)
    for _ in range(10):
        clock["t"] += 100.0
        c.add(1)
    sink.updates.clear()
    activity_mod.on_beat()
    assert sink.updates[0].self_stalled is False
    mint.completed()


def test_an_activity_with_no_counters_beats_nothing(
    clock: Dict[str, float], sink: _Sink,
) -> None:
    """A scope with no producer says nothing rather than borrowing somebody
    else's number — which is what the old registry-wide lookup did."""
    mint = activity_mod.begin("self_mint_compile", "trace_graph")
    progress_mod.counter("infer:steps", progress_mod.UNIT_STEPS,
                         owner="request:r1").add(1)
    sink.updates.clear()
    activity_mod.on_beat()
    assert sink.updates == []
    mint.completed()


def test_phase_scoped_lifetime_still_holds(clock: Dict[str, float]) -> None:
    """pgw#962 is not undone by pgw#894: a counter is still closed when the
    phase that registered it ends."""
    act = activity_mod.begin("self_mint_compile", "load")
    act.counter("load:bytes", progress_mod.UNIT_BYTES).add(1)
    assert progress_mod.freshest(act.id) is not None
    act.phase("inductor_compile")
    assert progress_mod.freshest(act.id) is None
    act.completed()


# ---------------------------------------------------------------------------
# The producers
# ---------------------------------------------------------------------------


def test_the_request_emitter_owns_its_own_counter() -> None:
    """`_make_ctx_emitter` used to do
    `activity.current().counter("infer:steps", ...)`, which on a pod running a
    background mint credits the MINT. It now registers under the request."""
    import inspect

    from gen_worker.executor import Executor

    src = inspect.getsource(Executor._make_ctx_emitter)
    assert 'owner=f"request:{job.request_id}"' in src
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#"))
    assert "activity_mod.current()" not in code, (
        "the request emitter reaches for the current activity again")


def test_scoped_counter_prefers_the_open_activity(clock: Dict[str, float]) -> None:
    """A download or a load belongs to the phase that asked for it."""
    act = activity_mod.begin("self_mint_compile", "load")
    c = activity_mod.scoped_counter("download:x", progress_mod.UNIT_BYTES)
    assert c.owner == act.id
    act.completed()


def test_scoped_counter_still_works_with_no_activity() -> None:
    """Library and CLI use has no activity at all, and must not start needing
    one."""
    assert activity_mod.current() is None
    assert activity_mod.scoped_counter("download:x", progress_mod.UNIT_BYTES).owner == ""


def test_the_request_counter_dies_with_the_request() -> None:
    """pgw#962's rule, applied to the new scope: a counter left open after its
    producer stopped is the min-age counter of work nobody is doing."""
    import inspect

    from gen_worker.executor import Executor

    src = inspect.getsource(Executor._finish)
    assert 'owner=f"request:{job.request_id}"' in src and ".finish()" in src


def test_the_in_process_liveness_view_is_deliberately_unscoped() -> None:
    """The in-call stall loop and the drain ask "is this process wedged", and
    registry-wide is the right answer to that. Pinned so a later sweep does
    not "finish the job" by scoping them too — that would make a request wait
    on its own counter, which many handlers never register."""
    import inspect

    from gen_worker.executor import Executor

    src = inspect.getsource(Executor._await_handler) if hasattr(
        Executor, "_await_handler") else inspect.getsource(Executor)
    assert "progress_mod.self_diagnosis()" in src
