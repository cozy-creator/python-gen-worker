"""pgw#1157: the mint child's evidence counter must survive a phase change.

The incident is measured, not hypothetical. RunPod A40 ``bgmdxhazxsugmk``,
release ``6ee9b4d4df2697a53da6f43a``, gen-worker **0.112.0** — every piece of
pgw#824's instrumentation present and shipped — spent 62 minutes inside
``trace_graph`` and sent the hub not one counter-carrying beat. The mint was
advancing the whole time (16 of 36 declared entries packed, ``export_s``
1378.52, ``compile_s`` 2065.36), and the lane that paid for the pod correctly
reported that it could not tell that from a wedge.

The locus is one captured object. ``mint_delegate._on_evidence`` acquired the
``mint_child_evidence`` counter ONCE and closed over it, while
``Activity.counter()`` binds a counter to the phase that registered it and
``Activity.phase()`` FINISHES — i.e. unregisters from the process registry —
every counter the new phase does not own (pgw#962). The mint crosses
``load`` -> ``warmup_forward`` -> ``trace_graph``, so from the first phase
change onward every ``set_done`` fed an object no reader could reach:
``activity.on_beat`` found no counter for the activity and returned WITHOUT
emitting, ``progress.self_diagnosis`` had nothing to diagnose, and the hub saw
only counterless heartbeats — which its rule treats as progress by definition,
because an old worker must never be condemned for a signal it cannot send.

These tests drive the REAL ``Activity``, the REAL process counter registry,
the REAL ``_on_evidence`` callback and the REAL ``activity.on_beat``.
"""
from __future__ import annotations

from typing import List

import pytest

from gen_worker import activity as activity_mod
from gen_worker import mint_delegate
from gen_worker import progress as progress_mod
from gen_worker.pb import worker_scheduler_pb2 as pb


@pytest.fixture(autouse=True)
def _clean_registry():
    activity_mod.reset_for_tests()
    progress_mod.reset()
    yield
    activity_mod.reset_for_tests()
    progress_mod.reset()


def _capture() -> List[pb.ActivityUpdate]:
    """Bind a sink that records every ActivityUpdate the process emits."""
    seen: List[pb.ActivityUpdate] = []
    activity_mod._sink = lambda update: seen.append(update)
    return seen


def _counter_beats(seen: List[pb.ActivityUpdate]) -> List[pb.ActivityUpdate]:
    return [u for u in seen if u.counter]


def test_evidence_counter_survives_the_mint_s_phase_changes() -> None:
    """The pod's own sequence: load -> warmup_forward -> trace_graph, with the
    child reporting evidence throughout.

    RED before pgw#1157: after the first ``phase()`` call the beat carries no
    counter at all, and ``on_beat`` emits nothing.
    """
    seen = _capture()
    act = activity_mod.begin("self_mint_compile", "load")
    apply = mint_delegate._on_evidence(act)

    # Phase `load`: the child is reading weights and the counter works.
    apply(10.0)
    activity_mod.on_beat()
    assert _counter_beats(seen), "no counter beat even before a phase change"
    seen.clear()

    # The two phase changes the real mint makes.
    act.phase("warmup_forward")
    apply(25.0)
    act.phase("trace_graph", 1, 36)

    # 62 minutes of this, on the real pod.
    for i, value in enumerate((40.0, 55.0, 70.0, 85.0), start=1):
        apply(value)
        activity_mod.on_beat()

    beats = _counter_beats(seen)
    assert beats, (
        "the mint reported evidence four times inside trace_graph and NOT ONE "
        "counter-carrying beat reached the wire — this is the 62-minute "
        "silence, and no hub-side rule can see through it"
    )
    last = beats[-1]
    assert last.counter == mint_delegate.EVIDENCE_COUNTER
    assert last.counter_done == pytest.approx(85.0), (
        f"beat carries counter_done={last.counter_done}; the reader must see "
        f"the LATEST evidence, not a value frozen at the phase change"
    )
    assert last.phase == "trace_graph"
    assert last.step == 1 and last.total_steps == 36, (
        "the position must ride the same beat as the counter"
    )


def test_the_counter_is_readable_through_the_process_registry() -> None:
    """The registry is what every reader actually queries — ``on_beat`` via
    ``progress.freshest(act.id)``, the self-diagnosis via
    ``progress.self_diagnosis(act.id)``. A counter that is fed but not
    REGISTERED is invisible to both, which is the whole defect."""
    _capture()
    act = activity_mod.begin("self_mint_compile", "load")
    apply = mint_delegate._on_evidence(act)
    apply(10.0)
    act.phase("trace_graph", 1, 36)
    apply(20.0)

    snap = progress_mod.freshest(act.id)
    assert snap is not None, (
        "the mint's own scope holds NO open counter during trace_graph — "
        "`freshest(act.id)` is None, so on_beat returns without emitting and "
        "self_diagnosis can never confess"
    )
    assert snap.name == mint_delegate.EVIDENCE_COUNTER
    assert snap.done == pytest.approx(20.0)


def test_a_frozen_mint_can_now_confess() -> None:
    """The other half: with the counter registered, a mint whose evidence
    stops advancing past its window is a typed ``self_stalled`` confession
    rather than an unbroken stream of healthy-looking heartbeats.

    RED before the fix for the same reason as above — with no counter in the
    activity's scope there is nothing to be stale, so the confession was
    structurally unreachable for the entire compile.
    """
    seen = _capture()
    clock = [1000.0]
    progress_mod._now = lambda: clock[0]
    try:
        act = activity_mod.begin("self_mint_compile", "load")
        apply = mint_delegate._on_evidence(act)
        apply(10.0)
        act.phase("trace_graph", 1, 36)
        apply(20.0)
        seen.clear()

        # Well inside the window: advancing evidence, no confession.
        clock[0] += 120.0
        apply(30.0)
        activity_mod.on_beat()
        assert not any(u.self_stalled for u in seen), "confessed while advancing"

        # Past the compile family's window with the value frozen.
        window = progress_mod.window_for(mint_delegate.EVIDENCE_COUNTER)
        clock[0] += window + 60.0
        apply(30.0)  # no advance
        activity_mod.on_beat()
        assert any(u.self_stalled for u in seen), (
            f"evidence frozen for {window + 60.0:.0f}s past a {window:.0f}s "
            f"window and the worker never confessed"
        )
    finally:
        progress_mod._now = __import__("time").monotonic


def test_evidence_counter_takes_the_compile_family_s_patience() -> None:
    """It measures an AOTI compile, so it must inherit the ``compile``
    window (600 s), not the unnamed default (300 s) it got by having no
    family prefix at all."""
    assert mint_delegate.EVIDENCE_COUNTER.startswith("compile:")
    assert progress_mod.window_for(mint_delegate.EVIDENCE_COUNTER) == (
        progress_mod.STALL_WINDOW_S["compile"])
    assert (progress_mod.window_for(mint_delegate.EVIDENCE_COUNTER)
            > progress_mod.DEFAULT_STALL_WINDOW_S), (
        "an entry that spends minutes inside one inductor call is exactly "
        "what the compile family's longer patience exists for")


def test_telemetry_still_never_costs_the_mint_its_work() -> None:
    """An Activity double with no counter registry keeps working: the
    heartbeat is the half the hub's liveness rule reads, and it must survive
    an activity that cannot hold counters at all."""

    class _ActNoCounter:
        def __init__(self) -> None:
            self.beats = 0

        def heartbeat(self) -> None:
            self.beats += 1

    act = _ActNoCounter()
    apply = mint_delegate._on_evidence(act)
    apply(12.5)
    apply(13.5)
    assert act.beats == 2
