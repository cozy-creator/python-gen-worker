"""pgw#962: a counter outlived the phase that fed it, and then confessed.

Counters registered through `Activity.counter()` were finished only when the
whole ACTIVITY ended. `self_mint_compile` spans load -> warmup_forward ->
load -> ... for every function on the pod, so `infer:steps` (fed one tick per
warmup ctx event) and `warmup:jobs` stayed open and frozen for the rest of the
mint. `progress.self_diagnosis()` is a registry-wide min-age query, so once the
next phase — which has no counter producer of its own; nothing in the SDK emits
a `load:` or `compile:` family counter — went quiet, the freshest OPEN counter
was the dead one, and the beat confessed `self_stalled` about work that had
already finished successfully.

The hub kills on that confession IMMEDIATELY (`legacyWorkerVerdicts`:
`case freshest.SelfStalled` fires without waiting out the hub's own window), so
a false confession destroys a healthy pod mid-load.

Observed in production, `master` stack, 2026-07-29 05:44:00Z, pod
g9f8f3nycoyueh (L4, gen-worker 0.76.5) — the ONLY self_stalled row in 616
durable activity rows:

    kind=self_mint_compile phase=load counter=infer:steps counter_done=30
    self_stalled=t stalled_for_ms=303614

phase `load`, counter `infer:steps`: the confession named a counter belonging
to a phase that had ended.
"""

from __future__ import annotations

import pytest

from gen_worker import activity, progress


@pytest.fixture(autouse=True)
def _reset():
    progress.reset()
    yield
    progress.reset()
    with activity._lock:
        activity._sink = None
        activity._current = None


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_a_finished_phases_counter_cannot_confess(monkeypatch):
    """The production shape, replayed."""
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
    # Warmup forwards run and finish cleanly: 30 ctx events, one per step.
    act.phase(activity.PHASE_WARMUP_FORWARD, 1, 1)
    for _ in range(30):
        act.counter("infer:steps", progress.UNIT_STEPS).add(1)
        clk.t += 1

    # The mint moves on to the NEXT function's load. That phase has no counter
    # producer, so nothing new is registered.
    act.phase(activity.PHASE_LOAD)
    clk.t += progress.DEFAULT_STALL_WINDOW_S + 5

    assert progress.self_diagnosis() is None, (
        "a counter whose phase ended must not be able to confess for the "
        "phase that replaced it")


def test_the_current_phases_counter_still_confesses(monkeypatch):
    """The guard must stay armed for the phase that actually owns a counter —
    scoping is not a mute button."""
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
    act.phase(activity.PHASE_WARMUP_FORWARD, 1, 4)
    act.counter("warmup:jobs", progress.UNIT_STEPS, total=4).set_done(1)
    clk.t += progress.STALL_WINDOW_S["warmup"] + 5

    snap = progress.self_diagnosis()
    assert snap is not None and snap.name == "warmup:jobs"


def test_registry_counters_are_not_phase_scoped(monkeypatch):
    """`progress.counter()` (the watchdog's `evidence:` counter, the model
    download counter) deliberately spans phases and owns its own finish() —
    phase scoping must not touch it, or a long load loses its only signal."""
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
    ev = progress.counter("evidence:self_mint_compile", progress.UNIT_EVIDENCE)
    ev.set_done(1.0)
    act.phase(activity.PHASE_WARMUP_FORWARD, 1, 1)
    clk.t += 5
    ev.set_done(2.0)

    fresh = progress.freshest()
    assert fresh is not None and fresh.name == "evidence:self_mint_compile"
    assert fresh.age_s == 0.0


def test_activity_end_still_finishes_everything(monkeypatch):
    """gw#621's original guarantee is unchanged."""
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)

    act = activity.begin(activity.KIND_WARMUP, activity.PHASE_WARMUP_FORWARD)
    act.counter("warmup:jobs", progress.UNIT_STEPS).set_done(1)
    act.completed()
    assert progress.snapshot() == []
