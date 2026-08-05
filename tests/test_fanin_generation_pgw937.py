"""pgw#937: a DEAD group's last frame must not outvote the live ones.

The parent's three per-group fan-in dicts (`_group_activities`,
`_group_fn_unavail`, `_group_fn_degraded`) plus `slot.last_state_delta` are
GENERATION-SCOPED facts: they say what the CURRENT incarnation of a group
reported. Before this issue nothing ever cleaned them, and nothing ever asked
whether the group behind an entry was still alive — so a group that died mid
activity could pin that activity RUNNING forever, veto a live group's
`self_stalled` confession, and retire a function a live group was serving.

The ruling this file proves is "down-group semantics" in ``procsplit/parent.py``:
**a group without a live child is not a participant, and its facts are UNKNOWN
— which is neither "still true" nor the live-group default.** So the identity
element of a down group is EXCLUSION from every merge; a bare `.pop()` is not
the fix, because absence-of-a-value already means "this group serves it" and
popping alone would make the dead group read as serving everything.

Two layers, both driving production code:

* the RUNTIME rows reuse the real G=2 harness — two real child processes, a
  real ParentControl, real gRPC to the hub-double — and kill a child while it
  holds an activity open;
* the PARENT rows drive `_fan_in`, `_retire_group_generation` and
  `begin_generation` (the exact functions `_on_child_frame` and
  `_handle_child_death` call) on a real ParentControl, for the two observables
  a real child cannot be made to produce on demand: a stall confession and a
  setup-failure retirement.
"""

from __future__ import annotations

from typing import Callable, Optional

import pytest

from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit.parent import DEATH_LABEL, ParentControl
from gen_worker.topology import ExecutionTopology

from harness.hub_double import is_ready, is_result_for
from test_group_spawn_pgw783 import (  # the real two-process harness
    G2Harness,
    _dials,          # noqa: F401  (fixture)
    _isolated_postmortem,  # noqa: F401  (fixture)
    _payload,
)

HOLD_KIND = "g_hold"


@pytest.fixture()
def g2(tmp_path, _dials):  # noqa: F811
    h = G2Harness(tmp_path)
    try:
        yield h
    finally:
        h.close()


def _activity(kind: str, state: Optional[int] = None) -> Callable[[pb.WorkerMessage], bool]:
    def pred(m: pb.WorkerMessage) -> bool:
        if m.WhichOneof("msg") != "activity_update":
            return False
        act = m.activity_update
        return act.kind == kind and (state is None or act.state == state)

    return pred


# ---------------------------------------------------------------------------
# RUNTIME: two real child processes, one of them dies holding an activity open
# ---------------------------------------------------------------------------


def test_a_dead_groups_open_activity_is_terminated_for_the_worker(g2):
    """Observable A, on real processes.

    g1 opens ``g_hold`` and SIGKILLs itself while it is still RUNNING. Nothing
    else in the worker ever mentions that kind again, so unless the parent emits
    the terminal ON THE DEAD GROUP'S BEHALF the hub folds a RUNNING update into
    ``info.Activities["g_hold"]`` and never receives anything to close it.

    Pre-fix this hangs on the FAILED wait: the only updates for the kind are the
    dead child's RUNNING ones.
    """
    conn = g2.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    conn.send(run_job=pb.RunJob(
        request_id="r-die-g1", attempt=1, function_name="activity-die",
        input_payload=_payload(HOLD_KIND), compute=pb.ResolvedCompute(gpu_index=1)))

    # The group really did open it, and the hub really did see it RUNNING.
    conn.wait_for(_activity(HOLD_KIND, pb.ACTIVITY_STATE_RUNNING))

    died = conn.wait_for(is_result_for("r-die-g1"))
    assert died.job_result.status == pb.JOB_STATUS_FATAL
    assert DEATH_LABEL in died.job_result.safe_message

    # THE ASSERTION: the worker's activity of that kind reaches a terminal.
    term = conn.wait_for(_activity(HOLD_KIND, pb.ACTIVITY_STATE_FAILED))
    # FAILED, not COMPLETED — the process doing it died, it did not finish.
    assert term.activity_update.error, "a terminal with no reason is not evidence"
    # The aggregation still never leaks a group ordinal to the wire.
    assert "g1" not in term.activity_update.detail
    assert "group=1" not in term.activity_update.detail

    # The sibling was undisturbed and still serves.
    conn.send(run_job=pb.RunJob(
        request_id="r-g0-after", attempt=1, function_name="whoami",
        input_payload=_payload(), compute=pb.ResolvedCompute(gpu_index=0)))
    ok = conn.wait_for(is_result_for("r-g0-after"))
    assert ok.job_result.status == pb.JOB_STATUS_OK
    assert b"g=0" in ok.job_result.inline


def test_the_dead_generation_is_retired_and_the_respawn_starts_a_new_one(g2):
    """The parent-side state behind the observable, measured on the live object.

    The dead group's fan-in entry is gone, its participation is over, and the
    respawned child comes back as a NEW generation with empty state — so
    "absent" means the live-group default again (§4.15), not an inherited fact.
    The surviving group's own entry is untouched throughout.
    """
    conn = g2.scheduler.wait_connection(0)
    conn.wait_for(is_ready)
    pc = g2.pc
    assert pc._slots[1].generation == 1 and pc._slots[1].participating

    # g0 holds the SAME kind open for the whole run; its fact must survive.
    conn.send(run_job=pb.RunJob(
        request_id="r-hold-g0", attempt=1, function_name="activity-hold",
        input_payload=_payload(HOLD_KIND), compute=pb.ResolvedCompute(gpu_index=0)))
    conn.wait_for(_activity(HOLD_KIND, pb.ACTIVITY_STATE_RUNNING))

    conn.send(run_job=pb.RunJob(
        request_id="r-die-g1", attempt=1, function_name="activity-die",
        input_payload=_payload(HOLD_KIND), compute=pb.ResolvedCompute(gpu_index=1)))
    died = conn.wait_for(is_result_for("r-die-g1"))
    assert died.job_result.status == pb.JOB_STATUS_FATAL

    # The dead group's fan-in entry is gone; the LIVE group still holds the kind.
    assert 1 not in pc._group_activities or HOLD_KIND not in pc._group_activities[1]
    assert HOLD_KIND in pc._group_activities.get(0, {})

    # The dead group's parent-side observation went with it: the death path is
    # the OTHER exit from `_observations`, and without it 512 orphans FIFO-evict
    # live jobs' observations and billing attestation silently stops.
    assert ("r-die-g1", 1) not in pc._observations
    assert ("r-hold-g0", 1) in pc._observations

    # The respawn is a NEW generation, participating again, with empty state.
    # `_on_child_connect` calls `begin_generation()` and THEN cycles the stream
    # to re-sync the worker, so the hub's second connection is the ordered proof
    # that the new generation exists — no clock involved.
    g2.scheduler.wait_connection(1)
    assert pc._slots[1].generation >= 2
    assert pc._slots[1].participating
    assert pc._group_activities.get(1, {}) == {}

    # And a live group's activity is NOT terminated by a sibling's death.
    assert conn.count(_activity(HOLD_KIND, pb.ACTIVITY_STATE_FAILED)) == 0
    assert conn.count(_activity(HOLD_KIND, pb.ACTIVITY_STATE_COMPLETED)) == 0


# ---------------------------------------------------------------------------
# PARENT: the two observables a real child cannot be asked to produce, driven
# through the same functions `_on_child_frame` / `_handle_child_death` call.
# ---------------------------------------------------------------------------

SOCK = "/tmp/pgw937.sock"


def _parent(groups: int = 2) -> ParentControl:
    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1", worker_id="w-937", worker_jwt="",
    )
    topo = ExecutionTopology(gpu_count=groups, gpus_per_execution_group=1)
    p = ParentControl(settings, socket_path=SOCK, topology=topo)
    for slot in p._slots:
        slot.begin_generation()
    return p


def _running(kind: str, *, stalled: bool, step: int = 1) -> pb.WorkerMessage:
    return pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind=kind, state=pb.ACTIVITY_STATE_RUNNING, step=step,
        counter="mint:graphs", counter_done=float(step), self_stalled=stalled,
    ))


def test_a_dead_groups_frame_cannot_veto_a_live_groups_stall_confession():
    """Observable B — the one that costs money.

    `merge.reconcile_activity_kind` computes `self_stalled` with `all(...)`, so
    ONE non-stalled voter suppresses the confession the hub recycles the pod on.
    A dead group is a permanent non-stalled voter.
    """
    p = _parent(2)
    p._fan_in(p._slots[1], _running("mint", stalled=False))
    p._fan_in(p._slots[0], _running("mint", stalled=False))

    p._retire_group_generation(p._slots[1])   # g1's child dies

    out = p._fan_in(p._slots[0], _running("mint", stalled=True, step=2))
    assert out is not None
    assert out.activity_update.self_stalled is True, (
        "a dead group's last frame vetoed a live group's stall confession"
    )


def test_a_dead_group_cannot_pin_an_activity_running_forever():
    """Observable A at the merge, complementing the runtime row: once the dead
    group is retired, the LIVE group's terminal is the worker's terminal."""
    p = _parent(2)
    p._fan_in(p._slots[1], _running("mint", stalled=False))
    p._fan_in(p._slots[0], _running("mint", stalled=False))

    # Retiring g1 emits the consequence for a kind only IT still held...
    p._retire_group_generation(p._slots[1])
    # ...and g0's own completion now closes the kind for the worker.
    out = p._fan_in(p._slots[0], pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind="mint", state=pb.ACTIVITY_STATE_COMPLETED)))
    assert out is not None
    assert out.activity_update.state == pb.ACTIVITY_STATE_COMPLETED


def test_retiring_a_group_terminates_only_the_kinds_nobody_else_runs():
    """The retirement is not a blanket terminal: a kind a LIVE group still runs
    is re-stated RUNNING (without the dead group's progress), and only a kind
    nobody runs any more is failed."""
    p = _parent(2)
    p._fan_in(p._slots[1], _running("shared", stalled=False, step=9))
    p._fan_in(p._slots[1], _running("g1_only", stalled=False, step=3))
    p._fan_in(p._slots[0], _running("shared", stalled=False, step=2))

    out = p._retire_group_generation(p._slots[1])
    acts = {m.activity_update.kind: m.activity_update
            for m in out if m.WhichOneof("msg") == "activity_update"}
    assert set(acts) == {"g1_only", "shared"}
    assert acts["g1_only"].state == pb.ACTIVITY_STATE_FAILED
    assert acts["shared"].state == pb.ACTIVITY_STATE_RUNNING
    # The dead group's step-9 progress is gone: the worker's is the live one's.
    assert acts["shared"].step == 2


def test_a_respawned_group_does_not_inherit_the_dead_ones_unavailability():
    """Observable C: there is no `fn_available` message, so a respawned group
    that serves the function again says NOTHING. Its unavailability must
    therefore have been dropped with the generation that reported it — or the
    next transient failure anywhere retires the function worker-wide."""
    p = _parent(2)
    fu = pb.WorkerMessage(fn_unavailable=pb.FnUnavailable(
        function_name="txt2img", reason="setup_failed"))
    assert p._fan_in(p._slots[1], fu) is None      # g0 serves it: suppressed

    p._retire_group_generation(p._slots[1])        # g1 dies...
    p._slots[1].begin_generation()                 # ...and comes back serving

    out = p._fan_in(p._slots[0], pb.WorkerMessage(fn_unavailable=pb.FnUnavailable(
        function_name="txt2img", reason="insufficient_vram")))
    assert out is None, (
        "the respawned group inherited the dead one's FnUnavailable, so a "
        "transient failure in g0 retired a function g1 was serving"
    )


def test_a_down_group_is_excluded_from_the_merge_not_defaulted_to_serving():
    """The reason a bare `.pop()` is WORSE than the stale entry.

    `worker_fn_unavailable` reads a `None` value as "this group serves it". With
    g1 DOWN, g0's unavailability is the whole worker's truth and the hub must
    hear it. A naive pop would leave `per_group[1] = None` and swallow it.
    """
    p = _parent(2)
    p._retire_group_generation(p._slots[1])

    out = p._fan_in(p._slots[0], pb.WorkerMessage(fn_unavailable=pb.FnUnavailable(
        function_name="txt2img", reason="insufficient_vram")))
    assert out is not None, (
        "the down group was defaulted to 'serves everything', so the worker "
        "never reported a function no live group can serve"
    )
    assert out.fn_unavailable.function_name == "txt2img"


def test_a_down_group_does_not_veto_a_live_groups_degradation_report():
    """Same inversion on FnDegraded: absence means "serves it native" there, so
    a down group left in the `served_native_somewhere` scan silently suppresses
    the placement hint the live group is asking for."""
    p = _parent(2)
    p._retire_group_generation(p._slots[1])

    out = p._fan_in(p._slots[0], pb.WorkerMessage(fn_degraded=pb.FnDegraded(
        function_name="txt2img", est_latency_multiplier=3.0)))
    assert out is not None
    assert out.fn_degraded.est_latency_multiplier == pytest.approx(3.0)


def test_a_dead_group_stops_advertising_its_functions_and_its_free_vram():
    """Observable D: `merge_state_deltas` UNIONS `available_functions` and SUMS
    `free_vram_bytes`, with no liveness filter — so the hub kept dispatching
    onto a down group and got "compute process restarting" on a loop, and the
    pod's reported free VRAM counted a process that no longer exists."""
    p = _parent(2)
    p._slots[0].last_state_delta = pb.WorkerMessage(state_delta=pb.StateDelta(
        available_functions=["a"], free_vram_bytes=10, phase=pb.WORKER_PHASE_READY))
    p._slots[1].last_state_delta = pb.WorkerMessage(state_delta=pb.StateDelta(
        available_functions=["b"], free_vram_bytes=30, phase=pb.WORKER_PHASE_READY))
    p._note_state_delta()
    assert list(p._last_state_delta.state_delta.available_functions) == ["a", "b"]

    out = p._retire_group_generation(p._slots[1])
    st = p._last_state_delta.state_delta
    assert list(st.available_functions) == ["a"]
    assert st.free_vram_bytes == 10
    # The hub LEARNS it, rather than inferring it from an update that stopped.
    assert any(m.WhichOneof("msg") == "state_delta" for m in out)


def test_with_every_group_down_the_worker_advertises_nothing():
    """The last group's function set is not the worker's while nothing is up."""
    p = _parent(2)
    for slot in p._slots:
        slot.last_state_delta = pb.WorkerMessage(state_delta=pb.StateDelta(
            available_functions=["a"], free_vram_bytes=10,
            phase=pb.WORKER_PHASE_READY))
    p._note_state_delta()
    for slot in p._slots:
        p._retire_group_generation(slot)
    st = p._last_state_delta.state_delta
    assert list(st.available_functions) == []
    assert st.free_vram_bytes == 0
    assert st.phase == pb.WORKER_PHASE_BOOTING


def test_a_retired_generations_late_frame_cannot_resurrect_it():
    """`_settle_link` is bounded, so a frame from a reaped child can still land
    after the death path ran. A worker-SCOPED claim from a retired generation is
    a dead process's view of a worker it has left."""
    p = _parent(2)
    p._retire_group_generation(p._slots[1])
    assert p._fan_in(p._slots[1], _running("mint", stalled=False)) is None
    assert p._group_activities.get(1, {}) == {}
    # Per-request frames still forward: they are about a job, not the worker.
    jr = pb.WorkerMessage(job_result=pb.JobResult(
        request_id="r", attempt=1, status=pb.JOB_STATUS_OK))
    assert p._fan_in(p._slots[1], jr) is jr


def test_G1_is_untouched_by_all_of_this():
    """The standing safety property: a one-child worker is byte-identical to the
    pre-pgw#783 relay. The fan-in dicts are never populated there, so retirement
    is a no-op BY CONSTRUCTION rather than by a flag."""
    p = _parent(1)
    msg = _running("mint", stalled=False)
    assert p._fan_in(p._slots[0], msg) is msg
    assert p._retire_group_generation(p._slots[0]) == []
    assert p._fan_in(p._slots[0], msg) is msg   # still verbatim after a death
