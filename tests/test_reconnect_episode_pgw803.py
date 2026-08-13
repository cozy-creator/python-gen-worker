"""pgw#803: a disconnect/reconnect episode is a hub row that ACCOUNTS for itself.

THE INCIDENT (th#1333's tape, verbatim). A serving pod's stream ended at
17:51:43Z — the hub's own cleanup line says *"the worker PROCESS may well be
alive"* — and the worker's next connect attempt reached the hub at 18:07:19Z.
**15m36s.** The hub then condemned it, revoked its credential, and refused the
reconnect that proved it was alive. The hub half is fixed (two-phase revoke,
readmission on refutation). The worker half was never explained, and could not
be: the reconnect loop's entire narration is `logger.info` on RunPod stdout,
which nothing can read (gw#640's premise).

WHAT THIS LANE ESTABLISHED ON `origin/master`, BY READING THE CODE. The loop's
own constants cannot produce 936 s, so "the backoff grew unbounded" — the
issue's leading hypothesis — is refuted:

* backoff is `random.uniform(0, min(cap, base * 2**attempt))` with
  ``cap = 30 s`` (`Worker.__init__`), i.e. FULL JITTER with a seconds ceiling;
* ``attempt`` resets to 0 after `_BACKOFF_RESET_AFTER_S` (60 s) of
  connectedness, so a long healthy stream's first retry is `uniform(0, 1 s)`;
* a dead peer is called by h2 keepalive within `keepalive_time_ms` (20 s) plus
  `KEEPALIVE_TIMEOUT_S` (10 s), with `keepalive_permit_without_calls`;
* a dial that hangs is cut at `_HELLO_ACK_TIMEOUT_S` (30 s);
* every one of those is older than the incident (all present since 0.56.0).

So the 936 s was time the reconnect loop WAS NOT RUNNING, and the fix for that
class is to make it NAMEABLE — not to add another duration. (The most probable
cause at the incident's 0.78.0 — torch sharing a process with the gRPC stream —
is separately gone: since 0.84.0 the control/compute split is unconditional and
the parent that owns the stream never imports torch.)

These tests fail against the pre-pgw#803 tree, where neither the event nor the
partition exists.
"""

from __future__ import annotations

from typing import List

import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.transport import RECONNECT_EVENT, _ReconnectEpisode

from harness.hub_double import hub_double


# The private `activity._sink` restore this file used to carry is now
# `tests/conftest.py::_fresh_report_sinks` — one authority for the whole suite,
# Not one file remembering.


def _reconnect_events(msgs: List[pb.WorkerMessage], phase: str) -> List[pb.ActivityUpdate]:
    return [
        m.activity_update for m in msgs
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == RECONNECT_EVENT
        and m.activity_update.phase == phase
    ]


# ---------------------------------------------------------------------------
# the partition, on the incident's own numbers
# ---------------------------------------------------------------------------

def test_the_incident_gap_is_attributed_to_a_loop_that_was_not_running() -> None:
    """936 s of gap against ~42 s of pacing: the row must say which is which.

    This is the whole point of the ledger. With only "reconnected after 936 s"
    the next investigator re-derives the same wrong hypothesis (unbounded
    backoff) that this lane had to refute by reading code. With the partition,
    894 s sits in `unaccounted` and the row names the cause class itself.
    """
    ep = _ReconnectEpisode(
        dropped_at=0.0, cause="grpc_unavailable", uptime_s=520.0,
        loop_silent_s=0.4,
    )
    # 62 attempts is what 936 s of a `uniform(0, 30 s)` schedule would need —
    # the shape the "exponential backoff without a cap" story requires.
    ep.attempts = 3
    ep.sched_s = 40.0
    ep.slept_s = 40.0
    ep.dialed_s = 2.0
    ep.teardown_s = 0.0

    at = 936.0
    assert ep.gap_s(at) == 936.0
    assert ep.overshoot_s() == 0.0            # the loop paced itself correctly
    assert ep.unaccounted_s(at) == 894.0      # and then did not run at all


def test_a_starved_sleep_is_overshoot_not_pacing() -> None:
    """The other shape of the same fault: the loop DID sleep, for far longer
    than it asked to. An asyncio sleep on a healthy loop overshoots by
    milliseconds, so minutes of overshoot cannot be confused with a slow
    network, a slow hub, or a long backoff."""
    ep = _ReconnectEpisode(
        dropped_at=0.0, cause="stream_ended", uptime_s=100.0, loop_silent_s=900.0,
    )
    ep.attempts = 1
    ep.sched_s = 0.5
    ep.slept_s = 900.5
    ep.dialed_s = 0.2

    assert ep.overshoot_s() == 900.0
    # ...and it is NOT double-counted as unaccounted: slept time is measured
    # wall time, so the partition still sums.
    assert ep.unaccounted_s(900.7) == pytest.approx(0.0, abs=1e-6)


def test_a_healthy_episode_accounts_for_all_of_its_gap() -> None:
    ep = _ReconnectEpisode(
        dropped_at=0.0, cause="stream_ended", uptime_s=3600.0, loop_silent_s=2.0,
    )
    ep.attempts = 1
    ep.sched_s = 0.4
    ep.slept_s = 0.41
    ep.dialed_s = 0.09
    ep.teardown_s = 0.01

    assert ep.unaccounted_s(0.51) == pytest.approx(0.0, abs=1e-6)
    # The exact difference between what was scheduled and what was slept —
    # a property of the two recorded numbers, not of the runner's speed.
    assert ep.overshoot_s() == pytest.approx(0.01, abs=1e-6)


def test_attempt_outcomes_coalesce_by_count_never_by_dropping() -> None:
    """An hour-long hub outage is two hub rows, not one row per attempt: the
    send queue is bounded, and a replay storm would cost the very evidence the
    episode is made of."""
    ep = _ReconnectEpisode(
        dropped_at=0.0, cause="grpc_unavailable", uptime_s=10.0, loop_silent_s=0.0,
    )
    for _ in range(200):
        ep.note_outcome("grpc_unavailable")
    ep.note_outcome("exc_ConnectionError")

    assert ep.histogram() == "grpc_unavailable=200, exc_ConnectionError=1"


def test_an_episode_starts_when_the_STREAM_ended_not_when_teardown_finished() -> None:
    """The frame the whole partition hangs on.

    `_account_connection` runs at the END of the reconnect loop's `finally`,
    after `reset_for_reconnect` and `on_disconnect`. Opening the episode at
    "now" would date the drop AFTER a teardown it then also charges the episode
    for — so the stream's measured uptime would swallow the teardown, and every
    later term would be read against a window that started too late. Both facts
    are asserted against an injected clock, which is why this row is exact.
    """
    from gen_worker.config import load_settings
    from gen_worker.transport import Transport

    t = Transport(load_settings(worker_id="pgw803-frame"), object())
    t._connected_at = 100.0          # HelloAck landed
    t._last_send_at = 109.0          # last proven scheduling instant
    t._dial_started = 99.0

    # The stream ended at 110.0; teardown then took 0.5 s.
    t._account_connection(
        outcome="stream_ended", ended_at=110.0, teardown_s=0.5)

    ep = t._episode
    assert ep is not None
    assert ep.dropped_at == 110.0, "the episode is dated after its own teardown"
    assert ep.uptime_s == 10.0, "the stream's uptime swallowed the teardown"
    assert ep.loop_silent_s == 1.0
    assert ep.teardown_s == 0.5
    # ...and the teardown is fully inside the episode's own window.
    assert ep.unaccounted_s(110.5) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# the real path: a real Worker, a real gRPC socket, a real involuntary drop
# ---------------------------------------------------------------------------

def test_an_involuntary_drop_and_its_reconnect_both_reach_the_hub() -> None:
    """Kill the stream under a live worker; both halves of the episode must
    arrive ON THE NEXT STREAM.

    That is not incidental: the `dropped` row is emitted while the worker has
    no hub at all, so it can only land because it rides pgw#869's durable
    evidence lane, which `reset_for_reconnect` preserves.

    (This row does NOT cover the earliest window — a drop before the first
    `Executor.ensure_setup`, which is where `activity.bind_sink` runs, i.e.
    during the 72-199 s weights fetch. Such an episode is logger-only and
    therefore invisible. Binding the sink at HelloAck instead — the argument
    `boot_phases` already makes at that call site — fixes it and is NOT done
    here: it is a process-global rebind that clobbers the `_sink` patch
    `test_superseded_disk_ref_th1330` installs, so it belongs to whoever owns
    that surface, not to a transport change.)
    """
    with hub_double(worker_id="pgw803-worker") as (scheduler, _harness):
        first = scheduler.wait_connection(0)
        assert first.hello is not None
        first.kill()

        second = scheduler.wait_connection(1)

        def _has_both(_m: pb.WorkerMessage) -> bool:
            return bool(
                _reconnect_events(second.received, "dropped")
                and _reconnect_events(second.received, "reconnected")
            )

        second.wait_for(_has_both)

        dropped = _reconnect_events(second.received, "dropped")
        reconnected = _reconnect_events(second.received, "reconnected")
        assert dropped, "the drop itself never reached the hub"
        assert reconnected, "the reconnect never reached the hub"

        # The drop names its cause and this process's own last-scheduled
        # instant; the reconnect names the partition of the gap.
        assert "cause=" in dropped[0].detail
        assert "loop_silent=" in dropped[0].detail

        row = reconnected[0].detail
        for axis in (
            "attempts=", "sched=", "slept=", "dialed=",
            "teardown=", "overshoot=", "unaccounted=", "loop_silent_at_drop=",
        ):
            assert axis in row, f"{axis!r} missing from the reconnect row: {row}"
        # A measured span, not a number interpolated into prose.
        assert reconnected[0].duration_ms >= 0
