"""#1660: the parent owns the emitted lifecycle sequence, across child respawns.

pgw's own 2026-08-19 audit §2 — filed there, never fixed: the compute child's
session id survives a respawn but its ``_state_seq`` resets to 1, and the hub
SILENTLY DROPS the lower seq. Every arm below is about what the HUB does with
what the parent emitted, not about what the child said.
"""

from __future__ import annotations

import time

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit import merge
from gen_worker.procsplit.lifecycle_relay import LifecycleRelay

SESSION = "sess-pgw1660"
RELEASE = "rel-pgw1660"


def _snapshot(
    *,
    state_seq: int,
    intent_status: "pb.LifecycleIntentStatus" = pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
    capability_state: "pb.FunctionCapabilityState" = pb.FUNCTION_CAPABILITY_STATE_APPLYING,
    session: str = "child-session",
    intent_id: str = "intent-a",
) -> pb.LifecycleSnapshot:
    at = int(time.time() * 1000)
    return pb.LifecycleSnapshot(
        worker_session_id=session,
        state_seq=state_seq,
        full_replace=True,
        generated_at_unix_ms=at,
        intents=[
            pb.IntentState(
                worker_session_id=session,
                state_seq=state_seq,
                goal_id="goal-a",
                intent_id=intent_id,
                release_id=RELEASE,
                status=intent_status,
                stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                since_unix_ms=at,
                updated_at_unix_ms=at,
            )
        ],
        capabilities=[
            pb.FunctionCapability(
                function_name="echo",
                release_id=RELEASE,
                state=capability_state,
            )
        ],
    )


def _relay() -> LifecycleRelay:
    return LifecycleRelay(SESSION, RELEASE)


def test_the_parent_stamps_its_own_session_over_every_child_field() -> None:
    relay = _relay()
    session_id, stamped = relay.hello([_snapshot(state_seq=4)])
    assert session_id == SESSION
    assert stamped is not None
    assert stamped.worker_session_id == SESSION
    assert all(i.worker_session_id == SESSION for i in stamped.intents)


def test_the_sequence_advances_across_a_child_respawn() -> None:
    """THE regression this relay exists for: after a respawn the child's own
    ``state_seq`` restarts at 1, and a snapshot the hub reads as lower is
    dropped with no rejection, no metric and no log."""
    relay = _relay()
    _session, hello = relay.hello([_snapshot(state_seq=1)])
    assert hello is not None
    seqs = [hello.state_seq]
    for child_seq in (2, 3, 4):
        stamped = relay.stamp(
            _snapshot(state_seq=child_seq, capability_state=_state(child_seq))
        )
        assert stamped is not None
        seqs.append(stamped.state_seq)

    # The child dies and its replacement starts counting from 1 again.
    for child_seq in (1, 2):
        stamped = relay.stamp(
            _snapshot(state_seq=child_seq, capability_state=_state(child_seq + 10))
        )
        assert stamped is not None
        seqs.append(stamped.state_seq)

    assert seqs == sorted(set(seqs)), (
        f"the emitted state_seq must be strictly increasing; got {seqs}. A "
        f"repeat is a hub-side `lifecycle state_seq conflict` rejection and a "
        f"regression is a SILENT drop"
    )


def _state(n: int) -> "pb.FunctionCapabilityState":
    """A distinct capability state per step, so each snapshot really differs."""
    states = [
        pb.FUNCTION_CAPABILITY_STATE_APPLYING,
        pb.FUNCTION_CAPABILITY_STATE_BOOT_STALE,
        pb.FUNCTION_CAPABILITY_STATE_READY,
        pb.FUNCTION_CAPABILITY_STATE_FAILED,
    ]
    return states[n % len(states)]


def test_an_unchanged_projection_is_not_resent() -> None:
    """The hub rejects a reused seq whose bytes differ and no-ops one that
    matches; re-sending an unchanged projection buys nothing either way."""
    relay = _relay()
    relay.hello([_snapshot(state_seq=1)])
    assert relay.stamp(_snapshot(state_seq=2)) is None


def test_a_terminal_intent_stays_terminal_after_the_child_respawns() -> None:
    """The hub discards the WHOLE snapshot when a previously terminal intent
    changes status, so the projection would freeze at its pre-respawn state."""
    relay = _relay()
    relay.hello([_snapshot(state_seq=1)])
    done = relay.stamp(
        _snapshot(state_seq=2, intent_status=pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED)
    )
    assert done is not None
    assert done.intents[0].status == pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED

    # The respawned child re-applies the command and reports the same intent id
    # as freshly ACCEPTED.
    after = relay.stamp(
        _snapshot(
            state_seq=1,
            intent_status=pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
            capability_state=pb.FUNCTION_CAPABILITY_STATE_READY,
        )
    )
    assert after is not None
    assert after.intents[0].status == pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED, (
        "a terminal intent that goes back to ACCEPTED costs the entire snapshot"
    )
    assert after.capabilities[0].state == pb.FUNCTION_CAPABILITY_STATE_READY, (
        "what dispatch actually reads still comes LIVE from the child — the pin "
        "is on intent history, never on readiness"
    )


def test_one_group_without_a_projection_withholds_the_whole_pair() -> None:
    relay = _relay()
    assert relay.hello([_snapshot(state_seq=1), None]) == ("", None)
    assert not relay.announced


def test_a_mid_stream_snapshot_is_dropped_when_the_hello_went_out_legacy() -> None:
    """No session id reached the hub, so a snapshot naming one is a
    `worker_session_mismatch` rejection."""
    relay = _relay()
    relay.hello([None])
    assert not relay.announced
    assert relay.receipt(pb.GoalReceipt(goal_id="g")) is None


def test_a_snapshot_the_hub_would_discard_is_held_back() -> None:
    relay = _relay()
    relay.hello([_snapshot(state_seq=1)])
    broken = _snapshot(state_seq=2, capability_state=pb.FUNCTION_CAPABILITY_STATE_READY)
    broken.capabilities[0].release_id = "another-release"
    assert relay.stamp(broken) is None


# ---------------------------------------------------------------------------
# G > 1: the hub sees ONE worker
# ---------------------------------------------------------------------------


def test_the_worker_is_ready_only_when_every_group_is() -> None:
    merged = merge.merge_lifecycle_snapshots([
        _snapshot(state_seq=1, capability_state=pb.FUNCTION_CAPABILITY_STATE_READY),
        _snapshot(state_seq=1, capability_state=pb.FUNCTION_CAPABILITY_STATE_APPLYING),
    ])
    assert merged is not None
    assert merged.capabilities[0].state == pb.FUNCTION_CAPABILITY_STATE_APPLYING, (
        "a wide worker that hides one unready group behind a ready one accepts "
        "work it cannot serve"
    )


def test_one_groups_rejection_is_the_workers_answer() -> None:
    accepted = pb.GoalReceipt(
        worker_session_id=SESSION, command_seq=1, goal_id="g", release_id=RELEASE,
        status=pb.GOAL_RECEIPT_STATUS_ACCEPTED, received_at_unix_ms=1,
    )
    rejected = pb.GoalReceipt(
        worker_session_id=SESSION, command_seq=1, goal_id="g", release_id=RELEASE,
        status=pb.GOAL_RECEIPT_STATUS_REJECTED, received_at_unix_ms=1,
        error_code=pb.LIFECYCLE_ERROR_CODE_UNKNOWN_FUNCTION,
    )
    answer = merge.worker_goal_receipt({0: accepted, 1: rejected})
    assert answer is not None
    assert answer.status == pb.GOAL_RECEIPT_STATUS_REJECTED
    assert merge.worker_goal_receipt({0: accepted, 1: None}) is None, (
        "an answer that speaks for a group which has not answered is a guess"
    )
    both = merge.worker_goal_receipt({0: accepted, 1: accepted})
    assert both is not None and both.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED


def test_a_missing_group_projection_is_never_papered_over() -> None:
    assert merge.merge_lifecycle_snapshots([_snapshot(state_seq=1), None]) is None
    assert merge.merge_lifecycle_snapshots([]) is None
