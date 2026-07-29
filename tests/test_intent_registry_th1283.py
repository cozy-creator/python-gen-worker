"""th#1283: per-intent mandatory — advisory mistakes decline, mandatory work
still fails closed, and a command-level conflict without new mandatory work
rejects without latching the process."""

from __future__ import annotations

from gen_worker.intent_registry import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb


def _base_command(registry: IntentRegistry, **kwargs) -> pb.DesiredStateCommand:
    defaults = dict(
        worker_session_id=registry.worker_session_id,
        command_seq=1,
        goal_id="goal-1",
        release_id="release-1",
    )
    defaults.update(kwargs)
    return pb.DesiredStateCommand(**defaults)


def _materialize(intent_id: str, *, mandatory: bool, digest: bytes = b"blake3:abc") -> pb.DesiredIntent:
    return pb.DesiredIntent(
        intent_id=intent_id,
        kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
        cause=(
            pb.DESIRED_INTENT_CAUSE_REQUEST if mandatory else pb.DESIRED_INTENT_CAUSE_PREPOSITION
        ),
        ref="owner/model:latest",
        snapshot_digest=digest,
        mandatory=mandatory,
    )


def test_advisory_intent_errors_decline_without_latching() -> None:
    """Symptom 3, worker half: an unknown-function FUNCTION_READY declared
    advisory is declined typed while the rest of the command is accepted and
    the process keeps serving."""
    registry = IntentRegistry("release-1", ["generate"])
    command = _base_command(
        registry,
        intents=[
            _materialize("materialize-1", mandatory=False),
            pb.DesiredIntent(
                intent_id="function-ghost",
                kind=pb.DESIRED_INTENT_KIND_FUNCTION_READY,
                cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
                function_name="ghost_fn",
                binding_digest=b"binding",
            ),
        ],
    )
    receipt = registry.apply_command(command)
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED
    assert not registry.protocol_rejected
    assert [r.intent_id for r in receipt.rejections] == ["function-ghost"]
    assert receipt.rejections[0].error_code == pb.LIFECYCLE_ERROR_CODE_UNKNOWN_FUNCTION

    states = {state.intent_id: state for state in registry.snapshot().intents}
    assert states["materialize-1"].status == pb.LIFECYCLE_INTENT_STATUS_ACCEPTED
    assert states["function-ghost"].status == pb.LIFECYCLE_INTENT_STATUS_FAILED
    assert states["function-ghost"].error_code == pb.LIFECYCLE_ERROR_CODE_UNKNOWN_FUNCTION
    # The declined intent never impersonates accepted desired work.
    assert registry.intent_id(pb.DESIRED_INTENT_KIND_FUNCTION_READY, function_name="ghost_fn") == ""

    # A byte-identical resend replays the original receipt.
    replay = registry.apply_command(command)
    assert replay.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED
    assert [r.intent_id for r in replay.rejections] == ["function-ghost"]


def test_command_seq_conflict_without_new_mandatory_work_fails_open() -> None:
    """Symptom 1, worker half: a goal_id drift at a fixed command_seq whose
    intents are all already registered rejects the resend but must NOT latch
    the process."""
    registry = IntentRegistry("release-1", ["generate"])
    first = _base_command(
        registry,
        intents=[_materialize("materialize-1", mandatory=True)],
    )
    assert registry.apply_command(first).status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    drifted = pb.DesiredStateCommand()
    drifted.CopyFrom(first)
    drifted.goal_id = "goal-1-drifted"
    receipt = registry.apply_command(drifted)
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_REJECTED
    assert receipt.error_code == pb.LIFECYCLE_ERROR_CODE_COMMAND_SEQ_CONFLICT
    assert not registry.protocol_rejected
    # The previously accepted goal survives untouched.
    states = {state.intent_id: state for state in registry.snapshot().intents}
    assert states["materialize-1"].status == pb.LIFECYCLE_INTENT_STATUS_ACCEPTED


def test_mandatory_intent_error_still_fails_closed() -> None:
    """A genuinely mandatory intent (request-blocking MATERIALIZE) with a
    validation error must still latch the process."""
    registry = IntentRegistry("release-1", ["generate"])
    receipt = registry.apply_command(
        _base_command(
            registry,
            intents=[_materialize("materialize-1", mandatory=True, digest=b"")],
        )
    )
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_REJECTED
    assert receipt.error_code == pb.LIFECYCLE_ERROR_CODE_SNAPSHOT_IDENTITY_MISSING
    assert registry.protocol_rejected
