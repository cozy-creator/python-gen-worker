from __future__ import annotations

import asyncio

import pytest

from gen_worker.intent_registry import IntentRegistry, UnreportedIntentWait
from gen_worker.pb import worker_scheduler_pb2 as pb


def test_long_unreported_await_fails_closed_and_replays_typed_failure() -> None:
    registry = IntentRegistry(
        "release-1",
        ["echo"],
        unreported_wait_timeout_s=0.01,
    )
    receipt = registry.apply_command(
        pb.DesiredStateCommand(
            worker_session_id=registry.worker_session_id,
            command_seq=1,
            goal_id="goal-1",
            release_id="release-1",
            intents=[
                pb.DesiredIntent(
                    intent_id="materialize-1",
                    kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
                    cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
                    ref="owner/model:latest",
                    snapshot_digest=b"blake3:abc",
                    mandatory=True,
                )
            ],
            mandatory=True,
        )
    )
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    async def run() -> None:
        never = asyncio.Event()
        with pytest.raises(UnreportedIntentWait):
            await registry.guard_await(
                "materialize-1",
                never.wait(),
                operation="test materialization",
            )

    asyncio.run(run())

    state = next(item for item in registry.snapshot().intents if item.intent_id == "materialize-1")
    assert registry.protocol_rejected
    assert state.status == pb.LIFECYCLE_INTENT_STATUS_FAILED
    assert state.error_code == pb.LIFECYCLE_ERROR_CODE_PROTOCOL_UNREPORTED_WAIT
