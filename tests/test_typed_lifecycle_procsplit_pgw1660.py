"""#1660, end to end through the REAL process split.

The split is unconditional in production — every pod is a control parent plus a
compute child — so the Hello the hub actually sees is the parent's merged one.
This drives a real `ParentControl`, a real child subprocess and a real gRPC hub
double, and reads the pair off the wire.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gen_worker.lifecycle import snapshot_refusal
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.split import SplitHarness, isolated_postmortem  # noqa: F401

RELEASE = "rel-pgw1660-split"


def _ack(hello: pb.Hello) -> pb.HelloAck:
    now = int(time.time() * 1000)
    return pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        file_base_url="http://127.0.0.1:1/files",
        desired_residency=pb.DesiredResidency(
            generation=2, release_id=RELEASE, config_generation=4,
        ),
        desired_state_command=pb.DesiredStateCommand(
            worker_session_id=hello.worker_session_id,
            command_seq=2,
            goal_id="goal-split-pgw1660",
            release_id=RELEASE,
            config_generation=4,
            config_digest=b"digest",
            parameter_snapshot=b"\x80",
            issued_at_unix_ms=now,
            accept_by_unix_ms=now + 2_000,
            first_action_by_unix_ms=now + 600_000,
        ),
    )


@pytest.mark.usefixtures("isolated_postmortem")
def test_the_parents_merged_hello_carries_the_whole_pair(tmp_path: Path) -> None:
    harness = SplitHarness(
        tmp_path,
        hello_ack=_ack,
        extra_child_env={
            "WORKER_RELEASE_ID": RELEASE,
            "GEN_WORKER_CONFIG_SNAPSHOT_PATH": str(tmp_path / "runtime_config.msgpack"),
        },
    )
    try:
        conn = harness.scheduler.wait_connection(0, timeout=120.0)
        hello = conn.hello
        assert hello is not None
        assert hello.worker_session_id, "the parent states no session id"
        assert hello.HasField("lifecycle_snapshot"), (
            "the parent shipped worker_session_id with NO snapshot — the exact "
            "hello_session_id_missing_snapshot the whole fleet books today"
        )
        snapshot = hello.lifecycle_snapshot
        assert snapshot.worker_session_id == hello.worker_session_id, (
            "the parent owns the session id; a child-minted one inside the "
            "projection is a worker_session_mismatch and the hub drops it whole"
        )
        assert snapshot_refusal(snapshot, RELEASE) == ""
        assert {c.function_name for c in snapshot.capabilities}, (
            "typed with no capabilities starves every dispatch on this release"
        )

        receipt = conn.wait_for(
            lambda m: m.WhichOneof("msg") == "goal_receipt", timeout=120.0,
        ).goal_receipt
        assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED, receipt.detail
        assert receipt.worker_session_id == hello.worker_session_id, (
            "a receipt carrying the CHILD's session id is refused by the hub"
        )
        assert receipt.command_seq == 2
        assert receipt.goal_id == "goal-split-pgw1660"
    finally:
        harness.close()
