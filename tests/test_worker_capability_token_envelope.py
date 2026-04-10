from __future__ import annotations

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.worker import _extract_worker_capability_token


def test_extract_worker_capability_token_prefers_canonical_field() -> None:
    req = pb.TaskExecutionRequest(
        request_id="req-1",
        file_token="legacy-token",
        worker_capability_token="canonical-token",
    )
    assert _extract_worker_capability_token(req) == "canonical-token"


def test_extract_worker_capability_token_falls_back_to_legacy_alias() -> None:
    req = pb.TaskExecutionRequest(
        request_id="req-2",
        file_token="legacy-token-only",
    )
    assert _extract_worker_capability_token(req) == "legacy-token-only"


def test_extract_worker_capability_token_works_for_realtime_command() -> None:
    cmd = pb.RealtimeOpenCommand(
        session_id="sess-1",
        function_name="realtime_fn",
        file_token="legacy-realtime-token",
    )
    assert _extract_worker_capability_token(cmd) == "legacy-realtime-token"
