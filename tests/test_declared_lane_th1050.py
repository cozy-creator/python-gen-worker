"""th#1050 SDK lane contract, real codepaths over the hub-double gRPC wire.

A toy endpoint DECLARES `handles=["fp8-w8a8-dynamic"]` and branches on
ctx.lane. Proven here:
  1. Discovery manifest emits the `handles` block (hub parse input).
  2. A dispatched declared lane executes with NO laddered binding rebind —
     ctx.lane reports it, the handler branches on it, and JobMetrics.lane
     reports the SAME value (post-degrade consistency).
  3. Undeclared dispatch (no RunJob.lane) = today's behavior exactly:
     ctx.lane reports bf16-w16a16+eager and the reference branch runs.
  4. An instructed lane the endpoint does NOT declare still refuses TYPED
     (declaration is divergence marking, never a blanket bypass).
  5. Decorator validation rejects coarse/unknown/execution-suffixed tokens.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker.api.decorators import endpoint
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn


def _payload() -> bytes:
    return msgspec.msgpack.encode(EchoIn(text="x"))


def _decode(res: "pb.JobResult") -> dict:
    return msgspec.msgpack.decode(res.inline)


def test_manifest_emits_handles_block() -> None:
    from gen_worker.discovery.discover import _extract_entries
    from harness import toy_endpoints

    (fn,) = _extract_entries(toy_endpoints.LaneAwareEndpoint, "harness.toy_endpoints")
    assert fn["handles"] == ["fp8-w8a8-dynamic"]
    plain = _extract_entries(toy_endpoints.Basics, "harness.toy_endpoints")
    assert all("handles" not in f for f in plain)


def test_declared_lane_executes_and_ctx_lane_reports_it() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-lane", attempt=1, function_name="lane-echo",
            input_payload=_payload(), lane="fp8-w8a8-dynamic+eager"))
        res = conn.wait_for(is_result_for("r-lane")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        body = _decode(res)
        # ctx.lane carried the executing lane into the author branch...
        assert body["response"] == "author-kernel:fp8-w8a8-dynamic+eager"
        # ...and JobMetrics.lane reports the SAME executing truth.
        assert res.metrics.lane == "fp8-w8a8-dynamic+eager"


def test_undeclared_dispatch_is_todays_behavior() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-base", attempt=1, function_name="lane-echo",
            input_payload=_payload()))
        res = conn.wait_for(is_result_for("r-base")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        assert _decode(res)["response"] == "reference:bf16-w16a16+eager"
        assert res.metrics.lane == "bf16-w16a16+eager"


def test_unhandled_lane_still_refuses_typed() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-nope", attempt=1, function_name="lane-echo",
            input_payload=_payload(), lane="nvfp4-w4a4-static+eager"))
        res = conn.wait_for(is_result_for("r-nope")).job_result
        assert res.status == pb.JOB_STATUS_INVALID
        assert "lane_unavailable" in res.safe_message
        assert "nvfp4-w4a4-static" in res.safe_message


def test_decorator_rejects_bad_handles_tokens() -> None:
    class _In(msgspec.Struct):
        text: str = ""

    class _Out(msgspec.Struct):
        response: str = ""

    def _make(tokens):
        @endpoint(handles=tokens)
        class _E:
            def go(self, ctx, data: _In) -> _Out:  # pragma: no cover
                return _Out()
        return _E

    for bad, needle in (
        (["fp8"], "coarse family"),
        (["fp8-w8a8-dynamic+compiled"], "execution axis"),
        (["int8-w8a8"], "not a known lane body"),
        (["fp8-w8a8-dynamic", "fp8-w8a8-dynamic"], "repeats"),
    ):
        with pytest.raises(ValueError, match=needle):
            _make(bad)
    # A valid declaration normalizes to a tuple of bodies.
    cls = _make(["FP8-W8A8-DYNAMIC"])
    from gen_worker.api.decorators import ATTR

    assert getattr(cls, ATTR).handles == ("fp8-w8a8-dynamic",)
