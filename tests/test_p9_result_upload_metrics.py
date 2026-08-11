"""P9 (th#960/pgw#609 design table): inline <64KB vs blob_ref presigned PUT
by size alone, over a real hub-double + the shared media-upload HTTP sink
(`harness.upload_sink` — a dedup response over tensorhub's real route table,
so no S3 multipart scripting is needed). JobMetrics' typed usage propagates
regardless of which wire form the result took (billing never scavenges the
payload — pgw#512/#513 class).

The owner-segment test this file used to carry (J19 run34: the capability
token's `tenant` claim, not the dispatch slug) is absorbed into
`test_media_upload_orgless_pgw1138.py`: th#1722 §C removed the segment
entirely, so there is no owner left to get wrong.
"""

from __future__ import annotations

from typing import Tuple

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn
from harness.upload_sink import (
    MEDIA_UPLOADS_PATH,
    DedupUploadSink,
    reset_upload_sink,
    serve_upload_sink,
)


def _serve() -> Tuple[object, str]:
    return serve_upload_sink()


def _payload() -> bytes:
    return msgspec.msgpack.encode(EchoIn(text="x"))


def test_small_output_ships_inline_with_typed_usage() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-small", attempt=1, function_name="small-usage",
            input_payload=_payload()))
        res = conn.wait_for(is_result_for("r-small")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert res.inline
        assert not res.blob_ref
        assert res.metrics.input_tokens == 12
        assert res.metrics.input_cached_tokens == 2
        assert res.metrics.output_tokens == 5


def test_large_output_ships_blob_ref_with_typed_usage_intact() -> None:
    """pgw#512/#513 class: a >64KB output goes blob_ref (executor's
    INLINE_RESULT_MAX_BYTES) via a real presigned upload round trip —
    JobMetrics' token usage is computed from the raw handler output BEFORE
    that serialization decision, so it survives regardless of wire form."""
    org_id = "00000000-0000-0000-0000-000000000001"
    httpd, base_url = _serve()
    try:
        with hub_double(file_base_url=base_url) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(
                run_job=pb.RunJob(
                    request_id="r-large",
                    attempt=1,
                    function_name="large-usage",
                    input_payload=_payload(),
                    org=org_id,
                    capability_token="cap-token",
                )
            )
            res = conn.wait_for(is_result_for("r-large")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert res.blob_ref, "a >64KB output must ship blob_ref, not inline"
            assert not res.inline
            assert res.metrics.input_tokens == 4000
            assert res.metrics.input_cached_tokens == 100
            assert res.metrics.output_tokens == 9000
            assert DedupUploadSink.requests_seen, "the real upload sink must have been hit"
            path, body = DedupUploadSink.requests_seen[-1]
            assert path == MEDIA_UPLOADS_PATH
            assert body["size_bytes"] > 64 * 1024
    finally:
        httpd.shutdown()
        reset_upload_sink()
