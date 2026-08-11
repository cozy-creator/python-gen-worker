"""pgw#767: OUTPUT_MODE_INLINE must not decide whether the result envelope's
blob actually exists.

The client's `Prefer: bytes=inline` hint is about MEDIA outputs. It reached
`ctx.save_bytes`, which under that hint returns bytes-in-memory WITHOUT
uploading — and `executor._serialize_output` kept only `asset.ref`. Every
result in (INLINE_RESULT_MAX_BYTES=64 KiB, _SAVE_BYTES_INLINE_THRESHOLD=4 MiB]
therefore shipped a `blob_ref` naming a blob no consumer could ever fetch.
That is the exact band, corrected from the issue's "any result >64 KiB":
above 4 MiB the inline shortcut declines and the real upload already ran.

Same real-codepath shape as test_p9_result_upload_metrics.py — a real hub
double plus a real local upload sink, so "was it uploaded" is answered by the
sink being hit, not by a mock assertion.
"""

from __future__ import annotations

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


def test_inline_dispatch_over_the_envelope_ceiling_still_really_uploads() -> None:
    """RED before the fix: the sink is never hit and the returned blob_ref
    names nothing. `large_usage` emits ~195 KiB — inside the affected band."""
    org_id = "00000000-0000-0000-0000-000000000001"
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(file_base_url=base_url) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-inline-big", attempt=1, function_name="large-usage",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
                org=org_id, capability_token="cap-token",
                output_mode=pb.OUTPUT_MODE_INLINE,
            ))
            res = conn.wait_for(is_result_for("r-inline-big")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert res.blob_ref, "over the envelope ceiling the result ships a ref"
            assert not res.inline
            assert DedupUploadSink.requests_seen, (
                "a returned blob_ref must name a blob that was REALLY uploaded — "
                "the inline media hint must not reach the result envelope"
            )
            path, body = DedupUploadSink.requests_seen[-1]
            assert path == MEDIA_UPLOADS_PATH
            assert body["size_bytes"] > 64 * 1024
    finally:
        httpd.shutdown()
        reset_upload_sink()


def test_save_bytes_still_inlines_media_for_the_client() -> None:
    """The fix is scoped to the envelope: the public media path keeps the
    `Prefer: bytes=inline` shortcut it exists to provide."""
    from gen_worker import RequestContext

    ctx = RequestContext(
        request_id="req-inline", owner="org",
        execution_hints={"output_format": "inline"},
    )
    asset = ctx.save_bytes("samples/small.bin", b"payload")
    assert asset.inline_bytes == b"payload"


def test_result_envelope_helper_ignores_the_inline_hint() -> None:
    """The envelope helper never takes the shortcut: it either uploads or
    refuses, but it must not hand back bytes-in-memory under a ref.

    Asserted without `pytest.raises(Exception)`, which would also pass on the
    `AttributeError` of the helper simply not existing.
    """
    from gen_worker import RequestContext

    ctx = RequestContext(
        request_id="req-envelope", owner="org",
        execution_hints={"output_format": "inline"},
    )
    assert hasattr(ctx, "_save_result_envelope")
    try:
        asset = ctx._save_result_envelope("results/req-envelope.msgpack", b"payload")
    except Exception:
        return  # refused with no upload endpoint configured — correct
    assert not asset.inline_bytes
