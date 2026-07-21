"""P9 (th#960/pgw#609 design table): inline <64KB vs blob_ref presigned PUT
by size alone, over a real hub-double + a real local media-upload HTTP sink
(dedup response — no S3 multipart scripting needed, matching
tests/test_media_upload_owner.py's real-codepath pattern). JobMetrics'
typed usage propagates regardless of which wire form the result took
(billing never scavenges the payload — pgw#512/#513 class).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, List, Tuple

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn


class _DedupUploadSink(BaseHTTPRequestHandler):
    """Real local stand-in for tensorhub's /api/v1/media/:owner/uploads —
    answers a dedup create so the test needs no S3 part PUT scripting, same
    approach as test_media_upload_owner.py."""

    requests_seen: ClassVar[List[Tuple[str, Dict[str, Any]]]] = []

    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests_seen.append((self.path, body))
        resp = json.dumps({
            "dedup": True, "ref": body.get("ref") or "", "filename": "out.msgpack",
            "blake3": body.get("blake3") or "", "size_bytes": body.get("size_bytes") or 0,
            "mime_type": "application/octet-stream", "media_id": "m1",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def _serve() -> Tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DedupUploadSink)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


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
    httpd, base_url = _serve()
    try:
        with hub_double(file_base_url=base_url) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-large", attempt=1, function_name="large-usage",
                input_payload=_payload(), tenant="acme", capability_token="cap-token"))
            res = conn.wait_for(is_result_for("r-large")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert res.blob_ref, "a >64KB output must ship blob_ref, not inline"
            assert not res.inline
            assert res.metrics.input_tokens == 4000
            assert res.metrics.input_cached_tokens == 100
            assert res.metrics.output_tokens == 9000
            assert _DedupUploadSink.requests_seen, "the real upload sink must have been hit"
            path, body = _DedupUploadSink.requests_seen[-1]
            assert path.startswith("/api/v1/media/acme/uploads")
            assert body["size_bytes"] > 64 * 1024
    finally:
        httpd.shutdown()
        _DedupUploadSink.requests_seen = []


def test_save_bytes_targets_token_bound_owner_not_dispatch_slug() -> None:
    """Absorbed from test_media_upload_owner.py (J19 run34): the capability
    token's `tenant` claim — NOT the dispatch-stamped ctx.owner (which can
    be a slug) — is the owner segment tensorhub's upload_media grant
    authorizes. A worker that used ctx.owner directly 403'd every upload
    whose owner resolved to a different org."""
    import base64
    import json as json_mod

    from gen_worker import RequestContext

    def _unsigned_jwt(claims: Dict[str, Any]) -> str:
        def seg(obj: Dict[str, Any]) -> str:
            raw = json_mod.dumps(obj).encode("utf-8")
            return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

        return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"

    owner_uuid = "019f4c33-f3a5-705b-9848-0b3b0863c416"
    httpd, base_url = _serve()
    try:
        token = _unsigned_jwt({"tenant": owner_uuid, "request_id": "req-run34"})
        # ctx.owner is the dispatch-stamped SLUG — the run34 failure mode.
        ctx = RequestContext(
            request_id="req-run34", owner="tensorhub",
            file_api_base_url=base_url, worker_capability_token=token,
        )
        asset = ctx.save_bytes("samples/pair-000.bin", b"payload")

        assert asset.ref
        assert _DedupUploadSink.requests_seen
        path, body = _DedupUploadSink.requests_seen[-1]
        assert path == f"/api/v1/media/{owner_uuid}/uploads", (
            "must ride the TOKEN-bound owner, never the dispatch slug"
        )
        assert body["ref"] == "samples/pair-000.bin"
    finally:
        httpd.shutdown()
        _DedupUploadSink.requests_seen = []
