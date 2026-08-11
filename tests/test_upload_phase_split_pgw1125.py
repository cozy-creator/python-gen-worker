"""pgw#1125 / th#1795: `stage_ms.upload` splits into its three real legs.

MEASURED on the standing `master` stack (2026-08-11): a fast image request
waits 2917 ms, of which the GPU slot is held 12 ms (0.4%) and `finalize` is
2623 ms (89.9%) — 98.6% of that being `upload`. The upload does NOT scale with
payload size (n=88 spanning 46 KB -> 13 MB: intercept 2971 ms, slope
0.72 ms/MB, R² = 1.5e-6), so it is round-trip count, not bandwidth. Which of
the three legs owns the round trips decides which fix is worth building, and
the whole 2587 ms was reported under ONE key.

The test drives the REAL client (`presigned_upload_file`: real
`requests.Session`, real `PutPool`, real part PUTs) against a real localhost
server that implements all three legs. Each leg is given a distinct, disjoint
artificial delay, so a callback that attributed time to the wrong leg cannot
pass: the assertion is per-leg and the bands do not overlap.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple

from gen_worker.presigned_upload import presigned_upload_file
from gen_worker.stage_timing import StageTimer, reconciliation, stage_ms_for_metrics

CREATE_DELAY_S = 0.30
PUT_DELAY_S = 0.60
COMPLETE_DELAY_S = 0.90


class _ThreeLegHub(BaseHTTPRequestHandler):
    """create -> PUT part -> complete, each leg deliberately slow."""

    def log_message(self, *_a: Any) -> None:
        pass

    def _json(self, code: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        if self.path.endswith("/complete"):
            time.sleep(COMPLETE_DELAY_S)
            self._json(200, {
                "ref": "outputs/r/abc.bin", "media_id": "m1",
                "filename": "abc.bin", "blake3": "", "sha256": "",
                "size_bytes": 0, "mime_type": "application/octet-stream",
            })
            return
        time.sleep(CREATE_DELAY_S)
        self._json(201, {
            "upload_id": "u1",
            "part_urls": [base + "/put/1"],
            "part_size": 8 << 20,
            "total_parts": 1,
        })

    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        time.sleep(PUT_DELAY_S)
        self.send_response(200)
        self.send_header("ETag", '"deadbeef"')
        self.send_header("Content-Length", "0")
        self.end_headers()


def _serve() -> Tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ThreeLegHub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_the_three_legs_are_attributed_separately(tmp_path) -> None:
    src = tmp_path / "out.webp"
    src.write_bytes(b"x" * 4096)

    httpd, base = _serve()
    seen: List[Tuple[str, float]] = []
    try:
        result = presigned_upload_file(
            file_path=str(src),
            base_url=base,
            endpoint_path="/api/v1/media/o/uploads",
            headers={"Authorization": "Bearer t"},
            create_payload={"ref": "out.webp"},
            blake3_hex="0" * 64,
            size_bytes=4096,
            on_phase=lambda name, secs: seen.append((name, secs)),
        )
    finally:
        httpd.shutdown()

    assert result.dedup is False
    phases = dict(seen)
    assert set(phases) == {"create", "put", "complete"}
    # Disjoint bands: an off-by-one-leg attribution cannot satisfy all three.
    assert CREATE_DELAY_S <= phases["create"] < PUT_DELAY_S
    assert PUT_DELAY_S <= phases["put"] < COMPLETE_DELAY_S
    assert COMPLETE_DELAY_S <= phases["complete"] < COMPLETE_DELAY_S + PUT_DELAY_S


def test_a_failed_leg_still_reports_its_cost(tmp_path) -> None:
    """A leg that raised still spent its time; a callback that only fires on
    success would make the slow failure mode invisible — exactly the case an
    operator is looking at when the tail blows up."""
    src = tmp_path / "out.webp"
    src.write_bytes(b"x" * 16)

    class _RefuseComplete(_ThreeLegHub):
        def do_POST(self) -> None:  # noqa: N802
            if self.path.endswith("/complete"):
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                time.sleep(COMPLETE_DELAY_S)
                self._json(400, {"error": {"code": "bad_request"}})
                return
            super().do_POST()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RefuseComplete)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    seen: List[Tuple[str, float]] = []
    try:
        try:
            presigned_upload_file(
                file_path=str(src),
                base_url=base,
                endpoint_path="/api/v1/media/o/uploads",
                headers={"Authorization": "Bearer t"},
                create_payload={"ref": "out.webp"},
                blake3_hex="0" * 64,
                size_bytes=16,
                on_phase=lambda name, secs: seen.append((name, secs)),
            )
        except Exception:
            pass
    finally:
        httpd.shutdown()

    phases = dict(seen)
    assert "complete" in phases
    assert phases["complete"] >= COMPLETE_DELAY_S


def test_sub_phases_do_not_disturb_the_reconciliation_invariant() -> None:
    """The stage map's headline property is that measured stages plus
    `resid.unattributed` equal `total.runtime`. Sub-phases are a rollup like
    `denoise.step_mean`: reported, never summed. If they were charged as
    stages, `upload` would silently start meaning "upload minus its legs" and
    every number already measured against it would stop comparing."""
    timer = StageTimer()
    timer.handler_open()
    with timer.stage("upload"):
        time.sleep(0.02)
    timer.record_phase("upload", "create", 0.005)
    timer.record_phase("upload", "put", 0.004)
    timer.record_phase("upload", "complete", 0.008)
    timer.handler_close()

    out = stage_ms_for_metrics(timer, runtime_ms=out_runtime(timer))
    assert out["upload.create"] == 5
    assert out["upload.put"] == 4
    assert out["upload.complete"] == 8
    # `upload` still means the whole bracket, unchanged.
    assert out["upload"] >= 20
    attributed, total = reconciliation(out)
    assert attributed == total


def out_runtime(timer: StageTimer) -> int:
    return int(timer.snapshot().get("total.handler", 0))


# ---------------------------------------------------------------------------
# th#1795 direct-to-final: the store-enforced single PUT
# ---------------------------------------------------------------------------


class _DirectFinalHub(BaseHTTPRequestHandler):
    """The hub answering a create that declared sha256: one PUT grant into the
    final key, with the signed checksum header, and no part URLs at all."""

    creates: List[Dict[str, Any]] = []
    puts: List[Tuple[str, Dict[str, str], int]] = []

    def log_message(self, *_a: Any) -> None:
        pass

    def _json(self, code: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        if self.path.endswith("/complete"):
            self._json(200, {
                "ref": "outputs/r/abc.webp", "media_id": "m1",
                "filename": "abc.webp", "sha256": body.get("sha256", ""),
                "size_bytes": 0, "mime_type": "image/webp",
            })
            return
        type(self).creates.append(body)
        self._json(201, {
            "upload_id": "u1",
            "put_url": base + "/final/abc.webp",
            "put_headers": {"x-amz-checksum-sha256": "Zm9vYmFy"},
            "total_parts": 1,
        })

    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        type(self).puts.append((
            self.path,
            {k.lower(): v for k, v in self.headers.items()},
            len(payload),
        ))
        self.send_response(200)
        self.send_header("ETag", '"deadbeef"')
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_a_direct_final_grant_is_put_once_with_the_signed_headers_verbatim(tmp_path) -> None:
    """The whole win depends on two things the client must not get wrong: it
    PUTs ONCE (no multipart ceremony) and it sends the grant's headers
    verbatim. Editing or dropping `x-amz-checksum-sha256` is a 403 from the
    store, because the checksum is inside the signature."""
    src = tmp_path / "out.webp"
    src.write_bytes(b"y" * 2048)
    _DirectFinalHub.creates = []
    _DirectFinalHub.puts = []

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DirectFinalHub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    seen: List[Tuple[str, float]] = []
    try:
        result = presigned_upload_file(
            file_path=str(src),
            base_url=base,
            endpoint_path="/api/v1/media/o/uploads",
            headers={"Authorization": "Bearer t"},
            create_payload={"ref": "out.webp", "sha256": "a" * 64},
            blake3_hex="0" * 64,
            size_bytes=2048,
            on_phase=lambda name, secs: seen.append((name, secs)),
        )
    finally:
        httpd.shutdown()

    assert result.dedup is False
    assert len(_DirectFinalHub.puts) == 1, "the direct path PUTs exactly once"
    path, headers, sent = _DirectFinalHub.puts[0]
    assert path == "/final/abc.webp"
    assert sent == 2048
    assert headers.get("x-amz-checksum-sha256") == "Zm9vYmFy"
    # All three legs are still attributed on this path.
    assert set(dict(seen)) == {"create", "put", "complete"}


def test_a_real_serve_declares_the_sha256_of_the_bytes_it_uploaded() -> None:
    """The two-line unlock, on the REAL serve path: a real worker, a real
    dispatch, a real upload create against a real local sink. The digest was
    computed during the writes and then dropped on the floor; without it on the
    create the hub cannot name the final content-addressed key, so every save
    is staged, copied and deleted. RED before the declare: the create body has
    no `sha256` key at all."""
    import hashlib

    import msgspec

    from gen_worker.pb import worker_scheduler_pb2 as pb

    from harness.hub_double import hub_double, is_ready, is_result_for
    from harness.toy_endpoints import EchoIn
    from harness.upload_sink import DedupUploadSink, serve_upload_sink

    org_id = "00000000-0000-0000-0000-000000000001"
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(file_base_url=base_url) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-1125-sha", attempt=1, function_name="large-usage",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
                org=org_id, capability_token="cap-token",
                output_mode=pb.OUTPUT_MODE_INLINE,
            ))
            res = conn.wait_for(is_result_for("r-1125-sha")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert DedupUploadSink.requests_seen, "the real upload sink must have been hit"
            _path, body = DedupUploadSink.requests_seen[-1]
            declared = body.get("sha256")
            assert declared, (
                "the create must declare sha256 — without it the hub has no final "
                "key to presign into (th#1795 candidate 1(i))"
            )
            assert len(declared) == 64 and int(declared, 16) >= 0, declared
            assert body.get("blake3"), "blake3 stays declared; this adds, it does not replace"
            assert hashlib.sha256(b"").hexdigest() != declared
    finally:
        httpd.shutdown()
        DedupUploadSink.requests_seen = []
