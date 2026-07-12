"""gw#504: save_image on ANY job kind rides the MEDIA route — renewed token included.

J19 runs 48b-52d post-mortem: gen-worker 0.13.19 was wire-correct all along
(samples uploaded via POST /api/v1/media/{token tenant}/uploads carrying
request_id; the hub keyed outputs/<request-id>/<blake3> and stamped
producer_request_id — verified in the run-51/52d hub logs). The runs went red
because th#724 flipped hub-side output-owner attribution (invoked org →
invoker org) between runs, which the harness's ?tenant= query didn't follow.

This pins the worker half of the contract so a REAL worker regression can
never hide behind stack-side attribution changes again:

  * save_image on a producer (training) job — with repo-CAS checkpoint
    routing armed — creates a media session with request_id + job_id equal
    to the capability token binding, PUTs parts, completes on the media
    route, and returns the hub-keyed outputs/<rid>/<hash> ref.
  * the /commits route family is NEVER touched by an image save.
  * the same holds AFTER capability-token renewal (~80% TTL): the renewed
    token keeps the request/job claims and authenticates the media create.

The stand-in hub mirrors tensorhub's real capability gates for the media
create (internal/api/media_presigned_batch.go): request_id/job_id must equal
the token binding, blake3 must be 64-hex, and the stored key comes back as
outputs/<request_id>/<blake3><ext>.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import msgspec
import pytest

from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

OWNER_UUID = "019f4c33-f3a5-705b-9848-0b3b0863c416"
REQUEST_ID = "req-gw504"
JOB_ID = "job-gw504"
DEST_REPO = "acme/lora-out"
UPLOAD_ID = "med-1"

_MEDIA_CREATE_RE = re.compile(r"^/api/v1/media/([^/]+)/uploads$")
_MEDIA_COMPLETE_RE = re.compile(r"^/api/v1/media/([^/]+)/uploads/([^/]+)/complete$")
_PART_RE = re.compile(r"^/media-parts/(\d+)$")
_COMMITS_FAMILY_RE = re.compile(r"^/api/v1/repos/")
_RENEW_PATH = "/v1/worker/capability/renew"


def _unsigned_jwt(claims: Dict[str, Any]) -> str:
    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


def _jwt_claims(token: str) -> Dict[str, Any]:
    seg = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))


def _cap_claims(exp: float, iat: float) -> Dict[str, Any]:
    return {
        "cap_kind": "worker_capability",
        "tenant": OWNER_UUID,
        "request_id": REQUEST_ID,
        "job_id": JOB_ID,
        "exp": exp,
        "iat": iat,
        "grants": [
            {"do": "upload_media", "tenant": OWNER_UUID, "job": REQUEST_ID},
            {"do": "create_repo", "name": DEST_REPO},
            {"do": "create_checkpoint", "repo": DEST_REPO},
        ],
    }


class _MediaHubHandler(BaseHTTPRequestHandler):
    """Stand-in tensorhub: REAL media route table + the renewal endpoint.

    Mirrors the hub's capability gates on create; anything unmatched (the
    whole /commits family included) 404s and is recorded."""

    requests_seen: ClassVar[List[Tuple[str, str, str, Any]]] = []
    unmatched: ClassVar[List[Tuple[str, str]]] = []
    renewals: ClassVar[List[str]] = []  # renewed tokens minted, in order
    base_url: ClassVar[str] = ""

    def log_message(self, *args: Any) -> None:
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def _json(self, status: int, obj: Dict[str, Any]) -> None:
        resp = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _bearer(self) -> str:
        auth = str(self.headers.get("Authorization") or "")
        return auth.removeprefix("Bearer ").strip()

    def _404(self) -> None:
        type(self).unmatched.append((self.command, self.path))
        self._json(404, {"error": {"code": "not_found", "message": "no such route"}})

    def do_PUT(self) -> None:
        m = _PART_RE.match(self.path)
        raw = self._read_body()
        type(self).requests_seen.append((self.command, self.path, self._bearer(), len(raw)))
        if not m:
            self._404()
            return
        self.send_response(200)
        self.send_header("ETag", f'"etag-{m.group(1)}"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        token = self._bearer()
        type(self).requests_seen.append((self.command, self.path, token, body))

        if self.path == _RENEW_PATH:
            presented = str(body.get("capability_token") or "")
            claims = _jwt_claims(presented)
            if str(body.get("request_id") or "") != str(claims.get("request_id") or ""):
                self._json(403, {"error": "capability token is for a different job"})
                return
            # Re-mint: SAME claims, fresh far-future exp (mirrors the hub's
            # worker_capability_renew handler key-copy loop).
            now = time.time()
            renewed = _unsigned_jwt({**claims, "iat": now, "exp": now + 3600.0})
            type(self).renewals.append(renewed)
            self._json(200, {"capability_token": renewed, "expires_at_unix": int(now + 3600)})
            return

        m = _MEDIA_CREATE_RE.match(self.path)
        if m:
            claims = _jwt_claims(token)
            # tensorhub's capability gates (media_presigned_batch.go).
            if str(claims.get("cap_kind") or "") != "worker_capability":
                self._json(403, {"error": "invalid capability kind"})
                return
            bound_req = str(claims.get("request_id") or "")
            bound_job = str(claims.get("job_id") or "")
            if not bound_req or not bound_job:
                self._json(403, {"error": "missing request/job binding"})
                return
            # Hub defaulting: empty values inherit the binding; a non-empty
            # mismatch rejects.
            req_id = str(body.get("request_id") or "") or bound_req
            job_id = str(body.get("job_id") or "") or bound_job
            if req_id != bound_req or job_id != bound_job:
                self._json(403, {"error": "request_id/job_id not allowed"})
                return
            if m.group(1) != str(claims.get("tenant") or ""):
                self._json(403, {"error": "no upload_media grant for this owner/job"})
                return
            if not body.get("ref") \
                    or not re.fullmatch(r"[0-9a-f]{64}", str(body.get("blake3") or "")) \
                    or int(body.get("size_bytes") or 0) <= 0:
                self._json(400, {"error": "bad_request"})
                return
            self._json(201, {
                "upload_id": UPLOAD_ID,
                "part_urls": [f"{type(self).base_url}/media-parts/1"],
                "part_size": 64 * 1024 * 1024,
                "total_parts": 1,
                "expires_at": "2027-01-01T00:00:00Z",
                "upload_url": "",
            })
            return

        m = _MEDIA_COMPLETE_RE.match(self.path)
        if m:
            if m.group(2) != UPLOAD_ID:
                self._404()
                return
            parts = body.get("parts") or []
            if not parts or any(not p.get("etag") or not p.get("part_number") for p in parts):
                self._json(400, {"error": "bad parts"})
                return
            # Hub-side keying: outputs/<request_id>/<blake3><ext> — recover
            # the session's create payload from the recorded requests.
            create = next(
                b for mth, p, _, b in type(self).requests_seen
                if mth == "POST" and _MEDIA_CREATE_RE.match(p)
            )
            key = f"outputs/{create['request_id']}/{create['blake3']}.webp"
            self._json(200, {
                "ref": key,
                "filename": key,
                "media_id": "med-obj-1",
                "blake3": create["blake3"],
                "size_bytes": create["size_bytes"],
                "mime_type": "image/webp",
            })
            return

        self._404()


@pytest.fixture()
def hub_server():
    _MediaHubHandler.requests_seen = []
    _MediaHubHandler.unmatched = []
    _MediaHubHandler.renewals = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MediaHubHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    _MediaHubHandler.base_url = f"http://{host}:{port}"
    try:
        yield _MediaHubHandler.base_url
    finally:
        server.shutdown()


class _In(msgspec.Struct):
    destination_repo: str = DEST_REPO
    destination_repo_tags: List[str] = msgspec.field(default_factory=list)


class _Out(msgspec.Struct):
    ok: bool


def _run_job(hub_url: str, method, *, kind: str, cap_token: str) -> List[pb.WorkerMessage]:
    async def _go() -> List[pb.WorkerMessage]:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        spec = EndpointSpec(
            name="fn", method=method, kind=kind,
            payload_type=_In, output_mode="single",
        )
        ex = Executor([spec], _send)
        ex.file_base_url = hub_url
        await ex.handle_run_job(pb.RunJob(
            request_id=REQUEST_ID, attempt=1, function_name="fn",
            input_payload=msgspec.msgpack.encode(_In()),
            tenant=OWNER_UUID,
            capability_token=cap_token,
        ))
        job = ex.jobs[(REQUEST_ID, 1)]
        assert job.task is not None
        await job.task
        for _ in range(20):
            await asyncio.sleep(0)
        return sent

    return asyncio.run(_go())


def _save_image(ctx) -> Any:
    from PIL import Image

    return ctx.save_image(Image.new("RGB", (8, 8), "red"), "samples/step_000000001_0.webp")


def _assert_media_contract(seen, expected_bearer: Optional[str] = None, job_bound: bool = True) -> None:
    assert _MediaHubHandler.unmatched == []
    assert not [p for _, p, _, _ in seen if _COMMITS_FAMILY_RE.match(p)]
    creates = [(p, tok, b) for mth, p, tok, b in seen if mth == "POST" and _MEDIA_CREATE_RE.match(p)]
    assert len(creates) == 1
    path, tok, body = creates[0]
    assert path == f"/api/v1/media/{OWNER_UUID}/uploads"
    assert body["ref"] == "samples/step_000000001_0.webp"
    assert body["request_id"] == REQUEST_ID
    if job_bound:
        assert body["job_id"] == JOB_ID
    assert body["content_type"] == "image/webp"
    if expected_bearer is not None:
        assert tok == expected_bearer
    completes = [p for mth, p, _, _ in seen if mth == "POST" and _MEDIA_COMPLETE_RE.match(p)]
    assert completes == [f"/api/v1/media/{OWNER_UUID}/uploads/{UPLOAD_ID}/complete"]


def test_save_image_on_training_job_rides_media_route(hub_server: str) -> None:
    """Producer job with repo-CAS routing ARMED: image saves still ride the
    media route with the request/job binding — never /commits."""
    cap_token = _unsigned_jwt(_cap_claims(exp=4_102_444_800, iat=time.time()))

    def _train(ctx, payload: _In) -> _Out:
        assert ctx._execution_hints["kind"] == "training"
        assert ctx._repo_job_upload_scope() == ("acme", "lora-out", JOB_ID)
        asset = _save_image(ctx)
        assert asset.ref.startswith(f"outputs/{REQUEST_ID}/")
        assert asset.media_id == "med-obj-1"
        return _Out(ok=True)

    sent = _run_job(hub_server, _train, kind="training", cap_token=cap_token)
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK
    _assert_media_contract(_MediaHubHandler.requests_seen, expected_bearer=cap_token)
    assert _MediaHubHandler.renewals == []


def test_save_image_with_renewed_token_keeps_binding(hub_server: str) -> None:
    """Short-TTL token: the executor's renewal loop fires (~80% TTL), the
    handler waits for the swap, and the save under the RENEWED token still
    carries request_id/job_id and rides the media route only."""
    now = time.time()
    original = _unsigned_jwt(_cap_claims(exp=now + 4.0, iat=now))

    def _train(ctx, payload: _In) -> _Out:
        deadline = time.time() + 30.0
        while ctx._worker_capability_token == original and time.time() < deadline:
            time.sleep(0.1)
        renewed = ctx._worker_capability_token
        assert renewed != original, "capability token was never renewed"
        claims = _jwt_claims(renewed)
        assert claims["request_id"] == REQUEST_ID
        assert claims["job_id"] == JOB_ID
        assert claims["tenant"] == OWNER_UUID
        asset = _save_image(ctx)
        assert asset.ref.startswith(f"outputs/{REQUEST_ID}/")
        return _Out(ok=True)

    sent = _run_job(hub_server, _train, kind="training", cap_token=original)
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK
    assert len(_MediaHubHandler.renewals) >= 1
    _assert_media_contract(_MediaHubHandler.requests_seen, expected_bearer=_MediaHubHandler.renewals[0])


def test_save_image_on_inference_job_rides_media_route(hub_server: str) -> None:
    """Inference jobs (no repo-job scope) ride the identical media path."""
    cap_token = _unsigned_jwt(_cap_claims(exp=4_102_444_800, iat=time.time()))

    def _infer(ctx, payload: _In) -> _Out:
        assert ctx._repo_job_upload_scope() is None
        asset = _save_image(ctx)
        assert asset.ref.startswith(f"outputs/{REQUEST_ID}/")
        return _Out(ok=True)

    sent = _run_job(hub_server, _infer, kind="inference", cap_token=cap_token)
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK
    _assert_media_contract(_MediaHubHandler.requests_seen, expected_bearer=cap_token, job_bound=False)
