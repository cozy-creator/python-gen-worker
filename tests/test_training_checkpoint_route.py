"""gw#453 + gw#471: checkpoint saves ride tensorhub's REAL /commits API.

gw#453 (J19 run41): executor-built training contexts must arm repo-CAS
checkpoint routing (destination_repo + job_id hints) so checkpoints never
fall back to the 256 MiB-capped media route.

gw#471 (J19 run43): the checkpoint publish path spoke a phantom
POST /repos/:o/:r/revisions dialect that tensorhub deleted in th#514/#515
(2026-07-03). The gw#453 version of this test missed it because its stand-in
hub accepted ANY path. This stand-in serves ONLY tensorhub's real route
table and 404s everything else — that strictness is the regression guard.

Real flow asserted end to end (mirrors internal/api/repo_commits.go):

    POST /api/v1/repos/{o}/{r}/commits          {operations:[{add,path,blake3,size_bytes}],...}
      -> {revision_id, uploads:[{upload_id, part_urls, part_size, complete_url, ...}]}
    PUT  <part_urls>                            (multipart body bytes)
    POST .../commits/{rev}/uploads/{id}/complete {parts:[{part_number, etag}]}
    POST .../commits/{rev}/finalize             (no body)

plus repo-absent first-publish (the /commits body carries the repo spec for
server-side auto-create under the job's create_repo grant — the client never
calls a create-repo endpoint), and the gw#471 scope-add: upload failures emit
a typed `request.warning` {code: artifact_upload_failed} event before raising.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, List, Tuple

import msgspec
import pytest

from gen_worker.executor import Executor, _producer_destination_repo
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.request_context import RequestContext, TrainingContext

OWNER_UUID = "019f4c33-f3a5-705b-9848-0b3b0863c416"
JOB_ID = "job-run43"
DEST_REPO = "acme/lora-out"
MEDIA_MAX_BYTES = 256 * 1024 * 1024  # tensorhub media route per-file cap
REVISION_ID = "rev-471"
UPLOAD_ID = "u-1"
PART_SIZE = 64 * 1024 * 1024

# Exact key set tensorhub's createCommitRequest decoder accepts
# (DisallowUnknownFields — internal/api/repo_commits.go). Unknown keys 400.
_COMMIT_BODY_KEYS = {
    "operations", "tags", "mode", "message", "display_label", "flavor",
    "flavors", "default_flavor", "dtype", "file_layout", "file_type",
    "metadata", "provenance", "lineage", "auto_create_external_parent",
    "kind", "library_name", "model_family", "class_name", "adapter_for_family",
}

_COMMITS_RE = re.compile(r"^/api/v1/repos/([^/]+)/([^/]+)/commits$")
_COMPLETE_RE = re.compile(
    r"^/api/v1/repos/([^/]+)/([^/]+)/commits/([^/]+)/uploads/([^/]+)/complete$")
_FINALIZE_RE = re.compile(r"^/api/v1/repos/([^/]+)/([^/]+)/commits/([^/]+)/finalize$")
_PART_RE = re.compile(r"^/parts/(\d+)$")
_MEDIA_RE = re.compile(r"^/api/v1/media/([^/]+)/uploads$")


def _unsigned_jwt(claims: Dict[str, Any]) -> str:
    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


class _StrictHubHandler(BaseHTTPRequestHandler):
    """Stand-in tensorhub that serves ONLY the real route table.

    Every request is recorded as (method, path, auth, body). Anything that
    doesn't match a real route 404s AND lands in `unmatched` — tests assert
    that list is empty, so a client speaking a phantom dialect fails loudly.
    """

    requests_seen: ClassVar[List[Tuple[str, str, str, Any]]] = []
    unmatched: ClassVar[List[Tuple[str, str]]] = []
    part_bytes: ClassVar[Dict[int, int]] = {}
    base_url: ClassVar[str] = ""

    def log_message(self, *args: Any) -> None:
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        remaining = length
        chunks: List[bytes] = []
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _json(self, status: int, obj: Dict[str, Any]) -> None:
        resp = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _record(self, body: Any) -> None:
        type(self).requests_seen.append(
            (self.command, self.path, str(self.headers.get("Authorization") or ""), body)
        )

    def _404(self) -> None:
        type(self).unmatched.append((self.command, self.path))
        self._json(404, {"error": {"code": "not_found", "message": "no such route"}})

    def do_PUT(self) -> None:
        m = _PART_RE.match(self.path)
        raw = self._read_body()
        self._record(len(raw))
        if not m:
            self._404()
            return
        idx = int(m.group(1))
        type(self).part_bytes[idx] = len(raw)
        self.send_response(200)
        self.send_header("ETag", f'"etag-{idx}"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        self._record(body)

        m = _COMMITS_RE.match(self.path)
        if m:
            unknown = set(body) - _COMMIT_BODY_KEYS
            if unknown:
                self._json(400, {"error": {"code": "bad_request",
                                           "message": f"unknown fields: {sorted(unknown)}"}})
                return
            ops = body.get("operations") or []
            if not ops:
                self._json(400, {"error": {"code": "bad_request",
                                           "message": "operations must be non-empty"}})
                return
            op = ops[0]
            if op.get("type") != "add" or not op.get("path") \
                    or not re.fullmatch(r"[0-9a-f]{64}", str(op.get("blake3") or "")) \
                    or int(op.get("size_bytes") or 0) <= 0:
                self._json(400, {"error": {"code": "bad_request",
                                           "message": f"malformed add operation: {op}"}})
                return
            size = int(op["size_bytes"])
            total_parts = max(1, (size + PART_SIZE - 1) // PART_SIZE)
            self._json(201, {
                "revision_id": REVISION_ID,
                "uploads": [{
                    "path": op["path"],
                    "blake3": op["blake3"],
                    "exists": False,
                    "upload_id": UPLOAD_ID,
                    "part_urls": [f"{type(self).base_url}/parts/{i + 1}" for i in range(total_parts)],
                    "part_size": PART_SIZE,
                    "total_parts": total_parts,
                    "expires_at": "2027-01-01T00:00:00Z",
                    "complete_url": "",
                    "size_bytes": size,
                }],
                "deletions": [], "copies": [], "tags": [],
                "mode": body.get("mode") or "merge",
            })
            return

        m = _COMPLETE_RE.match(self.path)
        if m:
            if m.group(3) != REVISION_ID or m.group(4) != UPLOAD_ID:
                self._404()
                return
            parts = body.get("parts") or []
            if not parts or any(not p.get("etag") or not p.get("part_number") for p in parts):
                self._json(400, {"error": {"code": "bad_request", "message": f"bad parts: {parts}"}})
                return
            self._json(200, {"ok": True})
            return

        m = _FINALIZE_RE.match(self.path)
        if m:
            if m.group(3) != REVISION_ID:
                self._404()
                return
            self._json(200, {"checkpoint": {
                "checkpoint_id": "ck-471",
                "snapshot_digest": "blake3:" + "ab" * 32,
            }})
            return

        m = _MEDIA_RE.match(self.path)
        if m:
            self._json(200, {
                "dedup": True,
                "ref": body.get("ref") or "",
                "blake3": body.get("blake3") or "",
                "size_bytes": body.get("size_bytes") or 0,
                "mime_type": "application/octet-stream",
            })
            return

        self._404()


@pytest.fixture()
def hub_server():
    _StrictHubHandler.requests_seen = []
    _StrictHubHandler.unmatched = []
    _StrictHubHandler.part_bytes = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StrictHubHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    _StrictHubHandler.base_url = f"http://{host}:{port}"
    try:
        yield _StrictHubHandler.base_url
    finally:
        server.shutdown()


class _In(msgspec.Struct):
    destination_repo: str = DEST_REPO
    destination_repo_tags: List[str] = msgspec.field(default_factory=list)


class _Out(msgspec.Struct):
    ok: bool


def _cap_token() -> str:
    return _unsigned_jwt(
        {
            "cap_kind": "worker_capability",
            "tenant": OWNER_UUID,
            "request_id": "req-run43",
            "job_id": JOB_ID,
            "exp": 4_102_444_800,  # far future: renewal task stays asleep
            "grants": [
                {"do": "upload_media", "tenant": OWNER_UUID, "job": "req-run43"},
                {"do": "create_repo", "name": DEST_REPO},
                {"do": "create_checkpoint", "repo": DEST_REPO},
            ],
        }
    )


def _run_training_job(hub_url: str, method) -> List[pb.WorkerMessage]:
    async def _go() -> List[pb.WorkerMessage]:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        spec = EndpointSpec(
            name="train", method=method, kind="training",
            payload_type=_In, output_mode="single",
        )
        ex = Executor([spec], _send)
        ex.file_base_url = hub_url
        await ex.handle_run_job(pb.RunJob(
            request_id="req-run43", attempt=1, function_name="train",
            input_payload=msgspec.msgpack.encode(_In()),
            tenant=OWNER_UUID,
            capability_token=_cap_token(),
        ))
        job = ex.jobs[("req-run43", 1)]
        assert job.task is not None
        await job.task
        for _ in range(20):
            await asyncio.sleep(0)
        return sent

    return asyncio.run(_go())


def _commit_calls(seen) -> List[Tuple[str, str, str, Any]]:
    return [r for r in seen if _COMMITS_RE.match(r[1])]


def test_training_checkpoint_publishes_via_commits(hub_server: str, tmp_path) -> None:
    """>256MiB checkpoint rides the FULL real sequence: create commit ->
    part PUTs -> complete -> finalize. Samples stay on the media route."""
    lora = tmp_path / "lora_000000500.safetensors"
    # Bigger than the media route's per-file cap: the exact run41/run43 payload class.
    size = MEDIA_MAX_BYTES + 1024 * 1024
    with open(lora, "wb") as f:
        f.truncate(size)

    def _train(ctx, payload: _In) -> _Out:
        assert ctx._execution_hints["kind"] == "training"
        assert ctx._execution_hints["destination_repo"] == DEST_REPO
        assert ctx._repo_job_upload_scope() == ("acme", "lora-out", JOB_ID)
        out = ctx.save_checkpoint(
            "checkpoints/lora_000000500.safetensors", str(lora),
            step_number=500, output_kind="lora",
        )
        assert out.size_bytes == size
        assert out.blake3
        assert out.blob_digest == f"blake3:{out.blake3}"
        # Samples are media outputs: they must stay on the media route.
        ctx.save_bytes("samples/sample_000000500.bin", b"sample-bytes")
        return _Out(ok=True)

    sent = _run_training_job(hub_server, _train)
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK

    token = _cap_token()
    seen = _StrictHubHandler.requests_seen

    # THE regression guard: nothing hit a route tensorhub doesn't serve.
    assert _StrictHubHandler.unmatched == []

    # 1. Exactly one commit-create on the destination repo, cap-token auth,
    #    with the full add operation declared up front.
    commits = _commit_calls(seen)
    assert [(m, p) for m, p, _, _ in commits] == [("POST", "/api/v1/repos/acme/lora-out/commits")]
    _, _, auth, body = commits[0]
    assert auth == f"Bearer {token}"
    assert body["mode"] == "merge"
    op = body["operations"][0]
    assert op["path"] == "checkpoints/lora_000000500.safetensors"
    assert op["size_bytes"] == size
    # Worker-addable provenance stamp rode the body.
    assert body["provenance"]["step_number"] == 500

    # 2. Every byte was PUT to the presigned part URLs.
    assert sum(_StrictHubHandler.part_bytes.values()) == size
    assert len(_StrictHubHandler.part_bytes) == (size + PART_SIZE - 1) // PART_SIZE

    # 3. complete (with ETags) then finalize (no body), in order.
    posts = [(p, b) for m, p, _, b in seen if m == "POST"]
    complete_idx = [i for i, (p, _) in enumerate(posts) if _COMPLETE_RE.match(p)]
    finalize_idx = [i for i, (p, _) in enumerate(posts) if _FINALIZE_RE.match(p)]
    assert len(complete_idx) == 1 and len(finalize_idx) == 1
    assert complete_idx[0] < finalize_idx[0]
    assert posts[complete_idx[0]][1]["parts"][0]["etag"] == "etag-1"
    assert posts[finalize_idx[0]][1] == {}

    # 4. The only media call is the sample asset, against the token-bound owner.
    media_calls = [(p, b) for m, p, _, b in seen if _MEDIA_RE.match(p)]
    assert [(p, b["ref"]) for p, b in media_calls] == [
        (f"/api/v1/media/{OWNER_UUID}/uploads", "samples/sample_000000500.bin"),
    ]


def test_first_publish_carries_repo_spec_no_precreate(hub_server: str, tmp_path) -> None:
    """Repo-absent first publish: the client sends ONE /commits request whose
    body carries the repo spec (server auto-creates under the create_repo
    grant) — it never calls a create-repo endpoint first."""
    src = tmp_path / "lora.safetensors"
    src.write_bytes(b"weights-bytes")

    def _train(ctx, payload: _In) -> _Out:
        ctx.set_repo_spec(
            kind="adapter",
            library_name="diffusers",
            model_family="qwen_image_edit",
            adapter_for_family="qwen_image_edit",
        )
        ctx.save_checkpoint("lora.safetensors", str(src), step_number=250, output_kind="lora")
        return _Out(ok=True)

    sent = _run_training_job(hub_server, _train)
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK
    assert _StrictHubHandler.unmatched == []

    commits = _commit_calls(_StrictHubHandler.requests_seen)
    assert len(commits) == 1
    body = commits[0][3]
    assert body["kind"] == "adapter"
    assert body["library_name"] == "diffusers"
    assert body["model_family"] == "qwen_image_edit"
    assert body["adapter_for_family"] == "qwen_image_edit"
    assert body["provenance"]["step_number"] == 250


def test_training_without_destination_fails_loudly_not_media(hub_server: str, tmp_path) -> None:
    """No destination bindings → save_checkpoint must raise (the designed
    fail-loud), never silently fall back to the media route."""
    src = tmp_path / "lora.safetensors"
    src.write_bytes(b"weights")

    class _NoDest(msgspec.Struct):
        pass

    def _train(ctx, payload) -> _Out:
        with pytest.raises(RuntimeError, match="repo job scope"):
            ctx.save_checkpoint("checkpoints/lora.safetensors", str(src))
        return _Out(ok=True)

    async def _go() -> List[pb.WorkerMessage]:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        spec = EndpointSpec(
            name="train", method=_train, kind="training",
            payload_type=_NoDest, output_mode="single",
        )
        ex = Executor([spec], _send)
        ex.file_base_url = hub_server
        await ex.handle_run_job(pb.RunJob(
            request_id="r2", attempt=1, function_name="train",
            input_payload=msgspec.msgpack.encode(_NoDest()),
            tenant=OWNER_UUID, capability_token=_cap_token(),
        ))
        job = ex.jobs[("r2", 1)]
        await job.task
        return sent

    sent = asyncio.run(_go())
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK
    assert not [r for r in _StrictHubHandler.requests_seen if _MEDIA_RE.match(r[1])]


# ---------------------------------------------------------------------------
# gw#471 scope-add: upload failures must be REPORTED (typed request.warning),
# not just logged — the phantom-route breakage was invisible for a dozen runs.
# ---------------------------------------------------------------------------


class _DeadHubHandler(BaseHTTPRequestHandler):
    """A hub that 404s everything — the phantom-route failure class."""

    def log_message(self, *args: Any) -> None:
        pass

    def _die(self) -> None:
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        resp = b'{"error":{"code":"not_found"}}'
        self.send_response(404)
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    do_POST = _die
    do_PUT = _die


@pytest.fixture()
def dead_hub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DeadHubHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()


def _ctx(dead_hub: str, events: List[Dict[str, Any]], *, hints: Dict[str, Any],
         cls: type = TrainingContext) -> RequestContext:
    return cls(
        request_id="req-warn",
        job_id=JOB_ID,
        emitter=events.append,
        owner=OWNER_UUID,
        file_api_base_url=dead_hub,
        worker_capability_token=_cap_token(),
        execution_hints=hints,
    )


def test_checkpoint_upload_failure_emits_typed_warning(dead_hub: str, tmp_path) -> None:
    src = tmp_path / "lora.safetensors"
    src.write_bytes(b"weights")
    events: List[Dict[str, Any]] = []
    ctx = _ctx(dead_hub, events, hints={
        "kind": "training", "destination_repo": DEST_REPO, "job_id": JOB_ID,
    })

    with pytest.raises(Exception, match="404"):
        ctx.save_checkpoint("checkpoints/lora.safetensors", str(src), step_number=250)

    warnings = [e for e in events if e.get("type") == "request.warning"]
    assert len(warnings) == 1
    payload = warnings[0]["payload"]
    assert payload["code"] == "artifact_upload_failed"
    assert payload["kind"] == "checkpoint"
    assert payload["ref"] == "checkpoints/lora.safetensors"
    assert payload["step_number"] == 250
    assert payload["attempt"] == 1
    assert "404" in payload["error"] and len(payload["error"]) <= 500


def test_media_upload_failure_emits_typed_warning(dead_hub: str) -> None:
    events: List[Dict[str, Any]] = []
    ctx = _ctx(dead_hub, events, hints={"kind": "inference"}, cls=RequestContext)

    with pytest.raises(Exception):
        ctx.save_bytes("samples/sample.bin", b"sample-bytes")

    warnings = [e for e in events if e.get("type") == "request.warning"]
    assert len(warnings) == 1
    payload = warnings[0]["payload"]
    assert payload["code"] == "artifact_upload_failed"
    assert payload["kind"] == "sample"
    assert payload["ref"] == "samples/sample.bin"


@pytest.mark.parametrize(
    ("info", "payload_attr", "want"),
    [
        ({"ref": "acme/lora-out"}, "", "acme/lora-out"),
        ({"ref": "acme/lora-out:latest#fp8"}, "", "acme/lora-out"),
        ({"repo": "acme/other"}, "", "acme/other"),
        ({}, "acme/flat", "acme/flat"),
        ({}, "acme/flat:tag", "acme/flat"),
        ({}, "", ""),
    ],
)
def test_producer_destination_repo_dual_form(info, payload_attr, want) -> None:
    class _P(msgspec.Struct):
        destination_repo: str = ""

    assert _producer_destination_repo(_P(destination_repo=payload_attr), info) == want
