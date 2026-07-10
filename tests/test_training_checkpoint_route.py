"""gw#453: executor-built training contexts arm repo-CAS checkpoint routing.

J19 run41 trained 500/500 steps then died publishing the final LoRA with
file_too_large: the executor built TrainingContext with only
``{"output_format": ...}`` hints — no kind/destination_repo/job_id — so
``_repo_job_upload_scope()`` was always None and ``ctx.save_checkpoint`` rode
the MEDIA route (256 MiB/file cap) instead of the repo-CAS checkpoint route
(the job-bound create_checkpoint grant tensorhub mints for the destination
repo at dispatch).

Real codepath: a local ThreadingHTTPServer stands in for tensorhub and a real
Executor runs a kind=training job end to end. The handler saves a checkpoint
LARGER than the media per-file cap plus a sample asset; the test asserts the
checkpoint rides POST /api/v1/repos/.../revisions[.../uploads] (session open
job-bound to the cap token's job_id claim, Bearer cap-token auth) while the
sample stays on /api/v1/media/<token-bound owner>/uploads.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, List, Tuple

import msgspec
import pytest

from gen_worker.executor import Executor, _producer_destination_repo
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

OWNER_UUID = "019f4c33-f3a5-705b-9848-0b3b0863c416"
JOB_ID = "job-run41"
DEST_REPO = "acme/lora-out"
MEDIA_MAX_BYTES = 256 * 1024 * 1024  # tensorhub media route per-file cap
REVISION_ID = "rev-453"


def _unsigned_jwt(claims: Dict[str, Any]) -> str:
    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


class _HubHandler(BaseHTTPRequestHandler):
    requests_seen: ClassVar[List[Tuple[str, str, Dict[str, Any]]]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests_seen.append(
            (self.path, str(self.headers.get("Authorization") or ""), body)
        )
        if self.path.endswith("/revisions"):
            # Checkpoint upload-session open.
            resp_obj: Dict[str, Any] = {"session_id": REVISION_ID, "expires_at": ""}
        else:
            # Upload create (repo-CAS or media): dedup outcome, no part PUTs.
            resp_obj = {
                "dedup": True,
                "ref": body.get("ref") or body.get("path") or "",
                "path": body.get("path") or "",
                "blake3": body.get("blake3") or "",
                "size_bytes": body.get("size_bytes") or 0,
                "mime_type": "application/octet-stream",
            }
        resp = json.dumps(resp_obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


@pytest.fixture()
def hub_server():
    _HubHandler.requests_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HubHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
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
            "request_id": "req-run41",
            "job_id": JOB_ID,
            "exp": 4_102_444_800,  # far future: renewal task stays asleep
            "grants": [
                {"do": "upload_media", "tenant": OWNER_UUID, "job": "req-run41"},
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
            request_id="req-run41", attempt=1, function_name="train",
            input_payload=msgspec.msgpack.encode(_In()),
            tenant=OWNER_UUID,
            capability_token=_cap_token(),
        ))
        job = ex.jobs[("req-run41", 1)]
        assert job.task is not None
        await job.task
        for _ in range(20):
            await asyncio.sleep(0)
        return sent

    return asyncio.run(_go())


def test_training_checkpoint_rides_repo_cas_not_media(hub_server: str, tmp_path) -> None:
    lora = tmp_path / "lora_000000500.safetensors"
    # Bigger than the media route's per-file cap: the exact run41 payload class.
    size = MEDIA_MAX_BYTES + 1024 * 1024
    with open(lora, "wb") as f:
        f.truncate(size)

    def _train(ctx, payload: _In) -> _Out:
        # The executor must have armed the repo-job scope from the payload's
        # destination + the cap token's job_id claim.
        assert ctx._execution_hints["kind"] == "training"
        assert ctx._execution_hints["destination_repo"] == DEST_REPO
        assert ctx._repo_job_upload_scope() == ("acme", "lora-out", JOB_ID)
        out = ctx.save_checkpoint(
            "checkpoints/lora_000000500.safetensors", str(lora),
            step_number=500, output_kind="lora",
        )
        assert out.size_bytes == size
        assert out.blake3  # repo-CAS route returns a blake3 digest
        # Samples are media outputs: they must stay on the media route.
        ctx.save_bytes("samples/sample_000000500.bin", b"sample-bytes")
        return _Out(ok=True)

    sent = _run_training_job(hub_server, _train)
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK

    token = _cap_token()
    seen = _HubHandler.requests_seen
    paths = [p for p, _, _ in seen]

    # 1. Checkpoint session opens on the destination repo, job-bound.
    open_calls = [(p, a, b) for p, a, b in seen if p == "/api/v1/repos/acme/lora-out/revisions"]
    assert len(open_calls) == 1
    _, auth, body = open_calls[0]
    assert auth == f"Bearer {token}"
    assert body["job_id"] == JOB_ID

    # 2. The checkpoint upload create rides the CHECKPOINT endpoints.
    up_path = f"/api/v1/repos/acme/lora-out/revisions/{REVISION_ID}/uploads"
    up_calls = [(p, a, b) for p, a, b in seen if p == up_path]
    assert len(up_calls) == 1
    _, auth, body = up_calls[0]
    assert auth == f"Bearer {token}"
    assert body["path"] == "checkpoints/lora_000000500.safetensors"
    assert body["size_bytes"] == size

    # 3. NOT the media route: the only media call is the sample asset,
    #    against the token-bound owner.
    media_calls = [(p, b) for p, _, b in seen if p.startswith("/api/v1/media/")]
    assert [(p, b["ref"]) for p, b in media_calls] == [
        (f"/api/v1/media/{OWNER_UUID}/uploads", "samples/sample_000000500.bin"),
    ]
    # And nothing else hit the hub.
    assert len(paths) == 3


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
    assert not [p for p, _, _ in _HubHandler.requests_seen if p.startswith("/api/v1/media/")]


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
