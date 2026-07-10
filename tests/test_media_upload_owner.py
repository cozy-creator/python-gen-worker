"""Media uploads target the capability-token-bound owner (J19 run34 fix).

TrainingContext.save_image (and every media-route save_*) must ride the SAME
worker-authorized mechanism as inference outputs: POST
/api/v1/media/<token-bound owner>/uploads with the worker capability token.
Tensorhub authorizes these writes by matching the token's upload_media grant,
which is bound to the canonical invoking-org uuid in the token's `tenant`
claim — NOT by the dispatch-stamped ctx.owner, which can be a slug or a
destination-repo owner resolving to a different org. J19 run34 (2026-07-10):
ai-toolkit trained 500/500 steps, then died on the sample-image upload because
the create POST went to /api/v1/media/tensorhub/uploads (slug) while the
grant was bound to the invoker org uuid -> 403.

Real codepath: a local ThreadingHTTPServer stands in for tensorhub; the ctx
drives the actual save_image -> save_bytes -> _RequestOutputStream ->
presigned_upload_file flow over real HTTP. The server answers the create POST
with a dedup response so the test needs no S3 part scripting.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, List, Tuple

import pytest

from gen_worker import RequestContext, TrainingContext

OWNER_UUID = "019f4c33-f3a5-705b-9848-0b3b0863c416"


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
        # Dedup response: valid create outcome that needs no part PUTs.
        resp = json.dumps(
            {
                "dedup": True,
                "ref": body.get("ref") or "",
                "filename": "out.webp",
                "blake3": body.get("blake3") or "",
                "size_bytes": body.get("size_bytes") or 0,
                "mime_type": "image/webp",
                "media_id": "m1",
            }
        ).encode("utf-8")
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
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()


def test_training_save_image_uses_token_bound_owner(hub_server: str) -> None:
    pil = pytest.importorskip("PIL.Image")
    token = _unsigned_jwt(
        {
            "cap_kind": "worker_capability",
            "tenant": OWNER_UUID,
            "request_id": "req-run34",
            "grants": [{"do": "upload_media", "tenant": OWNER_UUID, "job": "req-run34"}],
        }
    )
    # ctx.owner is the dispatch-stamped slug — the run34 failure mode.
    ctx = TrainingContext(
        request_id="req-run34",
        owner="tensorhub",
        file_api_base_url=hub_server,
        worker_capability_token=token,
    )
    asset = ctx.save_image(pil.new("RGB", (4, 4)), "samples/pair-000.webp")

    assert asset.ref
    assert len(_HubHandler.requests_seen) == 1
    path, auth, body = _HubHandler.requests_seen[0]
    # The create POST rides the worker-authorized media route for the
    # TOKEN-BOUND owner, with the capability token as bearer auth.
    assert path == f"/api/v1/media/{OWNER_UUID}/uploads"
    assert auth == f"Bearer {token}"
    assert body["ref"] == "samples/pair-000.webp"
    assert body["request_id"] == "req-run34"


def test_inference_save_bytes_same_route_and_owner(hub_server: str) -> None:
    token = _unsigned_jwt({"tenant": OWNER_UUID, "request_id": "req-inf"})
    ctx = RequestContext(
        request_id="req-inf",
        owner=OWNER_UUID,  # inference dispatch already stamps the uuid
        file_api_base_url=hub_server,
        worker_capability_token=token,
    )
    ctx.save_bytes("outputs/req-inf/out.bin", b"payload")

    (path, auth, _body) = _HubHandler.requests_seen[0]
    assert path == f"/api/v1/media/{OWNER_UUID}/uploads"
    assert auth == f"Bearer {token}"


def test_media_owner_falls_back_to_ctx_owner_without_jwt(hub_server: str) -> None:
    # Dev/local paths: token missing a tenant claim (or not a JWT) keeps the
    # historical ctx.owner routing.
    ctx = RequestContext(
        request_id="r-dev",
        owner="dev-org",
        file_api_base_url=hub_server,
        worker_capability_token="not-a-jwt",
    )
    ctx.save_bytes("outputs/r-dev/a.bin", b"x")

    (path, _auth, _body) = _HubHandler.requests_seen[0]
    assert path == "/api/v1/media/dev-org/uploads"
