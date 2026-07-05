"""Shared fake of tensorhub's /commits API (threaded HTTP server)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from cozy_convert.hub import HubClient


class _FakeHub(BaseHTTPRequestHandler):
    server_version = "FakeTensorhub/1.0"
    state: dict[str, Any] = {}

    def log_message(self, *args: Any) -> None:  # silence
        pass

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        return json.loads(body) if body else {}

    def _send(self, code: int, payload: dict | None = None) -> None:
        body = json.dumps(payload or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        st = _FakeHub.state
        if self.path.endswith("/commits"):
            if st.get("fail_commit_posts", 0) > 0:
                st["fail_commit_posts"] -= 1
                self._send(503, {"error": "unavailable"})
                return
            req = self._read_json()
            st["commit_request"] = req
            st.setdefault("commit_requests", []).append(req)
            st["auth"] = self.headers.get("Authorization", "")
            uploads = []
            for i, op in enumerate(req.get("operations", [])):
                if op["type"] != "add":
                    continue
                if op["blake3"] in st.get("existing_blobs", set()):
                    uploads.append({"path": op["path"], "blake3": op["blake3"], "exists": True})
                    continue
                uid = f"up-{i}"
                base = f"http://127.0.0.1:{self.server.server_port}"
                if st.get("grant_mode"):
                    # R2 SDK-transfer shape: scoped temp credential, no
                    # multipart part URLs.
                    uploads.append({
                        "path": op["path"], "blake3": op["blake3"], "exists": False,
                        "upload_id": uid,
                        "size_bytes": int(op["size_bytes"]),
                        "transfer_grant": {
                            "endpoint_url": "https://acct.r2.cloudflarestorage.com",
                            "bucket": "repo-cas",
                            "key": f"__presigned_staging/v1/{uid}/object",
                            "access_key_id": "k", "secret_access_key": "s",
                            "session_token": "t", "region": "auto",
                        },
                    })
                    continue
                uploads.append({
                    "path": op["path"], "blake3": op["blake3"], "exists": False,
                    "upload_id": uid,
                    "part_urls": [f"{base}/put/{uid}/1"],
                    "part_size": max(int(op["size_bytes"]), 1),
                    "total_parts": 1,
                })
            self._send(201, {"revision_id": "rev-1", "uploads": uploads,
                             "deletions": [], "copies": [], "tags": req.get("tags") or [],
                             "mode": req.get("mode") or "merge"})
            return
        if "/uploads/" in self.path and self.path.endswith("/complete"):
            if st.get("fail_completes", 0) > 0:
                st["fail_completes"] -= 1
                self._send(500, {"error": "boom"})
                return
            st.setdefault("completed", []).append(self.path)
            body = self._read_json()
            st.setdefault("complete_bodies", []).append(body)
            self._send(200, {"ok": True})
            return
        if self.path.endswith("/finalize"):
            n = st.get("finalize_calls", 0) + 1
            st["finalize_calls"] = n
            if n == 1:
                self._send(202, {"status": "running"})  # first call -> poll
            else:
                self._send(200, {"ok": True, "checkpoint_id": "blake3:abc"})
            return
        self._send(404, {"error": "not_found"})

    def do_PUT(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else b""
        st = _FakeHub.state
        if st.get("fail_puts", 0) > 0:
            st["fail_puts"] -= 1
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        _FakeHub.state.setdefault("put_bytes", {})[self.path] = data
        self.send_response(200)
        self.send_header("ETag", '"etag-1"')
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture()
def fake_hub():
    _FakeHub.state = {"existing_blobs": set()}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHub)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server
    server.shutdown()


def _client(server) -> HubClient:
    return HubClient(
        base_url=f"http://127.0.0.1:{server.server_port}",
        token="cap-token", owner="acme",
    )
