"""Shared fake of tensorhub's /commits API (threaded HTTP server)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any

from gen_worker.convert.hub import HubClient


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
        if self.path.endswith("/clone-manifests/lookup"):
            # th#592 download-skip bank lookup. `ready` mirrors tensorhub:
            # every manifest blob must still be in CAS.
            if st.get("fail_bank_lookups", 0) > 0:
                st["fail_bank_lookups"] -= 1
                self._send(503, {"error": "unavailable"})
                return
            req = self._read_json()
            st.setdefault("bank_lookups", []).append(req)
            manifests = st.setdefault("bank_manifests", {})
            blobs = st.setdefault("cas_blobs", set())
            results = []
            for key in req.get("keys") or []:
                payload = manifests.get(key)
                if payload is None:
                    results.append({"key": key, "found": False, "ready": False})
                    continue
                ready = all(f["blake3"] in blobs for f in payload.get("files") or [])
                entry = {"key": key, "found": True, "ready": ready}
                if ready:
                    entry["payload"] = payload
                results.append(entry)
            self._send(200, {"results": results})
            return
        if self.path.endswith("/clone-manifests"):
            # th#592 bank record: refuse manifests whose blobs aren't in CAS.
            req = self._read_json()
            st.setdefault("bank_records", []).append(req)
            manifests = st.setdefault("bank_manifests", {})
            blobs = st.setdefault("cas_blobs", set())
            results = []
            for m in req.get("manifests") or []:
                key, payload = m.get("key"), m.get("payload") or {}
                if any(f["blake3"] not in blobs for f in payload.get("files") or []):
                    results.append({"key": key, "status": "missing_blobs"})
                    continue
                manifests[key] = payload
                results.append({"key": key, "status": "recorded"})
            self._send(200, {"results": results})
            return
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
            cas = st.setdefault("cas_blobs", set())
            for i, op in enumerate(req.get("operations", [])):
                if op["type"] != "add":
                    continue
                in_cas = op["blake3"] in st.get("existing_blobs", set()) | cas
                if in_cas and op["blake3"] not in st.get("commit_pretend_missing", set()):
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
            # Uploaded blobs land in the fake CAS (tests simulating GC or
            # missing blobs mutate state["cas_blobs"] directly).
            cas.update(op["blake3"] for op in req.get("operations", [])
                       if op["type"] == "add")
            self._send(201, {"revision_id": "rev-1", "uploads": uploads,
                             "deletions": [], "copies": [], "tags": req.get("tags") or [],
                             "mode": req.get("mode") or "merge"})
            return
        if "/uploads/" in self.path and self.path.endswith("/complete"):
            if st.get("fail_completes", 0) > 0:
                st["fail_completes"] -= 1
                self._send(500, {"error": "boom"})
                return
            if st.get("complete_race_count", 0) > 0:
                # Simulates a still-finalizing concurrent attempt (tensorhub
                # verifies large single files synchronously and can outlast
                # the client's timeout -- e2e tracker #110): the caller must
                # poll rather than treat this 409 as fatal.
                st["complete_race_count"] -= 1
                st.setdefault("complete_race_polls", []).append(self.path)
                self._send(409, {"error": {"code": "upload_complete_in_progress",
                                           "message": "a concurrent completion is in progress"}})
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


def _client(server) -> HubClient:
    return HubClient(
        base_url=f"http://127.0.0.1:{server.server_port}",
        token="cap-token", owner="acme",
    )
