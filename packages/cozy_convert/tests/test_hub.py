"""HubClient against a threaded fake of tensorhub's /commits API.

Exercises the real HTTP code path: POST /commits (presign), part PUTs,
per-upload complete, finalize (202 poll -> 200), and blake3 dedup skips.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from cozy_convert.hub import CommitFile, HubClient, HubPublishError, blake3_file, files_from_tree


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
            req = self._read_json()
            st["commit_request"] = req
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


def test_commit_uploads_completes_and_finalizes(fake_hub, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    (tmp_path / "sub").mkdir()
    (tmp_path / "config.json").write_text('{"a":1}')
    (tmp_path / "sub" / "weights.safetensors").write_bytes(b"\x00" * 64)

    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=files_from_tree(tmp_path),
        tags=["prod"],
        mode="replace",
        flavor="bf16",
        dtype="bf16",
        file_layout="diffusers",
        file_type="safetensors",
        lineage=[{"parent_repo": "external-sources/upstream",
                  "parent_checkpoint_id": "hf:org/name",
                  "relationship_kind": "import"}],
        auto_create_external_parent=True,
    )
    assert result.revision_id == "rev-1"
    assert result.uploaded == 2 and result.deduped == 0

    st = _FakeHub.state
    req = st["commit_request"]
    assert req["mode"] == "replace"
    assert req["tags"] == [{"tag": "prod"}]
    assert req["flavor"] == "bf16"
    assert req["auto_create_external_parent"] is True
    ops = {op["path"]: op for op in req["operations"]}
    assert set(ops) == {"config.json", "sub/weights.safetensors"}
    assert ops["config.json"]["blake3"] == blake3_file(tmp_path / "config.json")
    # Bytes actually PUT + parts echoed on complete.
    assert len(st["put_bytes"]) == 2
    assert st["complete_bodies"][0]["parts"][0] == {"part_number": 1, "etag": "etag-1"}
    assert st["finalize_calls"] == 2  # 202 then 200
    assert st["auth"] == "Bearer cap-token"


def test_dedup_skips_put(fake_hub, tmp_path: Path) -> None:
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x01" * 32)
    _FakeHub.state["existing_blobs"] = {blake3_file(f)}
    _FakeHub.state["finalize_calls"] = 1  # finalize returns 200 immediately

    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=[CommitFile(path="model.safetensors", local_path=f)],
    )
    assert result.deduped == 1 and result.uploaded == 0
    assert "put_bytes" not in _FakeHub.state


def test_destination_must_be_owner_repo(fake_hub) -> None:
    with pytest.raises(HubPublishError, match="owner/repo"):
        _client(fake_hub).commit(
            destination_repo="just-a-name",
            files=[CommitFile(path="x", local_path=Path("/nonexistent"))],
        )
