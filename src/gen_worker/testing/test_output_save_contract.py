from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import uuid
import base64
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gen_worker.worker import RequestContext


class _UploadHandler(BaseHTTPRequestHandler):
    got_path: str = ""
    got_authz: str = ""
    got_owner: str = ""
    got_body: bytes = b""
    upload_sessions: dict[str, dict[str, object]] = {}
    upload_sessions_lock = threading.Lock()
    repo_metadata_exists: bool = True
    repo_metadata_body: dict[str, object] = {"mirror": {"mode": "mirror"}}
    got_repo_metadata_write: dict[str, object] = {}
    got_repo_create_payload: dict[str, object] = {}
    got_run_start_payload: dict[str, object] = {}
    got_run_finalize_payload: dict[str, object] = {}
    got_run_commit_payload: dict[str, object] = {}
    got_claims_search_payload: dict[str, object] = {}
    got_claim_upsert_payload: dict[str, object] = {}
    got_claim_delete_path: str = ""
    got_copy_by_ref_payload: dict[str, object] = {}

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        return

    def do_PUT(self) -> None:  # noqa: N802
        _UploadHandler.got_path = self.path
        _UploadHandler.got_authz = self.headers.get("Authorization") or ""
        _UploadHandler.got_owner = self.headers.get("X-Cozy-Owner") or ""
        n = int(self.headers.get("Content-Length") or "0")
        _UploadHandler.got_body = self.rfile.read(n) if n > 0 else b""
        if self.path == "/api/v1/repos/alice/model-a/metadata":
            try:
                payload = json.loads(_UploadHandler.got_body.decode("utf-8"))
            except Exception:
                payload = {}
            md = payload.get("metadata")
            if isinstance(md, dict):
                _UploadHandler.got_repo_metadata_write = dict(md)
            else:
                _UploadHandler.got_repo_metadata_write = {}
            body = json.dumps({"metadata": _UploadHandler.got_repo_metadata_write, "exists": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(
            {
                "ref": "custom/outputs/x.bin",
                "size_bytes": len(_UploadHandler.got_body),
                "sha256": "abc",
                "mime_type": "application/octet-stream",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _media_session_id(path: str, suffix: str = "") -> str:
        prefix = "/api/v1/media/uploads/"
        if not path.startswith(prefix):
            return ""
        tail = path[len(prefix) :]
        if suffix:
            if not tail.endswith(suffix):
                return ""
            tail = tail[: -len(suffix)]
        sid = tail.strip().strip("/")
        if sid == "" or "/" in sid:
            return ""
        return sid

    def do_POST(self) -> None:  # noqa: N802
        _UploadHandler.got_path = self.path
        _UploadHandler.got_authz = self.headers.get("Authorization") or ""
        _UploadHandler.got_owner = self.headers.get("X-Cozy-Owner") or ""
        n = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}
        if self.path == "/api/v1/media/uploads":
            ref = str(payload.get("ref") or "").strip()
            if ref == "":
                self.send_response(400)
                self.end_headers()
                return
            sid = f"sess-{uuid.uuid4().hex}"
            with _UploadHandler.upload_sessions_lock:
                _UploadHandler.upload_sessions[sid] = {
                    "ref": ref,
                    "buf": bytearray(),
                }
            body = json.dumps({"upload_id": sid, "upload_offset": 0, "max_chunk_bytes": 8 * 1024 * 1024}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        session_id = self._media_session_id(self.path, suffix="/complete")
        if session_id:
            with _UploadHandler.upload_sessions_lock:
                sess = _UploadHandler.upload_sessions.pop(session_id, None)
            if not isinstance(sess, dict):
                self.send_response(404)
                self.end_headers()
                return
            buf = sess.get("buf")
            data = bytes(buf) if isinstance(buf, (bytes, bytearray)) else b""
            ref = str(sess.get("ref") or "")
            body = json.dumps(
                {
                    "ref": ref,
                    "size_bytes": len(data),
                    "sha256": "abc",
                    "mime_type": "application/octet-stream",
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/v1/repos":
            _UploadHandler.got_repo_create_payload = payload if isinstance(payload, dict) else {}
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/v1/repos/alice/model-a/conversion-jobs/start":
            _UploadHandler.got_run_start_payload = payload if isinstance(payload, dict) else {}
            body = json.dumps({"job_id": "job-start-1", "status": "running"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if re.match(r"^/api/v1/repos/alice/model-a/jobs/[^/]+/finalize$", self.path):
            _UploadHandler.got_run_finalize_payload = payload if isinstance(payload, dict) else {}
            body = json.dumps({"ok": True, "status": "succeeded", "output_versions": list((_UploadHandler.got_run_finalize_payload or {}).get("output_versions") or [])}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if re.match(r"^/api/v1/repos/alice/model-a/jobs/[^/]+/commit$", self.path):
            _UploadHandler.got_run_commit_payload = payload if isinstance(payload, dict) else {}
            body = json.dumps({
                "ok": True,
                "status": "succeeded",
                "output_versions": list((_UploadHandler.got_run_commit_payload or {}).get("output_versions") or []),
                "output_variant_count": len(list((_UploadHandler.got_run_commit_payload or {}).get("output_variants") or [])),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/v1/metadata/claims/search":
            _UploadHandler.got_claims_search_payload = payload if isinstance(payload, dict) else {}
            body = json.dumps({
                "scope": "version",
                "items": [
                    {
                        "claim_id": 11,
                        "owner": "alice",
                        "repo": "seed-model",
                        "version_id": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "identity_hash": str((_UploadHandler.got_claims_search_payload or {}).get("identity_hash") or ""),
                        "metadata_json": {
                            "source": {
                                "provider": "huggingface",
                                "source_ref": "hf-internal-testing/tiny-stable-diffusion-pipe",
                                "source_revision": "main",
                            }
                        },
                    }
                ],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if re.match(r"^/api/v1/repos/alice/model-a/versions/[^/]+/metadata/claims$", self.path):
            _UploadHandler.got_claim_upsert_payload = payload if isinstance(payload, dict) else {}
            body = json.dumps({"ok": True, "claim_id": 77}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/v1/repos/copy-by-reference":
            _UploadHandler.got_copy_by_ref_payload = payload if isinstance(payload, dict) else {}
            body = json.dumps({"ok": True, "copied_version_id": str((_UploadHandler.got_copy_by_ref_payload or {}).get("source_version_id") or "")}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_PATCH(self) -> None:  # noqa: N802
        _UploadHandler.got_path = self.path
        _UploadHandler.got_authz = self.headers.get("Authorization") or ""
        _UploadHandler.got_owner = self.headers.get("X-Cozy-Owner") or ""
        session_id = self._media_session_id(self.path)
        if session_id == "":
            self.send_response(404)
            self.end_headers()
            return
        with _UploadHandler.upload_sessions_lock:
            sess = _UploadHandler.upload_sessions.get(session_id)
            if not isinstance(sess, dict):
                self.send_response(404)
                self.end_headers()
                return
            buf = sess.get("buf")
            if not isinstance(buf, bytearray):
                buf = bytearray()
                sess["buf"] = buf
            current_offset = len(buf)
        try:
            upload_offset = int(self.headers.get("Upload-Offset") or "0")
        except Exception:
            upload_offset = 0
        if upload_offset != current_offset:
            self.send_response(409)
            self.send_header("Upload-Offset", str(current_offset))
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length") or "0")
        chunk = self.rfile.read(n) if n > 0 else b""
        with _UploadHandler.upload_sessions_lock:
            sess2 = _UploadHandler.upload_sessions.get(session_id)
            if not isinstance(sess2, dict):
                self.send_response(404)
                self.end_headers()
                return
            buf2 = sess2.get("buf")
            if not isinstance(buf2, bytearray):
                buf2 = bytearray()
                sess2["buf"] = buf2
            buf2.extend(chunk)
            next_offset = len(buf2)
        self.send_response(204)
        self.send_header("Upload-Offset", str(next_offset))
        self.end_headers()

    def do_DELETE(self) -> None:  # noqa: N802
        _UploadHandler.got_path = self.path
        _UploadHandler.got_authz = self.headers.get("Authorization") or ""
        _UploadHandler.got_owner = self.headers.get("X-Cozy-Owner") or ""
        session_id = self._media_session_id(self.path)
        if session_id:
            with _UploadHandler.upload_sessions_lock:
                _UploadHandler.upload_sessions.pop(session_id, None)
            self.send_response(204)
            self.end_headers()
            return
        if re.match(r"^/api/v1/repos/alice/model-a/versions/[^/]+/metadata/claims/\d+$", self.path):
            _UploadHandler.got_claim_delete_path = self.path
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        _UploadHandler.got_path = self.path
        _UploadHandler.got_authz = self.headers.get("Authorization") or ""
        _UploadHandler.got_owner = self.headers.get("X-Cozy-Owner") or ""
        if self.path == "/api/v1/repos/alice/model-a/metadata":
            if not _UploadHandler.repo_metadata_exists:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"exists": True, "metadata": _UploadHandler.repo_metadata_body}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


class OutputSaveContractTest(unittest.TestCase):
    @staticmethod
    def _fake_jwt(claims: dict[str, object]) -> str:
        payload = json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
        b64 = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        return f"x.{b64}.y"

    def test_save_bytes_local_output_accepts_path_agnostic_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = RequestContext(
                request_id="run-1",
                owner="alice",
                local_output_dir=td,
            )
            asset = ctx.save_bytes("custom/outputs/final.bin", b"OK")
            self.assertEqual(asset.ref, "custom/outputs/final.bin")
            self.assertIsNotNone(asset.local_path)
            with open(asset.local_path or "", "rb") as f:
                self.assertEqual(f.read(), b"OK")

    def test_save_bytes_rejects_url_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = RequestContext(request_id="run-2", local_output_dir=td)
            with self.assertRaises(ValueError):
                ctx.save_bytes("https://example.test/out.bin", b"x")

    def test_save_bytes_uploads_to_tensorhub_file_api(self) -> None:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            ctx = RequestContext(
                request_id="run-3",
                owner="alice",
                file_api_base_url=base,
                worker_capability_token="worker-cap-token",
            )
            asset = ctx.save_bytes("refs/out.bin", b"ABC")
            self.assertEqual(asset.size_bytes, 3)
            self.assertEqual(_UploadHandler.got_authz, "Bearer worker-cap-token")
            self.assertEqual(_UploadHandler.got_owner, "alice")
            self.assertTrue(_UploadHandler.got_path.startswith("/api/v1/media/uploads/"))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_save_bytes_uses_worker_capability_token_env(self) -> None:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        prior_base = os.environ.get("FILE_API_BASE_URL")
        prior_worker = os.environ.get("WORKER_CAPABILITY_TOKEN")
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            os.environ["FILE_API_BASE_URL"] = base
            os.environ["WORKER_CAPABILITY_TOKEN"] = "worker-cap-env-token"
            ctx = RequestContext(
                request_id="run-4",
                owner="alice",
            )
            asset = ctx.save_bytes("refs/env-out.bin", b"ABC")
            self.assertEqual(asset.size_bytes, 3)
            self.assertEqual(_UploadHandler.got_authz, "Bearer worker-cap-env-token")
            self.assertTrue(_UploadHandler.got_path.startswith("/api/v1/media/uploads/"))
        finally:
            if prior_base is None:
                os.environ.pop("FILE_API_BASE_URL", None)
            else:
                os.environ["FILE_API_BASE_URL"] = prior_base
            if prior_worker is None:
                os.environ.pop("WORKER_CAPABILITY_TOKEN", None)
            else:
                os.environ["WORKER_CAPABILITY_TOKEN"] = prior_worker
            srv.shutdown()
            srv.server_close()

    def test_publish_repo_revision_rejects_token_repo_scope_mismatch(self) -> None:
        token = self._fake_jwt({
            "cap_kind": "worker_capability",
            "tensor_repos_version_create": ["alice/model-a"],
            "tensor_repo_create": {"owner": "alice", "allowed_names": ["model-a"], "allow_any_name": False},
        })
        ctx = RequestContext(
            request_id="run-5",
            owner="alice",
            file_api_base_url="http://127.0.0.1:1",
            worker_capability_token=token,
        )
        with self.assertRaisesRegex(ValueError, "create allow-list|repo-version:create scope"):
            ctx.publish_repo_revision(destination_repo="alice/model-b", artifact_refs=[], metadata={})

    def test_publish_repo_revision_accepts_matching_token_scope(self) -> None:
        token = self._fake_jwt({
            "cap_kind": "worker_capability",
            "org": "alice",
            "tensor_repos_version_create": ["alice/model-a"],
            "tensor_repo_create": {"owner": "alice", "allowed_names": ["model-a"], "allow_any_name": False},
        })
        ctx = RequestContext(
            request_id="run-6",
            owner="alice",
            worker_capability_token=token,
        )
        out = ctx.publish_repo_revision(destination_repo="alice/model-a", artifact_refs=[], metadata={})
        self.assertEqual(out.get("ok"), False)
        self.assertEqual(out.get("skipped"), True)

    def test_publish_repo_revision_rejects_non_worker_capability_token(self) -> None:
        token = self._fake_jwt({
            "cap_kind": "user",
            "tensor_repos_version_create": ["alice/model-a"],
            "tensor_repo_create": {"owner": "alice", "allowed_names": ["model-a"], "allow_any_name": False},
        })
        ctx = RequestContext(
            request_id="run-7",
            owner="alice",
            file_api_base_url="http://127.0.0.1:1",
            worker_capability_token=token,
        )
        with self.assertRaisesRegex(ValueError, "cap_kind=worker_capability"):
            ctx.publish_repo_revision(destination_repo="alice/model-a", artifact_refs=[], metadata={})

    def test_publish_repo_revision_sends_publish_intent_for_same_version_variant(self) -> None:
        _UploadHandler.got_repo_create_payload = {}
        _UploadHandler.got_run_start_payload = {}
        _UploadHandler.got_run_finalize_payload = {}
        _UploadHandler.got_run_commit_payload = {}
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            token = self._fake_jwt({
                "cap_kind": "worker_capability",
                "org": "alice",
                "tensor_repos_version_create": ["alice/model-a"],
                "tensor_repo_create": {"owner": "alice", "allowed_names": ["model-a"], "allow_any_name": False},
            })
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            ctx = RequestContext(
                request_id="run-10",
                owner="alice",
                file_api_base_url=base,
                worker_capability_token=token,
            )
            out = ctx.publish_repo_revision(
                destination_repo="alice/model-a",
                artifact_refs=["jobs/run-10/outputs/model.safetensors"],
                metadata={"source_provider": "cozy", "source_ref": "alice/model-a:prod"},
                create_if_missing=True,
                destination_repo_tags=["prod", "beta"],
                source_repo="alice/model-a:prod",
                source_version_id="blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
            self.assertEqual(out.get("ok"), True)
            self.assertEqual(_UploadHandler.got_repo_create_payload.get("repo_name"), "model-a")

            start_payload = dict(_UploadHandler.got_run_start_payload)
            self.assertEqual(start_payload.get("kind"), "conversion")
            self.assertEqual(
                start_payload.get("input_versions"),
                ["blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
            )
            start_intent = dict(start_payload.get("publish_intent") or {})
            self.assertEqual(start_intent.get("repo"), "alice/model-a")
            self.assertEqual(start_intent.get("version_mode"), "same_version_variant")
            self.assertEqual(
                start_intent.get("source_version_id"),
                "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

            finalize_payload = dict(_UploadHandler.got_run_commit_payload)
            self.assertEqual(
                finalize_payload.get("output_versions"),
                ["blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
            )
            self.assertEqual(finalize_payload.get("run_kind"), "conversion")
            self.assertEqual(finalize_payload.get("status"), "succeeded")
            self.assertTrue(str(finalize_payload.get("commit_idempotency_key") or "").startswith("worker:run-10:"))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_read_repo_metadata_uses_capability_channel(self) -> None:
        _UploadHandler.repo_metadata_exists = True
        _UploadHandler.repo_metadata_body = {"mirror": {"provider": "huggingface", "mode": "mirror"}}
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            token = self._fake_jwt({
                "cap_kind": "worker_capability",
                "org": "alice",
                "tensor_repos_read": ["alice/model-a"],
            })
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            ctx = RequestContext(
                request_id="run-8",
                owner="alice",
                file_api_base_url=base,
                worker_capability_token=token,
            )
            out = ctx.read_repo_metadata(destination_repo="alice/model-a")
            self.assertEqual(out.get("exists"), True)
            self.assertEqual((out.get("metadata") or {}).get("mirror", {}).get("provider"), "huggingface")
            self.assertEqual(_UploadHandler.got_authz, f"Bearer {token}")
            self.assertEqual(_UploadHandler.got_owner, "alice")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_write_repo_metadata_uses_capability_channel(self) -> None:
        _UploadHandler.got_repo_metadata_write = {}
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            token = self._fake_jwt({
                "cap_kind": "worker_capability",
                "org": "alice",
                "tensor_repos_version_create": ["alice/model-a"],
            })
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            ctx = RequestContext(
                request_id="run-9",
                owner="alice",
                file_api_base_url=base,
                worker_capability_token=token,
            )
            out = ctx.write_repo_metadata(
                destination_repo="alice/model-a",
                metadata={"mirror": {"provider": "huggingface", "mode": "mirror"}},
            )
            self.assertEqual(out.get("ok"), True)
            self.assertEqual((_UploadHandler.got_repo_metadata_write.get("mirror") or {}).get("provider"), "huggingface")
            self.assertEqual(_UploadHandler.got_authz, f"Bearer {token}")
            self.assertEqual(_UploadHandler.got_owner, "alice")
        finally:
            srv.shutdown()
            srv.server_close()


    def test_search_metadata_claims_and_copy_by_reference_use_capability_channel(self) -> None:
        _UploadHandler.got_claims_search_payload = {}
        _UploadHandler.got_copy_by_ref_payload = {}
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            token = self._fake_jwt({
                "cap_kind": "worker_capability",
                "org": "alice",
                "tensor_repos_read": ["alice/seed-model"],
                "tensor_repos_version_create": ["alice/model-a"],
            })
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            ctx = RequestContext(
                request_id="run-claims-search",
                owner="alice",
                file_api_base_url=base,
                worker_capability_token=token,
            )
            search_out = ctx.search_metadata_claims(
                scope="version",
                identity_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                metadata_contains={"source": {"provider": "huggingface"}},
            )
            self.assertEqual(search_out.get("ok"), True)
            self.assertEqual((_UploadHandler.got_claims_search_payload or {}).get("scope"), "version")
            self.assertEqual((_UploadHandler.got_claims_search_payload or {}).get("identity_hash"), "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

            copy_out = ctx.copy_repo_by_reference(
                source_repo="alice/seed-model",
                source_version_id="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                destination_repo="alice/model-a",
                destination_repo_tags=["prod"],
                release_visibility="public",
            )
            self.assertEqual(copy_out.get("ok"), True)
            self.assertEqual((_UploadHandler.got_copy_by_ref_payload or {}).get("source_owner"), "alice")
            self.assertEqual((_UploadHandler.got_copy_by_ref_payload or {}).get("source_repo"), "seed-model")
            self.assertEqual((_UploadHandler.got_copy_by_ref_payload or {}).get("destination_repo"), "model-a")
            self.assertEqual((_UploadHandler.got_copy_by_ref_payload or {}).get("destination_repo_tags"), ["prod"])
            self.assertEqual(_UploadHandler.got_authz, f"Bearer {token}")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_upsert_and_delete_version_metadata_claim_use_capability_channel(self) -> None:
        _UploadHandler.got_claim_upsert_payload = {}
        _UploadHandler.got_claim_delete_path = ""
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            token = self._fake_jwt({
                "cap_kind": "worker_capability",
                "org": "alice",
                "tensor_repos_version_create": ["alice/model-a"],
            })
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            ctx = RequestContext(
                request_id="run-claims-write",
                owner="alice",
                file_api_base_url=base,
                worker_capability_token=token,
            )
            upsert_out = ctx.upsert_version_metadata_claim(
                destination_repo="alice/model-a",
                version_id="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                identity_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                metadata_json={"source": {"provider": "huggingface", "source_ref": "hf/model", "source_revision": "main"}},
            )
            self.assertEqual(upsert_out.get("ok"), True)
            self.assertEqual((_UploadHandler.got_claim_upsert_payload or {}).get("identity_hash"), "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

            delete_out = ctx.delete_version_metadata_claim(
                destination_repo="alice/model-a",
                version_id="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                claim_id=77,
            )
            self.assertEqual(delete_out.get("ok"), True)
            self.assertIn("/metadata/claims/77", _UploadHandler.got_claim_delete_path)
            self.assertEqual(_UploadHandler.got_authz, f"Bearer {token}")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
