from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gen_worker.worker import ActionContext


class _UploadHandler(BaseHTTPRequestHandler):
    got_path: str = ""
    got_authz: str = ""
    got_owner: str = ""
    got_body: bytes = b""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        return

    def do_PUT(self) -> None:  # noqa: N802
        _UploadHandler.got_path = self.path
        _UploadHandler.got_authz = self.headers.get("Authorization") or ""
        _UploadHandler.got_owner = self.headers.get("X-Cozy-Owner") or ""
        n = int(self.headers.get("Content-Length") or "0")
        _UploadHandler.got_body = self.rfile.read(n) if n > 0 else b""
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


class OutputSaveContractTest(unittest.TestCase):
    def test_save_bytes_local_output_accepts_path_agnostic_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = ActionContext(
                run_id="run-1",
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
            ctx = ActionContext(run_id="run-2", local_output_dir=td)
            with self.assertRaises(ValueError):
                ctx.save_bytes("https://example.test/out.bin", b"x")

    def test_save_bytes_uploads_to_tensorhub_file_api(self) -> None:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _UploadHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            ctx = ActionContext(
                run_id="run-3",
                owner="alice",
                file_api_base_url=base,
                file_api_token="worker-cap-token",
            )
            asset = ctx.save_bytes("refs/out.bin", b"ABC")
            self.assertEqual(asset.size_bytes, 3)
            self.assertEqual(_UploadHandler.got_authz, "Bearer worker-cap-token")
            self.assertEqual(_UploadHandler.got_owner, "alice")
            self.assertTrue(_UploadHandler.got_path.startswith("/api/v1/file/"))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
