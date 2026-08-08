"""A real local stand-in for tensorhub's `/api/v1/media/:owner/uploads`.

Answers a dedup create, so a test needs no S3 part-PUT scripting to prove that
an upload really happened. `requests_seen` being non-empty is the observable:
it distinguishes "uploaded" from "returned a ref for bytes that never left the
process", which is the pgw#767 defect class and cannot be told apart from the
result envelope alone.

The v2 suite has its own richer `UploadSink` fixture (`tests_v2/conftest.py`,
with a `status=` knob for refusal rows). This is the v1 suite's minimal twin —
kept separate deliberately, because a cross-suite import would make the v2
fixture surface load-bearing for v1 tests.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, List, Tuple


class DedupUploadSink(BaseHTTPRequestHandler):
    requests_seen: ClassVar[List[Tuple[str, Dict[str, Any]]]] = []

    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests_seen.append((self.path, body))
        resp = json.dumps({
            "dedup": True, "ref": body.get("ref") or "", "filename": "out.bin",
            "blake3": body.get("blake3") or "", "size_bytes": body.get("size_bytes") or 0,
            "mime_type": "application/octet-stream", "media_id": "m1",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def serve_upload_sink() -> Tuple[ThreadingHTTPServer, str]:
    """Start the sink on an ephemeral port. Caller shuts it down and resets
    `DedupUploadSink.requests_seen`."""
    DedupUploadSink.requests_seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DedupUploadSink)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
