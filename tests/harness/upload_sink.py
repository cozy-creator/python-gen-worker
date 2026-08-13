"""A real local stand-in for tensorhub's media-upload route family.

Answers a dedup create, so a test needs no S3 part-PUT scripting to prove that
an upload really happened. `requests_seen` being non-empty is the observable:
it distinguishes "uploaded" from "returned a ref for bytes that never left the
process", which is the pgw#767 defect class and cannot be told apart from the
result envelope alone.

**The sink ROUTES; it does not accept everything.** `_ORG_LESS_ROUTES` mirrors
`registerMediaUploadRoutes(v1, "/media")` in tensorhub `internal/api/files.go`
(th#1722 §C, `de30113d`): the org is derived from the credential and is never a
path segment. Anything else 404s exactly as gin does — including the
transitional org-addressed alias `/api/v1/media/<org>/uploads`, which th#1799
deletes. That is what makes a client's URL construction testable end to end: a
test that let the server answer any path could not tell the two shapes apart,
which is the defect class pgw#1138 was filed against.

The v2 suite has its own richer `UploadSink` fixture (`tests_v2/conftest.py`,
with a `status=` knob for refusal rows). This is the v1 suite's minimal twin —
kept separate deliberately, because a cross-suite import would make the v2
fixture surface load-bearing for v1 tests.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Dict, List, Tuple

#: The canonical create route. Clients build this and nothing else.
MEDIA_UPLOADS_PATH = "/api/v1/media/uploads"

#: The org-less media-upload family, one compiled graph per tensorhub route. Kept as
#: patterns rather than prefixes so `/api/v1/media/<org>/uploads` — one extra
#: segment — cannot match by accident.
_ORG_LESS_ROUTES: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^/api/v1/media/uploads$",
        r"^/api/v1/media/uploads/batch$",
        r"^/api/v1/media/uploads/batch/complete$",
        r"^/api/v1/media/uploads/[^/]+$",
        r"^/api/v1/media/uploads/[^/]+/parts$",
        r"^/api/v1/media/uploads/[^/]+/complete$",
    )
)


def is_media_upload_route(path: str) -> bool:
    """True when `path` is served by tensorhub's org-less upload family."""
    bare = str(path or "").split("?", 1)[0]
    return any(rx.match(bare) for rx in _ORG_LESS_ROUTES)


class DedupUploadSink(BaseHTTPRequestHandler):
    requests_seen: ClassVar[List[Tuple[str, Dict[str, Any]]]] = []
    headers_seen: ClassVar[List[Dict[str, str]]] = []
    rejected: ClassVar[List[str]] = []

    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not is_media_upload_route(self.path):
            type(self).rejected.append(self.path)
            self._send(404, {
                "error": "not_found",
                "message": f"no route for POST {self.path}",
            })
            return
        body = json.loads(raw or b"{}")
        type(self).requests_seen.append((self.path, body))
        type(self).headers_seen.append({k: v for k, v in self.headers.items()})
        self._send(200, {
            "dedup": True, "ref": body.get("ref") or "", "filename": "out.bin",
            "blake3": body.get("blake3") or "", "size_bytes": body.get("size_bytes") or 0,
            "mime_type": "application/octet-stream", "media_id": "m1",
        })

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        resp = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def reset_upload_sink() -> None:
    DedupUploadSink.requests_seen = []
    DedupUploadSink.headers_seen = []
    DedupUploadSink.rejected = []


def serve_upload_sink() -> Tuple[ThreadingHTTPServer, str]:
    """Start the sink on an ephemeral port. Caller shuts it down and calls
    `reset_upload_sink()`."""
    reset_upload_sink()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DedupUploadSink)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
