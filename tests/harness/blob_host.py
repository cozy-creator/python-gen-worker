"""Real blake3-addressed HTTP blob host — stands in for tensorhub's presigned
GET URLs on ``pb.SnapshotFile.url``. No mocking of the download path itself:
the worker does a real HTTP GET and a real blake3 verify against this server.
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path
from typing import Any, List, Optional

from blake3 import blake3

from gen_worker.pb import worker_scheduler_pb2 as pb


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, directory: str, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, *_args: Any) -> None:  # silence
        pass


class BlobHost:
    """One ThreadingHTTPServer serving a tmp_path-scoped directory of blobs."""

    def __init__(self, root: Path) -> None:
        self._dir = root / "www"
        self._dir.mkdir(parents=True, exist_ok=True)

        def _handler(*args: Any, **kwargs: Any) -> _QuietHandler:
            return _QuietHandler(*args, directory=str(self._dir), **kwargs)

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def put(self, name: str, payload: bytes) -> str:
        (self._dir / name).write_bytes(payload)
        return f"{self.base_url}/{name}"

    def file(
        self, name: str, payload: bytes, *, path_in_snapshot: str = "model.safetensors",
    ) -> pb.SnapshotFile:
        url = self.put(name, payload)
        return pb.SnapshotFile(
            path=path_in_snapshot, size_bytes=len(payload),
            blake3=blake3(payload).hexdigest(), url=url,
        )

    def snapshot(self, digest: str, files: List[pb.SnapshotFile]) -> pb.Snapshot:
        return pb.Snapshot(digest=digest, files=files)

    def one_file_snapshot(
        self, digest: str, name: str, payload: bytes,
        *, path_in_snapshot: str = "model.safetensors",
    ) -> pb.Snapshot:
        """Convenience: the common single-weight-file case (P1-P3, P6, P9)."""
        return self.snapshot(digest, [self.file(name, payload, path_in_snapshot=path_in_snapshot)])

    def shutdown(self) -> None:
        self._httpd.shutdown()


class CorruptingBlobHost(BlobHost):
    """P2 quarantine case: serves the WRONG bytes for one named blob (digest
    mismatch is on the client verify side, so this is enough to trigger it),
    the correct bytes for everything else."""

    def __init__(self, root: Path, *, corrupt: Optional[str] = None) -> None:
        super().__init__(root)
        self._corrupt = corrupt

    def put(self, name: str, payload: bytes) -> str:
        if name == self._corrupt:
            payload = b"\x00" * len(payload) if payload else b"\x00"
        return super().put(name, payload)
