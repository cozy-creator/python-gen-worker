"""Real sha256-addressed HTTP blob host — stands in for tensorhub's presigned
GET URLs on ``pb.SnapshotFile.url``. No mocking of the download path itself:
the worker does a real HTTP GET and a real digest verify against this server.
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path
from typing import Any, List, Optional

import hashlib

from gen_worker.pb import worker_scheduler_pb2 as pb


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(
        self, *args: Any, directory: str, host: "BlobHost", **kwargs: Any,
    ) -> None:
        self._host = host
        super().__init__(*args, directory=directory, **kwargs)

    def send_head(self) -> Any:
        self._host.before_serve(self.path.lstrip("/"))
        return super().send_head()

    def log_message(self, *_args: Any) -> None:  # silence
        pass


class BlobHost:
    """One ThreadingHTTPServer serving a tmp_path-scoped directory of blobs."""

    def __init__(self, root: Path) -> None:
        self._dir = root / "www"
        self._dir.mkdir(parents=True, exist_ok=True)

        def _handler(*args: Any, **kwargs: Any) -> _QuietHandler:
            return _QuietHandler(
                *args, directory=str(self._dir), host=self, **kwargs)

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def before_serve(self, name: str) -> None:
        """Hook run on the handler thread just before ``name`` is served."""

    def put(self, name: str, payload: bytes) -> str:
        (self._dir / name).write_bytes(payload)
        return f"{self.base_url}/{name}"

    def file(
        self, name: str, payload: bytes, *, path_in_snapshot: str = "model.safetensors",
    ) -> pb.SnapshotFile:
        url = self.put(name, payload)
        return pb.SnapshotFile(
            path=path_in_snapshot, size_bytes=len(payload),
            digest="sha256:" + hashlib.sha256(payload).hexdigest(), url=url,
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


class GatedBlobHost(BlobHost):
    """Holds the GET for ONE named blob until :meth:`release`.

    pgw#955: a deterministic way to keep a residency reconcile pass genuinely
    in flight while the test does something else — the download the pass is
    awaiting simply does not answer yet. ``requested`` fires when the gated GET
    arrives, so the test waits on the EVENT rather than on a sleep. ``shutdown``
    releases, so a failing test cannot leave the handler thread parked.
    """

    def __init__(self, root: Path, *, gated: str) -> None:
        super().__init__(root)
        self._gated = gated
        self.requested = threading.Event()
        self._release = threading.Event()

    def before_serve(self, name: str) -> None:
        if name != self._gated:
            return
        self.requested.set()
        self._release.wait()

    def release(self) -> None:
        self._release.set()

    def shutdown(self) -> None:
        self.release()
        super().shutdown()
