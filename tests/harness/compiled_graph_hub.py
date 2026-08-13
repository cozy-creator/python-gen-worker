"""pgw#978: a LOCAL hub that speaks the real compiled graph publish + discover wire.

Promoted out of ``test_compiled_graph_publish_v2_pgw807.py``'s ``_Hub``, which already
implemented tensorhub's v2 publish contract exactly (including the R2 digest
enforcement that makes the protocol worth having) — and extended with the half
that test did not need: the DISCOVER side a second worker adopts through.

Why a double rather than a compose tensorhub: the rig's value is that a full
machinery cycle costs seconds on a laptop. A stack with Postgres, MinIO and
Vault costs minutes to boot and is a second thing that can be broken. What
matters is that the WIRE is real — every route below is the one
``fleet_compiled_graphs.CompiledGraphPublisher`` and the rig's exact-fetch adopter actually call, with the
same status codes and the same envelope shapes, and the store really refuses
bytes that do not hash to their digest. A compiled graph published here and adopted here
crossed the same seven HTTP calls it would cross in production.

What this deliberately does NOT model: attestation, trust tiers, quota, and the
hub-side stamping of ``compiled_graph_store`` from the token claim. Those are refusals the
hub owns; a double that invented its own version of them would be testing the
double. Publish-intent here always grants.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


class _Handler(http.server.BaseHTTPRequestHandler):
    """tensorhub's v2 publish contract + the checkpoint/resolve read surface."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a: Any) -> None:
        pass

    # -- helpers ------------------------------------------------------------

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _bytes(self, code: int, raw: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {}

    # -- PUT: the CAS object store ------------------------------------------

    def do_PUT(self) -> None:  # noqa: N802
        srv = self.server
        digest = urllib.parse.urlparse(self.path).path.rsplit("/", 1)[-1]
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        # The digest is signed into the grant, so the STORE refuses bytes that
        # do not hash to it. Integrity must never depend on the sender being
        # honest — this is the property the whole chunked protocol is for.
        if hashlib.sha256(body).hexdigest() != digest:
            self._json(400, {"error": {"code": "digest_mismatch"}})
            return
        with srv.lock:  # type: ignore[attr-defined]
            srv.objects["sha256:" + digest] = body  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- GET: CAS reads + the discover surface ------------------------------

    def do_GET(self) -> None:  # noqa: N802
        srv = self.server
        parsed = urllib.parse.urlparse(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        with srv.lock:  # type: ignore[attr-defined]
            srv.calls.append((path, dict(query)))  # type: ignore[attr-defined]

        if path.startswith("/cas/"):
            digest = "sha256:" + path.rsplit("/", 1)[-1]
            with srv.lock:  # type: ignore[attr-defined]
                blob = srv.objects.get(digest)  # type: ignore[attr-defined]
            if blob is None:
                self._json(404, {"error": {"code": "not_found"}})
                return
            self._bytes(200, blob)
            return

        # catalog read surface: GET /api/v1/repos/<repo>/checkpoints
        if path.endswith("/checkpoints"):
            repo = path.split("/api/v1/repos/", 1)[-1].rsplit("/checkpoints", 1)[0]
            with srv.lock:  # type: ignore[attr-defined]
                items = [
                    dict(row) for row in srv.checkpoints  # type: ignore[attr-defined]
                    if row["repo"] == repo
                ]
            self._json(200, {"items": [
                {"checkpoint_id": r["checkpoint_id"],
                 "updated_at": r["updated_at"],
                 "metadata": r["metadata"]}
                for r in items]})
            return

        # harness.rig_fetch.fetch_named_compiled_graph: GET /api/v1/repos/<repo>/resolve
        if path.endswith("/resolve"):
            repo = path.split("/api/v1/repos/", 1)[-1].rsplit("/resolve", 1)[0]
            want = (query.get("digest") or [""])[0]
            with srv.lock:  # type: ignore[attr-defined]
                row = next(
                    (r for r in srv.checkpoints  # type: ignore[attr-defined]
                     if r["repo"] == repo and r["checkpoint_id"] == want), None)
            if row is None:
                self._json(404, {"error": {"code": "not_found"}})
                return
            self._json(200, {"files": row["files"]})
            return

        self._json(404, {"error": {"code": "not_found"}})

    # -- POST: the publish protocol -----------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        srv = self.server
        path = urllib.parse.urlparse(self.path).path
        body = self._body()
        with srv.lock:  # type: ignore[attr-defined]
            srv.calls.append((path, body))  # type: ignore[attr-defined]

        if path.endswith("/v1/worker/compiled_graphs/publish-intent"):
            family = str(body.get("family") or "")
            with srv.lock:  # type: ignore[attr-defined]
                srv.intents.append(dict(body))  # type: ignore[attr-defined]
            self._json(200, {"capability_token": "local-rig-cap",
                             "repo": f"root/family-{family}"})
            return

        if path.endswith("/v1/worker/compiled_graphs/publish-complete"):
            with slock(srv):
                srv.completes.append(dict(body))  # type: ignore[attr-defined]
            self._json(200, {"recorded": True})
            return

        if path.endswith("/publishes"):
            files = body.get("files") or []
            declared: Dict[str, int] = {}
            for f in files:
                for c in f.get("chunks") or []:
                    declared["sha256:" + c["digest"]] = int(c["len"])
                if not f.get("chunks"):
                    declared[f["digest"]] = int(f["size_bytes"])
            need, have = [], []
            with slock(srv):
                for digest, size in declared.items():
                    if digest in srv.objects:  # type: ignore[attr-defined]
                        have.append(digest)
                    else:
                        need.append(self._grant(srv, digest, size))
                pid = str(uuid.uuid4())
                srv.sessions[pid] = (declared, dict(body))  # type: ignore[attr-defined]
            self._json(201, {"publish_id": pid, "have": have, "need": need,
                             "distinct_objects": len(declared),
                             "resident_objects": len(have)})
            return

        if path.endswith("/grants"):
            pid = path.split("/publishes/")[1].split("/")[0]
            with slock(srv):
                declared = (srv.sessions.get(pid) or ({}, {}))[0]  # type: ignore[attr-defined]
                need = [
                    self._grant(srv, d, s) for d, s in declared.items()
                    if d not in srv.objects  # type: ignore[attr-defined]
                ]
            self._json(200, {"need": need, "have": []})
            return

        if path.endswith("/complete"):
            pid = path.split("/publishes/")[1].split("/")[0]
            repo = path.split("/api/v1/repos/", 1)[-1].split("/publishes/")[0]
            with slock(srv):
                declared, plan = srv.sessions.get(pid) or ({}, {})  # type: ignore[attr-defined]
                missing = [d for d in declared
                           if d not in srv.objects]  # type: ignore[attr-defined]
            if missing:
                self._json(409, {"status": {
                    "stage": "repudiated", "terminal": True,
                    "failure": {"code": "objects_missing", "retryable": False,
                                "message": f"{len(missing)} object(s) never landed"}}})
                return
            checkpoint_id = srv.record_checkpoint(repo, plan)  # type: ignore[attr-defined]
            self._json(200, {"checkpoint": {"checkpoint_id": checkpoint_id}})
            return

        self._json(404, {"error": {"code": "not_found"}})

    @staticmethod
    def _grant(srv: Any, digest: str, size: int) -> dict:
        return {"digest": digest, "size_bytes": size,
                "put_url": f"{srv.base}/cas/{digest.split(':', 1)[1]}",
                "headers": {"x-amz-checksum-sha256": digest}}


def slock(srv: Any) -> Any:
    return srv.lock


class LocalCompiledGraphHub:
    """A running local hub. ``base`` is what a ``CompiledGraphPublisher`` /
    the rig's fetch/publish legs take as their ``base_url``."""

    def __init__(self) -> None:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.lock = threading.Lock()  # type: ignore[attr-defined]
        self.httpd.objects = {}  # type: ignore[attr-defined]
        self.httpd.sessions = {}  # type: ignore[attr-defined]
        self.httpd.calls = []  # type: ignore[attr-defined]
        self.httpd.intents = []  # type: ignore[attr-defined]
        self.httpd.completes = []  # type: ignore[attr-defined]
        self.httpd.checkpoints = []  # type: ignore[attr-defined]
        self.httpd.base = self.base  # type: ignore[attr-defined]
        self.httpd.record_checkpoint = self._record_checkpoint  # type: ignore[attr-defined]
        self._seq = 0
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    # -- state a test reads --------------------------------------------------

    @property
    def base(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def intents(self) -> List[dict]:
        return list(self.httpd.intents)  # type: ignore[attr-defined]

    @property
    def completes(self) -> List[dict]:
        return list(self.httpd.completes)  # type: ignore[attr-defined]

    @property
    def checkpoints(self) -> List[dict]:
        return list(self.httpd.checkpoints)  # type: ignore[attr-defined]

    @property
    def calls(self) -> List[Tuple[str, Any]]:
        return list(self.httpd.calls)  # type: ignore[attr-defined]

    def routes(self) -> List[str]:
        return [p for p, _ in self.calls]

    # -- the publish -> discover join ---------------------------------------

    def _record_checkpoint(self, repo: str, plan: dict) -> str:
        """Turn a completed publish into a listable checkpoint.

        This is the join the discover side reads, and it is the ONE place the
        double has to model hub bookkeeping: the manifest a ``resolve`` returns
        has to describe the very objects the publish landed, or an adopting
        worker downloads bytes that hash to nothing.
        """
        files = []
        for f in plan.get("files") or []:
            digest = str(f.get("digest") or "")
            chunks = f.get("chunks") or []
            entry: Dict[str, Any] = {
                "path": str(f.get("path") or ""),
                "size_bytes": int(f.get("size_bytes") or 0),
                "sha256": digest.split(":", 1)[-1] if digest else "",
                "digest": digest,
            }
            if chunks:
                entry["chunks"] = [
                    {"sha256": c["digest"], "len": int(c["len"]),
                     "url": f"{self.base}/cas/{c['digest']}"}
                    for c in chunks
                ]
                entry["chunk_size_bytes"] = int(chunks[0]["len"])
            else:
                entry["url"] = f"{self.base}/cas/{digest.split(':', 1)[-1]}"
            files.append(entry)
        self._seq += 1
        checkpoint_id = "sha256:" + hashlib.sha256(
            f"{repo}/{plan.get('flavor')}/{self._seq}".encode()).hexdigest()
        row = {
            "repo": repo,
            "checkpoint_id": checkpoint_id,
            "updated_at": f"2026-08-06T00:00:{self._seq:02d}Z",
            "metadata": dict(plan.get("metadata") or {}),
            "files": files,
        }
        with self.httpd.lock:  # type: ignore[attr-defined]
            self.httpd.checkpoints.append(row)  # type: ignore[attr-defined]
        return checkpoint_id

    def artifact_bytes(self) -> int:
        return sum(len(v) for v in self.httpd.objects.values())  # type: ignore[attr-defined]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@contextlib.contextmanager
def local_compiled_graph_hub() -> Iterator[LocalCompiledGraphHub]:
    hub = LocalCompiledGraphHub()
    try:
        yield hub
    finally:
        hub.close()


def dump_state(hub: LocalCompiledGraphHub, path: Optional[Path] = None) -> dict:
    """Everything the rig's driver reports about the hub leg."""
    state = {
        "base": hub.base,
        "routes": hub.routes(),
        "intents": hub.intents,
        "completes": hub.completes,
        "checkpoints": [
            {k: v for k, v in row.items() if k != "files"}
            for row in hub.checkpoints
        ],
        "cas_bytes": hub.artifact_bytes(),
    }
    if path is not None:
        Path(path).write_text(json.dumps(state, indent=2, sort_keys=True, default=str))
    return state


__all__ = ["LocalCompiledGraphHub", "dump_state", "local_compiled_graph_hub"]
