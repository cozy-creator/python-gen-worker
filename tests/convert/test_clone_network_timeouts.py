"""gw#456: clone downloads must never hang on a stalled connection.

Real sockets, real client codepaths: a threaded local HTTP server speaks just
enough of the HF hub protocol (repo_info, tree listing, HEAD/GET resolve with
Range) plus a civitai endpoint. Failure modes are injected at the socket level
(never respond / drop mid-body / 500), never by mocking the client.
"""

from __future__ import annotations

import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REV = "a" * 40


def _tiny_safetensors(n_bytes: int = 1 << 20) -> bytes:
    """A minimal valid safetensors blob of roughly n_bytes."""
    header = json.dumps({
        "__metadata__": {"format": "pt"},
        "w": {"dtype": "F16", "shape": [n_bytes // 2], "data_offsets": [0, n_bytes]},
    }).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * n_bytes


class _FakeHF(BaseHTTPRequestHandler):
    """Minimal HF hub + civitai file server with per-path failure modes:
    'ok' | 'stall' (accept, never respond) | 'drop_once' (half body, abort)
    | 'error_once' (one 500)."""

    protocol_version = "HTTP/1.0"

    def log_message(self, *a) -> None:  # silence
        pass

    # -- helpers -----------------------------------------------------------
    def _send_json(self, payload, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mode(self, name: str) -> str:
        return self.server.modes.get(name, "ok")  # type: ignore[attr-defined]

    def _consume_once(self, name: str, mode: str) -> bool:
        srv = self.server
        with srv.lock:  # type: ignore[attr-defined]
            if self._mode(name) == mode:
                srv.modes[name] = "ok"  # type: ignore[attr-defined]
                return True
        return False

    def _stall(self) -> None:
        # Accept the request, never respond — the live incident's shape.
        self.server.stop_event.wait(60)  # type: ignore[attr-defined]

    # -- routes ------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        srv = self.server
        if "/resolve/" in self.path:
            name = self.path.rsplit("/", 1)[-1]
            data = srv.files.get(name)  # type: ignore[attr-defined]
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("ETag", f'"{name}-etag"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Repo-Commit", REV)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        srv = self.server
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/") and srv.modes.get("_api") == "stall":  # type: ignore[attr-defined]
            self._stall()
            return
        repo = srv.repo_id  # type: ignore[attr-defined]

        # civitai model-version metadata
        if path.startswith("/api/v1/model-versions/"):
            self._send_json(srv.civitai_payload)  # type: ignore[attr-defined]
            return

        # HfApi.repo_info
        if path in (f"/api/models/{repo}", f"/api/models/{repo}/revision/{REV}",
                    "/api/models/" + repo + "/revision/main"):
            self._send_json({
                "id": repo, "modelId": repo, "sha": REV, "private": False,
                "tags": [], "downloads": 0, "likes": 0,
                "siblings": [{"rfilename": n} for n in sorted(srv.files)],  # type: ignore[attr-defined]
            })
            return

        # HfApi.list_repo_tree
        if path.startswith(f"/api/models/{repo}/tree/"):
            entries = []
            for name, data in sorted(srv.files.items()):  # type: ignore[attr-defined]
                e = {"type": "file", "path": name, "size": len(data), "oid": "git-" + name}
                if name.endswith(".safetensors"):
                    e["lfs"] = {"oid": "f" * 64, "size": len(data), "pointerSize": 135}
                entries.append(e)
            self._send_json(entries)
            return

        # file bytes: /{repo}/resolve/{rev}/{name} and civitai /dl/{name}
        if "/resolve/" in path or path.startswith("/dl/"):
            name = path.rsplit("/", 1)[-1]
            data = srv.files.get(name)  # type: ignore[attr-defined]
            if data is None:
                self._send_json({"error": "not found"}, code=404)
                return
            with srv.lock:  # type: ignore[attr-defined]
                srv.get_counts[name] = srv.get_counts.get(name, 0) + 1  # type: ignore[attr-defined]
            if self._mode(name) == "stall":
                self._stall()
                return
            if self._consume_once(name, "error_once"):
                self._send_json({"error": "boom"}, code=500)
                return
            start = 0
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                start = int(rng[6:].split("-", 1)[0] or 0)
                srv.ranges.append((name, start))  # type: ignore[attr-defined]
            body = data[start:]
            drop = self._consume_once(name, "drop_once")
            stall_mid = self._consume_once(name, "stall_mid_once")
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", f'"{name}-etag"')
            self.send_header("X-Repo-Commit", REV)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if drop:
                self.wfile.write(body[: max(1, len(body) // 2)])
                self.wfile.flush()
                self.connection.close()  # abrupt mid-body disconnect
                return
            if stall_mid:
                self.wfile.write(body[: max(1, len(body) // 2)])
                self.wfile.flush()
                self._stall()  # half the body, then dead air
                return
            self.wfile.write(body)
            return

        self._send_json({"error": "not found"}, code=404)


@pytest.fixture()
def hf_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHF)
    srv.daemon_threads = True
    srv.repo_id = "acme/tiny-model"
    weights = _tiny_safetensors()
    srv.files = {
        "config.json": json.dumps({"architectures": ["LlamaForCausalLM"],
                                   "model_type": "llama"}).encode(),
        "model.safetensors": weights,
    }
    srv.modes = {}
    srv.get_counts = {}
    srv.ranges = []
    srv.lock = threading.Lock()
    srv.stop_event = threading.Event()
    srv.civitai_payload = {}
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.stop_event.set()
        srv.shutdown()
        srv.server_close()


@pytest.fixture()
def hf_env(hf_server, monkeypatch, tmp_path):
    """Point huggingface_hub at the fake server; small timeouts; fresh cache."""
    import huggingface_hub.constants as hc

    from gen_worker import net

    base = f"http://127.0.0.1:{hf_server.server_port}"
    monkeypatch.setattr(hc, "ENDPOINT", base)
    monkeypatch.setattr(hc, "HUGGINGFACE_CO_URL_TEMPLATE",
                        base + "/{repo_id}/resolve/{revision}/{filename}")
    monkeypatch.setattr(hc, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setenv(net.CONNECT_TIMEOUT_ENV, "2")
    monkeypatch.setenv(net.READ_TIMEOUT_ENV, "2")
    monkeypatch.setenv("COZY_CLONE_DOWNLOAD_ATTEMPTS", "2")
    # Socket timeouts are the stall detector under test; keep the disk-scan
    # watchdog out of the way.
    monkeypatch.setenv("COZY_HF_DOWNLOAD_STALL_TIMEOUT_S", "60")
    net.install_hf_http_timeouts()
    return base


def test_metadata_stall_fails_fast(hf_server, hf_env):
    """HfApi.repo_info passes an explicit timeout=None — the floor hook must
    still time it out (the exact live-incident hang: request sent, upstream
    never answers, job stuck forever)."""
    from gen_worker.convert.ingest import plan_huggingface

    hf_server.modes["_api"] = "stall"
    t0 = time.monotonic()
    with pytest.raises(Exception) as exc_info:
        plan_huggingface(hf_server.repo_id)
    elapsed = time.monotonic() - t0
    assert elapsed < 20, f"metadata stall not bounded: {elapsed:.1f}s"
    assert "timed out" in str(exc_info.value).lower() or "timeout" in type(exc_info.value).__name__.lower()


def test_download_stall_aborts_cleanly(hf_server, hf_env, tmp_path):
    """Weights GET accepts the request and never responds: the clone must
    surface CloneDownloadError within the bounded retry budget, not hang."""
    from gen_worker.convert.ingest import CloneDownloadError, ingest_huggingface

    hf_server.modes["model.safetensors"] = "stall"
    t0 = time.monotonic()
    with pytest.raises(CloneDownloadError):
        ingest_huggingface(hf_server.repo_id, tmp_path / "snap")
    elapsed = time.monotonic() - t0
    assert elapsed < 60, f"stalled download not bounded: {elapsed:.1f}s"


def test_drop_mid_body_outer_retry_succeeds(hf_server, hf_env, tmp_path):
    """First weights GET dies mid-body (peer close): hf_hub does not retry
    that, so the clone's bounded outer retry must recover byte-identically."""
    from gen_worker.convert.ingest import ingest_huggingface

    hf_server.modes["model.safetensors"] = "drop_once"
    calls: list[tuple[int, int | None]] = []
    src = ingest_huggingface(hf_server.repo_id, tmp_path / "snap",
                             progress=lambda done, total: calls.append((done, total)))
    got = (Path(src.dir) / "model.safetensors").read_bytes()
    assert got == hf_server.files["model.safetensors"]
    assert (Path(src.dir) / "config.json").exists()
    assert hf_server.get_counts["model.safetensors"] >= 2
    dones = [d for d, _ in calls]
    assert dones == sorted(dones) and dones[-1] > 0  # monotonic, reached total


def test_stall_mid_body_resumes_via_range(hf_server, hf_env, tmp_path, monkeypatch):
    """Mid-body stall (the CDN CLOSE-WAIT shape): the floored read timeout
    aborts the dead socket and hf_hub resumes with a Range request instead of
    hanging or restarting from zero."""
    import huggingface_hub.constants as hc

    from gen_worker.convert.ingest import ingest_huggingface

    monkeypatch.setattr(hc, "DOWNLOAD_CHUNK_SIZE", 64 * 1024)
    hf_server.modes["model.safetensors"] = "stall_mid_once"
    t0 = time.monotonic()
    src = ingest_huggingface(hf_server.repo_id, tmp_path / "snap")
    elapsed = time.monotonic() - t0
    got = (Path(src.dir) / "model.safetensors").read_bytes()
    assert got == hf_server.files["model.safetensors"]
    assert any(name == "model.safetensors" and off > 0 for name, off in hf_server.ranges), \
        f"no Range resume observed: {hf_server.ranges}"
    assert elapsed < 60, f"mid-body stall not bounded: {elapsed:.1f}s"


def test_http_500_once_outer_retry_succeeds(hf_server, hf_env, tmp_path):
    """A transient 500 on the weights GET is retried by the clone's own
    bounded retry loop (hf_hub does not retry 5xx) and then succeeds."""
    from gen_worker.convert.ingest import ingest_huggingface

    hf_server.modes["model.safetensors"] = "error_once"
    src = ingest_huggingface(hf_server.repo_id, tmp_path / "snap")
    got = (Path(src.dir) / "model.safetensors").read_bytes()
    assert got == hf_server.files["model.safetensors"]
    assert hf_server.get_counts["model.safetensors"] >= 2


def test_civitai_drop_mid_body_retries(hf_server, hf_env, tmp_path, monkeypatch):
    """Civitai stream dies mid-body once; the per-file retry re-downloads and
    the sha256 check passes."""
    import hashlib

    from gen_worker.models import download as dl

    data = hf_server.files["model.safetensors"]
    name = "civi-model.safetensors"
    hf_server.files[name] = data
    hf_server.modes[name] = "drop_once"
    hf_server.civitai_payload = {
        "id": 123, "modelId": 7, "baseModel": "SDXL 1.0", "air": "",
        "model": {"type": "Checkpoint"},
        "files": [{
            "id": 1, "name": name, "primary": True,
            "downloadUrl": f"{hf_env}/dl/{name}",
            "sizeBytes": len(data),
            "hashes": {"SHA256": hashlib.sha256(data).hexdigest()},
        }],
    }
    monkeypatch.setattr(dl, "_CIVITAI_API", f"{hf_env}/api/v1")
    monkeypatch.setenv("COZY_CIVITAI_DOWNLOAD_ATTEMPTS", "2")

    out = dl.download_civitai(123, tmp_path / "civi")
    assert Path(out).read_bytes() == data
    assert hf_server.get_counts[name] == 2  # one drop, one clean re-download
