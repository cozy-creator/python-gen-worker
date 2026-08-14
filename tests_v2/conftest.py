"""tests_v2 shared fixtures — the v2 suite's ONE fixture surface.

Every v2 suite builds on exactly this interface; read this docstring before
writing a new suite. House style is non-negotiable: real worker over real
gRPC against the hub double, real ModelStore/CAS/files/subprocesses, zero
mocks, and every wait progress-gated (`harness.progress_wait`) — a wall-clock
deadline whose expiry FAILS a test is the forbidden shape.

Fixture / helper interface
--------------------------
``hub`` (fixture, factory)
    ``with hub() as (scheduler, harness): ...`` — one hub-double gRPC server
    plus one REAL in-process ``Worker`` loaded with ``tests_v2.catalog``.
    Keyword args pass through to ``harness.hub_double.hub_double`` (e.g.
    ``modules=``, ``file_base_url=``, ``gpu_slots=``, ``worker_id=``).
    ``scheduler.wait_connection(0)`` -> ``Conn`` with progress-gated
    ``wait_for(pred)`` / ``wait_for_count(pred, n)`` / ``count(pred)``.

``blob_host`` (fixture)
    A real blake3-addressed HTTP blob host (``harness.blob_host.BlobHost``):
    ``blob_host.one_file_snapshot(digest, name, payload)`` -> ``pb.Snapshot``
    whose file URLs the worker really GETs and verifies.

``upload_sink`` (fixture) / ``UploadSink`` (class)
    A real local HTTP stand-in for tensorhub's media-upload endpoint. The
    fixture yields a 200-dedup sink; instantiate ``UploadSink(status=403)``
    for refusal rows. ``sink.base_url`` feeds ``hub(file_base_url=...)``;
    ``sink.requests`` is the recorded ``[(path, body_dict), ...]``.

``standalone_scheduler()`` (context manager)
    A hub-double gRPC server with NO in-process worker — for REAL
    ``python -m gen_worker.entrypoint`` subprocess boots that dial in over
    localhost TCP. Yields ``(FakeScheduler, port)``.

``spawn_entrypoint(tmp_path, functions=..., env_overrides=...)`` (helper)
    Launches the real entrypoint as a subprocess against a baked manifest and
    returns an ``EntrypointProc``: ``.wait_for_output(pred)`` (progress-gated
    on output growth), ``.output()``, ``.phases()`` (parsed
    ``worker.startup.phase`` / ``worker.fatal`` lines), ``.send_signal()``,
    ``.terminate_and_wait()``. ``manifest_entry(...)`` builds function rows;
    by default they point at ``tests_v2.catalog``.

``torchless`` (fixture)
    in-process torch absence: blocks ``import torch`` via a meta-path finder
    and strips cached torch modules for the duration.

From the old harness (import directly; they are part of this interface):
    ``harness.hub_double``: ``hub_double``, ``custom_scheduler_server``,
    ``FakeScheduler``, ``is_ready``, ``is_result_for``, ``is_accept_for``,
    ``is_model_event``, ``is_fn_unavailable``.
    ``harness.progress_wait``: ``Cadence``, ``StalledError``, ``await_count``,
    ``await_progress``.
    ``harness.subprocess_runner``: ``run_entrypoint`` (blocking boot->exit),
    ``startup_phase_lines``, ``assert_no_unhandled_crash``.
    ``harness.hardware_report_hub``: ``recording_hub``, ``closed_port_addr``.

Endpoints come from ``tests_v2.catalog`` (see its docstring): scenarios select
``catalog.CATALOG`` rows; a new behavior is a new row, never a new module.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# The old tests/harness package is a first-class asset of v2 (hub double,
# progress-gated waits, blob host); make it importable when only tests_v2 runs.
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# point the postmortem carriers off the host BEFORE anything imports
# gen_worker.postmortem (paths resolve once at import). Subprocess boots
# inherit os.environ, and a crash streak recorded into the shared /tmp
# registry poisons every later boot in every lane.
os.environ.setdefault(
    "GEN_WORKER_BOOT_RECORD",
    str(Path(tempfile.mkdtemp(prefix="pgw-v2-postmortem-")) / "boot-record.json"),
)
import gen_worker  # noqa: E402

_LOCATION = Path(gen_worker.__file__).resolve()
if REPO_ROOT / "src" not in _LOCATION.parents:
    raise RuntimeError(
        f"gen_worker is imported from {_LOCATION}, NOT this repo's src/. A "
        "stale install shadows the working tree; run via `uv run --extra dev "
        "pytest` (see tests/conftest.py)."
    )

import msgspec  # noqa: E402  (used by manifest baking)
import pytest  # noqa: E402

from gen_worker import boot_phases  # noqa: E402
from gen_worker import config as gw_config  # noqa: E402
from gen_worker import worker_goals as gw_worker_goals  # noqa: E402

from harness.blob_host import BlobHost  # noqa: E402
from harness.hardware_report_hub import closed_port_addr  # noqa: E402
from harness.hub_double import FakeScheduler, hub_double  # noqa: E402
from harness.progress_wait import Cadence, StalledError  # noqa: E402

# ---------------------------------------------------------------------------
# Process-global state hygiene (mirrors tests/conftest.py; production state is
# process-lifetime by design, so each test starts and ends clean).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_process_settings():
    """pgw#931: Settings are PUBLISHED by a process entry, not cached lazily.
    Each test starts from a clean install over its own env (see
    tests/conftest.py for the full rationale)."""
    gw_config.reset_for_test()
    gw_worker_goals.reset_for_test()
    gw_config.install(gw_config.load_settings())
    gw_worker_goals.install(gw_worker_goals.SERVE_ONLY)
    yield
    gw_config.reset_for_test()
    gw_worker_goals.reset_for_test()


def pytest_configure(config):
    # Same treatment as tests/conftest.py — the declared
    # interpreter env (PYTHONHASHSEED=0), imposed by ONE re-exec with global
    # capture stopped first so the re-exec'd run still owns the terminal.
    # xdist workers inherit the env from the re-exec'd master.
    from gen_worker.settings_authority import (
        _interpreter_env_diffs, ensure_interpreter_env)

    if not _interpreter_env_diffs():
        return
    capman = config.pluginmanager.getplugin("capturemanager")
    if capman is not None:
        capman.stop_global_capturing()
    ensure_interpreter_env()  # execs; never returns (or raises for -E)


@pytest.fixture(autouse=True)
def _fresh_boot_phases():
    """boot_phases is process-global and one-shot per boot; without a reset a
    second in-process boot records nothing (the pgw#797 suppression shape)."""
    boot_phases.reset_for_tests()
    yield
    boot_phases.reset_for_tests()


@pytest.fixture(autouse=True)
def _fresh_receipt_gate():
    """The pgw#709 receipt gate arms at HelloAck and stays armed for the
    process; every hub-double test walks a HelloAck, so clear it."""
    from gen_worker import receipts

    receipts._reset_for_tests()
    yield
    receipts._reset_for_tests()


@pytest.fixture(autouse=True)
def _fresh_cell_ledgers():
    from gen_worker import compile_cache as _cc
    from gen_worker import fleet_cells as _fc

    def _clear() -> None:
        with _cc._PROVEN_CELLS_LOCK:
            _cc._QUARANTINED_CELLS.clear()
        with _fc._PENDING_LOCK:
            _fc._FINALIZED.clear()

    _clear()
    yield
    _clear()


@pytest.fixture(scope="session")
def _postmortem_root(tmp_path_factory):
    return tmp_path_factory.mktemp("postmortem")


@pytest.fixture(autouse=True)
def _postmortem_paths_off_the_host(_postmortem_root):
    """pgw#801: the postmortem carriers are HOST paths; the suite must never
    read or write the shared production ones (measured cross-lane poisoning)."""
    from gen_worker import postmortem as _pm

    names = ("BOOT_RECORD_PATH", "INFLIGHT_PATH",
             "CRASH_REGISTRY_PATH", "FAULT_DUMP_PATH")
    saved = {name: getattr(_pm, name) for name in names}
    redirected = [_postmortem_root / path.name for path in saved.values()]

    def _wipe() -> None:
        for path in redirected:
            path.unlink(missing_ok=True)

    for name, path in zip(names, redirected):
        setattr(_pm, name, path)
    _wipe()
    yield
    _wipe()
    for name, path in saved.items():
        setattr(_pm, name, path)


# ---------------------------------------------------------------------------
# The hub-double factory (in-process real Worker over a real gRPC socket).
# ---------------------------------------------------------------------------


@pytest.fixture
def hub():
    """Factory: ``with hub(**kw) as (scheduler, harness)``. Defaults the
    worker's endpoint modules to the declarative catalog."""

    def _hub(**kw: Any):
        kw.setdefault("modules", ("tests_v2.catalog",))
        return hub_double(**kw)

    return _hub


@pytest.fixture
def blob_host(tmp_path: Path) -> Iterator[BlobHost]:
    host = BlobHost(tmp_path)
    try:
        yield host
    finally:
        host.shutdown()


# ---------------------------------------------------------------------------
# Real local upload sink (tensorhub media-upload stand-in).
# ---------------------------------------------------------------------------


#: The canonical org-less create route (th#1722 §C / pgw#1138). The org is
#: derived from the CREDENTIAL and is never a path segment.
MEDIA_UPLOADS_PATH = "/api/v1/media/uploads"

#: tensorhub's org-less media-upload family, mirroring
#: `registerMediaUploadRoutes(v1, "/media")` in `internal/api/files.go`.
#: Patterns, not prefixes, so `/api/v1/media/<org>/uploads` — the transitional
#: alias th#1799 deletes — cannot match. Deliberately duplicated from
#: `tests/harness/upload_sink.py` rather than imported: a cross-suite import
#: would make the v1 harness load-bearing for v2.
_ORG_LESS_UPLOAD_ROUTES = tuple(
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


class UploadSink:
    """Real HTTP sink for the worker's result-blob upload path.

    ``status=200`` answers a dedup create (no S3 part scripting needed);
    any other status is returned verbatim — the refusal rows. Records every
    POST as ``(path, decoded_json_body)`` in ``self.requests``.

    It ROUTES: a path outside tensorhub's org-less upload family 404s exactly
    as gin does, so a client's URL construction is testable end to end rather
    than assumed by an assertion the sink itself could not contradict.
    """

    def __init__(self, status: int = 200) -> None:
        self.requests: List[Tuple[str, Dict[str, Any]]] = []
        self.rejected: List[str] = []
        sink = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                bare = self.path.split("?", 1)[0]
                if not any(rx.match(bare) for rx in _ORG_LESS_UPLOAD_ROUTES):
                    sink.rejected.append(self.path)
                    payload = json.dumps({"error": "not_found"}).encode()
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                body = json.loads(raw or b"{}")
                sink.requests.append((self.path, body))
                if sink.status != 200:
                    payload = json.dumps({"error": "refused by test sink"}).encode()
                    self.send_response(sink.status)
                else:
                    payload = json.dumps({
                        "dedup": True, "ref": body.get("ref") or "",
                        "filename": "out.msgpack",
                        "blake3": body.get("blake3") or "",
                        "size_bytes": body.get("size_bytes") or 0,
                        "mime_type": "application/octet-stream", "media_id": "m1",
                    }).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.status = status
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def shutdown(self) -> None:
        self._httpd.shutdown()


@pytest.fixture
def upload_sink() -> Iterator[UploadSink]:
    sink = UploadSink()
    try:
        yield sink
    finally:
        sink.shutdown()


# ---------------------------------------------------------------------------
# Real entrypoint subprocess against a live scheduler socket.
# ---------------------------------------------------------------------------


@contextmanager
def standalone_scheduler() -> Iterator[Tuple[FakeScheduler, int]]:
    """A hub-double server with NO in-process worker: subprocess boots dial it."""
    import grpc

    from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc

    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield scheduler, port
    finally:
        server.stop(grace=0)


def manifest_entry(
    *, name: str = "echo", module: str = "tests_v2.catalog",
    kind: str = "inference", gpu: bool = False,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"name": name, "module": module, "kind": kind}
    entry["resources"] = {"gpu": True} if gpu else {}
    return entry


class EntrypointProc:
    """One live ``python -m gen_worker.entrypoint`` subprocess with drained,
    inspectable output. All waits are progress-gated on output growth or on
    process death — never a wall clock."""

    def __init__(self, proc: "subprocess.Popen[str]") -> None:
        self.proc = proc
        self._chunks: List[str] = []
        self._cond = threading.Condition()
        self._readers = [
            threading.Thread(target=self._drain, args=(proc.stdout,), daemon=True),
            threading.Thread(target=self._drain, args=(proc.stderr,), daemon=True),
        ]
        for r in self._readers:
            r.start()

    def _drain(self, stream: Any) -> None:
        for line in iter(stream.readline, ""):
            with self._cond:
                self._chunks.append(line)
                self._cond.notify_all()
        stream.close()

    def output(self) -> str:
        with self._cond:
            return "".join(self._chunks)

    def phases(self) -> List[Dict[str, Any]]:
        from harness.subprocess_runner import startup_phase_lines

        return startup_phase_lines(self.output())

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait_for_output(self, pred: Callable[[str], bool], what: str = "output") -> str:
        """Progress-gated: gives up only when the process is dead AND the
        streams are drained, or when output itself has gone stale."""
        cadence = Cadence()
        last_len, last_advance = -1, time.monotonic()
        while True:
            with self._cond:
                text = "".join(self._chunks)
            if pred(text):
                return text
            drained = all(not r.is_alive() for r in self._readers)
            if not self.alive and drained:
                raise StalledError(
                    f"waiting for {what}: the entrypoint exited "
                    f"(rc={self.proc.returncode}) without producing it.\n{text}"
                )
            now = time.monotonic()
            if len(text) != last_len:
                if last_len >= 0:
                    cadence.record(now - last_advance)
                last_len, last_advance = len(text), now
            elif now - last_advance >= cadence.window_s:
                raise StalledError(
                    f"waiting for {what}: no new output in "
                    f"{now - last_advance:.1f}s ({cadence.describe()}).\n{text}"
                )
            with self._cond:
                self._cond.wait(0.05)

    def send_signal(self, sig: int) -> None:
        self.proc.send_signal(sig)

    def terminate_and_wait(self) -> int:
        if self.alive:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=Cadence().floor_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for r in self._readers:
            r.join(timeout=5.0)
        return int(self.proc.returncode)


def spawn_entrypoint(
    tmp_path: Path,
    *,
    functions: List[Dict[str, Any]],
    env_overrides: Optional[Dict[str, str]] = None,
) -> EntrypointProc:
    """Launch the REAL entrypoint module with a baked manifest. The default
    hello target is a definitively-closed port; pass ORCHESTRATOR_PUBLIC_ADDR
    in ``env_overrides`` to dial a live ``standalone_scheduler()``."""
    manifest_path = tmp_path / "endpoint.lock"
    manifest_path.write_bytes(msgspec.toml.encode({"functions": functions}))
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")]),
        "ORCHESTRATOR_PUBLIC_ADDR": closed_port_addr(),
        "TENSORHUB_CACHE_DIR": str(tmp_path / "cache"),
        "ENDPOINT_LOCK_PATH": str(manifest_path),
        "GEN_WORKER_BOOT_RECORD": str(tmp_path / "boot-record.json"),
        "GEN_WORKER_VIDEO_ENCODER": "x264",
    }
    env.update(env_overrides or {})
    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.entrypoint"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return EntrypointProc(proc)


# ---------------------------------------------------------------------------
# In-process torch absence (pgw#788 shape).
# ---------------------------------------------------------------------------


@pytest.fixture
def torchless(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make ``import torch`` raise ImportError for the duration, exactly like
    a torchless image; cached torch modules are stripped and restored."""
    import importlib.abc

    class _BlockTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
            if fullname == "torch" or fullname.startswith("torch."):
                raise ImportError(f"tests_v2 torchless: {fullname} is not installed")
            return None

    finder = _BlockTorch()
    saved = dict(sys.modules)
    for name in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(ImportError):
            __import__("torch")
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)
