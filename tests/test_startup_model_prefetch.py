from __future__ import annotations

import queue
import threading
import time
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator
from urllib.parse import unquote

import grpc
import pytest
from blake3 import blake3

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.worker import Worker


class _FileHandler(BaseHTTPRequestHandler):
    files_dir: Path

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        prefix = "/api/v1/file/"
        if not self.path.startswith(prefix):
            self.send_response(404)
            self.end_headers()
            return
        rel = unquote(self.path[len(prefix) :]).lstrip("/")
        dst = (self.files_dir / rel).resolve()
        root = self.files_dir.resolve()
        if (root not in dst.parents and dst != root) or not dst.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = dst.read_bytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(data)


def _start_file_server(files_dir: Path) -> ThreadingHTTPServer:
    files_dir.mkdir(parents=True, exist_ok=True)
    _FileHandler.files_dir = files_dir
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FileHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


class _WorkerSession:
    def __init__(
        self,
        out_q: "queue.Queue[pb.WorkerSchedulerMessage]",
        in_q: "queue.Queue[pb.WorkerSchedulerMessage]",
    ) -> None:
        self._out_q = out_q
        self._in_q = in_q

    def send(self, msg: pb.WorkerSchedulerMessage) -> None:
        self._out_q.put_nowait(msg)

    def recv(self, timeout_s: float) -> pb.WorkerSchedulerMessage | None:
        try:
            return self._in_q.get(timeout=timeout_s)
        except queue.Empty:
            return None


class _Scheduler(pb_grpc.SchedulerWorkerServiceServicer):
    def __init__(self) -> None:
        self._ready = threading.Event()
        self._session: _WorkerSession | None = None

    def get_session(self, timeout_s: float) -> _WorkerSession | None:
        if not self._ready.wait(timeout=timeout_s):
            return None
        return self._session

    def ConnectWorker(  # type: ignore[override]
        self,
        request_iterator: Iterator[pb.WorkerSchedulerMessage],
        context: grpc.ServicerContext,
    ) -> Iterator[pb.WorkerSchedulerMessage]:
        del context
        out_q: "queue.Queue[pb.WorkerSchedulerMessage]" = queue.Queue()
        in_q: "queue.Queue[pb.WorkerSchedulerMessage]" = queue.Queue()
        closed = threading.Event()

        def reader() -> None:
            try:
                for msg in request_iterator:
                    in_q.put_nowait(msg)
            except Exception:
                pass
            finally:
                closed.set()

        threading.Thread(target=reader, daemon=True).start()

        start = time.monotonic()
        while time.monotonic() - start < 30:
            if closed.is_set():
                break
            msg = None
            try:
                msg = in_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if msg is not None and msg.HasField("worker_registration"):
                self._session = _WorkerSession(out_q, in_q)
                self._ready.set()
                break

        while not closed.is_set():
            try:
                yield out_q.get(timeout=0.25)
            except queue.Empty:
                continue


def test_startup_prefetch_warms_disk_and_reports_disk_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-test")
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKER_MAX_CONCURRENT_DOWNLOADS", "1")

    # Start a tiny HTTP file server to act as the "presigned URL" origin.
    files_dir = tmp_path / "files"
    httpd = _start_file_server(files_dir)
    host, port = httpd.server_address

    # Create one small file that our resolved manifest will reference.
    rel_path = "models/weights.bin"
    data = b"hello-model-bytes"
    out = files_dir / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    url = f"http://{host}:{port}/api/v1/file/{rel_path}"

    b3 = blake3(data).hexdigest()

    # Start mock orchestrator gRPC server.
    orch = _Scheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_SchedulerWorkerServiceServicer_to_server(orch, server)
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    variant_ref = "cozy:demo/repo@blake3:snap-1"

    w = Worker(
        scheduler_addr=f"127.0.0.1:{grpc_port}",
        user_module_names=[],
        worker_jwt="jwt-1",
        reconnect_delay=0,
    )
    t = Thread(target=w.run, daemon=True)
    t.start()
    try:
        sess = orch.get_session(timeout_s=30.0)
        assert sess is not None

        cfg = pb.EndpointConfig(
            supported_repo_refs=[variant_ref],
            required_flavor_refs=[variant_ref],
            resolved_cozy_models_by_flavor_ref={
                variant_ref: pb.ResolvedCozyModel(
                    snapshot_digest="snap-1",
                    files=[
                        pb.ResolvedCozyModelFile(
                            path="weights.bin",
                            size_bytes=len(data),
                            blake3=b3,
                            url=url,
                        )
                    ],
                )
            },
        )
        sess.send(pb.WorkerSchedulerMessage(endpoint_config=cfg))

        # The worker prefetch thread triggers an immediate registration update after caching.
        start = time.monotonic()
        saw_started = False
        saw_completed = False
        disk_ready = False
        while time.monotonic() - start < 30:
            msg = sess.recv(timeout_s=0.5)
            if msg is None:
                continue
            if msg.HasField("worker_event"):
                et = msg.worker_event.event_type
                if et == "model.download.started":
                    saw_started = True
                if et == "model.download.completed":
                    saw_completed = True
            if not msg.HasField("worker_registration"):
                if disk_ready and saw_started and saw_completed:
                    return
                continue
            disk_models = list(msg.worker_registration.resources.disk_models)
            if variant_ref in disk_models:
                disk_ready = True
            if disk_ready and saw_started and saw_completed:
                return

        raise AssertionError("timed out waiting for worker to report disk_models after startup prefetch")
    finally:
        try:
            w.stop()
        except Exception:
            pass
        t.join(timeout=5)
        server.stop(grace=None)
        httpd.shutdown()
