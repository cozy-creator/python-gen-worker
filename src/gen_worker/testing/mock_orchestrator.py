from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import unquote

import grpc
import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc


class _FileAPIHandler(BaseHTTPRequestHandler):
    # Set by factory
    files_dir: Path
    token: str

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Silence stdlib http server logs by default; keep output focused on gRPC.
        return

    def _check_auth(self) -> bool:
        token = (self.token or "").strip()
        if not token:
            return True
        got = (self.headers.get("Authorization") or "").strip()
        return got == f"Bearer {token}"

    def _send_json(self, code: int, obj: dict[str, object]) -> None:
        raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def _ref_from_path(self) -> str:
        # expected: /api/v1/file/<urlencoded_ref>
        prefix = "/api/v1/file/"
        if not self.path.startswith(prefix):
            raise ValueError("invalid path")
        return unquote(self.path[len(prefix):]).lstrip("/")

    def _fs_path(self, ref: str) -> Path:
        # Keep it simple: write under files_dir, creating directories as needed.
        # Prevent path traversal.
        ref = ref.replace("\\", "/")
        ref = ref.lstrip("/")
        out = (self.files_dir / ref).resolve()
        base = self.files_dir.resolve()
        if base not in out.parents and out != base:
            raise ValueError("path traversal")
        return out

    def do_PUT(self) -> None:  # noqa: N802
        if not self._check_auth():
            self._send_json(401, {"error": "unauthorized"})
            return
        try:
            ref = self._ref_from_path()
            dst = self._fs_path(ref)
        except Exception:
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except Exception:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(body)
        sha = hashlib.sha256(body).hexdigest()
        self._send_json(200, {"ref": ref, "size_bytes": len(body), "sha256": sha})

    def do_POST(self) -> None:  # noqa: N802
        # Create behaves like overwrite for dev.
        self.do_PUT()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._check_auth():
            self.send_response(401)
            self.end_headers()
            return
        try:
            ref = self._ref_from_path()
            dst = self._fs_path(ref)
        except Exception:
            self.send_response(404)
            self.end_headers()
            return
        if not dst.exists():
            self.send_response(404)
            self.end_headers()
            return
        size = dst.stat().st_size
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._check_auth():
            self.send_response(401)
            self.end_headers()
            return
        try:
            ref = self._ref_from_path()
            dst = self._fs_path(ref)
        except Exception:
            self.send_response(404)
            self.end_headers()
            return
        if not dst.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = dst.read_bytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(data)


def _start_file_api_server(listen: str, files_dir: str, token: str) -> ThreadingHTTPServer:
    host, port_s = listen.rsplit(":", 1)
    port = int(port_s)
    root = Path(files_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    def handler_factory(*args: object, **kwargs: object) -> _FileAPIHandler:
        h = _FileAPIHandler(*args, **kwargs)  # type: ignore[misc]
        return h

    # Set shared config on handler class.
    _FileAPIHandler.files_dir = root
    _FileAPIHandler.token = token

    httpd = ThreadingHTTPServer((host, port), handler_factory)  # type: ignore[arg-type]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


@dataclass
class WorkerSession:
    worker_id: str
    release_id: str
    available_functions: Tuple[str, ...]
    function_schemas: Dict[str, pb.FunctionSchema]
    metadata: Tuple[Tuple[str, str], ...]

    _out_q: "queue.Queue[pb.WorkerSchedulerMessage]"
    _in_q: "queue.Queue[pb.WorkerSchedulerMessage]"
    _closed: threading.Event

    def send(self, msg: pb.WorkerSchedulerMessage) -> None:
        if self._closed.is_set():
            raise RuntimeError("worker session is closed")
        self._out_q.put_nowait(msg)

    def recv(self, timeout_s: float) -> Optional[pb.WorkerSchedulerMessage]:
        try:
            return self._in_q.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def run_task(
        self,
        *,
        function_name: str,
        payload_obj: Any,
        run_id: Optional[str] = None,
        owner: str = "",
        user_id: str = "",
        timeout_ms: int = 0,
        required_variant_refs: Tuple[str, ...] = (),
        file_base_url: str = "",
        file_token: str = "",
    ) -> str:
        rid = run_id or str(uuid.uuid4())
        raw = msgspec.msgpack.encode(payload_obj)
        req = pb.TaskExecutionRequest(
            run_id=rid,
            function_name=function_name,
            input_payload=raw,
            required_variant_refs=list(required_variant_refs),
            timeout_ms=timeout_ms,
            owner=owner,
            user_id=user_id,
            file_base_url=file_base_url,
            file_token=file_token,
        )
        self.send(pb.WorkerSchedulerMessage(run_request=req))
        return rid


class _MockOrchestrator(pb_grpc.SchedulerWorkerServiceServicer):
    def __init__(self) -> None:
        self._session_lock = threading.Lock()
        self._session_ready = threading.Event()
        self._session: Optional[WorkerSession] = None

    def get_session(self, timeout_s: float) -> Optional[WorkerSession]:
        if not self._session_ready.wait(timeout=timeout_s):
            return None
        with self._session_lock:
            return self._session

    def ConnectWorker(  # type: ignore[override]
        self, request_iterator: Iterator[pb.WorkerSchedulerMessage], context: grpc.ServicerContext
    ) -> Iterator[pb.WorkerSchedulerMessage]:
        out_q: "queue.Queue[pb.WorkerSchedulerMessage]" = queue.Queue()
        in_q: "queue.Queue[pb.WorkerSchedulerMessage]" = queue.Queue()
        closed = threading.Event()

        md = tuple((m.key, m.value) for m in context.invocation_metadata())

        def reader() -> None:
            try:
                for msg in request_iterator:
                    in_q.put_nowait(msg)
            except Exception:
                pass
            finally:
                closed.set()

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        # Wait for initial registration to populate the session.
        worker_id = ""
        release_id = ""
        available_functions: Tuple[str, ...] = ()
        function_schemas: Dict[str, pb.FunctionSchema] = {}

        start = time.monotonic()
        while time.monotonic() - start < 30:
            if closed.is_set():
                break
            msg = None
            try:
                msg = in_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if msg is None:
                continue
            if msg.HasField("worker_registration"):
                reg = msg.worker_registration
                res = reg.resources
                worker_id = res.worker_id
                release_id = res.release_id
                available_functions = tuple(res.available_functions)
                for fs in res.function_schemas:
                    function_schemas[fs.name] = fs
                session = WorkerSession(
                    worker_id=worker_id,
                    release_id=release_id,
                    available_functions=available_functions,
                    function_schemas=function_schemas,
                    metadata=md,
                    _out_q=out_q,
                    _in_q=in_q,
                    _closed=closed,
                )
                with self._session_lock:
                    self._session = session
                    self._session_ready.set()
                break

        # Main send loop.
        while not closed.is_set() and context.is_active():
            try:
                msg = out_q.get(timeout=0.25)
            except queue.Empty:
                continue
            yield msg


def _parse_payload_json(payload_json: str) -> Any:
    try:
        return json.loads(payload_json)
    except Exception as e:
        raise SystemExit(f"invalid --payload-json: {e}")


def _format_msg(msg: pb.WorkerSchedulerMessage) -> str:
    if msg.HasField("worker_registration"):
        return "worker_registration"
    if msg.HasField("run_result"):
        rr = msg.run_result
        return f"run_result run_id={rr.run_id} success={rr.success} error_type={rr.error_type!r} retryable={rr.retryable}"
    if msg.HasField("worker_event"):
        ev = msg.worker_event
        return f"worker_event run_id={ev.run_id} type={ev.event_type}"
    if msg.HasField("load_model_result"):
        return "load_model_result"
    if msg.HasField("unload_model_result"):
        return "unload_model_result"
    if msg.HasField("interrupt_run_cmd"):
        return "interrupt_run_cmd"
    if msg.HasField("deployment_artifact_config"):
        return "deployment_artifact_config"
    if msg.HasField("realtime_open_cmd"):
        return "realtime_open_cmd"
    if msg.HasField("realtime_frame"):
        return "realtime_frame"
    if msg.HasField("realtime_close_cmd"):
        return "realtime_close_cmd"
    if msg.HasField("run_request"):
        return "run_request"
    if msg.HasField("load_model_cmd"):
        return "load_model_cmd"
    if msg.HasField("unload_model_cmd"):
        return "unload_model_cmd"
    return "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mock-orchestrator",
        description=(
            "One-off local invoke helper for python-gen-worker. "
            "Starts a temporary mock scheduler, waits for a worker to connect, "
            "sends one task, prints the result, and exits."
        ),
    )
    ap.add_argument("--listen", default="0.0.0.0:8080", help="address to listen on (host:port)")
    ap.add_argument("--wait-s", type=float, default=30.0, help="seconds to wait for a worker to connect")

    ap.add_argument(
        "--run",
        dest="function_name",
        required=True,
        help="function name to run once a worker connects",
    )
    ap.add_argument("--payload-json", default="{}", help="JSON payload to encode and send")
    ap.add_argument("--timeout-ms", type=int, default=0)
    ap.add_argument("--owner", default="")
    ap.add_argument("--user-id", default="")
    ap.add_argument("--required-model", action="append", default=[], help="repeatable model id/key list for injections")

    ap.add_argument("--file-base-url", default="", help="file API base URL to pass to the worker (TaskExecutionRequest.file_base_url)")
    ap.add_argument("--file-token", default="", help="file token to pass to the worker (TaskExecutionRequest.file_token)")
    ap.add_argument("--serve-files-dir", default="", help="if set, start a tiny dev file API server writing into this directory")
    ap.add_argument("--serve-files-listen", default="0.0.0.0:8081", help="listen address for --serve-files-dir server (host:port)")

    ap.add_argument("--print-events", action="store_true", help="print worker_event messages during run")
    ap.add_argument(
        "--print-registrations",
        action="store_true",
        help="print worker_registration heartbeats during run",
    )

    args = ap.parse_args(argv)

    orch = _MockOrchestrator()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_SchedulerWorkerServiceServicer_to_server(orch, server)
    server.add_insecure_port(args.listen)
    server.start()
    file_server: Optional[ThreadingHTTPServer] = None
    if args.serve_files_dir:
        file_server = _start_file_api_server(args.serve_files_listen, args.serve_files_dir, args.file_token)
        print(f"[file-api] listening={args.serve_files_listen} dir={args.serve_files_dir!r}")
    try:
        sess = orch.get_session(timeout_s=float(args.wait_s))
        if sess is None:
            print(f"timed out waiting for worker to connect to {args.listen}", file=sys.stderr)
            return 2

        print(f"[connected] worker_id={sess.worker_id!r} release_id={sess.release_id!r}")
        if sess.available_functions:
            print(f"[connected] functions={list(sess.available_functions)}")
        if sess.metadata:
            print(f"[connected] metadata={list(sess.metadata)}")

        payload_obj = _parse_payload_json(args.payload_json)
        run_id = sess.run_task(
            function_name=args.function_name,
            payload_obj=payload_obj,
            timeout_ms=int(args.timeout_ms),
            owner=args.owner,
            user_id=args.user_id,
            required_variant_refs=tuple(args.required_model),
            file_base_url=str(args.file_base_url or ""),
            file_token=str(args.file_token or ""),
        )
        print(f"[sent] run_id={run_id} function={args.function_name!r}")

        # Wait for result.
        start = time.monotonic()
        while True:
            msg = sess.recv(timeout_s=0.5)
            if msg is None:
                if time.monotonic() - start > max(5.0, float(args.wait_s)):
                    print("timed out waiting for run_result", file=sys.stderr)
                    return 3
                continue
            if msg.HasField("worker_event") and args.print_events:
                try:
                    payload = msg.worker_event.payload_json.decode("utf-8", errors="replace")
                except Exception:
                    payload = "<binary>"
                print(f"[event] {msg.worker_event.event_type}: {payload}")
                continue
            if msg.HasField("run_result") and msg.run_result.run_id == run_id:
                rr = msg.run_result
                if rr.output_payload:
                    out_obj = msgspec.msgpack.decode(rr.output_payload)
                    print("[output]", json.dumps(out_obj, indent=2, sort_keys=True))
                else:
                    print("[output] <empty>")
                if not rr.success:
                    print(
                        f"[error] type={rr.error_type!r} retryable={rr.retryable} safe={rr.safe_message!r}",
                        file=sys.stderr,
                    )
                    if rr.error_message:
                        print(f"[error] message={rr.error_message}", file=sys.stderr)
                    return 1
                return 0
            if msg.HasField("worker_registration") and args.print_registrations:
                print("[registration]")
                continue
    finally:
        server.stop(grace=None)
        if file_server is not None:
            try:
                file_server.shutdown()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
