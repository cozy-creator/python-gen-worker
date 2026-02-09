from __future__ import annotations

import json
import sys
import time
from concurrent import futures
from pathlib import Path
from threading import Thread
from typing import Optional

import grpc
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.testing.mock_orchestrator import _MockOrchestrator
from gen_worker.worker import Worker


def _extract_bearer(md: tuple[tuple[str, str], ...]) -> Optional[str]:
    for k, v in md:
        if k.lower() == "authorization":
            vv = v.strip()
            if vv.lower().startswith("bearer "):
                return vv.split(" ", 1)[1].strip()
    return None


def test_worker_jwt_rotation_updates_next_reconnect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Start mock orchestrator server.
    orch = _MockOrchestrator()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_SchedulerWorkerServiceServicer_to_server(orch, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    # Create a tiny user module.
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "hello_mod.py").write_text(
        """
from __future__ import annotations

import msgspec
from gen_worker import ActionContext, ResourceRequirements, worker_function

class Input(msgspec.Struct):
    name: str

class Output(msgspec.Struct):
    message: str

@worker_function(ResourceRequirements())
def hello(ctx: ActionContext, payload: Input) -> Output:
    return Output(message=f"hello {payload.name}")
""".lstrip(),
        encoding="utf-8",
    )
    sys.path.insert(0, str(mod_dir))

    w = Worker(
        scheduler_addr=f"127.0.0.1:{port}",
        user_module_names=["hello_mod"],
        worker_jwt="jwt-1",
        reconnect_delay=0,
    )

    t = Thread(target=w.run, daemon=True)
    t.start()
    try:
        sess = orch.get_session(timeout_s=30.0)
        assert sess is not None
        assert _extract_bearer(sess.metadata) == "jwt-1"

        # Send rotation signal over the stream. Worker stores it for next reconnect.
        payload = json.dumps({"worker_jwt": "jwt-2"}, separators=(",", ":"), sort_keys=True).encode("utf-8")
        sess.send(pb.WorkerSchedulerMessage(worker_event=pb.WorkerEvent(run_id="", event_type="worker.jwt.rotate", payload_json=payload)))

        # Ensure the worker processed the rotation signal before we force a reconnect.
        start = time.monotonic()
        while time.monotonic() - start < 10:
            if w.worker_jwt == "jwt-2":
                break
            time.sleep(0.05)
        assert w.worker_jwt == "jwt-2"

        # Force a reconnect without killing the server. This simulates network interruption.
        w._handle_connection_error()

        # Wait for a new connection to establish with the rotated token.
        start = time.monotonic()
        while time.monotonic() - start < 30:
            sess2 = orch.get_session(timeout_s=0.5)
            if sess2 is None:
                continue
            if _extract_bearer(sess2.metadata) == "jwt-2":
                return
            time.sleep(0.1)

        raise AssertionError("timed out waiting for reconnect with rotated WORKER_JWT")
    finally:
        try:
            w.stop()
        except Exception:
            pass
        t.join(timeout=5)
        server.stop(grace=None)
