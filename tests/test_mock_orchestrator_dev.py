from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gen_worker.testing.mock_orchestrator import _MockOrchestrator
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc


@pytest.mark.skipif(os.getenv("COZY_DEV_GRPC_E2E") != "1", reason="set COZY_DEV_GRPC_E2E=1 to run gRPC e2e")
def test_mock_orchestrator_can_run_one_task(tmp_path: Path) -> None:
    # Start mock orchestrator server.
    orch = _MockOrchestrator()
    import grpc
    from concurrent import futures

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_SchedulerWorkerServiceServicer_to_server(orch, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    # Write a tiny module with a single worker function into a temp dir.
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

    # Start a worker process that connects to our mock orchestrator.
    env = dict(os.environ)
    env["SCHEDULER_ADDR"] = f"127.0.0.1:{port}"
    (tmp_path / "cozy.toml").write_text(
        """
schema_version = 1
name = "dev-test"
main = "hello_mod"
gen_worker = ">=0"
""".lstrip(),
        encoding="utf-8",
    )
    env["COZY_MANIFEST_PATH"] = str(tmp_path / "cozy.toml")
    env["PYTHONPATH"] = f"{mod_dir}:{env.get('PYTHONPATH','')}"
    env["WORKER_ID"] = "dev-test"
    env["WORKER_JWT"] = "dev-test-jwt"

    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.entrypoint"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        sess = orch.get_session(timeout_s=30.0)
        assert sess is not None
        assert "hello" in sess.available_functions

        run_id = sess.run_task(function_name="hello", payload_obj={"name": "world"})

        # Wait for a result.
        start = time.monotonic()
        while time.monotonic() - start < 30:
            msg = sess.recv(timeout_s=0.5)
            if msg is None:
                continue
            if msg.HasField("run_result") and msg.run_result.run_id == run_id:
                assert msg.run_result.success is True
                assert msg.run_result.output_payload
                import msgspec

                out = msgspec.msgpack.decode(msg.run_result.output_payload)
                assert out["message"] == "hello world"
                return

        raise AssertionError("timed out waiting for run_result")
    finally:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        server.stop(grace=None)
