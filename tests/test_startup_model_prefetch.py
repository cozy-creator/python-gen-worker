from __future__ import annotations

import time
from concurrent import futures
from pathlib import Path
from threading import Thread

import grpc
import pytest
from blake3 import blake3

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.testing.mock_orchestrator import _MockOrchestrator, _start_file_api_server
from gen_worker.worker import Worker


def test_startup_prefetch_warms_disk_and_reports_disk_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-test")
    monkeypatch.setenv("WORKER_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WORKER_MAX_CONCURRENT_DOWNLOADS", "1")

    # Start a tiny HTTP file server to act as the "presigned URL" origin.
    files_dir = tmp_path / "files"
    httpd = _start_file_api_server("127.0.0.1:0", files_dir=str(files_dir), token="")
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
    orch = _MockOrchestrator()
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

        cfg = pb.DeploymentArtifactConfig(
            supported_repo_refs=[variant_ref],
            required_variant_refs=[variant_ref],
            resolved_cozy_models_by_variant_ref={
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
        sess.send(pb.WorkerSchedulerMessage(deployment_artifact_config=cfg))

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
