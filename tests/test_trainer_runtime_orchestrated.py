from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any

import pytest

from gen_worker.trainer.orchestrated import StartupContractError
from gen_worker.trainer.runtime import run_training_runtime_from_env


def _register_trainer_module(monkeypatch: pytest.MonkeyPatch, module_name: str, class_name: str) -> str:
    mod = types.ModuleType(module_name)

    class _Trainer:
        def setup(self, ctx) -> None:
            _ = ctx

        def configure(self, ctx) -> dict[str, object]:
            _ = ctx
            return {"loaded": False}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, prepared_batch: object, state: dict[str, object], ctx):
            _ = prepared_batch
            _ = ctx
            loaded = 1.0 if bool(state.get("loaded")) else 0.0
            from gen_worker.trainer import StepResult

            return StepResult(metrics={"train/loss": loaded})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return dict(state)

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx) -> None:
            _ = ctx
            state.update(payload)

    _Trainer.__name__ = class_name
    setattr(mod, class_name, _Trainer)
    monkeypatch.setitem(sys.modules, module_name, mod)
    return f"{module_name}:{class_name}"


def _parquet_bytes() -> bytes:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"image_ref": ["a", "b"], "caption": ["x", "y"]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


class _TestHTTPHandler(BaseHTTPRequestHandler):
    auth_token = ""
    dataset_bytes = b""
    resume_payload = b"{}"
    fail_paths: set[str] = set()
    posts: list[dict[str, Any]] = []
    puts: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _authorized(self) -> bool:
        token = str(self.auth_token or "").strip()
        if not token:
            return True
        got = str(self.headers.get("Authorization") or "").strip()
        return got == f"Bearer {token}"

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        if self.path == "/inputs/train.parquet":
            body = self.dataset_bytes
        elif self.path == "/inputs/resume.json":
            body = self.resume_payload
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        payload = json.loads(raw.decode("utf-8") or "{}")
        self.posts.append({"path": self.path, "payload": payload})
        if self.path in self.fail_paths:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"forced failure"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_PUT(self) -> None:  # noqa: N802
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        self.puts.append(
            {
                "path": self.path,
                "size": len(body),
                "content_type": str(self.headers.get("Content-Type") or ""),
            }
        )
        if self.path in self.fail_paths:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"forced failure"}')
            return
        payload = {
            "ref": self.path.removeprefix("/api/v1/file/"),
            "sha256": "testsha256",
            "size_bytes": len(body),
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _start_test_server(token: str, dataset_bytes: bytes, resume_payload: bytes, fail_paths: set[str] | None = None) -> tuple[ThreadingHTTPServer, str]:
    _TestHTTPHandler.auth_token = token
    _TestHTTPHandler.dataset_bytes = dataset_bytes
    _TestHTTPHandler.resume_payload = resume_payload
    _TestHTTPHandler.fail_paths = set(fail_paths or set())
    _TestHTTPHandler.posts = []
    _TestHTTPHandler.puts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TestHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def test_trainer_runtime_startup_requires_capability_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer_import = _register_trainer_module(monkeypatch, "tmp_orch_startup_mod", "StartupTrainer")
    spec = {"run_id": "run-orch-startup", "trainer": trainer_import, "max_steps": 1, "mock_batches": [1]}
    spec_path = tmp_path / "trainer_job.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("TRAINER_ORCHESTRATED", "1")
    monkeypatch.delenv("TRAINER_CAPABILITY_TOKEN", raising=False)

    with pytest.raises(StartupContractError, match="startup.missing_capability_token"):
        run_training_runtime_from_env()


def test_trainer_runtime_orchestrated_happy_path_with_materialize_resume_and_uploads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer_import = _register_trainer_module(monkeypatch, "tmp_orch_happy_mod", "HappyTrainer")
    token = "cap-123"
    resume_payload = json.dumps({"state": {"loaded": True}}).encode("utf-8")
    server, base = _start_test_server(token=token, dataset_bytes=_parquet_bytes(), resume_payload=resume_payload)
    try:
        events = tmp_path / "events.jsonl"
        ckpt = tmp_path / "ckpt"
        samples = tmp_path / "samples"
        spec = {
            "run_id": "run-orch-happy",
            "trainer": trainer_import,
            "trainer_api_version": "v1",
            "max_steps": 2,
            "metric_every": 1,
            "checkpoint_every": 1,
            "sample_every": 1,
            "dataset": {"batch_size": 1, "readahead": 1, "columns": ["image_ref", "caption"]},
            "inputs": {
                "dataset_parquet_refs": [f"{base}/inputs/train.parquet"],
                "resume_checkpoint_ref": f"{base}/inputs/resume.json",
            },
        }
        spec_path = tmp_path / "trainer_job_happy.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        monkeypatch.setenv("TRAINER_ORCHESTRATED", "1")
        monkeypatch.setenv("TRAINER_CAPABILITY_TOKEN", token)
        monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
        monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
        monkeypatch.setenv("TRAINER_CHECKPOINTS_DIR", str(ckpt))
        monkeypatch.setenv("TRAINER_SAMPLES_DIR", str(samples))
        monkeypatch.setenv("TRAINER_UPLOAD_METRICS_URL", f"{base}/upload/metrics")
        monkeypatch.setenv("TRAINER_UPLOAD_CHECKPOINT_URL", f"{base}/upload/checkpoint")
        monkeypatch.setenv("TRAINER_UPLOAD_SAMPLE_URL", f"{base}/upload/sample")
        monkeypatch.setenv("TRAINER_UPLOAD_TERMINAL_URL", f"{base}/upload/terminal")

        assert run_training_runtime_from_env() == 0
        lines = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert any(x.get("event") == "started" for x in lines)
        assert any(x.get("event") == "metric" and x.get("name") == "train/loss" and float(x.get("value", -1)) == 1.0 for x in lines)
        assert any(x.get("event") == "completed" for x in lines)
        assert all(x.get("schema_version") == "trainer_event.v1" for x in lines)

        posted_paths = [x["path"] for x in _TestHTTPHandler.posts]
        assert "/upload/metrics" in posted_paths
        assert "/upload/checkpoint" in posted_paths
        assert "/upload/sample" in posted_paths
        assert "/upload/terminal" in posted_paths
        terminal = [x["payload"] for x in _TestHTTPHandler.posts if x["path"] == "/upload/terminal"][-1]
        assert terminal["status"] == "completed"
    finally:
        server.shutdown()
        server.server_close()


def test_trainer_runtime_orchestrated_uploads_checkpoint_bytes_to_tensorhub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer_import = _register_trainer_module(monkeypatch, "tmp_orch_upload_bytes_mod", "UploadBytesTrainer")
    token = "cap-bytes-1"
    server, base = _start_test_server(token=token, dataset_bytes=_parquet_bytes(), resume_payload=b"{}")
    try:
        events = tmp_path / "events.jsonl"
        ckpt = tmp_path / "ckpt"
        samples = tmp_path / "samples"
        spec = {
            "run_id": "run-orch-upload-bytes",
            "owner": "00000000-0000-0000-0000-000000000001",
            "trainer": trainer_import,
            "trainer_api_version": "v1",
            "max_steps": 1,
            "metric_every": 1,
            "checkpoint_every": 1,
            "sample_every": 0,
            "mock_batches": [1],
        }
        spec_path = tmp_path / "trainer_job_upload_bytes.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        monkeypatch.setenv("TRAINER_ORCHESTRATED", "1")
        monkeypatch.setenv("TRAINER_CAPABILITY_TOKEN", token)
        monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
        monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
        monkeypatch.setenv("TRAINER_CHECKPOINTS_DIR", str(ckpt))
        monkeypatch.setenv("TRAINER_SAMPLES_DIR", str(samples))
        monkeypatch.setenv("TRAINER_UPLOAD_METRICS_URL", f"{base}/upload/metrics")
        monkeypatch.setenv("TRAINER_UPLOAD_CHECKPOINT_URL", f"{base}/upload/checkpoint")
        monkeypatch.setenv("TRAINER_UPLOAD_TERMINAL_URL", f"{base}/upload/terminal")
        monkeypatch.setenv("TENSORHUB_URL", base)

        assert run_training_runtime_from_env() == 0

        assert _TestHTTPHandler.puts, "expected PUT uploads to tensorhub file API"
        assert any("/api/v1/file/v1/00000000-0000-0000-0000-000000000001/runs/run-orch-upload-bytes/checkpoints/" in x["path"] for x in _TestHTTPHandler.puts)

        terminal = [x["payload"] for x in _TestHTTPHandler.posts if x["path"] == "/upload/terminal"][-1]
        assert terminal["status"] == "completed"
        assert terminal["final_checkpoint_ref"] != ""
        assert terminal["final_checkpoint_sha256"] == "testsha256"
    finally:
        server.shutdown()
        server.server_close()


def test_trainer_runtime_cancel_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer_import = _register_trainer_module(monkeypatch, "tmp_orch_cancel_mod", "CancelTrainer")
    spec = {"run_id": "run-orch-cancel", "trainer": trainer_import, "max_steps": 3, "metric_every": 1, "mock_batches": [1, 2, 3]}
    spec_path = tmp_path / "trainer_job_cancel.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    events = tmp_path / "events.jsonl"
    cancel_file = tmp_path / "cancel.flag"
    cancel_file.write_text("1", encoding="utf-8")

    monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
    monkeypatch.setenv("TRAINER_CANCEL_FILE", str(cancel_file))

    with pytest.raises(Exception, match="canceled"):
        run_training_runtime_from_env()

    lines = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines() if x.strip()]
    failed = [x for x in lines if x.get("event") == "failed"]
    assert failed
    assert "canceled" in str(failed[-1].get("error", "")).lower()


def test_trainer_runtime_upload_failure_reports_upload_category(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer_import = _register_trainer_module(monkeypatch, "tmp_orch_upload_fail_mod", "UploadFailTrainer")
    token = "cap-up-1"
    server, base = _start_test_server(
        token=token,
        dataset_bytes=_parquet_bytes(),
        resume_payload=json.dumps({"state": {"loaded": True}}).encode("utf-8"),
        fail_paths={"/upload/sample"},
    )
    try:
        spec = {
            "run_id": "run-orch-upload-fail",
            "trainer": trainer_import,
            "max_steps": 1,
            "metric_every": 1,
            "checkpoint_every": 1,
            "sample_every": 1,
            "mock_batches": [1],
        }
        spec_path = tmp_path / "trainer_job_upload_fail.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        events = tmp_path / "events.jsonl"

        monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
        monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
        monkeypatch.setenv("TRAINER_CAPABILITY_TOKEN", token)
        monkeypatch.setenv("TRAINER_UPLOAD_METRICS_URL", f"{base}/upload/metrics")
        monkeypatch.setenv("TRAINER_UPLOAD_CHECKPOINT_URL", f"{base}/upload/checkpoint")
        monkeypatch.setenv("TRAINER_UPLOAD_SAMPLE_URL", f"{base}/upload/sample")

        with pytest.raises(Exception, match="upload"):
            run_training_runtime_from_env()

        lines = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines() if x.strip()]
        failed = [x for x in lines if x.get("event") == "failed"]
        assert failed
        assert str(failed[-1].get("error", "")).startswith("upload:")
    finally:
        server.shutdown()
        server.server_close()


def test_trainer_runtime_timeout_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = types.ModuleType("tmp_orch_timeout_mod")

    class SlowTrainer:
        def setup(self, ctx) -> None:
            _ = ctx

        def configure(self, ctx) -> dict[str, object]:
            _ = ctx
            return {}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, prepared_batch: object, state: dict[str, object], ctx):
            _ = prepared_batch
            _ = state
            _ = ctx
            time.sleep(1.1)
            from gen_worker.trainer import StepResult

            return StepResult(metrics={"train/loss": 0.1})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return dict(state)

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx) -> None:
            _ = ctx
            state.update(payload)

    mod.SlowTrainer = SlowTrainer
    monkeypatch.setitem(sys.modules, "tmp_orch_timeout_mod", mod)

    spec = {
        "run_id": "run-orch-timeout",
        "trainer": "tmp_orch_timeout_mod:SlowTrainer",
        "max_steps": 2,
        "metric_every": 1,
        "mock_batches": [1, 2, 3],
    }
    spec_path = tmp_path / "trainer_job_timeout.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
    monkeypatch.setenv("TRAINER_MAX_RUNTIME_SECONDS", "1")

    with pytest.raises(Exception, match="timeout"):
        run_training_runtime_from_env()

    lines = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines() if x.strip()]
    failed = [x for x in lines if x.get("event") == "failed"]
    assert failed
    assert "timeout" in str(failed[-1].get("error", "")).lower()


def test_trainer_runtime_resume_idempotent_skips_when_final_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer_import = _register_trainer_module(monkeypatch, "tmp_orch_resume_skip_mod", "ResumeSkipTrainer")
    token = "cap-up-2"
    server, base = _start_test_server(
        token=token,
        dataset_bytes=_parquet_bytes(),
        resume_payload=json.dumps({"state": {"loaded": True}}).encode("utf-8"),
    )
    try:
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "final.json").write_text(json.dumps({"state": {"loaded": True}}), encoding="utf-8")
        events = tmp_path / "events.jsonl"
        spec = {
            "run_id": "run-orch-resume-skip",
            "trainer": trainer_import,
            "max_steps": 5,
            "resume_from_latest": True,
            "mock_batches": [1, 2, 3],
        }
        spec_path = tmp_path / "trainer_job_resume_skip.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
        monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
        monkeypatch.setenv("TRAINER_CHECKPOINTS_DIR", str(ckpt))
        monkeypatch.setenv("TRAINER_CAPABILITY_TOKEN", token)
        monkeypatch.setenv("TRAINER_UPLOAD_TERMINAL_URL", f"{base}/upload/terminal")

        assert run_training_runtime_from_env() == 0
        lines = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert any(x.get("event") == "completed" for x in lines)
        posted_paths = [x["path"] for x in _TestHTTPHandler.posts]
        assert "/upload/terminal" not in posted_paths
    finally:
        server.shutdown()
        server.server_close()
