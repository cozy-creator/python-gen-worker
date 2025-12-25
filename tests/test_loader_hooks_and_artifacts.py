import os
import tempfile
import unittest
from pathlib import Path
from typing import Annotated

import msgspec

from gen_worker.injection import ModelArtifacts, ModelRef, ModelRefSource as Src, register_runtime_loader
from gen_worker.worker import ActionContext, Worker


class Payload(msgspec.Struct):
    x: int = 0


class Output(msgspec.Struct):
    ok: bool
    root: str
    file_exists: bool


class RuntimeHandle:
    def __init__(self, artifacts: ModelArtifacts) -> None:
        self.artifacts = artifacts


class TestLoaderHooksAndArtifacts(unittest.TestCase):
    def _make_worker(self) -> Worker:
        w = Worker.__new__(Worker)
        import threading

        w._gpu_busy_lock = threading.Lock()
        w._is_gpu_busy = False
        w.max_output_bytes = 0
        w._model_manager = None
        w._runtime_loaders = {}
        w._custom_runtime_cache = {}
        w._custom_runtime_locks = {}
        w._deployment_model_id_by_key = {}
        w._deployment_allowed_model_ids = None
        w._active_tasks_lock = threading.Lock()
        w._active_tasks = {}
        w._active_function_counts = {}
        w._send_message = lambda msg: w._sent.append(msg)  # type: ignore[method-assign]
        w._sent = []
        w._stop_event = threading.Event()
        w._running = True
        w._materialize_assets = lambda run_id, obj: None  # type: ignore[method-assign]
        w._discovered_resources = {}
        return w

    def test_loader_hook_builds_and_caches_runtime_handle(self) -> None:
        calls = {"n": 0}

        def loader(ctx: ActionContext, artifacts: ModelArtifacts) -> RuntimeHandle:
            _ = ctx
            calls["n"] += 1
            return RuntimeHandle(artifacts)

        register_runtime_loader(RuntimeHandle, loader)

        def fn(
            ctx: ActionContext,
            rt: Annotated[RuntimeHandle, ModelRef(Src.DEPLOYMENT, "model-a")],
            payload: Payload,
        ) -> Output:
            _ = payload
            f = rt.artifacts.get_path("weights")
            return Output(
                ok=True,
                root=str(rt.artifacts.root_dir),
                file_exists=bool(f and f.exists()),
            )

        w = self._make_worker()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            weights = root / "weights.bin"
            weights.write_bytes(b"x")

            os.environ["WORKER_MODEL_ARTIFACTS_JSON"] = msgspec.json.encode(
                {"model-a": {"root_dir": str(root), "files": {"weights": str(weights)}}}
            ).decode("utf-8")
            try:
                spec = w._inspect_task_spec(fn)  # type: ignore[arg-type]
                ctx1 = ActionContext("run-a", emitter=lambda _e: None)
                b = msgspec.msgpack.encode({"x": 1})
                w._execute_task(ctx1, spec, b)

                ctx2 = ActionContext("run-b", emitter=lambda _e: None)
                w._execute_task(ctx2, spec, b)
            finally:
                os.environ.pop("WORKER_MODEL_ARTIFACTS_JSON", None)

        self.assertEqual(calls["n"], 1)

    def test_tensorrt_metadata_mismatch_returns_resource_error(self) -> None:
        def loader(ctx: ActionContext, artifacts: ModelArtifacts) -> RuntimeHandle:
            _ = ctx
            return RuntimeHandle(artifacts)

        register_runtime_loader(RuntimeHandle, loader)

        def fn(
            ctx: ActionContext,
            rt: Annotated[RuntimeHandle, ModelRef(Src.DEPLOYMENT, "engine-a@tensorrt")],
            payload: Payload,
        ) -> Output:
            _ = (ctx, payload)
            return Output(ok=True, root=str(rt.artifacts.root_dir), file_exists=False)

        w = self._make_worker()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            os.environ["WORKER_CUDA_VERSION"] = "12.6"
            os.environ["WORKER_MODEL_ARTIFACTS_JSON"] = msgspec.json.encode(
                {
                    "engine-a@tensorrt": {
                        "root_dir": str(root),
                        "files": {},
                        "metadata": {"cuda_version": "12.5", "tensorrt_version": "10.0", "sm": "89"},
                    }
                }
            ).decode("utf-8")
            try:
                spec = w._inspect_task_spec(fn)  # type: ignore[arg-type]
                ctx = ActionContext("run-trt", emitter=lambda _e: None)
                b = msgspec.msgpack.encode({"x": 1})
                w._execute_task(ctx, spec, b)
            finally:
                os.environ.pop("WORKER_CUDA_VERSION", None)
                os.environ.pop("WORKER_MODEL_ARTIFACTS_JSON", None)

        run_results = [m.run_result for m in w._sent if getattr(m, "run_result", None) is not None]
        self.assertEqual(len(run_results), 1)
        rr = run_results[0]
        self.assertFalse(rr.success)
        self.assertEqual(rr.error_type, "resource")
