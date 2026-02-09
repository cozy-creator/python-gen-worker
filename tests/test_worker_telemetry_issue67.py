from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest

from gen_worker.model_cache import ModelCache
from gen_worker.worker import ActionContext, Worker, pb, _TaskSpec
from gen_worker.decorators import ResourceRequirements


class _Input(msgspec.Struct):
    sleep_s: float


class _Output(msgspec.Struct):
    ok: bool


def _sleep_fn(ctx: ActionContext, payload: _Input) -> _Output:
    time.sleep(payload.sleep_s)
    return _Output(ok=True)


def test_gpu_is_busy_refcount_overlapping_inference() -> None:
    w = Worker.__new__(Worker)
    w._has_gpu = True
    w._gpu_busy_lock = threading.Lock()
    w._gpu_busy_refcount = 0
    w._is_gpu_busy = False
    w._active_model_use_lock = threading.Lock()
    w._active_model_use_counts = {}
    w._active_tasks_lock = threading.Lock()
    w._active_tasks = {}
    w._active_function_counts = {}
    w._send_message = lambda msg: None  # type: ignore[method-assign]
    w._send_task_result = lambda *args, **kwargs: None  # type: ignore[method-assign]
    w._materialize_assets = lambda ctx, obj: None  # type: ignore[method-assign]

    spec = _TaskSpec(
        name="sleep",
        func=_sleep_fn,
        resources=ResourceRequirements(),
        ctx_param="ctx",
        payload_param="payload",
        payload_type=_Input,
        output_mode="single",
        output_type=_Output,
        injections=(),
    )

    ctx1 = ActionContext("r1", emitter=lambda e: None)
    ctx2 = ActionContext("r2", emitter=lambda e: None)
    payload = msgspec.msgpack.encode({"sleep_s": 0.25})

    t1 = threading.Thread(target=w._execute_task, args=(ctx1, spec, payload), daemon=True)
    t2 = threading.Thread(target=w._execute_task, args=(ctx2, spec, payload), daemon=True)

    t1.start()
    t2.start()

    # Wait until at least one entered busy.
    deadline = time.time() + 2.0
    while time.time() < deadline and not w._get_gpu_busy_status():
        time.sleep(0.01)
    assert w._get_gpu_busy_status() is True

    # While at least one task is running, busy must remain true.
    while t1.is_alive() or t2.is_alive():
        assert w._get_gpu_busy_status() is True
        time.sleep(0.01)

    assert w._get_gpu_busy_status() is False


class _StubModelManager:
    def __init__(self) -> None:
        self._loaded: set[str] = set()
        self.model_sizes: dict[str, float] = {}

    async def load_model_into_vram(self, model_id: str) -> bool:
        await asyncio.sleep(0.15)
        self._loaded.add(model_id)
        return True

    def unload(self, model_id: str) -> None:
        self._loaded.discard(model_id)

    def get_vram_loaded_models(self) -> list[str]:
        return sorted(self._loaded)


def test_load_model_emits_events_and_updates_vram_models_and_busy(tmp_path: Path) -> None:
    w = Worker.__new__(Worker)
    w._has_gpu = True
    w._gpu_busy_lock = threading.Lock()
    w._gpu_busy_refcount = 0
    w._is_gpu_busy = False
    w._active_model_use_lock = threading.Lock()
    w._active_model_use_counts = {}
    w._model_init_done_event = threading.Event()
    w._model_init_done_event.set()
    w._model_manager = _StubModelManager()
    w._model_cache = ModelCache(model_cache_dir=str(tmp_path / "cache"))
    w._task_specs = {}
    w._ws_specs = {}
    w._discovered_resources = {}
    w._function_schemas = {}
    w.max_concurrency = 0
    w.runpod_pod_id = ""

    sent: list[Any] = []
    w._send_message = lambda msg: sent.append(msg)  # type: ignore[method-assign]

    cmd = pb.LoadModelCommand(model_id="cozy:demo/repo@sha256:snap-1")

    th = threading.Thread(target=w._handle_load_model_cmd, args=(cmd,), daemon=True)
    th.start()

    # Busy should become true during the async sleep.
    deadline = time.time() + 2.0
    saw_busy = False
    while time.time() < deadline and th.is_alive():
        if w._get_gpu_busy_status():
            saw_busy = True
            break
        time.sleep(0.01)
    assert saw_busy is True

    th.join(timeout=5)
    assert w._get_gpu_busy_status() is False

    # Must emit load.started + load.completed/failed.
    event_types = [
        m.worker_event.event_type
        for m in sent
        if getattr(m, "worker_event", None) and m.HasField("worker_event")
    ]
    assert "model.load.started" in event_types
    assert "model.load.completed" in event_types

    # LoadModelResult must succeed.
    results = [m.load_model_result for m in sent if m.HasField("load_model_result")]
    assert results and results[-1].success is True

    # The immediate registration update should reflect vram_models after the load.
    regs = [m.worker_registration for m in sent if m.HasField("worker_registration")]
    assert regs
    assert "cozy:demo/repo@sha256:snap-1" in list(regs[-1].resources.vram_models)


def test_unload_model_rejected_when_in_use(tmp_path: Path) -> None:
    w = Worker.__new__(Worker)
    w._has_gpu = True
    w._gpu_busy_lock = threading.Lock()
    w._gpu_busy_refcount = 0
    w._is_gpu_busy = False
    w._active_model_use_lock = threading.Lock()
    w._active_model_use_counts = {}
    w._model_manager = _StubModelManager()
    w._model_cache = ModelCache(model_cache_dir=str(tmp_path / "cache"))
    w._task_specs = {}
    w._ws_specs = {}
    w._discovered_resources = {}
    w._function_schemas = {}
    w.max_concurrency = 0
    w.runpod_pod_id = ""

    sent: list[Any] = []
    w._send_message = lambda msg: sent.append(msg)  # type: ignore[method-assign]

    model_id = "cozy:demo/repo@sha256:snap-1"
    w._model_use_enter(model_id)
    try:
        w._handle_unload_model_cmd(pb.UnloadModelCommand(model_id=model_id))
    finally:
        w._model_use_exit(model_id)

    res = [m.unload_model_result for m in sent if m.HasField("unload_model_result")]
    assert res and res[-1].success is False
    assert "model_in_use" in (res[-1].error_message or "")

