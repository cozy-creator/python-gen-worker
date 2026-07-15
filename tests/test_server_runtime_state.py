"""Worker-owned engine startup is represented in typed worker state."""

from __future__ import annotations

import asyncio

import msgspec
import pytest

from gen_worker import HF, RequestContext, Resources, VLLMRuntime, endpoint
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from gen_worker.runtimes.server import ServerHandle


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    text: str


@endpoint(
    model=HF("test/qwen"),
    resources=Resources(vram_gb=44),
    runtime=VLLMRuntime(max_model_len=16_384, gpu_memory_utilization=0.94),
)
class _RuntimeEndpoint:
    def setup(self, model: str, server: ServerHandle) -> None:
        self.model = model
        self.server = server

    def complete(self, ctx: RequestContext, payload: _In) -> _Out:
        return _Out(text=payload.prompt)


class _Process:
    pid = 123


def _state(executor: Executor) -> pb.StateDelta:
    lifecycle = object.__new__(Lifecycle)
    lifecycle.executor = executor
    lifecycle.hardware = {"gpu_total_mem": 0}
    lifecycle.phase = pb.WORKER_PHASE_READY
    lifecycle._observed_residency_generation = 0
    return lifecycle._state_delta()


def test_engine_boot_stays_loading_until_health_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    asyncio.run(_assert_engine_boot_state(monkeypatch, tmp_path))


async def _assert_engine_boot_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def send(_message) -> None:
        return None

    (spec,) = extract_specs(_RuntimeEndpoint)
    executor = Executor([spec], send)
    executor.store.ensure_local = _returning(tmp_path)

    started = asyncio.Event()
    healthy = asyncio.Event()
    handle = ServerHandle(
        base_url="http://127.0.0.1:1234",
        process=_Process(),  # type: ignore[arg-type]
    )

    async def boot(_spec, _paths):
        started.set()
        await healthy.wait()
        return handle

    monkeypatch.setattr(executor, "_boot_engine_server", boot)
    monkeypatch.setattr("gen_worker.lifecycle.free_vram_bytes", lambda: 0)
    monkeypatch.setattr("gen_worker.runtimes.server.process_vram_bytes", lambda _pid: 0)

    setup = asyncio.create_task(executor.ensure_setup(spec))
    await started.wait()

    during = _state(executor)
    assert list(during.loading_functions) == ["complete"]
    assert list(during.available_functions) == []

    healthy.set()
    instance = await setup

    assert instance.server is handle
    ready = _state(executor)
    assert list(ready.loading_functions) == []
    assert list(ready.available_functions) == ["complete"]


def _returning(value):
    async def return_value(*_args, **_kwargs):
        return value

    return return_value
