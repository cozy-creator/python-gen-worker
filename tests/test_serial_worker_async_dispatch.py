"""#345 Improvement B: async-handler dispatch on the SerialWorker archetype.

These tests exercise the *runtime* dispatch path (not just registration):

  1. An `async def` single-output handler is scheduled onto the shared asyncio
     loop (`_batched_loop`) via `run_coroutine_threadsafe` and its result is
     sent back through `_send_request_result`.
  2. An `async def` generator handler (AsyncIterator[Delta]) streams deltas via
     `_emit_incremental_delta_typed` and terminates with a done event.
  3. The GPU semaphore (#337) is acquired on the dispatcher thread BEFORE the
     coroutine is scheduled — verified by asserting the semaphore value drops
     to 0 by the time the coroutine body runs, then is released afterward.

The Worker is built bare (no gRPC), with just the attributes the SerialWorker
dispatch path touches populated. We drive `_execute_serial_class_request`
directly on the calling thread (as the ThreadPoolExecutor would) and capture
its outputs via monkeypatched emit/send hooks.
"""

from __future__ import annotations

import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import msgspec
import pytest

from gen_worker import RequestContext, inference
from gen_worker.worker import Worker


class GenIn(msgspec.Struct):
    prompt: str = ""


class GenOut(msgspec.Struct):
    result: str = ""


class TokenDelta(msgspec.Struct):
    delta_text: str = ""
    finished: bool = False
    item_id: str = "item-0"


def _build_worker_with(cls: type) -> Tuple[Worker, List[dict], List[dict], List[dict]]:
    """Build a bare Worker, register the SerialWorker class on it, and wire up
    capture hooks for results / deltas / done events.

    Returns (worker, results, deltas, dones).
    """
    w = Worker.__new__(Worker)
    w._request_specs = {}
    w._training_specs = {}
    w._batched_specs = {}
    w._batched_instances = []
    w._serial_class_specs = {}
    w._serial_class_instances = []
    w._discovered_resources = {}
    w._function_schemas = {}
    w._batched_loop = None
    w._batched_loop_thread = None
    w._batched_inflight_lock = threading.Lock()
    w._batched_inflight = {}
    w._micro_batch_aggregators = {}
    w._active_requests = {}
    w._active_requests_lock = threading.Lock()
    w._request_handler_done_times = {}
    w._request_recv_times = {}
    w.scheduler_addr = ""
    w.worker_id = "test"
    w.max_output_bytes = 0
    w.max_input_bytes = 0
    # accelerator=none for these tests → no GPU semaphore acquire by default.
    w._gpu_semaphore = threading.Semaphore(4)

    # Register the class through the real registration path.
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)

    results: List[dict] = []
    deltas: List[dict] = []
    dones: List[dict] = []

    def _capture_result(
        request_id, success, output_payload, error_type, retryable,
        safe_message, error_message,
    ):
        results.append({
            "request_id": request_id,
            "success": success,
            "output_payload": output_payload,
            "error_type": error_type,
            "safe_message": safe_message,
            "error_message": error_message,
        })

    def _capture_delta(**kw):
        deltas.append(kw)

    def _capture_done(**kw):
        dones.append(kw)

    w._send_request_result = _capture_result  # type: ignore[assignment]
    w._emit_incremental_delta_typed = _capture_delta  # type: ignore[assignment]
    w._emit_incremental_done_typed = _capture_done  # type: ignore[assignment]
    return w, results, deltas, dones


def _ctx(request_id: str = "req-1") -> RequestContext:
    return RequestContext(request_id=request_id)


def test_async_single_output_dispatch() -> None:
    @inference()
    class AsyncEcho:
        def setup(self):
            pass

        @inference.function
        async def echo(self, ctx: RequestContext, payload: GenIn) -> GenOut:
            import asyncio
            await asyncio.sleep(0)  # force a real loop hop
            return GenOut(result=f"echo:{payload.prompt}")

    w, results, _deltas, _dones = _build_worker_with(AsyncEcho)
    sspec = w._serial_class_specs["echo"]
    assert sspec.is_async is True
    assert sspec.output_mode == "single"

    payload = msgspec.msgpack.encode(GenIn(prompt="hi"))
    w._execute_serial_class_request(_ctx(), sspec, payload)

    assert len(results) == 1
    assert results[0]["success"] is True
    out = msgspec.msgpack.decode(results[0]["output_payload"], type=GenOut)
    assert out.result == "echo:hi"


def test_async_streaming_dispatch() -> None:
    @inference()
    class AsyncStream:
        def setup(self):
            pass

        @inference.function
        async def chat(self, ctx: RequestContext, payload: GenIn) -> AsyncIterator[TokenDelta]:
            import asyncio
            for tok in payload.prompt.split():
                await asyncio.sleep(0)
                yield TokenDelta(delta_text=tok, item_id="item-0")
            yield TokenDelta(delta_text="", finished=True, item_id="item-0")

    w, results, deltas, dones = _build_worker_with(AsyncStream)
    sspec = w._serial_class_specs["chat"]
    assert sspec.is_async is True
    assert sspec.output_mode == "incremental"

    payload = msgspec.msgpack.encode(GenIn(prompt="alpha beta"))
    w._execute_serial_class_request(_ctx("req-stream"), sspec, payload)

    # 2 word deltas + 1 final finished delta = 3 emitted deltas.
    assert len(deltas) == 3
    assert [d["delta_text"] for d in deltas[:2]] == ["alpha", "beta"]
    assert len(dones) == 1
    assert len(results) == 1
    assert results[0]["success"] is True


def test_async_gpu_semaphore_acquired_before_coroutine() -> None:
    """The GPU semaphore must be held (value == 0 with capacity 1) by the time
    the coroutine body runs, proving it was acquired on the dispatcher thread
    BEFORE scheduling — never inside the coroutine (#345 Improvement B + #337).
    """
    from gen_worker import Resources

    observed: Dict[str, Any] = {}

    @inference(resources=Resources(accelerator="cuda", min_vram_gb=1))
    class AsyncGpu:
        def setup(self):
            pass

        @inference.function
        async def run(self, ctx: RequestContext, payload: GenIn) -> GenOut:
            import asyncio
            await asyncio.sleep(0)
            return GenOut(result="ok")

    w, results, _deltas, _dones = _build_worker_with(AsyncGpu)
    # Capacity-1 semaphore so we can detect the acquire by reading its value.
    w._gpu_semaphore = threading.Semaphore(1)
    sspec = w._serial_class_specs["run"]
    assert w._function_needs_gpu(sspec) is True

    # Patch _ensure_batched_loop so we can sample the semaphore value at the
    # moment the coroutine is about to be scheduled — i.e. on the dispatcher
    # thread, after the acquire.
    orig_ensure = w._ensure_batched_loop

    def _spy_ensure():
        observed["sema_at_schedule"] = w._gpu_semaphore._value  # type: ignore[attr-defined]
        return orig_ensure()

    w._ensure_batched_loop = _spy_ensure  # type: ignore[assignment]
    # Avoid touching real CUDA env.
    w._bind_gpu_for_request = lambda ctx: 0  # type: ignore[assignment]

    payload = msgspec.msgpack.encode(GenIn(prompt="x"))
    w._execute_serial_class_request(_ctx("req-gpu"), sspec, payload)

    # Acquired before scheduling → value was 0 when the loop was fetched.
    assert observed["sema_at_schedule"] == 0
    # Released afterwards → back to full capacity.
    assert w._gpu_semaphore._value == 1  # type: ignore[attr-defined]
    assert len(results) == 1 and results[0]["success"] is True
