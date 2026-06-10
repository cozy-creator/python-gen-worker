"""#447: async SerialWorker dispatch must not be bounded by the job-executor
width (Python default min(32, cpu+4)).

`async def` handlers are scheduled onto the shared asyncio loop and FREE their
dispatcher thread for the duration of the await; completion (result encode +
send) is callback-driven on the loop. These tests drive the real
``_execute_serial_class_request`` path on a bare Worker (same pattern as
test_worker_dispatch) with stubbed result capture.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable, Dict, List

import msgspec

from gen_worker import RequestContext, Resources, inference
from gen_worker.worker import Worker

# Strictly above the default ThreadPoolExecutor width of min(32, cpu+4) so the
# barrier below can ONLY release if the await holds no executor thread.
N_INFLIGHT = 64


class HoldIn(msgspec.Struct):
    tag: str = ""


class HoldOut(msgspec.Struct):
    tag: str = ""


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    w._request_specs = {}
    w._training_specs = {}
    w._batched_specs = {}
    w._batched_instances = []
    w._serial_class_specs = {}
    w._serial_class_instances = []
    w._conversion_class_specs = {}
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
    w._gpu_semaphore = threading.Semaphore(1)
    return w


def _wait(pred: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def test_async_dispatch_exceeds_executor_width() -> None:
    """Dispatch N_INFLIGHT (=64 > 32) async requests SEQUENTIALLY from one
    thread. Each handler parks on a shared barrier that only releases once all
    64 have STARTED — so the test completes iff all 64 coroutines are in
    flight simultaneously on the loop. Under the old thread-blocking design
    (dispatcher thread parks on future.result()) the very first dispatch call
    would never return and this would deadlock at started=1.
    """
    state: Dict[str, object] = {"started": 0, "release": None}

    @inference(resources=Resources(accelerator="none"))
    class Holder:
        def setup(self) -> None:
            pass

        @inference.function
        async def hold(self, ctx: RequestContext, payload: HoldIn) -> HoldOut:
            # All coroutines share the single loop thread — no locking needed.
            if state["release"] is None:
                state["release"] = asyncio.Event()
            state["started"] = int(state["started"]) + 1  # type: ignore[call-overload]
            if int(state["started"]) >= N_INFLIGHT:  # type: ignore[call-overload]
                state["release"].set()  # type: ignore[union-attr]
            await state["release"].wait()  # type: ignore[union-attr]
            return HoldOut(tag=payload.tag)

    w = _bare_worker()
    assert w._register_endpoint_class(Holder, Holder.__gen_worker_endpoint_spec__) == 1
    sspec = w._serial_class_specs["hold"]
    assert sspec.is_async and sspec.output_mode == "single"

    results: List[Dict[str, object]] = []
    all_done = threading.Event()

    def _capture(request_id, success, output_payload, error_type, retryable, safe_message, error_message):  # type: ignore[no-untyped-def]
        results.append({"request_id": request_id, "success": success, "error_type": error_type})
        if len(results) >= N_INFLIGHT:
            all_done.set()

    w._send_request_result = _capture  # type: ignore[method-assign]

    # Sequential dispatch from ONE thread: each call must return as soon as
    # the coroutine is scheduled, never blocking on the handler's await.
    t0 = time.monotonic()
    for i in range(N_INFLIGHT):
        ctx = RequestContext(request_id=f"r{i}")
        w._active_requests[ctx.request_id] = ctx
        w._execute_serial_class_request(
            ctx, sspec, msgspec.msgpack.encode(HoldIn(tag=str(i)))
        )
    dispatch_s = time.monotonic() - t0

    assert all_done.wait(timeout=30.0), (
        f"only {len(results)}/{N_INFLIGHT} results arrived "
        f"(started={state['started']}) — async dispatch is thread-bound again"
    )
    assert int(state["started"]) == N_INFLIGHT  # type: ignore[call-overload]
    assert all(r["success"] for r in results)
    assert {r["request_id"] for r in results} == {f"r{i}" for i in range(N_INFLIGHT)}
    # Dispatch itself is non-blocking; generous bound to stay CI-safe.
    assert dispatch_s < 10.0, f"sequential dispatch took {dispatch_s:.1f}s — a dispatch call blocked"
    # All requests were cleaned out of the active set on completion.
    _wait(lambda: len(w._active_requests) == 0, timeout=5.0)


def test_async_dispatch_failure_still_sends_terminal_result() -> None:
    """The callback-driven completion path must deliver a mapped failure
    result when the coroutine raises (no thread waiting to catch it)."""

    @inference(resources=Resources(accelerator="none"))
    class Exploder:
        def setup(self) -> None:
            pass

        @inference.function
        async def boom(self, ctx: RequestContext, payload: HoldIn) -> HoldOut:
            await asyncio.sleep(0)
            raise RuntimeError("kaboom")

    w = _bare_worker()
    w._register_endpoint_class(Exploder, Exploder.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["boom"]

    results: List[Dict[str, object]] = []
    w._send_request_result = lambda request_id, success, output_payload, error_type, retryable, safe_message, error_message: results.append(  # type: ignore[method-assign]
        {"success": success, "error_type": error_type, "retryable": retryable}
    )

    ctx = RequestContext(request_id="rfail")
    w._active_requests[ctx.request_id] = ctx
    w._execute_serial_class_request(ctx, sspec, msgspec.msgpack.encode(HoldIn(tag="x")))

    _wait(lambda: len(results) == 1)
    assert results[0]["success"] is False
    assert results[0]["error_type"]  # mapped, not empty
    assert "rfail" not in w._active_requests


def test_async_dispatch_cancel_before_run_maps_to_canceled() -> None:
    """cancel-on-disconnect invariant: a ctx canceled before/while dispatched
    yields a terminal `canceled` result, not a success."""

    @inference(resources=Resources(accelerator="none"))
    class Slow:
        def setup(self) -> None:
            pass

        @inference.function
        async def crawl(self, ctx: RequestContext, payload: HoldIn) -> HoldOut:
            await asyncio.sleep(60)
            return HoldOut(tag=payload.tag)

    w = _bare_worker()
    w._register_endpoint_class(Slow, Slow.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["crawl"]

    results: List[Dict[str, object]] = []
    w._send_request_result = lambda request_id, success, output_payload, error_type, retryable, safe_message, error_message: results.append(  # type: ignore[method-assign]
        {"success": success, "error_type": error_type}
    )

    ctx = RequestContext(request_id="rcancel")
    ctx.cancel()
    w._active_requests[ctx.request_id] = ctx
    w._execute_serial_class_request(ctx, sspec, msgspec.msgpack.encode(HoldIn(tag="x")))

    _wait(lambda: len(results) == 1)
    assert results[0] == {"success": False, "error_type": "canceled"}
