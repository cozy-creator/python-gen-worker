"""Worker archetype dispatch — collapsed integration suite.

Drives the REAL Worker registration + dispatch paths (built bare to skip the
gRPC init the dispatch tables don't need) for every archetype in the floor:

  7a. SerialWorker SYNC dispatch: register a sync @inference class, run a real
      request through the registered spec, drain via shutdown().
  7b. SerialWorker ASYNC dispatch: an `async def` single-output handler is
      scheduled on the shared loop and an `async def` generator streams deltas.
  8.  BatchedWorker dispatch + discovery: @batched_inference (tenant engine) and
      @inference(runtime=...) both register a routable _BatchedWorkerSpec, and
      the async-generator tenant body iterates deltas -> Done in order.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, Iterator, List, Tuple

import msgspec
import pytest

from gen_worker import (
    Repo,
    RequestContext,
    Resources,
    inference,
)
from gen_worker.api.decorators import batched_inference
from gen_worker.api.streaming import Done, IncrementalTokenDelta
from gen_worker._worker_support import _BatchedWorkerSpec, _SerialWorkerSpec
from gen_worker.worker import Worker


class GenIn(msgspec.Struct):
    prompt: str = ""


class GenOut(msgspec.Struct):
    result: str = ""


class TokenDelta(msgspec.Struct):
    delta_text: str = ""
    finished: bool = False
    item_id: str = "item-0"


def _wait_until(pred: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll until pred() is True. Async dispatch completes on the shared loop
    (#447), so tests must wait for the callback-driven result send."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


# --------------------------------------------------------------------------- #
# 7a. SerialWorker sync dispatch — register, run a real request, drain         #
# --------------------------------------------------------------------------- #


def test_serial_sync_register_dispatch_and_drain(bare_worker) -> None:
    setup_calls: list = []
    shutdown_calls: list = []

    @inference(models={"pipe": Repo("test-org/test-repo").flavor("bf16")})
    class TestSerial:
        def setup(self, pipe):
            setup_calls.append(pipe)
            self.pipe = pipe

        @inference.function
        def generate(self, ctx: RequestContext, payload: GenIn) -> GenOut:
            return GenOut(result=f"echo:{payload.prompt}")

        def shutdown(self):
            shutdown_calls.append(True)

    spec = TestSerial.__gen_worker_endpoint_spec__
    assert getattr(TestSerial, "__gen_worker_archetype__") == "SerialWorker"
    assert spec.runtime is None

    w = bare_worker()
    assert w._register_endpoint_class(TestSerial, spec) == 1
    sspec = w._serial_class_specs["generate"]
    assert isinstance(sspec, _SerialWorkerSpec)
    assert sspec.payload_type is GenIn and sspec.output_type is GenOut
    assert sspec.output_mode == "single"

    # setup() runs once with resolved model kwargs, then a real invoke.
    rec = w._serial_class_instances[0]
    w._ensure_serial_class_started(rec)
    assert len(setup_calls) == 1
    out = sspec.method(RequestContext.__new__(RequestContext), GenIn(prompt="hi"))
    assert out.result == "echo:hi"

    # Drain calls shutdown() exactly once (idempotent).
    w._shutdown_serial_workers()
    w._shutdown_serial_workers()
    assert shutdown_calls == [True]


def test_serial_streaming_method_registers_incremental(bare_worker) -> None:
    @inference()
    class Streamer:
        def setup(self):
            pass

        @inference.function
        def chat(self, ctx: RequestContext, payload: GenIn) -> Iterator[TokenDelta]:
            for tok in payload.prompt.split():
                yield TokenDelta(delta_text=tok)
            yield TokenDelta(finished=True)

    w = bare_worker()
    w._register_endpoint_class(Streamer, Streamer.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["chat"]
    assert sspec.output_mode == "incremental"
    assert sspec.delta_type is TokenDelta


# --------------------------------------------------------------------------- #
# 7b. SerialWorker async dispatch — single output + async-generator streaming   #
# --------------------------------------------------------------------------- #


def _build_with_capture(bare_worker, cls: type) -> Tuple[Worker, List[dict], List[dict], List[dict]]:
    w = bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    results: List[dict] = []
    deltas: List[dict] = []
    dones: List[dict] = []
    w._send_request_result = lambda request_id, success, output_payload, error_type, retryable, safe_message, error_message: results.append(
        {"success": success, "output_payload": output_payload}
    )
    w._emit_incremental_delta_typed = lambda **kw: deltas.append(kw)
    w._emit_incremental_done_typed = lambda **kw: dones.append(kw)
    return w, results, deltas, dones


def test_async_single_and_streaming_dispatch(bare_worker) -> None:
    @inference()
    class AsyncEcho:
        def setup(self):
            pass

        @inference.function
        async def echo(self, ctx: RequestContext, payload: GenIn) -> GenOut:
            await asyncio.sleep(0)  # force a real loop hop
            return GenOut(result=f"echo:{payload.prompt}")

        @inference.function
        async def chat(self, ctx: RequestContext, payload: GenIn) -> AsyncIterator[TokenDelta]:
            for tok in payload.prompt.split():
                await asyncio.sleep(0)
                yield TokenDelta(delta_text=tok)
            yield TokenDelta(finished=True)

    w, results, deltas, dones = _build_with_capture(bare_worker, AsyncEcho)
    echo_spec = w._serial_class_specs["echo"]
    chat_spec = w._serial_class_specs["chat"]
    assert echo_spec.is_async and echo_spec.output_mode == "single"
    assert chat_spec.is_async and chat_spec.output_mode == "incremental"
    assert chat_spec.delta_type is TokenDelta

    # Single-output async handler round-trips through the shared loop.
    # #447: dispatch returns once the coroutine is scheduled; the result is
    # sent from a loop-side callback, so wait for it.
    w._execute_serial_class_request(
        RequestContext(request_id="r1"), echo_spec, msgspec.msgpack.encode(GenIn(prompt="hi"))
    )
    _wait_until(lambda: len(results) == 1)
    assert results[0]["success"] is True
    assert msgspec.msgpack.decode(results[0]["output_payload"], type=GenOut).result == "echo:hi"

    # Async-generator handler streams 2 word deltas + 1 finished delta.
    w._execute_serial_class_request(
        RequestContext(request_id="r2"), chat_spec, msgspec.msgpack.encode(GenIn(prompt="a b"))
    )
    _wait_until(lambda: len(dones) == 1 and len(results) == 2)
    assert [d["delta_text"] for d in deltas[:2]] == ["a", "b"]
    assert len(deltas) == 3 and len(dones) == 1
    assert results[1]["success"] is True


# --------------------------------------------------------------------------- #
# 8. BatchedWorker dispatch + discovery (tenant engine + runtime=)             #
# --------------------------------------------------------------------------- #


class CaptionInput(msgspec.Struct):
    prompt: str
    max_new_tokens: int = 32


class CaptionDelta(msgspec.Struct):
    delta_text: str = ""
    finished: bool = False


def test_batched_inference_tenant_engine_registers_and_iterates(bare_worker) -> None:
    @batched_inference(
        models={"llm": Repo("fancyfeast/llama-joycaption")},
        resources=Resources(accelerator="cuda", min_vram_gb=24),
    )
    class JoyCaption:
        def setup(self, llm):
            self.llm = llm
            self.tokens = ["hello", " ", "world"]

        @batched_inference.function
        async def caption(self, ctx: RequestContext, payload: CaptionInput) -> AsyncIterator[object]:
            for tok in self.tokens:
                if ctx.cancelled():
                    break
                yield IncrementalTokenDelta(text=tok)
            yield Done()

    cls = JoyCaption
    assert getattr(cls, "__gen_worker_archetype__") == "BatchedWorker"

    w = bare_worker()
    assert w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__) == 1
    spec = w._batched_specs["caption"]
    assert isinstance(spec, _BatchedWorkerSpec)
    assert spec.runtime == "tenant"
    assert spec.payload_type is CaptionInput
    assert spec.delta_type is IncrementalTokenDelta
    assert w._batched_instances[0]["tenant_owned_engine"] is True

    # In-memory iteration: 3 deltas then Done, in order.
    inst = cls()
    inst.setup(llm="stub")

    class _Ctx:
        def __init__(self) -> None:
            self._c = False

        def cancelled(self) -> bool:
            return self._c

    async def _collect():
        return [s async for s in inst.caption(_Ctx(), CaptionInput(prompt="x"))]

    signals = asyncio.run(_collect())
    assert [s.text for s in signals[:3]] == ["hello", " ", "world"]
    assert isinstance(signals[-1], Done) and len(signals) == 4


def test_inference_runtime_sglang_registers_batched_spec(bare_worker) -> None:
    @inference(models={"engine": Repo("org/joy").flavor("bf16")}, runtime="sglang")
    class JoySglang:
        async def setup(self, engine):
            self.engine = engine

        @inference.function
        async def caption_image(self, ctx: RequestContext, payload: CaptionInput) -> AsyncIterator[CaptionDelta]:
            yield CaptionDelta(delta_text="hi", finished=True)

    spec = JoySglang.__gen_worker_endpoint_spec__
    assert spec.runtime == "sglang"
    assert getattr(JoySglang, "__gen_worker_archetype__") == "BatchedWorker"

    w = bare_worker()
    assert w._register_endpoint_class(JoySglang, spec) == 1
    bspec = w._batched_specs["caption-image"]  # slugified
    assert bspec.runtime == "sglang"
    assert bspec.payload_type is CaptionInput and bspec.delta_type is CaptionDelta
    assert "caption-image" in w._function_schemas

    # A sync @inference.function under runtime= is rejected at decoration.
    with pytest.raises(ValueError, match="async"):
        @inference(models={"engine": Repo("org/x")}, runtime="sglang")
        class BadSync:
            def setup(self, engine):
                self.engine = engine

            @inference.function
            def caption_image(self, ctx, payload):
                return None
