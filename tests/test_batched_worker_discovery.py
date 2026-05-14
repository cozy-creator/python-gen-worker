"""BatchedWorker class-shape discovery + dispatch wiring tests (#273).

Smoke-tests that the `@inference(runtime='sglang'|'vllm')` class decorator
attaches the expected metadata, that the Worker's `_register_endpoint_class`
hook constructs the instance + registers a routable function name keyed
by the slugified method name, and that the dispatch path lookup picks
the BatchedWorker spec over the legacy `_request_specs` table.

These tests do NOT spin up an actual SGLang / vLLM engine — the smoke
test for #273 is shape discovery + dispatch routing, not real inference.
The engine wrappers carry their own `is_available()` classmethod gate.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

import msgspec
import pytest

from gen_worker import Repo, RequestContext, inference
from gen_worker.engines import EngineBase, SGLangEngine, VLLMEngine, make_engine
from gen_worker.worker import Worker
from gen_worker._worker_support import _BatchedWorkerSpec


class CaptionInput(msgspec.Struct):
    prompt: str
    image_url: str = ""
    max_new_tokens: int = 64


class CaptionDelta(msgspec.Struct):
    delta_text: str = ""
    finished: bool = False
    item_id: str = "item-0"


def _make_joycaption_class():
    """Build a fresh JoyCaption BatchedWorker class for each test."""
    joycaption_repo = Repo("fancyfeast/llama-joycaption-beta-one-hf-llava")

    @inference(
        models={"engine": joycaption_repo.flavor("bf16")},
        runtime="sglang",
    )
    class JoyCaption:
        async def setup(self, engine):
            self.engine = engine

        async def warmup(self):
            pass

        @inference.function
        async def caption_image(
            self, ctx: RequestContext, payload: CaptionInput
        ) -> AsyncIterator[CaptionDelta]:
            yield CaptionDelta(delta_text="hello", finished=True)

        async def shutdown(self):
            pass

    return JoyCaption


def _bare_worker() -> Worker:
    """Build a Worker instance with only the dispatch dicts populated.

    Avoids the heavyweight gRPC init the real `__init__` does; tests
    against discovery + dispatch routing don't need any of it.
    """
    w = Worker.__new__(Worker)
    w._request_specs = {}
    w._training_specs = {}
    w._batched_specs = {}
    w._batched_instances = []
    w._serial_class_specs = {}
    w._conversion_class_specs = {}
    w._discovered_resources = {}
    w._function_schemas = {}
    w._batched_loop = None
    w._batched_loop_thread = None
    w._batched_inflight_lock = threading.Lock()
    w._batched_inflight = {}
    return w


def test_inference_class_runtime_sglang_attaches_metadata() -> None:
    cls = _make_joycaption_class()
    spec = getattr(cls, "__gen_worker_endpoint_spec__")
    assert spec.kind == "inference"
    assert spec.runtime == "sglang"
    assert getattr(cls, "__gen_worker_archetype__") == "BatchedWorker"
    methods = getattr(cls, "__gen_worker_function_methods__")
    assert len(methods) == 1
    name, _method, fn_spec = methods[0]
    assert name == "caption_image"
    assert fn_spec.name == "caption_image"


def test_register_endpoint_class_creates_batched_spec() -> None:
    cls = _make_joycaption_class()
    w = _bare_worker()
    n = w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    assert n == 1
    # Slugified method name (underscores → dashes) is the wire route.
    assert "caption-image" in w._batched_specs
    spec = w._batched_specs["caption-image"]
    assert isinstance(spec, _BatchedWorkerSpec)
    assert spec.runtime == "sglang"
    assert spec.payload_type is CaptionInput
    assert spec.delta_type is CaptionDelta
    # Instance is the singleton — same object shows up in
    # _batched_instances so drain can call shutdown() exactly once.
    assert len(w._batched_instances) == 1
    assert w._batched_instances[0]["instance"] is spec.instance
    assert w._batched_instances[0]["runtime"] == "sglang"
    # Schemas wired into _function_schemas so the function-capabilities
    # advertisement picks them up.
    assert "caption-image" in w._function_schemas
    assert "caption-image" in w._discovered_resources


def test_register_endpoint_class_rejects_sync_method_under_batched_runtime() -> None:
    """`@inference(runtime='sglang')` on a class whose @inference.function
    method is sync should fail at decoration — the SDK validates
    archetype/runtime alignment on the class-level decorator.
    """
    repo = Repo("fancyfeast/llama-joycaption-beta-one-hf-llava")

    with pytest.raises(ValueError, match="async"):

        @inference(models={"engine": repo}, runtime="sglang")
        class BadSync:
            def setup(self, engine):
                self.engine = engine

            @inference.function
            def caption_image(self, ctx, payload):
                return None

            def shutdown(self):
                pass


def test_make_engine_factory_branches() -> None:
    e1 = make_engine("sglang")
    assert isinstance(e1, SGLangEngine)
    e2 = make_engine("vllm")
    assert isinstance(e2, VLLMEngine)
    with pytest.raises(ValueError, match="Unknown runtime"):
        make_engine("trt-llm")


def test_engines_is_available_does_not_import_lazily() -> None:
    """is_available() must not raise even when the engine package is
    missing — it returns False instead. Tenants probe via this before
    pinning a runtime at deploy time.
    """
    # Both will be False in CI (sglang + vllm both need GPUs).
    assert SGLangEngine.is_available() in (True, False)
    assert VLLMEngine.is_available() in (True, False)


def test_ensure_batched_loop_starts_dedicated_thread() -> None:
    w = _bare_worker()
    loop = w._ensure_batched_loop()
    try:
        assert loop.is_running()
        assert w._batched_loop_thread is not None
        assert w._batched_loop_thread.is_alive()
        # Calling again returns the same loop.
        loop2 = w._ensure_batched_loop()
        assert loop2 is loop
    finally:
        # Best-effort cleanup so the test thread doesn't leak.
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass


def test_register_endpoint_class_skips_non_inference_kind() -> None:
    """@training class endpoints should NOT be routed through BatchedWorker
    dispatch even if their methods are async. Only @inference + runtime=
    set engages the engine path.
    """
    from gen_worker import training

    @training(models={})
    class TrainerCls:
        def setup(self):
            pass

        @training.function
        def train(self, ctx, payload):
            pass

        def shutdown(self):
            pass

    # training class doesn't carry runtime; discovery should accept it
    # without registering as a BatchedWorker. (This worker doesn't have
    # the SerialWorker class-dispatch wiring yet — that's #322.)
    w = _bare_worker()
    n = w._register_endpoint_class(TrainerCls, TrainerCls.__gen_worker_endpoint_spec__)
    assert n == 0
    assert not w._batched_specs


class _FakeEngine(EngineBase):
    """Drop-in EngineBase that records calls but doesn't touch a GPU."""

    name = "fake"

    def __init__(self) -> None:
        self.start_calls: list = []
        self.abort_calls: list = []
        self.shutdown_calls: int = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def start(self, model_path: str, **kwargs) -> None:
        self.start_calls.append((model_path, kwargs))

    async def generate(self, request_id: str, payload):
        # Yield one fake delta then finish.
        yield {"delta_text": "hello", "finished": True}

    async def abort(self, request_id: str) -> None:
        self.abort_calls.append(request_id)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_engine_base_protocol_is_subclassable() -> None:
    """Smoke-confirm tenant test code (or follow-up integration tests)
    can substitute a fake engine implementing the same protocol.
    """
    e = _FakeEngine()
    asyncio.run(e.start("dummy"))
    assert e.start_calls == [("dummy", {})]
    asyncio.run(e.abort("rid-1"))
    assert e.abort_calls == ["rid-1"]
    asyncio.run(e.shutdown())
    assert e.shutdown_calls == 1
