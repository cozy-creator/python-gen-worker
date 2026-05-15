"""@batched_inference decorator + worker dispatch tests (#273).

Covers the SDK SHAPE slice:
  * @batched_inference decorates a class and attaches the right markers.
  * @batched_inference.function requires an async-generator method shape.
  * Sync methods + plain (non-generator) async methods are rejected at
    decoration time with a clear error.
  * Method signature contract: (self, ctx, payload).
  * Worker registers the class through the parallel
    `__gen_worker_batched_inference_function_methods__` codepath
    (creates a _BatchedWorkerSpec keyed by slugified method name).
  * In-memory async-iteration: instantiating the class and running its
    function yields IncrementalTokenDelta -> Done in order.
  * Cancellation: setting ctx.cancel() mid-stream lets the tenant's
    loop break out cleanly on the next ctx.cancelled()/is_canceled() check.
  * setup(**models) injection works the same as @inference.

Engine integration (vLLM / SGLang / transformers) is OUT OF SCOPE —
follow-up session. See progress.json #273.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

import msgspec
import pytest

from gen_worker import (
    Done,
    Error,
    IncrementalTokenDelta,
    Repo,
    RequestContext,
    Resources,
    batched_inference,
    inference,
)
from gen_worker._worker_support import _BatchedWorkerSpec
from gen_worker.worker import Worker


# ----------------------------------------------------------------------------
# Test payload types
# ----------------------------------------------------------------------------


class CaptionInput(msgspec.Struct):
    prompt: str
    max_new_tokens: int = 32


# ----------------------------------------------------------------------------
# Class factory — fresh class per test so decoration runs each time.
# ----------------------------------------------------------------------------


def _make_joycaption_class():
    """JoyCaption-style @batched_inference class, no real engine."""

    @batched_inference(
        models={"llm": Repo("fancyfeast/llama-joycaption-beta-one-hf-llava")},
        resources=Resources(accelerator="cuda", min_vram_gb=24),
    )
    class JoyCaptionGenerate:
        def setup(self, llm):
            # Tenant constructs their engine here. For the SHAPE test we
            # just record what arrived from the SDK to verify injection
            # works.
            self.llm = llm
            self.tokens = ["hello", " ", "world"]

        def warmup(self):
            self.warmed_up = True

        @batched_inference.function
        async def caption(
            self,
            ctx: RequestContext,
            payload: CaptionInput,
        ) -> AsyncIterator[object]:
            for tok in self.tokens:
                if ctx.cancelled():
                    break
                yield IncrementalTokenDelta(text=tok)
            yield Done()

        def shutdown(self):
            self.shutdown_done = True

    return JoyCaptionGenerate


def _bare_worker() -> Worker:
    """Worker instance with only the dispatch dicts populated (no gRPC)."""
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


# ============================================================================
# Class-level decoration markers
# ============================================================================


def test_batched_inference_attaches_endpoint_spec() -> None:
    cls = _make_joycaption_class()
    spec = getattr(cls, "__gen_worker_endpoint_spec__")
    assert spec.kind == "inference"
    # No runtime= on @batched_inference — tenant owns the engine.
    assert spec.runtime is None
    # accelerator='cuda' was passed through (NOT 'gpu' — alias removed
    # per #326; we use the canonical name).
    assert spec.resources.accelerator == "cuda"
    assert spec.resources.min_vram_gb == 24
    # Models dict came through.
    assert set(spec.models.keys()) == {"llm"}


def test_batched_inference_attaches_archetype_and_function_methods() -> None:
    cls = _make_joycaption_class()
    assert getattr(cls, "__gen_worker_archetype__") == "BatchedWorker"
    methods = getattr(cls, "__gen_worker_batched_inference_function_methods__")
    assert len(methods) == 1
    name, method, fn_spec = methods[0]
    assert name == "caption"
    assert fn_spec.name == "caption"
    # The original @inference function-methods marker is NOT set — the
    # codepath is fully parallel.
    assert getattr(cls, "__gen_worker_function_methods__", None) is None or (
        getattr(cls, "__gen_worker_function_methods__") == []
    )


# ============================================================================
# Function-method signature validation
# ============================================================================


def test_batched_inference_function_rejects_sync_method() -> None:
    with pytest.raises(TypeError, match="sync"):

        class _Bad:
            def setup(self):
                pass

            @batched_inference.function
            def caption(self, ctx, payload):  # type: ignore[no-untyped-def]
                return None

            def shutdown(self):
                pass


def test_batched_inference_function_rejects_plain_coroutine() -> None:
    """An ``async def`` without ``yield`` is a coroutine, not a generator.

    @batched_inference.function requires the method to be an async
    generator so the dispatcher can stream incremental tokens.
    """
    with pytest.raises(TypeError, match="coroutine"):

        class _Bad:
            def setup(self):
                pass

            @batched_inference.function
            async def caption(self, ctx, payload):
                return None  # no yield → plain coroutine

            def shutdown(self):
                pass


def test_batched_inference_function_accepts_async_generator() -> None:
    """The canonical valid shape: ``async def`` with at least one ``yield``."""

    @batched_inference()
    class Good:
        def setup(self):
            pass

        @batched_inference.function
        async def caption(self, ctx, payload: CaptionInput):
            yield IncrementalTokenDelta(text="hi")
            yield Done()

        def shutdown(self):
            pass

    methods = getattr(Good, "__gen_worker_batched_inference_function_methods__")
    assert len(methods) == 1
    assert methods[0][0] == "caption"


def test_batched_inference_requires_class_not_function() -> None:
    with pytest.raises(TypeError, match="requires a class"):

        @batched_inference()
        async def caption(ctx, payload):
            yield IncrementalTokenDelta(text="hi")


def test_batched_inference_requires_setup() -> None:
    with pytest.raises(ValueError, match="setup"):

        @batched_inference()
        class _NoSetup:
            @batched_inference.function
            async def caption(self, ctx, payload: CaptionInput):
                yield IncrementalTokenDelta(text="hi")
                yield Done()


def test_batched_inference_requires_at_least_one_function() -> None:
    with pytest.raises(ValueError, match="no @batched_inference.function"):

        @batched_inference()
        class _NoFunction:
            def setup(self):
                pass

            def shutdown(self):
                pass


def test_batched_inference_rejects_stage_methods() -> None:
    """@inference.stage is a SerialWorker concept and isn't compatible
    with the BatchedWorker shape.
    """
    with pytest.raises(ValueError, match="@inference.stage"):

        @batched_inference()
        class _WithStage:
            def setup(self):
                pass

            @inference.stage(name="encode", gpu_class="small")
            def encode(self, prompt):
                return prompt

            @batched_inference.function
            async def caption(self, ctx, payload: CaptionInput):
                yield IncrementalTokenDelta(text="hi")
                yield Done()


def test_batched_inference_models_must_match_setup_kwargs() -> None:
    """Each models={} key must correspond to a setup() kwarg (same
    contract as @inference).
    """
    with pytest.raises(ValueError, match="doesn't match any setup"):

        @batched_inference(models={"missing_kw": Repo("org/repo")})
        class _Mismatched:
            def setup(self, present_kw):  # noqa: ARG002
                pass

            @batched_inference.function
            async def caption(self, ctx, payload: CaptionInput):
                yield IncrementalTokenDelta(text="hi")
                yield Done()

            def shutdown(self):
                pass


# ============================================================================
# Worker dispatch wiring
# ============================================================================


def test_worker_registers_via_parallel_codepath() -> None:
    cls = _make_joycaption_class()
    w = _bare_worker()
    n = w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    assert n == 1
    # Slugified method name (no underscores in 'caption' → stable).
    assert "caption" in w._batched_specs
    spec = w._batched_specs["caption"]
    assert isinstance(spec, _BatchedWorkerSpec)
    # Runtime marker for tenant-owned engines.
    assert spec.runtime == "tenant"
    assert spec.payload_type is CaptionInput
    # The wire-schema delta_type points at the canonical
    # IncrementalTokenDelta — that's the shape advertised to the
    # orchestrator. Done / Error travel on their own proto messages.
    assert spec.delta_type is IncrementalTokenDelta
    # Single instance recorded for drain/shutdown.
    assert len(w._batched_instances) == 1
    assert w._batched_instances[0]["instance"] is spec.instance
    assert w._batched_instances[0]["runtime"] == "tenant"
    assert w._batched_instances[0]["tenant_owned_engine"] is True


# ============================================================================
# In-memory iteration: instantiate the class, run the function, assert order
# ============================================================================


class _StubCtx:
    """Minimal stand-in for RequestContext for the in-memory iteration tests.

    Carries the cancellation flag the tenant's loop checks via
    ``ctx.cancelled()``. We don't need the rest of RequestContext for
    SHAPE-level tests.
    """

    def __init__(self) -> None:
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    def cancelled(self) -> bool:
        return self._canceled

    def is_canceled(self) -> bool:
        return self._canceled


def test_in_memory_iteration_yields_deltas_then_done() -> None:
    cls = _make_joycaption_class()
    instance = cls()
    instance.setup(llm="stub-model-path")

    ctx = _StubCtx()
    payload = CaptionInput(prompt="describe", max_new_tokens=8)

    async def _collect():
        signals = []
        async for sig in instance.caption(ctx, payload):
            signals.append(sig)
        return signals

    signals = asyncio.run(_collect())
    # 3 deltas + 1 Done.
    assert len(signals) == 4
    assert all(isinstance(s, IncrementalTokenDelta) for s in signals[:3])
    assert [s.text for s in signals[:3]] == ["hello", " ", "world"]
    assert isinstance(signals[3], Done)


def test_in_memory_iteration_cancellation_breaks_loop() -> None:
    """When ctx.cancelled() flips True mid-stream, the tenant's loop exits.

    Verifies the contract the SDK promises to BatchedWorker tenants: on
    client disconnect / interrupt the dispatcher flips ctx.cancel(), the
    tenant checks ctx.cancelled() on the next iteration, and the async
    generator stops emitting deltas.
    """
    cls = _make_joycaption_class()
    instance = cls()
    instance.setup(llm="stub-model-path")

    ctx = _StubCtx()
    payload = CaptionInput(prompt="describe", max_new_tokens=8)

    async def _collect_with_cancel():
        signals = []
        agen = instance.caption(ctx, payload)
        async for sig in agen:
            signals.append(sig)
            # Cancel after the first delta. The tenant's for-loop checks
            # ctx.cancelled() at the top of each iteration, so the second
            # delta should NOT be emitted; the loop falls through to the
            # trailing yield Done().
            if len(signals) == 1:
                ctx.cancel()
        return signals

    signals = asyncio.run(_collect_with_cancel())
    # First IncrementalTokenDelta + final Done (the tenant's for-loop
    # broke before the second delta).
    assert isinstance(signals[0], IncrementalTokenDelta)
    assert signals[0].text == "hello"
    assert isinstance(signals[-1], Done)
    assert len(signals) == 2


# ============================================================================
# Signal types — sanity for the public import surface
# ============================================================================


def test_signal_types_import_from_gen_worker_root() -> None:
    """The tenant pattern uses ``from gen_worker import IncrementalTokenDelta,
    Done, Error`` — verify the root re-export is wired up.
    """
    import gen_worker

    assert gen_worker.IncrementalTokenDelta is IncrementalTokenDelta
    assert gen_worker.Done is Done
    assert gen_worker.Error is Error


def test_signal_types_construct_with_kw_only_fields() -> None:
    d = IncrementalTokenDelta(text="hi")
    assert d.text == "hi"
    assert d.item_id is None
    Done()  # zero-field signal must be constructable
    e = Error(message="boom")
    assert e.message == "boom"
