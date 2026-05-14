"""Smoke tests for #324 cross-request micro-batching + prefer_distilled."""

from __future__ import annotations

import asyncio
import threading
import time
import types

import msgspec
import pytest

from gen_worker.api.binding import Repo
from gen_worker.api.decorators import inference
from gen_worker.api.micro_batch import (
    MicroBatchAggregator,
    should_disable_batching,
)
from gen_worker.discovery.discover import _extract_class_function_methods
from gen_worker.request_context import RequestContext


# -------------------- decorator + spec surface tests --------------------


class _Inp(msgspec.Struct):
    prompt: str


class _Out(msgspec.Struct):
    image: str


def _make_serial_class(**kwargs):
    @inference(models={"pipe": Repo("test/sana-sprint")}, **kwargs)
    class Gen:
        def setup(self, pipe=None):
            self.pipe = pipe or object()

        @inference.function
        def generate(self, ctx: RequestContext, payload: _Inp) -> _Out:
            return _Out(image="x")

        def shutdown(self):
            pass

    return Gen


def test_batch_kwargs_attach_to_endpoint_spec() -> None:
    cls = _make_serial_class(batch_window_ms=50, max_batch=4, prefer_distilled=True)
    spec = cls.__gen_worker_endpoint_spec__
    assert spec.batch_window_ms == 50
    assert spec.max_batch == 4
    assert spec.prefer_distilled is True
    assert spec.runtime is None  # SerialWorker


def test_batch_kwargs_default_none() -> None:
    cls = _make_serial_class()
    spec = cls.__gen_worker_endpoint_spec__
    assert spec.batch_window_ms is None
    assert spec.max_batch is None
    assert spec.prefer_distilled is False


def test_only_one_batch_kwarg_rejected() -> None:
    with pytest.raises(ValueError, match="declared together"):
        _make_serial_class(batch_window_ms=50)
    with pytest.raises(ValueError, match="declared together"):
        _make_serial_class(max_batch=4)


def test_max_batch_below_two_rejected() -> None:
    with pytest.raises(ValueError, match="max_batch must be >= 2"):
        _make_serial_class(batch_window_ms=50, max_batch=1)


def test_batch_kwargs_rejected_on_async_class() -> None:
    with pytest.raises(ValueError, match="SerialWorker-only"):

        @inference(
            models={"engine": Repo("test/batched")},
            runtime="vllm",
            batch_window_ms=50,
            max_batch=4,
        )
        class BadAsync:
            async def setup(self, engine=None):
                self.engine = engine

            @inference.function
            async def generate(self, ctx: RequestContext, payload: _Inp):
                yield _Out(image="x")

            async def shutdown(self):
                pass


# -------------------- discovery manifest surface tests --------------------


def test_discovery_emits_batch_kwargs_and_prefer_distilled() -> None:
    cls = _make_serial_class(batch_window_ms=50, max_batch=4, prefer_distilled=True)
    cls.__module__ = "test_mod"
    entries = _extract_class_function_methods(cls, "test_mod")
    assert len(entries) == 1
    e = entries[0]
    assert e["batch_window_ms"] == 50
    assert e["max_batch"] == 4
    assert e["prefer_distilled"] is True
    assert e["archetype"] == "SerialWorker"


def test_discovery_emits_defaults_when_not_declared() -> None:
    cls = _make_serial_class()
    cls.__module__ = "test_mod_b"
    entries = _extract_class_function_methods(cls, "test_mod_b")
    e = entries[0]
    assert e["batch_window_ms"] is None
    assert e["max_batch"] is None
    assert e["prefer_distilled"] is False


# -------------------- should_disable_batching --------------------


class _BatchBreakerCache:
    breaks_cross_request_batching = True


class _NeutralCache:
    breaks_cross_request_batching = False


def test_should_disable_batching_via_pipe_marker() -> None:
    class Pipe:
        pass

    class Instance:
        def __init__(self):
            self.pipe = Pipe()
            self.pipe._gen_worker_teacache_helper = _BatchBreakerCache()

    reason = should_disable_batching(Instance())
    assert reason is not None
    assert "_gen_worker_teacache_helper" in reason


def test_should_disable_batching_via_declared_wrapper() -> None:
    class Instance:
        pass

    reason = should_disable_batching(Instance(), declared_wrappers=[_BatchBreakerCache()])
    assert reason is not None


def test_should_disable_batching_clean_returns_none() -> None:
    class Pipe:
        pass

    class Instance:
        def __init__(self):
            self.pipe = Pipe()
            self.pipe._gen_worker_deepcache_helper = _NeutralCache()

    assert should_disable_batching(Instance()) is None


# -------------------- end-to-end aggregator behavior --------------------


@pytest.fixture()
def aggregator_loop():
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # tiny wait so loop is running before tests submit work.
    for _ in range(50):
        if loop.is_running():
            break
        time.sleep(0.01)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)


def test_aggregator_drains_window_with_concurrent_submits(aggregator_loop) -> None:
    call_log: list[list[int]] = []

    def call_fn(payloads: list[int]):
        call_log.append(list(payloads))
        return [p * 10 for p in payloads]

    agg = MicroBatchAggregator(
        function_name="t",
        batch_window_ms=80,
        max_batch=4,
        call_fn=call_fn,
    )
    agg.start(aggregator_loop)

    results: list[tuple[int, int]] = []

    def submit(i: int) -> None:
        fut = agg.submit(f"rid-{i}", i)
        results.append((i, fut.result(timeout=2.0)))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # 5 submissions, max_batch=4 → expect 2 batches summing to 5.
    assert sum(len(b) for b in call_log) == 5
    # Every result is payload * 10.
    for i, v in results:
        assert v == i * 10
    metrics = agg.metrics()
    assert metrics["items_processed"] == 5
    agg.shutdown()


def test_aggregator_broadcasts_exception_to_batch(aggregator_loop) -> None:
    def call_fn(payloads):
        raise RuntimeError("synthetic failure")

    agg = MicroBatchAggregator(
        function_name="err",
        batch_window_ms=40,
        max_batch=4,
        call_fn=call_fn,
    )
    agg.start(aggregator_loop)

    errs: list[BaseException] = []

    def submit() -> None:
        fut = agg.submit("rid", 1)
        try:
            fut.result(timeout=2.0)
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=submit) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(errs) == 3
    assert all(isinstance(e, RuntimeError) for e in errs)
    agg.shutdown()
