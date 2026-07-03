"""Cross-request micro-batching aggregator for SerialWorker endpoints (#324).

Tenants opt in by declaring BOTH ``batch_window_ms`` and ``max_batch`` on the
``@inference(...)`` class decorator:

.. code-block:: python

    @inference(
        models={'pipe': sana_sprint},
        batch_window_ms=50,   # admission window
        max_batch=4,          # cap on per-forward batch size
    )
    class SanaSprintGen:
        def setup(self, pipe):
            self.pipe = pipe

        @inference.function
        def generate(self, ctx, payload):
            # When invoked through the aggregator, `payload` arrives as a
            # LIST of the originally-decoded payload structs (one per
            # concurrent caller). The method must return a LIST in the same
            # order — the aggregator de-multiplexes results back to callers.
            #
            # When the aggregator is auto-disabled (e.g. TeaCache attached),
            # the method is called with a single payload struct as before.
            if isinstance(payload, list):
                return self.pipe(prompts=[p.prompt for p in payload]).images
            return self.pipe(prompts=[payload.prompt]).images[0]

        def shutdown(self): pass

Hard rules:
  * Both ``batch_window_ms`` and ``max_batch`` must be set; either-None means
    "no batching" and dispatch falls back to one-at-a-time.
  * Auto-disabled when any cache wrapper attached to the tenant's pipe
    objects has ``breaks_cross_request_batching=True`` (TeaCache today, per
    nunchaku #597). Convention: any cache/acceleration wrapper whose state
    must not be shared across concurrent requests stamps
    ``breaks_cross_request_batching = True`` on itself (attribute name is
    the contract; there is no base class).
  * Aggregator runs on an asyncio loop (one drain coroutine per registered
    function). The tenant ``@inference.function`` body is sync — it runs on
    a worker thread via ``loop.run_in_executor`` so it doesn't block the loop.
  * If the tenant method raises while processing a batch, the SAME exception
    is delivered to every caller in the batch — the aggregator can't pick a
    "winner" so failure is shared.

Validated by TetriServe (https://arxiv.org/html/2510.01565): admission
window ~50ms with max_batch=4 lifts throughput 1.5-3x on sub-300ms forwards
without breaking SLOs.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Cache-wrapper inspection — auto-disable when any wrapper attached to a pipe
# declares `breaks_cross_request_batching = True`.
# ============================================================================


def _wrapper_breaks_batching(obj: Any) -> bool:
    """Return True iff ``obj`` is a cache wrapper that forbids batch > 1."""
    return bool(getattr(obj, "breaks_cross_request_batching", False))


def _scan_pipe_for_batch_breakers(pipe: Any) -> List[str]:
    """Return human-readable names of cache wrappers on ``pipe`` that break
    cross-request batching.

    Looks at:
      1. Any attribute on ``pipe`` whose name starts with ``_gen_worker_``
         and which carries ``breaks_cross_request_batching = True``
         (e.g. ``pipe._gen_worker_deepcache_helper``, or future markers).
      2. Class-level introspection: if the pipe itself was patched in-place
         by a wrapper class with ``breaks_cross_request_batching=True``, the
         attribute lookup above catches it; for TeaCache (which patches via
         module-level monkeypatch and leaves no marker on the pipe), this
         function returns the literal class-name marker so callers can
         inspect at decorator-eval time via the wrapper instance directly.

    Note: the canonical way to know is at ``apply()``-time — the aggregator's
    ``register_function`` accepts an explicit list of cache wrappers and we
    prefer that path; the pipe scan here is the belt-and-suspenders fallback
    for tenants who attach caches outside of the SDK helper module.
    """
    out: List[str] = []
    for attr_name in dir(pipe):
        if not attr_name.startswith("_gen_worker_"):
            continue
        try:
            val = getattr(pipe, attr_name, None)
        except Exception:
            continue
        if val is None:
            continue
        if _wrapper_breaks_batching(val):
            out.append(f"{attr_name}={type(val).__name__}")
    return out


def should_disable_batching(
    instance: Any,
    declared_wrappers: Optional[Iterable[Any]] = None,
) -> Optional[str]:
    """Return a reason-string if batching should be disabled, else None.

    Inspects:
      * Any cache wrappers the tenant explicitly registered via
        ``declared_wrappers`` (the SDK can capture these at ``setup()``-time).
      * Every attribute on ``instance`` whose value is a pipe-like object
        and which carries a ``_gen_worker_<cache>_helper`` marker.

    Returns the first conflict found so the log message is short and points
    at the wrapper that triggered the disable.
    """
    if declared_wrappers:
        for w in declared_wrappers:
            if _wrapper_breaks_batching(w):
                return f"cache wrapper {type(w).__name__} declares breaks_cross_request_batching=True"

    # Walk attributes of the tenant instance looking for pipes with cache markers.
    for attr_name in dir(instance):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(instance, attr_name, None)
        except Exception:
            continue
        if val is None:
            continue
        breakers = _scan_pipe_for_batch_breakers(val)
        if breakers:
            return f"pipe `{attr_name}` carries batch-breaking cache wrapper(s): {', '.join(breakers)}"
    return None


# ============================================================================
# Aggregator runtime — one per (function-name) on a shared asyncio loop.
# ============================================================================


@dataclass
class _PendingItem:
    """One in-flight tenant request waiting for the next batched dispatch."""

    request_id: str
    payload: Any
    # ``future`` is settled with the per-caller result (or exception).
    future: "asyncio.Future[Any]"
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class MicroBatchAggregator:
    """Per-function aggregator: queue + window-timer + drain coroutine.

    Lifecycle:
      1. ``register`` (or constructor) records the per-function metadata.
      2. ``start(loop)`` launches the drain coroutine on the given loop.
      3. Callers invoke ``submit(payload)`` and ``await`` the returned future.
      4. The drain coroutine waits at most ``batch_window_ms`` for at least
         one item, then drains up to ``max_batch`` items and fires ONE call
         into ``call_fn(payloads_list)``. Results from the list are mapped
         back 1:1 to caller futures.
      5. ``shutdown()`` cancels the drain task and rejects any in-flight
         items with ``RuntimeError``.

    ``call_fn(payloads)`` runs the tenant code. It receives a LIST of
    payloads and MUST return either:
      - a list of per-caller results in the same order, OR
      - any non-list value, in which case the aggregator broadcasts that
        single value to every caller in the batch (escape hatch — tenants
        whose forward truly produces one output for the whole batch can
        use this, but it's almost never what you want).

    If ``call_fn`` raises, the same exception is delivered to every caller
    in the batch (the aggregator can't selectively retry one).
    """

    function_name: str
    batch_window_ms: int
    max_batch: int
    # call_fn is sync; the aggregator runs it via run_in_executor so the
    # asyncio loop stays responsive.
    call_fn: Callable[[List[Any]], Any]

    _queue: "asyncio.Queue[_PendingItem]" = field(init=False, repr=False)
    _drain_task: "Optional[asyncio.Task[None]]" = field(default=None, init=False, repr=False)
    _loop: Optional[asyncio.AbstractEventLoop] = field(default=None, init=False, repr=False)
    _shutdown: bool = field(default=False, init=False)
    _metrics_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _batches_fired: int = field(default=0, init=False)
    _items_processed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # asyncio.Queue must be created on the loop it will be used from;
        # we defer construction to ``start()``.
        self._queue = None  # type: ignore[assignment]

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the drain coroutine on the given asyncio loop.

        The loop must be running on its own thread (e.g. the shared
        BatchedWorker loop). Caller thread is unblocked once start completes.
        """
        if self._drain_task is not None:
            return
        self._loop = loop
        # Build queue + spawn drain on the target loop so internal asyncio
        # state (Queue's _loop ref, Task scheduling) is correct.
        if not loop.is_running():
            raise RuntimeError(
                f"MicroBatchAggregator({self.function_name}).start: loop is not running"
            )
        cfut = asyncio.run_coroutine_threadsafe(self._on_loop_init(), loop)
        cfut.result(timeout=5.0)
        logger.info(
            "MicroBatchAggregator started: function=%s window_ms=%d max_batch=%d",
            self.function_name, self.batch_window_ms, self.max_batch,
        )

    async def _on_loop_init(self) -> None:
        self._queue = asyncio.Queue()
        self._drain_task = asyncio.create_task(
            self._drain_forever(),
            name=f"micro-batch-drain-{self.function_name}",
        )

    def submit(self, request_id: str, payload: Any) -> Any:
        """Enqueue a request for the next batched forward and return a
        ``concurrent.futures.Future`` that resolves to the per-caller result
        (or carries the per-caller exception).

        Thread-safe — callers from the synchronous job dispatch thread can
        ``.result(timeout=...)`` on the returned future. Internally bridges
        the aggregator's asyncio.Future to a concurrent.futures.Future via
        ``asyncio.run_coroutine_threadsafe`` so dispatch threads don't need
        their own loop.
        """
        if self._shutdown:
            raise RuntimeError(
                f"MicroBatchAggregator({self.function_name}) is shut down"
            )
        loop = self._loop
        if loop is None:
            raise RuntimeError(
                f"MicroBatchAggregator({self.function_name}).submit before start()"
            )
        return asyncio.run_coroutine_threadsafe(
            self._submit_and_wait(request_id, payload), loop
        )

    async def _submit_and_wait(self, request_id: str, payload: Any) -> Any:
        """Enqueue + await the per-caller future on the aggregator loop."""
        running_loop = asyncio.get_running_loop()
        fut: "asyncio.Future[Any]" = running_loop.create_future()
        await self._queue.put(_PendingItem(
            request_id=request_id,
            payload=payload,
            future=fut,
        ))
        return await fut

    async def _drain_forever(self) -> None:
        """Window-timed drain loop. Pulls items, fires one batched call."""
        window_s = self.batch_window_ms / 1000.0
        while not self._shutdown:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return
            batch: List[_PendingItem] = [first]
            # Pull additional items up to max_batch, capped by the window.
            window_deadline = time.monotonic() + window_s
            while len(batch) < self.max_batch:
                remaining = window_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # Drain whatever we have + return cleanly.
                    return
                batch.append(item)

            await self._dispatch_batch(batch)

    async def _dispatch_batch(self, batch: List[_PendingItem]) -> None:
        """Run the tenant call on a worker thread; route results back."""
        payloads = [item.payload for item in batch]
        loop = asyncio.get_running_loop()
        with self._metrics_lock:
            self._batches_fired += 1
            self._items_processed += len(batch)
        t0 = time.monotonic()
        try:
            results = await loop.run_in_executor(
                None, self.call_fn, payloads
            )
        except BaseException as exc:  # noqa: BLE001
            # Broadcast the same exception to every caller in the batch.
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            logger.warning(
                "MicroBatch dispatch failed: function=%s batch_size=%d exc=%r",
                self.function_name, len(batch), exc,
            )
            return
        dur_ms = int((time.monotonic() - t0) * 1000)
        # Map results back. If the tenant returned a list of len == batch_size,
        # zip one-to-one. Otherwise broadcast the same value (escape hatch).
        if isinstance(results, list) and len(results) == len(batch):
            for item, res in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(res)
        else:
            for item in batch:
                if not item.future.done():
                    item.future.set_result(results)
        logger.debug(
            "MicroBatch dispatched: function=%s batch_size=%d dur_ms=%d",
            self.function_name, len(batch), dur_ms,
        )

    def metrics(self) -> Dict[str, int]:
        with self._metrics_lock:
            return {
                "batches_fired": self._batches_fired,
                "items_processed": self._items_processed,
            }

    def shutdown(self) -> None:
        """Cancel the drain task. Idempotent."""
        self._shutdown = True
        task = self._drain_task
        loop = self._loop
        if task is not None and loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except Exception:
                pass
        self._drain_task = None
        # Reject any items still on the queue. Must run on the loop thread
        # since asyncio.Queue.get_nowait isn't thread-safe.
        if self._queue is not None and loop is not None and loop.is_running():
            try:
                cfut = asyncio.run_coroutine_threadsafe(self._drain_queue_on_shutdown(), loop)
                cfut.result(timeout=2.0)
            except Exception:
                pass

    async def _drain_queue_on_shutdown(self) -> None:
        if self._queue is None:
            return
        try:
            while True:
                item = self._queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(
                        RuntimeError(
                            f"MicroBatchAggregator({self.function_name}) shut down"
                        )
                    )
        except asyncio.QueueEmpty:
            pass


__all__ = [
    "MicroBatchAggregator",
    "should_disable_batching",
]
