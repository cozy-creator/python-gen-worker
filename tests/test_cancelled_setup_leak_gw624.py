"""gw#624: cancelled setup/warmup attempts must not accumulate memory.

Live incident (ie#522 smoke, 2026-07-22): 5 cancelled load retries on ONE
worker process climbed container RAM 3%->97% and VRAM to 83.86GB (OOM on an
80GB card) — each cancelled attempt's partially loaded modules stayed alive
(pinned by the propagating CancelledError's traceback and by uncollected
reference cycles) while the next attempt loaded a fresh copy on top.

Two guards, both revert-turns-red here:
1. ``_to_thread_complete`` drops its joined Task reference on the cancel
   path, so the discarded load result is not pinned by the traceback frame
   for as long as the exception lives (rollback runs in that window).
2. A rolled-back setup schedules an allocation purge; the NEXT attempt runs
   ``gc.collect`` (+ ``torch.cuda.empty_cache``) before allocating, so a
   retry provably starts from baseline.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import threading
import weakref
from typing import List

import msgspec
import pytest

from gen_worker import Resources, endpoint
from gen_worker.executor import Executor, _to_thread_complete
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

from harness.progress_wait import Cadence, await_progress_async


class _In(msgspec.Struct):
    prompt: str = "x"


class _Out(msgspec.Struct):
    y: str


class _Buffer:
    """Trackable stand-in for a partially loaded pipeline: carries a
    reference cycle (real pipelines always do), so only a gc pass frees it."""

    def __init__(self) -> None:
        self.cycle = self


#: The load runs on the loop's default executor, so the test owns one of a
#: KNOWN width — the process default is ``min(32, cpu+4)`` and a round-trip
#: through it proves nothing about which thread served it.
_POOL_WIDTH = 2


async def _workers_past_their_work_items(pool: concurrent.futures.Executor) -> None:
    """Prove every pool thread returned to ``work_queue.get()``.

    CPython's worker does ``work_item.run()`` -> ``del work_item``, and between
    those two the item — hence the load's result — is still reachable from the
    thread. A rendezvous every worker must enter simultaneously cannot be
    reached until all of them are past that ``del``: an ordering fact, with no
    clock and no turn count in it.
    """
    loop = asyncio.get_running_loop()
    barrier = threading.Barrier(_POOL_WIDTH)
    await asyncio.gather(*(loop.run_in_executor(pool, barrier.wait)
                           for _ in range(_POOL_WIDTH)))


async def _released(refs: List[weakref.ref]) -> None:
    """Wait for the reference to go, or for proof it never can.

    A ``call_soon`` queued now runs after every callback queued before it, so
    one pass drains what is pending; when no task is left to run either, there
    is nothing that could still drop the reference and the property is FALSE
    now rather than in five turns' time.
    """
    loop = asyncio.get_running_loop()
    me = asyncio.current_task()
    while True:
        drained = loop.create_future()
        loop.call_soon(drained.set_result, None)
        await drained
        gc.collect()
        if refs and refs[0]() is None:
            return
        if not any(t is not me for t in asyncio.all_tasks(loop)):
            raise AssertionError(
                "the cancelled load's result is still pinned with the loop "
                "quiesced and every executor thread past its work item — "
                "rollback cannot free it")


def test_cancelled_to_thread_join_releases_result() -> None:
    """While the CancelledError (and its traceback) is still alive — the
    exact window setup rollback runs in — the joined thread's result must
    already be unreachable from ``_to_thread_complete``'s frame."""
    refs: List[weakref.ref] = []
    started = threading.Event()
    release = threading.Event()

    def load() -> _Buffer:
        started.set()
        release.wait(10)
        buf = _Buffer()
        refs.append(weakref.ref(buf))
        return buf

    async def run() -> None:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=_POOL_WIDTH)
        asyncio.get_running_loop().set_default_executor(pool)
        task = asyncio.create_task(_to_thread_complete(load))
        # pgw#1349, same reason as the sibling below: a 10 s expiry here
        # cancels before `load` ran, and every assertion afterwards then
        # describes a tree that was never exercised.
        await await_progress_async(
            started.is_set,
            bool,
            what="the load to start on a pool thread",
            cadence=Cadence(),
            gone=lambda: (
                "the load task ended without ever starting"
                if task.done() and not started.is_set() else None),
        )
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task
        # This used to judge reachability after five event-loop turns
        # — a wall clock spelled in turns, and the wrong clock besides. The
        # holder at turn five is the EXECUTOR thread, which had set the result
        # but not yet dropped its work item, so the row red whenever that
        # thread lost the CPU (11/80 locally). Both waits below are orderings.
        await _workers_past_their_work_items(pool)
        await _released(refs)
        del excinfo

    asyncio.run(run())


def test_cancelled_setup_frees_prior_attempt_before_retry() -> None:
    """Real executor path: cancel ensure_setup mid-``setup()``, then start a
    second attempt — the first attempt's buffers must be gone BEFORE the
    second attempt allocates. gc is disabled for the duration so only the
    executor's own purge (not an incidental collection) can pass the test."""
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    entered = threading.Event()
    release = threading.Event()
    refs: List[weakref.ref] = []
    alive_at_second_attempt: List[bool] = []

    @endpoint(resources=Resources(gpu=True))
    class Ep:
        def setup(self) -> None:
            if refs:
                # Attempt 2: judge attempt 1's leftovers at the exact point
                # a real load would start allocating on top of them.
                alive_at_second_attempt.append(refs[0]() is not None)
                return
            buf = _Buffer()
            refs.append(weakref.ref(buf))
            self.buf = buf
            entered.set()
            release.wait(10)

        def generate(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

    specs = extract_specs(Ep)
    ex = Executor(specs, _send)

    async def run() -> None:
        task = asyncio.create_task(ex.ensure_setup(specs[0]))
        # pgw#1349: this was `await asyncio.to_thread(entered.wait, 10)`, and a
        # 10-second give-up here does not FAIL the tape — it silently rewrites
        # it. Cancelling before `setup()` was entered leaves `refs` empty, so
        # attempt 2 takes the FIRST branch, records nothing, and the test dies
        # far below on `assert [] == [False]` naming a leak that never
        # happened. That is how it red master from a merge ref (banked as an
        # unattributed observation against pgw#1328). The wait now ends on the
        # EVENT, on the setup task ending without entering `setup()`, or on a
        # silence window this wait measured for itself.
        await await_progress_async(
            entered.is_set,
            bool,
            what="the first attempt to enter setup()",
            cadence=Cadence(),
            gone=lambda: (
                "the setup task ended without ever entering setup()"
                if task.done() and not entered.is_set() else None),
        )
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await ex.ensure_setup(specs[0])

    # Only the executor's OWN purge may pass this test, so the generational
    # collector is off for the duration. Restored to what it WAS rather than
    # unconditionally enabled: this is process-global state and an xdist worker
    # runs many modules through it.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        asyncio.run(run())
    finally:
        if gc_was_enabled:
            gc.enable()

    assert alive_at_second_attempt == [False], (
        "the cancelled attempt's partial load survived into the retry — "
        "retries stack allocations until OOM (gw#624)"
        if alive_at_second_attempt else
        "attempt 2 never judged attempt 1: the first attempt was cancelled "
        "before it allocated, so this run measured nothing (gw#624)")
