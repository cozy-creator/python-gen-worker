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
import gc
import threading
import weakref
from typing import List

import msgspec
import pytest

from gen_worker.api import Resources, endpoint
from gen_worker.executor import Executor, _to_thread_complete
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs


class _In(msgspec.Struct):
    prompt: str = "x"


class _Out(msgspec.Struct):
    y: str


class _Buffer:
    """Trackable stand-in for a partially loaded pipeline: carries a
    reference cycle (real pipelines always do), so only a gc pass frees it."""

    def __init__(self) -> None:
        self.cycle = self


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
        task = asyncio.create_task(_to_thread_complete(load))
        await asyncio.to_thread(started.wait, 10)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task
        # Let shield/done callbacks drain before judging reachability.
        for _ in range(5):
            await asyncio.sleep(0)
        gc.collect()
        assert refs and refs[0]() is None, (
            "the cancelled load's result is still pinned while the "
            "exception lives — rollback cannot free it")
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

    @endpoint(resources=Resources(vram_gb=8))
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
        await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await ex.ensure_setup(specs[0])

    gc.disable()
    try:
        asyncio.run(run())
    finally:
        gc.enable()

    assert alive_at_second_attempt == [False], (
        "the cancelled attempt's partial load survived into the retry — "
        "retries stack allocations until OOM (gw#624)")
