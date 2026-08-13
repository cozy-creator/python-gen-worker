"""pgw#687: handlers that ignore cancellation — the observed wedge shape.

A modelopt calibration loop never polls ``ctx.cancelled``, and a sync
handler's thread cannot be killed, so ``CancelJob`` is a no-op for it while
the job keeps its GPU permit and instance gate. ``stubborn`` reproduces
exactly that: it blocks on an event the test owns, so the test decides
whether the unwind never lands or lands late. ``polite`` is the peer job the
hub reassigns 61 s later, and ``patient`` is the same long handler done right
(it polls), so the unwind-works row and the wedge row use one code path.

No torch, no GPU: ``RunJob.compute.accelerator="cuda"`` declares the GPU
permit the wedge holds, exactly as ``test_p6``'s serialization row does.
"""

from __future__ import annotations

import threading
import time
from typing import List

import msgspec

from gen_worker import RequestContext, endpoint

#: Set by ``stubborn`` when its handler body is running.
STUBBORN_RUNNING = threading.Event()
#: Released by the test when (if ever) the wedged handler may return.
STUBBORN_RELEASE = threading.Event()
#: Every handler compiled graph, in order — proves which jobs actually EXECUTED.
CALLS: List[str] = []

_MAX_BLOCK_S = 120.0


def reset() -> None:
    STUBBORN_RUNNING.clear()
    STUBBORN_RELEASE.clear()
    CALLS.clear()


class WedgeIn(msgspec.Struct):
    text: str = ""


class WedgeOut(msgspec.Struct):
    response: str


@endpoint
class Wedge:
    def stubborn(self, ctx: RequestContext, data: WedgeIn) -> WedgeOut:
        CALLS.append("stubborn")
        STUBBORN_RUNNING.set()
        # Deliberately never polls ctx.cancelled while blocked.
        STUBBORN_RELEASE.wait(timeout=_MAX_BLOCK_S)
        ctx.raise_if_cancelled("late unwind")
        return WedgeOut(response="stubborn-finished")

    def polite(self, ctx: RequestContext, data: WedgeIn) -> WedgeOut:
        CALLS.append("polite")
        ctx.raise_if_cancelled()
        return WedgeOut(response="polite-ok")

    def patient(self, ctx: RequestContext, data: WedgeIn) -> WedgeOut:
        """Long, but cancellable — polls between units like a real loop."""
        CALLS.append("patient")
        STUBBORN_RUNNING.set()
        deadline = time.monotonic() + _MAX_BLOCK_S
        while time.monotonic() < deadline:
            ctx.raise_if_cancelled()
            if STUBBORN_RELEASE.wait(timeout=0.05):
                break
        return WedgeOut(response="patient-finished")
