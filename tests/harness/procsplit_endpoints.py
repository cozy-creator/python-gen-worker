"""pgw#763 split-harness endpoints: real uncatchable deaths and hangs.

Separate from toy_endpoints so the process-split suite owns its fixture file
outright (shared-worktree etiquette).
"""

from __future__ import annotations

import asyncio
import os
import signal
import time

import msgspec

from gen_worker import RequestContext, activity, endpoint


class ProbeIn(msgspec.Struct):
    text: str = ""


class ProbeOut(msgspec.Struct):
    response: str


@endpoint
class SplitProbe:
    def echo(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        return ProbeOut(response=f"echo:{data.text}")

    def die_hard(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """SIGKILL self: the cgroup-OOM shape — no exception, no finally."""
        os.kill(os.getpid(), signal.SIGKILL)
        return ProbeOut(response="unreachable")

    def sleepy(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Long cancellable job: measures cancel latency across the seam."""
        for _ in range(1200):
            ctx.raise_if_cancelled()
            time.sleep(0.05)
        return ProbeOut(response="done")

    def freeze(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """SIGSTOP self: a wedge the WatchdogSec analog must detect."""
        os.kill(os.getpid(), signal.SIGSTOP)
        return ProbeOut(response="unfrozen")

    async def starve_loop(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """pgw#771: an inductor compile's shape — an ASYNC handler that burns
        CPU without yielding, so the event loop (and every ping riding it) goes
        silent while the process is demonstrably alive and working.

        Wrapped in the real activity + evidence watchdog bracket, because that
        is what a self-mint compile does and it is what the parent's hang
        verdict is required to defer to.
        """
        seconds = float(data.text or "8")
        with activity.running("self_mint_compile") as act:
            with activity.watchdog(act):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    pow(7, 4001, 10**9 + 7)   # real CPU, no await, no yield
        return ProbeOut(response=f"compiled:{seconds:.0f}")

    async def async_wait(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """A job that legitimately WAITS: real asyncio sleeps, so the process
        burns no CPU and moves no disk while its loop keeps turning.

        The shape that falsified the parent's first stall report — a healthy
        15s marco-polo-slow was called stalled on the first real-stack run
        because /proc evidence alone cannot tell waiting from wedged.
        """
        for _ in range(int(float(data.text or "8") / 0.1)):
            await asyncio.sleep(0.1)
        return ProbeOut(response="waited")
