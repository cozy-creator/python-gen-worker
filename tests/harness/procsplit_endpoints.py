"""pgw#763 split-harness endpoints: real uncatchable deaths and hangs.

Separate from toy_endpoints so the process-split suite owns its fixture file
outright (shared-worktree etiquette).
"""

from __future__ import annotations

import os
import signal
import time

import msgspec

from gen_worker import RequestContext, endpoint


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
