"""gw#621 harness endpoint: setup performs a CPU-quiet "download" — a
ticking byte counter doing no work — then freezes it, driving the progress
beat and self-diagnosis path end-to-end over the hub double."""

from __future__ import annotations

import threading
import time

import msgspec

from gen_worker import RequestContext, endpoint
from gen_worker import activity as activity_mod
from gen_worker import progress

TICK_S = 1.2
TICK_EVERY_S = 0.05
FREEZE_S = 2.5
TOTAL_BYTES = 1000.0

SETUP_DONE = threading.Event()


class PingIn(msgspec.Struct):
    text: str = ""


class PingOut(msgspec.Struct):
    response: str


@endpoint
class SlowFill:
    def setup(self) -> None:
        act = activity_mod.current()
        assert act is not None, "setup must run inside an activity bracket"
        ctr = act.counter(
            "download:toy/model", progress.UNIT_BYTES, total=TOTAL_BYTES)
        end = time.monotonic() + TICK_S
        while time.monotonic() < end:
            ctr.add(10.0)
            time.sleep(TICK_EVERY_S)
        time.sleep(FREEZE_S)
        SETUP_DONE.set()

    def ping(self, ctx: RequestContext, data: PingIn) -> PingOut:
        return PingOut(response="pong")
