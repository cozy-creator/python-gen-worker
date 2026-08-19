"""Runtime-contract fixture: author code instrumented to OBSERVE the
pgw#1382 lifecycle/concurrency contract (single-flight, drain-then-unload,
best-effort unload, multi-model slots). Kept separate from the
main_v2-shaped fixture so that one stays contract-exact.

All coordination is event-based (progress-gated), never clock-based.
"""

from __future__ import annotations

import threading

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint
from gen_worker.models import SDXL

#: Cross-request observation log: ("request_done" | "unload:<cls>" | ...).
ORDER: list[str] = []
#: Set by the test to let held requests proceed.
RELEASE = threading.Event()
#: Released once per held request entering the model work section.
ENTERED = threading.Semaphore(0)

_gauge_lock = threading.Lock()
_active = 0
HIGH_WATER = 0


def reset() -> None:
    global _active, HIGH_WATER
    ORDER.clear()
    RELEASE.clear()
    while ENTERED.acquire(blocking=False):
        pass
    with _gauge_lock:
        _active = 0
        HIGH_WATER = 0


class In(msgspec.Struct):
    value: int = 0
    hold: bool = False


class Out(msgspec.Struct):
    value: int
    served_by: str


class SlowModel(
    Model[SDXL],
    eager_only="a residency/lifecycle fixture: it compiles nothing by design",
):
    """Eager-permanent; ``load`` builds cheap state, work is gauged."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.loaded = True
        self.name = type(self).__name__

    def work(self, payload: In) -> int:
        global _active, HIGH_WATER
        with _gauge_lock:
            _active += 1
            HIGH_WATER = max(HIGH_WATER, _active)
        try:
            if payload.hold:
                ENTERED.release()
                RELEASE.wait()
            return payload.value * 2
        finally:
            with _gauge_lock:
                _active -= 1

    def unload(self, ctx: LoadContext[SDXL]) -> None:
        ORDER.append(f"unload:{type(self).__name__}")


class OtherModel(SlowModel, eager_only="second residency key, same posture"):
    pass


class BrokenUnloadModel(
    SlowModel, eager_only="unload-failure fixture; compiles nothing"
):
    def unload(self, ctx: LoadContext[SDXL]) -> None:
        ORDER.append("unload:BrokenUnloadModel")
        raise RuntimeError("author unload bug — must not pin the eviction")


@entrypoint
def run(ctx: RequestContext, payload: In, m: SlowModel) -> Out:
    result = m.work(payload)
    ORDER.append("request_done")
    return Out(value=result, served_by=m.name)


@entrypoint
def pair(ctx: RequestContext, payload: In, right: SlowModel,
         left: OtherModel) -> Out:
    # Slot params in declaration order; slot NAME keys the envelope pick.
    return Out(value=payload.value, served_by=f"{right.name}+{left.name}")


@entrypoint
def broken(ctx: RequestContext, payload: In, m: BrokenUnloadModel) -> Out:
    return Out(value=m.work(payload), served_by=m.name)
