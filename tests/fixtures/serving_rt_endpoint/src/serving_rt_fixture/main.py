from __future__ import annotations

import threading

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const
from gen_worker.models import SDXL
#: THE REAL RATIFIED PAIR (pgw#1621). A lane is `(topology, quant)`; both
#: halves are documents in the vendored `spec/v2` corpus, so this fixture
#: cannot invent one. The v1 constant it replaces is deleted.
SDXL_DIFFUSERS_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")

ORDER: list[str] = []
RELEASE = threading.Event()
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
    lanes={SDXL_DIFFUSERS_BF16: lane(request=const(MiB(64)))},
):
    """Marks nothing; ``load`` builds cheap state, work is gauged."""

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


class OtherModel(SlowModel):
    """Second residency key, same posture — `SlowModel`'s lane, inherited."""


class BrokenUnloadModel(SlowModel):
    """Unload-failure fixture; marks nothing, same inherited lane."""

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
    return Out(value=payload.value, served_by=f"{right.name}+{left.name}")


@entrypoint
def broken(ctx: RequestContext, payload: In, m: BrokenUnloadModel) -> Out:
    return Out(value=m.work(payload), served_by=m.name)
