"""pgw#1067 (worker half of th#1723): the boot-forward roll-up is a countable
EVENT about a span that already ended — never the RUNNING setup+warmup
activity — so the two must not share the hub's one map slot per kind.

The hub keys `info.Activities` on kind alone (`map[kind]WorkerActivityState`,
ordered by `seq`), and the `warmup` slot feeds `SelfMintActivityRunning` — read
by the stall monitor, cap turnover and the unservable reaper — plus
`InFlightMintKinds`. Both shapes used to be emitted as kind `warmup`, so the
roll-up's COMPLETED state landed on the live activity's slot mid-load and a
still-loading worker read as finished to every one of those readers.

Driven through the real `Executor.ensure_setup` path (the same one
`test_activity_gw601.py` uses), because the collision is a property of the
ORDER production emits in, not of either call site read on its own.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

import msgspec

from gen_worker import activity
from gen_worker.api import Resources, endpoint
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

RUNNING = pb.ActivityState.ACTIVITY_STATE_RUNNING
COMPLETED = pb.ActivityState.ACTIVITY_STATE_COMPLETED


class _In(msgspec.Struct):
    prompt: str = "x"


class _Out(msgspec.Struct):
    y: str


def _boot_updates() -> List[pb.ActivityUpdate]:
    """Every ActivityUpdate a real boot setup+warmup emits, in `seq` order."""
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    @endpoint(resources=Resources(vram_gb_hint=8))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

        def generate_turbo(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

    specs = extract_specs(Ep)
    ex = Executor(specs, _send)

    async def _go() -> None:
        await ex.ensure_setup(specs[0])
        for _ in range(10):  # the sink schedules sends onto this loop
            await asyncio.sleep(0)

    asyncio.run(_go())

    ups = [
        m.activity_update for m in sent
        if m.WhichOneof("msg") == "activity_update"
    ]
    assert ups, "no activity envelopes emitted by a real boot setup"
    return sorted(ups, key=lambda u: u.seq)


def test_the_roll_up_never_lands_on_the_live_warmup_activitys_slot():
    ups = _boot_updates()

    on_the_warmup_slot = [u for u in ups if u.kind == activity.KIND_WARMUP]
    assert on_the_warmup_slot, "the setup+warmup activity reported nothing"

    # The activity's own terminal is its LAST update; everything the hub folds
    # onto that slot before it must still say RUNNING, or a reader asking
    # "is this worker inside its own load window" gets the wrong answer.
    for u in on_the_warmup_slot[:-1]:
        assert u.state == RUNNING, (
            f"a {pb.ActivityState.Name(u.state)} update (phase={u.phase!r}, "
            f"duration_ms={u.duration_ms}) landed on the hub's `warmup` map "
            f"slot while the setup+warmup activity was still running"
        )
    assert on_the_warmup_slot[-1].state == COMPLETED


def test_the_roll_up_is_emitted_under_its_own_kind_carrying_the_span():
    ups = _boot_updates()

    summaries = [u for u in ups if u.kind == activity.KIND_WARMUP_SUMMARY]
    assert len(summaries) == 1, [u.kind for u in ups]
    (summary,) = summaries
    assert summary.state == COMPLETED
    assert summary.phase == activity.PHASE_WARMUP_FORWARD
    # th#1322's numeric home: the roll-up exists to carry the measured span.
    assert summary.duration_ms > 0
    assert "boot warmup" in summary.detail

    # It is an EVENT: it must not disturb the running activity's `_current`.
    assert activity.KIND_WARMUP_SUMMARY != activity.KIND_WARMUP


def test_the_hub_fold_keeps_the_worker_inside_its_load_window():
    """The hub's own fold, replayed: `map[kind] -> state`, applied in `seq`
    order. Replayed up to the roll-up, the `warmup` slot must still read
    RUNNING — that slot IS `SelfMintActivityRunning`."""
    ups = _boot_updates()
    summary_seq = next(
        u.seq for u in ups if u.kind == activity.KIND_WARMUP_SUMMARY)

    slots: Dict[str, int] = {}
    for u in ups:
        if u.seq > summary_seq:
            break
        slots[u.kind] = u.state

    assert slots[activity.KIND_WARMUP_SUMMARY] == COMPLETED
    assert slots[activity.KIND_WARMUP] == RUNNING
