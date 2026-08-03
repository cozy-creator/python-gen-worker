"""The driver for a SCHEDULED MINT GOAL (th#1359 Part 2, recut by pgw#930).

Renamed from ``forge.py``. A pod that holds a mint goal boots the SAME release
image as any other; it mints what it was scheduled for, publishes, states a
TYPED TERMINAL, and — **only if it holds no serve goal** — retires.

That last conditional is the §1.17 change. This module used to be "the forge
pod's whole life", and it called ``start_drain`` unconditionally because a forge
pod had nothing else to do. Under composable goals a pod may hold a mint goal
AND a serve goal at once (Paul: *"you can spawn a GPU and tell it to mint some
AOT-cells, while also using it to serve jobs"*), and draining that pod when its
mint finishes would tear down a live serving instance. The terminal is still
stated either way — what it MEANS for the pod is what the goal set decides.

Why a driver at all — the two measured blockers (pgw#846)
---------------------------------------------------------
A mint on a SERVING pod is a background guest of something else's lifecycle,
and the program lost two multi-hour mints to exactly that:

* attempt sixteen ran 29 minutes and died ``self_mint_abort/abandoned_shutdown``
  — a DRAIN tore the endpoint instances down under it
  (``executor.shutdown_instances()`` abandons every background mint
  unconditionally; its only caller is the drain path). The worker itself never
  shut down. The pod then ended ``boot_ended_uncompiled``, which is terminal:
  it serves eager for the rest of its life and publishes nothing.
* attempts fourteen and fifteen died against a VRAM budget whose whole premise
  is a co-resident tenant model.

A serving pod has no way to say "I am not done". A forge pod does not need one:
minting IS its purpose, so its terminal state is the pod's terminal state, and
nothing else has standing to end it first. The hub half of that promise (no
tenant dispatch, no idle-turnover victimhood) is th#1359's tensorhub side; this
half makes the pod's own disposition legible and bounded.

The three terminals, and why they are typed
-------------------------------------------
Every forge pod ends in exactly one of :data:`TERMINAL_PUBLISHED`,
:data:`TERMINAL_NOTHING_OWED`, :data:`TERMINAL_REFUSED`, or
:data:`TERMINAL_FAILED` — and it says so on the wire before it drains. A forge
pod that ends silently is indistinguishable from one that hung, which is the
observation gap that made ``boot_ended_uncompiled`` necessary in the first
place.

No deadline lives here, deliberately (§ no magic timeouts): the forge waits on
the mint's own completion, never on a clock. A mint that stops making progress
is already the hub's ``worker_activity_stalled`` verdict against the live
``aot_mint_phases``/``self_mint_compile`` activity — a progress-staleness
judgement, which is the only kind allowed to kill a pod.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from . import activity as activity_mod
from . import fleet_cells
from . import worker_goals

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .lifecycle import Lifecycle

logger = logging.getLogger(__name__)

#: The wire kind every mint-goal terminal rides. One kind, four phases, so the
#: hub's view of "how did mint goals end" is a `GROUP BY phase`.
#:
#: The STRING is deliberately unchanged across the pgw#930 rename: it is an
#: emitted activity kind the hub already groups on, and renaming it would
#: silently orphan every historical row. The Python name says what it is now.
KIND_MINT_GOAL = "forge_terminal"

#: A cell was minted AND published. The only outcome that discharges what the
#: pod was bought for.
TERMINAL_PUBLISHED = "published"

#: Nothing was owed: the declaration arms an existing cell, or declares no
#: compile family at all. NOT a failure — and the pod retires immediately
#: rather than idling on a paid card.
TERMINAL_NOTHING_OWED = "nothing_owed"

#: A mint ran and a cell exists locally, but the publish was refused or
#: withheld (hub refusal, unattestable axes, quota). The pod did its job; the
#: fleet did not get the artifact.
TERMINAL_REFUSED = "publish_refused"

#: The mint itself did not produce a cell (declined, aborted, errored). The
#: typed WHY is already on the wire as `self_mint_skipped`/`self_mint_abort`;
#: this terminal says the pod is finished asking.
TERMINAL_FAILED = "no_cell"

#: Drain deadline a retiring mint-only pod gives itself. Such a pod holds no
#: tenant work by construction, so this bounds only the flush of its own
#: already-produced results — never a wait for work to finish. It is not
#: consulted at all when a serve goal is also held, because that pod does not
#: retire here.
MINT_GOAL_DRAIN_DEADLINE_MS = 30_000

#: How often the driver re-checks that its publish threads have finished. The
#: publish registry (`fleet_cells.publishes_in_flight`) is the fact; this is
#: only the polling cadence for a thread-to-loop handoff, not a deadline.
_PUBLISH_POLL_S = 2.0


_disposition: Optional[asyncio.Event] = None


def disposition_event() -> asyncio.Event:
    """The boot's "mint disposition is final" latch.

    Set from ``Executor._mark_warm_complete``, which is the ONE point that
    already means exactly this on every path — inline setup, adopted cell,
    eager-without-mint, and the background mint's own ``finally``. Reusing it
    keeps the forge from inventing a second notion of "boot is over" that
    could disagree with the one the boot telemetry already publishes.
    """
    global _disposition
    if _disposition is None:
        _disposition = asyncio.Event()
    return _disposition


def note_disposition_final() -> None:
    """Called (sync, on the loop thread) when a boot's mint disposition is
    final. Harmless when no forge driver is waiting."""
    try:
        disposition_event().set()
    except Exception:  # pragma: no cover - a latch must never break a boot
        logger.debug("forge disposition latch failed", exc_info=True)


def _classify() -> tuple[str, str]:
    """(phase, detail) for this pod's single terminal, from the publish ledger.

    The ledger is the only evidence that survives the mint driver: a published
    cell is a hub-acknowledged checkpoint, a refusal is the hub's own code.
    Everything else — why a mint declined, where it aborted — is already typed
    on the wire under its own kind and is not restated here.
    """
    published = fleet_cells.published_cells()
    if published:
        pairs = ", ".join(f"{k}->{v}" for k, v in sorted(published.items()))
        return TERMINAL_PUBLISHED, f"published {len(published)} cell(s): {pairs}"
    refused = fleet_cells.refused_publishes()
    if refused:
        pairs = ", ".join(f"{k} ({v})" for k, v in sorted(refused.items()))
        return TERMINAL_REFUSED, f"minted but not published: {pairs}"
    return TERMINAL_FAILED, (
        "no cell was published — see this pod's self_mint_skipped / "
        "self_mint_abort events for the typed cause"
    )


async def run(lifecycle: "Lifecycle") -> None:
    """Drive a scheduled mint goal from "boot finished" to its terminal.

    Never raises: a driver that dies leaves a pod that mints nothing and never
    reports, which is strictly worse than any outcome it could state.
    """
    if not worker_goals.current().drives_mint():
        return
    try:
        await _run(lifecycle)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("mint-goal driver failed")
        activity_mod.emit_event(
            KIND_MINT_GOAL,
            f"mint-goal driver failed: {type(exc).__name__}: {exc}",
            phase=TERMINAL_FAILED,
        )
        _retire_if_owed(lifecycle)


def _retire_if_owed(lifecycle: "Lifecycle") -> None:
    """Drain the pod iff the mint goal was the only thing asked of it.

    pgw#930: the single line that makes ``serve=True, mint=True`` safe. A pod
    that also holds a serve goal has more to do than this mint, so its mint
    terminal is a report, not an ending.
    """
    goals = worker_goals.current()
    if goals.retires_when_mint_completes():
        lifecycle.start_drain(MINT_GOAL_DRAIN_DEADLINE_MS)
        return
    logger.info(
        "mint goal complete; NOT retiring — this pod also holds a serve goal "
        "(pgw#930). Retiring here would drain a live serving instance.")


async def _run(lifecycle: "Lifecycle") -> None:
    goals = worker_goals.current()
    logger.info(
        "mint goal driver: goals serve=%s mint=%s; this pod mints, publishes "
        "and reports a typed terminal%s",
        goals.serve, goals.mint,
        "; it retires when done" if goals.retires_when_mint_completes()
        else "; it keeps serving when done")
    await disposition_event().wait()

    # A release may declare several compile families; `_mark_warm_complete`
    # latches on the first to finish. Join the rest before calling the pod
    # done — retiring with a live mint would reproduce exactly the abandonment
    # this mode exists to prevent.
    tasks = lifecycle.executor.background_mint_tasks()
    if tasks:
        logger.info("forge: joining %d background mint(s)", len(tasks))
        await asyncio.gather(*tasks, return_exceptions=True)

    # The upload is the part that must not be raced: "the pod must survive the
    # upload or the cell is lost" (fleet_cells._publish_async). A forge pod is
    # the one pod that can simply wait.
    while True:
        in_flight = fleet_cells.publishes_in_flight()
        if not in_flight:
            break
        logger.info("forge: waiting on %d publish(es): %s",
                    len(in_flight), ", ".join(sorted(in_flight)))
        await asyncio.sleep(_PUBLISH_POLL_S)

    phase, detail = _classify()
    if phase == TERMINAL_FAILED and not lifecycle.executor.declares_compile():
        phase, detail = TERMINAL_NOTHING_OWED, (
            "this release declares no compile family — a mint goal against "
            "it has nothing to mint")
    activity_mod.emit_event(KIND_MINT_GOAL, detail, phase=phase)
    logger.info("mint-goal terminal: %s — %s", phase, detail)
    _retire_if_owed(lifecycle)
