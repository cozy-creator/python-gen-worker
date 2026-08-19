"""pgw#1483 / th#2204: the v2 worker's consumer of ``HelloAck.desired_residency``.

The hub declares model residency ONCE, in the HelloAck, and then waits. Before
this module the v2 worker read exactly one field off that ack
(``file_base_url``) and dropped ``disk_refs``, ``hot`` and ``snapshots`` on the
floor — the generation was genuinely acked, so the hub logged the goal as
ACCEPTED while nothing on the pod acted on it. th#2204 measured the consequence
on a rented H100 at $3.29/hr: `placement declined … reason=model_download_pending
… download=[no_position_reported]` at 1 Hz for the life of the rental, twice
punctuated by th#1142's watchdog releasing a reservation whose owner had
"accepted the goal 6m14s ago with NO DOWNLOAD EVER OPENED" and re-electing the
same sole worker. **A livelock with no exit and no terminal state.**

The cause was one level below the deleted ``lifecycle.py``: ``ModelStore`` is
never constructed anywhere in ``src/`` (pgw#1483's census), and
``WorkerMessage.model_event`` is built INSIDE that class — so the hub was
waiting for a fact no live object on the pod was capable of stating.

## What this module is, and what it deliberately is NOT

It is a JOIN, not a re-implementation. Every mechanism the reconcile needs
already shipped and was merely unreachable:

* ``ModelStore.replace_desired_snapshots`` — full-replacement desired identity,
  and it bumps the republish epoch that forces a re-announce of unchanged bytes.
* ``ModelStore.ensure_local`` — the one instrumented funnel (pgw#1455's byte
  positions, pgw#1485's ``already_resident`` phase, lazy record open, terminal
  close in a ``finally``).
* ``ModelStore._confirm_cached_identity`` — the SATISFIED answer: a verified
  cached tree publishes its exact identity as an ``ON_DISK`` ModelEvent without
  a redundant download. Its docstring already described answering a re-received
  plan; nothing called it on v2.

So this file owns three things only: **when** to reconcile (a new generation
arrives), **what order** (declared priority), and **the echo** — the observed
generation the hub reads back off ``StateDelta`` to tell "answered" from
"silent". Everything else is delegated.

## The two answers, and why both are words

* **ref already resident** → ``already_resident`` on the ``weight_fetch``
  position stream plus the ``ON_DISK`` ModelEvent. No download record is
  opened (pgw#1485: a record is a liability, and one opened for bytes the pod
  holds can never close honestly). The hub's placement sees DISK locality and
  DISPATCHES — the wait terminates on a fact, not on a timeout.
* **ref absent** → the ordinary funnel opens, advances and closes. Positions
  ride ``weight_fetch`` so th#2191's decline explain can differentiate a
  progressing fetch from a wedged one.

**Silence is never one of the answers.** A ref that can be neither satisfied
nor fetched reports ``FAILED`` through the funnel's own terminal path, and the
generation is still echoed — the hub then knows the pod ANSWERED and failed,
which is a different state from a pod that never spoke.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from .models.refs import WireRef
from .models.store import ModelStore
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)


class ResidencyGoal:
    """Reconciles the hub's declared residency onto this pod's disk.

    One instance per worker. ``apply`` is called from ``on_hello_ack`` and
    returns immediately: a 134 GB fetch must never run inside the transport's
    message handler, and a reconnect must be able to supersede a reconcile that
    is still running.
    """

    def __init__(self, store: ModelStore) -> None:
        self._store = store
        self._task: Optional["asyncio.Task[None]"] = None
        #: The generation whose reconcile has been ANSWERED — every declared
        #: ref reached a terminal answer (resident, fetched, or failed).
        #: Echoed on StateDelta.observed_residency_generation.
        self.observed_generation: int = 0
        #: The generation currently being reconciled. Distinct from the one
        #: above on purpose: a goal in flight is not a goal answered, and
        #: collapsing them would let the hub read a 134 GB fetch as satisfied.
        self.accepted_generation: int = 0

    # ---- the HelloAck seam -------------------------------------------------

    def apply(self, desired: Optional[pb.DesiredResidency]) -> None:
        """Take the hub's goal. Non-blocking; supersedes any live reconcile.

        A goal with generation 0 is the wire's "unset" and is not a goal.
        """
        if desired is None:
            return
        generation = int(desired.generation or 0)
        if generation <= 0:
            return
        if generation < self.accepted_generation:
            # An older plan cannot supersede a newer one. The hub is
            # authoritative on ordering and never rewinds a live generation;
            # a rewind here would be a replayed frame, not an instruction.
            logger.warning(
                "residency goal generation %d is older than the accepted %d; ignored",
                generation, self.accepted_generation,
            )
            return

        snapshots: Dict[WireRef, pb.Snapshot] = {
            WireRef(str(ref)): snapshot
            for ref, snapshot in desired.snapshots.items()
            if str(ref)
        }
        refs: List[WireRef] = [
            WireRef(str(ref)) for ref in desired.disk_refs if str(ref).strip()
        ]
        # Full replacement, and it bumps the republish epoch — that is what
        # makes an unchanged, already-resident ref re-announce instead of
        # staying silent because nothing changed. The hub re-sending a plan IS
        # the hub asking to be told again.
        self._store.replace_desired_snapshots(snapshots, generation=generation)
        self.accepted_generation = generation

        logger.info(
            "residency goal accepted: generation=%d disk_refs=%d snapshots=%d",
            generation, len(refs), len(snapshots),
        )
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(
            self._reconcile(generation, refs),
            name=f"residency-goal-{generation}",
        )

    # ---- the reconcile -----------------------------------------------------

    async def _reconcile(self, generation: int, refs: List[WireRef]) -> None:
        """Drive every declared ref to a terminal answer, then echo.

        Order is the hub's declared order — ``disk_refs`` is documented as
        orchestrator priority, and a pod that fetched the 22 MB interpolator
        before the 134 GB checkpoint would satisfy placement's cheapest
        precondition last.
        """
        answered = 0
        for ref in refs:
            try:
                # THE SATISFIED ANSWER FIRST. A warm pod holds the tree the
                # goal names; asking the funnel to "fetch" it would open a
                # transfer record for zero bytes, which is the liability
                # pgw#1485/th#2205 exist to remove. Ask disk, answer, move on.
                if await self._store.announce_resident(ref):
                    answered += 1
                    logger.info(
                        "residency already_resident generation=%d ref=%s "
                        "— answered without opening a transfer",
                        generation, ref,
                    )
                    continue
                path = await self._store.ensure_local(ref)
            except asyncio.CancelledError:
                # A newer generation superseded this one. Say so — a cancelled
                # reconcile that echoed nothing is exactly the silence this
                # module exists to remove, and the successor will echo.
                logger.info(
                    "residency reconcile generation=%d superseded while on %s",
                    generation, ref,
                )
                raise
            except Exception as exc:  # noqa: BLE001 — every failure is an ANSWER
                # ensure_local already emitted the FAILED ModelEvent through
                # the funnel's terminal path. Nothing is re-emitted here; the
                # ref is counted as answered because the hub HEARD something.
                logger.error(
                    "residency reconcile generation=%d ref=%s failed: %s: %s",
                    generation, ref, type(exc).__name__, exc,
                )
                answered += 1
                continue
            answered += 1
            logger.info(
                "residency satisfied generation=%d ref=%s at %s", generation, ref, path,
            )

        self.observed_generation = generation
        logger.info(
            "residency goal ANSWERED: generation=%d refs=%d/%d "
            "— observed_residency_generation now rides every StateDelta",
            generation, answered, len(refs),
        )

    # ---- shutdown ----------------------------------------------------------

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
