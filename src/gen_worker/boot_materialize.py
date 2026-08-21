"""pgw#1490 / th#2204: this worker materializes what its config names, at BOOT.

Paul's ruling (2026-08-19, "simplify first"): *the production system should use
the exact same code paths as we are using and testing here, not invent or
require new ones*. Locally that chain is ``download`` then ``up`` then ``run``,
and ``run`` requires ``up`` — the weights are on disk before anything can be
asked to serve. A pod is the same three verbs in one process: boot IS ``up``,
so boot pulls what the runtime config names and the worker does not advertise
its functions until it has them.

## What this replaces, and why the shape changed

The predecessor (``residency_goal.ResidencyGoal``, pgw#1483) drove the same
materialization from a hub-instructed *reconcile*: the hub declared a desired
residency, the worker answered, the hub read the answer back off an echoed
generation and only then unparked the requests it had been holding. That
protocol had a live half and a dead half — [[pgw#1373]] deleted the worker's
acquisition loop and left the hub choreographing an acquisition that could
never happen, which billed a rented H100 at $3.29/hr in a livelock with no
terminal state ([[th#2204]]).

The ruling deletes the choreography rather than rebuilding its missing half.
So this module keeps the JOIN pgw#1483 found — ``ModelStore`` constructed,
``announce_resident`` for bytes a staged volume already holds,
``ensure_local`` for bytes it does not — and throws away the *trigger*:

* the refs are read as PLAIN CONFIG (the release's checkpoint list, delivered
  on the ack the pod already receives), not as a goal to be reconciled;
* there is no generation echo, because nothing hub-side waits on one: the hub
  routes on the worker's advertised READINESS, like any web service;
* there is no acquisition coordination of any kind. Paul, 2026-08-19:
  *"Separate workers should have no notion of coordinating with each other to
  optimize downloads to an NFS."* Every worker pulls its own config
  independently; a shared volume is a pre-warm that makes the pull cheap
  (``announce_resident`` answers off disk and no byte moves), never a thing to
  take turns on.

## The invariant, by construction

*Never serialize a multi-GB fetch inside a user request.* The hub used to
enforce this by parking the request. It now holds because the worker is not
routable until the fetch is done — the same reason ``run`` requires ``up``
locally. ``tree_for`` stays a pure lookup and that is now CORRECT rather than
a trap: by the time a dispatch arrives, boot has already put the tree there.

## Failure is a state, not a retry

``ensure_local`` already retries transient failures with backoff and raises
terminal ones. When it raises, this stops: the worker enters ``failed``,
reports ``FnUnavailable{reason=model_unavailable}`` for every function it
would have served, and stays connected-and-unroutable. A pod that silently
retried forever would be the livelock this issue exists to delete, wearing a
different hat.

## Progress reporting is the download's own, not a bolt-on

Byte positions ride the funnel inside ``ModelStore.ensure_local``
(``weight_position``/``activity``), which is ONE code path: with no bound
transport sink — cozy-local, the CLI, tests — the same events land on the
logger as local progress; on a pod they ride the existing worker stream as the
subset the hub consumes (progress, stall detection, completion estimates).
Nothing here reports progress separately.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Mapping, Optional, Tuple

from . import activity
from . import boot_phases
from . import weight_position
from .models.refs import WireRef
from .models.store import ModelStore
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

#: No config has arrived yet. The worker is connected and not routable.
STATE_PENDING = "pending"
#: The configured refs are being materialized. Connected, not routable.
STATE_MATERIALIZING = "materializing"
#: Every configured ref is on disk. The worker advertises its functions.
STATE_READY = "ready"
#: A configured ref could not be materialized. Typed, loud, and terminal for
#: this config — a new config supersedes it, nothing else does.
STATE_FAILED = "failed"

#: The ``FnUnavailable`` token for "this pod does not have the weights". Part
#: of that field's CLOSED vocabulary already (worker_scheduler.proto), and the
#: hub's per-reason lifetime policy (th#1100) re-probes it with backoff.
REASON_MODEL_UNAVAILABLE = "model_unavailable"


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """The checkpoint refs this release serves, as plain configuration.

    ``version`` is the config push's own counter. It orders pushes and lets an
    unchanged re-push be recognized as unchanged; it is NOT a fence anything
    waits on, and nothing echoes it back.
    """

    version: int = 0
    refs: Tuple[WireRef, ...] = ()
    snapshots: Mapping[WireRef, pb.Snapshot] = field(default_factory=dict)

    @property
    def identity(self) -> Tuple[Tuple[str, str], ...]:
        """(ref, digest) pairs — WHAT to materialize, with no version in it.

        Two configs with this identity ask for the same bytes, so the second
        is a no-op even though its version differs. The version is deliberately
        excluded: a re-push of an unchanged plan must not re-pull 134 GB.
        """
        return tuple(
            (str(ref), str(self.snapshots[ref].digest) if ref in self.snapshots else "")
            for ref in self.refs
        )

    @classmethod
    def from_wire(cls, desired: Optional[pb.DesiredResidency]) -> "CheckpointConfig":
        """Read the config off the channel that already carries it.

        ``DesiredResidency`` survives the ruling as a CONFIG MESSAGE — the
        release's checkpoint list plus the resolved snapshot each ref needs to
        fetch. Its reconcile semantics (the goal, the answer, the generation
        fence) do not survive, and nothing here reads them as such.
        """
        if desired is None:
            return cls()
        snapshots: Dict[WireRef, pb.Snapshot] = {
            WireRef(str(ref)): snapshot
            for ref, snapshot in desired.snapshots.items()
            if str(ref).strip() and snapshot is not None
        }
        refs = tuple(
            WireRef(str(ref).strip())
            for ref in desired.disk_refs
            if str(ref).strip()
        )
        return cls(version=int(desired.generation or 0), refs=refs, snapshots=snapshots)


class CheckpointMaterialization:
    """Boot-time materialization, and the readiness it gates.

    One instance per worker. :meth:`configure` is non-blocking — a 134 GB pull
    runs on its own task while the transport keeps reading — and the worker
    reports ``available_functions`` only while :attr:`ready` is true.
    """

    def __init__(
        self,
        store: ModelStore,
        *,
        announce: Optional[Callable[[], Awaitable[None]]] = None,
        warm: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._store = store
        #: Pushes a StateDelta the moment readiness changes, so the hub learns
        #: this worker is routable at materialization time and not up to one
        #: heartbeat later.
        self._announce_cb = announce
        #: pgw#1584: the BOOT WARM PASS, run between the last byte landing and
        #: this worker calling itself ready. The placement is the whole point:
        #: `ready` is still False while it runs, so the heartbeat still reports
        #: `loading_functions`, `first_request_servable` cannot be stamped, and
        #: the hub cannot route here — which turns that milestone from "the
        #: process is up" into "a real forward has completed on this pod"
        #: (th#2233's false-servable gets an actual readiness probe).
        self._warm_cb = warm
        self._task: Optional["asyncio.Task[None]"] = None
        self._applied: Optional[CheckpointConfig] = None
        self.state: str = STATE_PENDING
        #: Populated in STATE_FAILED: "<ref>: <ExcType>: <message>".
        self.failure: str = ""

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    @property
    def failed(self) -> bool:
        return self.state == STATE_FAILED

    def phase(self) -> "pb.WorkerPhase":
        """This worker's startup phase, as the hub's own vocabulary.

        The hub reads these directly (``applyStateDeltaLocked``): ERROR and a
        non-READY phase with no available functions both render as NOT
        routable, and READY renders as routable. No new wire state was needed
        for readiness routing — the vocabulary was already there.
        """
        if self.state == STATE_READY:
            return pb.WORKER_PHASE_READY
        if self.state == STATE_FAILED:
            return pb.WORKER_PHASE_ERROR
        if self.state == STATE_MATERIALIZING:
            return pb.WORKER_PHASE_DOWNLOADING_MODELS
        return pb.WORKER_PHASE_BOOTING

    # ---- the config seam ---------------------------------------------------

    def configure(self, config: CheckpointConfig) -> None:
        """Apply a config. Non-blocking; an unchanged config is a no-op.

        The argument is a `CheckpointConfig`, never a wire message: this is
        the ONE boot path, and the only thing that differs between an author
        running `gen-worker up` in a venv and the same code in a container is
        where the config came from (`from_wire` here, a local runtime config
        there). There is no pod-vs-local branch below this line.
        """
        if self._applied is not None and config.identity == self._applied.identity:
            # A RECONNECT re-delivers the same config. Re-pulling bytes this
            # pod already holds is exactly the phantom download th#2204
            # measured, so an unchanged plan changes nothing — including
            # readiness, which stays exactly where it was.
            logger.info(
                "checkpoint config unchanged (version %d -> %d, %d ref(s)); "
                "state stays %s",
                self._applied.version, config.version, len(config.refs), self.state,
            )
            return

        self._applied = config
        if self._task is not None and not self._task.done():
            # A NEW config supersedes an in-flight materialization of an old
            # one. This is config replacement, not a reconcile race.
            self._task.cancel()

        if not config.refs:
            # A weightless release, or a release whose config names nothing.
            # There is nothing to wait for, so readiness is immediate — the
            # gate must never be able to strand a pod that has no weights.
            #
            # NO WARM PASS HERE, and the reason is th#2233 rather than this
            # branch. A config naming zero refs also names zero checkpoint
            # BINDINGS, so `boot_picks` has nothing to bind and every function
            # would skip with "no boot-time checkpoint binding" — a warm call
            # placed here would do nothing but say so. th#2233 measured that
            # the hub sends zero `disk_refs` for `fixed` bindings FLEET-WIDE
            # (the only HelloAck seeder covers dynamic slots, of which the hub
            # has none), so today every pod lands in this branch and the warm
            # pass rides that fix — which is the same fix that makes
            # `first_request_servable` mean anything at all.
            self.state = STATE_READY
            self.failure = ""
            logger.info(
                "checkpoint config version=%d names NO refs; this worker is "
                "ready with nothing to materialize", config.version,
            )
            self._announce_soon()
            return

        self.state = STATE_MATERIALIZING
        self.failure = ""
        # Banks each ref's snapshot identity so the funnel can fetch by digest
        # and `announce_resident` can recognize bytes already on disk.
        self._store.replace_desired_snapshots(
            dict(config.snapshots), generation=config.version,
        )
        logger.info(
            "checkpoint config version=%d: materializing %d ref(s) before this "
            "worker advertises ready: %s",
            config.version, len(config.refs), ", ".join(str(r) for r in config.refs),
        )
        self._announce_soon()
        self._task = asyncio.create_task(
            self._materialize(config), name=f"boot-materialize-{config.version}",
        )

    # ---- the pull ----------------------------------------------------------

    async def _materialize(self, config: CheckpointConfig) -> None:
        """Put every configured ref on disk, then open the readiness gate.

        Order is the config's own order: a pod that fetched the 22 MB
        interpolator before the 134 GB checkpoint would finish the cheap half
        first and still not be able to serve.

        pgw#1613: THE FETCH RUNS UNDER AN OPEN ACTIVITY, and that is load-
        bearing, not bookkeeping. A cold multi-hundred-GB fill saturates the
        container's cores and holds this process's event loop quiet for
        minutes. The compute-child watchdog arms on that silence and then asks
        what is OPEN (`procsplit/parent.py` `_hang_verdict`): with nothing
        declared it returns `loop_wedged_no_activity` and SIGKILLs a child that
        is provably burning CPU. Its `held` branch — "the child's event loop is
        starved by accounted work, not hung" — exists for exactly this case and
        can only fire when an activity is open. Measured twice on minimax-h3
        (~105 GB), at two different pins, killed at 356 s and 687 s.

        Scoped to the FETCH ONLY: `_warm()` opens its own `warmup` activity and
        must not nest inside this one.
        """
        with activity.running(activity.KIND_BOOT_MATERIALIZE) as fetch:
            if not await self._fetch_refs(config, fetch):
                await self._announce()
                return

        # pgw#1584: WEIGHTS ARE DOWN, NOTHING IS SERVABLE YET. The warm pass
        # goes exactly here — after the last byte and before the state flips —
        # so the first synthetic forward happens while this worker is still
        # advertising `loading_functions`. It never raises (see `_warm`).
        await self._warm()

        self.state = STATE_READY
        self.failure = ""
        logger.info(
            "checkpoint config version=%d MATERIALIZED (%d ref(s)); this "
            "worker is now ready and routable",
            config.version, len(config.refs),
        )
        await self._announce()

    async def _fetch_refs(
        self, config: CheckpointConfig, fetch: "activity.Activity",
    ) -> bool:
        """Put every configured ref on disk. ``False`` when one of them did not
        land, with `state`/`failure` already set by the caller's contract."""
        for ref in config.refs:
            snapshot = config.snapshots.get(ref)
            fetch.note(f"ref={ref}")
            try:
                # pgw#1555: TIMED, because a `resident` verdict deletes the
                # only other row this ref would have produced. The check is a
                # verified manifest match (pgw#1511), so on a warm 134 GB
                # volume it IS the boot — and until this span existed that
                # boot rendered as an empty ladder.
                with boot_phases.span(
                    boot_phases.PHASE_RESIDENCY_CHECK, ref=str(ref),
                ) as check:
                    check.note(f"tree_bytes={weight_position.snapshot_bytes(snapshot)}")
                    resident = await self._store.announce_resident(ref, snapshot)
                    check.classify("resident" if resident else "absent")
                if resident:
                    # The volume-staged case, and the ordinary warm reboot.
                    # Asking the funnel to "fetch" bytes the pod holds opens a
                    # transfer record for zero bytes — the 0-of-N phantom
                    # pgw#1485 and th#2205 both measured.
                    logger.info("checkpoint %s already resident; no transfer", ref)
                    continue
                path = await self._store.ensure_local(ref, snapshot)
            except asyncio.CancelledError:
                logger.info(
                    "checkpoint materialization superseded while on %s", ref,
                )
                raise
            except Exception as exc:  # noqa: BLE001 — every failure is a STATE
                self.state = STATE_FAILED
                self.failure = f"{ref}: {type(exc).__name__}: {exc}"
                logger.error(
                    "checkpoint materialization FAILED on %s: %s: %s — this "
                    "worker will not advertise its functions. It is connected "
                    "and not routable; nothing retries this in the background.",
                    ref, type(exc).__name__, exc,
                )
                # The activity's terminal is FAILED, not COMPLETED: a silent
                # death is a bug by this module's own contract. `failed()` and
                # the `running` exit are both idempotent through `_done`.
                fetch.failed(exc)
                return False
            logger.info("checkpoint %s materialized at %s", ref, path)
        return True

    # ---- the boot warm pass ------------------------------------------------

    async def _warm(self) -> None:
        """Run the boot warm pass, if one is wired. NEVER raises.

        The posture is pgw#1584's ruling: a warm-pass failure degrades LOUDLY
        and does not brick the pod. `ServeLoop.boot_warmup` already confesses
        per entrypoint with a `serve_degrade` event and returns rather than
        raising; this is the second belt, because a pod that refuses to become
        ready over a failed OPTIMIZATION is strictly worse than one that serves
        and pays the first-call tax it would have paid anyway.
        """
        if self._warm_cb is None:
            return
        try:
            await self._warm_cb()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — an optimization never costs a boot
            logger.warning(
                "boot warm pass raised; this worker becomes ready anyway and "
                "the first real request pays the cold cost", exc_info=True,
            )

    # ---- readiness announcement -------------------------------------------

    def _announce_soon(self) -> None:
        """Announce from a synchronous caller, without blocking it."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.create_task(self._announce(), name="readiness-announce")

    async def _announce(self) -> None:
        if self._announce_cb is None:
            return
        try:
            await self._announce_cb()
        except Exception:  # noqa: BLE001 — a missed announce is not a failure
            logger.warning("readiness announce failed", exc_info=True)

    # ---- shutdown ----------------------------------------------------------

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
