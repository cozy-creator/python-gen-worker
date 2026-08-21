"""Boot materializes everything the runtime config names BEFORE the worker advertises its functions; the invariant "never serialize a multi-GB fetch inside a user request" holds because the worker is simply not routable until the fetch is done. A terminal fetch failure is a state (failed, connected-and-unroutable, FnUnavailable{model_unavailable}), never a silent retry loop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Mapping, Optional, Tuple

from . import activity
from . import boot_phases
from . import weight_position
from .models import disk_errors
from .models.refs import WireRef
from .models.store import ModelStore
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

STATE_PENDING = "pending"
STATE_MATERIALIZING = "materializing"
STATE_READY = "ready"
STATE_FAILED = "failed"

REASON_MODEL_UNAVAILABLE = "model_unavailable"


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """The checkpoint refs this release serves, as plain configuration."""

    version: int = 0
    refs: Tuple[WireRef, ...] = ()
    snapshots: Mapping[WireRef, pb.Snapshot] = field(default_factory=dict)

    @property
    def identity(self) -> Tuple[Tuple[str, str], ...]:
        """(ref, digest) pairs — WHAT to materialize, with no version in it."""
        return tuple(
            (str(ref), str(self.snapshots[ref].digest) if ref in self.snapshots else "")
            for ref in self.refs
        )

    @classmethod
    def from_wire(cls, desired: Optional[pb.DesiredResidency]) -> "CheckpointConfig":
        """Read the config off the channel that already carries it."""
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
    """Boot-time materialization, and the readiness it gates."""

    def __init__(
        self,
        store: ModelStore,
        *,
        announce: Optional[Callable[[], Awaitable[None]]] = None,
        warm: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._store = store
        self._announce_cb = announce
        self._warm_cb = warm
        self._task: Optional["asyncio.Task[None]"] = None
        #: pgw#1631: fills whose config has been superseded. They are HELD, not
        #: cancelled — a strong reference so the loop cannot garbage-collect a
        #: task that is still moving bytes into the shared CAS — and they
        #: decide nothing about readiness.
        self._superseded: set["asyncio.Task[None]"] = set()
        self._applied: Optional[CheckpointConfig] = None
        self.state: str = STATE_PENDING
        self.failure: str = ""

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    @property
    def failed(self) -> bool:
        return self.state == STATE_FAILED

    def phase(self) -> "pb.WorkerPhase":
        """This worker's startup phase, as the hub's own vocabulary."""
        if self.state == STATE_READY:
            return pb.WORKER_PHASE_READY
        if self.state == STATE_FAILED:
            return pb.WORKER_PHASE_ERROR
        if self.state == STATE_MATERIALIZING:
            return pb.WORKER_PHASE_DOWNLOADING_MODELS
        return pb.WORKER_PHASE_BOOTING

    def configure(self, config: CheckpointConfig) -> None:
        """Apply a config."""
        if self._applied is not None and config.identity == self._applied.identity:
            logger.info(
                "checkpoint config unchanged (version %d -> %d, %d ref(s)); "
                "state stays %s",
                self._applied.version, config.version, len(config.refs), self.state,
            )
            return

        previous = self._applied
        self._applied = config
        if self._task is not None and not self._task.done():
            # pgw#1631: A SUPERSEDE NEVER CANCELS BYTE MOVEMENT.
            #
            # This used to `cancel()`. Objects are release-agnostic — they are
            # keyed by content — so bytes the old plan is mid-way through
            # fetching are bytes the NEW plan's fill will simply find present.
            # Cancelling them buys nothing and costs the re-fetch, which is the
            # class th#2204 measured as phantom downloads and pgw#1596 turned
            # into a disk-capacity incident: the re-entry that armed it was a
            # supersede that changed nothing about the ref.
            #
            # What supersede DOES mean is that the old task's VERDICT no longer
            # decides anything — `_materialize` checks `self._applied is
            # config` before touching state. The old fill drains; at worst some
            # objects the new plan does not name wait for steady-state GC.
            self._superseded.add(self._task)
            self._task.add_done_callback(self._superseded.discard)
            logger.info(
                "checkpoint config version=%s superseded by version=%d; the "
                "in-flight fill DRAINS rather than cancelling — its objects "
                "are content-keyed and the new plan will find them present",
                previous.version if previous is not None else "none",
                config.version,
            )

        if not config.refs:
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

    async def _materialize(self, config: CheckpointConfig) -> None:
        """Put every configured ref on disk (in config order), then open the readiness gate. Order matters: a pod that fetched the 22 MB interpolator before the 134 GB checkpoint would finish the cheap half and still not serve. The fetch runs under an OPEN activity, which pgw#1630 demoted to TELEMETRY — the watchdog verdict is kernel evidence only, so forgetting it costs the stall report its label, not the process. Scoped to the fetch only: _warm() opens its own activity and must not nest inside this one."""
        with activity.running(activity.KIND_BOOT_MATERIALIZE) as fetch:
            if not await self._fetch_refs(config, fetch):
                if self._current(config):
                    await self._announce()
                return

        if not self._current(config):
            # pgw#1631: this fill was SUPERSEDED while it ran. Its bytes are in
            # the CAS and the live plan's fill will find them present — which is
            # the whole reason it was allowed to drain — but its verdict is
            # about a config nobody asked for any more, so it touches neither
            # readiness nor the warm pass.
            logger.info(
                "superseded fill for version=%d completed; %d ref(s) drained "
                "into the CAS, readiness untouched",
                config.version, len(config.refs),
            )
            return

        await self._warm()

        self.state = STATE_READY
        self.failure = ""
        logger.info(
            "checkpoint config version=%d MATERIALIZED (%d ref(s)); this "
            "worker is now ready and routable",
            config.version, len(config.refs),
        )
        await self._announce()

    async def _report_if_out_of_space(
        self, ref: WireRef, exc: BaseException
    ) -> None:
        """pgw#1612: report an ENOSPC as `insufficient_disk`, with the mount.

        ONE seam for the whole boot fetch, so every raiser under it is covered
        — a per-call-site catch is how half of them stay generic. The token is
        the EXISTING one because the hub-side handling already exists behind
        it; inventing a second vocabulary would mean building the migration
        path twice.
        """
        typed = disk_errors.as_insufficient_disk(
            exc,
            doing=f"materializing {ref}",
            fallback_path=self._store.cache_dir,
        )
        if typed is None:
            return
        logger.error(
            "checkpoint materialization ran OUT OF DISK on %s — this SHAPE "
            "cannot work and re-buying it at the same size will fail the same "
            "way: %s", ref, typed,
        )
        try:
            await self._store.report_insufficient_disk(ref, str(typed))
        except Exception:  # noqa: BLE001 — reporting must not mask the failure
            logger.exception("insufficient_disk report for %s could not be sent", ref)

    def _current(self, config: CheckpointConfig) -> bool:
        """Is ``config`` still the plan this worker is being judged on?

        A superseded fill keeps moving bytes (they are content-keyed and the
        successor wants them) but must not set state — a stale FAILED would
        strand a pod whose live config is fine, and a stale READY would
        advertise weights the live config does not name.
        """
        return self._applied is config

    async def _fetch_refs(
        self, config: CheckpointConfig, fetch: "activity.Activity",
    ) -> bool:
        for ref in config.refs:
            snapshot = config.snapshots.get(ref)
            fetch.note(f"ref={ref}")
            try:
                with boot_phases.span(
                    boot_phases.PHASE_RESIDENCY_CHECK, ref=str(ref),
                ) as check:
                    check.note(f"tree_bytes={weight_position.snapshot_bytes(snapshot)}")
                    resident = await self._store.announce_resident(ref, snapshot)
                    check.classify("resident" if resident else "absent")
                if resident:
                    logger.info("checkpoint %s already resident; no transfer", ref)
                    continue
                path = await self._store.ensure_local(ref, snapshot)
            except asyncio.CancelledError:
                logger.info(
                    "checkpoint materialization superseded while on %s", ref,
                )
                raise
            except Exception as exc:  # noqa: BLE001 — every failure is a STATE
                # pgw#1612: an ENOSPC anywhere under this call is a claim about
                # the SHAPE, not about the attempt. Classified at this ONE boot
                # seam so every raiser under it is covered, and reported on the
                # channel the hub's `insufficient_disk` migration path already
                # reads. Without it the hub sees "this attempt failed" and
                # requeues onto a machine with the identical
                # `container_disk_gb_requested` — measured on th#2246:
                # `8gpqows0j349gm` -> `3zod6pwvn10f4y`, both A100-SXM4-80GB with
                # the same 100 GB, at $1.59/hr until a human cancelled it.
                await self._report_if_out_of_space(ref, exc)
                if not self._current(config):
                    logger.info(
                        "superseded fill for version=%d failed on %s (%s: %s); "
                        "state untouched — a stale verdict must not strand a "
                        "pod whose live config is fine",
                        config.version, ref, type(exc).__name__, exc,
                    )
                    fetch.failed(exc)
                    return False
                self.state = STATE_FAILED
                self.failure = f"{ref}: {type(exc).__name__}: {exc}"
                logger.error(
                    "checkpoint materialization FAILED on %s: %s: %s — this "
                    "worker will not advertise its functions. It is connected "
                    "and not routable; nothing retries this in the background.",
                    ref, type(exc).__name__, exc,
                )
                fetch.failed(exc)
                return False
            logger.info("checkpoint %s materialized at %s", ref, path)
        return True

    async def _warm(self) -> None:
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

    def _announce_soon(self) -> None:
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

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
