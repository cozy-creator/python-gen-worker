"""The instance record book: which record holds which ref, whether tearing it
down would disturb live work, and the teardown itself.

`record_refs` / `records_holding` / `record_in_use` are pure QUERIES: they read
records, jobs and residency and mutate nothing. `vacate_record` and
`shutdown_instances` are the two MUTATORS over the same book — the residency
side of an instance's death.

Everything here is a free function taking exactly what it reads — the executor
passes `self._classes.values()`, `self.jobs.values()` and a `RecordTeardown`
seam — rather than a class with a back-reference. A back-reference would
re-import the whole executor and put us back where we started.

`abandon_background_mint` is NOT ours (th#1834's ruling, 2026-08-13): under
per-graph-class accretion there is no project called "the mint" to abandon, so
what survives is *"stop supervising further compiles for this record, and give
the card back"* — a supervision verb with a residency side effect. It stays
with the mint supervisor and arrives here as an injected callable on
`RecordTeardown`. Residency CALLS supervision; the direction is the ruling.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
)

from ..api.binding import wire_ref
from .memory import aflush_memory
from . import residency as residency_mod

logger = logging.getLogger(__name__)


class _Record(Protocol):
    """The record fields these queries read. Structural on purpose: the record
    type lives in `executor.py` and importing it here would be the cycle."""

    ready: bool
    held_refs: List[str]
    specs: Any


class _Job(Protocol):
    finished: bool
    superseded: bool
    spec: Any


#: Generic so the executor's concrete record type survives the seam — a
#: Protocol-typed return would erase it at every call site.
R = TypeVar("R", bound=_Record)


def record_refs(rec: _Record) -> List[str]:
    """The wire refs a record's instance holds: the load-time booking keys when
    stamped, else the current binding derivation (records that never
    completed a setup)."""
    if rec.held_refs:
        return list(rec.held_refs)
    return [wire_ref(b) for s in rec.specs for b in s.models.values()]


def records_holding(records: Iterable[R], ref: str) -> List[R]:
    return [rec for rec in records if rec.ready and ref in record_refs(rec)]


def record_in_use(
    rec: _Record,
    *,
    records: Iterable[_Record],
    jobs: Iterable[_Job],
    residency: Any,
    reclaim_ref: Optional[str] = None,
) -> bool:
    """Whether teardown would disturb live work.

    ``reclaim_ref`` narrows a pressure-driven teardown to the candidate that
    selected this record. A different held ref can be pinned by an incoming job
    before its own setup (the common SDXL VAE); that does not make this
    record's idle checkpoint active. ``_vacate_record`` leaves such a pinned ref
    resident because ``release_to_disk`` refuses it. Full-record invalidation
    omits the argument and remains conservative.

    A job on a rebound spec no longer references the record's held refs;
    membership of the job's spec in this record is the honest instance-use
    signal.
    """
    for job in jobs:
        if job.finished or job.superseded or job.spec is None:
            continue
        if job.spec in rec.specs:
            return True
    records = list(records)
    refs = [reclaim_ref] if reclaim_ref is not None else record_refs(rec)
    for ref in refs:
        owners = records_holding(records, ref)
        if len(owners) == 1 and owners[0] is rec and residency.in_use(ref):
            return True
    return False


class _TeardownRecord(_Record, Protocol):
    """The record fields teardown MUTATES, on top of what the queries read."""

    cls: type
    instance: Any
    server: Any
    posture: Any
    stale: bool
    compile_targets: Dict[str, Any]
    applied_lanes: List[Any]
    shared_keys: List[Any]
    held_objects: Dict[str, Any]
    held_snapshot_digests: Dict[str, str]
    held_bindings: List[Any]
    execution_lane_refs: Set[Any]
    slot_pipelines: Dict[str, Any]


class AbandonBackgroundMint(Protocol):
    """th#1834's ruled seam, typed here so the contract is checked and not
    merely described. The body behind it — one background-mint task today, a
    per-graph-class supervisor after the reroute — is the supervisor's to
    change without touching this module."""

    def __call__(
        self, rec: Any, *, reason: str, code: str = ...,
        free_targets: bool = ...,
    ) -> Awaitable[None]: ...


@dataclass(frozen=True)
class RecordTeardown:
    """What tearing an instance record down needs from outside the record book.

    Built per call by the executor, never cached: `on_state_change` is
    reassigned after construction (worker.py wires it once Lifecycle exists),
    so a seam that captured it at `__init__` would call the placeholder.
    """

    records: Iterable[Any]
    residency: Any
    abandon_background_mint: AbandonBackgroundMint
    on_state_change: Callable[[], None]
    close_sequence_group: Callable[[Any], None]
    observe_host_ram_progress: Callable[..., Awaitable[None]]


async def shutdown_instances(seam: RecordTeardown) -> None:
    for rec in seam.records:
        await seam.abandon_background_mint(
            rec, reason="worker shutdown", code="shutdown")
        inst, rec.instance, rec.ready = rec.instance, None, False
        rec.compile_targets.clear()
        rec.applied_lanes.clear()
        rec.posture.clear()
        shutdown = getattr(inst, "shutdown", None)
        if inst is not None and callable(shutdown):
            try:
                if asyncio.iscoroutinefunction(shutdown):
                    await shutdown()
                else:
                    await asyncio.to_thread(shutdown)
            except Exception:
                logger.exception("shutdown() failed for %s", rec.cls.__name__)
        server, rec.server = rec.server, None
        if server is not None:
            await asyncio.to_thread(server.stop)
    seam.on_state_change()


async def vacate_record(rec: _TeardownRecord, seam: RecordTeardown) -> List[str]:
    """Tear an instance down and return refs whose owner was released."""
    # pgw#671: a departing instance takes its background mint with it —
    # stop the driver before any module teardown races a warm forward.
    await seam.abandon_background_mint(
        rec, reason="instance vacate", code="vacate")
    held_refs = record_refs(rec)
    held_objects = rec.held_objects
    released_refs: List[str] = []
    old_obj: Any = None
    inst, rec.instance, rec.ready = rec.instance, None, False
    rec.compile_targets.clear()
    # pgw#1104: the applied lane belonged to THESE weights; the next setup
    # re-reports it or the lane honestly reverts to the binding's.
    rec.applied_lanes.clear()
    # th#1871 P1: and so does the posture. A lever applied to weights that
    # no longer exist would qualify the NEXT instance's measurements with
    # the last one's degradation.
    rec.posture.clear()
    # The next full StateDelta must remove the old address before any
    # replacement can become READY. Do this synchronously before teardown
    # awaits; adoption's second validation then rejects the stale ID.
    seam.on_state_change()
    shutdown = getattr(inst, "shutdown", None)
    if inst is not None and callable(shutdown):
        try:
            if asyncio.iscoroutinefunction(shutdown):
                await shutdown()
            else:
                await asyncio.to_thread(shutdown)
        except Exception:
            logger.exception("shutdown() during vacate failed")
    # A bound method owns its instance. Drop it before measuring cgroup
    # headroom, otherwise this teardown frame itself can retain the whole
    # departing pipeline and suppress a genuine capacity transition.
    shutdown = None
    del inst
    server, rec.server = rec.server, None
    if server is not None:
        await asyncio.to_thread(server.stop)
    server = None
    # No gc pass here: the caller holds the load lock and the departing
    # objects' owners were just dropped above, so only the allocator cache
    # needs returning (pgw#657 fold).
    await aflush_memory(collect=False)
    # gw#494: inspect exactly what the instance BOOKED (held_refs) —
    # re-deriving from spec.models would inspect the wrong keys after a
    # resolution rebind. A multiply-held ref stays resident until its last
    # ready record owner leaves.
    for ref in held_refs:
        tier_before = seam.residency.tier(ref)
        old_obj = held_objects.get(ref)
        owners = records_holding(seam.records, ref)
        if owners:
            # Residency keeps one representative object per wire ref. If
            # it points at the departing record, transfer it to a survivor
            # so the old pipeline can actually be collected. This is an
            # ownership handoff, not an ON_DISK transition.
            if old_obj is not None and seam.residency.obj(ref) is old_obj:
                replacement = next(
                    (owner.held_objects.get(ref) for owner in reversed(owners)
                     if owner.held_objects.get(ref) is not None),
                    None,
                )
                seam.residency.replace_object(ref, replacement)
            if old_obj is not None:
                released_refs.append(ref)
            continue
        if (
            tier_before in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM)
            and seam.residency.release_to_disk(ref)
        ):
            released_refs.append(ref)
    rec.held_refs = []
    rec.held_snapshot_digests = {}
    rec.held_bindings = []
    rec.execution_lane_refs = set()
    rec.held_objects = {}
    rec.slot_pipelines = {}  # pgw#678: pipelines die with the instance
    # pgw#748: the rank siblings are an implementation detail of THIS
    # instance's pipeline; they must not outlive it holding D cards.
    seam.close_sequence_group(rec)
    # Do not let this teardown frame itself retain a departing pipeline
    # while the cgroup probe decides whether capacity really progressed.
    old_obj = None
    replacement = None
    owners = []
    held_objects.clear()
    rec.stale = False
    if rec.shared_keys:
        # Drop this record's holds on content-keyed shared components
        # (gw#479). pgw#636: entries no other record references are NOT
        # drained eagerly — a hot GPU keeps them resident as ordinary LRU
        # candidates so the next pick that matches their bytes aliases
        # them for free; real pressure reclaims them through make_room.
        for key in rec.shared_keys:
            seam.residency.release_shared(key)
        rec.shared_keys.clear()
    seam.on_state_change()
    released_refs = list(dict.fromkeys(released_refs))
    await seam.observe_host_ram_progress(released_refs, collect_host=True)
    return released_refs


__all__ = [
    "AbandonBackgroundMint",
    "RecordTeardown",
    "record_in_use",
    "record_refs",
    "records_holding",
    "shutdown_instances",
    "vacate_record",
]
