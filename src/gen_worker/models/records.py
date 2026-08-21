"""The instance record book: which record holds which ref, whether tearing it down would disturb live work, and the teardown itself."""

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

    ready: bool
    held_refs: List[str]
    specs: Any


class _Job(Protocol):
    finished: bool
    superseded: bool
    spec: Any


R = TypeVar("R", bound=_Record)


def record_refs(rec: _Record) -> List[str]:
    """The wire refs a record's instance holds: the load-time booking keys when stamped, else the current binding derivation (records that never completed a setup)."""
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
    """Whether teardown would disturb live work."""
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

    def __call__(
        self, rec: Any, *, reason: str, code: str = ...,
        free_targets: bool = ...,
    ) -> Awaitable[None]: ...


@dataclass(frozen=True)
class RecordTeardown:
    """What tearing an instance record down needs from outside the record book."""

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
    await seam.abandon_background_mint(
        rec, reason="instance vacate", code="vacate")
    held_refs = record_refs(rec)
    held_objects = rec.held_objects
    released_refs: List[str] = []
    old_obj: Any = None
    inst, rec.instance, rec.ready = rec.instance, None, False
    rec.compile_targets.clear()
    rec.applied_lanes.clear()
    rec.posture.clear()
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
    shutdown = None
    del inst
    server, rec.server = rec.server, None
    if server is not None:
        await asyncio.to_thread(server.stop)
    server = None
    # No gc pass here: the caller holds the load lock and the departing objects' owners were just dropped above, so only the allocator cache needs returning .
    await aflush_memory(collect=False)
    for ref in held_refs:
        tier_before = seam.residency.tier(ref)
        old_obj = held_objects.get(ref)
        owners = records_holding(seam.records, ref)
        if owners:
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
    rec.slot_pipelines = {}
    seam.close_sequence_group(rec)
    old_obj = None
    replacement = None
    owners = []
    held_objects.clear()
    rec.stale = False
    if rec.shared_keys:
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
