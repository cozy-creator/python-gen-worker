"""Which instance record holds which ref, and whether tearing it down would
disturb live work.

These three are pure QUERIES over the executor's record book:
they read records, jobs and residency and mutate nothing. Lifting them out as
free functions is what lets the residency-reconcile lifetime be reasoned about
without the 11,000-line job engine around it.

They stay free functions taking exactly what they read — the executor passes
`self._classes.values()` and `self.jobs.values()` — rather than becoming a
class with a back-reference. A back-reference would re-import the whole
executor and put us back where we started.

Their two MUTATING callers (`Executor.shutdown_instances` and
`Executor._vacate_record`) deliberately stay on the executor: both call
`abandon_background_mint`, which is inside a frozen region, and dragging a
frozen method across a module seam is what that fence forbids.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Protocol, TypeVar

from ..api.binding import wire_ref


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


__all__ = ["record_in_use", "record_refs", "records_holding"]
