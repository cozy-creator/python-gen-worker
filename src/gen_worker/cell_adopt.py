"""The ONE adoption vocabulary (pgw#923).

Adopting a pre-built compiled cell — hub-delivered or catalog-discovered — used
to be described twice. The typed description rode ``ModelEvent{ADOPTED}``
(``duration_ms``, ``cache_hits``, ``cache_misses``, ``warmup_s``), which the hub
persists as ``worker_activity_events.kind='compile_cache_adopt'`` with two
partial indexes and a p50/p95/max admin surface. The other description was a
free-text ``aot_adopt`` activity event that put ``family=… key=… sku=…`` in
prose and no numbers anywhere.

Only the free-text one was ever reachable from the path adoptions actually take
(boot attach through ``fleet_cells``); the typed one was reachable only from the
hub-commanded ``ADOPT_COMPILE_CACHE`` operation, which no stack has ever
dispatched. So the measured lane had zero rows on both live stacks while every
real adoption landed at ``duration_ms=0``, and the percentile endpoint
aggregated a population with no members.

The free-text lane is DELETED. This module is what replaced it: the arm returns
a typed outcome instead of a bare bool, the arming policy times it and names the
candidate, and the executor — the one component that owns the wire — turns it
into the ``ModelEvent`` the hub already knows how to store. One fact, one
spelling, and the spelling that carries numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AdoptOutcome", "CellAdoption"]


@dataclass(frozen=True)
class AdoptOutcome:
    """The result of arming ONE candidate artifact on ONE pipeline.

    Truthy exactly when the cell armed, so the many ``if enable_compiled(...)``
    call sites read unchanged while the classified refusal — previously
    reachable only as the ``phase`` of a free-text event — becomes a value the
    caller can act on and put on the wire.

    ``reason`` is the short, stable, countable token (an ``AdoptError.reason``,
    a lane-gate refusal, ``no_cell``); ``detail`` is the human sentence.
    ``identity`` carries ``family=… key=…`` when the candidate's own metadata
    could be read — a refusal must still name the cell it refused, including
    when the refusal IS a metadata problem.
    """

    armed: bool
    reason: str = ""
    detail: str = ""
    identity: str = ""

    def __bool__(self) -> bool:
        return self.armed

    @classmethod
    def hit(cls, identity: str = "") -> "AdoptOutcome":
        return cls(armed=True, identity=identity)

    @classmethod
    def miss(cls, reason: str, detail: str = "", identity: str = "") -> "AdoptOutcome":
        return cls(armed=False, reason=reason, detail=detail[:2000], identity=identity)


@dataclass(frozen=True)
class CellAdoption:
    """One adoption ATTEMPT, measured, with the identity the hub fences on.

    ``arm_ms`` is the wall time of the arm itself — load, bind, wrap, gate —
    and is the same quantity the hub stores as the adoption's ``duration_ms``.
    The warmup half is deliberately absent: a boot-attached cell is armed during
    injection and warmed later, by the setup warmup, so the two numbers are
    known at two different instants and the executor joins them. A hot
    (hub-commanded) adoption arms and warms in one frame and fills both there.
    """

    ref: str
    snapshot_digest: str
    artifact_kind: str
    arm_ms: int
    armed: bool
    reason: str = ""
    detail: str = ""
    #: ``id()`` of the pipeline this attempt armed, so the executor can join the
    #: adoption to that object's own warmup proof (its cache hit/miss deltas)
    #: instead of attributing another slot's evidence to it.
    pipeline_id: int = 0
