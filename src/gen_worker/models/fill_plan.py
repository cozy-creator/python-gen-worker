"""pgw#1631: the fill PLAN — one predicate, so a divergent precondition is unwritable.

pgw#1596 was not a bug in the CAS. It was a bug in the SHAPE of a gate: the
headroom check computed its cost from the REQUEST (walk the manifest, sum every
file) while the fill computed its work from the DELTA (skip what
``contains(digest, size)`` already answers for). Two derivations of one quantity
drifted, and a 105 GB pull died 157 MB from the end on a disk that fit it fine.

The fix that landed subtracted the resident bytes. The fix here removes the
second derivation: there is ONE function that decides whether an object is
present, ONE plan built out of it, and the gate is HANDED that plan rather than
handed a manifest to price. A gate that cannot see file sizes cannot re-derive a
cost, and a precondition that cannot be re-derived cannot disagree.

The predicate is `LocalCAS.contains` — presence, not integrity, resolved by one
``lstat``. That is deliberate and it is the store's own documented contract: a
declared size makes a truncated object answer False for free, and bytes
corrupted in place answer True, which is `verify_object`'s job on the read side
and not a planning question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence


class _Cas(Protocol):
    def contains(self, ref: Any, *, size: int | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlannedObject:
    """One content-addressed object a manifest names."""

    digest: str
    size_bytes: int
    path: str = ""


@dataclass(frozen=True, slots=True)
class FillPlan:
    """What a fill will SKIP and what it will FETCH, decided once.

    ``missing_bytes`` is the only cost any precondition on this path is allowed
    to charge for. It is not an estimate of the tree; it is the arithmetic
    consequence of the same skip decision the fetch loop will make.
    """

    present: tuple[PlannedObject, ...] = ()
    missing: tuple[PlannedObject, ...] = ()
    #: Objects the manifest named without a digest. They are counted as
    #: MISSING (the honest direction — they will be fetched) and kept
    #: separately so a refusal can say the plan was built on incomplete input.
    undigested: tuple[PlannedObject, ...] = ()

    @property
    def present_bytes(self) -> int:
        return sum(o.size_bytes for o in self.present)

    @property
    def missing_bytes(self) -> int:
        return sum(o.size_bytes for o in self.missing)

    @property
    def total_bytes(self) -> int:
        return self.present_bytes + self.missing_bytes

    @property
    def object_count(self) -> int:
        return len(self.present) + len(self.missing)

    def describe(self) -> str:
        """The plan's own arithmetic, for a refusal that shows its working."""

        return (
            f"{self.missing_bytes} missing of {self.total_bytes} "
            f"({len(self.missing)} of {self.object_count} objects; "
            f"{self.present_bytes} already banked)"
        )


def is_present(cas: _Cas, digest: str, size_bytes: int) -> bool:
    """THE skip predicate. Both the plan and the fetch loop call this one.

    An unreadable or unparseable object answers False — absent, which is the
    honest direction: it will be fetched, and the fetch is what decides.
    """

    ref = str(digest or "").strip()
    if not ref:
        return False
    try:
        return bool(cas.contains(ref, size=int(size_bytes)))
    except Exception:  # noqa: BLE001 — a probe must never be the failure
        return False


def plan_fill(cas: _Cas, entries: Iterable[Any]) -> FillPlan:
    """Split a manifest into what this store holds and what it must fetch.

    ``entries`` is anything with ``digest`` and ``size_bytes``: a
    ``pb.SnapshotFile``, a ``WorkerResolvedRepoFile``, a ``TransferGrant``.
    """

    present: list[PlannedObject] = []
    missing: list[PlannedObject] = []
    undigested: list[PlannedObject] = []
    for entry in entries:
        digest = str(getattr(entry, "digest", "") or "").strip()
        size = int(getattr(entry, "size_bytes", 0) or 0)
        path = str(getattr(entry, "path", "") or "")
        obj = PlannedObject(digest=digest, size_bytes=size, path=path)
        if not digest:
            undigested.append(obj)
            missing.append(obj)
            continue
        (present if is_present(cas, digest, size) else missing).append(obj)
    return FillPlan(
        present=tuple(present),
        missing=tuple(missing),
        undigested=tuple(undigested),
    )


def plan_for_snapshot(cache_dir: Any, files: Sequence[Any]) -> FillPlan:
    """The plan for a snapshot's file list against this pod's CAS.

    An unopenable CAS yields an all-missing plan rather than an exception: a
    store nobody can read holds nothing, which is the conservative direction
    for a headroom gate and the correct one for a fetch.
    """

    from .cache_paths import open_worker_cas

    try:
        cas = open_worker_cas(cache_dir)
    except Exception:  # noqa: BLE001 — a probe must not be the failure
        return FillPlan(missing=tuple(
            PlannedObject(
                digest=str(getattr(f, "digest", "") or ""),
                size_bytes=int(getattr(f, "size_bytes", 0) or 0),
                path=str(getattr(f, "path", "") or ""),
            )
            for f in files
        ))
    return plan_fill(cas, files)


__all__ = [
    "FillPlan",
    "PlannedObject",
    "is_present",
    "plan_fill",
    "plan_for_snapshot",
]
