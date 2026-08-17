"""pgw#1351: what a pod actually TRANSFERRED to boot a model, as a wire fact.

e2e#1892's full journey set out to measure warm-boot dedup across three pods
and could not: ``worker_activity_events`` exposes no per-pod downloaded-bytes
counter, so all three pods reported the same ``tree on disk 12.93 GiB`` — which
is TREE SIZE, not wire bytes, and identical on a pod that fetched everything
and a pod that fetched nothing. The journey reported the absence rather than
inventing a number, and the one proxy left (three pulls within 4% of each other
by wall clock) is coarse enough to be worth almost nothing.

The pull path has known these numbers all along. ``_ensure_objects`` decides,
per object, whether the pod's own CAS already had it, whether the endpoint
volume supplied it, or whether it must come over the wire, and the tensorfs
transfer returns a report of what it moved. Nothing carried them off-pod.

So the pull emits ONE self-contained completed event when a snapshot lands, and
warm-boot dedup becomes a query:

    GET /v1/admin/worker-activity-events?kind=snapshot_pull&pod=<id>

with ``fetched_bytes`` over ``tree_bytes`` as the answer and
``resident_objects`` as the evidence behind it.

# Why the residency split is three ways and not two

A pod-local CAS hit and an endpoint-volume fill are BOTH "not fetched", and
folding them would make the dedup number a property of the volume rather than
of the pod. They are the numerator of different questions — the pod's own warm
fraction versus the volume's — so they are counted apart and named apart.

``late_resident_objects`` is the third: an object the residency scan judged
missing that the transfer then found already present, because a concurrent pull
landed it in between. It is a real event on a pod running two boots, and
counting it as fetched would overstate the wire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: The phase token on the pull's completion row. There is exactly one today —
#: this is an EVENT, not an activity with a phase sequence — and it is stated
#: rather than left empty so a reader filtering by phase gets a row.
PHASE_PULLED = "pulled"


@dataclass(frozen=True, slots=True)
class SnapshotPullStats:
    """One snapshot pull, reduced to what it cost the wire.

    Every field is a COUNT or a BYTE COUNT of objects in the snapshot's own
    manifest, so ``fetched_* + resident_* + filled_* + late_resident_*``
    accounts for ``requested_objects`` exactly. A reader that finds it does not
    is looking at a shape this dataclass does not model, which is worth knowing.
    """

    #: Every object the snapshot's manifest requires, deduped by digest.
    requested_objects: int = 0
    #: The sum of those objects' sizes — the DENOMINATOR of the dedup fraction.
    #: Deliberately not "the size of the tree on disk": a projection writes
    #: stubs and symlinks beside the objects, and `disk_gc.tree_bytes` counts
    #: those too. This is the number `fetched_bytes` is comparable with.
    tree_bytes: int = 0

    #: Objects pulled over the wire, and their bytes. From the transfer's own
    #: report, never re-derived from the grant sizes.
    fetched_objects: int = 0
    fetched_bytes: int = 0

    #: Objects THIS POD'S CAS already held. The dedup numerator.
    resident_objects: int = 0
    resident_bytes: int = 0

    #: Objects supplied by the endpoint volume. Not pod-local dedup — a
    #: different question, kept separate so it cannot flatter this one.
    filled_objects: int = 0
    filled_bytes: int = 0

    #: Objects the residency scan called missing and the transfer found
    #: present: a concurrent pull landed them in between.
    late_resident_objects: int = 0

    def accounted(self) -> int:
        """Objects this record accounts for. Equals requested_objects."""
        return (
            self.fetched_objects
            + self.resident_objects
            + self.filled_objects
            + self.late_resident_objects
        )

    def detail(self, *, snapshot: str, key: str, components: int) -> str:
        """The ``k=v`` detail line, in the grammar the th#1839 route serves
        verbatim and the pgw#1232 legs already parse (``(\\w+)=(\\S+)``).

        Values never contain spaces, so a snapshot digest or key that somehow
        does would break every reader downstream — they are emitted through
        ``_token`` for that reason.
        """
        return (
            f"snapshot={_token(snapshot)} key={_token(key)} "
            f"components={components} "
            f"requested_objects={self.requested_objects} "
            f"tree_bytes={self.tree_bytes} "
            f"fetched_objects={self.fetched_objects} "
            f"fetched_bytes={self.fetched_bytes} "
            f"resident_objects={self.resident_objects} "
            f"resident_bytes={self.resident_bytes} "
            f"filled_objects={self.filled_objects} "
            f"filled_bytes={self.filled_bytes} "
            f"late_resident_objects={self.late_resident_objects} "
            f"accounted_objects={self.accounted()}"
        )


def _token(value: str) -> str:
    """A `k=v` value with no spaces in it, and never empty.

    An empty value ends the token at the `=` and silently merges the next pair
    into it; a value with a space splits one pair into two. Both produce a
    detail line that parses without error and means something else.
    """
    cleaned = "".join(ch for ch in str(value or "") if not ch.isspace())
    return cleaned or "-"


def emit_pull(
    stats: SnapshotPullStats,
    *,
    snapshot: str,
    key: str,
    components: int = 0,
    duration_ms: int = 0,
) -> None:
    """One completed event stating what this pull moved.

    An EVENT and not an activity, deliberately: it is a self-contained roll-up
    of work that has already finished, with no progress counter and no running
    state. The pull's LIVE progress is already reported through the caller's
    ``progress`` callback and `boot_phases`; admitting a second running
    activity for the same span would put "a message arrived recently" into the
    hub's serving-blocked predicates.

    Never raises — telemetry never fails a boot that succeeded.
    """
    try:
        from . import activity as activity_mod

        activity_mod.emit_event(
            activity_mod.KIND_SNAPSHOT_PULL,
            stats.detail(snapshot=snapshot, key=key, components=components),
            phase=PHASE_PULLED,
            duration_ms=duration_ms,
        )
    except Exception:  # pragma: no cover — telemetry never fails a boot
        logger.debug("snapshot pull event failed", exc_info=True)
