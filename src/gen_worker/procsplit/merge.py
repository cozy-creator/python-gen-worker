"""pgw#783 fan-in: G children produce G views; the hub sees ONE worker.

Pure functions over the wire protobufs — no I/O, no torch, no state. Every
rule here has a plausible wrong answer that would be silently harmful on a wide
pod, so each carries the reason it is not that answer.

**At G == 1 every function returns its single input UNCHANGED** (the same
object, so the serialized bytes are trivially identical). The N-child path is
never a different code path for a one-child worker — it is the same code with a
list of length one.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from ..pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

__all__ = [
    "merge_state_deltas",
    "merge_residency",
    "merge_hello",
    "merge_phase",
]

# Readiness order for WorkerPhase. ERROR is not "most ready" despite being the
# largest enum value, so the proto's numeric order cannot be used directly.
_PHASE_RANK = {
    pb.WORKER_PHASE_ERROR: -1,
    pb.WORKER_PHASE_UNSPECIFIED: 0,
    pb.WORKER_PHASE_BOOTING: 1,
    pb.WORKER_PHASE_DOWNLOADING_MODELS: 2,
    pb.WORKER_PHASE_LOADING_PIPELINES: 3,
    pb.WORKER_PHASE_WARMING: 4,
    pb.WORKER_PHASE_READY: 5,
}


def merge_phase(phases: Sequence[int]) -> int:
    """The worker's phase is its LEAST ready group — and any group in ERROR
    makes the worker ERROR, because a wide worker that hides one broken group
    behind three healthy ones is exactly the pod nobody can debug."""
    if not phases:
        return pb.WORKER_PHASE_UNSPECIFIED
    if len(phases) == 1:
        return phases[0]
    return min(phases, key=lambda p: _PHASE_RANK.get(p, 0))


def merge_state_deltas(deltas: Sequence[pb.StateDelta]) -> pb.StateDelta:
    """One worker-level StateDelta from G per-group ones."""
    if not deltas:
        return pb.StateDelta()
    if len(deltas) == 1:
        return deltas[0]

    # A function is dispatchable only where EVERY group can serve it: the hub
    # picks the slot, so advertising a union invites dispatch into a group that
    # cannot serve. The cost — one slow group holds the worker back — is the
    # honest one; hiding it is not.
    available = set(deltas[0].available_functions)
    for d in deltas[1:]:
        available &= set(d.available_functions)

    loading = set()
    for d in deltas:
        loading |= set(d.loading_functions)
    loading -= available

    targets: List[pb.CompileTarget] = []
    seen_targets = set()
    lookups: List[pb.CellLookup] = []
    seen_lookups = set()
    for d in deltas:
        for t in d.compile_targets:
            # incarnation_id is a uuid4 minted per live object, so children
            # cannot collide; the dedup is belt for a replayed snapshot.
            if t.incarnation_id in seen_targets:
                continue
            seen_targets.add(t.incarnation_id)
            targets.append(t)
        for cl in d.cell_lookups:
            key = (cl.family, cl.cell_key)
            if key in seen_lookups:
                continue
            seen_lookups.add(key)
            lookups.append(cl)

    merged = pb.StateDelta(
        phase=merge_phase([d.phase for d in deltas]),
        available_functions=sorted(available),
        loading_functions=sorted(loading),
        # Each child measures only the cards CUDA_VISIBLE_DEVICES let it see,
        # so the pod's free VRAM is the sum. (Under one process this was one
        # measurement over all cards; the total is the same number.)
        free_vram_bytes=sum(d.free_vram_bytes for d in deltas),
        finalizing_jobs=sum(d.finalizing_jobs for d in deltas),
        # The WORKER has observed generation N only when every group has.
        # A max here would tell the hub a config/residency edit had landed
        # while a group was still running the previous one.
        observed_residency_generation=min(
            d.observed_residency_generation for d in deltas
        ),
        observed_config_generation=min(d.observed_config_generation for d in deltas),
        compile_targets=targets,
        cell_lookups=lookups,
    )
    # THE TRAP: disk is NOT summable. All G children share ONE container
    # filesystem, so summing their statvfs reports would tell the hub the pod
    # has G times the disk it has — and every residency budget on a wide pod
    # would be computed against a number that does not exist. One child's
    # report IS the pod's report.
    for d in deltas:
        if d.HasField("disk_usage"):
            merged.disk_usage.CopyFrom(d.disk_usage)
            break
    return merged


def merge_residency(
    snapshots: Sequence[Sequence[pb.ModelResidency]],
) -> List[pb.ModelResidency]:
    """The pod's residency baseline from G per-group snapshots.

    Same discipline as ``available_functions``: a ref counts as resident only
    where EVERY group holds it, because the hub may dispatch to any group. Tier
    is the weakest group's (a ref in VRAM on one card and on disk on another is
    a disk-tier ref for dispatch purposes); ``vram_bytes`` is SUMMED because it
    is a measured pod footprint, not a per-group claim.
    """
    if not snapshots:
        return []
    if len(snapshots) == 1:
        return list(snapshots[0])

    by_ref: List[Dict[str, pb.ModelResidency]] = [
        {m.ref: m for m in snap} for snap in snapshots
    ]
    common = set(by_ref[0])
    for d in by_ref[1:]:
        common &= set(d)

    out: List[pb.ModelResidency] = []
    for ref in sorted(common):
        records = [d[ref] for d in by_ref]
        weakest = min(records, key=lambda m: int(m.tier))
        digests = {m.snapshot_digest for m in records if m.snapshot_digest}
        if len(digests) > 1:
            # Two groups materialized DIFFERENT immutable bytes for one ref.
            # That is a real divergence (a mid-flight snapshot change caught
            # groups at different generations); report the weakest and say so.
            logger.warning(
                "residency merge: ref %s has %d distinct snapshot digests "
                "across groups (%s) — reporting the weakest tier's",
                ref, len(digests), sorted(digests),
            )
        out.append(pb.ModelResidency(
            ref=ref,
            tier=weakest.tier,
            vram_bytes=sum(m.vram_bytes for m in records),
            snapshot_digest=weakest.snapshot_digest,
            residency_generation=min(m.residency_generation for m in records),
        ))
    return out


def merge_hello(
    hellos: Sequence[pb.Hello],
    *,
    worker_session_id: Optional[str] = None,
    extra_in_flight: Sequence[Tuple[str, int]] = (),
) -> pb.Hello:
    """One Hello from G children's Hellos.

    ``worker_session_id`` OVERRIDES whatever the children minted. It is
    ``uuid.uuid4().hex`` in ``intent_registry.__init__`` today, i.e. minted by
    the child, so it changes on every child respawn — and the hub rejects
    cross-session shadow state. At G == 1 stage 1 got away with it because a
    respawn also cycles the stream; with G children one group's respawn must
    not invalidate the whole worker's shadow state. The parent is the process
    with the session, so the parent mints it.

    ``extra_in_flight`` is the parent's own durable pending-result keys, merged
    exactly as stage 1 does.
    """
    if not hellos:
        raise ValueError("merge_hello needs at least one Hello")

    if len(hellos) == 1 and not worker_session_id and not extra_in_flight:
        return hellos[0]

    # Group 0 is the template: protocol version, identity, and (until pgw#763
    # delta 2 moves hardware measurement to the parent) `resources`. Delta 2
    # replaces exactly this line and nothing else.
    merged = pb.Hello()
    merged.CopyFrom(hellos[0])

    if len(hellos) > 1:
        merged.state.CopyFrom(merge_state_deltas([h.state for h in hellos]))
        del merged.models[:]
        merged.models.extend(merge_residency([list(h.models) for h in hellos]))
        # The promised cadence must be one every group can keep.
        promised = [h.heartbeat_interval_ms for h in hellos if h.heartbeat_interval_ms]
        merged.heartbeat_interval_ms = min(promised) if promised else 0

    seen = set()
    del merged.in_flight[:]
    for h in hellos:
        for j in h.in_flight:
            key = (j.request_id, j.attempt)
            if key in seen:
                continue
            seen.add(key)
            merged.in_flight.add(request_id=j.request_id, attempt=j.attempt)
    for rid, att in extra_in_flight:
        if (rid, att) in seen:
            continue
        seen.add((rid, att))
        merged.in_flight.add(request_id=rid, attempt=att)

    if worker_session_id:
        merged.worker_session_id = worker_session_id
    return merged
