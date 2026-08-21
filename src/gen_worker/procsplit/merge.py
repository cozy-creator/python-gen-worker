"""Fan-in: G children produce G views; the hub sees ONE worker."""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

__all__ = [
    "merge_state_deltas",
    "merge_residency",
    "merge_hello",
    "merge_phase",
    "reconcile_activity_kind",
    "worker_fn_unavailable",
    "worker_fn_degraded",
]

_PHASE_RANK = {
    pb.WORKER_PHASE_ERROR: -1,
    pb.WORKER_PHASE_UNSPECIFIED: 0,
    pb.WORKER_PHASE_BOOTING: 1,
    pb.WORKER_PHASE_DOWNLOADING_MODELS: 2,
    pb.WORKER_PHASE_LOADING_PIPELINES: 3,
    pb.WORKER_PHASE_WARMING: 4,
    pb.WORKER_PHASE_READY: 5,
}


def merge_phase(phases: Sequence["pb.WorkerPhase"]) -> "pb.WorkerPhase":
    """The worker's phase is its LEAST ready group — and any group in ERROR makes the worker ERROR, because a wide worker that hides one broken group behind three healthy ones is exactly the pod nobody ca..."""
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

    available = set()
    for d in deltas:
        available |= set(d.available_functions)

    loading = set()
    for d in deltas:
        loading |= set(d.loading_functions)
    loading -= available

    targets: List[pb.CompileTarget] = []
    seen_targets = set()
    for d in deltas:
        for t in d.compile_targets:
            if t.incarnation_id in seen_targets:
                continue
            seen_targets.add(t.incarnation_id)
            targets.append(t)

    merged = pb.StateDelta(
        phase=merge_phase([d.phase for d in deltas]),
        available_functions=sorted(available),
        loading_functions=sorted(loading),
        free_vram_bytes=sum(d.free_vram_bytes for d in deltas),
        finalizing_jobs=sum(d.finalizing_jobs for d in deltas),
        observed_residency_generation=min(
            d.observed_residency_generation for d in deltas
        ),
        observed_config_generation=min(d.observed_config_generation for d in deltas),
        compile_targets=targets,
    )
    for d in deltas:
        if d.HasField("disk_usage"):
            merged.disk_usage.CopyFrom(d.disk_usage)
            break
    return merged


def merge_residency(
    snapshots: Sequence[Sequence[pb.ModelResidency]],
) -> List[pb.ModelResidency]:
    """The worker's residency baseline from G per-group snapshots."""
    if not snapshots:
        return []
    if len(snapshots) == 1:
        return list(snapshots[0])

    by_ref: List[Dict[str, pb.ModelResidency]] = [
        {m.ref: m for m in snap} for snap in snapshots
    ]
    all_refs = set()
    for d in by_ref:
        all_refs |= set(d)

    out: List[pb.ModelResidency] = []
    for ref in sorted(all_refs):
        records = [d[ref] for d in by_ref if ref in d]
        strongest = max(records, key=lambda m: int(m.tier))
        digests = {m.snapshot_digest for m in records if m.snapshot_digest}
        if len(digests) > 1:
            logger.warning(
                "residency merge: ref %s has %d distinct snapshot digests "
                "across groups (%s) — reporting the strongest tier's",
                ref, len(digests), sorted(digests),
            )
        out.append(pb.ModelResidency(
            ref=ref,
            tier=strongest.tier,
            vram_bytes=sum(m.vram_bytes for m in records),
            snapshot_digest=strongest.snapshot_digest,
            residency_generation=max(m.residency_generation for m in records),
        ))
    return out


_ACTIVITY_TERMINAL = (pb.ACTIVITY_STATE_COMPLETED, pb.ACTIVITY_STATE_FAILED)


def reconcile_activity_kind(
    per_group: Mapping[int, pb.ActivityUpdate],
    *,
    seq: int,
) -> pb.ActivityUpdate:
    """Collapse ONE activity ``kind`` across the groups reporting it into ONE worker-level ActivityUpdate — the parent-side aggregation Paul's ruling 2 requires (the hub folds ActivityUpdate into ``info.A..."""
    updates = [per_group[g] for g in sorted(per_group)]
    if not updates:
        raise ValueError("reconcile_activity_kind needs at least one update")

    running = [u for u in updates if u.state == pb.ACTIVITY_STATE_RUNNING]
    if running:
        state = pb.ACTIVITY_STATE_RUNNING
        live = running
    elif any(u.state == pb.ACTIVITY_STATE_FAILED for u in updates):
        state = pb.ACTIVITY_STATE_FAILED
        live = updates
    else:
        state = pb.ACTIVITY_STATE_COMPLETED
        live = updates

    lead = max(live, key=lambda u: (u.step, u.counter_done, u.updated_at_unix_ms))
    merged = pb.ActivityUpdate()
    merged.CopyFrom(lead)
    merged.state = state
    merged.seq = int(seq)
    merged.step = max(u.step for u in live)
    merged.total_steps = max(u.total_steps for u in live)
    if any(u.counter for u in live):
        merged.counter_done = sum(u.counter_done for u in live)
        merged.counter_total = sum(u.counter_total for u in live)
        merged.rate_per_s = sum(u.rate_per_s for u in live)
    merged.self_stalled = bool(live) and all(u.self_stalled for u in live)
    merged.stalled_for_ms = (
        min(u.stalled_for_ms for u in live) if merged.self_stalled else 0
    )
    return merged


def worker_fn_unavailable(
    per_group: Mapping[int, Optional[pb.FnUnavailable]],
) -> Optional[pb.FnUnavailable]:
    """The worker-level truth for one function's UNAVAILABILITY."""
    if not per_group:
        return None
    reports = list(per_group.values())
    if any(r is None for r in reports):
        return None
    ranked = sorted(
        (r for r in reports if r is not None),
        key=lambda r: 0 if r.reason != "setup_failed" else 1,
    )
    return ranked[0] if ranked else None


def worker_fn_degraded(
    per_group: Mapping[int, Optional[pb.FnDegraded]],
    *,
    served_native_somewhere: bool,
) -> Optional[pb.FnDegraded]:
    """The worker-level truth for one function's DEGRADATION."""
    if served_native_somewhere:
        return None
    reports = [r for r in per_group.values() if r is not None]
    if not reports:
        return None
    return min(reports, key=lambda r: r.est_latency_multiplier or float("inf"))


def merge_hello(
    hellos: Sequence[pb.Hello],
    *,
    worker_session_id: Optional[str] = None,
    extra_in_flight: Sequence[Tuple[str, int]] = (),
) -> pb.Hello:
    """One Hello from G children's Hellos."""
    if not hellos:
        raise ValueError("merge_hello needs at least one Hello")

    if len(hellos) == 1 and not worker_session_id and not extra_in_flight:
        return hellos[0]

    merged = pb.Hello()
    merged.CopyFrom(hellos[0])

    if len(hellos) > 1:
        merged.state.CopyFrom(merge_state_deltas([h.state for h in hellos]))
        del merged.models[:]
        merged.models.extend(merge_residency([list(h.models) for h in hellos]))
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
