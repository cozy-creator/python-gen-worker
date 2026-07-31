"""pgw#783 fan-in: G children produce G views; the hub sees ONE worker.

Pure functions over the wire protobufs — no I/O, no torch, no state. Every
rule here has a plausible wrong answer that would be silently harmful on a wide
pod, so each carries the reason it is not that answer.

**The orchestrator sees ONE worker, never N sub-units** (Paul, 2026-07-30):
*"from the orchestrator's point of view, they should be thinking of a worker as
one unit they can assign work to. The worker worries about the sub-execution
units, but from the orchestrator's point of view, they are one worker. This
keeps the system boundaries intact."* So the hub must not learn that groups
exist at all — no group-keyed wire field, ever. The parent is a genuine
AGGREGATOR, not a relay: it reconciles G children into one coherent worker view
(one function set, one activity stream, one liveness claim, one capacity
picture) BEFORE anything reaches the stream.

**A function is advertised while ANY group can serve it** (Paul, 2026-07-30):
*"The worker should advertise any function it can serve. If it has 4 execution
groups, and only one of them can serve function-X, then it should advertise
function-X as serveable."* So availability UNIONS. The hub dispatches to the
worker; the worker routes to a group that can serve (procsplit.group.route),
and a dispatch landing on a group that cannot serve is the worker's problem to
re-home internally, not the hub's to avoid. Whether the scheduler should ALSO
know staffing depth (1 group vs 4 can serve X) is deliberately left open and
filed separately (th#, "does the scheduler benefit from knowing how many groups
can serve X") — NOT built speculatively here.

**At G == 1 every function returns its single input UNCHANGED** (the same
object, so the serialized bytes are trivially identical). The N-child path is
never a different code path for a one-child worker — it is the same code with a
list of length one.
"""

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


def merge_phase(phases: Sequence["pb.WorkerPhase"]) -> "pb.WorkerPhase":
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

    # UNION (Paul's ruling): the worker advertises any function ANY group can
    # serve. The hub assigns to the worker as one unit; the worker routes to a
    # group that can serve it. A function still loading in one group but ready
    # in another is READY for the worker — so union-then-subtract, never the
    # reverse.
    available = set()
    for d in deltas:
        available |= set(d.available_functions)

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
    """The worker's residency baseline from G per-group snapshots.

    Same discipline as ``available_functions``, and for the same reason: a ref
    is resident on the WORKER while ANY group holds it (UNION), because the hub
    assigns to the worker and the worker routes a dispatch to a group that has
    the ref. Tier is the STRONGEST any group has it at — the best the worker can
    do for that ref — and ``vram_bytes`` is SUMMED across the groups holding it
    (a measured pod footprint, not a per-group claim).
    """
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
            # Two groups materialized DIFFERENT immutable bytes for one ref —
            # a real divergence (a mid-flight snapshot change caught groups at
            # different generations). Report the strongest tier's identity and
            # say so.
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
            # The freshest generation that materialized this ref anywhere: the
            # ref IS resident, as recently as any group made it so.
            residency_generation=max(m.residency_generation for m in records),
        ))
    return out


# A ref/kind whose freshest state across groups is one of these is over — it
# should not keep the worker-level activity or residency claim alive.
_ACTIVITY_TERMINAL = (pb.ACTIVITY_STATE_COMPLETED, pb.ACTIVITY_STATE_FAILED)


def reconcile_activity_kind(
    per_group: Mapping[int, pb.ActivityUpdate],
    *,
    seq: int,
) -> pb.ActivityUpdate:
    """Collapse ONE activity ``kind`` across the groups reporting it into ONE
    worker-level ActivityUpdate — the parent-side aggregation Paul's ruling 2
    requires (the hub folds ActivityUpdate into ``info.Activities[kind]``, so G
    children producing the same kind would overwrite each other and a group's
    mint would look like the whole worker's).

    ``per_group`` is the latest update of this kind, keyed by group ordinal, for
    every group that currently has it open. The worker's activity of this kind:

    * is RUNNING while ANY group runs it (the worker IS minting if any group
      is); it is terminal only when EVERY group's latest is terminal, and FAILED
      then only if any of those terminals failed;
    * carries the AGGREGATE progress — summed counters, the furthest step — so
      the hub's counter-advancement liveness (gw#621) sees the union of the
      groups' work advancing, never one group's stall masking three groups'
      progress;
    * is re-stamped with the PARENT's monotonic ``seq``. Children have
      independent seq counters that would collide and regress across groups; the
      hub records last-progress on every seq increase, so the seq the hub sees
      must be the worker's own, minted here.

    Never leaks a group ordinal: the output is a plain worker-level update.
    """
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

    # The representative event we elaborate from: the furthest-along live one.
    lead = max(live, key=lambda u: (u.step, u.counter_done, u.updated_at_unix_ms))
    merged = pb.ActivityUpdate()
    merged.CopyFrom(lead)
    merged.state = state
    merged.seq = int(seq)
    # Aggregate progress across the live groups, so the counter the hub judges
    # liveness by is the WORKER's advancement, not one group's.
    merged.step = max(u.step for u in live)
    merged.total_steps = max(u.total_steps for u in live)
    if any(u.counter for u in live):
        merged.counter_done = sum(u.counter_done for u in live)
        merged.counter_total = sum(u.counter_total for u in live)
        merged.rate_per_s = sum(u.rate_per_s for u in live)
    # self_stalled is a worker confession the hub recycles the pod on: the
    # worker is only stalled when EVERY live group is (one group advancing is
    # the worker advancing).
    merged.self_stalled = bool(live) and all(u.self_stalled for u in live)
    merged.stalled_for_ms = (
        min(u.stalled_for_ms for u in live) if merged.self_stalled else 0
    )
    return merged


def worker_fn_unavailable(
    per_group: Mapping[int, Optional[pb.FnUnavailable]],
) -> Optional[pb.FnUnavailable]:
    """The worker-level truth for one function's UNAVAILABILITY.

    Ruling 2 + ruling 1 together: the worker does not serve a function only when
    NO group serves it. ``per_group[g]`` is that group's FnUnavailable for this
    function, or ``None`` if the group serves it. If ANY group serves it (a
    ``None`` present), the worker serves it — return ``None``, emit nothing, and
    do NOT retire it worker-wide. Only when every group reports it unavailable
    does the worker report unavailable, carrying one representative reason (the
    admin availability view is per-worker, not per-group).
    """
    if not per_group:
        return None
    reports = list(per_group.values())
    if any(r is None for r in reports):
        return None  # at least one group serves it -> the worker serves it
    # Every group is unavailable. Prefer a hardware-gating reason over a
    # transient setup failure for the admin view; otherwise the first.
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
    """The worker-level truth for one function's DEGRADATION.

    FnDegraded is placement guidance — "this release runs degraded on this card,
    it wants a bigger one". Under ruling 1 the worker routes a dispatch to a
    group that can serve, so if ANY group serves the function NATIVE the worker
    is not degraded: report nothing. The worker is degraded only when it serves
    the function AND the best any group can do is degraded — then report the
    LEAST degraded of the group reports (the worker's true best), so the hub's
    placement hint reflects the best card the worker actually has for it.
    """
    if served_native_somewhere:
        return None
    reports = [r for r in per_group.values() if r is not None]
    if not reports:
        return None
    # Least degraded = smallest realistic slowdown: the worker's real best.
    return min(reports, key=lambda r: r.est_latency_multiplier or float("inf"))


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
