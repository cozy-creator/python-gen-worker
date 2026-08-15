"""Job execution: intake, GPU semaphore, deadline + cancellation watchdog,
sync-on-thread / async-on-loop, JobProgress deltas, result send, and the
worker-side model seam (ensure-local, setup injection, declarative residency,
and compile-cache adoption).

One dispatch path for every endpoint kind. Everything runs on the single
asyncio loop; sync tenant code runs in threads via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import itertools
import logging
import os
import tempfile
import threading
import time
import typing
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple


import msgspec

from . import activity as activity_mod
from . import adopt_fit
from . import aot_declaration, aot_identity
from . import boot_adopt
from . import boot_phases as boot_mod
from . import cell_adopt
from . import dispatch
from . import handler_proof
from .procsplit import broker as procsplit_broker
from . import cpu_budget
from . import measured_posture as posture_mod
from . import mint_workers
from . import settings_authority
from . import progress as progress_mod
from . import serve_posture
from . import serving_mode as serving_mode_mod
from . import warmup
from . import worker_credential
from . import worker_goals
from .api.binding import (
    ModelRef,
    wire_ref,
)
from .hubio.client import HubPublishError
from .hub_error import HubApiError
from . import graph_facts
from .child_contract import MintSlot, slot_subjects
from .wire_snapshots import index_snapshots
from .api.errors import (
    ArtifactTransferError,
    CanceledError,
    EndpointSetupFailed,
    GpuSlotUnreachable,
    IllegalCombination,
    ModelSlotIdentityError,
    RetryableError,
    ValidationError,
    WorkerError,
)
from .api.streaming import (
    BatchItemDelta,
    Done,
    Error,
    IncrementalTokenDelta,
    StreamAccumulator,
    StreamResult,
    TokenUsage,
)
from .api.types import Asset
from .capability import (
    HOST_RAM_REFUSALS,
    HardwareUnmetError,
    HostRamCapacityError,
    InsufficientDiskError,
    InsufficientHostRamError,
)
from .input_assets import (
    InputManifestEntry,
    cleanup_input_assets,
    manifest_from_run_job,
    materialize_input_assets,
)
from .lifecycle_intents import IntentRegistry
from .models import disk_gc
from .models import provision
from .models.refs import WireRef, normalize_model_ref
from .models import residency as residency_mod
from .models.memory import (
    aflush_memory,
    cuda_allocated_bytes,
    estimate_cuda_resident_gb,
    estimate_pipeline_size_gb,
    flush_memory,
    get_available_vram_gb,
    is_cuda_oom,
    low_vram_mode,
    release_unused_pinned_host_cache,
)
from .models import rung as rungspec
from .models.rung import touches_host_ram, transition_line
from .models.records import (
    RecordTeardown,
    record_in_use,
    record_refs,
    records_holding,
    vacate_record,
)
from .models.errors import MissingSnapshotError, UrlExpiredError
from .models.execution_lanes import ExecutionLaneUnavailableError
from .topology import (
    ExecutionTopology,
    TopologyError,
    current_device_group,
    device_group_scope,
    pin_cuda_device_for_group,
)
from .pb import worker_scheduler_pb2 as pb
from .redact import sanitize as _sanitize
from .models.store import ModelStore, _ResidencyIdentity
from .registry import EndpointSpec
from .runtime_config import ConfigStore, extract_job_config
from .stage_timing import stage_ms_for_metrics

#: pgw#848 item 1: cadence for the publish-durability wait. A POLL interval,
#: not a deadline — nothing here decides when a publish has taken too long;
#: the activity's own staleness window does, and only durable movement feeds it.
_PUBLISH_SETTLE_POLL_S = 2.0

if typing.TYPE_CHECKING:
    from . import compile_cache
    from . import fleet_cells
    from .models.serve_fit import ServePlan
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)
from .request_context._helpers import _decode_unverified_jwt_claims
from .utils import lora as lora_util
import errno as _errno
import inspect as _inspect
import struct as _struct
from .models.refs import parse_model_ref
from . import compile_cache
from .models.loading import (
    is_modular_pipeline_class,
    plan_streamed_hydration,
)
from .compile_cache import CompiledExecutionLaneUnavailableError
from .preload import Preloader
from .api.binding import rebind_pick
from . import hostfacts
from .models.hub_policy import TensorhubWorkerCapabilities
from .models.serve_fit import (RUN_FP8_STORAGE,
                                   RUN_OFFLOAD, plan_serve)
from . import postmortem
from .models.serve_fit import replan
from .models.loading import pipeline_weight_lane
from .models import attention_modes as attnspec
from .models import execution_lanes as lanespec
from . import warmup as warmup_mod
from .api.decorators import ATTR as _DECL_ATTR
from . import compile_cache as _cc_execution_lane
from .parallel import ContextParallelUnavailable
from .parallel import BootPlan, GroupPlan
from .parallel.cp import w8a8_gemm_mode
from .parallel.runtime import SequenceRuntime, arm_sequence_gate
from .runtimes.server import RUNTIME_FACTORIES
from .models.loading import composition_compute_dtype
from .runtimes.server import ServerHandle
from .models.lane_residency_gate import LaneResidencyGate, arm_lane_residency_gate
from .models.memory import rearm_offload
from . import fleet_cells
from . import aot_serve, numerics_ladder, shape_growth
from . import fleet_cells as fleet_cells_mod
from . import hot_swap
from . import mint_supervisor
from .hostfacts import cuda_ready

_CONTEXT_BY_KIND: Dict[str, type] = {
    "inference": RequestContext,
    "conversion": ConversionContext,
    "dataset": DatasetContext,
    "training": TrainingContext,
    # th#1255: an eval materializes the same reserved refs a conversion does
    # (its `source` IS the reference arm) and writes request output assets.
    # It just publishes nothing, and there is no separate surface for that —
    # publish authority is the hub's call, and the hub refuses repo writes
    # for kind=eval. A distinct EvalContext would only restate that refusal
    # somewhere it cannot be enforced.
    "eval": ConversionContext,
}

logger = logging.getLogger(__name__)

INLINE_RESULT_MAX_BYTES = 64 * 1024


async def _to_thread_complete(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Join after cancellation: ``to_thread`` itself cannot be cancelled.

    Diffusers/Accelerate mutate process-global meta-device hooks while loading,
    so a surrounding model-load lock must outlive the worker thread.
    """
    # pgw#748: this is THE loop->thread hop of the model-load path (setup,
    # slot injection, warmup). `torch.cuda.set_device` is thread-local, so a
    # pool thread points at card 0 no matter which group's job scheduled it —
    # and every `.to("cuda")` in the loader follows the CURRENT device. The
    # group rides the contextvar into the thread; this makes the thread's
    # device follow it. No-op for group 0, i.e. for every pod today.
    def _pinned(*a: Any, **kw: Any) -> Any:
        pin_cuda_device_for_group()
        return func(*a, **kw)

    work = asyncio.create_task(asyncio.to_thread(_pinned, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        try:
            await work
        except BaseException:
            pass
        # gw#624: the joined thread's result (possibly a fully-loaded
        # multi-GB pipeline) lives on the Task. This frame rides the
        # propagating CancelledError's traceback through rollback — keeping
        # ``work`` referenced here would pin the whole discarded load in
        # memory across the retry.
        del work
        raise


# ctx.progress/log/checkpoint events ride the JobProgress stream; the hub fans
# them to /v1/requests/:id/events SSE as output.delta envelopes whose
# payload.delta carries this JSON verbatim (th#640).
EVENT_CONTENT_TYPE = "application/x-request-event+json"
# How often `_execute` re-consults `progress.self_diagnosis()` while a handler
# runs. A poll cadence, not a limit: it bounds how quickly a confession is
# NOTICED, never the work itself (§4.24 — the stall windows live per-phase in
# `progress.STALL_WINDOW_S`).
_STALL_POLL_S = 5.0
_STUCK_THREAD_RECYCLE_S = 30.0
# How often the reaper re-asks whether the abandoned handler thread has ended.
_STUCK_THREAD_POLL_S = 0.5
# th#1779: how often the request's evidence sampler re-reads process CPU+I/O.
# Same cadence as the stall poll, so the freshest sample is never older than
# one poll when the diagnosis is taken.
_HANDLER_EVIDENCE_INTERVAL_S = _STALL_POLL_S
# pgw#687: a cancel that never unwinds. Cancellation of a SYNC handler is
# cooperative — the thread cannot be killed — so a handler that never polls
# ctx.cancelled (observed: a modelopt calibration loop) keeps the GPU permit
# and its instance gate forever. The next assignment is then accepted and
# parks pre-execution, emitting NOTHING: 46 minutes of silent absorption with
# every hub-side signal reading healthy.
#
# The bound is on CANCEL -> TERMINAL latency, never on handler progress:
# after a cancel there is no legitimate work left to protect, so this does
# not re-introduce the wall-clock bound gw#666 / th#1157 / th#1160 forbid (a
# 51-minute silent source download is untouched — it is not a cancelled job).
_CANCEL_UNWIND_REASON = "cancel_unwind_stuck"
#: Cancel -> terminal result. Past it the executor is presumed unable to
#: return to idle: stop advertising and refuse work (REVERSIBLE).
_CANCEL_UNWIND_GRACE_S = 45.0
#: Further wait once quarantined. Past it the process is recycled so the pod
#: is replaced — a wedged thread cannot be reclaimed any other way.
_CANCEL_UNWIND_RECYCLE_S = 300.0
# pgw#738: how often a blocked #382 re-acquire OBSERVES the permit ledger.
# An observation cadence, never a bound: nothing gives up because this much
# time elapsed. The refusal is the ledger predicate (an outstanding permit
# with no live holder), confirmed across two probes that span no ledger
# transition, so a permit handed off between them can never read as leaked.
_PERMIT_PROBE_S = 0.1
_GiB = 1024 ** 3
# gw#587: a store-served boot (a compile cell was ATTACHED, not self-minted)
# must pay ~0 inductor compile wall time — the whole point of a delivered
# cell is that the graph is already compiled. This is seconds, not ms: a
# trivial guard recompile is noise, a real cold compile burns the economic
# claim the cell system exists to avoid (see gw#587's boot-to-ready value
# proposition). Fixed floor rather than a learned per-fleet baseline —
# simple, robust, and every real cold compile clears it by a wide margin.
_STORE_SERVED_COMPILE_ALARM_S = 30.0

try:  # torch is optional at import time; the executor works without it.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# Credential material inside exception messages (auth headers, presigned-URL
# query params) and worker-filesystem paths (pgw#514/P8: a FileNotFoundError
# ships "ExcClass: first-line" to the client — absolute paths leak pod
# internals). Redacted in place — replacing the whole message with
# "internal error" made every download/publish failure undiagnosable from the
# hub (pods ship no logs; presigned URLs carry X-Amz-* params).


def _unwrap_optional(ann: Any) -> Any:
    """``X | None`` -> ``X``. An OPTIONAL slot is annotated ``Pipe | None =
    None`` (that default is what declares it optional), but injection is
    typed off the annotation — so the loader must see ``Pipe``, not the
    union. Non-optional and non-union annotations pass through unchanged;
    a union of 2+ real types is left alone (no basis to pick one)."""
    args = [a for a in typing.get_args(ann) if a is not type(None)]
    if len(args) == 1 and type(None) in typing.get_args(ann):
        return args[0]
    return ann




def _snapshot_digest(snapshots: Any, ref: str) -> str:
    """The resolved snapshot's own digest for ``ref``, or ``""``.

    pgw#1117 names it in the envelope refusal: "the binding resolved to THIS
    artifact" is the fact an operator needs, and a ref alone does not carry it
    — the whole ie#642 shape is a mutable bare tag head pointing somewhere
    new."""
    snap = (snapshots or {}).get(ref) if snapshots else None
    return str(getattr(snap, "snapshot_digest", "") or "")


def _reserved_repo_info(payload: Any, field_name: str) -> Dict[str, Any]:
    """``payload.source`` / ``payload.destination`` / ``payload.text_encoder``
    / ``payload.candidate`` / ``payload.resume_from`` as a plain dict ({} when
    absent). Producer payloads carry these reserved-name structs (#376,
    pgw#594, pgw#684, pgw#1242). The set of names is hardcoded here; pgw#690
    tracks making it declarative."""
    obj = getattr(payload, field_name, None)
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        out = msgspec.to_builtins(obj)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def _producer_destination_repo(payload: Any, destination_info: Dict[str, Any]) -> str:
    """Bare ``owner/repo`` the producer publishes into, or "".

    The reserved struct (``payload.destination.ref``) wins; the flat
    ``payload.destination_repo`` scalar is the wire form gen-orchestrator
    dispatches. Tag/flavor/checkpoint selectors are stripped.
    """
    ref = str(destination_info.get("ref") or destination_info.get("repo") or "").strip()
    if not ref:
        ref = str(getattr(payload, "destination_repo", "") or "").strip()
    for sep in (":", "@", "#"):
        ref = ref.split(sep, 1)[0]
    return ref.strip().strip("/")


def _capability_job_id(token: str) -> Optional[str]:
    """job_id claim from the worker capability token ("" claims → None).

    Repo-CAS checkpoint sessions are job-bound: tensorhub requires the
    session's job_id to equal the cap token's job_id claim (gw#453).
    """
    raw = str(token or "").strip()
    if not raw:
        return None
    try:
        return str(_decode_unverified_jwt_claims(raw).get("job_id") or "").strip() or None
    except Exception:
        return None


def _undeclared_model_slots(spec: EndpointSpec, run: "pb.RunJob") -> List[str]:
    """``ModelBinding.slot`` names the hub dispatched that the endpoint never
    declared in ``@endpoint(models={...})`` (gw#583, the ie#518 silence).

    Not fatal — a new hub-side model param must stay forward-compatible with
    older workers — but never silent: the caller logs one warning per name.
    """
    return sorted({b.slot for b in run.models if b.slot and b.slot not in spec.models})


class _AllComponents(frozenset):
    """SDK v2 automatic-sharing sentinel: membership test always true."""

    def __contains__(self, item: object) -> bool:  # noqa: D105
        return True


_ALL_COMPONENTS = _AllComponents()


#: pgw#828: the slot-resolution chain and the root-slot rule moved to
#: ``warmup`` so the delegated mint child can reach them WITHOUT importing
#: the executor. These names stay as aliases because the dispatch path and
#: several tests call them here; there is one implementation.
_resolve_slots_kwargs = warmup.resolved_slots_kwargs
_spec_root_slot = warmup.spec_root_slot


def _hub_binding_for_wire_ref(ref: str) -> ModelRef:
    """A tensorhub-source binding for a hub-named wire ref (pgw#532).

    ``RunJob.models`` / desired-instance refs name hub-CAS repos in the canonical
    ``owner/repo[@release|@digest]`` grammar; this mints the binding the
    executor materializes them through (``ensure_local`` then follows the
    tensorhub lane: orchestrator snapshots or the th#763 missing_snapshot
    re-mint — never an upstream self-fetch). Raises ``ValueError`` when
    ``ref`` does not parse under that grammar (e.g. a raw upstream id the
    hub stamped for an unmirrored slot default)."""

    parsed = parse_model_ref(ref)
    th = parsed.tensorhub
    if th is None:  # pragma: no cover - parse_model_ref(tensorhub) guarantees it
        raise ValueError(f"{ref!r} is not a tensorhub ref")
    return ModelRef(
        source="tensorhub",
        path=f"{th.owner}/{th.repo}",
        release=th.release,
    )






def _exported_arm(pipeline: Any, ref: str = "") -> bool:
    """Is THIS object serving on the EXPORTED (AOTI) lane?

    pgw#1141b, and it is the whole issue: the answer decides which failure
    detector applies. The dynamo lane keeps a per-class cache-hit ledger with
    teeth (§4.31 — a dynamo arm that misses RECOMPILES silently, so the ledger
    is its only detector); the exported lane has none, because an AOTI cell
    that cannot serve RAISES and the wrapper answers eager in-request. Score an
    exported cell on the dynamo ledger and it is disproven by construction: an
    artifact performs no FX lookup, so its hit count is permanently zero.

    That is exactly what happened on a real pod. The question used to be asked
    of the ref STRING through ``aot_serve.is_aot_ref``, which consults keys
    this process was TOLD about — and the ordered/boot-adopt arm route told it
    nothing. The object is asked first now: a wrapped cell is a fact about the
    object, not about who announced it. The ref remains a second route in
    (a cell whose wrap this frame cannot see still names itself).
    """
    return aot_serve.holds_exported_cell(pipeline) or (
        bool(ref) and aot_serve.is_aot_ref(ref))


def _alias_binding_matches(alias: "EndpointSpec", slot_key: str, ref: str) -> bool:
    """Does ``alias`` hold this load-time binding fact?"""
    binding = alias.models.get(slot_key)
    if binding is None:
        return False
    return wire_ref(binding).strip() == ref


def _map_exception(exc: BaseException) -> Tuple["pb.JobStatus", str]:
    """-> (JobStatus, safe_message)."""
    if isinstance(exc, (CanceledError, asyncio.CancelledError)):
        return pb.JOB_STATUS_CANCELED, "canceled"
    # INVALID (400, never retried) is reserved for typed validation errors and
    # msgspec payload decode failures. A BARE ValueError is NOT invalid input
    # (pgw#514/P9): PIL/numpy/tenant code raise ValueError for internal bugs,
    # and mapping those to INVALID blamed the client and suppressed retries —
    # they fall through to FATAL (class name + sanitized detail) below.
    if isinstance(exc, (ValidationError, msgspec.ValidationError, msgspec.DecodeError)):
        return pb.JOB_STATUS_INVALID, _sanitize(str(exc) or "invalid input")
    if isinstance(exc, RetryableError):
        return pb.JOB_STATUS_RETRYABLE, _sanitize(str(exc) or "retryable error")
    if isinstance(exc, ArtifactTransferError) and getattr(exc, "retryable", False):
        return pb.JOB_STATUS_RETRYABLE, _sanitize(str(exc) or "artifact transfer failed")
    if isinstance(exc, HubPublishError):
        # pgw#1002. The hub's th#1301 refusal carries its OWN `retryable` bit;
        # PROVENANCE decides the class (th#1259), not our reading of the
        # message. `True` -> RETRYABLE, so the orchestrator's MaxJobAttempts
        # budget is actually spent on a publish the hub asked us to retry.
        # `False` is a repudiation (audit findings, contract failure,
        # possession refusal) and `None` honestly means the hub named nothing
        # — neither invents a retry. The hub's `code` LEADS the detail so the
        # refusal groups by a stable token instead of by prose.
        detail = _sanitize(str(exc) or "publish failed")
        if exc.code:
            detail = f"{exc.code}: {detail}"
        status = (pb.JOB_STATUS_RETRYABLE if exc.retryable is True
                  else pb.JOB_STATUS_FATAL)
        return status, detail[:512]
    if isinstance(exc, HubApiError):
        # pgw#1229. The hub named the code AND the remedy; `str(exc)` already
        # carries both on one line, so it goes on the wire verbatim rather than
        # being re-derived into "403 Client Error: Forbidden for url: ...".
        status = pb.JOB_STATUS_RETRYABLE if exc.retryable else pb.JOB_STATUS_FATAL
        return status, _sanitize(str(exc) or "hub refused the call")[:512]
    if isinstance(exc, HardwareUnmetError):
        return pb.JOB_STATUS_RETRYABLE, _sanitize(str(exc) or "hardware unmet")
    if isinstance(exc, UrlExpiredError):
        # Hub-side URL staleness, not a client problem — retry re-mints URLs.
        return pb.JOB_STATUS_RETRYABLE, "model download url expired"
    if isinstance(exc, MissingSnapshotError):
        # A cold worker mid-resolution must never fatal a user request
        # (th#763): the missing_snapshot ModelEvent makes the hub re-mint,
        # so a retry (here or on a warmer worker) succeeds.
        return pb.JOB_STATUS_RETRYABLE, "model snapshot not resolved yet"
    if is_cuda_oom(exc):
        # Never FATAL (gw#463): a bigger/idler card can serve this. The
        # degraded-mode retry already ran by the time this maps.
        return pb.JOB_STATUS_RETRYABLE, "out of memory"
    # Unexpected exception: keep it terse but NEVER opaque — "internal error"
    # made every novel worker-side failure undiagnosable from the hub (pods
    # ship no logs). Class name + sanitized first line is safe and decisive.
    detail = _sanitize(str(exc).splitlines()[0] if str(exc) else "")
    label = type(exc).__name__
    return pb.JOB_STATUS_FATAL, f"{label}: {detail}"[:512] if detail else label


#: Setup phases in which the worker drives its OWN synthetic forwards. No
#: request payload reaches any of them — that is the entry condition, and the
#: reason a fault raised here is the RELEASE's whoever reads it. ``load`` is
#: deliberately absent: a caller-routed slot can fail to resolve there, and
#: th#1259's rule is that nothing a payload participates in producing may be
#: labelled release-owned.
_WORKER_OWNED_SETUP_PHASES = frozenset({
    activity_mod.PHASE_TRACE_GRAPH,
    activity_mod.PHASE_INDUCTOR_COMPILE,
    activity_mod.PHASE_WARMUP_FORWARD,
})


def _typed_setup_fault(
    function: str, phase: str, exc: BaseException,
) -> "Optional[EndpointSetupFailed]":
    """pgw#1118/th#1773 -> the typed release fault, or None to re-raise as-is.

    Three exclusions, each load-bearing:

    * a phase outside ``_WORKER_OWNED_SETUP_PHASES`` — a payload may
      participate there, so the origin is not ours to claim;
    * an exception that already maps to a non-FATAL status — a warm-phase OOM
      is still an OOM, and re-typing it would fatal a job a bigger card serves;
    * anything already a ``WorkerError`` — those carry their own origin claim
      (``ModelSlotIdentityError``, ... are
      exactly the labels the hub routes on), and wrapping would erase it.
    """
    if phase not in _WORKER_OWNED_SETUP_PHASES:
        return None
    if isinstance(exc, WorkerError) or not isinstance(exc, Exception):
        return None
    status, _ = _map_exception(exc)
    if status != pb.JOB_STATUS_FATAL:
        return None
    return EndpointSetupFailed(function, phase, exc)


def _runtime_term_values(
    spec: Any, payload: Any, ctx: Any = None,
) -> "Optional[Dict[str, float]]":
    """th#1051: evaluate the declared runtime formula's terms on RESOLVED
    EFFECTIVE values (pgw#654 gap #4): explicit payload value, else the
    same-named field of the catalog-resolved recipe (``ctx.defaults``).
    None = undeclared or unevaluable — the hub then falls back."""
    rf = getattr(spec, "runtime_formula", None)
    if rf is None:
        return None
    defaults = None
    if ctx is not None:
        try:
            defaults = ctx._root_slot().defaults
        except Exception:
            defaults = None
    try:
        return rf.term_values_from_struct(payload, defaults)
    except Exception:
        return None


def _scan_output_assets(output: Any) -> Tuple[float, int]:
    """One walk over the job output: (summed MEDIA seconds, count of output
    ``Asset``s). Billing sources for ``per_output_second`` (th#572) and
    ``per_output`` (pgw#512) settlement — the ONLY ones; settlement must
    never scavenge the result payload by field name."""
    total_duration = 0.0
    count = 0
    seen: set = set()
    stack = [output]
    while stack:
        item = stack.pop()
        if item is None or isinstance(item, (str, bytes, bytearray, int, float, bool)):
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, Asset):
            count += 1
            d = getattr(item, "duration_s", None)
            if isinstance(d, (int, float)) and d > 0:
                total_duration += float(d)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        elif isinstance(item, msgspec.Struct):
            stack.extend(getattr(item, f, None) for f in item.__struct_fields__)
    return total_duration, count


def _output_token_usage(output: Any) -> Optional[TokenUsage]:
    """The terminal ``TokenUsage`` signal, when the job was a token stream
    (pgw#512). Non-streaming handlers report no token usage — that's a
    tenant/runtime authoring a ``TokenUsage`` explicitly (see
    ``runtimes/llama.py``), not something inferable from an arbitrary
    output shape."""
    if isinstance(output, StreamResult):
        return output.usage
    return None
# check-then-create races.
# ---------------------------------------------------------------------------




#: Traced weight lanes that are MANDATORY once evidence names them
#: (fail-closed serving): "w8a8" (gw#534), "w4a4" (gw#540). pgw#1148: the
#: evidence is the hub-RESOLVED execution lane, never a `#flavor` token —
#: §1.32(d) deleted the token, and an assertion in a ref was never evidence.
_MANDATORY_EXECUTION_LANES = ("w8a8", "w4a4")

# pgw#671 eager-first boot: background-mint driver pacing. The abandon grace
# bounds "finish the current unit" (one eager forward) before a hard cancel.
_MINT_ABANDON_GRACE_S = 60.0
_MINT_SEED_MAX_PASSES = 8
#: Consecutive OOM-truncated seed passes before the mint aborts loudly
#: (pgw#677 reopen: never finalize a partial capture off an OOM'd plan).
_MINT_OOM_MAX_PASSES = 3
_MINT_POLL_INTERVAL_S = 1.0

# pgw#677 background-turn gate: tenant requests always win the GPU; mint
# seeds and shape-warm compiles run only in granted turns.
#: Minimum continuous demand-blocked time before the background lane may
#: STEAL one turn (the minimum-progress guarantee — background work still
#: finishes under sustained tenant load).
_BG_STEAL_FLOOR_S = 30.0
#: COMPILE turns get a far higher steal floor (pgw#677 reopen sizing
#: correction): a stolen turn is not preemptible, and a real inductor
#: compile is 4-7 unabortable minutes on an L4 — ~100x the advertised
#: 30-90s residual. Compiles therefore run in tenant-idle gaps (arrivals
#: cluster around completions, so gaps exist under real load) and may
#: steal only against MINUTES of truly continuous demand — and when one
#: does, it announces itself on the wire (``bg_turn_steal``).
_BG_COMPILE_STEAL_FLOOR_S = 600.0
#: A turn that ran against live demand "costs" this multiple of its own
#: duration before the next steal — bounds the stolen duty cycle to
#: 1/(1+factor) even when every turn is a multi-minute compile.
_BG_STEAL_DEBT_FACTOR = 4.0
#: Idle-granted COMPILE turns additionally wait for this much tenant quiet
#: (arrivals cluster; this slashes the arrive-mid-compile collision).
_BG_COMPILE_QUIESCENCE_S = 5.0
#: How long one shape-warm thread admission attempt may wait before the job
#: re-queues (TurnGateBusy) — bounds head-of-line blocking of the global
#: warm queue; the persistent blocked-since clock keeps steals honest.
_BG_THREAD_ADMIT_WAIT_S = 0.5



def _model_failure_vocab(exc: BaseException) -> str:
    """Contract §9 ModelEvent.error vocabulary for residency failures."""
    if is_cuda_oom(exc):
        return "oom"
    if isinstance(exc, MissingSnapshotError):
        return "missing_snapshot"
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    return "load_failed"


def _shared_execution_lanes_need_fp8(
    slot_sizes: Dict[str, Dict[str, int]],
    shared_components: typing.Iterable[str],
    free_vram_bytes: int,
    *,
    margin_bytes: int = 2 * (1 << 30),
) -> bool:
    """Joint VRAM-fit decision for a shared-component multi-lane record
    (th#1043): loading each lane's precision reactively, one at a time,
    lets the FIRST lane to load consume all free VRAM at native precision
    — the shared-component invariant then refuses the offload placement
    the STARVED lane needs, hard-failing a fit that was achievable all
    along. Decide precision for the WHOLE group against its combined
    footprint (shared components counted once) before any lane loads.

    True when native precision doesn't fit the group but fp8-storage
    (denoiser weights ~halved) does — every lane in the group should force
    fp8 storage. False when native precision already fits (no forcing
    needed) or fp8 storage still wouldn't fit (per-lane reactive sizing
    makes its own honest call instead).
    """
    shared = list(shared_components)
    if len(slot_sizes) < 2 or not shared:
        return False
    first = next(iter(slot_sizes.values()))
    shared_bytes = sum(first.get(c, 0) for c in shared)
    exclusive_bytes = sum(
        b for sizes in slot_sizes.values()
        for comp, b in sizes.items() if comp not in shared
    )
    needed = shared_bytes + exclusive_bytes
    if needed + margin_bytes <= free_vram_bytes:
        return False
    fp8_needed = shared_bytes + 0.5 * exclusive_bytes
    return fp8_needed + margin_bytes <= free_vram_bytes


def _estimate_setup_need(per_ref: typing.Sequence[Tuple[int, int]]) -> int:
    """Pre-load VRAM headroom estimate for one setup's refs (pgw#636).

    ``per_ref`` carries ``(vram_hint, snapshot_bytes)`` per ref: a prior
    MEASURED footprint wins, else the wire snapshot's byte total (an honest
    first-load footprint for stored-precision lanes — make_room's margin covers
    slack). Both terms are MEASUREMENTS of bytes at hand.

    th#1867 deleted the third term. A ref with NEITHER fact used to raise the
    total to the endpoint's declared ``vram_gb``, and that declaration is gone
    (§2.4 ruling 4). The estimate is now strictly what is known: an unweighed
    ref contributes nothing, ``make_room`` evicts what the known refs need, and
    a genuine shortfall is found by the load and carried down the rung ladder —
    the same trade §1.35 makes everywhere else. Guessing here was never safe
    anyway: reserving a declared minimum wholesale for every never-seen
    checkpoint pick evicted the resident pipeline on 24 GB cards and pinned
    workers to one pipeline (the 2026-07-24 9.8/24 GB incident)."""
    needed = 0
    for hint, snapshot_bytes in per_ref:
        needed += hint if hint > 0 else max(0, int(snapshot_bytes))
    return needed


def _is_corrupt_load_error(exc: BaseException) -> bool:
    """Errors a truncated/corrupt snapshot produces at weights-load time
    (gw#408). Broad on purpose: the digest re-verify gate downstream
    separates real corruption from code bugs — a verified-clean tree
    re-raises the original error instead of quarantining."""

    if isinstance(exc, OSError):
        # e.g. "Unable to load weights from checkpoint file" (raised as
        # OSError by transformers/diffusers), FileNotFoundError from a
        # half-built tree. Resource exhaustion is not corruption.
        return getattr(exc, "errno", None) not in (_errno.ENOSPC, _errno.ENOMEM)
    if isinstance(exc, _struct.error):
        return True
    return type(exc).__name__ in (
        "SafetensorError", "HeaderTooLarge", "MetadataIncompleteBuffer",
        "UnpicklingError", "JSONDecodeError",
    )










# ---------------------------------------------------------------------------
# Endpoint instances (setup/warmup lifecycle)
# ---------------------------------------------------------------------------


@dataclass
class _CompileTargetRecord:
    """One exact live pipeline object eligible for compile-cell adoption."""

    incarnation_id: str
    spec: EndpointSpec
    pipeline: Any
    pipeline_weight_lane: str
    lora_bucket: int
    contract_digest: str
    active_compile_ref: str = ""
    active_compile_snapshot_digest: str = ""
    # gw#604: True when the active artifact is this worker's OWN mint (the
    # advertised digest is then the self-attested tar digest, not the store's
    # snapshot manifest digest — same bytes, different transport form).
    active_self_mint: bool = False
    function_names: Tuple[str, ...] = ()
    model_bindings: Tuple[Tuple[str, str, str], ...] = ()
    # Runtime guard failure is signaled from a handler thread. Guard every
    # mutable advertised field so StateDelta never observes a half-revoked
    # cell identity.
    state_lock: threading.Lock = dc_field(
        default_factory=threading.Lock, repr=False, compare=False)


@dataclass(frozen=True)
class _ArmOrder:
    """The arming decision for one dispatched attempt. The worker OBEYS it:
    ``aot_cell`` arms exactly ``selection``
    (already materialized and content-digest-verified), ``dynamo`` arms JIT
    intake, ``eager_only`` arms nothing. No discovery, ranking or self-mint
    fallback exists on this path — a failed exact arm is a typed refusal.

    ``adopt`` (pgw#1122) marks the ONE order the hub did not give: §4.27
    boot-adopt builds an identical order out of a cell this pod resolved by its
    OWN derived key. Nothing named that arm, so its refusal is a degrade to
    eager with a typed event, not a dead function — carrying the journey's
    ``BootAdoptOutcome`` here is what lets the degrade report itself under the
    same family/function/key the ``hit`` was reported under.
    """

    backend: str
    selection: Optional["_CompileArtifactSelection"] = None
    expected: Optional["aot_identity.ExpectedIdentity"] = None
    publisher_org: str = ""
    adopt: Optional["boot_adopt.BootAdoptOutcome"] = None
    #: pgw#1176: the OTHER entries this boot resolved. A boot derives a key SET
    #: and coverage ACCRETES, so several hits are the expected shape — each is
    #: armed into the same registry, the same target pool and the same live
    #: wrap after the first. A failure on one of these is a per-entry degrade
    #: (that class serves eager), never terminal: the first arm already proved
    #: the pod can serve compiled.
    extra: Tuple[Tuple[Path, Optional["aot_identity.ExpectedIdentity"], str],
                 ...] = ()

    @classmethod
    def for_artifact(
        cls,
        *,
        path: Path,
        ref: str,
        snapshot_digest: str,
        expected: Optional["aot_identity.ExpectedIdentity"],
        publisher_org: str,
        adopt: Optional["boot_adopt.BootAdoptOutcome"] = None,
        extra: Tuple[
            Tuple[Path, Optional["aot_identity.ExpectedIdentity"], str], ...
        ] = (),
    ) -> "_ArmOrder":
        """THE artifact -> arming-order map, in one place (pgw#1152).

        ONE route builds this object today — ``_setup_locked_inner``'s §4.27
        BOOT-ADOPT order — after pgw#1206 D deleted the Plan head that built
        the other. The constructor stays because the duplication it prevents is
        the same mapping pgw#1150 found between ``compile_cell()`` and ``cli.run``: a
        field ADDED here that one site sets and the other forgets silently
        diverges the two arm routes, which is this repo's most expensive defect
        shape: pgw#1108, pgw#1122, pgw#1141 and pgw#1141b were all "a rule the
        self-mint/plan path keeps and the adopt path does not".

        The selection is built here too, because the two sites also built THAT
        independently from the same three fields.
        """
        return cls(
            backend="aot_cell",
            selection=_CompileArtifactSelection(
                path=path, ref=ref, snapshot_digest=snapshot_digest),
            expected=expected,
            publisher_org=publisher_org,
            adopt=adopt,
            extra=extra,
        )


@dataclass(frozen=True)
class _JobOrder:
    """The NEUTRAL per-attempt order the dispatch driver executes (pgw#904).

    Produced by the wire head ``_legacy_order`` (from ``pb.RunJob``). The
    driver and every shared helper read this value and never a wire message,
    which is what kept the head swappable; head semantics that cannot be
    neutral (the ``required_compile`` fence) ride ``fence`` as a head-owned
    closure, and config snapshotting rides ``config_snapshot``.

    ``snapshots`` is TRANSPORT (ref-keyed presigned material for the store),
    never identity — identity lives in the derived spec's bindings.
    """

    request_id: str
    attempt: int
    function_name: str
    payload: bytes
    group: int
    slots: Mapping[str, dispatch.SlotOrder]
    adapters: Mapping[str, Tuple[dispatch.AdapterOrder, ...]]
    snapshots: Dict[WireRef, pb.Snapshot]
    input_manifest: Tuple[InputManifestEntry, ...]
    fence: Callable[[EndpointSpec], None]
    config_snapshot: Callable[[str, Dict[str, Any]], Optional[Any]]
    org: str = ""
    invoker_id: str = ""
    capability_token: str = ""
    inline_output: bool = False
    accelerator: str = ""  # "" = unstated (the spec decides), "cuda" | "none"
    gpu_index: int = 0
    lane_report: str = ""  # instruction surfaced to ctx.lane/metrics only
    # th#1871 P1: the hub DEMANDED a compiled cell for this dispatch. Carried on
    # the order because that is where the RunJob is read; copied onto the job
    # below, where the terminal posture is stamped.
    compile_required: bool = False
    stamped_config: Optional[Mapping[str, Any]] = None
    arm: Optional[_ArmOrder] = None


@dataclass(frozen=True)
class _CompileArtifactSelection:
    """One immutable compiled-artifact identity active on a pipeline.

    ``self_mint=False``: a hub-attached (store-served) cell selected before
    model setup — the gw#577 digest receipt governs it. ``self_mint=True``:
    this worker's OWN boot-warmup mint (gw#587 serving bootstrap) — ref is
    the worker's self-computed key ref and the digest is self-attested; the
    warmup proof, not a store receipt, gates serving.
    """

    path: Path
    ref: str
    snapshot_digest: str
    self_mint: bool = False


def _selection_for(
    delivered: Optional["_CompileArtifactSelection"],
    mint: Any,
) -> Optional["_CompileArtifactSelection"]:
    """The artifact identity a just-armed pipeline actually serves from.

    A self-mint outcome WINS over the boot's delivered family selection —
    when a delivered artifact failed to arm this object and the fleet policy
    minted instead, recording the delivered identity would advertise bytes
    this object does not serve (the gw#586 defect shape).

    ``mint`` is a finalized ``fleet_cells.SelfMint`` — an adopted cell, whose
    ``artifact`` is packed on this disk and whose ``ref`` carries the key
    STAMPED on that envelope. That is the only identity this function will
    hand back for a mint.

    A ``fleet_cells.PendingSelfMint`` yields the ``delivered`` selection (or
    nothing): pgw#805 — an exported cell's key folds the COMBINED GRAPH HASH
    of its class set, so it does not exist until the export finishes, and the
    pending's own ``ref`` is a COMPUTED ``kind="inductor"`` key that no
    artifact will ever carry. Advertising it would publish a self-attested ref
    against bytes that will be stamped with a different one. Nothing is
    advertised for an owed mint until ``adopt_delegated_mint`` reads the real
    key off the packed envelope.

    pgw#1033: that guard used to test ``mint.recipe == "aot"``, an attribute
    pgw#1010 deleted from every mint object — so it could not fire, and only
    the caller's ``delegated`` branch (which drops the selection it just asked
    for) kept the computed ref off the wire. The predicate is the mint's own
    state: a pending has no packed artifact.
    """
    if mint is not None:
        path = getattr(mint, "artifact", None)
        if path is None:
            return delivered
        return _CompileArtifactSelection(
            path=Path(path), ref=str(mint.ref),
            snapshot_digest=str(getattr(mint, "snapshot_digest", "") or ""),
            self_mint=True)
    return delivered


@dataclass
class _CompileObjectCandidate:
    """One setup-created pipeline and only the model slots that own it."""

    pipeline: Any
    slots: set[str] = dc_field(default_factory=set)


@dataclass
class _WarmupEvidence:
    """Successful handler warmups and the exact compile objects they proved."""

    count: int = 0
    functions_by_object: Dict[int, set[str]] = dc_field(default_factory=dict)
    #: pgw#677 reopen: non-empty when the warm plan was CUT SHORT (OOM
    #: backoff) — names the truncation. A truncated plan must never publish
    #: its partial capture as the family cell.
    aborted: str = ""


# gw#661: setup failures whose contract is "will be re-attempted". These are
# reported to the hub as a still-RUNNING activity, not the FAILED terminal —
# the hub reads FAILED as "no progress here" (th#1160) and condemns the pod.
_TRANSIENT_SETUP_ERRORS = (InsufficientDiskError, RetryableError, MissingSnapshotError)

# Consecutive transient losses on ONE record before the condition is this
# function's terminal truth. Each attempt already carries its own internal
# wait (the lane gate polls VRAM headroom for 45s), so this is minutes of
# real patience, not a tight spin.
MAX_TRANSIENT_SETUP_ATTEMPTS = 5


@contextmanager
def _pipeline_load_span(spec: EndpointSpec) -> typing.Iterator[Optional[boot_mod.BootSpan]]:
    """Open `pipeline_load` for one setup, or yield None outside the boot
    window (pgw#797).

    A plain `boot_mod.span(...)` cannot express "measure only during boot"
    without an `if` at every call site, and the one call site that forgot it is
    how steady-state work lands in a boot ladder.
    """
    if not boot_mod.in_boot():
        yield None
        return
    with boot_mod.span(boot_mod.PHASE_PIPELINE_LOAD, function=spec.name) as sp:
        yield sp


def _setup_error_will_retry(exc: BaseException) -> bool:
    """Whether this setup loss is contractually re-attempted (gw#661).

    CompiledLaneUnavailableError subclasses RetryableError but the worker
    DOES give up on it — it disables every handler requiring the unproven
    lane — so it is a failure, not a retry.
    """
    if not isinstance(exc, _TRANSIENT_SETUP_ERRORS):
        return False

    return not isinstance(exc, CompiledExecutionLaneUnavailableError)


@dataclass
class _ClassRecord:
    cls: type
    specs: List[EndpointSpec] = dc_field(default_factory=list)
    instance: Any = None
    server: Any = None  # ServerHandle for runtime="vllm"/"llama-server"
    ready: bool = False
    failed: Optional[str] = None
    # pgw#797: ordinal of the OPEN `pipeline_load` boot span covering the
    # in-flight setup, 0 when not booting. The nested `warmup` span names it
    # explicitly rather than inferring a parent from an implicit stack.
    boot_load_ordinal: int = 0
    lock: asyncio.Lock = dc_field(default_factory=asyncio.Lock)
    # pgw#647 concurrency contract: one live instance == one binding set with
    # mutable buffers (resident LoRA branches, adapter attach state), so
    # handler execution on it is SINGLE-FLIGHT by default. Held across
    # adapter activation + handler + deactivation; skipped only when the
    # endpoint class declared ``reentrant=True``. One-job-per-GPU used to
    # mask this; multi-GPU permits and multi-residency do not.
    run_lock: asyncio.Lock = dc_field(default_factory=asyncio.Lock)
    # pgw#677: module-exclusion mutex between loop-side executors (tenant
    # handler + adapter mutation, mint seed forwards — all already
    # serialized under run_lock) and the loop-LESS shape-warm thread, whose
    # compile executes these same modules. threading.Lock on purpose: the
    # warm thread must take it without an event loop; loop-side users take
    # it via a joined to_thread so the loop never blocks.
    turn_mutex: threading.Lock = dc_field(default_factory=threading.Lock)
    # Content-keyed shared components this record holds (gw#479): released
    # (refcount--) at vacate so the entries become LRU/drain candidates.
    shared_keys: List[Any] = dc_field(default_factory=list)
    # gw#494: the wire refs this record's instance BOOKED at load time —
    # teardown releases exactly these (never a re-derivation from the
    # possibly-rebound spec.models), so booking and clearing are provably
    # the same key space.
    held_refs: List[str] = dc_field(default_factory=list)
    # Exact snapshot digest behind each held model instance. A mutable tag can
    # keep the same wire ref while moving to new bytes; refs alone cannot
    # decide whether a ready instance is reusable.
    held_snapshot_digests: Dict[str, str] = dc_field(default_factory=dict)
    # Canonical load-time slot/ref/digest triples. Compile-target applicability
    # freezes these facts so two same-family SDXL checkpoints cannot
    # cross-certify merely because their graph/lane contracts match.
    held_bindings: List[Tuple[str, str, str]] = dc_field(default_factory=list)
    # The per-record object behind each booking. Residency has one entry per
    # wire ref, so a multiply-held ref needs this map to transfer its strong
    # representative when the latest owner leaves.
    held_objects: Dict[str, Any] = dc_field(default_factory=dict)
    # pgw#678: slot -> the worker-constructed pipeline injected into setup.
    # ``held_objects``/residency hold the LANE handle (an nn.ModuleDict of the
    # lane's exclusive modules whenever any component is unshared), which is
    # not a diffusers pipeline: adapters and the OOM offload rung must act on
    # THIS map instead.
    slot_pipelines: Dict[str, Any] = dc_field(default_factory=dict)
    # gw#494: a resolution re-pick moved the specs' bindings away from
    # held_refs; the instance serves the OLD pick and must be vacated.
    stale: bool = False
    # gw#551: wire refs of lane-registered slots (gw#479). Lane residency is
    # call-time-owned (LaneResidencyGate promotes + pins around each pipeline call);
    # the executor must neither whole-job-pin nor eagerly promote them, or
    # the idle sibling can never be LRU-swapped out.
    execution_lane_refs: set = dc_field(default_factory=set)
    # pgw#572: exact compile-capable objects owned by this READY record. The
    # IDs are minted after successful setup and cleared before vacate; they do
    # not derive from mutable refs, authored specs, or object memory addresses.
    compile_targets: Dict[str, _CompileTargetRecord] = dc_field(default_factory=dict)
    # pgw#1104: lanes this record's setup() APPLIED to its own weights
    # (`gen_worker.report_applied_lane`). The binding names the checkpoint the
    # hub resolved; a serve-time recipe moves the executed lane away from it,
    # and `_served_execution_lane` must report the lane that RUNS. Dies with
    # the instance — a new setup re-reports or the lane reverts to the binding.
    applied_lanes: List[lanespec.AppliedLane] = dc_field(default_factory=list)
    # pgw#1043 §PRODUCTIZATION: the attention path this record's setup()
    # INSTALLED (`gen_worker.report_applied_attention`). Empty == dense, which
    # is why no endpoint is obliged to report. Dies with the instance.
    applied_attention: List[Any] = dc_field(default_factory=list)
    # th#1871 P1 (pgw#1225): the typed POSTURE this record is serving under —
    # every lever reached for, in order, with the shortfall that forced it.
    # Owned by the record for the same reason `applied_lanes` is: a lever
    # applied to THESE weights stops being a fact the moment they are torn
    # down, and a posture that outlived its pipeline would qualify the next
    # instance's measurements with the last one's degradation.
    posture: posture_mod.PostureLedger = dc_field(
        default_factory=posture_mod.PostureLedger)
    # gw#661: consecutive will-retry setup losses; reset by any success.
    transient_setup_failures: int = 0
    # pgw#748 phase 1: the armed degree-D rank group serving THIS record, or
    # None at degree 1 (every record today). Owned by the record because it is
    # bound to the record's pipeline objects and must die with them.
    sp_runtime: Optional[Any] = None
    # pgw#671 eager-first boot: the in-flight background self-mint for this
    # record's live instance, when the boot went READY(eager) with the mint
    # deferred. Cleared when the mint completes, is disproven, or is
    # abandoned (peer-cell adoption, vacate, shutdown).
    background_mint: Optional["_BackgroundMint"] = None
    # pgw#824: WHY this record is not serving from a cell, as the arming
    # brain's own classified token (fleet_cells.ArmOutcome.eager_reason /
    # serving_mode.POSTURE_*). Every request served eager reports it as
    # `fallback_reason`, so "why is this fleet eager right now" is one GROUP BY
    # over request rows that joins the worker's own `self_mint_skipped` events
    # on the same string. "" once a cell is armed.
    eager_posture: str = ""


class _MintAbandoned(Exception):
    """The background mint was asked to stop (adoption/vacate/shutdown)."""


class _SeedPreempted(Exception):
    """pgw#677: a tenant arrival cooperatively cancelled the in-flight mint
    seed forward; the driver re-queues the unit and yields the turn."""


@dataclass
class _BackgroundMint:
    """One deferred boot self-mint (pgw#671, worker half of th#1187).

    The instance went READY serving EAGER; this carries everything the
    background driver needs to seed the full derived warm plan through the
    hot-swap routers, wait for the background compiles, prove, finalize,
    publish, and hot-swap the record to compiled. ``abandon`` is the clean
    stop signal — the driver checks it at unit boundaries (finish the
    current forward, then discard wholesale; local state is never left
    half-mutated)."""

    spec: EndpointSpec
    instance: Any
    snapshots: Optional[Dict[WireRef, "pb.Snapshot"]]
    # id(pipeline) -> fleet_cells.PendingSelfMint (same objects the arming
    # scope produced; shared captures keep their sharing structure).
    pendings: Dict[int, Any]
    # id(pipeline) -> the actual pipeline object (id() keys alone cannot
    # keep the object alive or recover it).
    pipes: Dict[int, Any]
    abandon: asyncio.Event = dc_field(default_factory=asyncio.Event)
    task: Optional["asyncio.Task[None]"] = None
    act: Optional[Any] = None  # the handed-over self_mint_compile Activity
    # pgw#677: the RequestContext of the in-flight PREEMPTIBLE seed forward
    # (idle-granted turns only — stolen turns run to completion). A tenant
    # admission cancels it; the driver re-queues the unit.
    seed_ctx: Optional[Any] = None
    # pgw#784: the two facts a DELEGATED mint's child process needs and cannot
    # rediscover for itself — the endpoint module(s) to walk (the child re-runs
    # discovery in a fresh interpreter) and the parent's own RESOLUTION of each
    # setup slot: identity, already-materialized local tree, and pgw#617
    # composition, in ONE value per slot (pgw#974, `child_contract.MintSlot`).
    # The paths matter because a mint is compute, and a mint process that could
    # download is one that can stall on a lemon host (pgw#786); the refs matter
    # because `ctx.slots` is built from bindings and the child rediscovers none
    # for a hub-catalog slot (pgw#969).
    modules: Tuple[str, ...] = ()
    slots: Dict[str, MintSlot] = dc_field(default_factory=dict)
    # th#1299: WHY this mint was asked to stop. The abort event used to report
    # "(adopt-on-arm / vacate / shutdown)" — three unrelated causes in one
    # string — so a mint that died could not be told from a mint that was
    # legitimately superseded without joining the hub's pod tables by hand.
    # Set by abandon_background_mint before the signal, read by the terminal
    # handler; a code (queryable) and the human sentence beside it.
    abandon_code: str = "unspecified"
    abandon_reason: str = ""


def _mint_origin(bg: "_BackgroundMint", spec: EndpointSpec) -> str:
    """WHICH mint a warm context belongs to (pgw#969), for the deferred
    slot-resolution errors it may raise. The delegated child's twin is
    ``mint_child.mint_identity``; both exist because ``ValueError: slot
    'pipeline': no resolved model ref`` named a symptom and no mint."""
    keys = sorted({str(getattr(p, "compiled_graph_key", "")) for p in bg.pendings.values()})
    return (
        f"in-process mint fn={spec.name!r} "
        f"key={(keys[0] if len(keys) == 1 else keys) or '(none)'!r}")


def _mint_modules(spec: EndpointSpec) -> Tuple[str, ...]:
    """The module list a mint child re-runs discovery over (pgw#784).

    The spec's own declaring module — which is what the baked manifest named
    for this function. ``registry.collect_endpoints`` walks it and its
    submodules, so the child rediscovers this class AND its sibling functions
    (the warm plan is class-scoped, pgw#654) without the parent serializing
    anything live.
    """
    module = str(getattr(spec, "module", "") or "").strip()
    return (module,) if module else ()


def _delegated_pendings(pendings: typing.Mapping[int, Any]) -> bool:
    return any(getattr(p, "delegated", False) for p in pendings.values())


@dataclass
class _HostRamBlock:
    """One exact, still-unsatisfied host-RAM admission observation."""

    failure_event: pb.ModelEvent
    last_available_bytes: int


def _canonical_host_ram_refs(refs: typing.Iterable[str]) -> List[WireRef]:
    """Keep only canonical model refs suitable for protocol evidence."""
    return list(dict.fromkeys(
        WireRef(ref)
        for value in refs
        if (ref := str(value or "").strip()) and not ref.startswith("shared::")
    ))


def _shared_loader_must_hit() -> Any:
    """acquire_shared loader for peeked keys (gw#479): the object was seen in
    the cache under the load lock, so a miss here is a bookkeeping bug."""
    raise RuntimeError("shared component vanished between peek and acquire")


@dataclass
class _InjectionResult:
    """What one setup injection produced (gw#479): the setup kwargs, the
    per-slot residency objects+bytes, which slots were lane-registered
    inline, the shared keys this record now holds, and the VRAM booked on
    shared:: entries (counted once, excluded from per-slot residuals)."""

    kwargs: Dict[str, Any]
    loaded: Dict[str, Tuple[Any, int]]
    # pgw#678: the worker-constructed PIPELINE per slot, kept apart from
    # ``loaded`` for the same reason ``compile_objects`` is: a shared-component
    # lane books an ``nn.ModuleDict`` of the lane's EXCLUSIVE modules as its
    # residency/movement handle, so ``loaded``/``residency.obj`` is not the
    # object adapters, offload rungs or the LoRA registry may act on.
    slot_pipelines: Dict[str, Any] = dc_field(default_factory=dict)
    execution_lane_slots: set = dc_field(default_factory=set)
    shared_keys: List[Any] = dc_field(default_factory=list)
    shared_bytes: int = 0
    # gw#551: slots whose pipeline __call__ the LaneResidencyGate wrapped. Only these
    # may become call-time-owned; an un-gateable pipeline (no instance
    # __call__) keeps the eager whole-job pin + promote path.
    gated_slots: set = dc_field(default_factory=set)
    # Actual worker-constructed pipelines whose declared compile targets
    # resolve. Kept separately because shared-lane residency may replace the
    # bookkeeping object with a ModuleDict while setup receives the pipeline.
    compile_objects: List[_CompileObjectCandidate] = dc_field(default_factory=list)
    # pgw#1093: the ARM FACT — every object an arm RETURNED TRUE for, on
    # either scope (slot injection or `arm_compile()` inside setup()).
    # Recorded before any later re-scan can drop it, and never re-derived
    # from `is_compile_armed()`: a permanent degrade flips that probe to
    # False, which would let the end-of-setup invariant excuse exactly the
    # boot it exists to catch.
    armed_objects: List[Any] = dc_field(default_factory=list)
    # id(pipeline) -> exact attached artifact that successfully armed it.
    # Installed only after the setup warmup completes.
    active_compile_artifacts: Dict[int, _CompileArtifactSelection] = dc_field(
        default_factory=dict)
    # gw#587 CORRECT FIX: id(pipeline) -> fleet_cells.PendingSelfMint for
    # objects armed from a fresh self-mint capture, not yet proven or
    # packed. The warmup-proof loop finalizes (packs + publishes) exactly
    # the proven entries and abandons the rest — never before the proof.
    pending_self_mints: Dict[int, Any] = dc_field(default_factory=dict)
    # pgw#824: the arming brain's classified reason for every compile object
    # that ended this setup WITHOUT an armed cell. Carried into the record, so
    # the reason outlives the function that computed it.
    eager_postures: List[str] = dc_field(default_factory=list)
    # pgw#923: every measured adoption ATTEMPT this injection made, in order.
    # The arm happens here; the warmup that prices it happens after setup
    # returns, so the terminal wire event is sent once BOTH halves are known —
    # which is exactly why the boot-attached adoption never had a measured row
    # while the hub-commanded one (arm and warm in a single frame) did.
    adoptions: List["fleet_cells.CellAdoption"] = dc_field(default_factory=list)

    def add_compile_object(
        self, pipeline: Any, slots: typing.Iterable[str],
    ) -> _CompileObjectCandidate:
        """Record exact object ownership without duplicating shared objects."""
        for candidate in self.compile_objects:
            if candidate.pipeline is pipeline:
                candidate.slots.update(str(slot) for slot in slots if str(slot))
                return candidate
        candidate = _CompileObjectCandidate(
            pipeline=pipeline,
            slots={str(slot) for slot in slots if str(slot)},
        )
        self.compile_objects.append(candidate)
        return candidate


@dataclass
class _Job:
    request_id: str
    attempt: int
    spec: Optional[EndpointSpec]
    intent_id: str = ""
    ctx: Optional[RequestContext] = None
    task: Optional[asyncio.Task] = None
    exec_task: Optional[asyncio.Task] = None
    renew_task: Optional[asyncio.Task] = None
    finished: bool = False
    superseded: bool = False
    cancel_requested: bool = False
    # pgw#687: True once this job owns its GPU permit + instance gate and the
    # handler is (about to be) running. A job that is NOT executing is parked
    # pre-execution — safe to fail RETRYABLE so the hub replans it elsewhere.
    executing: bool = False
    # pgw#687: watches cancel -> terminal for THIS job.
    unwind_watch: Optional[asyncio.Task] = None
    # gw#516: True while the job is past the decode->finalize handoff (GPU
    # slot terminally released, encode/upload tail running, result unshipped).
    finalizing: bool = False
    # th#913/gw#596: the CONCRETE lane serving this job (stamped post-setup as
    # a forecast for ctx.lane, RE-composed at the terminal from the served
    # identity — ie#655). "" = not yet determined.
    execution_lane: str = ""
    # ie#655: the hub's lane instruction, kept so the terminal composition can
    # honor a declared (handles=) body without re-reading the dispatch order.
    lane_report: str = ""
    # th#1871 P1: the hub DEMANDED a compiled cell for this dispatch. It is a
    # second, independent statement of the compile axis and both are needed:
    # `lane_report` is empty whenever the lane rides HelloAck's ModelResolution
    # instead of the per-request override (`scheduler_dispatch.go:1037` — "" =
    # policy), and on that path the declared axis would otherwise be unknown to
    # the worker. Reported, never enforced: `_validate_required_compile` is what
    # enforces it, at dispatch, and this is only its shadow on the measurement.
    compile_required: bool = False
    # pgw#789 (th#1293 dimensions): this request was served EAGER by a compiled
    # lane — a pgw#680 guard miss, a router heal/volatile verdict, or an
    # aot_serve ingress refusal. Set from the guard-miss callback, which fires
    # DURING the request and names it via postmortem.current_inflight_request().
    # Without it a fallback sample reports lane=...+compiled and silently
    # contaminates every compiled-vs-eager latency comparison with eager data.
    served_eager_fallback: bool = False
    fallback_reason: str = ""
    # pgw#888: the dispatch fence runs TWICE for one job (at intake, and again
    # as the last execution fence before the GPU turn), so a degrade that
    # persists across both would confess twice and double every hub-side count
    # of it. One request, one confession.
    pinned_cell_degrade_reported: bool = False
    # pgw#789: (steps, width, height) of the EXECUTED payload, defaults
    # applied — the axes latency is a function of. Stamped beside `lane`,
    # where the resolved payload is in scope; 0 means "not applicable"
    # (non-spatial function), never "zero".
    shape: Tuple[int, int, int] = (0, 0, 0)
    # th#1779: set by the handler THREAD itself when it returns. A sync
    # handler cannot be killed and its asyncio wrapper task says nothing about
    # it once cancelled, so this event is the only truthful answer to "is the
    # abandoned handler still on the card".
    handler_thread_done: threading.Event = dc_field(default_factory=threading.Event)
    admitted_at: float = dc_field(default_factory=time.monotonic)
    # One JobProgress seq space per job, shared by stream chunks and ctx
    # events so interleaved sends stay monotonic. itertools.count.__next__
    # is atomic under the GIL — safe from handler threads.
    seq: "itertools.count[int]" = dc_field(default_factory=lambda: itertools.count(1))


class DispatchGroupUnresolved(RetryableError):
    """pgw#779: this dispatch does not name an execution group this pod has.

    RETRYABLE, not INVALID: nothing about the tenant's input is wrong — the
    hub's `ResolvedCompute` is missing or disagrees with the delivered packing,
    and a dispatch that carries the right one serves fine. Refused rather than
    floored onto group 0, which is the group that is always busiest."""


class _PermitHold:
    """One live claim on one GPU permit."""

    __slots__ = ("label", "task")

    def __init__(self, label: str, task: "Optional[asyncio.Task[Any]]") -> None:
        self.label = label
        self.task = task

    def dead(self) -> bool:
        return self.task is not None and self.task.done()

    def __str__(self) -> str:
        return f"{self.label}{' [task already finished]' if self.dead() else ''}"


class _PermitLedger:
    """Who holds each GPU permit — so *can this permit ever come back?* is a
    decidable state question instead of a guess about how long to wait.

    pgw#738 / gw#666. The #382 mid-handler re-acquire has no honest progress
    signal of its own, and the two candidates are both wrong:

    * **FIFO position.** With ``per_group=1`` the queue is one deep and a
      healthy holder computing for four hours moves it zero times. "Position
      did not advance" is the NORMAL state of a busy card, not a stall.
    * **The holder's heartbeat / stage progress.** A holder's silence is not
      the waiter's fault, and this very issue proves a healthy holder can be
      log-silent and GPU-idle by construction — the ~20 GB publish phase that
      got 6eb50902 killed on the dead-pod signature. And if a holder really is
      wedged, the HOLDER is the thing to condemn; pgw#687's cancel-unwind
      watch, the request deadline and the stuck-thread recycler already own
      that. The waiter must not second-guess them.

    So the wait is bounded by REACHABILITY, not by progress and not by a
    clock: it is refused only in the one state that cannot resolve itself —
    an outstanding permit attributable to no live holder, i.e. a raw acquirer
    outside this ledger or a hold whose owning task already finished.
    """

    __slots__ = (
        "depth", "_holds", "_next_token", "transitions", "_idle_since",
        "_idle_span",
    )

    def __init__(self, depth: int) -> None:
        self.depth = max(1, int(depth))
        self._holds: Dict[int, Dict[int, _PermitHold]] = {}
        self._next_token = 0
        # pgw#1154, THE ZERO-BUBBLE METER. id(sem) -> the monotonic instant
        # this group's permits went wholly unheld, and the span that ended
        # when the next holder took one. A permit is the group's card, so an
        # unheld permit is a card nobody is scheduled on — the inter-request
        # bubble Paul's bar is stated against, measured on every request
        # instead of inferred from a one-off harness. Seeded only by the
        # FIRST release, so request 1 reports no span at all: "unmeasured"
        # and "zero" must not render alike.
        self._idle_since: Dict[int, float] = {}
        self._idle_span: Dict[int, float] = {}
        # Bumped on every take/drop. Two probes spanning no transition saw a
        # settled ledger, which is what makes the predicate safe to act on: a
        # permit handed to another waiter between them would otherwise read as
        # unaccounted for exactly one loop turn.
        self.transitions = 0

    def take(
        self, sem: asyncio.Semaphore, label: str, *, owned: bool = True,
    ) -> int:
        """Register a hold. Call with NO await between ``sem.acquire()``
        returning and this call, so the ledger cannot miss a grant.

        ``owned=False`` for a hold whose owner is a HANDLER THREAD rather than
        the calling task (the #382 re-acquire): the calling task there is the
        one-shot ``run_coroutine_threadsafe`` wrapper, which finishes
        immediately and would read as a dead owner for the rest of the job.
        """
        self._next_token += 1
        token = self._next_token
        task: Optional[asyncio.Task[Any]] = None
        if owned:
            try:
                task = asyncio.current_task()
            except RuntimeError:
                task = None
        holds = self._holds.setdefault(id(sem), {})
        if not holds:
            # First holder after an unheld window: close the bubble.
            since = self._idle_since.pop(id(sem), None)
            if since is not None:
                self._idle_span[id(sem)] = max(0.0, time.monotonic() - since)
        holds[token] = _PermitHold(label, task)
        self.transitions += 1
        return token

    def drop(self, sem: asyncio.Semaphore, token: int) -> None:
        holds = self._holds.get(id(sem), {})
        if holds.pop(token, None) is not None:
            self.transitions += 1
            if not holds:
                # Nobody is scheduled on this group's card as of now.
                self._idle_since[id(sem)] = time.monotonic()

    def consume_idle(self, sem: asyncio.Semaphore) -> Optional[float]:
        """Seconds this group's card sat with no permit holder immediately
        before the current holder took it, or ``None`` when no such window
        has been observed yet (the worker's first job). Read-once."""
        return self._idle_span.pop(id(sem), None)

    def unreachable(self, sem: asyncio.Semaphore) -> Optional[str]:
        """Reason iff some outstanding permit has no live holder, else None."""
        free = getattr(sem, "_value", None)
        if not isinstance(free, int):
            return None  # unknown semaphore internals: never condemn
        outstanding = self.depth - free
        if outstanding <= 0:
            return None
        holds = list(self._holds.get(id(sem), {}).values())
        live = [h for h in holds if not h.dead()]
        if len(live) >= outstanding:
            return None
        who = ", ".join(str(h) for h in holds) or "nobody this worker knows"
        return (
            f"{outstanding} of {self.depth} GPU permit(s) outstanding but only "
            f"{len(live)} live holder(s): {who}"
        )


class _GpuSlotLease:
    """Thread-safe handle for a job's GPU slot (#382).

    Blob uploads and result sends are network/CPU work; holding the GPU
    semaphore across them idles the GPU for longer than the model's own
    compute on turbo image models. The lease lets ``RequestContext`` release
    the slot from the handler thread while ``save_bytes`` waits on the
    network (re-acquiring before returning to tenant code), and lets the
    executor free the slot as soon as ``_execute`` returns — before
    result-blob upload and result send. Transitions are lock-guarded so a
    hold is released at most once.
    """

    __slots__ = (
        "_sem", "_loop", "_lock", "_held", "released_at",
        "_ledger", "_label", "_token",
    )

    def __init__(
        self,
        sem: asyncio.Semaphore,
        loop: asyncio.AbstractEventLoop,
        ledger: _PermitLedger,
        label: str,
        token: int,
    ) -> None:
        self._sem = sem
        self._loop = loop
        self._ledger = ledger
        self._label = label
        self._token = token
        self._lock = threading.Lock()
        self._held = True
        # Monotonic time of the FIRST release — the terminal finalize handoff
        # (gw#476/gw#516) or the executor's post-handler release, whichever
        # came first. Reads out the finalize-overlap window.
        self.released_at: Optional[float] = None

    def yield_slot(self) -> bool:
        """Release the slot if held (any thread). True iff this call released."""
        with self._lock:
            if not self._held:
                return False
            self._held = False
            self.released_at = time.monotonic()
        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            self._release_on_loop()
        else:
            self._loop.call_soon_threadsafe(self._release_on_loop)
        return True

    def _release_on_loop(self) -> None:
        # Ledger drop and semaphore release in one loop callback: the ledger
        # never describes a permit this lease no longer owns.
        self._ledger.drop(self._sem, self._token)
        self._sem.release()

    def reacquire(self) -> None:
        """Blocking re-acquire from a handler thread.

        Waits for as long as a live holder takes — no clock (gw#666), and the
        pgw#954 order (instance gate -> permit) guarantees the permit is
        reachable while one exists. Raises :class:`GpuSlotUnreachable` only
        when the ledger proves no live holder can return it; see
        :class:`_PermitLedger` for why reachability, and not progress, is the
        honest bound here.
        """
        asyncio.run_coroutine_threadsafe(self._reacquire(), self._loop).result()

    async def _reacquire(self) -> None:
        acquire = asyncio.ensure_future(self._sem.acquire())
        watch = asyncio.ensure_future(self._watch_unreachable())
        try:
            await asyncio.wait(
                {acquire, watch}, return_when=asyncio.FIRST_COMPLETED)
            if acquire.done() and not acquire.cancelled():
                acquire.result()
                self._token = self._ledger.take(
                    self._sem, self._label, owned=False)
                with self._lock:
                    self._held = True
                return
            raise GpuSlotUnreachable(
                f"GPU permit reacquire refused for {self._label}: "
                f"{watch.result()}"
            )
        finally:
            for task in (acquire, watch):
                if not task.done():
                    task.cancel()

    async def _watch_unreachable(self) -> str:
        """Return the reason once the permit provably cannot come back."""
        while True:
            await asyncio.sleep(_PERMIT_PROBE_S)
            settled = self._ledger.transitions
            if self._ledger.unreachable(self._sem) is None:
                continue
            await asyncio.sleep(_PERMIT_PROBE_S)
            if self._ledger.transitions != settled:
                continue  # a handoff happened: the ledger was mid-transition
            reason = self._ledger.unreachable(self._sem)
            if reason is not None:
                return reason


class Executor:
    def __init__(
        self,
        specs: List[EndpointSpec],
        send: Callable[[pb.WorkerMessage], Awaitable[None]],
        *,
        settings: Any = None,
        store: Optional[ModelStore] = None,
        gpu_slots: Optional[int] = None,
        topology: Optional[ExecutionTopology] = None,
    ) -> None:
        self.specs: Dict[str, EndpointSpec] = {s.name: s for s in specs}
        self._send = send
        self._settings = settings
        self.store = store or ModelStore(send)
        # pgw#654: TF32 is PROCESS-GLOBAL state — set once at executor
        # bootstrap, never inside per-instance endpoint setup. pgw#1049: the
        # write is the settings authority's (its declared table IS the
        # pgw#654 posture); calling it here keeps embedder/test processes
        # that never ran env_seal.establish on the declared posture too.
        if torch is not None:
            settings_authority.impose_torch()
        self.intent_registry: Optional[IntentRegistry] = None
        for s in specs:
            for b in s.models.values():
                self.store.register_binding(wire_ref(b), b)
        # th#697: declared (pre-resolution) model bindings per spec, so hub
        # precision resolutions apply full-replace against the AUTHORED refs.
        self._declared_models: Dict[str, Dict[str, Any]] = {
            s.name: dict(s.models) for s in specs
        }
        # pgw#532: hub-named ref -> the ONE tensorhub binding object minted
        # for it. Identity-stable so equal picks across requests derive equal
        # instance keys (one resident instance per (class, resolved pick)).
        self._hub_bindings: Dict[str, ModelRef] = {}
        # pgw#748 phase 1 / th#1285: the hub's `G×D` packing is authoritative.
        # G == the slot semaphore, and it comes from the delivered topology —
        # NEVER from torch.cuda.device_count() (at D>1 slots are not devices)
        # and never from an operator knob. An explicit gpu_slots= still wins
        # for the local `cli/serve` path and for tests.
        self.topology = topology or ExecutionTopology.single()
        # pgw#748: id(record) -> the GroupPlan its ranks agreed on. Keyed by
        # identity because a record is the group's owner and dies with it.
        self._sequence_plans: Dict[int, Any] = {}
        self.store.bind_topology(self.topology)
        self._gpu_slots = max(1, int(gpu_slots) if gpu_slots else self.topology.execution_groups)
        # pgw#779: G INDEPENDENT permits, not a COUNT of G. A count admits G
        # concurrent jobs but binds none of them to a group, so four dispatches
        # naming the same card serialized on one `run_lock` while three cards
        # idled — reported as four healthy slots. One permit per group makes
        # "one GPU job per group" the invariant the object expresses.
        # ``gpu_slots=`` (the `cli serve`/test override, never a production
        # knob) still means what it meant: its concurrency is divided among the
        # groups, so at G==1 group 0's permit IS today's whole pool.
        per_group = max(1, self._gpu_slots // max(1, self.topology.execution_groups))
        self._gpu_permits: Tuple[asyncio.Semaphore, ...] = tuple(
            asyncio.Semaphore(per_group)
            for _ in range(max(1, self.topology.execution_groups))
        )
        self._gpu_permits_each = per_group
        # pgw#738: every permit acquisition in this file registers here, so a
        # blocked #382 re-acquire can ASK whether the permit is reachable
        # instead of guessing how long to wait for it.
        self._permits = _PermitLedger(per_group)
        # Group 0's permit. At G == 1 — every pod today — this IS the pool.
        self._gpu_semaphore = self._gpu_permits[0]
        # pgw#782: the slot count is also the CPU divisor. torch sizes its
        # intra-op pool from the HOST's logical processors, so a 4-group pod
        # runs four 48-thread teams against a 32-core quota. De-escalation
        # only, and measured NEUTRAL on the width-4 sdxl burst (37.1s vs
        # 36.7s) — the collapse is the shared interpreter (pgw#783), not this.
        # Kept because the oversubscription is real and bites narrow-quota
        # pods. Here, with the other process-global torch state (TF32 above),
        # keyed off the same authoritative slot count.
        self.cpu_budget = cpu_budget.impose_intra_op_threads(self._gpu_slots)
        # Model loads/promotions serialize so allocator-delta measurements
        # and free-VRAM reads don't cross-contaminate (#369).
        self._load_lock = asyncio.Lock()
        self._setup_active: Dict[Any, str] = {}
        # gw#624: set by a rolled-back setup; the next attempt gc-purges the
        # cancelled load's cycle-held modules before allocating.
        self._pending_alloc_purge = False
        # pgw#678: wire refs whose content-keyed share plan proved impossible
        # on THIS host — the lane's placement fell to an offload rung, which
        # the shared-component invariant refuses (hooks on a shared module
        # poison sibling lanes). Learned once so the retry loads monolithically
        # instead of re-deriving the same refused plan until retry_exhausted.
        self._no_share_refs: typing.Set[str] = set()
        # pgw#548: worker-local capacity blocks retain the exact numeric
        # requirement that failed. They are cleared only by a later measured
        # observation after owner/pin release; no timer or prose retry path.
        self._host_ram_lock = asyncio.Lock()
        self._host_ram_send_lock = asyncio.Lock()
        self._host_ram_generation = 0
        self._host_ram_blocks: Dict[WireRef, _HostRamBlock] = {}
        # Commit-ordered, latest-per-ref producer outbox. Transport capacity
        # enqueue is nonblocking, but this outbox still makes global generation
        # order explicit under concurrent failure/progress producers.
        self._host_ram_outbox: Dict[str, pb.ModelEvent] = {}
        # Active failures survive until residency or measured satisfaction.
        # Satisfied progress survives only until its exact generation completes
        # stream.write; before delivery it replays after reconnect because
        # Transport.reset_for_reconnect() intentionally sheds transient lanes.
        # Once progress satisfies a block its failure is no longer replayed:
        # older hubs ignore the additive progress enum and must not be handed a
        # stale FAILED that they can never clear.
        self._host_ram_progress: Dict[str, pb.ModelEvent] = {}
        # Parsed per-request LoRA state dicts, keyed by ref@digest (gw#393).
        self._adapter_cache = lora_util.AdapterCache()
        # Adapters attached to resident pipelines; requests toggle the active
        # set (gw#399). Demotion out of VRAM drops attachments.
        self._adapters = lora_util.AdapterResidency()
        self.store.residency.pre_demote = self._adapters.detach
        # Real wiring is worker.py assigning this attribute directly
        # (Executor is constructed before Lifecycle exists).
        self._on_state_change: Callable[[], None] = lambda: None
        self.file_base_url: str = ""
        # th#1087: worker-local mutable config (declared-parameter values +
        # snapshot file for subprocesses). Worker wiring replaces it with the
        # settings-pathed store; the default keeps embedded/CLI runs working.
        self.runtime_config = ConfigStore()
        # Current worker JWT for hub HTTP calls (capability renewal). Worker
        # wiring points this at the transport's rotated credential; until then
        # the process-wide credential source answers (pgw#848). It read
        # `getattr(settings, "worker_jwt", "")` until pgw#876 §2 — a field
        # pgw#848 RENAMED, so the getattr default swallowed the AttributeError
        # the rename exists to raise and this provider silently returned "".
        self.worker_jwt_provider: Callable[[], str] = (
            lambda: worker_credential.current()
        )
        self.draining = False
        self.jobs: Dict[Tuple[str, int], _Job] = {}
        # pgw#687 cancel-unwind quarantine: (request_id, attempt) -> detail for
        # every cancel that has not reached a terminal result within the grace,
        # and the function names WE marked unavailable for them.
        self._unwind_stuck: Dict[Tuple[str, int], str] = {}
        self._unwind_quarantined: set = set()
        # Process replacement seam, shared with the deadline reaper: tests
        # substitute a recorder instead of really exiting.
        self._process_exit: Callable[[int], None] = os._exit
        self._idle = asyncio.Event()
        self._idle.set()
        # pgw#677 background-turn gate state. Threading primitives on
        # purpose: the shape-warm thread must block on them without a
        # running event loop. _bg_unit_mutex = single-flight across ALL
        # background GPU consumers (mint seed units + shape-warm compiles);
        # _bg_quiet mirrors _idle for thread-side waits; the debt floats
        # implement the minimum-progress steal accounting.
        self._bg_unit_mutex = threading.Lock()
        self._bg_state_lock = threading.Lock()
        self._bg_quiet = threading.Event()
        self._bg_quiet.set()
        self._bg_steal_debt_until = 0.0
        self._bg_last_tenant_activity = 0.0
        # First refused thread-turn admission (spans TurnGateBusy requeue
        # cycles); None once admitted. The steal clock reads it.
        self._bg_blocked_since: Optional[float] = None
        # pgw#674 rotation preload: stages the hub's desired NEXT instances
        # (download -> pinned host -> VRAM double-buffer) WHILE jobs compute.
        # Lifecycle feeds it desired state; job admit/finish pokes it.

        self.preloader = Preloader(self)
        # gw#516: count of jobs in their slotless finalize tail. Mutated from
        # handler threads at the terminal slot release, so lock-guarded;
        # surfaced to the hub via StateDelta.finalizing_jobs.
        self._finalizing_lock = threading.Lock()
        self._finalizing_count = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Instance groups: specs sharing (cls, bindings) share one instance;
        # variant specs of the same class get separate instances. Function-
        # shaped endpoints (cls=None) have no instance at all.
        self._classes: Dict[Any, _ClassRecord] = {}
        for s in specs:
            if s.cls is None:
                continue
            rec = self._classes.setdefault(s.instance_key, _ClassRecord(cls=s.cls))
            rec.specs.append(s)
        # pgw#654 warm-tax fix: graph_keys already warm-RUN in this process,
        # keyed by warm CONTRACT (class + per-slot lane facts + component
        # overrides — everything that selects graphs/kernels, NEVER the
        # checkpoint ref). A new checkpoint instance of an already-warmed
        # contract is a cache hit: it runs one verification job, not the
        # plan. Process-local by design — allocator pool, cuBLAS/cuDNN
        # heuristics and dynamo's code cache die with the process.
        self._warm_contract_runs: Dict[Any, set] = {}
        # pgw#797: warm forwards counted by the in-flight `warmup` span.
        self._warm_iterations: int = 0
        # pgw#923: the boot warmup's measured cost, joined onto the
        # adoption event for every cell this boot attached.
        self._boot_warm_ms: int = 0
        # Hardware-gate failures: fn name -> (reason, detail, axes).
        self.unavailable: Dict[str, Tuple[str, str, Dict[str, str]]] = {}
        # Runtime compile failures are owned by the exact record/target that
        # disabled each alias. A successful fresh setup may clear only these
        # marks, and only after a new active target proves the alias again.
        self._compile_failure_owners: Dict[
            str, Tuple[_ClassRecord, str]
        ] = {}
        # gw#494: entries in `unavailable` that gate_functions owns — cleared
        # and re-derived on every (re-)gate so gating is idempotent; setup
        # failures (owned by _mark_setup_failed) survive re-gates.
        self._gate_owned: set = set()
        # Last hardware probe, so resolutions can re-run the gates.
        self._last_gpu_info: Optional[hostfacts.HostFacts] = None
        # th#683 P3: how each serveable function will run on the actual card
        # (native / emergency / offload / cpu) + honest-guidance advisory.
        self.serve_plans: Dict[str, "ServePlan"] = {}
        # Gate-time placement is immutable between hardware re-gates. Runtime
        # degradation updates serve_plans for FnDegraded telemetry, but must
        # not force an unrelated dynamic model pick down the same rung.
        self._gate_serve_plans: Dict[str, "ServePlan"] = {}
        # gw#463: learned degraded floor per model ref — "this model+GPU
        # needed offload mode X". In-process only; consulted at every load so
        # a doomed fully-resident attempt is never paid twice (ie#369).
        self.degraded_floor: Dict[str, str] = {}
        # th#913/gw#596: last-applied hub resolutions, keyed by declared wire
        # ref -> (resolved_ref, cast, lane[, lane_pinned]). Per-request lane
        # instructions expand family forms through these picks.
        self._model_resolutions: Dict[str, Tuple[Any, ...]] = {}

    def bind_intent_registry(self, registry: IntentRegistry) -> None:
        self.intent_registry = registry
        self.store.bind_intent_registry(registry)

    def _setup_intent(self, spec: EndpointSpec) -> str:
        registry = self.intent_registry
        if registry is None:
            return ""
        return registry.intent_id(
            pb.DESIRED_INTENT_KIND_FUNCTION_READY,
            function_name=spec.name,
        ) or registry.ensure_local_intent(
            "setup",
            repr(spec.instance_key),
            function_name=spec.name,
            detail=f"prepare function {spec.name}",
        )

    def _job_intent(self, request_id: str, attempt: int, function_name: str) -> str:
        registry = self.intent_registry
        if registry is None:
            return ""
        return registry.ensure_local_intent(
            "job",
            f"{request_id}\0{attempt}",
            function_name=function_name,
            detail=f"run request {request_id} attempt {attempt}",
        )

    def _intent_transition(
        self,
        intent_id: str,
        status: "pb.LifecycleIntentStatus",
        stage: "pb.LifecycleIntentStage",
        **kw: Any,
    ) -> None:
        if self.intent_registry is not None and intent_id:
            self.intent_registry.transition(intent_id, status, stage, **kw)

    async def _intent_await(
        self,
        intent_id: str,
        awaitable: Awaitable[Any],
        *,
        operation: str,
        status: "pb.LifecycleIntentStatus",
        stage: "pb.LifecycleIntentStage",
        reason: "pb.LifecycleWaitReason" = pb.LIFECYCLE_WAIT_REASON_UNSPECIFIED,
        blocker_intent_id: str = "",
        detail: str = "",
    ) -> Any:
        if self.intent_registry is None:
            return await awaitable
        return await self.intent_registry.reported_await(
            intent_id,
            awaitable,
            operation=operation,
            status=status,
            stage=stage,
            reason=reason,
            blocker_intent_id=blocker_intent_id,
            detail=detail,
        )

    @asynccontextmanager
    async def _intent_lock(
        self,
        intent_id: str,
        lock: asyncio.Lock,
        *,
        operation: str,
        stage: "pb.LifecycleIntentStage",
        reason: "pb.LifecycleWaitReason",
        resume_stage: "pb.LifecycleIntentStage",
        blocker_intent_id: str = "",
    ) -> typing.AsyncIterator[None]:
        acquired = False
        try:
            await self._intent_await(
                intent_id,
                lock.acquire(),
                operation=operation,
                status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                stage=stage,
                reason=reason,
                blocker_intent_id=blocker_intent_id,
            )
            acquired = True
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                resume_stage,
            )
            yield
        except asyncio.CancelledError:
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_CANCELED,
                stage,
                detail=f"canceled while waiting: {operation}",
            )
            raise
        finally:
            if acquired:
                lock.release()

    @asynccontextmanager
    async def _setup_singleflight(
        self,
        spec: EndpointSpec,
        rec: "_ClassRecord",
    ) -> typing.AsyncIterator[str]:
        key = spec.instance_key
        blocker_intent_id = self._setup_active.get(key, "")
        registry = self.intent_registry
        if registry is None:
            intent_id = ""
        elif blocker_intent_id:
            task = asyncio.current_task()
            intent_id = registry.ensure_local_intent(
                "setup-waiter",
                f"{repr(key)}\0{id(task)}",
                function_name=spec.name,
                detail=f"waiting to prepare function {spec.name}",
            )
        else:
            intent_id = self._setup_intent(spec)
            self._setup_active[key] = intent_id
        self._intent_transition(
            intent_id,
            pb.LIFECYCLE_INTENT_STATUS_RUNNING,
            pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
        )
        try:
            async with self._intent_lock(
                intent_id,
                rec.lock,
                operation=f"setup single-flight for {spec.name}",
                stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
                reason=pb.LIFECYCLE_WAIT_REASON_SINGLE_FLIGHT_OWNER,
                resume_stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                blocker_intent_id=blocker_intent_id,
            ):
                if intent_id:
                    self._setup_active[key] = intent_id
                yield intent_id
        except BaseException as exc:
            if registry is not None and registry.is_active(intent_id):
                registry.transition(
                    intent_id,
                    (
                        pb.LIFECYCLE_INTENT_STATUS_CANCELED
                        if isinstance(exc, asyncio.CancelledError)
                        else pb.LIFECYCLE_INTENT_STATUS_FAILED
                    ),
                    pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                    detail=_sanitize(str(exc))[:512],
                )
            raise
        finally:
            if self._setup_active.get(key) == intent_id:
                self._setup_active.pop(key, None)

    # ---- precision resolutions (th#697) -----------------------------------

    def apply_model_resolutions(
        self, resolutions: Dict[str, Tuple[Any, ...]],
    ) -> None:
        """Rebind model slots to the hub's precision-ladder picks.

        ``resolutions`` maps a DECLARED wire ref to ``(resolved_ref, cast,
        lane[, lane_pinned])`` — lane is the th#913 concrete execution-lane
        descriptor ("" = unspecified, pre-lane hub); ``lane_pinned``
        (pgw#714, optional for pre-pin hubs) is True when the lane came from
        an operator endpoint-pin, which makes an ``+eager`` execution axis a
        compile kill switch
        (HelloAck full-replace semantics: refs absent from the map revert to
        their authored bindings). Rebinding folds the resolved flavor into
        the binding via :func:`rebind_pick` (THE single fold, shared with the
        local ladder) and stamps ``cast`` as ``storage_dtype``, so every
        downstream consumer — wire_ref residency keys, downloads, setup,
        loading — follows the pick with no per-call-site changes.

        Application is TRANSACTIONAL (gw#494): a ready instance whose loaded
        refs no longer match its (re)bound refs is marked stale and vacated —
        its residency bookings under the OLD resolved refs are released and
        the next setup/LOAD loads the new pick — and the hardware gates +
        serve plans re-run against the rebound bindings.
        """

        self._model_resolutions = dict(resolutions)
        changed = False
        rehomed: List[Tuple[Any, EndpointSpec]] = []
        for spec in self.specs.values():
            declared = self._declared_models.get(spec.name)
            if not declared:
                continue
            key_before = spec.instance_key
            for slot, base_binding in declared.items():
                base_ref = wire_ref(base_binding)
                pick = resolutions.get(base_ref)
                new_binding = base_binding
                if pick is not None:
                    resolved_ref, cast, _execution_lane = pick
                    try:
                        new_binding = rebind_pick(
                            base_binding,
                            resolved_ref=(
                                resolved_ref if resolved_ref != base_ref else ""),
                            cast=cast)
                    except (ValueError, TypeError, AttributeError) as exc:
                        logger.warning(
                            "precision resolution %s -> %r rejected: %s",
                            base_ref, pick, exc)
                        new_binding = base_binding
                if spec.models.get(slot) is not new_binding:
                    spec.models[slot] = new_binding
                    self.store.register_binding(wire_ref(new_binding), new_binding)
                    changed = True
                    if new_binding is not base_binding:
                        logger.info(
                            "precision resolution applied: %s %s/%s -> %s (cast=%s)",
                            spec.name, slot, base_ref, wire_ref(new_binding),
                            getattr(new_binding, "storage_dtype", ""))
            if spec.cls is not None and spec.instance_key != key_before:
                rehomed.append((key_before, spec))
        # spec.instance_key is a live property over spec.models — a rebind
        # above MOVES the spec's key, so the self._classes instance-group
        # record must move with it. Leaving it under the stale key makes
        # every later lookup (state delta, setup, readiness) a KeyError that
        # crash-loops the hello handler (found live: ie#382 dozen lane, the
        # sm90 cast=fp8 pick on a bf16 upsampler killed every worker stream
        # ~1s after HelloAck, churning H100 pods at 60s intervals).
        for old_key, spec in rehomed:
            assert spec.cls is not None  # only cls-specs are rehomed
            rec = self._classes.get(old_key)
            if rec is not None and spec in rec.specs:
                rec.specs.remove(spec)
            new_key = spec.instance_key
            target = self._classes.get(new_key)
            if target is None:
                if rec is not None and not rec.specs:
                    # whole group moved (the common case): carry the record —
                    # and any live instance — to the new key.
                    self._classes.pop(old_key, None)
                    target = rec
                else:
                    target = _ClassRecord(cls=spec.cls)
                self._classes[new_key] = target
            if spec not in target.specs:
                target.specs.append(spec)
            if rec is not None and not rec.specs and self._classes.get(old_key) is rec:
                self._classes.pop(old_key, None)
        if changed:
            # gw#494: transactional application — (1) a ready instance whose
            # loaded refs diverged from the rebound refs is stale: vacate it
            # so nothing stays booked under the old resolved refs (pins,
            # promotions, adapters, and eviction all key off the CURRENT wire
            # refs; a divergent record orphans its VRAM forever).
            stale: List[_ClassRecord] = []
            seen: set = set()
            for rec in self._classes.values():
                if id(rec) in seen:
                    continue
                seen.add(id(rec))
                if not rec.ready or not rec.held_refs:
                    continue
                wanted = {wire_ref(b) for s in rec.specs for b in s.models.values()}
                if set(rec.held_refs) != wanted:
                    rec.stale = True
                    stale.append(rec)
            if stale:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None  # sync context: ensure_setup vacates on next touch
                if loop is not None:
                    for rec in stale:
                        loop.create_task(self._revalidate_record(rec))
            # (2) gates + serve plans re-run against the rebound bindings
            # (idempotent; also settles the startup()-vs-HelloAck order race).
            if self._last_gpu_info is not None:
                self.gate_functions(self._last_gpu_info)
            self._on_state_change()

    async def _revalidate_record(self, rec: "_ClassRecord") -> None:
        """Vacate a stale instance (gw#494): its pipelines were loaded for a
        superseded pick, so release the residency bookings under the OLD
        resolved refs; the next setup/LOAD loads the current pick. Records
        with jobs in flight are left for ``ensure_setup`` to vacate on the
        next touch."""
        async with rec.lock:
            if not rec.ready or not rec.stale:
                return
            async with self._load_lock:
                if record_in_use(rec, records=self._classes.values(), jobs=self.jobs.values(), residency=self.store.residency):
                    return
                await vacate_record(rec, self.teardown_seam)
        self._on_state_change()

    async def revalidate_snapshot_identity(
        self, ref: WireRef, snapshot: Optional[pb.Snapshot],
    ) -> None:
        """Vacate ready instances built from an older digest of the same ref.

        Desired disk preposition runs before any hot-instance request, so this
        must work even when no DesiredInstance follows. Otherwise the worker
        keeps reporting old RAM/VRAM bytes and the hub waits forever for the
        newer ON_DISK identity it requested.
        """
        wanted = self.store.snapshot_digest(ref, snapshot)
        if not wanted:
            return
        stale: List[_ClassRecord] = []
        seen: set[int] = set()
        for rec in self._classes.values():
            if id(rec) in seen:
                continue
            seen.add(id(rec))
            if (
                rec.ready
                and ref in rec.held_refs
                and rec.held_snapshot_digests.get(ref) != wanted
            ):
                rec.stale = True
                stale.append(rec)
        for rec in stale:
            await self._revalidate_record(rec)

    # ---- availability ----------------------------------------------------

    def gate_functions(self, facts: hostfacts.HostFacts) -> None:
        """Run hardware gates; populate self.unavailable + self.serve_plans.

        th#683 P3, as th#1867 left it — the worker NEVER refuses a function
        because of a card's SIZE, and after §1.35 there is no size input left
        for it to refuse on. Exactly TWO gates survive here, and both name OUR
        code: a quant library this IMAGE does not carry
        (``missing_cuda_library``), and no CUDA device at all
        (``cuda_unavailable`` — the owned pgw#1212 exception, see
        ``models/serve_fit``). Everything else serves by the best available
        means, and WHICH means is measured at load time by
        ``models/memory.select_auto_mode`` rather than predicted here. Needing
        offload is NEVER a refusal (Paul's ruling 2026-07-10: gen workers
        offload out of necessity, not preference — better to run degraded than
        not run). Every degraded serve is reported structurally (FnDegraded)
        as evidence for the OPERATOR; nothing on this path sizes a card.
        """

        # Idempotent re-gate (gw#494): drop only the marks THIS gate made
        # last time; setup failures and other owners survive. Remember the
        # probe so apply_model_resolutions can re-run us.
        self._last_gpu_info = facts
        for fn in self._gate_owned:
            self.unavailable.pop(fn, None)
        self._gate_owned = set()

        total_vram_gb = float(facts.vram_total_bytes) / (1024 ** 3)
        # pgw#940: no substitution. `or gpu_total_mem` treated a legitimate 0
        # as "absent" and replaced it with the largest plausible number — and
        # `gpu_free_mem` is genuinely 0 on a SATURATED card, which is exactly
        # the state where the native/fp8/4-bit/offload/CPU ladder must engage
        # and exactly the state that then read as "all of VRAM is free". This
        # figure feeds what the pod advertises
        # to the hub, so an unmeasured card must present as no room, not as an
        # empty one. `lifecycle.probe_hardware` initialises the key to 0 and
        # wraps its whole CUDA probe in `except Exception: pass`, so "the
        # probe raised" arrives here indistinguishable from "the card is
        # full" — both are non-permissive now, which is the point.
        free_vram_gb = float(facts.vram_free_bytes) / (1024 ** 3)
        detected_sm = facts.gpu_sm
        libs = set(facts.installed_libs)
        caps = TensorhubWorkerCapabilities(
            cuda_version=facts.cuda_version,
            gpu_sm=int(detected_sm) if detected_sm.isdigit() else 0,
            torch_version=facts.torch_version,
            installed_libs=list(libs),
        )
        # pgw#676: per-pod native-crash streaks (SIGSEGV & friends recorded
        # by the supervisor/boot-record post-mortem). A function that keeps
        # killing the PROCESS on this card is refused here — loudly, typed —
        # so its siblings keep serving instead of the whole pod crash-looping
        # into th#878's wedge terminate. Per-SKU-instance by construction:
        # the registry lives on the pod's container fs.

        crash_streaks = postmortem.native_crash_streaks()
        # pgw#714: a previous PROCESS DEATH attributed to a background
        # compile disables COMPILING on this pod, not serving — the pod
        # reboots into eager-only instead of re-running the native crash
        # (and instead of refusing the serving function / condemning the
        # SKU for a software bug, th#1226/th#1236).
        compile_rows = postmortem.compile_crash_rows()
        if compile_rows:
            worst = max(
                compile_rows.items(),
                key=lambda kv: int((kv[1] or {}).get("count") or 0))
            from . import compile_cache as _cc_gate

            _cc_gate.disable_process_compiles(
                f"{int((worst[1] or {}).get('count') or 0)} process signal "
                f"death(s) during background compile on this pod "
                f"(last={worst[0]}, "
                f"signal={(worst[1] or {}).get('last_signal') or 'unknown'})")
        for name, spec in self.specs.items():
            r = spec.resources
            # pgw#778: never ADVERTISE what dispatch will refuse. The
            # multi-group async-handler refusal used to fail every request
            # JOB_STATUS_INVALID — the status that means the CALLER's input was
            # bad, so the hub neither retried elsewhere nor charged the worker
            # — while the function stayed in available_functions and the hub
            # kept routing to it. A wide pod with an async endpoint became a
            # 100%-INVALID black hole blamed on the caller. Withdrawing it here
            # is the same seam every other hardware refusal uses, so the hub
            # re-packs or routes elsewhere.
            group_refusal = self._multi_group_handler_refusal(spec)
            if group_refusal:
                logger.warning(
                    "withdrawing %r on this pod: %s", name, group_refusal)
                self.unavailable[name] = (
                    "multi_group_async_handler", group_refusal,
                    {"execution_groups": str(self.topology.execution_groups),
                     "degree": str(self.topology.degree)})
                self._gate_owned.add(name)
                continue
            streak_row = crash_streaks.get(name)
            streak = int((streak_row or {}).get("count") or 0)
            if streak >= postmortem.NATIVE_CRASH_REFUSE_STREAK:
                detail = (
                    f"{streak} worker-process signal death(s) mid-"
                    f"{(streak_row or {}).get('last_kind') or 'execution'} on "
                    f"this pod (last={((streak_row or {}).get('last_signal')) or 'unknown'}); "
                    "refusing this function on this hardware — siblings keep "
                    "serving (pgw#676 degrade-never-die across process death)"
                )
                logger.error(
                    "NATIVE CRASH STREAK: function %r disabled on this pod — %s",
                    name, detail)
                self.unavailable[name] = (
                    "native_crash_streak", detail,
                    {"streak": str(streak),
                     "last_signal": str(
                         (streak_row or {}).get("last_signal") or ""),
                     # pgw#714/th#1236: crash phase, so the hub can spare
                     # the SKU table when the death was not a serving
                     # forward.
                     "last_kind": str(
                         (streak_row or {}).get("last_kind") or "")})
                self._gate_owned.add(name)
                continue
            # SDK v2 (pgw#647): NO compute-capability gate — precision per
            # card class is the fit ladder's decision (sdxl runs fine in fp16
            # on sm75). pgw#1148 moved the stored-flavor SM windows onto the
            # loaders' tensor-layout contracts, so nothing SM-shaped is left
            # in this gate at all.
            missing = [lib for lib in (r.libraries or ()) if lib not in libs]
            if missing:
                import importlib.util
                missing = [m for m in missing if importlib.util.find_spec(m) is None]
            if missing:
                self.unavailable[name] = (
                    "missing_cuda_library", f"missing required libraries: {', '.join(missing)}",
                    {"missing": ",".join(missing)})
                self._gate_owned.add(name)
                continue

            # Serve-time plan. th#1867: this no longer asks a size question —
            # `plan_serve` returns non-serveable ONLY for the two gates that
            # name our own code.
            primary = next(iter(spec.models.values()), None)
            plan = plan_serve(r, caps, free_vram_gb, binding=primary)
            self.serve_plans[name] = plan
            self._gate_serve_plans[name] = plan
            if not plan.serveable:
                # After th#1867 the planner has ONE non-serveable verdict left
                # and it is a library one — our IMAGE is short a dependency.
                # The `missing_cuda_library` gate above normally catches it
                # first (it re-checks with importlib), so reaching here means
                # the two library views disagreed. Report it under the same
                # honest token rather than inventing a card verdict for it:
                # naming a GPU problem that does not exist is exactly what
                # `compute_capability_unmet` was deleted for (§2.7).
                self.unavailable[name] = (
                    "missing_cuda_library", plan.reason,
                    {"detected_vram_gb": f"{total_vram_gb:.0f}"})
                self._gate_owned.add(name)
                continue
            if plan.degraded:
                logger.warning(transition_line(
                    event="planned", fn=name, phase="gate",
                    from_rung=plan.wanted, to_rung=plan.ran or plan.run_mode,
                    free_gb=free_vram_gb,
                    detail=f"~{plan.est_latency_multiplier:.1f}x latency: {plan.warning}",
                ))

    def _record_rung_transition(
        self,
        spec: EndpointSpec,
        *,
        ref: str,
        phase: str,
        from_rung: str = "",
        to_rung: str = "",
        run_mode: str = "",
        wanted: str = "",
        ran: str = "",
        needed_gb: float = 0.0,
        detail: str,
    ) -> None:
        """THE ladder-transition bookkeeper (pgw#1206 A2; folds gw#463
        demotion, gw#491 load-rung engagement and th#737 cast-drop): learned
        per-ref placement floor + updated ServePlan via serve_fit.replan +
        loud DEGRADED_MODE line + FnDegraded re-emit via the state-delta
        path — never a log-line-only fallback."""

        if ref and rungspec.touches_host_ram(to_rung):
            self.degraded_floor[ref] = rungspec.floor_of(
                self.degraded_floor.get(ref, ""), to_rung)
        free_gb = get_available_vram_gb() if (from_rung or to_rung) else 0.0
        # th#1871 P1: this is the ONE place every ladder transition passes
        # through, so it is the one place the typed posture is written. The
        # numbers were already being computed for the log line and then thrown
        # away — `needed_gb` and the live free VRAM ARE the §1.36 shortfall
        # ("needed N, had M, short by N-M"), and prose was the only thing
        # carrying them off this pod.
        self._record_posture_transition(
            spec, ref=ref, run_mode=run_mode, to_rung=to_rung,
            wanted=wanted, ran=ran, needed_gb=needed_gb, free_gb=free_gb)
        line = transition_line(
            event="engaged", fn=spec.name, model=ref, phase=phase,
            from_rung=from_rung, to_rung=to_rung or run_mode,
            needed_gb=needed_gb,
            free_gb=free_gb,
            detail=detail,
        )
        logger.warning(line)
        self.serve_plans[spec.name] = replan(
            self.serve_plans.get(spec.name),
            run_mode=run_mode, wanted=wanted, ran=ran, detail=line,
        )
        self._on_state_change()

    def _posture_ledger(
        self, spec: EndpointSpec,
    ) -> "Optional[posture_mod.PostureLedger]":
        """This spec's instance ledger, or None when there is no record yet.

        None is not an error: a transition can fire before the record exists
        (a refusal during the very first load), and inventing a ledger to hold
        it would attribute a lever to an instance that never served."""
        rec = self._classes.get(spec.instance_key)
        return None if rec is None else rec.posture

    def _record_posture_transition(
        self, spec: EndpointSpec, *, ref: str, run_mode: str, to_rung: str,
        wanted: str, ran: str, needed_gb: float, free_gb: float,
    ) -> None:
        """Project one ladder transition onto the typed posture.

        The projection is deliberately NOT the wire's `run_mode`: that token is
        coarse by design (`offload` covers three rungs whose prices differ by
        60%), and th#1871 §6.6 item 5 is precisely that those three stop sharing
        it. The named rung wins whenever the transition named one.
        """
        ledger = self._posture_ledger(spec)
        if ledger is None:
            return
        technique = posture_mod.technique_for_run_mode(run_mode, to_rung)
        if technique:
            ledger.technique(
                technique,
                # A cast that was ASKED FOR and one that was FORCED are
                # different postures even when the applied value matches, so
                # `wanted` rides the lever rather than being reconciled away.
                wanted=str(wanted or ""),
                reason=(posture_mod.REASON_LANE_CAST_DROPPED
                        if wanted and ran and wanted != ran
                        else posture_mod.REASON_VRAM_SHORTFALL))
        if to_rung:
            ledger.residency(to_rung)
        if needed_gb > 0.0:
            ledger.shortfall(posture_mod.ResourceShortfall.from_gb(
                posture_mod.RESOURCE_VRAM, needed_gb, free_gb,
                component=str(ref or "")))

    def _stamp_posture(
        self, metrics: "pb.JobMetrics", spec: EndpointSpec,
        served: "serving_mode_mod.ServedIdentity", lane: str, *,
        instructed: str = "", compile_required: bool = False,
    ) -> None:
        """Stamp the typed posture on one terminal ``JobMetrics``.

        THE ONE THING THIS MUST NEVER DO is send an empty posture. An all-empty
        record does not mean "clean" — it means nobody looked — and the hub keys
        the two differently on purpose (`endpoint_measurements`' unreported
        posture has its own digest). Claiming a clean posture over a worker that
        never observed one is ie#707 with the polarity flipped: instead of a
        degraded run filed as clean, a silent run filed as measured.
        """
        ledger = self._posture_ledger(spec)
        if ledger is None:
            return
        posture = ledger.snapshot(
            execution_lane=lane,
            compile_state=posture_mod.compile_axis(served.serving_mode),
            # What the lane DECLARED, off the hub's own dispatch instruction —
            # never off `lane`, which is composed from what actually ran and
            # would make every run trivially self-consistent. This is the axis
            # that made minimax-h3's declared-compiled/ran-eager hours
            # unrepresentable.
            compile_state_wanted=(
                posture_mod.compile_axis_of_lane(instructed or "")
                # The two hub paths, in the order of specificity. An instructed
                # lane states the axis outright; a `required_compile` fence says
                # `compiled` without naming a lane at all, and on the
                # ModelResolution path it is the only thing that says it.
                or (posture_mod.COMPILE_COMPILED if compile_required else "")),
        )
        if not posture.observed:
            return
        metrics.posture.CopyFrom(posture.to_proto())
        if posture.degraded:
            # §1.36's amendment, verbatim: *"the worker should obviously
            # complain loudly if it has to use a bunch of optimization
            # techniques"*. The typed record is what a DECISION reads; this line
            # is what a human reads, and it is derived from the same object so
            # the two cannot say different things. Never a gate — the request
            # already succeeded by the time this runs.
            logger.warning(
                "serve-posture: DEGRADED fn=%s %s",
                spec.name, posture_mod.summarize(posture))

    def _record_placement_posture(
        self, spec: EndpointSpec, *, ref: str, placed: Dict[str, Any],
    ) -> None:
        """Project one PLACEMENT onto the typed posture — proactive or not.

        `memory.place_pipeline` answers with the rung it actually used, whether
        it chose that rung up front against free VRAM or descended into it on a
        CUDA OOM. Both are degradations of the same kind and both key the same
        measurement; only the `reason` differs, and the reason is a field rather
        than the difference between reporting and silence.
        """
        ledger = self._posture_ledger(spec)
        if ledger is None:
            return
        mode = str(placed.get("mode") or "")
        ledger.residency(mode)
        reactive = bool(placed.get("oom_demotions"))
        technique = posture_mod.residency_for_placement(mode)
        if technique and technique != posture_mod.RESIDENCY_ALL_RESIDENT:
            ledger.technique(
                technique, component=str(ref or ""),
                # The rung ASKED FOR, when a descent moved off it. Absent on a
                # proactive selection: nothing was asked, the fit decided.
                wanted=str(placed.get("requested_mode") or ""),
                reason=(posture_mod.REASON_CUDA_OOM if reactive
                        else posture_mod.REASON_VRAM_SHORTFALL))
        if mode == "vae_only":
            # Resident, but NOT the same run: slicing changes the traced decode
            # graph and the decode's cost. A lever with no wire name at all
            # until now (§6.6 item 5).
            ledger.technique(
                posture_mod.TECHNIQUE_VAE_SLICING, component="vae",
                reason=posture_mod.REASON_VRAM_SHORTFALL)
        if mode == "cpu":
            ledger.technique(
                posture_mod.TECHNIQUE_CPU, component=str(ref or ""),
                reason=posture_mod.REASON_NO_CUDA)
        needed_gb = float(placed.get("fit_needed_gb") or 0.0)
        if needed_gb > 0.0:
            ledger.shortfall(posture_mod.ResourceShortfall.from_gb(
                posture_mod.RESOURCE_VRAM, needed_gb,
                float(placed.get("fit_available_gb") or 0.0),
                component=str(ref or "")))

    def available_functions(self) -> List[str]:
        out = []
        for name, spec in self.specs.items():
            if name in self.unavailable or self.draining:
                continue
            if spec.cls is None:
                out.append(name)
                continue
            if spec.slots:
                # pgw#532 dynamic slots: the hub owns the slot's model set —
                # serveability is per-dispatch (RunJob carries the resolved
                # refs + snapshots; setup materializes THEM, never the code
                # seed). Gate only on hardware/setup failures, never on a
                # resident instance the worker cannot create by itself.
                out.append(name)
                continue
            if spec.kind != "inference":
                # ie#522 finding: non-inference (e.g. conversion) functions
                # are dispatched and set up per-request — _warmup_plan (see
                # below) never schedules them for boot warmup, so a declared-
                # but-never-yet-dispatched conversion function must not read
                # as "awaiting readiness" (that bucket is loading_functions()
                # below, which th#965 layer 3 stall-watches: a conversion
                # function that simply hasn't been invoked yet was tripping
                # a 10m worker_activity_stalled kill on an otherwise-healthy
                # worker mid-job, taking down unrelated in-flight requests).
                out.append(name)
                continue
            rec = self._classes[spec.instance_key]
            if rec.ready or (not spec.models and rec.failed is None):
                out.append(name)
        return sorted(out)

    def loading_functions(self) -> List[str]:
        avail = set(self.available_functions())
        return sorted(
            name for name, spec in self.specs.items()
            if name not in avail and name not in self.unavailable
            and spec.cls is not None and not spec.slots
            and spec.kind == "inference"
            and self._classes[spec.instance_key].failed is None
        )

    @staticmethod
    def _refresh_compile_target(target: _CompileTargetRecord) -> None:
        """Refresh compatibility evidence after an in-place lane mutation."""

        cfg = target.spec.compile_cell()
        assert cfg is not None
        contract_digest = compile_cache.execution_contract_digest(
            target.pipeline, cfg)
        execution_lane = pipeline_weight_lane(target.pipeline)
        bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
        with target.state_lock:
            target.pipeline_weight_lane = execution_lane
            target.lora_bucket = bucket
            target.contract_digest = contract_digest

    def _compile_guard_failed(
        self,
        rec: _ClassRecord,
        target: _CompileTargetRecord,
        detail: str,
    ) -> None:
        """Synchronously revoke compiled proof before a runtime fallback.

        The target remains addressable with an empty active identity so the
        causal failure can be correlated regardless of event/StateDelta order.
        pgw#672: mandatory (w8a8/w4a4) lanes no longer disable their aliases
        or force a reload here — the guard wrapper degrades the object to
        explicit eager serving and this revocation flips the wire tier; the
        failed identity is quarantined process-wide so it is never re-adopted
        or re-minted this boot (`fleet_cells` reads that quarantine on the arm
        path).

        pgw#1032: the per-target `failed_compile_identities` set and the causal
        `adopt_failed:runtime_guard` ModelEvent are gone with the hub-commanded
        adoption they answered — both were fed only by the ADOPT_COMPILE_CACHE
        handler, which no stack has ever dispatched. The revocation itself is
        unchanged and still rides the StateDelta tier flip.
        """
        if rec.compile_targets.get(target.incarnation_id) is not target:
            raise RuntimeError("compiled target is no longer live")
        # pgw#1082: classify BEFORE the active-ref gate below. A JIT INTAKE
        # arm names no artifact by construction (pgw#1010), so that gate
        # returns early for exactly the lane this defect lives on — and the
        # graph-broken pod would keep reporting an empty `fallback_reason`.
        broke = compile_cache.graph_break_reason(target.pipeline)
        out_of_range = compile_cache.declared_range_refusal(target.pipeline)
        if broke:
            self._note_eager_posture(
                rec, cell_adopt.EagerPhase.GRAPH_BREAK.value,
                f"the declared regional target did not trace WHOLE under "
                f"fullgraph; serving eager rather than eager-glued fragments "
                f"reported as compiled: {broke}")
        elif out_of_range:
            self._note_eager_posture(
                rec, cell_adopt.EagerPhase.DECLARED_RANGE_EXCEEDED.value,
                out_of_range)
        elif detail:
            # pgw#1082: ANY permanent guard degrade must reach the request
            # row. The gate below returns early for a JIT INTAKE arm (it
            # names no artifact by construction, pgw#1010) — which is the
            # only JIT lane the fleet has — so before this, a degraded
            # intake pod recorded nothing at all and served eager silently.
            self._note_eager_posture(
                rec, cell_adopt.EagerPhase.JIT_ARM_FAILED.value, detail)
        with target.state_lock:
            if not (
                target.active_compile_ref
                or target.active_compile_snapshot_digest
            ):
                return
            failed_ref = target.active_compile_ref
            target.active_compile_ref = ""
            target.active_compile_snapshot_digest = ""

        compile_cache.record_compiled_graph_quarantined(failed_ref)
        logger.warning(
            "compile target %s runtime guard tripped; compiled proof revoked, "
            "serving degrades to explicit eager: %s",
            target.incarnation_id,
            detail,
        )
        self._signal_state_change_threadsafe()

    def _mark_compile_target_unavailable(
        self,
        rec: _ClassRecord,
        target: _CompileTargetRecord,
        detail: str,
    ) -> None:
        """Disable every alias owned by one failed mandatory compile target."""
        self._mark_compile_names_unavailable(
            rec, target.function_names, target.incarnation_id, detail)

    def _mark_compile_setup_unavailable(
        self, rec: _ClassRecord, spec: EndpointSpec, detail: str,
    ) -> None:
        """Fail loud for every handler requiring the unproven W8A8 setup."""
        names = self._required_compile_names(spec, rec) or {spec.name}
        self._mark_compile_names_unavailable(rec, names, "", detail)

    def _mark_compile_names_unavailable(
        self,
        rec: _ClassRecord,
        names: typing.Iterable[str],
        target_incarnation_id: str,
        detail: str,
    ) -> None:
        sanitized = _sanitize(detail)
        for name in names:
            existing = self.unavailable.get(name)
            owner = self._compile_failure_owners.get(name)
            if existing is not None and (
                existing[0] != "compile_cell_failed"
                or owner is None
                or owner[0] is not rec
                or owner[1] != target_incarnation_id
            ):
                # Never erase a hardware/setup disable or another target's
                # ownership merely because this target also named the alias.
                continue
            # pgw#1278: kept spelling. `FnUnavailable.reason` is a CLOSED
            # vocabulary enumerated in the proto, so this token moves with the
            # proto lane, not with the route/event rename.
            self.unavailable[name] = (
                "compile_cell_failed", sanitized, {},
            )
            self._compile_failure_owners[name] = (
                rec, target_incarnation_id,
            )

    def _clear_recovered_compile_failures(self, rec: _ClassRecord) -> None:
        """Re-advertise only aliases proven by a fresh active target."""
        recovered: set[str] = set()
        for target in rec.compile_targets.values():
            with target.state_lock:
                if (
                    target.active_compile_ref
                    and target.active_compile_snapshot_digest
                ):
                    recovered.update(target.function_names)
        for name in recovered:
            owner = self._compile_failure_owners.get(name)
            unavailable = self.unavailable.get(name)
            if (
                owner is not None
                and owner[0] is rec
                and unavailable is not None
                and unavailable[0] == "compile_cell_failed"
            ):
                self.unavailable.pop(name, None)
                self._compile_failure_owners.pop(name, None)

    def _abandon_pending_mint(self, inj: "_InjectionResult", pipe: Any) -> None:
        """Discard an unfinalized self-mint capture for ``pipe`` (gw#587
        CORRECT FIX): a disproven/unexercised candidate's capture must never
        be packed or published — only a passed proof produces the mint."""
        pending = inj.pending_self_mints.pop(id(pipe), None)
        if pending is not None:

            fleet_cells_mod.abandon_self_mint(pending)

    def _bind_compile_guard(
        self, rec: _ClassRecord, target: _CompileTargetRecord,
    ) -> bool:
        """Bind one live wrapper's first failure to exact target revocation."""

        # pgw#677: every shape-warm/heal compile for this pipeline must run
        # inside a background GPU turn (yield to tenant demand; mutual
        # exclusion with tenant forwards on this instance).
        self._wire_turn_gate(rec, target.pipeline)

        def callback(detail: str) -> None:
            self._compile_guard_failed(rec, target, detail)

        # pgw#844: the EXPORTED lane owns its own revocation signal and this
        # was never asked for it — `enable_compiled` returns as soon as
        # `arm_aot` succeeds, so an AOT-armed pipeline never gets the dynamo
        # `failure_signal` marker `compile_cache.set_guard_failure_callback`
        # reads. Every aot arm therefore answered "no runtime guard
        # revocation signal", had its `active_compile_ref` cleared, and
        # advertised eager — a compiled AOT serve was structurally
        # unreachable on the boot path regardless of dispatch. Ask the lane
        # that is actually armed.
        if aot_serve.set_guard_failure_callback(target.pipeline, callback):
            def refusal_callback(reason: str, detail: str) -> None:
                self._compile_ingress_refused(target, reason, detail)

            aot_serve.set_ingress_refusal_callback(
                target.pipeline, refusal_callback)
            return True
        if not compile_cache.set_guard_failure_callback(
            target.pipeline, callback,
        ):
            return False
        # pgw#680: serve-window guard misses confess through the same
        # target (telemetry only — no state mutation, no revocation).
        def miss_callback(miss: compile_cache.GuardMiss) -> None:
            self._compile_guard_missed(rec, target, miss)

        compile_cache.set_guard_miss_callback(target.pipeline, miss_callback)
        return True

    def _compile_guard_missed(
        self,
        rec: _ClassRecord,
        target: _CompileTargetRecord,
        miss: "compile_cache.GuardMiss",
    ) -> None:
        """pgw#680 confession: one tenant request hit fail-on-recompile.

        Pure observability — the compiled identity stays active, the tier
        stays compiled (the lane is healthy for its known input classes;
        THIS class is healing in background). The typed event rides the
        activity stream so the hub can count misses per (release, SKU,
        guard-reason): kind=guard_miss, phase=reason class, detail=the
        verbatim torch reason + shape identity + cell key + request id."""

        with target.state_lock:
            cell = target.active_compile_ref
            digest = target.active_compile_snapshot_digest
        reason_class = compile_cache.guard_miss_reason_class(miss.reason)
        request_id = postmortem.current_inflight_request()
        detail = (
            f"fn={sorted(target.function_names)} target={miss.target} "
            f"cell={cell or '<none>'} digest={digest[:16] or '<none>'} "
            f"request={request_id or '<unknown>'} heal={miss.heal} "
            f"miss_n={miss.misses} sig={miss.sig[:400]} "
            f"reason={miss.reason}"
        )
        logger.warning(
            "guard_miss (pgw#680): compiled %s served eager for request %s "
            "— reason class %r, heal=%s, cell=%s",
            miss.target, request_id or "<unknown>", reason_class, miss.heal,
            cell or "<none>",
        )
        activity_mod.emit_event(
            activity_mod.KIND_GUARD_MISS, detail, phase=reason_class)
        # pgw#789: charge the fallback to THIS request's JobMetrics. The event
        # above makes the miss countable per (release, SKU, reason); this makes
        # the LATENCY SAMPLE honest, which is a different question — an eager
        # sample tagged serving_mode=aot_cell argues against the optimization
        # that is in fact working. `heal` is the router's own verdict
        # (healing = transient, volatile = permanently eager for this shape),
        # so it outranks the generic guard_miss class.
        self._mark_request_eager_fallback(
            request_id,
            (miss.heal if miss.heal in (
                serving_mode_mod.FALLBACK_HEALING,
                serving_mode_mod.FALLBACK_VOLATILE) else
             serving_mode_mod.FALLBACK_GUARD_MISS),
        )

    def _compile_ingress_refused(
        self, target: _CompileTargetRecord, reason: str, detail: str,
    ) -> None:
        """pgw#844: one tenant request was refused at the artifact's ingress
        and served eager by an ARMED compiled lane.

        The typed `aot_ingress_refused` event already counts the refusal per
        (release, SKU, reason). This charges the same fact to THIS request's
        JobMetrics, which is the different question: an eager latency sample
        tagged `serving_mode=aot_cell` argues against the optimization that is
        working for every other shape. It matters now precisely because a
        partially dispatchable cell stays armed instead of costing the pod its
        whole compiled lane — the eager shapes must be subtractable from the
        compiled measurement, by name.

        Observability only: the artifact stays armed and the target keeps its
        active identity, exactly as the per-call refusal contract says.
        """

        request_id = postmortem.current_inflight_request()
        if not request_id:
            return
        logger.debug(
            "aot ingress refusal charged to request %s (%s): %s",
            request_id, reason, detail[:200])
        self._mark_request_eager_fallback(
            request_id, serving_mode_mod.FALLBACK_INGRESS_REFUSED)

    def _mark_request_eager_fallback(self, request_id: str, reason: str) -> None:
        """Record that ``request_id`` was served eager by a compiled lane.

        Called from a HANDLER thread (the guard-miss callback), so it only
        writes two plain fields on the job — no locks, no event-loop hop. A
        request id that names no live job (a background warm compile, a raced
        terminal) is dropped: there is no sample to correct.
        """
        if not request_id:
            return
        for job in list(self.jobs.values()):
            if job.request_id != request_id or job.finished:
                continue
            job.served_eager_fallback = True
            # First reason wins: `volatile` is a terminal verdict and must not
            # be downgraded to `guard_miss` by a later miss on the same request.
            if not job.fallback_reason:
                job.fallback_reason = reason

    def _note_eager_posture(
        self, rec: _ClassRecord, token: str, detail: str = "",
        *, override: bool = False,
    ) -> None:
        """pgw#824: record (and, once, confess) WHY this record has no cell.

        First token wins: the earliest honest cause outranks a later generic
        one. The typed event fires only on the transition, so a decline that
        happens per-object on a many-slot record coalesces to ONE row instead
        of N identical ones — counts, not silence, and not a flood.

        pgw#1093 ``override``: for the ONE token that is never a competing
        cause but the CAUSE OF the causes already recorded. "An arm returned
        True and nothing owns it" makes "no compile candidates survived" a
        consequence, not a rival — and first-token-wins would otherwise leave
        the record naming its own symptom.
        """
        token = str(token or "").strip()
        if not token or (rec.eager_posture and not override):
            return
        if override and rec.eager_posture == token:
            return
        rec.eager_posture = token
        activity_mod.emit_event(
            "serve_eager_posture",
            f"fn={','.join(s.name for s in rec.specs) or '?'}: this instance "
            f"serves EAGER — {detail or token}. Every request it serves "
            f"reports fallback_reason={token}.",
            phase=token,
        )

    def _note_boot_degrade(self, rec: _ClassRecord, pipeline: Any) -> None:
        """pgw#1093: confess a degrade that happened BEFORE any guard was bound.

        `_bind_compile_guard` installs the revocation callback in
        `_install_compile_targets`, which runs AFTER the boot warmup. So a
        target that armed, compiled, and then broke DURING that warmup has no
        callback to fire and no record to write on: pgw#1082's whole
        confession path is unreachable for it, and every reader afterwards
        falls through to the generic `uncompiled`. This reads the reason
        straight off the pipeline's own failure signal instead.
        """
        broke = compile_cache.graph_break_reason(pipeline)
        out_of_range = compile_cache.declared_range_refusal(pipeline)
        reason = compile_cache.degrade_reason(pipeline)
        if broke:
            self._note_eager_posture(
                rec, cell_adopt.EagerPhase.GRAPH_BREAK.value,
                f"the declared regional target did not trace WHOLE under "
                f"fullgraph during the boot warmup: {broke}")
        elif out_of_range:
            self._note_eager_posture(
                rec, cell_adopt.EagerPhase.DECLARED_RANGE_EXCEEDED.value,
                out_of_range)
        elif reason:
            self._note_eager_posture(
                rec, cell_adopt.EagerPhase.COMPILED_DEGRADED.value,
                f"the compiled target was ARMED and a call during the boot "
                f"warmup failed permanently, so this instance is eager for "
                f"the rest of its life: {reason}")

    def _assert_armed_targets_installed(
        self, rec: _ClassRecord, spec: EndpointSpec,
        armed_objects: typing.Iterable[Any],
    ) -> None:
        """pgw#1093 THE INVARIANT: armed at setup => an installed target owns it.

        "``compile_cache`` minted graphs into this object and nothing on this
        record can dispatch to them" is not a degraded state — it is an
        IMPOSSIBLE one, and it has now cost two releases (pgw#1078 D2, this
        issue) because every route to it was a log line or a bare ``continue``.

        Keyed on the ARM FACT — what ``arm_compile()``/the injection scan
        RETURNED — never on a live ``is_compile_armed()`` probe. A permanent
        degrade flips that probe to False, so probing would let the invariant
        excuse exactly the boot it exists to catch.

        Not fatal (pgw#672: a broken optimization never kills a serving
        worker). It is a TYPED, wire-visible refusal with a named cause, which
        is what the pod telemetry did not have.
        """
        owned = {id(t.pipeline) for t in rec.compile_targets.values()}
        orphans = [p for p in armed_objects if id(p) not in owned]
        if not orphans:
            return
        detail = "; ".join(
            f"{type(p).__name__} armed={compile_cache.is_compile_armed(p)} "
            f"targets_resolve="
            f"{compile_cache.has_compile_target(p, spec.compile_cell())} "
            f"degrade={compile_cache.degrade_reason(p) or '-'}"
            for p in orphans
        )
        logger.error(
            "%s: %d ARMED compile object(s) own NO installed target — this "
            "boot compiled graphs nothing can dispatch to (%s)",
            spec.name, len(orphans), detail)
        self._note_eager_posture(
            rec, cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value,
            f"{len(orphans)} object(s) armed during setup own no installed "
            f"compile target, so the compiled graphs this boot paid for are "
            f"undispatchable: {detail}",
            override=True)
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            detail=(
                f"fn={spec.name}: {len(orphans)} ARMED compile object(s) own "
                f"no installed target after setup — {detail}"),
            phase=cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value,
        )

    def _install_compile_targets(
        self,
        rec: _ClassRecord,
        spec: EndpointSpec,
        objects: typing.Iterable[Any],
        active_artifacts: Optional[Dict[int, _CompileArtifactSelection]] = None,
        function_proofs: Optional[Dict[int, set[str]]] = None,
    ) -> None:
        """Mint one incarnation for every compile-capable object just set up."""

        cfg = spec.compile_cell()
        rec.compile_targets = {}
        if cfg is None:
            return
        # pgw#775: no targets at degree>1. A target is what routes a novel
        # signature to the shared warm thread, which forwards the compile-
        # capable OBJECT (not the gated pipeline) on rank 0 alone — the exact
        # forward that hangs the group. No targets means no hot-swap routing,
        # no background mint turn and no adoption for this record.
        eager_only = self._eager_only_reason()
        if eager_only:
            logger.info(
                "%s: no compile targets installed — %s", spec.name, eager_only)
            self._note_eager_posture(
                rec, "no_compile_targets_installed", eager_only)
            return
        # Production injection supplies object-scoped slot ownership. Keep
        # bare objects accepted for focused unit construction only, deriving
        # their ownership from the record's already-frozen held bindings.
        all_slots = {
            slot.partition(".")[0] for slot, _ref, _digest in rec.held_bindings
        }
        candidates = [
            item if isinstance(item, _CompileObjectCandidate)
            else _CompileObjectCandidate(item, set(all_slots))
            for item in objects
        ]
        # pgw#1093: a permanent degrade that happened during the BOOT WARMUP
        # has no record to confess on — `_bind_compile_guard` installs the
        # revocation callback HERE, after the warmup, so pgw#1082's whole
        # confession path is structurally unreachable for a target that broke
        # while it was being warmed. This is the late confession: read the
        # reason off the pipeline's own failure signal, at the first moment a
        # record exists to carry it.
        for candidate in candidates:
            self._note_boot_degrade(rec, candidate.pipeline)
        if not candidates:
            # The candidate loop below never runs, so not one of its omission
            # tokens can fire — the exact shape that reached pgw#1093 as zero
            # rows and a generic `uncompiled` on every request.
            self._note_eager_posture(
                rec, cell_adopt.EagerPhase.NO_COMPILE_CANDIDATES.value,
                f"setup produced no compile-capable object for declared "
                f"family {str(getattr(cfg, 'family', '') or '?')!r} "
                f"targets={[str(t) for t in (getattr(cfg, 'targets', ()) or ())]} "
                f"over held slots {sorted(all_slots)}")
        requested_execution_lane = self._mandatory_execution_lane_of_bound(
            wire_ref(spec.models[slot]) for slot in self._setup_slots(spec)
        )
        active_artifacts = active_artifacts or {}
        function_proofs = function_proofs or {}
        contract_names = self._compile_contract_names(spec, rec)
        required_names = self._required_compile_names(spec, rec)
        seen: set[int] = set()
        for candidate in candidates:
            pipeline = candidate.pipeline
            if pipeline is None or id(pipeline) in seen:
                continue
            seen.add(id(pipeline))
            if not compile_cache.has_compile_target(pipeline, cfg):
                # pgw#1093: NOT a bare `continue` any more. An object reaches
                # this list only because something already decided it was
                # compile-capable — the injection scan or an `arm_compile()`
                # that RETURNED TRUE. Resolving no target here therefore means
                # the object changed under the arm (a lazily-hydrated
                # component replaced after setup, a slot swapped by the
                # endpoint), which is a WIRING fact and has to say so. It was
                # one of exactly three exits from this method that emitted
                # nothing at all, and this issue burned a $1.15 pod because of
                # it.
                self._note_eager_posture(
                    rec, cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value
                    if compile_cache.is_compile_armed(pipeline)
                    else cell_adopt.EagerPhase.NO_COMPILE_TARGET.value,
                    f"{type(pipeline).__name__} owning slots "
                    f"{sorted(candidate.slots)} resolves none of the declared "
                    f"targets "
                    f"{[str(t) for t in (getattr(cfg, 'targets', ()) or ())]} "
                    f"at install time"
                    + (" — yet compile_cache reports it ARMED, so this boot "
                       "compiled graphs it can never dispatch to"
                       if compile_cache.is_compile_armed(pipeline) else ""))
                continue
            bindings = tuple(sorted(
                binding for binding in rec.held_bindings
                if binding[0].partition(".")[0] in candidate.slots
            ))
            bindings_valid = bool(bindings) and all(
                slot.strip() and ref.strip() and digest.strip()
                for slot, ref, digest in bindings
            ) and len({slot for slot, _ref, _digest in bindings}) == len(bindings)
            active_selection = active_artifacts.get(id(pipeline))
            # pgw#1141 (Paul's ruling, 2026-08-11): on the EXPORTED lane the
            # warm ledger GATES NOTHING. It used to decide which aliases an
            # installed target may serve, so an object the boot warmup happened
            # not to dispatch through was handed `permitted_names=set()` ->
            # `function_names=()` -> `target_applicability_incomplete` -> a pod
            # that had just verified its cell at `cos=1.00000` served eager for
            # life. An AOTI artifact is ahead-of-time machine code for this
            # exact sm/toolchain: the first call is full speed, and the warm
            # pass never made it faster — it only checked it. What the cell
            # advertises is what it may serve; a class it does not carry is
            # refused BY NAME at ingress and served eager per request
            # (pgw#844), and a cell-attributable failure revokes the arm
            # in-request through the wrapper's own fallback.
            #
            # The DYNAMO lane keeps the ledger, and the difference is the
            # failure MODE, not the vintage: a dynamo arm that does not serve
            # its cell RECOMPILES — correct output, silently slower, no
            # exception for try-serve to catch and no numerics gate on that
            # lane at all. Its per-class cache-hit ledger is the only detector
            # that exists, so deleting it would remove a detector with no
            # replacement.
            exported_arm = _exported_arm(
                pipeline, active_selection.ref if active_selection else "")
            if exported_arm and not aot_serve.is_armed(pipeline):
                # pgw#1141: the sticky de-arm reaches the INSTALL. The artifact
                # revoked itself (a failed target, a constants fault) before
                # any guard was bound to hear it, so installing its target
                # would advertise `serving_mode=aot_cell` on a pipeline whose
                # every call now runs eager — the wire lie pgw#1082/#1093 spent
                # two pods closing. Under the old barrier the disarm sweep hid
                # this case by unwrapping first; serve-first reaches it, so it
                # is named here.
                detail = (
                    f"{type(pipeline).__name__} owning slots "
                    f"{sorted(candidate.slots)} holds a REVOKED exported cell "
                    f"({active_selection.ref if active_selection else '?'}): "
                    f"the artifact de-armed itself during boot, so every call "
                    f"serves eager and no compiled target may advertise it")
                logger.warning("compile target omitted for %s: %s",
                               spec.name, detail)
                self._note_eager_posture(
                    rec, cell_adopt.EagerPhase.COMPILED_DEGRADED.value, detail)
                continue
            permitted_names = (
                contract_names if exported_arm
                else function_proofs[id(pipeline)]
                if id(pipeline) in function_proofs
                else contract_names
            )
            object_proven_by_custom_warmup = bool(
                not exported_arm
                and spec.cls is not None
                and callable(getattr(spec.cls, "warmup", None))
                and function_proofs.get(id(pipeline))
            )
            incarnation_id = uuid.uuid4().hex
            target = _CompileTargetRecord(
                incarnation_id=incarnation_id,
                spec=spec,
                pipeline=pipeline,
                pipeline_weight_lane="",
                lora_bucket=0,
                contract_digest="",
                model_bindings=bindings,
            )
            self._refresh_compile_target(target)
            # Aliases apply only when they address this exact object through
            # the same owned slots and share its graph/lane contract. A class
            # sibling with a different checkpoint may share Python code but
            # cannot inherit this target's immutable applicability.
            compatible_names: set[str] = set()
            for alias in rec.specs:
                alias_cfg = alias.compile_cell()
                if alias_cfg is None:
                    continue
                if (
                    str(getattr(alias_cfg, "family", "") or "").strip()
                    != str(getattr(cfg, "family", "") or "").strip()
                    or int(getattr(alias_cfg, "lora_bucket", 0) or 0)
                    != target.lora_bucket
                    or not compile_cache.has_compile_target(pipeline, alias_cfg)
                ):
                    continue
                try:
                    if compile_cache.execution_contract_digest(
                        pipeline, alias_cfg,
                    ) != target.contract_digest:
                        continue
                except Exception:
                    continue
                if any(
                    not _alias_binding_matches(alias, slot, ref)
                    for slot, ref, _digest in bindings
                ):
                    continue
                name = str(alias.name).strip()
                if name:
                    compatible_names.add(name)
            expected_names = compatible_names & required_names
            if object_proven_by_custom_warmup:
                # gw#603 ruling (2026-07-20, supersedes the ac0bab9 single-
                # name attribution): proof is a property of the WARMED
                # OBJECT and the graph set actually exercised, not of the
                # initiating handler's name — the same identity reasoning as
                # gw#587 design pt 5. A custom object-level warmup (author
                # surface: "wins outright", e.g. LTX's two-stage synthetic
                # that warms EVERY declared graph) therefore attributes its
                # proof to every CONTRACT-COMPATIBLE sibling alias of this
                # exact object (same family, lora bucket, execution-contract
                # digest, and bindings — the compatible_names gate above).
                # pgw#654 removed the `warmup={...: None}` per-handler
                # opt-out with the declared-dict surface itself — the warm
                # plan is derived, so there is no author skip to honor.
                # Runtime backstop stays: every advertised alias serves
                # through the per-call guarded wrapper with hit/miss
                # counters, so an attributed alias whose real requests miss
                # is visible and degrades loudly, never silently.
                permitted_names = set(compatible_names)
            target.function_names = tuple(sorted(
                compatible_names & permitted_names))
            target_quant_execution_lane = next(
                (execution_lane for execution_lane in _MANDATORY_EXECUTION_LANES
                 if target.pipeline_weight_lane.startswith(execution_lane)), "")
            candidate_requested_execution_lane = self._mandatory_execution_lane_of_bound(
                ref for _slot, ref, _digest in bindings)
            mandatory_quant = bool(target_quant_execution_lane)
            if (
                (mandatory_quant or candidate_requested_execution_lane)
                and not expected_names <= set(target.function_names)
            ):
                # Every REQUIRED alias should be proven; a proven superset
                # (custom-warmup attribution covering a non-required
                # sibling) is not a defect. pgw#672: an unproven required
                # alias now serves explicit eager (no compile attribution)
                # instead of killing the whole setup.
                logger.error(
                    "mandatory quantized-lane function proof incomplete "
                    "(expected=%r proven=%r); unproven aliases serve "
                    "explicit eager (pgw#672)",
                    sorted(expected_names), list(target.function_names),
                )
            if not target.function_names or not bindings_valid:
                detail = (
                    "immutable object applicability is incomplete "
                    f"(functions={target.function_names!r} "
                    f"bindings={bindings!r} owned_slots={sorted(candidate.slots)!r})"
                )
                logger.warning(
                    "compile target omitted for %s: %s", spec.name, detail,
                )
                # pgw#672: mandatory lanes no longer raise here — the
                # functions serve explicit eager instead of dying.
                # pgw#824: and a log line is not "instead of dying" on a pod
                # with no stdout — it is dying quietly.
                self._note_eager_posture(
                    rec, "target_applicability_incomplete", detail)
                continue
            execution_lane_error = compile_cache.compile_target_execution_lane_error(
                target.pipeline_weight_lane, target.lora_bucket)
            if execution_lane_error:
                logger.warning(
                    "compile target omitted for %s: %s", spec.name, execution_lane_error)
                self._note_eager_posture(
                    rec, "target_lane_unsupported", execution_lane_error)
                continue
            if (candidate_requested_execution_lane
                    and target_quant_execution_lane != candidate_requested_execution_lane):
                raise compile_cache.CompiledExecutionLaneUnavailableError(
                    f"{candidate_requested_execution_lane.upper()} binding for "
                    f"{spec.name!r} materialized pipeline lane "
                    f"{target.pipeline_weight_lane!r}"
                )
            active_ref = active_selection.ref if active_selection else ""
            active_digest = (
                active_selection.snapshot_digest if active_selection else "")
            if bool(active_ref) != bool(active_digest):
                logger.warning(
                    "compile target omitted for %s: active artifact identity "
                    "is incomplete (ref=%r digest=%r)",
                    spec.name, active_ref, active_digest,
                )
                self._note_eager_posture(
                    rec, "artifact_identity_incomplete",
                    f"ref={active_ref!r} digest={active_digest!r}")
                continue
            if mandatory_quant and not active_ref:
                # pgw#672: a quantized-lane object without a proven exact
                # cell used to fail closed here. It now registers as an
                # ADDRESSABLE, active-less target — serving_tier projects
                # "eager" for its aliases and the hub sees the degrade; the
                # incarnation stays adoptable so a later armed cell can
                # restore the compiled tier without a reload.
                logger.error(
                    "%s compile target for %r has no proven active compiled "
                    "artifact; registering active-less — its aliases serve "
                    "explicit eager (pgw#672)",
                    target_quant_execution_lane.upper(), spec.name,
                )
                self._note_eager_posture(
                    rec, "mandatory_lane_active_less",
                    f"{target_quant_execution_lane.upper()} target registered with no "
                    f"proven active artifact")
            with target.state_lock:
                target.active_compile_ref = active_ref
                target.active_compile_snapshot_digest = active_digest
                target.active_self_mint = bool(getattr(
                    active_selection, "self_mint", False))
            rec.compile_targets[incarnation_id] = target
            # pgw#1010: bind the guards for anything ARMED, not only for a
            # target that names a cell. A JIT INTAKE arm is compiled code with
            # no artifact, and it is now the only dynamo lane there is — gating
            # the pgw#680 guard-miss confession on `active_ref` would take
            # every guard miss on the platform off the wire.
            armed_here = bool(
                active_ref or compile_cache.is_compile_armed(pipeline))
            if armed_here and not self._bind_compile_guard(rec, target):
                # Production wrappers always expose one of the two guard
                # signals. A hand-built/custom wrapper without revocation
                # cannot be advertised as compiled (pgw#672: eager, loudly).
                with target.state_lock:
                    target.active_compile_ref = ""
                    target.active_compile_snapshot_digest = ""
                logger.warning(
                    "compile target %s has no runtime guard revocation signal; "
                    "advertising eager", incarnation_id,
                )
            if target.active_compile_ref or armed_here:
                # pgw#622: post-proof, novel request shapes serve eager while
                # the compiled path warms in the background.
                #
                # pgw#1010: no republish callback. A grown JIT cache belongs
                # to this pod alone now — the cell it used to republish was a
                # dynamo artifact nothing could adopt.
                if hot_swap.enable(pipeline):
                    logger.info(
                        "hot-swap: eager-while-compiling enabled for %s",
                        spec.name)
                else:
                    self._report_no_growth_path(spec, target, pipeline)
        if requested_execution_lane and not rec.compile_targets:
            # pgw#672: degrade, never die — the loaded pipeline serves its
            # functions eagerly; the missing compile target is loud on the
            # wire (serving_tier=eager) and in the activity stream.
            logger.error(
                "%s setup for %r produced no addressable compile-capable "
                "pipeline target; serving explicit eager (pgw#672)",
                requested_execution_lane.upper(), spec.name,
            )
            activity_mod.current_note(
                f"{requested_execution_lane} lane serving eager: no addressable "
                "compile target survived the proof")

    def compile_targets(self) -> List[pb.CompileTarget]:
        """Full-replace READY compile-target snapshot for StateDelta.

        pgw#1032: a target states the identity it IS SERVING (the ACTIVE ref,
        whose key was STAMPED on the artifact at mint) and nothing else. The
        `requested_cell_key`/`requested_cell_axes` fields it used to fill are
        a COMPUTED (`kind="inductor"`) key, a space with no producer since
        pgw#1010 — so the hub's exact-key delivery machinery on them could
        never fire. `requested_cell_axes` is now `reserved 11` on the wire
        (§4.28, th#1751 W4); `requested_cell_key` survives unfilled.
        """
        out: List[pb.CompileTarget] = []
        for rec in self._classes.values():
            if not rec.ready:
                continue
            for target in rec.compile_targets.values():
                with target.state_lock:
                    cfg = target.spec.compile_cell()
                    family = str(getattr(cfg, "family", "") or "").strip()
                    if not family:
                        continue
                    out.append(pb.CompileTarget(
                        incarnation_id=target.incarnation_id,
                        family=family,
                        pipeline_weight_lane=target.pipeline_weight_lane,
                        lora_bucket=target.lora_bucket,
                        contract_digest=target.contract_digest,
                        active_compile_ref=target.active_compile_ref,
                        active_compile_snapshot_digest=(
                            target.active_compile_snapshot_digest),
                        function_names=target.function_names,
                        model_bindings=[pb.CompileTargetBinding(
                            slot=slot, ref=ref, snapshot_digest=digest,
                        ) for slot, ref, digest in target.model_bindings],
                    ))
        return sorted(out, key=lambda target: target.incarnation_id)

    def _compile_target(
        self, incarnation_id: str,
    ) -> Optional[Tuple[_ClassRecord, _CompileTargetRecord]]:
        """Return an exact still-READY target; never infer by family/ref."""
        for rec in self._classes.values():
            if not rec.ready:
                continue
            target = rec.compile_targets.get(incarnation_id)
            if target is not None:
                return rec, target
        return None

    def _resolved_mandatory_execution_lane(self, ref: str) -> str:
        """th#1059 twin (hub: ``mandatoryTracedLane``): mandatory-ness follows
        the hub-resolved EXECUTION lane. Storage never implied execution —
        SDXL's mixed fp8 variant serves the w8a16 upcast lane (plain graphs,
        never scaled_mm) while qwen's serves real w8a8 — and pgw#1148 deleted
        the `#flavor` FALLBACK that guessed at it: §1.32(d) made the token a
        non-address, and a token in a ref was an assertion, not evidence.
        With no resolved lane there is no mandate, so the caller is free
        rather than fail-closed against a guess.
        """

        ref = (ref or "").strip()
        known = False
        mandatory = ""
        for declared, pick in (self._model_resolutions or {}).items():
            resolved_ref = (pick[0] or declared).strip()
            execution_lane_str = (pick[2] or "").strip()
            if not execution_lane_str or ref not in (declared.strip(), resolved_ref):
                continue
            try:
                execution_lane = lanespec.parse_execution_lane(execution_lane_str)
            except ValueError:
                continue
            known = True
            if execution_lane.activation == lanespec.ACT_W8A8:
                mandatory = "w8a8"
            elif execution_lane.activation == lanespec.ACT_W4A4:
                mandatory = "w4a4"
        return mandatory if known else ""

    def _mandatory_execution_lane_of_bound(self, refs: typing.Iterable[str]) -> str:
        """Resolution-aware :func:`_mandatory_lane_of` (th#1059)."""
        for ref in refs:
            execution_lane = self._resolved_mandatory_execution_lane(ref)
            if execution_lane:
                return execution_lane
        return ""

    def _execution_lane_for_ref(self, ref: str) -> str:
        """The hub-resolved th#913 execution-lane descriptor for ``ref``
        (declared or resolved form), "" when the hub sent no lane."""
        return self._execution_lane_pick_for_ref(ref)[0]

    def _execution_lane_pick_for_ref(self, ref: str) -> Tuple[str, bool]:
        """(lane, pinned) for ``ref`` — ``pinned`` True when the hub marked
        the lane as an operator endpoint-pin (pgw#714 kill switch)."""
        ref = (ref or "").strip()
        for declared, pick in (self._model_resolutions or {}).items():
            resolved_ref = (pick[0] or declared).strip()
            execution_lane_str = (pick[2] or "").strip()
            if execution_lane_str and ref in (declared.strip(), resolved_ref):
                pinned = bool(pick[3]) if len(pick) > 3 else False
                return execution_lane_str, pinned
        return "", False

    def _validate_required_compile(
        self, spec: EndpointSpec, run: pb.RunJob,
    ) -> None:
        """Fence scheduler compile evidence against the exact live object.

        This is deliberately repeated before execution. A target ID is a
        worker-session address, not a durable model identity; vacate/reload,
        mutable-tag republish, or an alias/model mismatch must requeue rather
        than execute on a merely same-family pipeline.

        pgw#888 (Paul, 2026-08-15): *"a worker should serve-eager if
        compilation doesn't work … although it should loudly report when it's
        performing in a degraded mode."* So this fence now answers TWO
        questions that it used to fold into one refusal:

        - **Is this the right model?** — incarnation, lane, function and every
          model binding. A mismatch is a different object, there is nothing
          correct to serve, and it still requeues.
        - **Is the pinned COMPILED GRAPH still what this pod serves?** — the
          cell ref/digest half. A cell that de-armed for cause (§4.31), was
          revoked, or was replaced by a newer one is a SPEED fact, not a
          correctness one: the pipeline, its weights and its lane are exactly
          what the hub picked. Refusing there is what burned 11 real requests
          through five retries each. It now serves and confesses.

        The mandatory-quantized carve-out stands: `w8a8`/`w4a4` is a lane the
        author declared serves only from a cell, so a dispatch that cannot
        name one is still refused rather than answered with numerics the
        endpoint never sanctioned.
        """
        setup_slots = self._setup_slots(spec)
        want_execution_lane = self._mandatory_execution_lane_of_bound(
            wire_ref(spec.models[slot]) for slot in setup_slots
        )
        if not run.HasField("required_compile"):
            if want_execution_lane:
                raise RetryableError(
                    f"required_compile_missing: {want_execution_lane.upper()} dispatch "
                    "requires an exact active compile incarnation"
                )
            return
        required = run.required_compile
        identity = (
            required.target_incarnation_id.strip(),
            required.cell_ref.strip(),
            required.cell_snapshot_digest.strip(),
            required.contract_digest.strip(),
        )
        if not all(identity):
            raise RetryableError(
                "required_compile_invalid: target, cell ref/digest, and "
                "contract digest must all be nonempty"
            )
        found = self._compile_target(identity[0])
        if found is None:
            raise RetryableError(
                "required_compile_replaced: selected compile incarnation is "
                "no longer READY"
            )
        _rec, target = found
        with target.state_lock:
            target_execution_lane = target.pipeline_weight_lane
            target_functions = target.function_names
            target_active = (
                target.active_compile_ref,
                target.active_compile_snapshot_digest,
                target.contract_digest,
            )
            target_bindings = target.model_bindings
        if want_execution_lane and not target_execution_lane.startswith(want_execution_lane):
            raise RetryableError(
                f"required_compile_lane_mismatch: {want_execution_lane.upper()} "
                "dispatch selected a live pipeline on lane "
                f"{target_execution_lane!r}"
            )
        if spec.name not in target_functions:
            raise RetryableError(
                "required_compile_function_mismatch: target does not serve "
                f"{spec.name!r}"
            )
        if target_active[2] != identity[3]:
            # The EXECUTION CONTRACT, not the cell. A changed contract digest
            # means the target's call ingress is not the one the hub validated
            # this dispatch against, so serving it would run a different
            # function signature — an identity fault, not a degrade.
            raise RetryableError(
                "required_compile_contract_mismatch: execution contract changed"
            )

        expected: List[Tuple[str, str, str]] = []
        for slot, held_ref, _held_digest in target_bindings:
            binding = spec.models.get(slot)
            ref = wire_ref(binding).strip() if binding is not None else ""
            snap = run.snapshots.get(ref)
            digest = str(getattr(snap, "digest", "") or "").strip()
            if not slot.strip() or not ref or not digest:
                raise RetryableError(
                    "required_compile_binding_missing: every target-owned "
                    "model requires its exact RunJob ref and snapshot digest"
                )
            if ref != held_ref:
                raise RetryableError(
                    "required_compile_binding_mismatch: selected target holds "
                    "a different model ref"
                )
            expected.append((slot, ref, digest))
        if tuple(sorted(expected)) != target_bindings:
            raise RetryableError(
                "required_compile_binding_mismatch: selected target holds a "
                "different model ref or snapshot digest"
            )

        # LAST, and only once every identity fence above has passed: this is
        # the right pipeline, holding the right models, on the right lane —
        # and the compiled graph the hub pinned is not the one it serves.
        if target_active[0] != identity[1] or target_active[1] != identity[2]:
            if want_execution_lane:
                raise RetryableError(
                    f"required_compile_identity_mismatch: {want_execution_lane.upper()} "
                    "dispatch pinned a compile cell this target no longer "
                    "serves, and the lane is declared mandatory — eager would "
                    "serve numerics this endpoint never sanctioned"
                )
            self._report_pinned_cell_unavailable(spec, run, identity, target_active)

    def _report_pinned_cell_unavailable(
        self,
        spec: EndpointSpec,
        run: pb.RunJob,
        identity: Tuple[str, str, str, str],
        active: Tuple[str, str, str],
    ) -> None:
        """pgw#888: SERVE this request, and say loudly that it is degraded.

        A log line is not reporting — a hub-spawned worker exposes no stdout
        (pgw#760) — so the confession is a typed `serve_degrade` event naming
        the degraded mode, the compiled-graph key that failed to be here, and
        the cause. The hub banks it in `worker_activity_events`, which is what
        lets the fleet's republish/re-mint machinery fix the root cause while
        users keep getting outputs.

        Two distinguishable degrades share this exit, and conflating them
        would be the exact `serving_mode` contamination pgw#764 exists to
        prevent: nothing armed means this request really is served EAGER and
        the latency sample must be subtractable; a DIFFERENT armed cell still
        serves compiled, and `serving_mode` reports the cell it actually used.
        """
        job = self.jobs.get((run.request_id, int(run.attempt)))
        if job is not None:
            if job.pinned_cell_degrade_reported:
                return  # the intake fence already confessed this one
            job.pinned_cell_degrade_reported = True
        served_eager = not active[0]
        detail = (
            f"fn={spec.name} request={run.request_id or '<unknown>'} "
            f"attempt={int(run.attempt)} "
            f"pinned_cell={identity[1]} pinned_digest={identity[2][:16]} "
            f"active_cell={active[0] or '<none>'} "
            f"active_digest={active[1][:16] or '<none>'} "
            f"cause=the pinned compiled graph is not armed on this target "
            f"(de-armed for cause, revoked, or superseded); serving "
            f"{'eager' if served_eager else 'the armed cell'} instead of "
            f"refusing (pgw#888)"
        )
        logger.warning(
            "serving request %s DEGRADED: the hub pinned compile cell %s and "
            "this target serves %s — answering it anyway",
            run.request_id or "<unknown>", identity[1], active[0] or "eager",
        )
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            detail,
            phase=serving_mode_mod.FALLBACK_PINNED_CELL_UNAVAILABLE,
            compiled_graph_key=identity[1],
        )
        if served_eager:
            self._mark_request_eager_fallback(
                run.request_id,
                serving_mode_mod.FALLBACK_PINNED_CELL_UNAVAILABLE,
            )

    def in_flight_keys(self) -> List[Tuple[str, int]]:
        return [k for k, j in self.jobs.items() if not j.finished and not j.superseded]

    # ---- finalize tracking (gw#516) ------------------------------------------

    def finalizing_jobs(self) -> int:
        """Jobs past the decode->finalize handoff: GPU slot terminally
        released, encode/upload tail still running, result unshipped. The
        hub must treat these as live work (drain/retire gating) even though
        the GPU is already serving the next request."""
        with self._finalizing_lock:
            return self._finalizing_count

    def _enter_finalize(self, job: _Job) -> None:
        """Handler-thread callback at the terminal GPU-slot release."""
        with self._finalizing_lock:
            if job.finalizing or job.finished:
                return
            job.finalizing = True
            self._finalizing_count += 1
        self._signal_state_change_threadsafe()

    def _exit_finalize(self, job: _Job) -> None:
        """Job coroutine, after its result shipped (any terminal path)."""
        with self._finalizing_lock:
            if not job.finalizing:
                return
            job.finalizing = False
            self._finalizing_count -= 1
        self._signal_state_change_threadsafe()

    def _signal_state_change_threadsafe(self) -> None:
        """_on_state_change from any thread: lifecycle.state_changed needs a
        running loop, so handler-thread callers hop onto the executor loop."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._on_state_change)
                return
            except RuntimeError:
                pass
        self._on_state_change()

    # ---- dynamic slot materialization (pgw#532 / th#767) --------------------

    def _hub_binding(self, ref: str) -> ModelRef:
        """The one binding object for a hub-named wire ref (raises
        ``ValueError`` on non-CAS grammar). Registered with the store so
        provider classification stays confident on bare-ref paths."""
        binding = self._hub_bindings.get(ref)
        if binding is None:
            binding = self._hub_bindings.setdefault(ref, _hub_binding_for_wire_ref(ref))
            self.store.register_binding(wire_ref(binding), binding)
        return binding

    def _bound_slot(self, spec: EndpointSpec, slot: str, ref: str) -> ModelRef:
        """Refuse-never-default slot binding (pgw#904): the hub is the only
        resolver. A dispatch-named CAS ref binds; the declared tensorhub
        binding stands when the dispatch names nothing; anything else is a
        typed refusal — never an upstream fetch, never a fallback. The gw#583
        fixed-slot identity gate stays: a dispatch naming a DIFFERENT repo for
        a slot with no ``selected_by=`` catalog is silent drift, refused."""
        declared = spec.models.get(slot)
        if ref:
            if (
                declared is not None
                and declared.source == "tensorhub"
                and ref == wire_ref(declared)
            ):
                return declared
            try:
                binding = self._hub_binding(ref)
            except ValueError:
                raise RetryableError(
                    f"slot {slot!r} of {spec.name!r}: dispatched ref {ref!r} "
                    "is not a tensorhub-CAS ref; a connected worker resolves "
                    "nothing itself (pgw#904)") from None
            catalog_slot = spec.slots.get(slot)
            fixed_repo = (
                declared is not None
                and declared.source == "tensorhub"
                and not (catalog_slot is not None and catalog_slot.selected_by)
            )
            if fixed_repo and declared is not None and binding.path != declared.path:
                raise ModelSlotIdentityError(
                    spec.name, slot,
                    declared_ref=wire_ref(declared), dispatched_ref=ref,
                )
            return binding
        if declared is not None and declared.source == "tensorhub":
            return declared
        raise RetryableError(
            f"slot {slot!r} of {spec.name!r} has no loadable hub ref for this "
            f"request (dispatched ref={ref!r}, declared default "
            f"source={getattr(declared, 'source', None)!r}); a connected "
            "worker never fetches a Slot's raw upstream default "
            "(pgw#532/gw#465) — the hub must resolve the slot to a "
            "tensorhub-CAS ref"
        )

    def _multi_group_handler_refusal(self, spec: EndpointSpec) -> str:
        """pgw#748 residual, made LOUD instead of silently wrong.

        ``ctx.device`` still resolves to ``cuda:{torch.cuda.current_device()}``.
        For a SYNC handler that is correct by construction — the handler thread
        runs ``set_device(group rank-0)`` and, since 6424bce, so does the load
        thread. An ASYNC handler runs on the shared event loop, whose thread
        device belongs to whichever group most recently touched it, so on a
        multi-group worker ``ctx.device`` can name a card the job does not own.

        A wrong device is not a crash; it is a request that quietly computes on
        a sibling's card, competing for its VRAM. So it is refused, by name,
        until ``ctx.device`` is explicit from the job's group. Single-group
        workers — every pod today — are untouched.
        """
        if self.topology.execution_groups <= 1 and self.topology.degree <= 1:
            return ""
        if not (spec.is_async or spec.is_async_gen):
            return ""
        return (
            f"{spec.name}: async handlers are not yet served on a multi-group "
            f"worker ({self.topology}). ctx.device resolves from the calling "
            "thread's current CUDA device, which on the shared event loop is "
            "not this job's group. Refused rather than served on a sibling's "
            "card (pgw#748)."
        )

    def _eager_only_reason(self) -> str:
        """Non-empty when this pod's topology forbids compile arming (pgw#775).

        Once ``enable_parallelism`` installs the CP hooks, EVERY forward
        through the sharded modules issues collectives — and the only
        participant-supplying seam is the pipeline-level SP gate. A hot-swap
        warm compile, a mint seed, a proof warmup or an activation probe all
        forward on rank 0 only, outside that gate, and hang the group. "Eager
        only at degree>1" is therefore enforced by construction: no compile
        selection is fetched, no arming scope opens, no targets install, no
        cell adopts. This is the code the a08a3bd commit message claimed.

        pgw#1113/pgw#819: the condition is ``degree > 1``, FULL STOP — it used
        to be ``degree > 1 and parallel == "sequence"``, an allowlist by mode
        NAME, and every mode not on the list inherited a hole. ``internal``
        was the measured one (a model that spans its cards by its own device
        map bakes that placement into its kernels, so its cell keyed
        byte-identically to the single-GPU one, in both directions), and
        ``cfg`` — the platform's next declared sharding mode
        (``topology.PARALLEL_CFG``) — would have inherited the same hole the
        day it got a serve-side implementation. A gate that has to be widened
        once per new mode is not a rule; ``degree > 1`` is.

        This costs nothing today (no ``internal``-parallel release compiles)
        and it is SUPERSEDED, not contradicted, by the ``placement`` keying
        fact (``aot_serve.class_hash``): once a cell can state which cards it
        was baked for, cells at degree>1 become servable and this gate can
        narrow again to the modes whose collectives genuinely forbid an
        ungated forward.
        """
        topo = self.topology
        if topo.degree > 1:
            return (
                f"eager only at {topo}: compile/hot-swap/self-mint are "
                f"disabled at degree>1 (pgw#775/pgw#819) — under "
                f"{topo.parallel or 'internal placement'} a compile cell "
                f"cannot state the {topo.degree}-card placement it would be "
                f"baked for, and under platform sharding any forward outside "
                f"the parallelism gate would hang the group"
            )
        return ""

    def _dispatch_group(self, run: "pb.RunJob") -> int:
        """Which execution group this dispatch names (pgw#748 / th#1285 §2a).

        ``ResolvedCompute.gpu_index`` is unchanged on the wire and now names
        the group's RANK-0 device — 0, D, 2D, … — so the whole derivation is
        ``gpu_index // D``. At D == 1 this is the identity, which is why
        nothing about today's dispatch path moves.

        On a WIDE pod the hub's answer is load-bearing and a wrong one is
        silent: a missing ``compute`` used to floor to group 0 and an index
        that is not a rank-0 device was floored to a real group with a
        `logger.warning`, so four jobs could pile onto one card while three sat
        idle and the pod still reported four healthy slots. At G>1 both are
        typed refusals (pgw#779) — RETRYABLE, because a dispatch that carries
        the field serves fine, and nothing about the tenant's input is wrong.
        Single-group pods keep the historical behaviour exactly: CPU functions
        and pre-topology hubs have no compute and there is only one group to
        mean.
        """
        if self.topology.execution_groups <= 1:
            return 0
        if not run.HasField("compute"):
            raise DispatchGroupUnresolved(
                f"dispatch carries no resolved compute on a {self.topology} "
                "pod: the execution group cannot be derived, and serving it as "
                "group 0 would pack this job onto a card another group owns"
            )
        try:
            return self.topology.group_ordinal_exact(int(run.compute.gpu_index))
        except TopologyError as exc:
            raise DispatchGroupUnresolved(
                f"dispatched gpu_index={int(run.compute.gpu_index)} is not a "
                f"rank-0 device of {self.topology}: {exc}. The hub and the "
                "worker disagree about the packing; refused rather than "
                "floored onto a group this job does not own"
            ) from exc

    def _gpu_permit_for_group(self, group: int) -> asyncio.Semaphore:
        """The permit that IS group ``group``'s card (pgw#779).

        Out of range is a programming error, not a thing to floor: the group
        was derived by `_dispatch_group`, which already refused anything the
        topology does not contain.
        """
        g = int(group)
        if g < 0 or g >= len(self._gpu_permits):
            raise DispatchGroupUnresolved(
                f"no GPU permit for group {g} of {len(self._gpu_permits)} "
                f"({self.topology})"
            )
        return self._gpu_permits[g]

    def _gpu_permit_for_record(self, rec: "_ClassRecord") -> asyncio.Semaphore:
        """A record belongs to exactly one group — its specs carry the
        ordinal, and `instance_key` includes it, so this is a lookup and not a
        guess."""
        ordinals = {
            int(s.device_group_ordinal) for s in rec.specs
        } or {int(current_device_group())}
        return self._gpu_permit_for_group(sorted(ordinals)[0])

    def _group_effective_spec(
        self, spec: EndpointSpec, group: int
    ) -> EndpointSpec:
        """Bind the dispatch to its group. Two groups are two cards, so they
        are two resident instances: the ordinal joins ``instance_key`` and the
        existing one-record-per-key machinery does the rest."""
        if int(group) == int(spec.device_group_ordinal):
            return spec
        return dc_replace(spec, device_group_ordinal=int(group))

    def _dispatched_spec(
        self, spec: EndpointSpec, slots: Mapping[str, dispatch.SlotOrder],
    ) -> EndpointSpec:
        """The spec THIS dispatch runs (pgw#532): every declared Slot rebound
        to the hub-resolved pick in the neutral slot orders. A pick that
        differs from the declared binding derives a NEW instance key — one
        resident instance per (class, resolved binding set), so ``setup()``
        re-runs for the pick and setup-held state (``self.pipeline``) stays
        coherent per checkpoint while the LRU machinery evicts whole
        instances. Function-shaped (``cls=None``) specs rebind too — their
        slots inject via ``_handler_kwargs``, which reads ``spec.models``."""
        if not spec.slots:
            return spec
        run_refs = {
            slot: so.ref for slot, so in slots.items() if slot and so.ref
        }
        effective = dict(spec.models)
        for slot, decl in spec.slots.items():
            if decl.optional and not run_refs.get(slot, ""):
                # Unbound optional slot: the deploy chose not to serve this
                # lane, and the deploy decides (th#980/ie#524) — a code
                # default_checkpoint is a hub-less bootstrap, never a reason
                # to resurrect a lane the hub did not bind. Dropping it from
                # `effective` is what makes the rest fall out: `_setup_slots`
                # skips it, nothing is materialized, and setup() runs with
                # the parameter's own default.
                effective.pop(slot, None)
                continue
            effective[slot] = self._bound_slot(spec, slot, run_refs.get(slot, ""))
        if effective == spec.models:
            return spec
        return dc_replace(spec, models=effective)

    def _effective_config(
        self, spec: EndpointSpec,
        stamped: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """th#1087 effective declared-parameter values for one dispatch:
        declared defaults <- worker's current config store <- the head's
        dispatch-stamped values (read-at-dispatch class; a stamped job keeps
        its values even if a gen bump lands mid-flight). Wire extraction —
        and the legacy store advance — is HEAD code, not this function's
        (pgw#904)."""
        if not spec.config:
            return {}
        values = {p.name: p.default for p in spec.config}
        declared = set(values)
        for name, v in self.runtime_config.parameters_for(spec.name).items():
            if name in declared:
                values[name] = v
        if stamped:
            values.update(
                {name: v for name, v in stamped.items() if name in declared})
        return values

    def _served_identity(
        self, spec: Optional[EndpointSpec], job: Optional["_Job"] = None,
    ) -> "serving_mode_mod.ServedIdentity":
        """pgw#764/th#1293 dimensions for one completed request.

        `_served_lane` already reports `...+compiled`, but that axis is BINARY
        platform-wide: it cannot tell an AOT `.pt2` replay from a JIT dynamo
        cell and it names no artifact, so "AOT vs JIT p50 on 4090s for sdxl
        w8a8" was unanswerable over our own production traffic even though the
        worker knew the answer for every request. The discriminator is the
        ARMED artifact (`aot_serve` owns it), never the lane string.

        Reads the same `rec.compile_targets` scan `_served_lane` uses, so the
        two can never disagree about whether this request ran compiled.
        """
        ref, pipeline = "", None
        posture = ""
        if spec is not None and spec.cls is not None:
            rec = self._classes.get(spec.instance_key)
            if rec is not None:
                armed_pipeline = None
                for target in rec.compile_targets.values():
                    with target.state_lock:
                        active = str(target.active_compile_ref or "")
                    if active:
                        ref, pipeline = active, target.pipeline
                        break
                    # pgw#1078: a JIT INTAKE arm names no artifact — that is
                    # the whole point of pgw#1010's `is_compile_armed` — so a
                    # ref-only scan reports every intake-served request as
                    # eager. Carry the pipeline so `classify_mode` can ask it.
                    if armed_pipeline is None and compile_cache.is_compile_armed(
                            target.pipeline):
                        armed_pipeline = target.pipeline
                if not ref:
                    pipeline = armed_pipeline
                    if pipeline is None:
                        posture = self._eager_posture(spec, rec)
        return serving_mode_mod.resolve(
            active_compile_ref=ref,
            pipeline=pipeline,
            guard_missed=bool(job is not None and job.served_eager_fallback),
            verdict=(job.fallback_reason if job is not None else ""),
            eager_posture=posture,
        )

    def _eager_posture(self, spec: EndpointSpec, rec: "_ClassRecord") -> str:
        """pgw#824: the classified reason this record has no armed cell.

        Read LIVE rather than only from the stored token, because the two
        transient postures are properties of right now, not of the arming
        decision: a mint in flight means this eager request is a warming pod
        (its `fallback_reason` stops appearing on its own), while a stored
        decline means it never will.

        Order matters. An in-flight mint outranks whatever the arming brain
        said when it opened, because the background driver can start after a
        DIFFERENT decline was already recorded.
        """
        if rec.background_mint is not None:
            return serving_mode_mod.POSTURE_MINT_IN_PROGRESS
        stored = str(rec.eager_posture or "")
        if stored:
            return stored
        # pgw#1093: a target that ARMED and then degraded is not "uncompiled"
        # — it is a named execution failure, and reporting it as the generic
        # terminal token is what made an installed-then-degraded target read
        # identically to a never-installed one. Live, because a degrade can
        # land after boot on a target whose guard callback was never bound.
        for target in rec.compile_targets.values():
            if compile_cache.degrade_reason(target.pipeline):
                return cell_adopt.EagerPhase.COMPILED_DEGRADED.value
        if not any(
            s.compile is not None and s.compile.family for s in rec.specs
        ):
            # Eager is this release's CONTRACT, not a degradation. Naming it
            # keeps the honest zero out of every defect-class count.
            return serving_mode_mod.POSTURE_NO_COMPILE_DECLARED
        if not rec.ready:
            return serving_mode_mod.POSTURE_ARM_PENDING
        return serving_mode_mod.POSTURE_UNCOMPILED

    def _refuse_unservable_lane(self, spec: EndpointSpec, instructed: str) -> None:
        """A lane instruction is honored only when it needs NO worker-side
        resolution: the bf16 family (the declared base) or a lane the endpoint
        declares (``handles=``, author code). Everything else used to run the
        coarse-family ladder twin pgw#904 deleted, so it refuses typed —
        never a silent fallback, never a rebind."""
        raw = str(instructed or "").strip()
        if not raw:
            return
        try:
            req = lanespec.parse_execution_lane_spec(raw)
        except ValueError as exc:
            raise ValidationError(str(exc)) from None
        if req.is_zero or req.family == lanespec.FAMILY_BF16:
            return
        if self._handled_execution_lane_body(spec, raw):
            return
        raise lanespec.ExecutionLaneUnavailableError(
            raw,
            "worker-side lane expansion is deleted (pgw#904): this endpoint "
            "does not declare the lane and the worker rebinds nothing — the "
            "hub dispatches resolved bindings")

    def _handled_execution_lane_body(self, spec: EndpointSpec, instructed: str) -> str:
        """th#1050: the instructed lane's body when the endpoint DECLARES it
        (handles=) — the author's code, not binding surgery, serves it."""

        if not instructed or not getattr(spec, "handles", ()):
            return ""
        try:
            req = lanespec.parse_execution_lane_spec(instructed)
        except ValueError:
            return ""
        if req.execution_lane is None:
            return ""
        body = lanespec.execution_lane_body_id(req.execution_lane)
        return body if body in spec.handles else ""

    def _record_applied_lanes(
        self,
        spec: EndpointSpec,
        rec: _ClassRecord,
        applied: Tuple[lanespec.AppliedLane, ...],
    ) -> None:
        """pgw#1104: bank what ``setup()`` reported it did to the weights.

        A report that DIVERGES from the binding is the whole point of the
        mechanism, so it is a wire row and not a log line: the reader must be
        able to see, from the events alone, that this instance stopped
        executing the checkpoint's lane and which lane it executes instead."""
        rec.applied_lanes = list(applied)
        if not applied:
            return
        bound = self._bound_execution_body(spec)
        for entry in applied:
            activity_mod.emit_event(
                activity_mod.KIND_APPLIED_LANE,
                detail=f"{entry.detail()} bound={bound}",
                phase=entry.component)
            # th#1871 P1: the same fact, typed. `applied=fp8-w8a8-dynamic
            # bound=bf16-w16a16` is exactly the per-component posture the hub
            # needs to know whether two numbers describe the same thing — and
            # the prose line above was, until now, the only place it existed.
            rec.posture.component(
                entry.component, applied_quant=entry.body, bound_quant=bound)

    def _record_applied_attention(
        self, rec: _ClassRecord, applied: Tuple[Any, ...],
    ) -> None:
        """pgw#1043: bank the attention path setup() installed, one wire row
        per component. Reporting nothing is dense — the absence is the default,
        not a gap, so silence emits nothing."""
        rec.applied_attention = list(applied)
        for entry in applied:
            activity_mod.emit_event(
                activity_mod.KIND_APPLIED_ATTENTION,
                detail=entry.detail(),
                phase=entry.component)
            # th#1871 P1: the KERNEL half of the report, typed. The ledger
            # raises the `attention_fallback` technique itself when the engaged
            # backend is not the one that was asked for — a fallback nobody has
            # to notice is a fallback nobody notices (ie#707).
            backend = str(getattr(entry, "backend", "") or "")
            if backend:
                rec.posture.attention(
                    backend, wanted=str(getattr(entry, "backend_wanted", "") or ""))

    def _served_attention_mode(self, spec: EndpointSpec) -> str:
        """The attention mode `metrics.attention_mode` reports for a request on
        this spec. Never guessed: it is what the installing code reported, and
        the ABSENCE of a report is dense — so an endpoint with no sparse path
        reports "" and nothing downstream has to learn a new default."""
        if spec.cls is None:
            return ""
        rec = self._classes.get(spec.instance_key)
        entries = list(getattr(rec, "applied_attention", []) or []) if rec else []
        # th#1871 P1: a BACKEND-only report says nothing about sparsity, so it
        # must not turn "unreported" into a claim of dense. The two axes share
        # one record and one scope; they do not share a default.
        modes = [str(e.mode or "") for e in entries if str(e.mode or "")]
        if not modes:
            return ""
        return attnspec.most_sparse_mode(modes)

    def _served_attention_detail(self, spec: EndpointSpec) -> str:
        """The full applied-attention row for this instance (k, block, measured
        density, selector, index artifact) — what `attention_mode` alone cannot
        carry. One line per component, joined."""
        if spec.cls is None:
            return ""
        rec = self._classes.get(spec.instance_key)
        entries = list(getattr(rec, "applied_attention", []) or []) if rec else []
        return "; ".join(e.detail() for e in entries)

    def _bound_execution_body(self, spec: EndpointSpec) -> str:
        """The most-quantized lane BODY this spec's BINDINGS resolve to — what
        the hub handed the worker, before any serve-time recipe."""
        return lanespec.most_quantized_body(
            lanespec.execution_lane_body_of_binding(
                getattr(binding, "storage_dtype", "") or "")
            for binding in spec.models.values())

    def _served_execution_body(
        self, spec: EndpointSpec, instructed: str = "",
    ) -> str:
        """The WEIGHTS half of the lane this instance executes: the
        most-quantized body over the pipeline bindings AND whatever this
        instance's ``setup()`` reported it APPLIED to them. A declared
        (handles=) instruction owns the body outright.

        pgw#1104: the applied half is not decoration. minimax-h3 binds a bare
        tag (empty flavor) and quantizes 300 Linears to w8a8 fp8 inside
        setup(), so a binding-only derivation priced, verdicted and "proved" a
        37.4 GiB bf16 lane against a 21.7 GiB fp8 one that was really running.
        The lane id is a KEY (th#935 verdicts, compile cells, floors,
        pricing), so it follows the WEIGHTS AS EXECUTED — reported by the
        recipe that converted them, never sniffed off tensor subclasses."""
        handled = self._handled_execution_lane_body(spec, instructed)
        if handled:
            return handled
        applied: Tuple[lanespec.AppliedLane, ...] = ()
        if spec.cls is not None:
            rec = self._classes.get(spec.instance_key)
            if rec is not None:
                applied = tuple(rec.applied_lanes)
        bodies = [
            lanespec.execution_lane_body_of_binding(
                getattr(binding, "storage_dtype", "") or "")
            for binding in spec.models.values()
        ]
        # The applied report is validated against the lane table at report
        # time (`report_applied_lane`), so it can only name a real lane body.
        bodies.extend(entry.body for entry in applied)
        return lanespec.most_quantized_body(bodies)

    def _served_execution_lane(
        self,
        spec: EndpointSpec,
        instructed: str = "",
        served: Optional[serving_mode_mod.ServedIdentity] = None,
    ) -> str:
        """The CONCRETE lane this spec's instance executes as, for
        JobMetrics.lane and ctx.lane reporting: the executed WEIGHTS body at
        the OBSERVED execution posture.

        ie#655: there is exactly ONE reading of the execution axis on this
        worker, and it is ``ServedIdentity.serving_mode`` — the same value
        stamped on ``metrics.serving_mode``. The lane cannot contradict the
        serving mode because it is COMPOSED from it, not derived beside it.
        Two separate derivations is how a wan-2.2 H100 that declined its own
        mint for `insufficient_vram`, served eager, and said so three times in
        its own boot rows still reported `fp8-w8a8-dynamic+compiled` on both
        billed requests: the second reading ran the lane table's PLANNING
        coercion (`fp8-w8a8-dynamic` is a compiled-only CHOICE) over an
        observed eager posture and rewrote the fact. A declared instruction
        owns the body, never the execution axis: what the hub asked for is not
        evidence of what ran."""
        if served is None:
            served = self._served_identity(spec)
        compiled = served.serving_mode != serving_mode_mod.MODE_EAGER
        return lanespec.execution_lane_id(
            lanespec.observed_execution_lane(
                self._served_execution_body(spec, instructed), compiled))

    async def ensure_desired_instance(
        self,
        desired: "pb.DesiredInstance",
        snapshots: Dict[WireRef, "pb.Snapshot"],
    ) -> None:
        """Best-effort warm of one declarative, fully bound instance.

        Every failure — including a binding-shape refusal before setup —
        emits MODEL_STATE_FAILED for the instance's refs (th#1055: the old
        pre-setup ValidationErrors were pod-local only, so a refused hot
        intent stalled the worker fleet-invisibly forever)."""
        instance_refs = [
            r for r in dict.fromkeys(m.ref.strip() for m in desired.models) if r
        ]
        try:
            spec = self.specs.get(desired.function_name)
            if spec is None:
                raise ValidationError(
                    f"unknown function {desired.function_name!r}")
            await self._ensure_desired_instance_validated(spec, desired, snapshots)
        except Exception as exc:
            # Host-RAM admission already emitted the precise largest staged
            # ref(s) that caused the capacity failure. Do not overwrite that
            # signal by failing smaller shared refs such as an SDXL VAE.
            if not isinstance(exc, HOST_RAM_REFUSALS):
                error = _model_failure_vocab(exc)
                for ref in instance_refs:
                    await self._send(pb.WorkerMessage(
                        model_event=self.store.model_event(
                            ref, pb.MODEL_STATE_FAILED, error=error,
                        )
                    ))
            raise

    async def _ensure_desired_instance_validated(
        self,
        spec: EndpointSpec,
        desired: "pb.DesiredInstance",
        snapshots: Dict[WireRef, "pb.Snapshot"],
    ) -> None:
        if spec.cls is None:
            raise ValidationError(
                f"function {desired.function_name!r} has no persistent instance to warm"
            )
        # th#697/hello_ack contract: hot bindings may arrive in DECLARED ref
        # space; remap through the hub's precision picks so the warm instance
        # derives the SAME per-pick key a laddered dispatch derives.
        remapped: List[pb.ModelBinding] = []
        for m in desired.models:
            binding = pb.ModelBinding()
            binding.CopyFrom(m)
            binding.slot = m.slot.strip()
            ref = m.ref.strip()
            pick = self._model_resolutions.get(ref)
            if pick is not None and pick[0]:
                ref = pick[0]
            binding.ref = ref
            remapped.append(binding)
        pairs = [(m.slot, m.ref) for m in remapped]
        bindings = dict(pairs)
        if any(not slot or not ref for slot, ref in pairs):
            raise ValidationError(
                f"desired instance {desired.function_name!r} has an empty slot or ref"
            )
        # th#1055: deploy-bound Slots (ie#524/th#980) carry NO code default,
        # so spec.models does not name them — demanding set-equality against
        # spec.models refused EVERY hot intent on slot-only endpoints. The
        # instance must cover exactly the declared slots, where a missing
        # slot is acceptable only when a code default can fill it (the same
        # fallback dispatch uses).
        declared = set(spec.models) | set(spec.slots)
        undeclared = sorted(set(bindings) - declared)
        # An optional slot (setup param has a default) may be left unbound —
        # a single-lane deploy of a multi-lane endpoint is legal config.
        optional = {s for s, d in spec.slots.items() if d.optional}
        unbound = sorted(
            s for s in declared
            if s not in bindings and s not in spec.models and s not in optional
        )
        if len(bindings) != len(pairs) or undeclared or unbound:
            raise ValidationError(
                f"desired instance {desired.function_name!r} must bind the "
                f"declared slots {sorted(declared)!r} (code defaults exist for "
                f"{sorted(spec.models)!r}); got {sorted(bindings)!r}"
            )

        orders = {m.slot: dispatch.SlotOrder(ref=m.ref) for m in remapped}
        effective = self._dispatched_spec(spec, orders)
        mismatched = {
            slot: wire_ref(effective.models[slot])
            for slot, ref in bindings.items()
            if slot in effective.models and wire_ref(effective.models[slot]) != ref
        }
        if mismatched:
            raise ValidationError(
                f"desired instance {desired.function_name!r} does not match the "
                f"worker's resolved bindings: {mismatched!r}"
            )
        await self.ensure_setup(effective, snapshots)

    def _job_pin_refs(self, spec: EndpointSpec, slots: List[str]) -> List[WireRef]:
        """Refs a job pins for its whole lifetime: every routed slot EXCEPT
        lane refs (gw#551 — the LaneGate pins those around the actual
        pipeline call, so the idle sibling stays LRU-demotable), PLUS the
        record's shared-component entries (pgw#636: holders alone no longer
        block demotion, so an executing job must pin the TE/VAE entries its
        pipeline aliases)."""
        rec = self._classes.get(spec.instance_key) if spec.cls is not None else None
        execution_lane_refs = rec.execution_lane_refs if rec is not None else set()
        shared_ids = (
            [k.cache_id() for k in rec.shared_keys] if rec is not None else []
        )
        return list(dict.fromkeys(
            [
                r for s in slots
                for r in [wire_ref(spec.models[s])]
                if r not in execution_lane_refs
            ]
            + shared_ids
        ))

    def _job_admission_sizes(
        self, spec: EndpointSpec, slots: List[str],
        snapshots: Mapping[WireRef, pb.Snapshot],
    ) -> Dict[WireRef, int]:
        """ref -> expected VRAM bytes for one job's admission lease (pgw#641
        Stage 2). Same ref set as :meth:`_job_pin_refs`; bytes follow the
        pgw#636 ask ladder — a prior MEASURED hint wins, else the dispatch's
        own snapshot byte total (honest for a never-seen pick), else the
        banked snapshot's total, else 0 (lease-protected, no reservation)."""
        res = self.store.residency
        run_snapshots = dict(snapshots)

        def _expect(ref: WireRef) -> int:
            hint = res.vram_hint(ref)
            if hint > 0:
                return hint
            snap = run_snapshots.get(ref)
            if snap is not None:
                return sum(int(f.size_bytes) for f in snap.files)
            return sum(self.store.component_sizes(ref).values())

        return {ref: _expect(ref) for ref in self._job_pin_refs(spec, slots)}

    @staticmethod
    def _activation_key(spec: EndpointSpec) -> str:
        """Key the learned activation footprint by FUNCTION, not by pick
        (pgw#652). Transient VRAM is a property of the shape and the graph the
        function runs — a 1024^2 SDXL denoise costs the same latents and
        attention workspace whichever checkpoint is bound — so keying by
        function means a never-seen checkpoint inherits a real measurement
        instead of reserving nothing on its first request."""
        return spec.name

    def _class_record(self, spec: EndpointSpec) -> _ClassRecord:
        """Instance-group record for ``spec``, created on first sight for
        DERIVED (per-pick) specs. Never removed: records are tiny and the
        distinct-pick set a worker sees is bounded by its disk anyway."""
        assert spec.cls is not None
        rec = self._classes.get(spec.instance_key)
        if rec is None:
            rec = self._classes.setdefault(spec.instance_key, _ClassRecord(cls=spec.cls))
        if not any(s is spec or s == spec for s in rec.specs):
            rec.specs.append(spec)
        return rec

    @asynccontextmanager
    async def _exclusive_gpu(
        self,
        intent_id: str = "",
        *,
        resume_stage: "pb.LifecycleIntentStage" = pb.LIFECYCLE_INTENT_STAGE_WARMING,
    ) -> typing.AsyncIterator[None]:
        """Hold every worker GPU permit for setup/adoption proof warmups.

        Inductor exposes process-global cache counters. Acquiring only one
        permit on a multi-slot worker would let another graph increment them
        inside this target's before/after window and falsely certify it.
        These maintenance paths run before a job holds a permit themselves.
        """
        # pgw#779: EVERY group's permit, each to its full depth — "exclusive"
        # means no other job on this pod, and with per-group permits that is
        # G x per-group, not a count of G.
        all_permits = [
            permit for permit in self._gpu_permits
            for _ in range(self._gpu_permits_each)
        ]
        acquired = 0
        tokens: List[Tuple[asyncio.Semaphore, int]] = []
        try:
            for permit in all_permits:
                await self._intent_await(
                    intent_id,
                    permit.acquire(),
                    operation="exclusive GPU permit",
                    status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                    stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_GPU_SLOT,
                    reason=pb.LIFECYCLE_WAIT_REASON_GPU_SLOT,
                )
                tokens.append(
                    (permit, self._permits.take(permit, "exclusive GPU warmup")))
                acquired += 1
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                resume_stage,
            )
            yield
        except asyncio.CancelledError:
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_CANCELED,
                (
                    resume_stage
                    if acquired == len(all_permits)
                    else pb.LIFECYCLE_INTENT_STAGE_WAIT_GPU_SLOT
                ),
                detail="exclusive GPU wait canceled",
            )
            raise
        finally:
            for permit, token in tokens:
                self._permits.drop(permit, token)
                permit.release()

    # ---- setup -------------------------------------------------------------

    async def ensure_setup(
        self,
        spec: EndpointSpec,
        snapshots: Optional[Dict[WireRef, pb.Snapshot]] = None,
        promote_slots: Optional[List[str]] = None,
        arm: Optional[_ArmOrder] = None,
    ) -> Any:
        if spec.cls is None:
            return None  # function-shaped endpoint: no instance, no setup
        self.store.bind_loop()
        try:
            activity_mod.bind_sink(self._send, asyncio.get_running_loop())
        except RuntimeError:
            pass
        rec = self._class_record(spec)
        async with self._setup_singleflight(spec, rec) as intent_id:
            if rec.ready and not rec.stale:
                setup_refs = [
                    wire_ref(spec.models[slot])
                    for slot in self._setup_slots(spec)
                ]
                for ref in setup_refs:
                    wanted = self.store.snapshot_digest(
                        ref, (snapshots or {}).get(ref)
                    )
                    if wanted and rec.held_snapshot_digests.get(ref) != wanted:
                        logger.info(
                            "snapshot identity moved for %s: %s %s -> %s; "
                            "vacating stale instance",
                            spec.name, ref,
                            rec.held_snapshot_digests.get(ref) or "<unknown>",
                            wanted,
                        )
                        rec.stale = True
                        break
            if rec.ready and rec.stale:
                # gw#494: the instance was loaded for a superseded pick —
                # vacate (releasing its OLD-ref bookings) and set up fresh
                # with the current bindings.
                async with self._intent_lock(
                    intent_id,
                    self._load_lock,
                    operation=f"vacate stale setup for {spec.name}",
                    stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
                    reason=pb.LIFECYCLE_WAIT_REASON_LOAD_LOCK,
                    resume_stage=pb.LIFECYCLE_INTENT_STAGE_LOADING_HOST,
                ):
                    await vacate_record(rec, self.teardown_seam)
            if rec.ready:
                await self._promote_setup_refs(spec, promote_slots, rec=rec)
                self._intent_transition(
                    intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                    pb.LIFECYCLE_INTENT_STAGE_READY,
                )
                return rec.instance
            if self._record_has_setup_ownership(rec):
                # A prior process-local cancellation/failure may have reached
                # tenant setup before this worker version could roll it back.
                # Never layer a fresh instance over uncertain ownership.
                logger.warning(
                    "clearing incomplete setup ownership before retrying %s",
                    spec.name,
                )
                await self._rollback_failed_setup(rec)
            # gw#601: setup+warmup is one reportable activity. The watchdog
            # heartbeats through long wire-silent calls (inductor etc.) while
            # they provably burn CPU; a hang stops the beat within one
            # interval and the hub's generic stall rule owns termination.
            act = activity_mod.begin(
                activity_mod.KIND_SELF_MINT_COMPILE if spec.compile is not None
                else activity_mod.KIND_WARMUP,
                activity_mod.PHASE_LOAD,
            )
            try:
                # pgw#797: THE `pipeline_load` span, and its only owner. Every
                # setup — boot scan, hub-delivered `dynamic` spec, hot
                # instance, RunJob — funnels through here, which is why the two
                # `Lifecycle` sites pgw#789 used could not see a real load.
                # `in_boot()` keeps steady-state re-setups out of the ladder.
                # The ordinal is threaded to the nested `warmup` span so
                # `pipeline_load` reads as weights->VRAM by subtraction.
                with _pipeline_load_span(spec) as load_span:
                    rec.boot_load_ordinal = (
                        load_span.ordinal if load_span is not None else 0)
                    with activity_mod.watchdog(act):
                        instance = await self._setup_locked(
                            spec,
                            rec,
                            snapshots,
                            intent_id=intent_id,
                            arm=arm,
                        )
            except BaseException as exc:
                # gw#661: a will-retry condition is not a failure. Only
                # exhausting the budget is, and then the hub must see it.
                will_retry = _setup_error_will_retry(exc)
                exhausted = False
                if will_retry:
                    rec.transient_setup_failures += 1
                    exhausted = (
                        rec.transient_setup_failures >= MAX_TRANSIENT_SETUP_ATTEMPTS
                    )
                if will_retry and not exhausted:
                    act.retrying(
                        exc,
                        rec.transient_setup_failures,
                        MAX_TRANSIENT_SETUP_ATTEMPTS,
                    )
                else:
                    act.failed(exc)
                # Setup is a transaction: endpoint construction, tenant
                # setup/warmup, residency registration, and compile-target
                # publication either all reach READY or all ownership is
                # removed through the ordinary record-vacate path. Include
                # cancellation — _to_thread_complete has already joined any
                # tenant thread before it propagates CancelledError here.
                try:
                    await self._rollback_failed_setup(rec)
                except BaseException:
                    logger.exception("failed to roll back incomplete setup for %s", spec.name)
                self._intent_transition(
                    intent_id,
                    (
                        pb.LIFECYCLE_INTENT_STATUS_CANCELED
                        if isinstance(exc, asyncio.CancelledError)
                        else pb.LIFECYCLE_INTENT_STATUS_FAILED
                    ),
                    (
                        pb.LIFECYCLE_INTENT_STAGE_COMPILING
                        if spec.compile is not None
                        else pb.LIFECYCLE_INTENT_STAGE_WARMING
                    ),
                    detail=_sanitize(str(exc))[:512],
                )
                if not isinstance(exc, Exception):
                    raise
                # Honest failure (th#581): a function whose model download /
                # pipeline setup fails must surface a terminal per-function
                # error to the hub, not sit in loading_functions forever
                # while the worker reports READY.

                if isinstance(exc, CompiledExecutionLaneUnavailableError):
                    self._mark_compile_setup_unavailable(rec, spec, str(exc))
                    self._on_state_change()
                self._mark_setup_failed(rec, exc, exhausted=exhausted)
                # pgw#1118/th#1773: name the pod's OWN warm/compile fault
                # before it leaves this boundary — the job path cannot tell
                # afterwards, and untyped it becomes the caller's `fatal`.
                typed = _typed_setup_fault(spec.name, act.phase_name, exc)
                if typed is not None:
                    raise typed from exc
                raise
            if rec.failed is not None:
                # Recovery (desired-state retry succeeded): lift the
                # per-function disable; the next StateDelta re-advertises.
                rec.failed = None
                for s in rec.specs:
                    self.unavailable.pop(s.name, None)
            rec.transient_setup_failures = 0
            rec.instance = instance
            rec.ready = True
            # pgw#1087: the FIRST user-visible timestamp. A ready record is an
            # instance that can answer a request — armed or eager — and on the
            # pgw#671 eager-first boot below it is reached long before any cell
            # exists. Paired with `compiled_swap` it gives the eager-serving
            # window, which is the interval the compiled-serving campaign is
            # trying to shrink and which nothing measured. Distinct from
            # `first_request_servable`, which additionally requires the hub to
            # have been told (a worker the hub cannot reach is not servable).
            boot_mod.mark_once(boot_mod.PHASE_EAGER_READY, function=spec.name)
            bg = rec.background_mint
            if bg is not None and bg.task is None:
                # pgw#671 eager-first boot: READY is advertised now (eager
                # tier); the self_mint_compile activity is handed to the
                # background driver — it stays RUNNING on the wire (the
                # hub's minting classification and "serving (optimizing in
                # background)" messaging key off exactly that) and only the
                # driver completes or fails it.
                bg.act = act
                bg.task = asyncio.create_task(
                    self._background_mint(rec, bg),
                    name=f"eager-mint-{spec.name}",
                )
            else:
                act.completed()
                # pgw#797: WARM-COMPLETE on the non-deferred paths. pgw#789 put
                # this milestone in `_background_mint`'s finally only, and
                # `rec.background_mint` is set ONLY under eager-first — so a
                # boot that minted inline, adopted a delivered cell, or served
                # eager without minting emitted no `warm_complete` at all. That
                # is most boots, and it was read as "compiled serving never
                # reached" when the truth was "never measured". Reached here,
                # setup is done and no deferred mint will run, so this boot's
                # serving tier is final NOW.
                self._mark_warm_complete(rec, spec.name)
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                pb.LIFECYCLE_INTENT_STAGE_READY,
            )
            self._clear_recovered_compile_failures(rec)
            self._on_state_change()
            return instance

    def _mark_warm_complete(self, rec: "_ClassRecord", function: str) -> None:
        """Latch this process's compiled-serving disposition as FINAL.

        pgw#924 removed the `warm_complete` BOOT PHASE this used to record. It
        was never a phase of the boot: live rows reached 4,863,664 ms — the
        deferred background mint finishes eighty minutes after the worker
        became servable — so summing it against the boot window was arithmetic
        on two different clocks, and the `first_request_servable` milestone
        already answers "when could this worker serve". What survives is the
        part that was always real: the mint-goal latch and the pgw#805 backstop
        below.
        """
        # pgw#1082: "is this record serving compiled" is ONE question with
        # ONE answer. Reading only `active_compile_ref` asks "does it serve a
        # NAMED artifact", and a JIT INTAKE arm names none BY CONSTRUCTION
        # (pgw#1010) — so this row could never clear for an intake pod even
        # while it served compiled, and 0.4.3 emitted `boot_ended_uncompiled`
        # on a healthy H100. `is_compile_armed` is the same reading
        # `_served_identity` and `serving_mode` take.
        armed = next(
            (t.active_compile_ref for t in rec.compile_targets.values()
             if t.active_compile_ref), "") or (
            "jit_intake" if any(
                compile_cache.is_compile_armed(t.pipeline)
                for t in rec.compile_targets.values()) else "")
        if armed or rec.background_mint is not None:
            return
        # pgw#805: a boot that DECLARED a compile target and ends with no
        # artifact and no mint in flight must say so. This is the terminal
        # backstop for the whole miss policy — the individual declines
        # (fleet_cells._fail_closed, mint_recipe, mint_supervisor) each name
        # themselves, and this one catches whatever route a future decline
        # takes. Five real L4 pods reached exactly this state and emitted
        # nothing at all, which reads identically to a hung worker.
        if not any(
            s.compile is not None and s.compile.family for s in rec.specs
        ):
            return
        activity_mod.emit_event(
            "self_mint_skipped",
            f"fn={function}: setup finished with a declared compile target, "
            f"no compiled artifact armed and no mint in flight — this worker "
            f"serves eager for the rest of its life and publishes no cell",
            phase=cell_adopt.EagerPhase.BOOT_ENDED_UNCOMPILED.value,
        )

    @contextmanager
    def _warmup_span(
        self, spec: EndpointSpec, rec: "_ClassRecord", inj: Any
    ) -> typing.Iterator[None]:
        """`warmup`, nested under the open `pipeline_load` (pgw#797).

        The armed/unarmed tag is the point of the row, not decoration: an
        UNARMED warm pays the compile, an ARMED one pays only the call, and the
        difference between the two IS what a cell saves on warmup. They are
        separate ROWS here rather than two code paths that happen to share a
        name, so the question is a `GROUP BY`.

        pgw#924: the row is emitted only when a warm forward actually ran. This
        bracket used to open unconditionally, and most setup slots plan no warm
        units at all (a non-inference spec, a class with no declared warmup, a
        bare component slot), so 240 of 240 live `warmup` rows and 245 of 245
        `warmup` activity events reported `duration_ms=0` — a bracket around
        nothing, read by everyone downstream as "boot warmup is free". Whether
        work ran is known only at the END of the body, so the row is recorded
        then, against the `pipeline_load` ordinal it belongs to, rather than
        opened on a guess.
        """
        if not boot_mod.in_boot():
            yield
            return
        armed_refs = sorted(
            {sel.ref for sel in inj.active_compile_artifacts.values() if sel.ref}
        )
        armed = bool(armed_refs)
        minting = bool(inj.pending_self_mints)
        started = time.monotonic()
        failure: Optional[BaseException] = None
        try:
            yield
        except BaseException as exc:
            failure = exc
            raise
        finally:
            warm_ms = int(round((time.monotonic() - started) * 1000))
            ran = self._warm_iterations
            self._warm_iterations = 0
            self._boot_warm_ms = warm_ms if ran else 0
            if ran or failure is not None:
                boot_mod.mark(
                    boot_mod.PHASE_WARMUP,
                    duration_ms=warm_ms,
                    function=spec.name,
                    # An armed warm is a LOAD-class cost; unarmed it is COMPILE.
                    klass=(boot_mod.CLASS_LOAD if armed
                           else boot_mod.CLASS_COMPILE),
                    ref=armed_refs[0] if armed_refs else "",
                    parent=rec.boot_load_ordinal or 0,
                    outcome=(boot_mod.OUTCOME_FAILED if failure is not None
                             else boot_mod.OUTCOME_OK),
                    reason=("" if failure is None
                            else (str(getattr(failure, "reason", "") or "")
                                  or type(failure).__name__)),
                    detail=(
                        f"armed={int(armed)} minting={int(minting)} "
                        f"forwards={ran} refs={','.join(armed_refs[:4])}"
                    ),
                )
                # th#1322's numeric home, so the boot span and the activity
                # stream agree on one number rather than two derivations.
                activity_mod.emit_event(
                    activity_mod.KIND_WARMUP_SUMMARY,
                    f"boot warmup for {spec.name} "
                    f"({'armed' if armed else 'unarmed'}, {ran} forward"
                    f"{'' if ran == 1 else 's'})",
                    phase=activity_mod.PHASE_WARMUP_FORWARD,
                    duration_ms=warm_ms,
                )

    def _warmup_plan(
        self, spec: EndpointSpec, rec: _ClassRecord,
    ) -> Tuple[list[Any], list[Any]]:
        """Return gw#470's authoritative per-handler warmup contract."""
        if spec.kind != "inference" or spec.cls is None:
            return [], []

        decl = getattr(spec.cls, _DECL_ATTR, None)
        if decl is None:
            # Not an @endpoint class (internally-constructed spec): no
            # declaration surface exists, so no synthesized warmup either.
            return [], []
        # Instance group = every spec sharing this instance: the code-table
        # siblings (matching instance_key) plus whatever this record has
        # already seen (covers pgw#532 derived per-pick specs).
        siblings: Dict[str, EndpointSpec] = {
            s.name: s for s in self.specs.values()
            if s.cls is spec.cls and s.instance_key == spec.instance_key
        }
        for s in rec.specs:
            siblings[s.name] = s
        siblings[spec.name] = spec
        return warmup_mod.plan(
            siblings.values(),
            decl_warmup=decl.warmup,
            has_warmup_method=False,
        )

    def _warm_contract_key(self, spec: EndpointSpec) -> Any:
        """The identity under which warm RUNS are shared across checkpoint
        instances (pgw#654 warm-tax fix): the class plus every per-slot fact
        that selects graphs or kernels — the precision lane (storage_dtype /
        dtype) — and NEVER the checkpoint ref itself. Two fine-tunes of one
        family land on the same key by construction; a lane rebind derives a
        different one."""
        rows = tuple(
            (
                slot,
                getattr(b, "storage_dtype", "") or "",
                getattr(b, "dtype", "") or "",
            )
            for slot, b in sorted(spec.models.items())
        )
        return (spec.cls, rows)

    def _compile_contract_names(
        self, spec: EndpointSpec, rec: _ClassRecord,
    ) -> set[str]:
        """Handler aliases this setup can attribute its warmup proof to."""
        if spec.cls is not None and callable(getattr(spec.cls, "warmup", None)):
            # Absent a completed object proof, a custom object-level warmup
            # has no per-handler attribution (the gw#603 attribution in
            # _install_compile_targets applies only to a PROVEN object).
            return {spec.name}
        return self._required_compile_names(spec, rec)

    def _required_compile_names(
        self, spec: EndpointSpec, rec: _ClassRecord,
    ) -> set[str]:
        """Non-skipped aliases that a mandatory compiled setup must prove.

        pgw#654: the derived plan dedupes warm RUNS per graph class, so an
        alias may own zero runs yet still be required — it is covered (and
        proven) by the sibling runs of its graph classes (``job.covers``)."""
        jobs, _skips = self._warmup_plan(spec, rec)
        names = {name for job in jobs for name in (job.covers or (job.spec.name,))}
        if spec.cls is not None and callable(getattr(spec.cls, "warmup", None)):
            # The custom object warmup directly proves only its initiating
            # handler. Other warmable aliases remain required and therefore
            # make W8A8 fail loud until they have attributable proof.
            names.add(spec.name)
        return names

    async def _run_synthesized_warmup(
        self, spec: EndpointSpec, rec: _ClassRecord, instance: Any,
        snapshots: Optional[Dict[WireRef, pb.Snapshot]],
        *,
        proof_objects: typing.Iterable[Any] = (),
        cold_proof_ids: typing.Collection[int] = (),
        allow_contract_skip: bool = False,
        armed_cell_refs: typing.Iterable[str] = (),
    ) -> _WarmupEvidence:
        """Run the declared per-handler warmup contract pre-READY.

        In addition to the successful call count, record which exact compiled
        objects served each handler. A sibling handler is never certified by
        another handler's cache hit merely because both share config or an
        instance. Output remains local and discarded.

        ``cold_proof_ids`` (gw#587 CORRECT FIX): object ids armed from a
        FRESH self-mint capture — for them a successful compiled call is the
        proof (there is nothing pre-existing on disk to HIT against; the
        capture this very call populates becomes the cell). Delivered cells
        keep requiring a real cache hit.

        ``allow_contract_skip`` (pgw#654 warm-tax fix, setup path only):
        permit the contract-keyed run memory to collapse this warmup to a
        single verification job when every planned run already executed in
        this process for the same warm contract. Inheritance is refused
        when a self-mint capture is pending (its cell must trace every
        graph) or when any armed cell is not yet proven in-process (a
        1-job run must never disprove a cell the full plan would have
        proven). The hot-adopt path never passes it: a NEW cell on a live
        instance requires its own full proof.
        """

        jobs, skips = self._warmup_plan(spec, rec)
        for skip in skips:
            # pgw#669: an illegal-combination row is a COVERAGE claim, not a
            # skipped handler — the handler still warms its legal graph set.
            if getattr(skip, "illegal", 0):
                logger.info("boot warm coverage for %s: %s",
                            skip.spec.name, skip.reason)
            else:
                logger.info("boot warmup skipped for %s: %s",
                            skip.spec.name, skip.reason)
        objects = tuple({id(obj): obj for obj in proof_objects}.values())
        memory = self._warm_contract_runs.setdefault(
            self._warm_contract_key(spec), set())
        armed_refs = tuple(armed_cell_refs)
        # Tracing == some object under proof still needs the full class x
        # bucket cross-product, because its graphs must trace INTO something:
        # a dynamo lane whose per-class FX cache-hit ledger is its only
        # detector of a silent recompile, or a fresh self-mint capture every
        # declared graph must land in.
        #
        # pgw#1184 CUTS THE EXPORTED LANE OUT OF IT (th#1834 Phase 4). An
        # armed `.pt2` is ahead-of-time machine code for this exact
        # sm x toolchain: it performs no FX lookup, there is no ledger to
        # move, and §4.31 already ruled that warmup "never made a cell faster,
        # it only checked it." The 18 runs sdxl was paying per handler bought
        # exactly one thing the arm did not already have — a BOOT-TIME census
        # of which declared classes the dispatcher could route to — and that
        # census is deleted with them (see `_arm_class_coverage`, and the
        # commit message for why 16 extra full generates is not what that
        # question is worth). The per-shape truth still arrives, per request,
        # typed, at the ingress that already refuses a class BY NAME and
        # charges the request `fallback_reason=ingress_refused`.
        tracing = bool(cold_proof_ids) or any(
            not aot_serve.holds_exported_cell(obj) for obj in objects)
        skip_ok = (
            allow_contract_skip
            and not cold_proof_ids
            and all(compile_cache.compiled_graph_proven_in_process(r) for r in armed_refs)
        )
        run_jobs, warm_mode = warmup_mod.select_runs(
            jobs,
            tracing=tracing,
            executed=(memory if skip_ok else frozenset()),
        )
        if warm_mode != "full":
            logger.info(
                "boot warm plan for %s: mode=%s runs=%d/%d planned "
                "(contract-keyed warm memory holds %d graph keys)",
                spec.name, warm_mode, len(run_jobs), len(jobs), len(memory))
        evidence = _WarmupEvidence()
        # pgw#735: two compiled backends, two proofs. Dynamo proves by FX
        # cache hits, an EXPORTED artifact by its own
        # invocations — an exported cell performs no FX lookup at all, so a
        # cache-hit requirement would score every honest .pt2 adoption as a
        # failure. Never synthesize a hit counter for it: this is the one path
        # whose whole job is to detect a lie about serving compiled.
        start_counts = {
            id(obj): (
                compile_cache.execution_count(obj),
                aot_serve.execution_count(obj),
            )
            for obj in objects
        }
        # pgw#654 coverage attribution: runs prove GRAPH CLASSES; an alias
        # is proven on an object once ALL of its graph classes proved there.
        proven_keys: Dict[int, set] = {}
        # pgw#844: which objects proved through the EXPORTED lane. An exported
        # artifact refuses a shape outside its declared envelope BY NAME and
        # serves that call eager while staying armed, so a class it did not serve is
        # a per-shape posture, not a silent recompile — which is what lets the
        # attribution below be per-class for this lane and stay all-or-nothing
        # for dynamo, where an unproven class means an unannounced recompile.
        #
        # pgw#1184: the exported lane no longer runs the full plan, so this set
        # is seeded from the LANE rather than discovered by executing 18
        # generates. An armed exported object is on it from the start; what
        # `_one` still adds is nothing this lane needs, and what the
        # attribution below still does with it is attribute every alias
        # (§4.31: the arm has no warm prerequisite).
        exported_proof_ids: set = {
            id(obj) for obj in objects if aot_serve.holds_exported_cell(obj)
        }

        async def _one(wj: Any, build: Any, mode: str, *, variant: bool) -> bool:
            """One warmup forward; False = OOM, stop warming."""
            before = {
                id(obj): (
                    compile_cache.execution_count(obj),
                    aot_serve.execution_count(obj),
                )
                for obj in objects
            }
            handler_kwargs = await self._handler_kwargs(wj.spec, snapshots or {})
            t0 = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="gw-warmup-") as tmp:
                try:
                    payload = build(tmp)
                except IllegalCombination as exc:
                    # pgw#669: the endpoint declares this field combination
                    # outside its contract. The derived plan already filters
                    # these at plan time; reaching here means a media-variant
                    # row or a constraint only expressible against the fully
                    # built payload. Not a boot failure either way — the
                    # combination is not a servable request.
                    logger.info(
                        "boot warmup %s (%s): combination declared illegal, "
                        "not warmed: %s", wj.spec.name, mode, exc)
                    return True
                except Exception as exc:
                    if not variant:
                        raise
                    # A variant that cannot construct a valid payload is a
                    # schema mismatch, never a boot failure.
                    logger.warning(
                        "coverage warmup %s (%s) skipped: %s",
                        wj.spec.name, mode, exc)
                    return True
                if payload is None:
                    return True  # variant base already carries media
                ctx: RequestContext[Any] = warmup.warm_context(
                    wj.spec, request_id=f"boot-warmup-{wj.spec.name}",
                    local_output_dir=tmp,
                    execution_lane=self._served_execution_lane(wj.spec),
                    config=self._effective_config(wj.spec))
                try:
                    await self._invoke_warmup(wj.spec, instance, ctx, payload, handler_kwargs)
                except Exception as exc:
                    if not is_cuda_oom(exc):
                        raise
                    # A warmup OOM must not take the function down: the
                    # runtime fit ladder (gw#521) still serves it degraded
                    # on the first real request. Flush and stop warming.
                    logger.warning(
                        "boot warmup %s OOMed (%s) — skipping remaining "
                        "warmups; the first-request fit ladder owns this",
                        wj.spec.name, exc)
                    # pgw#677 reopen: a truncated plan is recorded — the
                    # caller withholds any pending mint's publish (the
                    # partial cell would brick adopters) — and, when this
                    # boot IS minting, the truncation reaches the hub as a
                    # typed event instead of dying in pod logs.
                    evidence.aborted = (
                        f"cuda_oom at warm unit "
                        f"{evidence.count + 1} ({wj.spec.name}): {exc}")
                    if tracing:
                        activity_mod.emit_event(
                            "self_mint_abort",
                            f"boot warm plan cut short: {evidence.aborted}",
                            phase="warmup_oom",
                        )
                    if torch is not None and cuda_ready():
                        torch.cuda.empty_cache()
                    return False
            evidence.count += 1
            if boot_mod.in_boot():
                # pgw#924: this counter is what makes the `warmup` boot row
                # honest — it is the difference between "the warm pass cost
                # nothing" and "there was no warm pass". Counted INSIDE the
                # gate: a steady-state warm hours later has no boot row to
                # belong to, and letting it bump the counter would misreport
                # the next boot's warmup.
                #
                # The per-forward `warmup_iteration` row that used to be
                # recorded here is DELETED. It was wired and never fired once
                # on either live stack (0 rows against 240 `warmup` rows),
                # because the plan that reaches this line is empty on the
                # shipping path; a per-iteration decomposition of a pass that
                # does not run is not a measurement anyone can read.
                self._warm_iterations += 1
            for obj in objects:
                calls_before, aot_before = before[id(obj)]
                inductor_proven = (
                    compile_cache.execution_count(obj) > calls_before
                    and (
                        compile_cache.cache_hit_count(obj) > 0
                        or id(obj) in cold_proof_ids
                    )
                )
                # pgw#735: an exported artifact proves itself by executing —
                # and by still being armed, so a call that ended in a revoked
                # (failed) artifact cannot count as proof.
                aot_proven = aot_serve.proven_since(obj, aot_before)
                if aot_proven:
                    exported_proof_ids.add(id(obj))
                if inductor_proven or aot_proven:
                    proven_keys.setdefault(id(obj), set()).add(wj.graph_key)
            logger.info(
                "boot warmup %s (%s): %.1fs",
                wj.spec.name, mode, time.monotonic() - t0)
            return True

        for wj_index, wj in enumerate(run_jobs, start=1):
            activity_mod.current_phase(
                activity_mod.PHASE_WARMUP_FORWARD, wj_index, len(run_jobs))
            act = activity_mod.current()
            if act is not None:
                act.counter("warmup:jobs", progress_mod.UNIT_STEPS,
                            total=len(run_jobs)).set_done(wj_index)
            mode = (
                "verify" if warm_mode == "verify"
                else "declared" if wj.declared else "synthesized"
            )
            if not await _one(wj, wj.build, mode, variant=False):
                return evidence
            memory.add(wj.graph_key)

        def _unexercised() -> list[Any]:
            return [
                obj for obj in objects
                if compile_cache.execution_count(obj) == start_counts[id(obj)][0]
                and aot_serve.execution_count(obj) == start_counts[id(obj)][1]
            ]

        # gw#614 coverage pass: an input-routed sibling lane the planned
        # warmups never reached (e.g. edit needing an input image) leaves a
        # compile object at calls=0 — the mint then withholds publish
        # (gw#612) and an adopt arms it unproven. Synthesized media variants
        # of the same base payloads exercise those lanes with matching
        # compile-key derivation (only the media field differs).
        if warm_mode == "full" and run_jobs and _unexercised():
            variant_jobs = [
                (wj, label, build)
                for wj in run_jobs
                for label, build in warmup_mod.media_variants(
                    wj.spec.payload_type, wj.build)
            ]
            total = len(run_jobs) + len(variant_jobs)
            for v_index, (wj, label, build) in enumerate(
                    variant_jobs, start=len(run_jobs) + 1):
                if not _unexercised():
                    break
                activity_mod.current_phase(
                    activity_mod.PHASE_WARMUP_FORWARD, v_index, total)
                act = activity_mod.current()
                if act is not None:
                    act.counter("warmup:jobs", progress_mod.UNIT_STEPS,
                                total=total).set_done(v_index)
                if not await _one(wj, build, label, variant=True):
                    break
        if warm_mode == "verify" and evidence.count:
            # Contract inheritance (pgw#654 warm-tax fix): the verification
            # run proving this object, together with every graph key already
            # EXECUTED in this process for the same warm contract (the skip
            # precondition), certifies the full plan on this object —
            # handler code paths and graph classes are instance-invariant;
            # only the weights changed. Without this, a 1-job verify would
            # attribute a single alias and fail the mandatory-lane
            # completeness gate that the full plan (whose runs would land on
            # dynamo's in-memory code anyway) passes.
            all_keys = {wj.graph_key for wj in jobs}
            for obj_id in list(proven_keys):
                proven_keys[obj_id] |= all_keys
        # Coverage attribution (pgw#654): name -> its full graph-class set,
        # from the plan; an alias attributes to an object only when EVERY
        # one of its classes proved there — a partially-traced alias is
        # never certified by one sibling run.
        keys_by_name: Dict[str, set] = {}
        for wj in jobs:
            for name in (wj.covers or (wj.spec.name,)):
                keys_by_name.setdefault(name, set()).add(wj.graph_key)
        for obj_id in set(proven_keys) | exported_proof_ids:
            proven = proven_keys.get(obj_id, set())
            # pgw#844's ORIGINAL FINDING, now answered at the lane instead of
            # per class (pgw#1184). The measured shape (attempt twelve, pod
            # o0legpgj5olhic): a regional sdxl cell armed all 72 entries,
            # dispatched 1024x1024 correctly, and refused the other eight
            # aspect buckets `entry_ambiguous`. Those eight classes went
            # unproven, the all-or-nothing rule attributed NO alias, the
            # target was omitted `target_applicability_incomplete`, and the
            # boot ended `boot_ended_uncompiled` — so the ONE bucket that was
            # armed, correct and unambiguous served eager too. One
            # undispatchable shape cost the pod every shape.
            #
            # `boot_ended_uncompiled` must mean "nothing is dispatchable", not
            # "something wasn't". An exported artifact refuses a shape it
            # cannot serve BY NAME, counts it, emits `aot_ingress_refused`,
            # charges the request `fallback_reason=ingress_refused`, and stays
            # armed — so the degradation is per shape and fully visible.
            #
            # pgw#844 bought that with a per-class warm census. §4.31 deleted
            # the premise underneath it: the arm has NO warm prerequisite, so
            # an armed exported object is attributed to every alias its plan
            # covers, and the count of classes it served AT BOOT decides
            # nothing. That is also what makes the eager plan sound above —
            # attribution can no longer depend on how many runs happened.
            #
            # Dynamo keeps the strict rule verbatim: there an unproven class
            # means an unannounced recompile at serve time, which is silent.
            if obj_id in exported_proof_ids:
                names = {name for name, keys in keys_by_name.items() if keys}
            else:
                names = {
                    name for name, keys in keys_by_name.items()
                    if keys and keys <= proven
                }
            if names:
                evidence.functions_by_object[obj_id] = names
        return evidence

    async def _invoke_warmup(
        self, spec: EndpointSpec, instance: Any, ctx: "RequestContext",
        payload: Any, kwargs: Dict[str, Any],
    ) -> None:

        bound = getattr(instance, spec.attr_name)
        call_kwargs = {spec.ctx_param: ctx, spec.payload_param: payload, **kwargs}
        # pgw#676: warm forwards (foreground eager warm AND background mint
        # seeds) get the same signal-death attribution as real requests.
        inflight_token = postmortem.note_inflight(
            "warmup", spec.name, request_id=str(ctx.request_id or ""))
        try:
            # pgw#1265: mint seeds and the boot warm forward go through here
            # and nowhere else, so this is what gives the adopt's headroom
            # verdict a measurement on a pod that has not served a tenant
            # request yet — which is every pod that mints at boot.
            with adopt_fit.forward_watermark():
                if spec.is_async_gen:
                    async for _ in bound(**call_kwargs):
                        pass
                elif spec.is_async:
                    await bound(**call_kwargs)
                else:
                    def _consume() -> None:
                        out = bound(**call_kwargs)
                        if spec.output_mode == "stream":
                            for _ in out:
                                pass

                    await _to_thread_complete(_consume)
            # pgw#1199: THE proof, recorded where it already happens. This call
            # is the endpoint's own handler, running on the RESIDENT pipeline
            # with real checkpoint values, and every warm path in this process
            # goes through it. A delegated mint reads the record instead of
            # materialising a second copy of the weights in its child to
            # re-prove the same sentence with random values (§4.33 steps 4-5).
            handler_proof.record(
                spec.name, f"boot warm forward {spec.name!r} (real weights)")
        finally:
            postmortem.clear_inflight(inflight_token)

    def _mark_setup_failed(
        self, rec: _ClassRecord, exc: BaseException, *, exhausted: bool = False,
    ) -> None:
        if isinstance(exc, _TRANSIENT_SETUP_ERRORS) and not exhausted:
            # Transient pressure (disk GC frees space / warm-tier RAM drains /
            # the hub re-mints a snapshot): fail the op RETRYABLE, never
            # disable the function.
            return
        if isinstance(exc, _TRANSIENT_SETUP_ERRORS):
            # gw#661: the budget is spent. "Retryable" described the class of
            # error, not an infinite entitlement — a condition that survives
            # every attempt is this function's terminal truth (th#1159's
            # genuinely-unfittable VRAM lane is the case this exists for).
            reason, axes = "retry_exhausted", {}
        elif isinstance(exc, HardwareUnmetError):
            reason = getattr(exc, "reason", "hardware_unmet")
            axes = {str(k): str(v) for k, v in (exc.axes() or {}).items()}
        else:
            reason, axes = "setup_failed", {}
        detail = _sanitize(f"{type(exc).__name__}: {exc}")
        rec.failed = detail
        for s in rec.specs:
            self.unavailable[s.name] = (reason, detail, axes)
        self._on_state_change()

    def mark_ref_unmaterializable(self, ref: str, detail: str) -> List[str]:
        """pgw#655: a model this worker fetches ITSELF never landed — gate
        every function that statically binds it.

        The alternative (log and continue) is what produced the live wedge:
        the worker walked on to READY, advertised the function, and the hub
        dispatched paid GPU jobs that each re-discovered the missing model as
        a per-request load failure. A worker never advertises a function whose
        model is absent. This is deliberately NOT a process kill — a sibling
        function whose model DID land keeps serving, exactly as a hardware
        gate leaves the rest of the endpoint alive.

        Hub-resolved slots are excluded: their refs arrive by delivery
        (pgw#532), so the worker never prefetches them and their absence at
        boot is not a failure.
        """
        gated: List[str] = []
        for name, spec in self.specs.items():
            bound = any(
                ref == wire_ref(binding)
                for slot, binding in spec.models.items()
                if slot not in spec.slots
            )
            if not bound:
                continue
            self.unavailable[name] = (
                "model_unavailable", _sanitize(detail), {"ref": ref},
            )
            gated.append(name)
        if gated:
            self._on_state_change()
        return sorted(gated)

    @staticmethod
    def _record_has_setup_ownership(rec: _ClassRecord) -> bool:
        """Whether a non-READY record owns anything that needs teardown."""
        return bool(
            rec.instance is not None
            or rec.server is not None
            or rec.held_refs
            or rec.held_objects
            or rec.shared_keys
            or rec.compile_targets
        )

    async def _rollback_failed_setup(self, rec: _ClassRecord) -> None:
        """Remove every provisional owner left by an incomplete setup.

        The normal vacate path is the single teardown implementation: it
        invokes endpoint shutdown, stops an engine server, releases loaded
        residency objects and shared-component holds, clears compile targets,
        and emits the resulting state. Failed setup can also leave freed
        staging buffers in PyTorch's pinned-host cache, so return those unused
        blocks after the owners have gone.
        """
        # gw#624: even with no record ownership (cancellation landed inside
        # _injection_kwargs before the load finished), the aborted attempt's
        # partially built modules are almost always in reference cycles and
        # can be pinned by the propagating exception's traceback until it
        # dies — schedule a purge so the NEXT attempt provably starts from
        # baseline instead of stacking a fresh multi-GB load on top
        # (observed live: 5 cancelled retries climbed one worker to 83.86GB
        # VRAM / 97% container RAM on an 80GB card).
        self._pending_alloc_purge = True
        if not self._record_has_setup_ownership(rec):
            return
        async with self._load_lock:
            released = await vacate_record(rec, self.teardown_seam)
        await aflush_memory()
        released_pinned = await asyncio.to_thread(
            release_unused_pinned_host_cache)
        logger.info(
            "rolled back incomplete setup refs=%s pinned_host_bytes=%d",
            released,
            released_pinned,
        )

    async def _purge_cancelled_setup_allocations(self) -> None:
        """gw#624: run once at the start of a setup attempt that follows a
        rolled-back one. By now the previous attempt's exception has died,
        so a full gc pass actually frees its cycle-held modules; only then
        can empty_cache return their VRAM to the allocator."""
        if not getattr(self, "_pending_alloc_purge", False):
            return
        self._pending_alloc_purge = False
        freed_before = time.monotonic()
        await aflush_memory()
        logger.info(
            "purged prior cancelled-setup allocations in %.1fs",
            time.monotonic() - freed_before,
        )

    async def _setup_locked(
        self, spec: EndpointSpec, rec: _ClassRecord,
        snapshots: Optional[Dict[WireRef, pb.Snapshot]],
        *,
        intent_id: str = "",
        arm: Optional[_ArmOrder] = None,
    ) -> Any:
        assert spec.cls is not None  # guarded by ensure_setup
        setup_slots = self._setup_slots(spec)
        # pgw#677 reopen: open the setup-scoped execution-lane window so
        # EVERY arm path (slot injection AND self-loaded arm_compile via
        # ArmingScope) stamps its pipelines with the hub-resolved th#913
        # lane — the ONE serveability brain compile_cache.mandatory_serving
        # reads. ContextVars ride to_thread, so the tenant setup thread and
        # its arms inherit the window.

        _setup_exec_execution_lane, _setup_execution_lane_pinned = "", False
        for _slot in setup_slots:
            _setup_exec_execution_lane, _setup_execution_lane_pinned = (
                self._execution_lane_pick_for_ref(
                    wire_ref(spec.models[_slot])))
            if _setup_exec_execution_lane:
                break
        _execution_lane_token = _cc_execution_lane._SETUP_EXEC_EXECUTION_LANE.set(_setup_exec_execution_lane)
        _pin_token = _cc_execution_lane._SETUP_EXEC_EXECUTION_LANE_PINNED.set(_setup_execution_lane_pinned)
        try:
            return await self._setup_locked_inner(
                spec, rec, snapshots, intent_id=intent_id,
                setup_slots=setup_slots, arm=arm)
        finally:
            _cc_execution_lane._SETUP_EXEC_EXECUTION_LANE_PINNED.reset(_pin_token)
            _cc_execution_lane._SETUP_EXEC_EXECUTION_LANE.reset(_execution_lane_token)

    async def _setup_locked_inner(
        self, spec: EndpointSpec, rec: _ClassRecord,
        snapshots: Optional[Dict[WireRef, pb.Snapshot]],
        *,
        intent_id: str = "",
        setup_slots: List[str],
        arm: Optional[_ArmOrder] = None,
    ) -> Any:
        assert spec.cls is not None
        # gw#494: residency keys for this setup are derived ONCE, here, in
        # resolved space; downloads, booking and the record's held_refs all
        # use these exact strings (a HelloAck rebind during an await below
        # cannot split download/booking/teardown identities).
        slot_refs: Dict[str, WireRef] = {
            slot: wire_ref(spec.models[slot]) for slot in setup_slots
        }
        slot_identities: Dict[str, _ResidencyIdentity] = {}
        # pgw#974: ONE resolution per slot — the binding and the tree its
        # bytes were materialized into. Written by a single statement per
        # slot, so the two cannot drift apart or arrive without one another.
        resolved_slots: Dict[str, MintSlot] = {}
        self._intent_transition(
            intent_id,
            pb.LIFECYCLE_INTENT_STATUS_RUNNING,
            pb.LIFECYCLE_INTENT_STAGE_LOADING_HOST,
        )
        for slot in setup_slots:
            binding = spec.models[slot]
            ref = slot_refs[slot]
            snap = (snapshots or {}).get(ref)
            materialized = await self.store._materialize_local(
                ref, snap, binding=binding)
            slot_identities[slot] = materialized.identity
            resolved_slots[slot] = MintSlot(
                ref=binding, path=str(materialized.path))
        paths: Dict[str, str] = {
            slot: res.path for slot, res in resolved_slots.items()}
        topology_eager = self._eager_only_reason()
        # pgw#1142 / §4.32 item 4. The order joins the topology reason for
        # every "do not go looking for a cell" decision below — this is the
        # gate that runs BEFORE the hub round trip and the materialize, so an
        # operator who says "stop compiling" is obeyed at the first boot phase
        # rather than at the arm, having paid for a download in between.
        #
        # It deliberately does NOT join the REFUSAL two blocks down. The two
        # are different in kind: a degree>1 topology CANNOT run the named cell
        # (the collectives would hang), so a spec naming one is unsatisfiable
        # and must fail typed; an operator order is a decision that the pod
        # serve eager, and `arm_ordered` obeys it by arming nothing and
        # serving — killing the function instead would be the opposite of what
        # was asked for, and it would not be reversible.
        ordered_eager = serve_posture.block()
        eager_only = topology_eager or ordered_eager
        if eager_only and spec.compile is not None:
            logger.info("%s: %s", spec.name, eager_only)
        # The ONLY source of a pre-materialized artifact is §4.27 boot-adopt
        # (pgw#1206 D deleted the Plan head that was the other one). The
        # connected snapshot scan that used to run here is deleted — the hub
        # no longer attaches cells to snapshots, and a worker that could pick
        # one would be a second resolver.
        if arm is not None and arm.backend == "aot_cell" and topology_eager:
            raise compile_cache.CompiledExecutionLaneUnavailableError(
                f"the spec names an exact cell but this pod cannot arm one: "
                f"{topology_eager}")
        compile_selection = arm.selection if arm is not None else None
        compile_artifact = compile_selection.path if compile_selection else None
        # §4.27 steps 1-3 (pgw#1089/pgw#1090): with no Plan-named artifact, this
        # pod derives its OWN cell key from code alone and asks the hub by that
        # key BEFORE `setup()` puts a weight in this process. On a hit the
        # answer becomes an ordinary `_ArmOrder`, so the adopted cell runs the
        # Plan path's gates and not one gate fewer.
        #
        # This is what makes boot-time adoption possible at all: the hub's other
        # resolver only VERIFIES a cell the worker already armed, so a cold pod
        # advertises nothing, is named nothing, and never adopts.
        #
        # OWED (pgw#1091's overlap box): the derivation runs HERE, after
        # `_materialize_local` finished the weights download, so it does not yet
        # RACE the fetch the way §4.27 step 4 asks. It is already off the
        # request path — no dispatch has occurred — and moving it earlier is a
        # restructure of this method's await order, not of the derivation.
        # pgw#1127 S2: the `ck1` key THIS MACHINE's own store answered on. Not
        # an `_ArmOrder`: a self-minted cell carries no hub receipt and no
        # publisher org, so `arm_ordered` would refuse it
        # `receipt_gate_unconfigured`. It is an ADDRESS, handed to the arming
        # brain as a second lookup route into the same CAS the arm-token memo
        # addresses — one key, two routes, and `_arm_exported_cell` is the one
        # gate at the end of both.
        boot_local_key = ""
        if arm is None and spec.compile is not None and not eager_only:
            adopts = await asyncio.to_thread(
                self._boot_adopt, spec, resolved_slots)
            # pgw#1176: the boot resolves ONE outcome per declared graph class.
            # Coverage accretes, so several hits are the expected shape and
            # each is armed on its own.
            #
            # UNFINISHED, AND LOUD RATHER THAN SILENT (owner: pgw#1176; expiry:
            # before this branch opens a PR). This call site still builds ONE
            # `_ArmOrder`, so only the first hit is armed here. The remaining
            # hits are NOT dropped quietly — they are named on the wire below,
            # and the fix is to carry them on the order so `_enable_compiled`
            # arms each into the same registry after the pipeline is up, which
            # is what `aot_serve.arm_entry` already supports. A silent subset
            # here would be the exact defect this whole change deletes.
            resolved = [o for o in adopts if o.adoption is not None]
            adopt = resolved[0] if resolved else (
                adopts[0] if adopts else boot_adopt.BootAdoptOutcome())
            boot_local_key = adopt.local_key
            if adopt.adoption is not None:
                got = adopt.adoption
                compile_artifact = got.artifact
                arm = _ArmOrder.for_artifact(
                    path=got.artifact, ref=got.ref,
                    snapshot_digest=got.snapshot_digest,
                    expected=got.expected,
                    publisher_org=got.cell.publisher_org,
                    # pgw#1122: this order is the POD's, not the hub's.
                    adopt=adopt,
                    # pgw#1176: every OTHER class this boot resolved, armed
                    # into the same registry after this one.
                    extra=tuple(
                        (got_other.artifact, got_other.expected,
                         got_other.cell.publisher_org)
                        for got_other in (
                            o.adoption for o in resolved[1:]
                            if o.adoption is not None)))
                compile_selection = arm.selection
        elif arm is None and spec.compile is not None:
            # pgw#1116: a compiled family that boots WITHOUT asking is a fact
            # somebody has to be able to read. This is the only branch where
            # that is correct by design (pgw#775 forbids arming here at all) —
            # so it says so, rather than being the ninth way to look like a pod
            # that quietly self-minted.
            boot_adopt.refused(
                # pgw#1142: WHICH eager, named. A pod that never asked because
                # its topology forbids arming and a pod that never asked
                # because an operator said so are two different boots, and the
                # second one can be undone.
                boot_adopt.OPERATOR_EAGER_ONLY if not topology_eager
                else "eager_only",
                eager_only,
                family=str(getattr(spec.compile, "family", "") or ""),
                function=str(spec.name or ""))
        # Loads serialize: concurrent setups would cross-contaminate each
        # other's allocator deltas and place_pipeline's free-VRAM reads.
        async with self._intent_lock(
            intent_id,
            self._load_lock,
            operation=f"model load for {spec.name}",
            stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
            reason=pb.LIFECYCLE_WAIT_REASON_LOAD_LOCK,
            resume_stage=pb.LIFECYCLE_INTENT_STAGE_LOADING_DEVICE,
        ):
            # gw#624: a prior cancelled attempt's cycle-held modules must be
            # collected BEFORE this attempt allocates, or retries stack
            # partial loads until OOM.
            await self._purge_cancelled_setup_allocations()
            await self._make_room_for(spec, setup_slots)
            # VRAM make-room may demote the old pipeline into host RAM. Admit
            # the incoming load only AFTER that transition so the probe sees
            # the actual post-demotion pressure (pgw#541).
            await self._ensure_host_ram_for(spec, paths)
            instance = spec.cls()
            # Stamp provisional ownership BEFORE tenant setup/warmup. The
            # record is not advertised until rec.ready becomes true, but an
            # exception or cancellation can now tear down the exact instance
            # and exact resolved refs instead of losing them in stack locals.
            rec.instance = instance
            rec.held_refs = sorted(set(slot_refs.values()))
            rec.held_snapshot_digests = {
                slot_refs[slot]: identity[0]
                for slot, identity in slot_identities.items()
                if slot in slot_refs and identity[0]
            }
            rec.held_bindings = sorted(
                (slot, ref, rec.held_snapshot_digests.get(ref, ""))
                for slot, ref in slot_refs.items()
            )
            setup = getattr(instance, "setup", None)
            inj = _InjectionResult(kwargs={}, loaded={})

            vram_before = cuda_allocated_bytes()
            if spec.runtime:
                rec.server = await self._boot_engine_server(spec, paths)
            if callable(setup):
                inj = await self._injection_kwargs(
                    spec, setup, paths, server=rec.server,
                    compile_selection=compile_selection,
                    snapshots=snapshots,
                    slot_identities=slot_identities,
                    arm=arm, boot_local_key=boot_local_key)
                rec.shared_keys.extend(inj.shared_keys)
                # pgw#517: a self-loading (str/Path-slot) endpoint builds its
                # own pipeline inside setup() and the executor never sees it
                # to arm compile automatically (the branch above only fires
                # for class-annotated slots) — hold the arming scope open so
                # a `gen_worker.arm_compile(pipe)` call from inside setup()
                # reaches the same cache-artifact-gated policy. No-op when
                # spec.compile is None.
                arming_scope = provision.ArmingScope(
                    # pgw#775: a None cell makes the scope a no-op, so an
                    # endpoint's own `gen_worker.arm_compile(pipe)` inside
                    # setup() cannot arm a compile — and cannot self-mint —
                    # on a context-parallel pod.
                    None if eager_only else spec.compile_cell(),
                    self.store._cache_dir, compile_artifact,
                    enable=functools.partial(
                        self._arming_enable,
                        subject=slot_subjects(
                            resolved_slots,
                            {name: ident[0]
                             for name, ident in slot_identities.items()})),
                )
                # pgw#1104: NOT gated on spec.compile — a serve-time recipe
                # quantizes whether or not this release compiles, and the lane
                # it applied is what every request then executes.
                applied_lane_scope = provision.AppliedLaneScope()
                applied_attn_scope = provision.AppliedAttentionScope()
                with arming_scope, applied_lane_scope, applied_attn_scope:
                    if asyncio.iscoroutinefunction(setup):
                        await setup(**inj.kwargs)
                    else:
                        await _to_thread_complete(setup, **inj.kwargs)
                self._record_applied_lanes(spec, rec, applied_lane_scope.applied)
                self._record_applied_attention(rec, applied_attn_scope.applied)
                # arm_compile() is the sole unambiguous ownership seam for a
                # self-loaded pipeline. Such a pipeline may be built from any
                # path-valued setup input, so freeze every self-loaded slot
                # into its applicability rather than guessing one later.
                self_loaded_slots = tuple(
                    slot for slot in setup_slots
                    if isinstance(inj.kwargs.get(slot), (str, Path))
                )
                # pgw#1078: …and a WORKER-loaded slot object that only became
                # compile-capable DURING setup owns itself. A lazy loader (a
                # `ModularPipeline` whose weight-bearing components hydrate
                # inside setup) has no compile target at injection time, so the
                # automatic branch skips it; the endpoint then hydrates and
                # calls `arm_compile(pipeline)`, which lands here. Attributing
                # that pipeline to `self_loaded_slots` — EMPTY for a
                # class-annotated slot — gave `_install_compile_targets` a
                # candidate with no owned slots, hence no bindings, hence
                # `target_applicability_incomplete`: the arm succeeded and NO
                # target was installed, so the guard was never bound, the
                # hot-swap router was never enabled, and every request reported
                # `+eager` (ie#632, minimax-h3 0.4.2).
                injected_slot_of = {
                    id(obj): slot
                    for slot in setup_slots
                    if (obj := inj.kwargs.get(slot)) is not None
                    and not isinstance(obj, (str, Path))
                }
                scope_mints = arming_scope.self_mints
                for bug in arming_scope.selection_bugs.values():
                    # th#1031: the fleet policy already self-minted a working
                    # cell instead of aborting — still report the th#883
                    # invariant loudly.
                    await self._report_cell_selection_bug(
                        spec, compile_selection, bug)
                for pipe, armed in arming_scope.objects:
                    if armed:
                        if not any(p is pipe for p in inj.armed_objects):
                            # pgw#1093: BEFORE the re-scan below can drop it.
                            # This object is armed compiled code; if nothing
                            # ends up owning it, that is the impossible state
                            # the end-of-setup invariant refuses — never a
                            # silent skip.
                            inj.armed_objects.append(pipe)
                        # pgw#1078: this arm DISPROVES whatever the injection-
                        # time attempt concluded about the same object — that
                        # attempt ran before setup hydrated it.
                        #
                        # pgw#1093 moved the clear ABOVE the re-scan gate. It
                        # used to sit after it, so an armed object that failed
                        # the re-scan kept the stale injection-time
                        # `no_compile_target` — and first-token-wins then let
                        # that stale token outrank the REAL cause the install
                        # was about to name. Same wrong-cause defect pgw#1078
                        # fixed, on the one path its fix could not reach.
                        inj.eager_postures.clear()
                    if not compile_cache.has_compile_target(pipe, spec.compile):
                        continue
                    owning = injected_slot_of.get(id(pipe))
                    inj.add_compile_object(
                        pipe, (owning,) if owning else self_loaded_slots)
                    mint = scope_mints.get(id(pipe))
                    selection = _selection_for(compile_selection, mint)
                    if getattr(mint, "delegated", False):
                        # pgw#784: see the slot path above — recorded, but
                        # never advertised as an active artifact.
                        inj.pending_self_mints[id(pipe)] = mint
                    elif armed and selection is not None:
                        inj.active_compile_artifacts[id(pipe)] = selection
            # pgw#671 eager-first boot (worker half of th#1187): a fresh
            # self-mint on an eager-compatible lane no longer gates READY.
            # Stash the arm-time placeholder selections (their digest is
            # empty until finalize — the install below must not see a half
            # identity), enable the pgw#622 routers NOW (pre-proof) so the
            # eager warm subset and every real request route EAGER while the
            # background thread compiles each signature into the pending
            # capture, and defer trace/proof/finalize/publish to the
            # background driver spawned after READY. Everything downstream
            # of this block then follows the plain-eager shape naturally:
            # no active artifacts => no exclusive-GPU window, eager warm
            # selection, no proof loop, targets registered active-less
            # (advertising the requested cell key for peer adoption).
            # §4.33 / pgw#1175: a `mint_budget.probe` gate stood here and
            # could turn a boot's eager-first capture off on an arithmetic
            # whose activation term was a quarter of the RESIDENT SET — a
            # fraction nobody measured, against a card whose free figure
            # already excluded those weights. Nothing predicts VRAM: the boot
            # arms, the compile children run weight-free, and a genuine
            # shortfall comes back as a classified child death.
            eager_first = self._eager_first_eligible(spec, inj)
            delegated_mints = _delegated_pendings(inj.pending_self_mints)
            if delegated_mints and not eager_first:
                # pgw#784: nothing is armed on these pipes, so the foreground
                # compile-then-serve path below cannot drive them. Discard the
                # obligation and serve eager with the cell absent — the honest
                # miss policy — rather than run a warmup proof against an
                # unarmed pipeline. (fleet_cells.delegation_refusal already
                # refused to delegate anything that MUST serve compiled, so
                # this is the custom-warmup / mixed-delivered-artifact
                # remainder.)
                from . import fleet_cells as _fc_undelegate

                for _pid, _mint in list(inj.pending_self_mints.items()):
                    if not getattr(_mint, "delegated", False):
                        continue
                    logger.info(
                        "%s: delegated mint discarded — this boot is not "
                        "eager-first, so there is no eager tier to serve "
                        "from while a child compiles; serving eager with the "
                        "cell absent", spec.name)
                    _fc_undelegate.abandon_self_mint(_mint)
                    inj.pending_self_mints.pop(_pid, None)
            if eager_first:

                mint_pipes: Dict[int, Any] = {}
                for candidate in inj.compile_objects:
                    pid = id(candidate.pipeline)
                    if pid not in inj.pending_self_mints:
                        continue
                    # pgw#1033: an OWED mint advertises no artifact. Every
                    # pending is delegated (pgw#1010), so nothing is armed on
                    # this pipe and it never entered
                    # `active_compile_artifacts`; the arm-time placeholder
                    # selection this loop used to stash was the pending's
                    # COMPUTED `kind="inductor"` ref, which the cell the child
                    # is exporting will never carry — and the only consumer of
                    # the stash was a `_BackgroundMint` field nothing read.
                    # The pipe serves eager until `adopt_delegated_mint` reads
                    # the STAMPED key off the packed envelope.
                    mint_pipes[pid] = candidate.pipeline
                    # pgw#677: the mint's own background compiles are the
                    # first consumers of the turn gate — wire it before the
                    # first seed can enqueue a warm job. pgw#1215 step 4: this
                    # pair used to run in the opposite order, which the prose
                    # already said was wrong; `Router.enable` now refuses an
                    # ungated router typed, so the order is enforced instead
                    # of merely intended.
                    self._wire_turn_gate(rec, candidate.pipeline)
                    hot_swap.enable(candidate.pipeline)
                rec.background_mint = _BackgroundMint(
                    spec=spec,
                    instance=instance,
                    snapshots=dict(snapshots) if snapshots else None,
                    pendings=dict(inj.pending_self_mints),
                    pipes=mint_pipes,
                    modules=_mint_modules(spec),
                    slots=dict(resolved_slots),
                )
                logger.info(
                    "eager-first boot for %s (pgw#671): READY at eager tier "
                    "after the minimal warm pass; self-mint runs in the "
                    "background and hot-swaps on arm",
                    spec.name,
                )
            # gw#587 serving bootstrap: the warmup PROOF gates every inductor
            # arm — delivered (store-served) AND self-minted alike. Only the
            # artifact SOURCE differs; a self-mint that does not actually
            # serve its own warmup graphs must fail closed below exactly
            # like a delivered cell that doesn't (never silent eager).
            # pgw#735: EXPORTED artifacts prove themselves by executing,
            # not by an FX cache hit — only the dynamo lane is scored by
            # hits below.
            # pgw#1141b: the lane split is decided per OBJECT (`_exported_arm`),
            # never off the ref string alone. A boot-adopted cell wraps a live
            # pipeline through the ordered arm, which taught `is_aot_ref`
            # nothing — so on a real pod every adopted artifact landed in
            # `proof_before` (the DYNAMO ledger), scored calls=0 against
            # counters an AOTI artifact cannot move, and was folded into
            # `unproven` and unwrapped. `aot_proof_before` was empty, so §4.31's
            # keep-the-arm branch below could not fire for the one object it
            # exists for.
            def _proves_by_fx(pipeline: Any, ref: str) -> bool:
                return not _exported_arm(pipeline, ref)

            _armed_now = [
                (candidate.pipeline, sel)
                for candidate in inj.compile_objects
                if (sel := inj.active_compile_artifacts.get(
                    id(candidate.pipeline))) is not None
            ]
            proves_inductor = any(
                _proves_by_fx(pipe, sel.ref) for pipe, sel in _armed_now)
            proof_before = {
                id(pipe): (
                    compile_cache.execution_count(pipe),
                    compile_cache.cache_miss_count(pipe),
                    aot_serve.execution_count(pipe),
                )
                for pipe, sel in _armed_now
                if proves_inductor and _proves_by_fx(pipe, sel.ref)
            }
            # Exported arms are proven separately: same fail-closed rule, its
            # own counter.
            aot_proof_before = {
                id(pipe): aot_serve.execution_count(pipe)
                for pipe, sel in _armed_now
                if _exported_arm(pipe, sel.ref)
            }
            # pgw#722 finding 2 (the #735 boot-proof gap): the proof loop
            # below used to run only under `proves_inductor`, so a worker
            # whose ONLY arm is an exported cell (the F1 adopt shape — the
            # dynamo artifact is skipped) stayed armed UNPROVEN through
            # boot. An exported arm demands the same fail-closed boot proof
            # as a dynamo arm; only the per-object scoring differs.
            proves_exported = bool(aot_proof_before)
            # pgw#815: every mint obligation this boot opened, snapshotted
            # BEFORE the proof pass pops entries. The assertion after the
            # block below reads it — a pending that reaches readiness having
            # touched none of {sealed, publishing, withheld, aborted,
            # abandoned} was never resolved by anything, which is exactly the
            # 24-minute L4 mint that ended `finalize completed` with no cell,
            # no receipt, no local arm and no error.
            mint_obligations = list(inj.pending_self_mints.values())
            warmup = getattr(instance, "warmup", None)

            async def run_warmup() -> Tuple[int, Dict[int, set[str]], str]:
                if callable(warmup):
                    warm_t0 = time.monotonic()
                    if asyncio.iscoroutinefunction(warmup):
                        await warmup()
                    else:
                        await _to_thread_complete(warmup)
                    warm_ms = int(round((time.monotonic() - warm_t0) * 1000))
                    # pgw#797: UNGATED. This was `if spec.compile is not None`
                    # and log-only, so a release without a `Compile`
                    # declaration did not even log its warmup, and one with it
                    # logged to a hub-spawned pod's unreachable stdout. The
                    # number now rides the wire on both paths.
                    activity_mod.emit_event(
                        "warmup_custom",
                        f"custom warmup() for {spec.name}",
                        phase=activity_mod.PHASE_WARMUP_FORWARD,
                        duration_ms=warm_ms,
                    )
                    logger.info("custom warmup %s completed in %.1fs",
                                spec.name, warm_ms / 1000.0)
                    return 1, {}, ""

                # gw#470: no custom warmup() — run every declared handler of
                # this instance group. A failure propagates as a load failure.
                evidence = await self._run_synthesized_warmup(
                    spec,
                    rec,
                    instance,
                    snapshots,
                    proof_objects=(
                        candidate.pipeline for candidate in inj.compile_objects
                        if id(candidate.pipeline) in inj.active_compile_artifacts
                    ),
                    cold_proof_ids=frozenset(inj.pending_self_mints),
                    # pgw#654 warm-tax fix: a checkpoint instance of an
                    # already-warmed contract collapses to one verification
                    # job (setup path only; hot-adopt keeps full proof).
                    allow_contract_skip=True,
                    armed_cell_refs=tuple(
                        sel.ref
                        for sel in inj.active_compile_artifacts.values()
                    ),
                )
                return (evidence.count, evidence.functions_by_object,
                        evidence.aborted)

            # pgw#671: an eager-first foreground pass is an eager warm, not
            # the inductor compile — that runs in the background driver.
            foreground_minting = bool(inj.pending_self_mints) and not eager_first
            activity_mod.current_phase(
                activity_mod.PHASE_INDUCTOR_COMPILE
                if foreground_minting
                else activity_mod.PHASE_WARMUP_FORWARD
            )
            warmup_stage = (
                pb.LIFECYCLE_INTENT_STAGE_COMPILING
                if foreground_minting
                else pb.LIFECYCLE_INTENT_STAGE_WARMING
            )
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                warmup_stage,
            )
            # pgw#1010: measured UNCONDITIONALLY. It used to be gated on
            # `proves_inductor`, which needs an active artifact — and the JIT
            # lane that survives (INTAKE) has none by construction, so the only
            # remaining dynamo compile on the platform would have been the one
            # nobody timed. A counter read costs nothing.
            compile_seconds_before = compile_cache.compile_wall_seconds()
            # pgw#1082: dynamo's own graph/break counters across the SAME
            # window. "How many graphs did this arm produce, and what cut
            # them" was unanswerable over our telemetry until this sample.
            graph_audit_before = compile_cache.graph_audit()
            # pgw#797: THE warmup split. `pipeline_load` used to be
            # load+warmup as one number, so "what does a cell save on warmup"
            # was only ever an estimate (`pipeline_load` minus a guessed
            # load). This span nests under the open `pipeline_load` — parent
            # named EXPLICITLY, not inferred — so the ladder still reconciles
            # and `pipeline_load` reads as weights->VRAM by subtraction.
            #
            # armed=1 and armed=0 are the two rows the question needs, and
            # they are the SAME quantity th#1329 records as `warmup_ms` on an
            # adopt event: what a warm pass costs with a compiled artifact
            # already armed. Unarmed, the pass pays the compile (CLASS_COMPILE);
            # armed, it pays only the call (CLASS_LOAD).
            with self._warmup_span(spec, rec, inj):
                if inj.active_compile_artifacts:
                    # Cache-hit counters are process-global. Hold every GPU
                    # permit so each exact guard window belongs to only this
                    # warmup.
                    async with self._exclusive_gpu(
                        intent_id,
                        resume_stage=warmup_stage,
                    ):
                        # pgw#672 honesty: drop stale in-memory compiled code
                        # for every object under proof so the warmup MUST go
                        # through the real lookup path — a mint truly compiles
                        # into its capture, an adoption truly hits its seeded
                        # FX entries. In a warm process, dynamo's class-keyed
                        # code cache otherwise serves these calls
                        # counter-silently (calls>0, hits=0, misses=0) and the
                        # proof disproves a healthy lane. No sibling GPU work
                        # runs inside this window, so the per-code reset is
                        # race-free; siblings re-trace to FX hits afterwards
                        # (additive live root).
                        for _cand in inj.compile_objects:
                            _sel = inj.active_compile_artifacts.get(
                                id(_cand.pipeline))
                            if _sel is not None:
                                compile_cache.reset_target_code(_cand.pipeline)
                        warmed, function_proofs, warm_aborted = await run_warmup()
                else:
                    warmed, function_proofs, warm_aborted = await run_warmup()
            compile_seconds = max(
                0.0,
                compile_cache.compile_wall_seconds() - compile_seconds_before)
            # id(pipeline) -> (calls, cache_hits, cache_misses) observed across
            # this setup's warmup. Declared out here because pgw#923's adoption
            # report reads it whether or not this boot proved anything: a cell
            # that armed and then warmed to zero hits is exactly the adoption
            # the measurement lane exists to price.
            proof_by_obj: Dict[int, Tuple[int, int, int]] = {}
            if compile_seconds > 0 and not inj.active_compile_artifacts:
                # pgw#1010 / th#1322: the JIT compile this pod paid for itself.
                # The `jit_compile` numeric event used to be emitted by the
                # parent OF A MINT CHILD, which no longer runs the JIT recipe —
                # so without this the INTAKE lane (the only JIT left) would
                # compile silently and "AOT vs JIT compile cost" would have one
                # arm. Sited OUTSIDE the proof block on purpose: an intake arm
                # names no artifact, so `proves_inductor` is false for it and
                # anything inside that block is unreachable from the lane this
                # measures.
                compile_cache.emit_jit_compile_event(
                    {"boot": compile_seconds},
                    family=str(getattr(spec.compile, "family", "") or ""),
                    execution_lane=self._served_execution_lane(spec),
                    route="intake",
                    audit=compile_cache.graph_audit_delta(graph_audit_before),
                )
            if proves_inductor or proves_exported:
                # gw#595 per-object provability: the proof scopes to objects
                # the warmup actually EXERCISED (calls>0). An exercised object
                # must serve its own cache hit or it disproves the cell. An
                # unexercised object (the warmup has no modality for it, e.g.
                # an edit lane needing an input image) neither proves nor
                # disproves — with a proven sibling it must not block
                # adoption. Zero proven objects still fails closed (gw#586).
                disproven: list[_CompileObjectCandidate] = []
                unexercised: list[_CompileObjectCandidate] = []
                #: pgw#1141: per object, why the boot warmup landed no dispatch
                #: on an ARMED artifact — the posture that follows is a row on
                #: the wire, not something a reader has to infer from the two
                #: later events that only describe its consequences
                #: (`target_applicability_incomplete`, `armed_target_unresolved`).
                arm_without_dispatch: Dict[int, str] = {}
                proven = 0
                hits = 0
                misses = 0
                calls_by_obj: Dict[int, int] = {}
                for candidate in inj.compile_objects:
                    pipe = candidate.pipeline
                    aot_before = aot_proof_before.get(id(pipe))
                    if aot_before is not None:
                        # pgw#735: the EXPORTED lane's proof — its own
                        # invocations, still armed. An exported artifact
                        # performs no FX lookup, so scoring it by cache hits
                        # would disprove every honest .pt2 adoption. Fail
                        # closed exactly like the dynamo lane: no execution or
                        # a revoked artifact means unexercised, never a
                        # synthesized hit.
                        aot_calls = aot_serve.execution_count(pipe) - aot_before
                        calls_by_obj[id(pipe)] = aot_calls
                        proof_by_obj[id(pipe)] = (aot_calls, 0, 0)
                        if warmed and aot_serve.proven_since(pipe, aot_before):
                            proven += 1
                            if callable(warmup):
                                function_proofs[id(pipe)] = {spec.name}
                            proved_sel = inj.active_compile_artifacts.get(id(pipe))
                            if proved_sel is not None:
                                compile_cache.record_compiled_graph_proven(proved_sel.ref)
                            continue
                        # pgw#1141 / §4.31 + §4.32: an adopted cell arms BEFORE
                        # setup, so no dispatch can have landed by now, and
                        # nothing measures it here either — quality was proven
                        # once, on the pod that MINTED it, and adoption runs no
                        # quality gate. The absence of a dispatch is therefore
                        # not a verdict: the arm stands, the first real request
                        # is the proof, and a cell-attributable failure de-arms
                        # it in-request.
                        arm_without_dispatch[id(pipe)] = (
                            "adoption runs no quality gate (§4.32) — this cell "
                            "was proven at its mint")
                        unexercised.append(candidate)
                        continue
                    before = proof_before.get(id(pipe))
                    if before is None:
                        continue
                    calls = compile_cache.execution_count(pipe) - before[0]
                    calls_by_obj[id(pipe)] = calls
                    pipe_hits = compile_cache.cache_hit_count(pipe)
                    pipe_misses = compile_cache.cache_miss_count(pipe) - before[1]
                    proof_by_obj[id(pipe)] = (calls, pipe_hits, pipe_misses)
                    hits += max(0, pipe_hits)
                    misses += max(0, pipe_misses)
                    if not warmed or calls <= 0:
                        arm_without_dispatch[id(pipe)] = (
                            "the dynamo lane takes no parity measurement, so "
                            "this boot holds no evidence either way")
                        unexercised.append(candidate)
                    elif pipe_hits > 0:
                        proven += 1
                        if callable(warmup):
                            function_proofs[id(pipe)] = {spec.name}
                        proved_sel = inj.active_compile_artifacts.get(id(pipe))
                        if proved_sel is not None:
                            compile_cache.record_compiled_graph_proven(proved_sel.ref)
                    elif (
                        pipe_misses <= 0
                        and (inmem_sel := inj.active_compile_artifacts.get(
                            id(pipe))) is not None
                        and compile_cache.compiled_graph_proven_in_process(inmem_sel.ref)
                        and compile_cache.has_inmemory_compiled_code(pipe)
                    ):
                        # pgw#637: calls>0 with ZERO counter movement against
                        # a cell this process ALREADY proved, AND dynamo
                        # confirming live compiled code for this object's
                        # targets, is torch 2.13's in-memory dynamo code
                        # cache serving a sibling checkpoint's compiled code
                        # — a legitimate third serving surface (cell keys are
                        # checkpoint-free by design), not silent eager.
                        # Disproving it bricked the compiled lane on every
                        # 2nd same-family pick. Both conditions are load-
                        # bearing: the registry alone would let a SIBLING
                        # object's hit certify this object's silence, which
                        # gw#603/gw#611 forbid; the dynamo probe alone would
                        # credit a cell never proven anywhere.
                        proven += 1
                        if callable(warmup):
                            function_proofs[id(pipe)] = {spec.name}
                        logger.info(
                            "compile-cache: %s served warmup from dynamo's "
                            "in-memory code cache (cell %s already proven "
                            "in-process; calls=%d) — counted as serving "
                            "evidence (pgw#637)",
                            spec.name, inmem_sel.ref, calls,
                        )
                    else:
                        disproven.append(candidate)
                # gw#612: everything after the per-object proof — unproven
                # handling, sibling resolution, the publish decision, and
                # the bookkeeping down to readiness — reports an honest
                # phase instead of a stale seal_publish.
                activity_mod.current_phase(activity_mod.PHASE_FINALIZE)

                def _confess_arm_without_dispatch(
                    candidate: "_CompileObjectCandidate",
                ) -> None:
                    """pgw#1141: name the DECISION, at the decision point.

                    Every emission the old disarm produced described its
                    wreckage two frames later (`target_applicability_
                    incomplete`, then `armed_target_unresolved`), so a reader
                    had to infer that an armed, resolvable cell had been thrown
                    away. The decision is the opposite one now — the arm STANDS
                    — and it is still a row, because an unannounced posture is
                    indistinguishable from a gate that never ran."""
                    reason = arm_without_dispatch.get(id(candidate.pipeline), "")
                    if not reason:
                        return
                    activity_mod.emit_event(
                        activity_mod.KIND_COMPILED_GRAPH_NUMERICS,
                        f"{spec.name}: the exported cell on slots "
                        f"{sorted(candidate.slots)} took no warm dispatch — "
                        f"{reason}. It STAYS ARMED and serves; a "
                        f"cell-attributable failure revokes it in-request",
                        phase=numerics_ladder.PHASE_ARMED_UNDISPATCHED,
                    )

                # pgw#1141 (Paul's ruling, 2026-08-11), and it is a DELETION:
                # *"skip the warmup/arm check, so we can serve right away; try
                # to serve, and if an error is encountered and it is the cell's
                # fault, de-arm the cell and serve eager instead. If our cell is
                # correct this adds zero cost."* An ABSENCE of warm evidence is
                # no longer a verdict about the artifact — an adopted cell arms
                # before setup, so nothing has dispatched through it BY
                # CONSTRUCTION, and disarming on that destroyed cells verified
                # at cos=1.00000 on two real pods while the self-mint arm (which
                # gets its dispatch from the warmup that drives its own capture)
                # sailed through. The two arms are symmetrical now: neither is
                # disarmed for want of a dispatch.
                #
                # SCOPED TO THE EXPORTED LANE, because the difference is the
                # failure MODE. An AOTI cell that cannot serve RAISES, and the
                # wrapper answers that request eager; a DYNAMO arm that does not
                # serve its cell RECOMPILES — correct output, silently slower,
                # no exception for try-serve to catch and no numerics gate on
                # that lane at all — so its per-class cache-hit ledger is the
                # only detector in existence and keeps its teeth.
                #
                # What still has teeth, unchanged:
                #   * the pgw#868 numerics gate REFUSES a cell that does not
                #     reproduce eager — the only detector for a cell that runs
                #     cleanly and returns a WRONG image, which try-serve cannot
                #     see;
                #   * EVIDENCE AGAINST still disarms (`disproven`: the object was
                #     exercised and demonstrably did not serve its own graph —
                #     a measured fault, not a missing measurement);
                #   * a cell-attributable failure at serve time revokes the arm
                #     IN-REQUEST (`aot_serve.wrap_module` / the pgw#680
                #     guard-miss doctrine): the tenant still gets a correct eager
                #     answer, the disarm is sticky for the process, and it is
                #     typed on the wire.
                #   * PUBLISHING to the fleet stays evidence-gated below —
                #     serving optimistically costs this pod one eager fallback,
                #     publishing an unverified cell costs every pod that adopts
                #     it.
                unproven = list(disproven)
                # The DYNAMO lane's silent-recompile detector, unchanged: with
                # nothing proven, an unexercised dynamo object is still folded
                # in and disarmed. Exported candidates never reach here — they
                # are scored above and keep their arm either way.
                dynamo_unexercised = [
                    candidate for candidate in unexercised
                    if not _exported_arm(candidate.pipeline)
                ]
                if not proven and dynamo_unexercised:
                    unproven.extend(dynamo_unexercised)
                    unexercised = [
                        candidate for candidate in unexercised
                        if candidate not in dynamo_unexercised
                    ]
                if unproven:

                    quant_execution_lane = any(
                        pipeline_weight_lane(
                            candidate.pipeline).startswith(_MANDATORY_EXECUTION_LANES)
                        for candidate in unproven
                    )
                    for candidate in unproven:
                        pipe = candidate.pipeline
                        function_proofs[id(pipe)] = set()
                        _confess_arm_without_dispatch(candidate)
                        # pgw#722 finding 2: an exported arm disarms through
                        # its own lane — aot_serve.unwrap restores the
                        # forward it captured (under the F2 flip that is the
                        # lifted LoRA forward), then the lifted lanes come
                        # off so the pod lands back on the exact pre-flip
                        # eager shape. Order is load-bearing.
                        if aot_serve.unwrap(pipe):
                            from .models import lora_lifted

                            lora_lifted.remove_lifted_lora_execution_lanes(pipe)
                        compile_cache.unwrap(pipe)
                        if spec.lora_bucket:
                            compile_cache.drop_lora_execution_lane(pipe)
                        # pgw#672: quarantine the disproven identity in this
                        # process so neither selection nor a fresh self-mint
                        # arm loops on it this boot.
                        failed_sel = inj.active_compile_artifacts.pop(
                            id(pipe), None)
                        if failed_sel is not None:
                            compile_cache.record_compiled_graph_quarantined(
                                failed_sel.ref)
                        failed_pending = inj.pending_self_mints.get(id(pipe))
                        if failed_pending is not None:
                            compile_cache.record_compiled_graph_quarantined(
                                str(failed_pending.ref))
                        self._abandon_pending_mint(inj, pipe)
                    # gw#611: `calls` discriminates the failure classes on the
                    # wire — calls=0 is an orphaned/never-invoked wrapper (or
                    # no warmup modality), calls>0 with 0/0 counters is a
                    # compiled call served by a cache layer the counters
                    # don't watch (pod logs are unreachable; this line is
                    # the only forensic surface).
                    unproven_calls = sum(
                        calls_by_obj.get(id(c.pipeline), 0) for c in unproven)
                    # gw#608: compile_seconds discriminates crediting bugs
                    # (~0s) from real recompiles (minutes) without pod logs;
                    # per-object calls/hits/misses scope a multi-object boot.
                    per_object = ""
                    if len(proof_by_obj) > 1:
                        per_object = ", objects=[" + ", ".join(
                            f"{c}/{h}/{m}"
                            for c, h, m in proof_by_obj.values()) + "]"
                    detail = (
                        f"{len(unproven)} attached compile object(s) did not "
                        "serve their own warmup graph "
                        f"(warmups={warmed}, calls={unproven_calls}, "
                        f"cache_hits={hits}, cache_misses={misses}, "
                        f"compile_seconds={compile_seconds:.1f}{per_object})"
                    )
                    # gw#608 FX-cache census: this boot's recompiles already
                    # saved their entries into the live cache dir, so the
                    # report says how many keys exist and what extern-libs key
                    # this process presents.
                    # pgw#1200 removed the CELL side and the B1/B2
                    # classification with it — every class was a difference
                    # against FX entries read from a `torch-inductor-cache`
                    # tarball, and that format has no writer and is deleted, so
                    # B1 was being named on every boot. The report no longer
                    # takes the artifact; passing one carried no information.
                    # pgw#722 finding 2 still scopes the CALL: FX state
                    # describes the dynamo lane only, which `proves_inductor`
                    # is what says.
                    if proves_inductor and compile_selection is not None:
                        try:
                            forensics = compile_cache.fx_cache_failure_report()
                        except Exception:
                            forensics = ""
                            logger.debug(
                                "fx-key forensics unavailable", exc_info=True)
                        if forensics:
                            logger.error(
                                "compile-cache: FX-key forensics: %s",
                                forensics)
                            # The activity error is capped at 2000 chars on
                            # the wire; keep the leading counts + first
                            # divergence intact.
                            detail += f"; fx forensics: {forensics[:1500]}"
                    if quant_execution_lane:
                        # pgw#672 posture change: a failed serve/finalize
                        # proof on a mandatory (w8a8/w4a4) lane used to raise
                        # here -> compiled_graph_quarantined -> every declared function
                        # disabled -> pod retired -> the replacement re-mints
                        # the same key (5 cycles / 4 dead workers on the L4
                        # burst). A broken optimization must never kill a
                        # serving worker: withhold the unproven publish,
                        # quarantine the identity (above), and DEGRADE to
                        # explicit eager — serving_tier flips on the wire and
                        # the activity carries the confession; never silent
                        # (gw#586).
                        logger.error(
                            "%s; mandatory lane DEGRADED to explicit eager "
                            "serving (pgw#672)", detail)
                        activity_mod.current_note(
                            f"compiled lane degraded to eager: {detail}")
                        # pgw#677 reopen: countable typed event — the
                        # degrade + withheld publish must be visible
                        # without pod logs.
                        activity_mod.emit_event(
                            "self_mint_abort",
                            f"proof failed; degraded to eager: {detail}",
                            phase="proof_failed",
                        )
                    else:
                        logger.warning("%s; serving eager", detail)
                for candidate in unexercised:
                    pipe = candidate.pipeline
                    if _exported_arm(pipe):
                        # THE DELETED BARRIER (pgw#1141). This branch used to
                        # unwrap the artifact, drop the lifted lanes, pop the
                        # active selection and abandon the mint — for an object
                        # the warm plan simply never dispatched through, which
                        # is EVERY boot-adopted cell by construction. It keeps
                        # the arm now and says so; the publish half is the only
                        # decision an absent measurement may still make.
                        logger.warning(
                            "compile object (slots=%s) armed with no warm "
                            "dispatch (calls=0); it SERVES, and a "
                            "cell-attributable failure revokes it in-request "
                            "(pgw#1141)", sorted(candidate.slots))
                        _confess_arm_without_dispatch(candidate)
                        # NOTE the publish is NOT withheld here any more. §4.32
                        # moves that authority to the mint-time gate on this
                        # same pod (`fleet_cells.adopt_delegated_mint` ->
                        # `provision.arm_aot(verify_numerics=True)`), which runs
                        # the freshly compiled artifact against the eager
                        # forward it was traced from and refuses to publish
                        # anything that is not identical. A warm dispatch count
                        # decides nothing at either end.
                        continue
                    mandatory = pipeline_weight_lane(pipe).startswith(
                        _MANDATORY_EXECUTION_LANES)
                    if mandatory:
                        # Eager is not a lane for it and a proven sibling
                        # vouches for the cell: stays armed unproven. Its
                        # own graphs, absent from the cell by design, fail
                        # loud at first use instead of at every boot.
                        logger.warning(
                            "compile object (slots=%s) armed unproven: no "
                            "warmup modality exercised it (calls=0); the "
                            "proof covers only its exercised siblings",
                            sorted(candidate.slots))
                        continue
                    # The dynamo lane keeps its disarm: an unexercised dynamo
                    # object that starts serving RECOMPILES silently, and no
                    # detector downstream would ever say so.
                    logger.warning(
                        "compile object (slots=%s) unproven (no warmup "
                        "modality, calls=0); serving eager",
                        sorted(candidate.slots))
                    _confess_arm_without_dispatch(candidate)
                    function_proofs[id(pipe)] = set()
                    compile_cache.unwrap(pipe)
                    if spec.lora_bucket:
                        compile_cache.drop_lora_execution_lane(pipe)
                    inj.active_compile_artifacts.pop(id(pipe), None)
                    self._abandon_pending_mint(inj, pipe)
                if (
                    proven
                    and compile_selection is not None
                    and compile_seconds >= _STORE_SERVED_COMPILE_ALARM_S
                ):
                    # gw#587 runtime assertion: this boot ATTACHED a cell
                    # (compile_selection is set — store-served; a MINTING
                    # boot has compile_selection=None and legitimately
                    # compiles, so it is exempt by the explicit gate above)
                    # and at least one candidate proved a warm
                    # cache hit, yet the process burned real inductor compile
                    # wall time getting there. A delivered cell should cost
                    # ~0 here; this is the gw#586 defect class generalized —
                    # a cell that claims to serve while the boot silently
                    # recompiles (stale/shape-mismatched artifact, or the
                    # wrong cell attested through th#910). Loud, greppable,
                    # and mirrored onto the wire via the existing ADOPTED
                    # ModelEvent shape (gw#391) so it is visible hub-side —
                    # boot-attached cells never sent this event before;
                    # duration_ms carries the measured compile wall here
                    # specifically (not the ordinary hot-adopt op-wall
                    # meaning) since this call site only fires on alarm.
                    family = str(getattr(spec.compile, "family", "") or "")
                    ref = WireRef(compile_selection.ref if compile_selection else "")
                    digest = (
                        compile_selection.snapshot_digest
                        if compile_selection else "")
                    logger.error(
                        "compile-cache: STORE_SERVED_BOOT_COMPILED family=%s "
                        "cell=%s digest=%s compile_seconds=%.1fs (>= alarm "
                        "threshold %.1fs) — a store-served boot should pay "
                        "~0 compile time (cache_hits=%d cache_misses=%d)",
                        family, ref, digest, compile_seconds,
                        _STORE_SERVED_COMPILE_ALARM_S, hits, misses,
                    )
                    try:
                        # gw#608: full cache-state report for the
                        # partial-recompile shape too.
                        logger.error(
                            "compile-cache: FX-key forensics: %s",
                            compile_cache.fx_cache_failure_report())
                    except Exception:
                        logger.debug(
                            "fx-key forensics unavailable", exc_info=True)
                    # pgw#923: this alarm used to ride the ADOPTED ModelEvent
                    # with `duration_ms` redefined to mean "inductor compile
                    # wall" — a second meaning for the one field the adoption
                    # measurement lane percentiles over. It has its own typed
                    # event now, so `compile_cache_adopt.duration_ms` means the
                    # arm, always, and this alarm keeps its own number.
                    activity_mod.emit_event(
                        "store_served_boot_compiled",
                        f"family={family} cell={ref} digest={digest}: a "
                        f"store-served boot burned {compile_seconds:.1f}s of "
                        f"inductor compile wall (alarm threshold "
                        f"{_STORE_SERVED_COMPILE_ALARM_S:.0f}s, cache_hits="
                        f"{hits} cache_misses={misses}) — the delivered cell "
                        f"is not serving the graphs this boot compiled",
                        phase="alarm",
                        duration_ms=int(compile_seconds * 1000),
                    )
            await self._report_adoptions(inj, proof_by_obj)
            # pgw#815: THE assertion. `finalize completed` must be
            # unreachable while a mint obligation is unresolved — running it
            # OUTSIDE the `proves_inductor or proves_exported` block is the
            # point, because a boot that armed a pending and then answered
            # "nothing proves by FX or export" skipped the whole publish gate
            # and resolved nothing, silently. Obligations handed to the
            # background driver (eager-first / delegated) are ITS to resolve
            # and are excluded here — the driver's own sweep, in
            # `_background_mint`'s `finally`, is where they are asserted.
            driver = rec.background_mint
            owned_by_driver = (
                {id(p) for p in driver.pendings.values()}
                if driver is not None else set())
            self._assert_mint_termini(
                spec,
                [p for p in mint_obligations if id(p) not in owned_by_driver])
            vram_delta = max(0, cuda_allocated_bytes() - vram_before)
            if rec.server is not None:
                # Engine subprocess VRAM is invisible to torch's allocator;
                # book the measured per-PID footprint so the LRU ledger is
                # honest and eviction (record teardown -> server.stop) works.
                from .runtimes.server import process_vram_bytes

                vram_delta += await asyncio.to_thread(
                    process_vram_bytes, rec.server.process.pid)
            self._register_residency(
                spec, setup_slots, inj.loaded, vram_delta,
                execution_lane_slots=inj.execution_lane_slots, shared_bytes=inj.shared_bytes,
                slot_refs=slot_refs, slot_identities=slot_identities)
            # gw#551: call-time-owned refs. Any record holding 2+ worker-
            # constructed pipelines can overcommit VRAM (content-keyed lanes
            # AND monolithic siblings alike) — those swap per use via the
            # LaneResidencyGate instead of being job-pinned + eagerly promoted.
            pipe_slots = {s for s, (obj, _) in inj.loaded.items() if obj is not None}
            swap_owned = pipe_slots if len(pipe_slots) >= 2 else set(inj.execution_lane_slots)
            swap_owned &= inj.gated_slots  # un-gateable pipes stay eager
            rec.execution_lane_refs = {slot_refs[s] for s in swap_owned if s in slot_refs}
            rec.held_objects = {}
            for slot, ref in slot_refs.items():
                obj = inj.loaded.get(slot, (None, 0))[0]
                if obj is not None or ref not in rec.held_objects:
                    rec.held_objects[ref] = obj
            # pgw#678: the pipeline identities, kept out of held_objects (which
            # is the residency/movement handle space).
            rec.slot_pipelines = dict(inj.slot_pipelines)
            # pgw#748: a degree-D group becomes D ranks HERE — after the
            # pipeline exists and its attention backend is set, before
            # compile. Degree 1 (every pod today) is a no-op.
            await self._arm_sequence_group(rec, spec, slot_refs)
            # pgw#824: the arming brain's own classified declines, carried
            # from the injection into the record BEFORE target installation so
            # its own (coarser) omission tokens never outrank them.
            for token in inj.eager_postures:
                self._note_eager_posture(
                    rec, token, f"the arming policy declined: {token}")
            self._install_compile_targets(
                rec,
                spec,
                inj.compile_objects,
                inj.active_compile_artifacts,
                function_proofs,
            )
            # pgw#1093: the terminus. Every route from "an arm returned True"
            # to "no installed target owns it" now ends on ONE typed,
            # wire-visible refusal instead of a log line nobody on a
            # hub-spawned pod can read.
            self._assert_armed_targets_installed(rec, spec, inj.armed_objects)
            rec.stale = False
            await self._clear_host_ram_capacity(list(slot_refs.values()))
        return instance

    # ---- sequence parallelism (pgw#748 phase 1) ------------------------------

    def _sequence_boot_slot(
        self, spec: EndpointSpec, rec: "_ClassRecord",
    ) -> str:
        """The ONE class-annotated pipeline slot a degree-D group shards.

        Refused typed, never guessed: a follower rebuilds its copy through
        ``provision.load_slot`` from the pod's shared CAS path, so a
        self-loading (str/Path) slot has no reproducible construction, and two
        candidate pipelines have no single SPMD unit.
        """

        candidates = [s for s, pipe in rec.slot_pipelines.items() if pipe is not None]
        if len(candidates) == 1:
            return candidates[0]
        raise ContextParallelUnavailable(
            f"{spec.name}: sequence parallelism needs exactly ONE "
            f"class-annotated pipeline slot to shard, found {candidates or 'none'}"
        )

    async def _arm_sequence_group(
        self,
        rec: "_ClassRecord",
        spec: EndpointSpec,
        slot_refs: Dict[str, WireRef],
    ) -> None:
        """Turn this record's execution group into D ranks, or refuse loudly.

        A refusal here is TERMINAL for the setup: the pod was bought for a
        degree-D promise and serving degree 1 against it would silently
        deliver a fraction of the tier that was sold. The hub's own answer to
        an unservable degree is to re-pack the pod, and it can only do that if
        the worker says so.
        """
        topo = self.topology
        if topo.degree <= 1 or topo.parallel != "sequence":
            return

        group = current_device_group()
        device_group = topo.group(group)
        slot = self._sequence_boot_slot(spec, rec)
        pipe = rec.slot_pipelines[slot]
        ref = slot_refs.get(slot, WireRef(""))
        path = self.store.local_path(ref) if ref else None
        binding = spec.models.get(slot)

        boot = BootPlan(
            modules=tuple(sorted({
                s.cls.__module__ for s in rec.specs if s.cls is not None
            })),
            function_name=spec.name,
            slot=slot,
            path=str(path or ""),
            cache_dir=str(self.store._cache_dir),
            degree=topo.degree,
            dtype=str(getattr(binding, "dtype", "") or ""),
            storage_dtype=str(getattr(binding, "storage_dtype", "") or ""),
        )
        # Rank 0 DECIDES; every rank obeys. Nothing below rank 0 ever measures
        # its own card and adapts (pgw#748 §5.4).
        plan = GroupPlan(
            precision_execution_lane=compile_cache.cell_base_execution_lane(pipe),
            gemm_mode=w8a8_gemm_mode(pipe),
            sp_degree=topo.degree,
        )
        runtime = SequenceRuntime(device_group.devices)
        installed = await asyncio.to_thread(runtime.arm, pipe, boot, plan)
        if not arm_sequence_gate(pipe, runtime):
            await asyncio.to_thread(runtime.close)

            raise ContextParallelUnavailable(
                f"{spec.name}: could not route {type(pipe).__name__}.__call__ "
                "through the rank group; refusing rather than serving degree 1 "
                "against a degree-{0} promise".format(topo.degree))
        rec.sp_runtime = runtime
        self._sequence_plans[id(rec)] = plan
        logger.info(
            "%s armed sequence parallelism degree=%d on devices %s (%s)",
            spec.name, topo.degree, list(device_group.devices), list(installed))

    def _close_sequence_group(self, rec: "_ClassRecord") -> None:
        """Tear down a record's rank group OFF the event loop (pgw#774).

        ``runtime.close()`` joins/terminates follower processes (bounded, but
        seconds) and must never run on the loop: the old collective-based
        close could block the loop — and the heartbeat with it — forever,
        presenting a wedged group as a platform stall."""
        runtime, rec.sp_runtime = rec.sp_runtime, None
        self._sequence_plans.pop(id(rec), None)
        if runtime is None:
            return

        def _do() -> None:
            try:
                runtime.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the vacate
                logger.warning("closing the sequence group failed", exc_info=True)

        threading.Thread(target=_do, name="sp-close", daemon=True).start()

    def group_plan_for(self, rec: "_ClassRecord") -> Optional[Any]:
        """The plan every rank of ``rec``'s group agreed on, or None at
        degree 1. Read by the adaptive paths that must NOT decide locally."""
        return self._sequence_plans.get(id(rec))

    def _register_residency(
        self,
        spec: EndpointSpec,
        setup_slots: List[str],
        loaded: Dict[str, Tuple[Any, int]],
        total_delta: int,
        *,
        execution_lane_slots: Optional[set] = None,
        shared_bytes: int = 0,
        slot_refs: Optional[Dict[str, WireRef]] = None,
        slot_identities: Optional[Dict[str, _ResidencyIdentity]] = None,
    ) -> None:
        """Honest per-ref residency after a setup (#369). Worker-constructed
        pipelines carry their own measured allocator delta AND the object
        (Residency owns it: demote/promote actually move memory). Refs the
        tenant loaded inside setup() split the residual delta — no object,
        so their VRAM is only reclaimable by record teardown. Lane slots
        (gw#479) were registered inline during injection — their bytes and
        the shared-entry bytes still reduce the residual, but re-tracking
        them here would clobber a mid-setup demotion."""
        res = self.store.residency
        execution_lanes = execution_lane_slots or set()
        refs = slot_refs or {}
        identities = slot_identities or {}
        per_ref: Dict[WireRef, Tuple[Any, int]] = {}
        per_ref_identity: Dict[str, _ResidencyIdentity] = {}
        for slot in setup_slots:
            if slot in execution_lanes:
                continue
            # gw#494: book under the SAME key the setup derived (never a
            # fresh wire_ref over possibly-rebound spec.models).
            ref = refs.get(slot) or wire_ref(spec.models[slot])
            obj, measured = loaded.get(slot, (None, 0))
            prev_obj, prev_bytes = per_ref.get(ref, (None, 0))
            per_ref[ref] = (obj or prev_obj, prev_bytes + measured)
            identity = identities.get(slot, ("", 0))
            prior_identity = per_ref_identity.get(ref)
            if prior_identity is not None and identity[0] and identity != prior_identity:
                raise RuntimeError(
                    f"setup slots for {ref!r} captured conflicting snapshot "
                    f"identities: {prior_identity!r} != {identity!r}"
                )
            if identity[0] or prior_identity is None:
                per_ref_identity[ref] = identity
        execution_lane_bytes = sum(loaded[s][1] for s in execution_lanes if s in loaded)
        residual = max(0, total_delta - sum(b for _, b in per_ref.values())
                       - execution_lane_bytes - max(0, int(shared_bytes)))
        opaque = [r for r, (obj, _) in per_ref.items() if obj is None]
        share = residual // len(opaque) if opaque else 0
        for ref, (obj, measured) in per_ref.items():
            self.store.activate_load_identity(
                ref, per_ref_identity.get(ref, ("", 0)))
            vram = measured + (share if obj is None else 0)
            if vram > 0:
                res.track_vram(ref, obj, vram_bytes=vram)
            elif obj is not None and int(estimate_cuda_resident_gb(obj) * _GiB) > 0:
                res.track_vram(ref, obj)  # measured via cuda-resident estimate
            else:
                res.track_ram(ref, obj)   # CPU-only host / offloaded load

    async def _promote_setup_refs(
        self,
        spec: EndpointSpec,
        slots: Optional[List[str]] = None,
        rec: Optional[_ClassRecord] = None,
    ) -> None:
        """RunJob/LOAD for a demoted (RAM-tier) instance: swap the pipelines
        back into VRAM instead of a cold reload (#371). Lane refs (gw#479)
        are excluded (gw#551): lane dispatch is handler-side, so eagerly
        promoting EVERY declared lane can never fit an overcommitted card —
        the LaneGate promotes exactly the lane a request touches, at call
        time."""
        res = self.store.residency
        setup_slots = self._setup_slots(spec)
        if slots is not None:
            setup_slots = [s for s in setup_slots if s in slots]
        execution_lane_refs = rec.execution_lane_refs if rec is not None else set()
        refs = [
            r for s in setup_slots
            if (r := wire_ref(spec.models[s])) not in execution_lane_refs
        ]
        # pgw#636: shared-component entries (TE/VAE) the record's pipelines
        # alias are independently demotable now — swap any that went warm
        # back in before this job executes, alongside the setup refs.
        if rec is not None and rec.shared_keys:
            refs.extend(
                cid for k in rec.shared_keys
                if (cid := k.cache_id()) not in refs
            )
        cuda_host = torch is not None and cuda_ready()
        if any(res.tier(r) is residency_mod.Tier.RAM for r in refs):
            async with self._load_lock:
                for ref in refs:
                    if res.tier(ref) is residency_mod.Tier.RAM:
                        ok = await asyncio.to_thread(res.promote, ref)
                        self._on_state_change()
                        if (not ok and cuda_host
                                and res.tier(ref) is residency_mod.Tier.RAM
                                and res.movable(ref)):
                            # Promote refused/rolled back (gw#409): fail the
                            # job RETRYABLE at promote time — never hand a
                            # handler a pipeline that fatals mid-denoise.
                            # Non-movable entries (object-less ledger refs,
                            # offload-hooked pipelines) can never promote —
                            # promote-or-die on them livelocks (gw#417).
                            raise RetryableError(
                                f"promotion of {ref} to VRAM failed; retrying"
                            )
        for ref in refs:
            res.touch(ref)

    @staticmethod
    def _worker_loaded_slot_types(spec: EndpointSpec) -> Dict[str, type]:
        """Setup slots the WORKER materializes in host RAM (class-typed
        annotations loaded via ``from_pretrained``), with their annotated
        classes. str/Path slots and engine runtimes (vllm/llama-server)
        stream weights themselves and must not be counted against the
        host-RAM admission gate."""
        if spec.cls is None or spec.runtime:
            return {}
        setup = getattr(spec.cls, "setup", None)
        if setup is None:
            return {}
        try:
            hints = typing.get_type_hints(setup)
        except Exception:
            return {}
        return {
            name: ann for name, ann in hints.items()
            if isinstance(ann, type) and callable(getattr(ann, "from_pretrained", None))
        }

    @staticmethod
    def _worker_loaded_slots(spec: EndpointSpec) -> set:
        return set(Executor._worker_loaded_slot_types(spec))

    async def _record_host_ram_failure(
        self, refs: List[str],
        error: typing.Union[InsufficientHostRamError, HostRamCapacityError],
    ) -> None:
        """Publish and retain one typed capacity block per causal ref."""
        causal_refs = sorted(_canonical_host_ram_refs(refs))
        if not causal_refs:
            return
        evicted = _canonical_host_ram_refs(error.evicted_refs)
        async with self._host_ram_lock:
            self._host_ram_generation += 1
            generation = self._host_ram_generation
            for ref in causal_refs:
                event = self.store.model_event(
                    ref, pb.MODEL_STATE_FAILED,
                    error=error.reason,
                    host_ram_required_bytes=error.required_bytes,
                    host_ram_available_before_bytes=error.available_before_bytes,
                    host_ram_available_after_bytes=error.available_after_bytes,
                    host_ram_evicted_refs=evicted,
                    host_ram_capacity_generation=generation,
                )
                self._host_ram_progress.pop(ref, None)
                self._host_ram_blocks[ref] = _HostRamBlock(
                    failure_event=event,
                    last_available_bytes=error.available_after_bytes,
                )
                self._queue_host_ram_event_locked(event)
        # Commit every causal ref before exposing the first event, and never
        # hold the state lock across a potentially backpressured/cancelled send.
        await self._flush_host_ram_outbox()

    def _queue_host_ram_event_locked(self, event: pb.ModelEvent) -> None:
        self._host_ram_outbox.pop(event.ref, None)
        self._host_ram_outbox[event.ref] = event

    async def _flush_host_ram_outbox(self) -> None:
        """Serialize committed generations without holding the state lock."""
        async with self._host_ram_send_lock:
            while True:
                async with self._host_ram_lock:
                    if not self._host_ram_outbox:
                        return
                    event = next(iter(self._host_ram_outbox.values()))
                await self._send(pb.WorkerMessage(model_event=event))
                async with self._host_ram_lock:
                    current = self._host_ram_outbox.get(event.ref)
                    if current is not None and current == event:
                        self._host_ram_outbox.pop(event.ref, None)

    async def _observe_host_ram_progress(
        self,
        released_refs: List[str],
        *,
        collect_host: bool = False,
    ) -> None:
        """Emit progress only when a release measurably satisfies a block.

        Callers invoke this after an owner record or execution pin has been
        released. A release that leaves headroom unchanged or still below the
        exact remembered requirement only advances the local numeric baseline;
        it never wakes the orchestrator.
        """
        released = _canonical_host_ram_refs(released_refs)
        # Let the RunJob pin/teardown frame release its references before the
        # host-only cgroup probe. This is a yield, not a retry timer. In
        # particular, do not call flush_memory here: it mutates CUDA cache and
        # resets peak-memory metrics even for an ordinary RunJob pin release.
        await asyncio.sleep(0)
        async with self._host_ram_lock:
            if not self._host_ram_blocks:
                return
            if collect_host:
                # Actual endpoint teardown can leave cyclic host objects after
                # all explicit owners are cleared. Collect host objects only;
                # flush_memory would also mutate CUDA cache/peak metrics.
                await asyncio.to_thread(gc.collect)
            observed = await asyncio.to_thread(self.store.residency.host_ram_headroom, 0)
            available = observed.available_bytes
            satisfied: List[Tuple[WireRef, _HostRamBlock]] = []
            for ref, block in sorted(self._host_ram_blocks.items()):
                previous = block.last_available_bytes
                if available <= previous:
                    # Keep the immediately preceding observation exact. A
                    # later event must prove a positive change from this real
                    # state, not from a stale high-water mark.
                    block.last_available_bytes = available
                    continue
                required = int(block.failure_event.host_ram_required_bytes)
                if available < required:
                    block.last_available_bytes = available
                    continue
                satisfied.append((ref, block))
            if not satisfied:
                return

            self._host_ram_generation += 1
            generation = self._host_ram_generation
            events: List[Tuple[WireRef, pb.ModelEvent]] = []
            for ref, block in satisfied:
                event = self.store.model_event(
                    ref, pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
                    identity=(
                        block.failure_event.snapshot_digest,
                        int(block.failure_event.residency_generation),
                    ),
                    host_ram_required_bytes=block.failure_event.host_ram_required_bytes,
                    host_ram_available_before_bytes=block.last_available_bytes,
                    host_ram_available_after_bytes=available,
                    host_ram_evicted_refs=released,
                    host_ram_capacity_generation=generation,
                )
                # Cache/pop before enqueue: a transport rotation cannot lose
                # the satisfying observation; HelloAck replays this generation.
                self._host_ram_progress[ref] = event
                self._host_ram_blocks.pop(ref, None)
                self._queue_host_ram_event_locked(event)
                events.append((ref, event))
        # As with failures, every satisfied ref is committed atomically before
        # the first send and remains replayable if this task is cancelled.
        await self._flush_host_ram_outbox()
        for ref, event in events:
            logger.info(
                "host-RAM capacity progressed ref=%s generation=%d "
                "required=%d available_before=%d available_after=%d released_refs=%s",
                ref,
                generation,
                event.host_ram_required_bytes,
                event.host_ram_available_before_bytes,
                event.host_ram_available_after_bytes,
                list(event.host_ram_evicted_refs),
            )

    async def _clear_host_ram_capacity(self, refs: List[WireRef]) -> None:
        """Drop stale block/replay state after the ref is actually resident."""
        async with self._host_ram_lock:
            for ref in refs:
                self._host_ram_blocks.pop(ref, None)
                self._host_ram_progress.pop(ref, None)
            ref_set = set(refs)
            for ref in ref_set:
                self._host_ram_outbox.pop(ref, None)

    async def host_ram_capacity_delivered(self, event: pb.ModelEvent) -> None:
        """Retire only matching satisfied evidence after stream.write succeeds."""
        if event.state != pb.MODEL_STATE_HOST_CAPACITY_PROGRESS:
            return
        async with self._host_ram_lock:
            current = self._host_ram_progress.get(event.ref)
            if (
                current is not None
                and current.host_ram_capacity_generation
                == event.host_ram_capacity_generation
            ):
                self._host_ram_progress.pop(event.ref, None)

    async def host_ram_capacity_replay(self) -> List[pb.WorkerMessage]:
        """Snapshot active failures, then undelivered progress, for reconnect."""
        async with self._host_ram_lock:
            failures = sorted(
                (block.failure_event for block in self._host_ram_blocks.values()),
                key=lambda event: (event.host_ram_capacity_generation, event.ref),
            )
            progress = sorted(
                self._host_ram_progress.values(),
                key=lambda event: (event.host_ram_capacity_generation, event.ref),
            )
            return [
                pb.WorkerMessage(model_event=event)
                for event in [*failures, *progress]
            ]

    async def _reclaim_disk_file_cache(
        self, candidate_refs: List[str], incoming_paths: List[Path],
    ) -> int:
        """Advise only idle immutable snapshots out of file cache.

        A DISK transition preserves model bytes, but recently-read clean pages
        can still fill the pod cgroup and make the following load look
        impossible.  Protect the incoming snapshot and every still-loaded or
        executing ref by inode; then advise only candidate refs that truthfully
        reached DISK.  Candidates may have been evicted during this admission
        or during an earlier rotation.  The caller always re-probes measured
        headroom afterward.

        This runs inside ``_setup_locked``'s process-wide load lock.  Every
        setup and ordinary RAM->VRAM promotion takes that same lock, so a DISK
        ref cannot be reloaded or promoted between this tier check and the
        blocking file-advice scan.
        """
        if not self._load_lock.locked():
            raise RuntimeError("snapshot file-cache reclaim requires the load lock")
        res = self.store.residency
        preserve = list(incoming_paths)
        for live_ref, tier, _vram_bytes in res.snapshot():
            if (
                tier not in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM)
                and not res.in_use(live_ref)
            ):
                continue
            local = res.local_path(live_ref)
            if local is not None:
                preserve.append(local)

        advised = 0
        seen_paths: set[Path] = set()
        for ref in dict.fromkeys(candidate_refs):
            if res.tier(ref) is not residency_mod.Tier.DISK or res.in_use(ref):
                continue
            local = res.local_path(ref)
            if local is None or local in seen_paths:
                continue
            seen_paths.add(local)
            advised += await asyncio.to_thread(
                disk_gc.reclaim_file_cache,
                local,
                preserve_paths=tuple(preserve),
            )
        return advised

    async def _ensure_host_ram_for(
        self,
        spec: EndpointSpec,
        paths: Dict[str, str],
    ) -> None:
        """Owner-aware host-RAM admission (gw#407/pgw#541). ``from_pretrained``
        stages the full weight set in host RAM before placement; loading into
        a nearly-full host pushes it into reclaim-thrash that stalls the whole
        process — including gRPC keepalive acks — so the hub disconnects and
        requeues in a livelock (J17: 16 SDXL variants on a 31GB host).

        A warm pipeline is owned by both Residency and its endpoint
        ``_ClassRecord``. Clearing only the Residency reference reports
        ON_DISK while ``record.instance`` still owns every tensor. Evict
        record-owned victims through ``records.vacate_record``; only ownerless
        entries may use ``release_to_disk`` directly. Re-probe observed RAM
        after every teardown and fail RETRYABLE if the real headroom still
        cannot cover the incoming bytes plus the derived floor.

        Only worker-loaded (pipeline-typed) slots count: tenant-owned and
        engine-runtime slots do not stage full weight sets in host RAM.

        Multi-slot setups stage SEQUENTIALLY under the load lock — each
        slot's weights move to VRAM (freeing host RAM) before the next slot
        loads — so the honest staging requirement is the LARGEST slot, not
        the sum (gw#479 live: two 28GiB fp8 lanes were refused as "56.2GiB
        incoming" on a 61GiB host that stages at most 28GiB at once).

        pgw#1026 applies the SAME rule one level down for a modular slot the
        loader will stage component-by-component onto the device: its
        requirement is its largest COMPONENT, not its tree. That is the only
        thing standing between a tree the card holds and a structural
        `HostRamCapacityError` (ie#615's H3: 134.1 GiB tree, 116.4 GiB host).
        The verdict comes from `plan_streamed_hydration`, which the loader
        re-reads — one authority, and it engages only when the whole tree
        does not fit while the card does."""
        slots = self._worker_loaded_slot_types(spec)
        if not paths or not slots:
            return
        incoming = 0
        incoming_refs: List[str] = []
        for slot, p in paths.items():
            if slot in slots:
                slot_bytes = await asyncio.to_thread(disk_gc.tree_bytes, Path(p))
                ref = wire_ref(spec.models[slot])
                if is_modular_pipeline_class(slots[slot]):
                    # pgw#1063: the discount below is the loader's promise
                    # that each component LEAVES the host for the card. The
                    # rung decides whether that promise can be kept, so the
                    # admission and the loader read the same one — a
                    # CPU-offload rung (including the sticky floor an OOM
                    # degrade learned) is charged its whole tree here and
                    # stages whole-tree there.
                    plan = await asyncio.to_thread(
                        functools.partial(
                            plan_streamed_hydration, Path(p),
                            placement_mode=self._placement_mode(spec, ref)),
                    )
                    if plan.engaged:
                        logger.info(
                            "host-RAM admission charges %s slot %s its "
                            "largest COMPONENT, not its tree: %s",
                            spec.name, slot, plan.summary())
                        slot_bytes = plan.largest_unit_bytes
                if slot_bytes > incoming:
                    incoming = slot_bytes
                    incoming_refs = [ref]
                elif slot_bytes == incoming and slot_bytes > 0:
                    incoming_refs.append(ref)
        if incoming <= 0:
            return
        res = self.store.residency
        before = await asyncio.to_thread(res.host_ram_headroom, incoming)
        if before.sufficient:
            return

        evicted: List[str] = []
        after = before
        for ref in (WireRef(v) for v in res.lru_ram_victims()):
            # A previous record teardown may already have transitioned every
            # ref that appeared in the snapshot of LRU candidates.
            if res.tier(ref) is not residency_mod.Tier.RAM:
                continue
            owners = records_holding(self._classes.values(), ref)
            if len(owners) > 1:
                # A ref shared by several endpoint instances is not an
                # ownership key. Their unique refs drive record teardown.
                continue
            rec = owners[0] if owners else None
            if rec is not None:
                if record_in_use(rec, records=self._classes.values(), jobs=self.jobs.values(), residency=self.store.residency, reclaim_ref=ref):
                    continue
                owned = [
                    held for held in record_refs(rec)
                    if res.tier(held) in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM)
                ]
                released = await vacate_record(rec, self.teardown_seam)
                evicted.extend(released)
                logger.info(
                    "host-RAM admission vacated warm record refs=%s for %s",
                    released or owned, spec.name,
                )
            elif await asyncio.to_thread(res.release_to_disk, ref):
                released = [ref]
                evicted.extend(released)
                logger.info(
                    "host-RAM admission released ownerless warm ref=%s for %s",
                    ref, spec.name,
                )
            else:
                continue

            # Let completed demotion/to_thread frames release their arguments,
            # then collect after the record owner is gone.  Pinned swap returns
            # dead tensors to PyTorch's process-wide host cache, not the OS, so
            # release its unused blocks first and re-probe.  Only if that is
            # still insufficient do we chill clean snapshot pages for refs
            # that truthfully reached DISK; every model byte stays local.
            await asyncio.sleep(0)
            await asyncio.to_thread(flush_memory)
            released_pinned = await asyncio.to_thread(
                release_unused_pinned_host_cache)
            if released_pinned:
                logger.info(
                    "host-RAM admission released %d unused pinned-host bytes "
                    "after vacating refs=%s for %s",
                    released_pinned, released, spec.name,
                )
            after = await asyncio.to_thread(res.host_ram_headroom, incoming)
            await self._observe_host_ram_progress(released)
            if after.sufficient:
                return
            advised = await self._reclaim_disk_file_cache(
                released, [Path(p) for p in paths.values()],
            )
            if advised:
                logger.info(
                    "host-RAM admission advised %d immutable snapshot bytes "
                    "out of file cache for %s",
                    advised, spec.name,
                )
            after = await asyncio.to_thread(res.host_ram_headroom, incoming)
            await self._observe_host_ram_progress(released)
            if after.sufficient:
                return

        # A prior rotation may already have truthfully moved every old model
        # to DISK.  In that state there is no RAM-tier victim above, but clean
        # pages from loading those immutable snapshots can still occupy the
        # conservative cgroup working set (active_file is deliberately not
        # counted as immediately available).  Chill the oldest idle DISK refs
        # before declaring the host incapable.  This does not delete model
        # bytes and the inode-preservation gate above protects the incoming
        # model plus every loaded/executing component.
        incoming_ref_set = set(incoming_refs)
        disk_refs = self.store.lru_disk_refs(
            exclude=tuple(incoming_ref_set),
        )
        advised = 0
        for ref in disk_refs:
            advised += await self._reclaim_disk_file_cache(
                [ref], [Path(p) for p in paths.values()],
            )
            after = await asyncio.to_thread(res.host_ram_headroom, incoming)
            await self._observe_host_ram_progress([])
            if after.sufficient:
                if advised:
                    logger.info(
                        "host-RAM admission advised %d already-DISK immutable "
                        "snapshot bytes out of file cache for %s",
                        advised, spec.name,
                    )
                return
        if advised:
            logger.info(
                "host-RAM admission advised %d already-DISK immutable "
                "snapshot bytes out of file cache for %s without reaching "
                "required headroom",
                advised, spec.name,
            )

        # pgw#752: a shortfall the whole host cannot cover is not pressure —
        # it is this pod SIZE's verdict, and re-dispatching buys an identical
        # pod that refuses identically (th#1228). Report it as a hardware axis
        # so the function self-disables here and the orchestrator learns the
        # required-vs-total placement fact.
        cls = HostRamCapacityError if after.structural else InsufficientHostRamError
        error = cls(
            spec.name,
            incoming_bytes=incoming,
            floor_bytes=after.floor_bytes,
            required_bytes=after.required_bytes,
            available_before_bytes=before.available_bytes,
            available_after_bytes=after.available_bytes,
            evicted_refs=tuple(_canonical_host_ram_refs(evicted)),
            total_bytes=after.total_bytes,
        )
        # th#807: model-failure state is the scheduler's typed capacity seam.
        # Only the largest sequentially staged ref(s) caused this admission
        # decision; failing a smaller shared VAE would poison unrelated jobs.
        await self._record_host_ram_failure(incoming_refs, error)
        raise error

    async def _make_room_for(self, spec: EndpointSpec, setup_slots: List[str]) -> None:
        """Evict idle LRU pipelines before loading instead of degrading the
        new load down the offload ladder (#371).

        Estimate per ref (pgw#636): a prior measured vram_hint, else the
        snapshot's actual byte total (the wire manifest sizes — an honest
        first-load footprint for stored-precision lanes; make_room's margin
        covers slack), else — only when NO byte facts exist at all — the
        endpoint's declared ``vram_gb``. The declaration is a PLACEMENT
        MINIMUM ("give me a card with at least this much"), never a per-load
        reservation: reserving it wholesale on every never-seen checkpoint
        pick evicted the resident pipeline on 24 GB cards and pinned the
        fleet to one-pipeline-per-worker (the live 9.8/24 GB incident)."""
        res = self.store.residency
        refs = [wire_ref(spec.models[s]) for s in setup_slots]
        needed = _estimate_setup_need(
            [
                (res.vram_hint(r), sum(self.store.component_sizes(r).values()))
                for r in refs
            ]
        )
        if needed <= 0:
            return
        # CPU-only workers do not have a VRAM tier to admit against.
        if torch is None or not cuda_ready():
            return
        # This job's own reservations are the demand being satisfied here —
        # exclude them from the outstanding-claim accounting (pgw#641 Stage 2).
        make_room = functools.partial(res.make_room, needed, for_refs=refs)
        if await asyncio.to_thread(make_room):
            self._on_state_change()
            return
        # Movable demotions weren't enough: tear down idle records holding
        # non-movable LRU victims (tenant-loaded refs).
        for ref in res.lru_vram_victims():
            owners = records_holding(self._classes.values(), ref)
            if len(owners) != 1:
                # Shared refs cannot identify which instance owns the
                # residency object; wait for a unique record-owned victim.
                continue
            rec = owners[0]
            if record_in_use(rec, records=self._classes.values(), jobs=self.jobs.values(), residency=self.store.residency, reclaim_ref=ref):
                continue
            await vacate_record(rec, self.teardown_seam)
            if await asyncio.to_thread(make_room):
                self._on_state_change()
                return
        # No arbitrary refusal at the recommendation boundary: vram_gb names
        # a target card, not a hard free-byte requirement. If pinned work
        # prevents full headroom, the freshly materialized pipeline's exact
        # size drives place_pipeline's offload decision before any CUDA move.
        self._on_state_change()

    def _placement_mode(self, spec: EndpointSpec, ref: str) -> str:
        """Placement for one concrete model ref on this worker.

        The hardware gate is function-wide and stable. Reactive OOM floors
        are ref-specific: one large or malformed dynamic pick must never
        spill every sibling pick to CPU offload.
        """

        plan = self._gate_serve_plans.get(spec.name)
        mode = "model_offload" if (
            plan is not None and plan.run_mode == RUN_OFFLOAD
        ) else "auto"
        floor = self.degraded_floor.get(ref, "")
        if floor:
            mode = rungspec.floor_of("" if mode == "auto" else mode, floor)
        return mode

    @staticmethod
    def _setup_slots(spec: EndpointSpec) -> List[str]:
        """Model slots loaded once at setup time. Classes without setup()
        take their models per call via handler-parameter injection."""
        if spec.cls is None or not spec.models:
            return []
        if getattr(spec.cls, "setup", None) is None:
            return []
        return list(spec.models)

    async def _boot_engine_server(self, spec: EndpointSpec, paths: Dict[str, str]) -> Any:
        """Boot the runtime="vllm"/"llama-server" subprocess and health-wait."""

        assert spec.runtime  # validated at decoration
        factory = RUNTIME_FACTORIES[spec.runtime]
        if not paths:
            raise ValidationError(
                f"runtime={spec.runtime!r} on {spec.name!r} requires a model binding"
            )
        model_path = next(iter(paths.values()))
        proc = factory(model_path)
        return await asyncio.to_thread(proc.start)

    def _component_share_plan(
        self, spec: EndpointSpec, paths: Dict[str, str], hints: Dict[str, Any]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Content-keyed shared-component plan (gw#479 / pgw#636, SDK v2):
        ``{slot: {component: LoadedComponentKey}}``. Sharing is AUTOMATIC by
        content address (pgw#647 deleted the ``Slot.share_components``
        opt-in — an endpoint can no longer forget to share): every
        component of a hub-resolvable (Slot-declared) pipeline slot becomes
        an independent content-keyed residency entry, so later picks with
        equal bytes alias it and unequal bytes stay honestly exclusive. A
        component also participates when its content key appears under 2+
        pipeline slots of THIS record (the multi-lane z-image/qwen shape)
        or when an entry for the key is ALREADY resident in the shared
        cache (a sibling pick's record seeded it). None when nothing
        qualifies — loading then stays monolithic."""
        pipe_slots = [
            s for s in paths
            if isinstance(hints.get(s), type)
            and callable(getattr(hints[s], "from_pretrained", None))
        ]
        declared: Dict[str, frozenset] = {
            s: _ALL_COMPONENTS for s in pipe_slots if s in spec.slots
        }
        if len(pipe_slots) < 2 and not declared:
            return None
        keys: Dict[str, Dict[str, Any]] = {}
        for slot in pipe_slots:
            binding = spec.models.get(slot)
            if binding is None:
                return None
            ref = wire_ref(binding)
            if ref in self._no_share_refs:
                # pgw#678: this ref proved un-shareable on this host (its lane
                # landed offloaded, which the shared-component invariant
                # refuses). Retrying the same shared plan is a dead end —
                # load it monolithically instead.
                continue
            # pgw#683: identity must carry the EFFECTIVE compute dtype, not the
            # binding's DECLARED one. Hub bindings declare no dtype, so every
            # flavor of a ref answered "" and byte-identical non-denoiser
            # components (a quantizer only rewrites the denoiser, so the VAE and
            # text encoders of `X` and `X#fp8-w8a8` are the SAME bytes) shared
            # ONE cache entry across compositions that compute at DIFFERENT
            # dtypes: a quant-artifact tree computes bf16, a plain fp16-stored
            # mirror computes fp16. Whichever pick loaded first won, and the
            # loser aliased a foreign-precision module into its own
            # composition — a Half nn.Linear meeting a bf16 activation is
            # `mat1 and mat2 must have the same dtype, but got BFloat16 and
            # Half`, with no component override anywhere on the wire.

            effective_dtype = composition_compute_dtype(
                paths[slot], str(getattr(binding, "dtype", "") or ""))
            digests = self.store.component_digests(ref, local_path=Path(paths[slot]))
            keys[slot] = {
                comp: residency_mod.LoadedComponentKey.for_component(
                    content_digest=digest, component=comp, binding=binding,
                    dtype=effective_dtype, label=f"{ref}/{comp}",
                )
                for comp, digest in digests.items() if comp
            }
        if not keys:
            return None
        counts: Dict[Any, int] = {}
        for slot_keys in keys.values():
            for k in slot_keys.values():
                counts[k] = counts.get(k, 0) + 1
        res = self.store.residency
        plan = {
            slot: {
                c: k for c, k in slot_keys.items()
                if counts[k] >= 2
                or c in declared.get(slot, frozenset())
                or res.shared_obj(k) is not None
            }
            for slot, slot_keys in keys.items()
        }
        if not any(plan.values()):
            return None
        shared = sorted({c for m in plan.values() for c in m})
        logger.info(
            "content-keyed lanes for %s: shared components %s across %d slots",
            spec.name, shared, len(pipe_slots),
        )
        return plan

    def _shared_group_force_fp8(
        self, spec: EndpointSpec, share_plan: Optional[Dict[str, Dict[str, Any]]],
    ) -> set:
        """Slots that must force fp8 denoiser storage to fit their shared-
        component group jointly resident (th#1043) — empty when the group
        fits at native precision, or fits nobody's precision at all."""
        plan = share_plan or {}
        slots = [s for s, m in plan.items() if m]
        if len(slots) < 2:
            return set()
        shared_components = sorted({c for m in plan.values() for c in m})
        slot_sizes: Dict[str, Dict[str, int]] = {}
        for slot in slots:
            binding = spec.models.get(slot)
            if binding is None:
                return set()
            slot_sizes[slot] = self.store.component_sizes(wire_ref(binding))
        free = self.store.residency.free_vram_bytes()
        if not _shared_execution_lanes_need_fp8(slot_sizes, shared_components, free):
            return set()
        logger.info(
            "th#1043: shared-lane group %s for %s doesn't fit resident at "
            "native precision (free=%.1fGiB) — forcing fp8 storage on every "
            "lane before any of them loads",
            slots, spec.name, free / (1 << 30),
        )
        return set(slots)

    @staticmethod
    def _model_index_components(path: str) -> set:
        """Component names the snapshot's model_index.json declares — the
        only names safe to pass as preloaded modules to from_pretrained."""
        return provision.model_index_components(path)

    async def _injection_kwargs(
        self,
        spec: EndpointSpec,
        setup: Callable[..., Any],
        paths: Dict[str, str],
        *,
        server: Any = None,
        compile_selection: Optional[_CompileArtifactSelection] = None,
        snapshots: Optional[Dict[WireRef, pb.Snapshot]] = None,
        slot_identities: Optional[Dict[str, _ResidencyIdentity]] = None,
        arm: Optional[_ArmOrder] = None,
        boot_local_key: str = "",
    ) -> "_InjectionResult":
        """Typed injection: each slot receives exactly what its ``setup``
        annotation says — a ``str``/``Path`` local path, or a constructed
        pipeline for a class annotation exposing ``from_pretrained`` (built off
        the loop; the binding dtype is honored and the worker applies its
        placement/offload policy to the result). A parameter annotated
        ``ServerHandle`` receives the booted engine server.

        Multi-lane records (gw#479): when 2+ pipeline slots carry
        byte-identical components (content keys), the first lane loads them
        and registers them in the shared cache; later lanes inject the very
        same module objects into ``from_pretrained`` and load only their
        exclusive weights. Each lane's residency entry is then the exclusive
        module set — LRU swap moves ONLY the transformer, never the shared
        encoder. Lane slots are residency-registered inline (per slot) so
        make_room can demote lane N-1 while lane N loads."""

        try:
            hints = {
                k: _unwrap_optional(v)
                for k, v in typing.get_type_hints(setup).items()
            }
        except Exception:
            hints = {}
        kwargs: Dict[str, Any] = {}
        loaded: Dict[str, Tuple[Any, int]] = {}
        result = _InjectionResult(kwargs=kwargs, loaded=loaded)
        compile_artifact = compile_selection.path if compile_selection else None
        share_plan = self._component_share_plan(spec, paths, hints)
        force_fp8_slots = self._shared_group_force_fp8(spec, share_plan)
        if server is not None:
            for pname, ann in hints.items():
                if ann is ServerHandle:
                    kwargs[pname] = server
        try:
            for slot, path in paths.items():
                ann = hints.get(slot)
                if ann is None or ann is str:
                    kwargs[slot] = path
                elif ann is Path:
                    kwargs[slot] = Path(path)
                elif isinstance(ann, type) and callable(getattr(ann, "from_pretrained", None)):
                    binding = spec.models.get(slot)
                    # Worker-owned placement/offload policy: one decider for the
                    # whole worker; endpoints never write device/offload code.
                    # Plan-time offload verdicts and the learned degraded floor
                    # pick the starting rung so a doomed fully-resident attempt
                    # is never paid (gw#463 / ie#369); a CUDA OOM inside is a
                    # ladder transition, not a failure.
                    ref = wire_ref(binding) if binding is not None else WireRef("")
                    mode = self._placement_mode(spec, ref)
                    slot_share = dict((share_plan or {}).get(slot) or {})
                    if slot_share and mode != "auto":
                        # Offload hooks on a shared module would poison sibling
                        # lanes; a planned-offload record loads monolithically.
                        logger.warning(
                            "content-keyed sharing disabled for %s slot %s: "
                            "placement mode %s", spec.name, slot, mode)
                        slot_share = {}
                    res = self.store.residency
                    injected: Dict[str, Any] = {}
                    if slot_share:
                        valid = self._model_index_components(path)
                        for comp, key in list(slot_share.items()):
                            if comp not in valid:
                                del slot_share[comp]
                                continue
                            if res.shared_obj(key) is not None:
                                injected[comp] = res.acquire_shared(
                                    key, _shared_loader_must_hit)
                                result.shared_keys.append(key)
                        # Exclusive-weights headroom BEFORE the load: demote idle
                        # LRU lanes now so placement never has to walk the
                        # offload ladder mid-lane (dual-resident when the budget
                        # admits, swap-mode otherwise — existing make_room path).
                        sizes = self.store.component_sizes(ref)
                        excl_bytes = sum(
                            b for comp, b in sizes.items() if comp not in injected)
                        if excl_bytes > 0:
                            await _to_thread_complete(functools.partial(
                                res.make_room, excl_bytes, for_refs=(ref,)))
                    before = cuda_allocated_bytes()
                    try:
                        sl = await _to_thread_complete(
                            provision.load_slot, ann, path, binding=binding,
                            slot=slot, ref=ref, mode=mode, components=injected,
                            force_storage_dtype=(
                                "fp8" if slot in force_fp8_slots else ""),
                        )
                    except Exception as exc:
                        # Corruption-shaped load failure (gw#408): digest-verify
                        # the snapshot; quarantine + re-materialize + retry ONCE
                        # when corruption is confirmed, re-raise otherwise.
                        fresh: Optional[Path] = None
                        if binding is not None and _is_corrupt_load_error(exc):
                            fresh = await self.store.refetch_corrupt(
                                ref, (snapshots or {}).get(ref), binding=binding
                            )
                        if fresh is None:
                            raise
                        logger.warning(
                            "weights load for slot %r failed on a corrupt snapshot "
                            "(%s: %s); retrying once after re-materialization",
                            slot, type(exc).__name__, exc,
                        )
                        path = str(fresh)
                        paths[slot] = path
                        sl = await _to_thread_complete(
                            provision.load_slot, ann, path, binding=binding,
                            slot=slot, ref=ref, mode=mode, components=injected,
                            force_storage_dtype=(
                                "fp8" if slot in force_fp8_slots else ""),
                        )
                    pipe = sl.obj
                    # pgw#678: record the PIPELINE identity for this slot
                    # before any lane bookkeeping can shadow it.
                    result.slot_pipelines[slot] = pipe
                    # pgw#654: generic materialization tuning — per-request
                    # progress bars are worker noise on every diffusers
                    # pipeline; endpoints never write this line.
                    if callable(getattr(pipe, "set_progress_bar_config", None)):
                        try:
                            pipe.set_progress_bar_config(disable=True)
                        except Exception:
                            pass
                    # Reconcile the load outcomes into ServePlan/FnDegraded via
                    # the state-delta path — the shared core decides WHAT
                    # degraded (details non-empty), the executor reports it.
                    if sl.pre_drop_detail:
                        self._record_rung_transition(
                            spec, ref=ref, phase="load",
                            wanted=sl.pre_drop_wanted, ran=sl.ran,
                            detail=sl.pre_drop_detail)
                    if sl.rung_detail:
                        self._record_rung_transition(
                            spec, ref=ref, phase="load",
                            run_mode=RUN_FP8_STORAGE,
                            detail=sl.rung_detail)
                    elif sl.cast_fail_detail:
                        self._record_rung_transition(
                            spec, ref=ref, phase="load",
                            wanted=sl.cast_fail_wanted, ran=sl.ran,
                            detail=sl.cast_fail_detail)
                    placed = sl.placed
                    # th#1871 P1 §6.6 item 3: the posture is recorded for EVERY
                    # placement, not only the OOM-demoted one below. The
                    # `oom_demotions` gate is the biggest blind spot the census
                    # found — a pipeline that `select_auto_mode` proactively put
                    # on the offload ladder never OOMs, so it reported nothing,
                    # served 2.5-4x slow, and its numbers were filed as
                    # measurements of a resident run.
                    self._record_placement_posture(spec, ref=ref, placed=placed)
                    if placed.get("oom_demotions"):
                        self._record_rung_transition(
                            spec, ref=ref, phase="load",
                            from_rung=str(placed.get("requested_mode") or mode),
                            to_rung=str(placed.get("mode") or ""),
                            run_mode=RUN_OFFLOAD,
                            needed_gb=estimate_pipeline_size_gb(pipe),
                            detail="CUDA OOM at load; pipeline placed offloaded",
                        )
                    if slot_share and str(placed.get("mode") or "") not in (
                        "", "off", "vae_only", "cpu",
                    ):
                        # pgw#678: LEARN it. Re-deriving the same share plan on
                        # every retry made this a silent dead end
                        # (retry_exhausted -> worker_function_unavailable, live
                        # on the sdxl turbo lane). The ref is marked
                        # un-shareable so the retry composes monolithically,
                        # where an offload rung is legal.
                        self._no_share_refs.add(ref)
                        raise RetryableError(
                            f"lane {slot!r} of {spec.name} placed "
                            f"{placed.get('mode')!r}: shared-component lanes "
                            "require resident placement; retrying without "
                            "content-keyed sharing for this ref")
                    if spec.compile is not None:
                        # Opt-in acceleration against a pre-built per-SKU
                        # inductor cache (#384). No verified artifact =>
                        # stays eager. ``compile_artifact`` is hub-attached (#569).

                        # pgw#677 reopen: stamp the hub-resolved execution
                        # lane on the pipe BEFORE arming, so the router's
                        # fail_closed and the eager-first eligibility both
                        # read the ONE serveability brain
                        # (compile_cache.mandatory_serving) instead of the
                        # weight-lane prefix. Never overwritten once set.
                        # pgw#1113: and stamp WHAT this pipe was resolved
                        # from, for the same reason and at the same seam. The
                        # arm token is an obligation, and an obligation that
                        # cannot name its subject dedups two checkpoints into
                        # one pending, one child and one memo row.
                        if binding is not None:
                            fleet_cells.stamp_arm_subject(
                                pipe, slot, [wire_ref(binding)],
                                (slot_identities or {}).get(slot, ("", 0))[0],
                            )
                        exec_execution_lane, lane_pinned = (
                            self._execution_lane_pick_for_ref(ref))
                        if exec_execution_lane and not getattr(
                                pipe, compile_cache.EXECUTION_LANE_ATTR,
                                None):
                            try:
                                setattr(
                                    pipe,
                                    compile_cache.EXECUTION_LANE_ATTR,
                                    exec_execution_lane)
                                setattr(
                                    pipe,
                                    compile_cache.EXECUTION_LANE_PINNED_ATTR,
                                    lane_pinned)
                            except Exception:
                                pass

                        try:
                            outcome = await _to_thread_complete(
                                self._enable_compiled,
                                pipe, spec.compile_cell(), compile_artifact,
                                compile_selection, arm, boot_local_key,
                            )
                        except compile_cache.CompiledExecutionLaneUnavailableError as exc:
                            # Mandatory (w8a8/w4a4) lane: self-mint also hit a
                            # genuine impossibility (no CUDA/toolchain/target).
                            # When this refusal was chained from a caught
                            # compiled_graph_selection_bug (th#1031), report it — the
                            # lane refusal must not silently swallow the
                            # loud invariant event.
                            bug = exc.__cause__
                            if isinstance(bug, compile_cache.CellSelectionBugError):
                                await self._report_cell_selection_bug(
                                    spec, compile_selection, bug)
                            raise
                        armed = outcome.armed
                        if armed and not any(
                                p is pipe for p in result.armed_objects):
                            # pgw#1093: the injection-scope half of the arm
                            # fact (the scope half is in `_setup_instance`).
                            result.armed_objects.append(pipe)
                        pipe_mint = outcome.self_mint
                        result.adoptions.extend(outcome.adoptions)
                        # pgw#824: the arming brain already classified WHY it
                        # is not arming; without carrying it here the reason
                        # died at the end of this function and every eager
                        # request the pod then served reported "".
                        if not armed and outcome.eager_reason:
                            result.eager_postures.append(outcome.eager_reason)
                        if outcome.selection_bug is not None:
                            # th#1031: no longer fatal — this SAME call
                            # already fell through to self-mint (or, for a
                            # plain lane, eager); still reported loudly so
                            # the th#883 invariant stays wire-visible.
                            await self._report_cell_selection_bug(
                                spec, compile_selection, outcome.selection_bug)

                        if compile_cache.has_compile_target(pipe, spec.compile):
                            result.add_compile_object(pipe, (slot,))
                            selection = _selection_for(compile_selection, pipe_mint)
                            # pgw#784: a DELEGATED mint arms NOTHING, so it
                            # reports armed=False and must still be recorded —
                            # the obligation is real, it is just owed to a
                            # child process. It deliberately does NOT enter
                            # active_compile_artifacts: this pipe serves eager,
                            # and claiming an active artifact for it would
                            # advertise bytes it does not serve (gw#586).
                            if getattr(pipe_mint, "delegated", False):
                                result.pending_self_mints[id(pipe)] = pipe_mint
                            elif armed and selection is not None:
                                result.active_compile_artifacts[id(pipe)] = selection
                    delta = max(0, cuda_allocated_bytes() - before)
                    if slot_share:
                        execution_lane_obj, execution_lane_bytes = self._register_execution_lane(
                            slot,
                            ref,
                            pipe,
                            slot_share,
                            injected,
                            delta,
                            result,
                            (slot_identities or {}).get(slot, ("", 0)),
                        )
                        loaded[slot] = (execution_lane_obj, execution_lane_bytes)
                        result.execution_lane_slots.add(slot)
                    else:
                        loaded[slot] = (pipe, delta)
                        if self._arm_lane_residency_gate(pipe, ref, spec=spec):
                            result.gated_slots.add(slot)
                    kwargs[slot] = pipe
                else:
                    kwargs[slot] = path
        except BaseException:
            # gw#624: a failed/cancelled injection never reached
            # ``rec.shared_keys.extend`` — roll back the shared-component
            # holds acquired so far or their refcounts leak forever (the
            # components become permanently unevictable, and each retry
            # re-acquires on top).
            for key in result.shared_keys:
                try:
                    self.store.residency.release_shared(key)
                except Exception:
                    logger.exception(
                        "failed to release shared hold %r after aborted "
                        "injection", key)
            raise
        return result

    def _register_execution_lane(
        self,
        slot: str,
        ref: WireRef,
        pipe: Any,
        slot_share: Dict[str, Any],
        injected: Dict[str, Any],
        delta: int,
        result: "_InjectionResult",
        load_identity: _ResidencyIdentity,
    ) -> Tuple[Any, int]:
        """Book one lane's residency (gw#479): freshly loaded shared
        components go into the content-keyed cache (VRAM counted once, held
        by refcount); the lane's own entry is its EXCLUSIVE module set, so
        LRU demote/promote swaps only lane-owned weights (the transformer),
        never the shared encoder."""
        import torch.nn as nn

        res = self.store.residency
        fresh_bytes = 0
        for comp, key in slot_share.items():
            if comp in injected:
                continue
            module = getattr(pipe, comp, None)
            if module is None:
                continue
            measured = 0
            if isinstance(module, nn.Module):
                measured = int(estimate_cuda_resident_gb(module) * _GiB)
            def _hold(m: Any = module) -> Any:
                return m

            res.acquire_shared(key, _hold, vram_bytes=measured)
            result.shared_keys.append(key)
            fresh_bytes += measured
        comps = getattr(pipe, "components", None) or {}
        exclusive = {
            name: m for name, m in comps.items()
            if isinstance(m, nn.Module) and name not in slot_share
        }
        execution_lane_obj: Any = nn.ModuleDict(exclusive) if exclusive else pipe
        execution_lane_bytes = max(0, delta - fresh_bytes)
        result.shared_bytes += fresh_bytes
        logger.info(
            "lane %s (%s): exclusive %s (%.2f GiB), shared %s (%.2f GiB %s)",
            slot, ref, sorted(exclusive) or ["<none>"], execution_lane_bytes / _GiB,
            sorted(slot_share), fresh_bytes / _GiB,
            "fresh" if fresh_bytes else "reused",
        )
        self.store.activate_load_identity(ref, load_identity)
        if execution_lane_bytes > 0:
            res.track_vram(ref, execution_lane_obj, vram_bytes=execution_lane_bytes)
        elif int(estimate_cuda_resident_gb(execution_lane_obj) * _GiB) > 0:
            res.track_vram(ref, execution_lane_obj)
        else:
            res.track_ram(ref, execution_lane_obj)
        if self._arm_lane_residency_gate(pipe, ref):
            result.gated_slots.add(slot)
        return execution_lane_obj, execution_lane_bytes

    def _arm_lane_residency_gate(
        self, pipe: Any, ref: str, spec: Optional[EndpointSpec] = None,
    ) -> bool:
        """gw#551: wrap a worker-constructed pipeline's ``__call__`` so a
        demoted/incomplete residency entry is promoted (pinned, idle sibling
        LRU-swapped out) before it executes — a cpu-resident lane must never
        run. No-op for offload-hooked pipelines (they own their placement).
        Monolithic pipelines (``spec`` given) additionally get the last-resort
        offload fallback; shared-component lanes never do (hooks on a shared
        module would poison sibling lanes)."""

        fallback = None
        if spec is not None:
            bound_spec = spec

            def fallback() -> bool:
                return self._serve_offload_fallback(bound_spec, pipe, ref)
        return arm_lane_residency_gate(pipe, LaneResidencyGate(
            ref=ref, residency=self.store.residency, label=ref,
            retry_exc=RetryableError, offload_fallback=fallback,
        ))

    def _serve_offload_fallback(self, spec: EndpointSpec, pipe: Any, ref: str) -> bool:
        """Serve-time last resort (gw#551): promote could not fit even after
        LRU demotions — arm a coherent CPU-offload rung on the (cpu-resident)
        pipeline and rebook it honestly, instead of failing the request."""

        if not rearm_offload(pipe):
            return False
        # Offload-hooked objects book the RAM tier (their VRAM is hook-owned).
        self.store.residency.track_vram(ref, pipe)
        self._record_rung_transition(
            spec, ref=ref, phase="serve", from_rung="resident",
            to_rung="model_offload", run_mode=RUN_OFFLOAD,
            needed_gb=estimate_pipeline_size_gb(pipe),
            detail="VRAM promote could not fit after LRU demotions; serving "
                   "CPU-offloaded (gw#551)",
        )
        return True

    def _cell_publisher(self) -> "fleet_cells.CellPublisher":
        """The fleet publish sink for self-minted cells (gw#587/th#910).
        Built per call: file_base_url arrives with HelloAck and the worker
        JWT rotates (#561). ``enabled()`` is false until both exist."""

        return fleet_cells.CellPublisher(
            base_url=self.file_base_url,
            worker_jwt=self.worker_jwt_provider,
            image_digest=str(
                getattr(self._settings, "worker_image_digest", "") or ""),
        )

    def _report_no_growth_path(
        self, spec: EndpointSpec, target: "_CompileTargetRecord", pipeline: Any,
    ) -> None:
        """pgw#916: this armed target has NO serve-window shape-growth path.

        ``hot_swap.enable`` returns False whenever the pipeline carries no
        dynamo router, which is every AOT arm by construction —
        ``provision.enable_compiled`` returns as soon as ``arm_aot`` succeeds,
        so ``compile_cache.enable`` (the only thing that installs the router)
        is never reached.  The consequence is total: a class the cell does not
        cover serves eager for the life of the pod, every pod, forever.

        Until this existed the ONLY named observable was a success log line
        that simply never printed — an absence nobody can query.  A silent
        no-op on the serving path is the pgw#760 defect class; this is its
        confession, and it is countable per (release, SKU, arm) hub-side
        exactly the way pgw#680 counts dynamo guard misses.
        """

        arm = shape_growth.ARM_DYNAMO
        if aot_serve.is_armed(pipeline):
            arm = shape_growth.ARM_AOT
        with target.state_lock:
            cell = target.active_compile_ref
        logger.warning(
            "shape-growth: %s is armed on arm=%s with NO serve-window growth "
            "path (pgw#916); every declared class the cell does not cover "
            "serves eager for the life of this pod", spec.name, arm)
        activity_mod.emit_event(
            activity_mod.KIND_SHAPE_GAP,
            f"arm={arm} fn={sorted(target.function_names)} "
            f"cell={cell or '<none>'}: this armed target has no serve-window "
            f"shape-growth path — a request at a class the cell does not "
            f"cover is served eager and NOTHING will grow the cell, on this "
            f"pod or any other",
            phase="no_growth_path",
        )

    # ---- pgw#671 eager-first boot (worker half of th#1187) -----------------

    def _eager_first_eligible(
        self, spec: EndpointSpec, inj: "_InjectionResult",
    ) -> bool:
        """Whether this setup may go READY(eager) with the mint deferred.

        Eager-first applies ONLY to a boot whose every armed artifact is a
        fresh self-mint on an eager-compatible lane: delivered cells keep
        their sequential proof window (they pay ~0 compile), custom object
        warmups have no derived plan for the driver to seed, and regional
        targets have no separable eager callable to route to.

        pgw#813 removes two stacked misclassifications:

        * the ``_mandatory_lane_of_bound`` early-out read a model ref's
          STORAGE flavor (``#fp8-w8a8``) as "cannot serve eager" — the exact
          proxy pgw#677's reopen removed from the router brain, still live
          one layer up, and sdxl's mixed fp8 checkpoint (fp8 storage, w8a16
          execution) is precisely the ref it misreads. The per-candidate
          brains below answer the question correctly and completely;
        * the per-candidate arm demanded a hot-swap ROUTER. A DELEGATED
          pending never has one, because nothing is armed on its pipe by
          construction — so EVERY delegated mint failed this test and was
          discarded, and pgw#784's out-of-process route could not run on any
          lane. A delegated pending's eager tier is the untouched pipeline
          itself; the router question belongs only to an in-process capture,
          whose eager-while-compiling routing is what a router performs."""
        if not inj.pending_self_mints:
            return False
        if spec.cls is not None and callable(getattr(spec.cls, "warmup", None)):
            return False
        cfg = spec.compile_cell()
        if cfg is None or bool(getattr(cfg, "regional", False)):
            return False
        # Any armed artifact that is NOT a pending self-mint (a delivered
        # cell) keeps today's foreground proof for the whole
        # record — mixing tiers inside one proof window is not worth it.
        if set(inj.active_compile_artifacts) - set(inj.pending_self_mints):
            return False

        saw_candidate = False
        for candidate in inj.compile_objects:
            pending = inj.pending_self_mints.get(id(candidate.pipeline))
            if pending is None:
                continue
            if getattr(pending, "delegated", False):
                # Nothing is armed on this pipe; the child owns the compile.
                # The only real question is whether the live object can still
                # answer a forward — which every lane the fleet serves can,
                # quantized ones included (pgw#813).
                if not compile_cache.eager_tier_available(candidate.pipeline):
                    return False
                saw_candidate = True
                continue
            # In-process capture: the pipe IS armed cold, so eager serving
            # happens through the pgw#622 router. pgw#677 reopen: the ONE
            # serveability brain decides fail-closed here (hub execution lane
            # first, weight-lane stamp as fallback), never the ref flavor.
            if compile_cache.mandatory_serving(candidate.pipeline):
                return False
            router = hot_swap.router_of(candidate.pipeline)
            if router is None or router.fail_closed:
                return False
            saw_candidate = True
        return saw_candidate

    def _assert_mint_termini(
        self, spec: EndpointSpec, obligations: List[Any],
        *, driver_owns_delegated: bool = True,
    ) -> None:
        """Every self-mint this boot opened must have ENDED somewhere (pgw#815).

        A mint obligation has exactly five honest ends: sealed-and-publishing,
        withheld, aborted, abandoned, or handed to a background/delegated
        driver that still owns it. Anything else is the vanishing publish —
        a pod pays a full GPU compile, reports `finalize completed`, and the
        store, the receipts, the local arm and the wire are all empty. That
        combination is unfalsifiable from outside, so it is asserted here and
        confessed on the wire rather than left to be re-measured on a rental.
        """
        if not obligations:
            return

        for pending in obligations:
            if fleet_cells_mod.terminus_of(pending):
                continue
            if driver_owns_delegated and getattr(pending, "delegated", False):
                # The supervisor owns it; `_supervise_mint` resolves it and is
                # the one that must confess if it does not.
                continue
            family = str(getattr(pending, "family", "") or "")
            key = str(getattr(pending, "compiled_graph_key", "") or "")
            logger.error(
                "%s: SELF_MINT_UNRESOLVED family=%s key=%s — the boot opened "
                "a mint capture and reached readiness without packing, "
                "publishing, withholding or abandoning it (pgw#815)",
                spec.name, family, key)
            activity_mod.emit_event(
                "self_mint_abort",
                f"family={family} key={key}: this boot opened a mint capture "
                f"and reached readiness without packing, publishing, "
                f"withholding or abandoning it — no cell, no receipt, no "
                f"local arm and no refusal. The capture is discarded so the "
                f"next pod re-mints instead of inheriting a phantom.",
                phase="no_terminus",
            )
            try:
                fleet_cells_mod.abandon_self_mint(pending)
            except Exception:  # noqa: BLE001 — the confession is the point
                logger.debug("abandoning the unresolved mint failed",
                             exc_info=True)

    def serving_tiers(self) -> Dict[str, str]:
        """Per-function serving tier for the capability projection (th#1187
        wire contract): ``"compiled"`` when a READY record's compile target
        covering the function has a proven active artifact, ``"eager"``
        otherwise (including functions without a compile declaration —
        eager by construction). Never returns ``""``: the empty tier is
        reserved for pre-0.65 workers on the wire.

        The tier is NOT ``serving_mode`` at a coarser grain, and pgw#1032
        deliberately did not merge them. The tier answers *"is this worker
        serving from a CELL"* — the hub reads it as adoption evidence
        (``WorkerServingCompiledTier`` -> ``WorkerAdoptedDeliveredCell``,
        th#1216), so a JIT-intake pod reporting ``compiled`` would testify that
        the cell exchange worked on a pod that adopted nothing. ``serving_mode``
        answers *"what code ran this request"*, where an intake arm is honestly
        ``jit_cell`` (pgw#1010). Two questions, two answers; the apparent
        divergence is the design.
        """
        compiled: set[str] = set()
        for rec in self._classes.values():
            if not rec.ready:
                continue
            for target in rec.compile_targets.values():
                with target.state_lock:
                    if target.active_compile_ref:
                        compiled.update(target.function_names)
        return {
            name: ("compiled" if name in compiled else "eager")
            for name in self.available_functions()
        }

    async def abandon_background_mint(
        self, rec: _ClassRecord, *, reason: str, code: str = "unspecified",
        free_targets: bool = False,
    ) -> None:
        """Cleanly stop an in-flight background mint (adopt-on-arm, vacate,
        shutdown): signal, let the driver finish its current unit, then
        cancel wholesale. Local state is never left half-mutated — the
        capture is either finalized (a sibling proved it) or discarded."""
        bg = rec.background_mint
        if bg is None:
            return
        bg.abandon_code = code
        bg.abandon_reason = reason
        bg.abandon.set()
        task = bg.task
        if task is not None and not task.done():
            logger.info(
                "abandoning background mint for %s (%s)", bg.spec.name, reason)
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=_MINT_ABANDON_GRACE_S)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # The driver's own terminal path cleans up when it observed the
        # signal; this covers a driver that never started or already died.
        if rec.background_mint is bg:
            self._abandon_mint_state(rec, bg)
        if free_targets:
            self._free_mint_targets(bg)

    def _abandon_mint_state(
        self, rec: _ClassRecord, bg: "_BackgroundMint",
        *, free_targets: bool = False,
    ) -> None:
        """Discard-wholesale cleanup: suspend routing concurrency (novel
        signatures return to sequential compile-then-serve) and abandon
        every unfinalized capture. Serving is untouched — the record keeps
        serving eager on the live instance.

        ``free_targets`` (pgw#737): this process will not mint at all, so
        also give the card back — see :meth:`_free_mint_targets`."""

        for pipe in bg.pipes.values():
            router = hot_swap.router_of(pipe)
            if router is not None:
                router.suspend()
        for pending in {id(p): p for p in bg.pendings.values()}.values():
            try:
                fleet_cells_mod.abandon_self_mint(pending)
            except Exception:
                logger.exception("background mint capture cleanup failed")
        if rec.background_mint is bg:
            rec.background_mint = None
        if free_targets:
            self._free_mint_targets(bg)

    def _free_mint_targets(self, bg: "_BackgroundMint") -> None:
        """pgw#737: give the card back after a mint that will not happen.

        A suspended router still leaves the guarded wrappers installed, the
        LoRA branch containers allocated and the allocator holding whatever
        the abandoned capture touched — the tenant's next peak has to fit
        around all of it, and on wan-2.2 it did not. Unwrap to true eager
        (the same end state as the Phase-3 unproven-object branch), drop the
        branch lane, then empty the cache."""

        for pipe in bg.pipes.values():
            try:
                router = hot_swap.router_of(pipe)
                if router is not None:
                    # CLOSE, not suspend: a queued warm job would otherwise
                    # still take its turn and compile onto the card we just
                    # decided we cannot capture on.
                    router.close()
                compile_cache.unwrap(pipe)
                if bg.spec.lora_bucket:
                    compile_cache.drop_lora_execution_lane(pipe)
            except Exception:
                logger.exception("mint target unwrap failed")
        flush_memory()

    async def _background_mint(
        self, rec: _ClassRecord, bg: "_BackgroundMint",
    ) -> None:
        """Drive one deferred boot self-mint to arm, off the serving path.

        Owns the handed-over ``self_mint_compile`` activity: it stays
        RUNNING on the wire for the whole background build (the hub's
        minting classification consumes exactly that) and terminates here —
        COMPLETED on arm or clean abandonment, FAILED on disproof/error
        (serving stays eager either way; a mint failure never un-serves)."""
        act = bg.act
        assert act is not None
        try:
            with activity_mod.watchdog(act):
                await self._supervise_mint(rec, bg, act)
                await self._await_publish_durable(act)
        except (_MintAbandoned, asyncio.CancelledError):
            self._abandon_mint_state(rec, bg)
            logger.info(
                "background mint for %s abandoned cleanly (%s: %s); serving "
                "continues at its current tier",
                bg.spec.name, bg.abandon_code, bg.abandon_reason)
            activity_mod.emit_event(
                "self_mint_abort",
                f"background mint for {bg.spec.name} abandoned "
                f"({bg.abandon_code}"
                + (f": {bg.abandon_reason}" if bg.abandon_reason else "")
                + "); serving continues at its current tier",
                phase=f"abandoned_{bg.abandon_code}",
            )
            act.completed()
        except Exception as exc:
            # pgw#737: free_targets — a failed mint that leaves its wrappers
            # and branch buffers installed keeps charging the tenant for a
            # capture that will never finalize.
            self._abandon_mint_state(rec, bg, free_targets=True)
            logger.warning(
                "background mint for %s failed (%s: %s); serving stays eager "
                "for this process", bg.spec.name, type(exc).__name__, exc)
            # pgw#677 reopen: the abort cause rides the wire typed and
            # countable, in addition to the FAILED activity terminal.
            activity_mod.emit_event(
                "self_mint_abort",
                f"background mint for {bg.spec.name} failed: "
                f"{type(exc).__name__}: {exc}",
                phase="error",
            )
            act.failed(exc)
        else:
            act.completed()
        finally:
            # pgw#815: the driver's own terminus sweep. Every branch above
            # already resolves the common shapes; this makes "resolved" an
            # asserted property of EVERY exit rather than a property of the
            # branches somebody remembered to write.
            self._assert_mint_termini(
                bg.spec, list(bg.pendings.values()),
                driver_owns_delegated=False)
            if rec.background_mint is bg:
                rec.background_mint = None
            # pgw#789: WARM-COMPLETE. The eager-first boot (pgw#671) advertises
            # READY on the eager tier and then mints in the background, so
            # `first_request_servable` is NOT when the pod starts serving
            # compiled — and the gap between them is exactly the window a pod
            # bills at eager speed. This milestone closes it: measured from
            # process start (cumulative), it IS the "time to compiled serving"
            # number an AOT-vs-JIT comparison needs, and `outcome` says whether
            # compiled serving was reached at all.
            # pgw#797: one owner, shared with the inline path in
            # `ensure_setup` — see `_mark_warm_complete`.
            self._mark_warm_complete(rec, bg.spec.name)
            self._on_state_change()

    async def _await_publish_durable(self, act: Any) -> None:
        """Keep the mint's activity RUNNING until its cell is DURABLE.

        pgw#848 item 1. This method's docstring used to be a lie by omission:
        the activity "stays RUNNING for the whole background build" and
        terminates COMPLETED on ARM — but the publish is a background thread
        that outlives the arm, so the window in which the cell EXISTS AND IS
        NOT YET DURABLE had no running activity at all. For a pod nobody is
        watching that window is unprotected, and a mint reaped there has paid
        its entire cost and produced nothing.

        THE CONSTRAINT THAT SHAPES ALL OF THIS: the counter advances on
        DURABLE STATE — a new key starting its upload, a key landing — and
        NEVER on a retry attempt or a message. So a publish that fails and
        retries forever goes stale here and IS condemned, which is the whole
        point: a liveness signal a failing retry loop can satisfy is not a
        liveness signal, and widening the window wrongly would be worse than
        the bug it closes. Same reasoning that refused `self_mint_publish` as
        a podguard progress kind.

        Beats the activity ONLY when the durable counter moves, so the
        activity's `UpdatedAt` remains "last PROGRESS", not "last poll".
        """
        try:
            from . import fleet_cells as fc

            last = fc.publish_durable_progress()
            while fc.publishes_in_flight():
                await asyncio.sleep(_PUBLISH_SETTLE_POLL_S)
                current = fc.publish_durable_progress()
                if current == last:
                    continue  # no durable movement: let the window age
                last = current
                phase = getattr(act, "phase", None)
                if callable(phase):
                    phase("publishing")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — never fail a mint on its own telemetry
            logger.debug("publish-durability wait failed", exc_info=True)

    def _advertise_compiled_graphs(
        self, rec: _ClassRecord, bg: "_BackgroundMint", act: Any,
        finalized: Dict[int, Any],
    ) -> None:
        """Activate a finalized self-mint identity on the live targets.

        State stays READY throughout — the tier flips eager->compiled in the
        next capability projection — and pgw#622 stays alive for post-mint
        novel shapes. RESIDENCY, deliberately kept out of ``mint_supervisor``:
        which pipe holds what, and what the wire is told about it, is this
        module's question — the supervisor answers only "which compiled graphs
        exist and did they arm".

        pgw#1113: ``finalized`` holds exactly the pipes that ARMED the cell.
        The caller no longer expands it across every pid that happened to hold
        the same pending — an advertisement is a claim about what a target
        serves, and a pipe that never had the bytes installed serves eager
        whatever this map says.
        """

        act.phase(activity_mod.PHASE_FINALIZE)
        # pgw#824: the eager posture is DISCHARGED — this record now serves
        # from a cell. Left behind, a stale token would misattribute a later,
        # unrelated un-arm (guard revocation) to whatever declined at boot.
        rec.eager_posture = ""
        for pid, outcome in finalized.items():
            pipe = bg.pipes[pid]
            # pgw#1265: THE FLIP IS CHECKED BEFORE IT HAPPENS. Between the arm
            # (where `arm_aot` took the same verdict) and here sit the publish
            # and whatever a sibling instance did to the card meanwhile, so the
            # question is asked again about the state that actually exists at
            # the flip. The advertisement is a claim that this pipe SERVES
            # compiled; making it on a card that can no longer fit a forward is
            # how a mint's last step became a SIGSEGV and then a crash-loop the
            # hub read as demand (th#1959).
            device = mint_workers.device_of(pipe)
            no_room = adopt_fit.refusal(
                f"advertising the compiled graphs for {bg.spec.name}", device)
            if no_room:
                self._abandon_advertisement(rec, pipe, outcome, no_room)
                continue
            armed_here = False
            # pgw#1262's marker, one question over: binding the guard and
            # flipping the router touch the live compiled objects, so a signal
            # death here is a compile's, not the tenant's.
            with postmortem.compile_inflight(f"advertise:{bg.spec.name}"):
                for target in rec.compile_targets.values():
                    if target.pipeline is not pipe:
                        continue
                    with target.state_lock:
                        target.active_compile_ref = str(outcome.ref)
                        target.active_compile_snapshot_digest = str(
                            outcome.snapshot_digest)
                        target.active_self_mint = True
                    # pgw#686: the mint stamped the pipe's lane; re-advertise so
                    # the target's lane/bucket/contract descriptors match what it
                    # now serves.
                    self._refresh_compile_target(target)
                    if not self._bind_compile_guard(rec, target):
                        with target.state_lock:
                            target.active_compile_ref = ""
                            target.active_compile_snapshot_digest = ""
                        logger.warning(
                            "compile target %s has no runtime guard revocation "
                            "signal; advertising eager", target.incarnation_id)
                        continue
                    armed_here = True
                # pgw#1265: eager-while-compiling is turned on for a pipe that
                # ADVERTISES something. Unconditional, it enabled concurrent
                # routing on a pipe whose every target had just been rolled
                # back to eager for want of a revocation signal.
                if armed_here:
                    hot_swap.enable(pipe)

    def _abandon_advertisement(
        self, rec: "_ClassRecord", pipe: Any, outcome: Any, detail: str,
    ) -> None:
        """pgw#1265: refuse the flip and GIVE THE ARM BACK.

        Invariants 2 and 3 at the wire seam. Nothing is advertised, so no
        capability projection claims a compiled tier; the armed entries are
        de-armed and their runners released, so the residency floor this adopt
        raised comes back down; and the de-arm is sticky for the boot, so the
        next request cannot walk into the same refusal. The worker serves
        eager and stays alive — which is the whole difference between a
        degraded pod and `ComputeProcessDied`.

        The artifact is not condemned and stays PUBLISHED: it was minted and
        parity-gated by this process, and a card that is full at this instant
        says nothing about it.
        """
        for target in rec.compile_targets.values():
            if target.pipeline is not pipe:
                continue
            with target.state_lock:
                target.active_compile_ref = ""
                target.active_compile_snapshot_digest = ""
                target.active_self_mint = False
        for entry in list(aot_serve.entry_states(pipe)):
            try:
                aot_serve.disarm_entry(pipe, entry, adopt_fit.REASON)
            except Exception:  # noqa: BLE001 — the refusal survives its cleanup
                logger.warning(
                    "adopt-fit: de-arming %r at the advertisement failed",
                    entry, exc_info=True)
        flush_memory()
        logger.warning("adopt-fit: %s", detail)
        activity_mod.emit_event(
            "adopt_headroom_refused", detail, phase=adopt_fit.REASON,
            compiled_graph_key=str(getattr(outcome, "ref", "") or ""),
        )

    async def _supervise_mint(
        self, rec: _ClassRecord, bg: "_BackgroundMint", act: Any,
    ) -> None:
        """th#1834 Phase 3 (pgw#1215 step 4): supervise this record's compile
        children directly, then advertise what armed.

        This replaced ``_background_mint_run`` / ``_delegated_mint_run``, which
        were a wrapper and a driver for a MIDDLE PROCESS TIER that no longer
        exists. The parent used to hand one mint child the whole job; that
        child loaded a second weight-free pipeline and drove the compile pool
        itself. The pipeline it loaded is one this record already holds a real,
        resident, SERVING copy of, and the process boundary it bought is now
        bought by a lint fence (``scripts/lint_serving_process_compiles.py``).

        What stays here is exactly what only a serving worker can do: keep
        serving eager and beating while it happens, decide publish on gw#612's
        sibling-coverage rule, and advertise the identity on the live compile
        targets. Everything between is ``mint_supervisor``'s — deliberately,
        because that is the compiled-graph interior and it is the surface the
        ``torch-compiled-graphs`` extraction lifts whole. Residency (which pipe,
        which target, what the wire is told) does not leak into it.

        Raises what ``_background_mint`` handles and nothing else:
        ``_MintAbandoned`` (adopt-on-arm / vacate / shutdown), or a plain
        ``Exception`` (a failed mint). Serving continues in every branch: the
        worker never dies with its mint.
        """

        spec = bg.spec
        # One supervised mint per DISTINCT pending. Pipes whose obligation
        # identity is the same token share one pending and therefore one mint —
        # since pgw#1113 that identity names the SUBJECT (which slot, resolved
        # to which checkpoint), so a shared pending means one thing to compile
        # rather than one family on one card.
        holders: Dict[int, List[int]] = {}
        for pid, pending in bg.pendings.items():
            holders.setdefault(id(pending), []).append(pid)
        if not holders:
            raise RuntimeError("supervised mint has no pending cell to build")

        #: id(pipeline) -> the compiled graphs its OWN pipeline armed.
        #: pgw#1113 deleted the "sharers" fiction that used to fill this for
        #: every pid holding the pending: exactly one pipe is supervised and
        #: exactly one pipe is passed to `adopt_delegated_mint`, so exactly one
        #: pipe ever had those bytes installed on it. Advertising the other
        #: pids' targets as compiled was a wire lie that only
        #: `_bind_compile_guard`'s incidental `False` ("advertising eager")
        #: stopped from reaching the hub.
        finalized: Dict[int, Any] = {}
        #: id(pending) -> what discharged it. The publish-coverage rule
        #: (gw#612) is about the OBLIGATION, not about how many pipes hold it,
        #: so it reads this rather than `finalized`.
        discharged: Dict[int, Any] = {}
        # pgw#999: every classified refusal this run saw, so the terminal
        # RuntimeError names them instead of restating "nothing to advertise".
        declined_reasons: List[str] = []
        for pids in holders.values():
            pending = bg.pendings[pids[0]]
            pipe = bg.pipes[pids[0]]
            result = await mint_supervisor.supervise(
                mint_supervisor.MintTask(
                    pending=pending,
                    pipe=pipe,
                    function=spec.name,
                    modules=bg.modules or _mint_modules(spec),
                    slots=dict(bg.slots),
                    weight_lane=compile_cache.cell_base_execution_lane(pipe),
                    execution_lane=self._served_execution_lane(spec),
                    configs={spec.name: self._effective_config(spec)},
                    device=mint_workers.device_of(pipe),
                    # pgw#1199: this boot ran the endpoint's own handler on the
                    # resident pipeline before any mint was supervised (setup
                    # completes, THEN `bg.task` is created), so the compile
                    # children get pgw#984's guarantee for free and allocate
                    # nothing for it. Empty when a custom `warmup()` stood in
                    # for the synthesized plan and no handler actually ran —
                    # the mint refuses on that, honestly, rather than proving it
                    # at the cost of a checkpoint.
                    handler_proof=handler_proof.provenance(spec.name),
                ),
                act=act, abandon=bg.abandon)
            if result.status == mint_supervisor.ABANDONED:
                raise _MintAbandoned()
            minted = result.minted
            if not result.ok or minted is None:
                logger.warning(
                    "supervised mint for %s produced no adoptable compiled "
                    "graph (%s); that object stays eager",
                    spec.name, result.detail)
                # pgw#815: resolve the obligation instead of dropping it — a
                # `continue` here left the pending with no terminus and no wire
                # trace whenever a SIBLING pending succeeded (the
                # `if not finalized: raise` below never fires then).
                fleet_cells_mod.abandon_self_mint(pending)
                # pgw#999: `phase` carries the CLASSIFIED reason when the
                # compiled graphs were built and then refused arming; it falls
                # back to the call-site token only when there is genuinely no
                # classification (nothing was produced at all).
                activity_mod.emit_event(
                    "self_mint_abort",
                    f"family={pending.family} key={pending.arm_token}: the "
                    f"supervised mint produced no adoptable compiled graph "
                    f"({result.detail or result.status}); this object stays "
                    f"eager and nothing is published",
                    phase=result.reason or "supervised_no_graph",
                )
                declined_reasons.append(result.reason or result.status)
                continue
            # pgw#1113: the ARMED pipe, and only it. `adopt_delegated_mint`
            # installed the compiled graphs on this one pipeline; the other pids
            # holding this pending were never armed with these bytes and must
            # not advertise them.
            finalized[pids[0]] = minted
            discharged[id(pending)] = minted
            if len(pids) > 1:
                activity_mod.emit_event(
                    "self_mint_unarmed_holder",
                    f"family={pending.family} key={pending.arm_token}: "
                    f"{len(pids) - 1} further compile object(s) hold this "
                    f"obligation and were NOT armed with its compiled graphs — "
                    f"one pipe is armed per supervised mint, so they serve "
                    f"eager until their own arm. They are not advertised as "
                    f"compiled (pgw#1113)",
                    phase="unarmed_obligation_holder",
                )
            compile_cache.record_compiled_graph_proven(str(minted.ref))

        if not finalized:
            raise RuntimeError(
                "the supervised mint produced no advertisable compiled graph; "
                "serving stays eager"
                + (f" (refused: {', '.join(sorted(set(declined_reasons)))})"
                   if declined_reasons else ""))

        # Publish per OBLIGATION on gw#612's rule: an artifact ships only when
        # the obligation it was built for was actually discharged — a partial
        # set bricks every adopting boot at the gw#607 per-object proof.
        for pids in holders.values():
            pending = bg.pendings[pids[0]]
            if id(pending) not in discharged:
                fleet_cells_mod.withhold_self_mint_publish(
                    pending,
                    "the supervised mint produced nothing for this obligation")
            else:
                fleet_cells_mod.publish_self_mint(pending)

        self._advertise_compiled_graphs(rec, bg, act, finalized)
        logger.info(
            "supervised mint for %s armed: %d compile object(s) hot-swapped to "
            "compiled — this worker served eager and beat at its normal "
            "cadence for the whole mint (th#1834 Phase 3)",
            spec.name, len(finalized))

    async def _report_cell_selection_bug(
        self,
        spec: EndpointSpec,
        compile_selection: Optional["_CompileArtifactSelection"],
        exc: BaseException,
    ) -> None:
        """th#883 invariant: a SELF-REQUESTED, identity-verified cell failed
        to arm — by construction a bug in the one selection brain. Loud
        event class on the wire (th#1031: no longer fatal to serving — the
        fleet policy already fell through to self-mint; this only makes
        sure the invariant stays wire-visible)."""
        bug_ref = WireRef(
            compile_selection.ref if compile_selection is not None else "")
        bug_digest = (
            compile_selection.snapshot_digest
            if compile_selection is not None else "")
        logger.error("compiled_graph_selection_bug on %s (%s): %s", spec.name, bug_ref, exc)
        await self._send(pb.WorkerMessage(
            model_event=self.store.model_event(
                bug_ref,
                pb.MODEL_STATE_FAILED,
                identity=((bug_digest, 0) if bug_digest else None),
                error=f"compiled_graph_selection_bug: {str(exc)[:300]}",
            )
        ))

    def _boot_adopt(
        self, spec: EndpointSpec, slots: Dict[str, MintSlot],
    ) -> "Tuple[boot_adopt.BootAdoptOutcome, ...]":
        """§4.27 steps 1-3 for one boot, off the event loop.

        ALWAYS an outcome, never ``None`` (pgw#1116). The three gates below
        used to return a bare ``None`` — "this pod cannot even attempt the
        derivation" — which is a true statement that names nothing: no family,
        no gate, no event, and a caller unable to tell it from a pod that asked
        the hub and was told no. Three real pods on 0.103.0 called
        ``/v1/worker/compiled-graphs/resolve`` ZERO times and no artifact anywhere said
        which of these gates did it. Each one now names itself and emits.

        None of them is fatal, and none of them is new behaviour: every non-hit
        outcome still means "boot as this pod booted yesterday".
        """
        cfg = spec.compile_cell()
        family = str(getattr(cfg, "family", "") or "")
        fn = str(spec.name or "")
        # pgw#1107: a registry read, not an evaluation. The pgw#853 thunk that
        # could raise out of here (and, uncaught, failed the whole model setup)
        # is retired; a blocked family carries its refusal as
        # `Compile.blockers` and the mint gate reads it.
        from .api.export_contract import export_declaration

        decl = export_declaration(family)
        if decl is None:
            return (boot_adopt.refused(
                "no_export_declaration",
                f"family {family!r} has no registered export declaration, so "
                f"this boot cannot state the class set a cell key names",
                family=family, function=fn),)
        try:
            declared_hint = len(list(aot_declaration.cell_plans(decl)))
        except Exception as exc:  # noqa: BLE001 — never fatal
            return (boot_adopt.refused(
                "declaration_unreadable",
                f"family {family!r} has a declaration this boot cannot "
                f"enumerate: {type(exc).__name__}: {exc}",
                family=family, function=fn),)
        base_url = str(self.file_base_url or "")
        bearer = str(self.worker_jwt_provider() or "")
        # pgw#1108: the credential lives in the PARENT under the split (pgw#783),
        # which is the only execution model. This executor runs in the compute
        # child, whose `worker_jwt_provider` returns "" BY CONSTRUCTION (it holds
        # no credential — pgw#763 delta 1), so a `not bearer` gate here refused
        # boot-adopt on EVERY real serving pod: derive never ran, resolve never
        # fired, and the pod fell straight through to self-mint — the whole reuse
        # circle stayed open. The seam being up (`broker.active()`) is the child's
        # honest "there is somebody to ask": the resolve is a parent-mediated
        # action (`compiled_graphs.resolve`), so the parent supplies base_url + bearer and
        # ignores what the child passes. Mirrors `fleet_cells.CellPublisher`'s
        # own readiness (base_url AND (local bearer OR broker.active())).
        hub_absent = ""
        if not base_url or (not bearer and not procsplit_broker.active()):
            hub_absent = "nobody to ask: base_url={} bearer={} seam={}".format(
                base_url or "<unset>", "set" if bearer else "<unset>",
                "up" if procsplit_broker.active() else "down")
        # pgw#1127 S2: this used to RETURN here, before the derivation, on the
        # premise that "deriving a key nobody will answer is pure boot latency".
        # The premise is false on exactly the machines §4.28 is about: the
        # derived `ck1` key IS `local_cell_store`'s own address, so an offline
        # box holding the exact cell it needs was being told there was nobody
        # to ask. The gate survives in its honest form — refuse only when BOTH
        # answerers are absent — and `attempt` decides the rest, after the
        # local store has been asked.
        if boot_adopt.no_compiled_graph_source(hub_absent):
            return (boot_adopt.refused(
                "no_compiled_graph_source",
                f"{hub_absent}, and this machine's own cell store is empty",
                family=family, function=fn),)
        work_root = Path(
            self.store._cache_dir or Path.home() / ".cache" / "gen-worker"
        ) / "boot-key" / (spec.name or "endpoint")
        return boot_adopt.attempt(
            function=spec.name,
            modules=_mint_modules(spec),
            cfg=cfg,
            slots=slots,
            declared_hint=declared_hint,
            work_root=work_root,
            # The memo lives beside the cell cache and OUTLIVES one boot on a
            # pod with a volume — which is the whole point (§4.28's
            # compile-once-run-forever promise for cozy-local reads the same
            # memo through the same closure digest).
            memo_dir=Path(self.store._cache_dir) if self.store._cache_dir else None,
            cache_dir=self.store._cache_dir,
            base_url=base_url,
            bearer=bearer,
            hub_absent=hub_absent,
        )

    def _enable_compiled(
        self, pipe: Any, cfg: Any, artifact: Optional[Path],
        delivered: Optional["_CompileArtifactSelection"] = None,
        arm: Optional[_ArmOrder] = None,
        boot_local_key: str = "",
    ) -> "fleet_cells.ArmOutcome":
        """Arm the best available compiled path for a freshly loaded pipeline.

        gw#587: delivered cell first — a th#1031 ``compiled_graph_selection_bug``
        (self-requested cell fails contract_drift) is reported loudly but no
        longer fatal: this falls through to SELF-MINT exactly like an
        ordinary miss. The boot warmup compiles the real serving graphs
        once, serves compiled immediately, and publishes through the hub's
        attested gate so the next worker on this key is store-served. Eager
        fallback and the fail-closed cell wait are gone for reachable mints;
        genuine mint impossibilities keep the old miss policy (plain=eager,
        quantized=typed refusal).

        Returns the fleet ``ArmOutcome``; a ``self_mint`` result is recorded
        into ``active_compile_artifacts`` exactly like a cell this pod
        DISCOVERED and pulled, so the warmup proof runs and the target
        advertises the key STAMPED on the bytes it serves.

        pgw#1032/th#1702: that advertised key is what the hub's dispatch fence
        verifies against its own store. The older "self-attested" spelling —
        ``ActiveCompileRef == KeyRef(family, requested_cell_key)`` — compared a
        stamped key against a COMPUTED one, disjoint spaces since pgw#1010, so
        it could never match; it is retired with the requested key itself.

        pgw#904: with an ``_ArmOrder`` (a Plan dispatch) the fleet POLICY does
        not run at all. The hub already decided: ``aot_cell`` arms exactly the
        named artifact or refuses typed, ``dynamo`` arms JIT intake,
        ``eager_only`` arms nothing.

        pgw#1122: with ONE exception, and it is the exception that keeps a
        refusal from costing a pod. §4.27 boot-adopt builds the same
        ``_ArmOrder`` shape out of a cell this pod resolved by its own derived
        key — the hub ordered nothing — so a typed refusal there drops the
        order and runs the ordinary policy, instead of failing the function and
        leaving the pod to be reaped and replaced."""

        if arm is not None:
            try:
                outcome = fleet_cells.arm_ordered(
                    pipe, cfg, self.store._cache_dir,
                    backend=arm.backend,
                    artifact=artifact,
                    delivered_ref=delivered.ref if delivered else "",
                    delivered_digest=(
                        delivered.snapshot_digest if delivered else ""),
                    expected=arm.expected,
                    publisher_org=arm.publisher_org,
                )
                # pgw#1176: THE ACCRETION LOOP. Every other class this boot
                # resolved arms into the SAME registry, target pool and live
                # wrap. A failure here costs that class and nothing else —
                # the pod is already serving compiled, so degrading one entry
                # to eager is the design's normal state, not a fallback.
                for extra_path, extra_expected, extra_org in arm.extra:
                    try:
                        fleet_cells.arm_ordered(
                            pipe, cfg, self.store._cache_dir,
                            backend=arm.backend, artifact=extra_path,
                            delivered_ref="", delivered_digest="",
                            expected=extra_expected,
                            publisher_org=extra_org,
                        )
                    except Exception as extra_exc:  # noqa: BLE001
                        activity_mod.emit_event(
                            "aot_entry_arm_failed",
                            f"a sibling entry of an armed cell would not arm "
                            f"({type(extra_exc).__name__}: {extra_exc}); that "
                            f"CLASS serves eager and every armed sibling keeps "
                            f"serving compiled",
                            phase=str(getattr(extra_exc, "reason", "")
                                      or "arm_failed"),
                            family=str(getattr(cfg, "family", "") or ""),
                            compiled_graph_key=str(
                                getattr(extra_expected, "compiled_graph_key", "") or ""),
                        )
                return outcome
            except fleet_cells.OrderedArmError as exc:
                # pgw#1122: a HUB-ordered arm stays terminal (pgw#904 — the hub
                # named one exact artifact and a substitute would not be it).
                # A BOOT-ADOPTED one was ordered by nobody: this pod derived the
                # key, asked, and was answered, so a refusal here means what
                # every other boot-adopt refusal means — boot as this pod booted
                # yesterday. Measured cost of not distinguishing them: three
                # pods that resolved and materialized a cell correctly, then
                # reported `worker_function_unavailable reason=compile_cell_
                # failed`, never served, were reaped `state_blocked_idle`, and
                # had replacements bought.
                if arm.adopt is None:
                    raise
                if compile_cache.mandatory_serving(pipe):
                    # A mandatory (w8a8/w4a4) lane serves ONLY from a cell
                    # (pgw#1010), so "boot as yesterday" is not available here
                    # and the refusal is genuinely terminal. Fail closed, named.
                    raise
                boot_adopt.arm_refused(
                    arm.adopt, cause=exc.reason, detail=str(exc))
                # Drop the order and run the ORDINARY fleet policy with no
                # delivered artifact — bit for bit the call this method makes
                # when boot-adopt returns a MISS, which is the boot every pod
                # did before §4.27 existed. The refused cell is not retried: it
                # is not passed back in.
                outcome = self._enable_compiled(
                    pipe, cfg, None, boot_local_key=boot_local_key)
                if outcome.armed or outcome.eager_reason:
                    return outcome
                return dc_replace(
                    outcome,
                    eager_reason=cell_adopt.EagerPhase.ADOPTED_COMPILED_GRAPH_REFUSED)
        return fleet_cells.enable_compiled(
            pipe, cfg, self.store._cache_dir, artifact,
            publisher=self._cell_publisher(),
            delivered_ref=delivered.ref if delivered else "",
            delivered_digest=delivered.snapshot_digest if delivered else "",
            # pgw#1127 S2: the boot's own derived `ck1`, when THIS MACHINE's
            # store answered on it. Empty on the fleet path, where the store is
            # empty and the lookup is one stat.
            boot_local_key=boot_local_key,
        )

    def _arming_enable(
        self, pipe: Any, cfg: Any, cache_dir: Optional[Path],
        artifact: Optional[Path],
        subject: Tuple[graph_facts.SlotSubject, ...] = (),
    ) -> "fleet_cells.ArmOutcome":
        """ArmingScope adapter: a self-loaded pipeline's ``arm_compile()``
        gets the same fleet policy (delivered cell first, self-mint on miss)
        as a worker-loaded slot. ``cache_dir`` comes from the scope, which
        the executor constructed with its own store cache dir.

        pgw#1113: this pipeline was built by the ENDPOINT, out of path-valued
        slots, so nothing here can say which of them it read. The obligation
        therefore names EVERY slot this setup resolved. That over-splits when
        the endpoint used only one of them — which costs one re-mint — and
        the alternative under-splits, which binds a pipeline to a cell nobody
        proved is its computation.
        """
        for sub in subject:
            fleet_cells.stamp_arm_subject(
                pipe, sub.slot, sub.refs, sub.snapshot_digest)
        return fleet_cells.enable_compiled(
            pipe, cfg, cache_dir, artifact,
            publisher=self._cell_publisher(),
        )

    @property
    def teardown_seam(self) -> RecordTeardown:
        """What `models.records`' two mutators need from here.

        Built per call, never cached: `_on_state_change` is reassigned after
        `__init__` (worker.py wires it once Lifecycle exists).
        `abandon_background_mint` is th#1834's ruled seam — abandonment stays
        with the mint supervisor and residency calls it through this."""
        return RecordTeardown(
            records=self._classes.values(),
            residency=self.store.residency,
            abandon_background_mint=self.abandon_background_mint,
            on_state_change=self._on_state_change,
            close_sequence_group=self._close_sequence_group,
            observe_host_ram_progress=self._observe_host_ram_progress,
        )

    # ---- Compile-cache adoption -------------------------------------------

    async def _report_adoptions(
        self,
        inj: "_InjectionResult",
        proof_by_obj: Dict[int, Tuple[int, int, int]],
    ) -> None:
        """Send ONE terminal `ModelEvent` per boot-attached adoption (pgw#923).

        This is the producer the `compile_cache_adopt` measurement lane never
        had. th#1329/th#1352 gave the hub a durable, indexed, percentile-backed
        home for "what did arming this cell cost, and what does it still cost
        at warm time" — and both live stacks held ZERO rows in it, because the
        only worker-side sender was the hub-commanded `ADOPT_COMPILE_CACHE`
        handler and no stack has ever issued that operation. Every adoption
        that actually happens is armed at boot — "boot attach" names WHEN, not
        a hub push; since th#1702 nothing is pushed to a pod at all and the
        cell arrives as a Plan's exact `Arm.artifact` (pgw#904) — and that arm
        reported itself in prose (`aot_adopt`, `duration_ms=0`) on a lane with
        no numbers in it. Two builders, one fact, and only the unmeasured builder
        reached the consumer.

        `operation_id` is empty by construction here: the wire contract already
        specifies empty as "boot-attached cell", so the hub stores these
        without being taught a second spelling.

        Telemetry only. A send failure never changes what this worker serves —
        the cell is already armed or already refused by the time this runs.
        """
        if not inj.adoptions:
            return
        warmup_s = round(max(0, self._boot_warm_ms) / 1000.0, 3)
        for adoption in inj.adoptions:
            if not adoption.ref:
                # pgw#1176: THIS DROP INVERTED AND IS NOW A REPORT.
                #
                # Under ck1 it was sound: one cell, one ref, and no ref meant
                # there was nothing anyone could attribute. Under a resolved
                # KEY SET it swallows the commonest per-entry outcome there
                # is — an entry that MISSED has no artifact ref BY
                # CONSTRUCTION, so a pod resolving 30 of 36 keys would have
                # reported the 30 and silently discarded the six that are the
                # actual news. That is exactly how `compile_cache_adopt` went
                # three readers and zero writers for five days.
                #
                # `ModelEvent` is keyed by ref and genuinely cannot carry
                # this, so the miss goes out on the channel that CAN: the
                # typed activity event, whose family/compiled_graph_key/graph_class
                # fields (proto 18-20) land in the hub's own columns.
                activity_mod.emit_event(
                    "aot_entry_missed",
                    f"this pod derived the key and nothing entitled answered "
                    f"it ({adoption.reason or 'no_cell'}"
                    f"{': ' + adoption.detail if adoption.detail else ''}); "
                    f"the class serves EAGER and is queued to compile",
                    phase=adoption.reason or "no_compiled_graph",
                    family=str(getattr(inj, "family", "") or ""),
                    compiled_graph_key=adoption.compiled_graph_key,
                    graph_class=adoption.entry,
                )
                continue
            calls, hits, misses = proof_by_obj.get(adoption.pipeline_id, (0, 0, 0))
            if adoption.armed:
                event = pb.ModelEvent(
                    ref=adoption.ref,
                    snapshot_digest=adoption.snapshot_digest,
                    # Empty: the wire contract's own name for a boot-attached
                    # cell, so the hub stores these without a second spelling.
                    operation_id="",
                    target_incarnation_id="",
                    state=pb.MODEL_STATE_ADOPTED,
                    duration_ms=adoption.arm_ms,
                    cache_hits=max(0, hits),
                    cache_misses=max(0, misses),
                    warmup_s=warmup_s,
                )
            else:
                event = pb.ModelEvent(
                    ref=adoption.ref,
                    snapshot_digest=adoption.snapshot_digest,
                    operation_id="",
                    target_incarnation_id="",
                    state=pb.MODEL_STATE_FAILED,
                    # The same `adopt_failed:<reason>` grammar the commanded
                    # path uses, so one `kind=compile_cache_adopt` query
                    # returns the whole outcome distribution rather than two
                    # half-populations that have to be unioned by hand.
                    error=f"adopt_failed:{adoption.reason or 'no_cell'}",
                    duration_ms=adoption.arm_ms,
                )
            logger.info(
                "cell adoption %s ref=%s digest=%s arm_ms=%d warmup_s=%.3f "
                "calls=%d hits=%d misses=%d%s",
                "ADOPTED" if adoption.armed else "FAILED",
                adoption.ref, adoption.snapshot_digest, adoption.arm_ms,
                warmup_s, calls, hits, misses,
                "" if adoption.armed else f" reason={adoption.reason}")
            await self._send(pb.WorkerMessage(model_event=event))
        inj.adoptions.clear()

    # ---- job intake --------------------------------------------------------

    async def handle_run_job(self, run: pb.RunJob) -> None:
        """The LEGACY wire head. Dies whole with ``RunJob`` at th#1457's cut:
        it and the ``_legacy_order`` projection it schedules are the only
        frames on the dispatch path that read ``pb.RunJob``."""
        job = await self._admit_dispatch(
            run.request_id, int(run.attempt), run.function_name)
        if job is None:
            return
        job.task = asyncio.create_task(
            self._supervise_job(
                job, functools.partial(self._legacy_order, job, run)),
            name=f"job-{run.request_id}")
        # pgw#674: the serving set may have changed — re-derive what to
        # stage next while this job computes.
        self.preloader.poke()

    async def _admit_dispatch(
        self, request_id: str, attempt: int, function_name: str,
    ) -> Optional[_Job]:
        """Shared admission preamble for both wire heads: retransmit re-ack,
        stale-attempt supersede, serve-goal/drain/function gates. Returns the
        admitted job with ``JobAccepted`` sent, or ``None`` when this
        dispatch was answered (re-acked or refused) here."""
        key = (request_id, attempt)
        existing = self.jobs.get(key)
        if existing is not None and not existing.superseded:
            if not existing.finished:
                await self._send(pb.WorkerMessage(job_accepted=pb.JobAccepted(
                    request_id=request_id, attempt=attempt)))
            return None
        # Same request, different attempt: abort the old attempt silently.
        for (rid, att), job in list(self.jobs.items()):
            if rid == request_id and att != attempt and not job.finished:
                job.superseded = True
                self._intent_transition(
                    job.intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED,
                    pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
                    detail=f"superseded by attempt {attempt}",
                )
                job.cancel_requested = True
                if job.ctx is not None:
                    job.ctx._cancel()
                if job.exec_task is not None:
                    job.exec_task.cancel()
                self._arm_cancel_unwind_watch(job)

        intent_id = self._job_intent(request_id, attempt, function_name)
        if not worker_goals.current().serve_admitted():
            # pgw#930 (§1.17): this pod holds no SERVE goal. It is not a mode
            # check — a pod holding BOTH a serve and a mint goal passes here,
            # which is the whole point of the ruling; only the absence of the
            # serve goal refuses.
            #
            # Kept rather than deleted (pgw#930 proposed deleting it outright
            # as a second copy of the hub's placement decision). It is not a
            # second copy once it reads the goal set: the hub DECLARES the
            # goals and this honours them, so there is one carrier. And the
            # measured failure it prevents is real — a pod that accepted a
            # tenant job would put that tenant's latency behind a multi-hour
            # compile. RETRYABLE, not INVALID: the request is perfectly valid
            # and belongs on a worker holding a serve goal.
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail="no serve goal: this worker was not asked to serve",
            )
            activity_mod.emit_event(
                "serve_goal_absent_dispatch_refused",
                f"request {request_id} attempt {attempt} for "
                f"{function_name!r} reached a worker holding no serve "
                f"goal — the hub placed tenant work on a mint-only pod "
                f"(pgw#930)",
                phase="goal_admission",
            )
            await self._send_result(
                request_id, attempt, pb.JOB_STATUS_RETRYABLE,
                safe_message="worker holds no serve goal",
            )
            return None
        if self.draining:
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail="worker draining",
            )
            await self._send_result(
                request_id, attempt, pb.JOB_STATUS_RETRYABLE,
                safe_message="worker draining",
            )
            return None
        spec = self.specs.get(function_name)
        if spec is None:
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail=f"unknown function {function_name!r}",
            )
            await self._send_result(
                request_id,
                attempt,
                pb.JOB_STATUS_INVALID,
                safe_message=f"unknown function {function_name!r}",
            )
            return None
        if function_name in self.unavailable:
            reason, detail, _ = self.unavailable[function_name]
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail=f"function unavailable: {reason}",
            )
            await self._send_result(
                request_id,
                attempt,
                pb.JOB_STATUS_RETRYABLE,
                safe_message=f"function unavailable: {reason}",
            )
            return None

        job = _Job(
            request_id=request_id,
            attempt=attempt,
            spec=spec,
            intent_id=intent_id,
        )
        self.jobs[key] = job
        self._idle.clear()
        # pgw#677: tenant demand exists NOW — no new background turn is
        # granted (outside the minimum-progress rule) and any preemptible
        # in-flight mint seed is cooperatively cancelled.
        self._bg_quiet.clear()
        self._bg_last_tenant_activity = time.monotonic()
        self._preempt_background_seeds()
        logger.info("job admitted %s attempt=%d", request_id, attempt)
        await self._send(pb.WorkerMessage(job_accepted=pb.JobAccepted(
            request_id=request_id, attempt=attempt)))
        return job

    async def _legacy_order(self, job: _Job, run: pb.RunJob) -> _JobOrder:
        """Project ``pb.RunJob`` into the neutral order. LEGACY head code —
        the last dispatch-path frame that reads the wire message, deleted
        whole with ``RunJob`` at th#1457's cut.

        Note what is NOT projected: ``timeout_ms`` (retired — kill/condemn
        authority is liveness + progress-staleness, never a clock) and any
        coarse lane to expand (the ladder twin is deleted; an instruction the
        endpoint does not itself serve refuses typed in
        ``_refuse_unservable_lane``)."""
        spec = job.spec
        assert spec is not None
        for undeclared in _undeclared_model_slots(spec, run):
            logger.warning(
                "UNDECLARED_MODEL_SLOT function=%s slot=%s request_id=%s: "
                "dispatched model param not declared in @endpoint(models={...}) "
                "— ignored, not loaded", spec.name, undeclared, run.request_id)
        self._refuse_unservable_lane(spec, run.lane)
        group = self._dispatch_group(run)

        slots: Dict[str, dispatch.SlotOrder] = {}
        adapters: Dict[str, Tuple[dispatch.AdapterOrder, ...]] = {}
        for b in run.models:
            if not b.slot:
                continue
            slots[b.slot] = dispatch.SlotOrder(
                ref=str(b.ref or "").strip(),
                inference_defaults=str(b.inference_defaults or ""),
                objective=str(b.objective or ""),
                distilled=bool(b.distilled),
                distilled_status=str(b.distilled_status or ""),
            )
            if b.loras:
                adapters[b.slot] = tuple(
                    dispatch.AdapterOrder(
                        ref=str(o.ref or "").strip(),
                        weight=float(o.weight),
                        inference_defaults=str(o.inference_defaults or ""),
                    )
                    for o in b.loras)

        stamped: Optional[Dict[str, Any]] = None
        if spec.config:
            gen, raw_stamped = extract_job_config(run)
            if raw_stamped is not None:
                declared = {p.name for p in spec.config}
                stamped = {
                    k: v for k, v in raw_stamped.items() if k in declared}
                # Advance the worker store + snapshot file to this dispatch's
                # stamped values, so subprocesses read the latest on invoke.
                self.runtime_config.stamp_function(spec.name, stamped, gen)

        def _config_snapshot(name: str, values: Dict[str, Any]) -> Optional[Any]:
            generation = int(
                run.config_generation or self.runtime_config.generation)
            return self.runtime_config.invocation_snapshot(
                name, values, generation)

        compute = run.compute if run.HasField("compute") else None
        return _JobOrder(
            request_id=run.request_id,
            attempt=int(run.attempt),
            function_name=run.function_name,
            payload=bytes(run.input_payload),
            group=group,
            slots=slots,
            adapters=adapters,
            snapshots=index_snapshots(run.snapshots, run.models),
            input_manifest=manifest_from_run_job(run.input_assets),
            # Positional call through the LIVE method (tests and tooling
            # observe/replace `_validate_required_compile` by name).
            fence=lambda s: self._validate_required_compile(s, run),
            config_snapshot=_config_snapshot,
            org=str(run.org or ""),
            invoker_id=str(run.invoker_id or ""),
            capability_token=str(run.capability_token or ""),
            inline_output=run.output_mode == pb.OUTPUT_MODE_INLINE,
            accelerator=str(compute.accelerator) if compute is not None else "",
            gpu_index=int(compute.gpu_index) if compute is not None else 0,
            lane_report=str(run.lane or ""),
            compile_required=run.HasField("required_compile"),
            stamped_config=stamped,
            arm=None,
        )

    def handle_cancel(self, cancel: pb.CancelJob) -> None:
        job = self.jobs.get((cancel.request_id, cancel.attempt))
        if job is None or job.finished:
            return  # unknown pair or natural result already stands
        # JobAccepted means this exact attempt is cancellable even before its
        # context or handler task exists. Retain the request across every
        # pre-execution await instead of dropping an early CancelJob.
        job.cancel_requested = True
        if job.ctx is not None:
            job.ctx._cancel()  # cooperative: sync handlers poll ctx
        if job.exec_task is not None and job.spec is not None and job.spec.is_async:
            job.exec_task.cancel()  # async handlers are cancelled on the loop
        self._arm_cancel_unwind_watch(job)

    # ---- pgw#687 cancel unwind ---------------------------------------------

    def _arm_cancel_unwind_watch(self, job: _Job) -> None:
        """A cancel is only real once the job reaches a TERMINAL result.

        Nothing else watches that edge: a sync handler that ignores
        ``ctx.cancelled`` keeps its GPU permit and instance gate, and the next
        assignment parks pre-execution forever with no event of any kind. So
        watch cancel -> terminal, and if it never lands, refuse work loudly
        (and ultimately replace the pod) instead of absorbing assignments.
        """
        if job.finished or job.unwind_watch is not None:
            return

        async def _watch() -> None:
            try:
                if await self._await_unwound(job, _CANCEL_UNWIND_GRACE_S):
                    return
                await self._enter_cancel_quarantine(job)
                if await self._await_unwound(job, _CANCEL_UNWIND_RECYCLE_S):
                    self._leave_cancel_quarantine(job)
                    return
                logger.critical(
                    "handler for %s attempt=%d ignored cancel for %.0fs; "
                    "recycling worker process so the pod is replaced",
                    job.request_id, job.attempt,
                    _CANCEL_UNWIND_GRACE_S + _CANCEL_UNWIND_RECYCLE_S,
                )
                self._process_exit(70)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("cancel-unwind watch failed for %s", job.request_id)

        job.unwind_watch = asyncio.create_task(
            _watch(), name=f"unwind-{job.request_id}")

    @staticmethod
    async def _await_unwound(job: _Job, timeout: float) -> bool:
        """True iff the job reached a terminal result within ``timeout``."""
        deadline = time.monotonic() + timeout
        interval = min(0.25, max(timeout / 8.0, 0.01))
        while True:
            if job.finished:
                return True
            if time.monotonic() >= deadline:
                return job.finished
            await asyncio.sleep(interval)

    async def _enter_cancel_quarantine(
        self, job: _Job, *, detail: str = "",
    ) -> None:
        """Fail closed: stop advertising, and refuse work already parked
        behind the wedged job instead of letting it sit eventless.

        pgw#738: ``detail`` names the death-without-cancel face — the same
        wedge reached without anyone cancelling anything."""
        detail = detail or (
            f"cancel of request {job.request_id} attempt {job.attempt} has not "
            f"unwound after {_CANCEL_UNWIND_GRACE_S:.0f}s; the handler still "
            "holds the GPU permit / instance gate"
        )
        logger.critical("CANCEL_UNWIND_STUCK %s", detail)
        self._unwind_stuck[(job.request_id, job.attempt)] = detail
        for name in self.specs:
            if name in self.unavailable:
                continue  # never erase another owner's disable
            self.unavailable[name] = (_CANCEL_UNWIND_REASON, _sanitize(detail), {})
            self._unwind_quarantined.add(name)
        for other in list(self.jobs.values()):
            if other is job or other.finished or other.executing:
                continue
            logger.warning(
                "refusing parked request %s attempt=%d: %s",
                other.request_id, other.attempt, _CANCEL_UNWIND_REASON,
            )
            await self._finish(
                other, pb.JOB_STATUS_RETRYABLE,
                safe_message=f"worker unfit: {_CANCEL_UNWIND_REASON}",
            )
            if other.task is not None:
                other.task.cancel()
        self._on_state_change()

    def _leave_cancel_quarantine(self, job: _Job) -> None:
        """The unwind landed late — re-advertise what we (and only we) took."""
        self._unwind_stuck.pop((job.request_id, job.attempt), None)
        if self._unwind_stuck:
            return  # another wedged cancel still open
        for name in list(self._unwind_quarantined):
            entry = self.unavailable.get(name)
            if entry is not None and entry[0] == _CANCEL_UNWIND_REASON:
                self.unavailable.pop(name, None)
        restored = len(self._unwind_quarantined)
        self._unwind_quarantined.clear()
        logger.warning(
            "cancel of %s attempt=%d unwound late; re-advertising %d function(s)",
            job.request_id, job.attempt, restored,
        )
        self._on_state_change()

    def _quarantine_after_cancel(self, spec: EndpointSpec, job: _Job) -> None:
        """A PRODUCER handler cancelled mid-run leaves its own mutations on the
        live instance (modelopt installs module-level quantizer hooks; a
        trainer swaps in adapter/optimizer state), and the next ``setup()``
        inherits them. Reload clean instead. Inference is excluded on purpose:
        a cancelled forward mutates nothing, and discarding a warm serving
        pipeline on every user cancel would be its own regression.
        """
        if spec.kind == "inference" or not job.executing:
            return
        rec = self._classes.get(spec.instance_key)
        if rec is not None and rec.ready and not rec.stale:
            rec.stale = True
            logger.warning(
                "%s cancelled mid-run for %s; marking the instance stale so the "
                "next dispatch reloads it clean", spec.kind, spec.name,
            )

    async def wait_idle(self, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ---- pgw#677 background-turn gate --------------------------------------
    #
    # Doctrine: tenant requests ALWAYS win the GPU; background work (mint
    # seed units, shape-warm/heal compiles) yields. A background unit runs
    # only inside a granted TURN: single-flight with every other background
    # unit, holding the GPU permit + the owning instance's run gate, so it
    # can never race a tenant forward on shared mutable state (the pgw#676
    # SIGSEGV class) nor contend for the device (the measured 8.6x tenant
    # degradation). Turns are granted when the worker is tenant-idle; under
    # sustained load the minimum-progress rule STEALS one bounded turn per
    # debt window so the mint still finishes (starvation both ways
    # considered). A tenant arriving mid-turn waits at most ONE unit — and
    # preempts idle-granted seed units cooperatively.

    def _bg_admit(
        self,
        kind: str,
        abort_check: Callable[[], None],
        max_wait: Optional[float] = None,
    ) -> bool:
        """Thread-BLOCKING admission: wait until the worker is tenant-quiet
        (plus compile quiescence) or the minimum-progress steal is due.
        Returns ``stole``. ``abort_check`` raises to give up. With
        ``max_wait`` set, raises ``hot_swap.TurnGateBusy`` instead of
        waiting past it — the shape-warm thread re-queues rather than
        head-of-line blocking every other router's jobs."""
        attempt_start = time.monotonic()
        blocked_since = attempt_start
        # The steal clock spans requeue cycles (TurnGateBusy re-queues the
        # job): continuous demand-block is measured from the FIRST refused
        # admission, not per attempt.
        if max_wait is not None:
            with self._bg_state_lock:
                if self._bg_blocked_since is not None:
                    blocked_since = self._bg_blocked_since

        def _admitted() -> None:
            if max_wait is not None:
                with self._bg_state_lock:
                    self._bg_blocked_since = None

        while True:
            abort_check()
            now = time.monotonic()
            if self._bg_quiet.is_set():
                quiet_for = now - self._bg_last_tenant_activity
                if kind != "compile" or quiet_for >= _BG_COMPILE_QUIESCENCE_S:
                    _admitted()
                    return False
                wait_s = max(_BG_COMPILE_QUIESCENCE_S - quiet_for, 0.0)
            else:
                floor = (
                    _BG_COMPILE_STEAL_FLOOR_S if kind == "compile"
                    else _BG_STEAL_FLOOR_S)
                with self._bg_state_lock:
                    due = max(
                        self._bg_steal_debt_until,
                        blocked_since + floor,
                    )
                if now >= due:
                    _admitted()
                    if kind == "compile":
                        # pgw#677 reopen: a stolen compile turn stalls the
                        # next tenant on this instance for one unabortable
                        # multi-minute compile — never silently.
                        activity_mod.emit_event(
                            "bg_turn_steal",
                            f"stole a background compile turn after "
                            f"{now - blocked_since:.0f}s of continuous "
                            "tenant demand; the next tenant on this "
                            "instance waits out one unabortable compile "
                            "(attributed to instance_gate_wait)",
                            phase="compile",
                        )
                    return True
                wait_s = due - now
            if max_wait is not None and now - attempt_start >= max_wait:
                with self._bg_state_lock:
                    if self._bg_blocked_since is None:
                        self._bg_blocked_since = blocked_since

                raise hot_swap.TurnGateBusy(kind)
            if self._bg_quiet.is_set():
                time.sleep(min(wait_s, 0.25))
            else:
                self._bg_quiet.wait(timeout=min(wait_s, 0.25))

    def _bg_charge_debt(self, stole: bool, duration: float) -> None:
        """A turn that ran against (or into) live tenant demand charges its
        cost — the stolen background duty cycle stays bounded at
        1/(1+debt_factor)."""
        if stole or not self._bg_quiet.is_set():
            with self._bg_state_lock:
                self._bg_steal_debt_until = time.monotonic() + max(
                    _BG_STEAL_FLOOR_S, _BG_STEAL_DEBT_FACTOR * duration)

    @staticmethod
    def _bg_mutex_acquire(
        mutex: threading.Lock, abort_check: Callable[[], None],
    ) -> None:
        """Thread-blocking mutex acquire with an abort escape hatch."""
        while not mutex.acquire(timeout=0.5):
            abort_check()

    async def _bg_locked(
        self, mutex: threading.Lock, abort_check: Callable[[], None],
    ) -> None:
        """Loop-side acquire of a threading mutex, cancellation-safe: on
        cancel the worker thread is JOINED and an acquire it completed is
        released, so the mutex can never leak held."""
        work = asyncio.create_task(
            asyncio.to_thread(self._bg_mutex_acquire, mutex, abort_check))
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            try:
                await work
            except BaseException:
                pass
            else:
                mutex.release()
            raise

    @asynccontextmanager
    async def _bg_turn(
        self,
        rec: _ClassRecord,
        kind: str,
        abort: Optional[asyncio.Event] = None,
    ) -> typing.AsyncIterator[bool]:
        """Grant the mint driver one background GPU turn; yields ``stole``
        (True when granted against live tenant demand under the
        minimum-progress rule — stolen turns are not preemptible)."""

        def abort_check() -> None:
            if abort is not None and abort.is_set():
                raise _MintAbandoned()
            if self.draining:
                if abort is not None:
                    raise _MintAbandoned()

                raise hot_swap.TurnGateClosed("worker draining")

        await self._bg_locked(self._bg_unit_mutex, abort_check)
        try:
            stole = await asyncio.to_thread(self._bg_admit, kind, abort_check)
            turn_t0 = time.monotonic()
            # pgw#954: same order as the tenant path — run_lock -> turn_mutex
            # -> permit. A turn must never hold the permit while queued on an
            # instance gate: that is the half of the cycle that wedges a
            # tenant's mid-handler #382 reacquire.
            async with rec.run_lock:
                await self._bg_locked(rec.turn_mutex, abort_check)
                try:
                    # pgw#779: the RECORD's group permit. A background turn on
                    # group 2 must not consume group 0's slot - with a count it
                    # did, so a mint anywhere stalled a tenant everywhere.
                    permit = self._gpu_permit_for_record(rec)
                    await permit.acquire()
                    token = self._permits.take(permit, f"background {kind} turn")
                    try:
                        yield stole
                    finally:
                        self._permits.drop(permit, token)
                        permit.release()
                finally:
                    rec.turn_mutex.release()
                    self._bg_charge_debt(
                        stole, time.monotonic() - turn_t0)
        finally:
            self._bg_unit_mutex.release()

    def _bg_turn_threaded(
        self, rec: _ClassRecord,
    ) -> Callable[[str], typing.ContextManager[None]]:
        """Thread-callable turn factory handed to hot-swap routers: the one
        shape-warm thread blocks here — loop-free by construction — until
        its turn is granted, then owns the instance's modules for exactly
        one compile."""

        @contextmanager
        def turn(kind: str) -> typing.Iterator[None]:
            def abort_check() -> None:
                if self.draining:
                    raise hot_swap.TurnGateClosed("worker draining")

            self._bg_mutex_acquire(self._bg_unit_mutex, abort_check)
            try:
                # Bounded admission: refused turns raise TurnGateBusy and
                # the warm job RE-QUEUES — the one shape-warm thread never
                # head-of-line blocks other routers' jobs behind one
                # instance's demand-blocked compile.
                stole = self._bg_admit(
                    kind, abort_check, max_wait=_BG_THREAD_ADMIT_WAIT_S)
                turn_t0 = time.monotonic()
                self._bg_mutex_acquire(rec.turn_mutex, abort_check)
                try:
                    yield
                finally:
                    rec.turn_mutex.release()
                    self._bg_charge_debt(stole, time.monotonic() - turn_t0)
            finally:
                self._bg_unit_mutex.release()

        return turn

    def _wire_turn_gate(self, rec: _ClassRecord, pipeline: Any) -> None:
        """Hand this pipeline's hot-swap router the background-turn gate so
        every shape-warm/heal compile serializes with — and yields to —
        tenant work (pgw#677). Idempotent; no-op without a router.

        pgw#995: unconditional. ``GEN_WORKER_BG_YIELD`` used to be able to skip
        this, restoring the pre-pgw#677 shape where shape-warm compiles ran
        ungated against tenant forwards. Nothing ever set it."""
        router = hot_swap.router_of(pipeline)
        if router is not None:
            router.set_turn_gate(self._bg_turn_threaded(rec))

    def _preempt_background_seeds(self) -> None:
        """pgw#677: a tenant admission preempts the in-flight PREEMPTIBLE
        mint seed at its next cooperative cancel point; the driver
        re-queues the unit and the tenant takes the gate."""
        for rec in self._classes.values():
            bg = rec.background_mint
            if bg is None:
                continue
            ctx = bg.seed_ctx
            if ctx is not None and not ctx.cancelled:
                ctx._cancel()

    async def abort_all(self, safe_message: str = "worker draining") -> None:
        for job in list(self.jobs.values()):
            if job.finished or job.superseded:
                continue
            if job.ctx is not None:
                job.ctx._cancel()
            if job.exec_task is not None:
                job.exec_task.cancel()
            await self._finish(job, pb.JOB_STATUS_RETRYABLE, safe_message=safe_message)

    # ---- job execution -----------------------------------------------------

    async def _supervise_job(
        self, job: _Job, make_order: Callable[[], Awaitable[_JobOrder]],
    ) -> None:
        """pgw#738 never-silent guarantee: a job task that ends WITHOUT having
        reported terminal state is reaped into one.

        The 62922680 face of this issue was 3h51m of `assigned` on a live
        heartbeat with a dead task — the worker is the only component
        positioned to know its own task died, and it stayed silent. Every
        escape from ``_run_job``'s own handlers lands here, as does a plain
        return that somehow skipped ``_finish``.

        ``make_order`` is the wire head's projection (pgw#904): everything
        from here down reads the neutral ``_JobOrder``, never a wire message.
        """
        escaped: Optional[BaseException] = None
        try:
            await self._run_job(job, make_order)
        except asyncio.CancelledError:
            # Worker shutdown / explicit task cancellation. The stream drop is
            # itself a terminal signal to the hub and the loop is going away,
            # so there is no silence to break here.
            raise
        except BaseException as exc:  # noqa: BLE001 — reporting IS the contract
            escaped = exc
            logger.exception(
                "job task for %s attempt=%d escaped without reporting",
                job.request_id, job.attempt)
        if job.finished:
            return
        detail = (
            f"{type(escaped).__name__}: {_sanitize(str(escaped))}"
            if escaped is not None
            else "the task returned with no result and no exception"
        )
        await self._reap_silent_job(job, detail)

    async def _reap_silent_job(self, job: _Job, detail: str) -> None:
        message = f"job task died without reporting terminal state: {detail}"[:512]
        logger.critical(
            "REAPED SILENT JOB %s attempt=%d: %s",
            job.request_id, job.attempt, message)
        try:
            await self._finish(
                job, pb.JOB_STATUS_RETRYABLE, safe_message=message)
        except Exception:
            logger.exception(
                "failed to report the reaped terminal state for %s",
                job.request_id)
        # pgw#687 quarantine doctrine, DEATH-WITHOUT-CANCEL face: the task is
        # gone but a sync handler thread cannot be killed. If it is still
        # running it still owns the card, so the pod stops advertising and
        # refuses the work parked behind it rather than absorbing it silently.
        exec_task = job.exec_task
        if job.executing and exec_task is not None and not exec_task.done():
            await self._enter_cancel_quarantine(
                job,
                detail=(
                    f"request {job.request_id} attempt {job.attempt} died "
                    f"without reporting and its handler thread is still "
                    f"running: {detail}"
                ),
            )

    async def _run_job(
        self, job: _Job, make_order: Callable[[], Awaitable[_JobOrder]],
    ) -> None:
        spec = job.spec
        assert spec is not None
        # The head's projection runs INSIDE the task so its refusals end the
        # job with a terminal state instead of going quiet (pgw#779 shape).
        try:
            order = await make_order()
        except DispatchGroupUnresolved as exc:
            logger.error("refusing %s: %s", job.request_id, exc)
            await self._finish(
                job, pb.JOB_STATUS_RETRYABLE, safe_message=_sanitize(str(exc)))
            return
        except ExecutionLaneUnavailableError as exc:
            await self._finish(
                job, pb.JOB_STATUS_INVALID, safe_message=_sanitize(str(exc)))
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status, msg = _map_exception(exc)
            await self._finish(job, status, safe_message=msg)
            return
        # pgw#748 phase 1: stamp the execution group BEFORE anything reads
        # residency, admits, loads or sets a device. Contextvars propagate
        # into every coroutine and to_thread hop this job makes, so the whole
        # job — admission, staging, handler, teardown — speaks one group.
        with device_group_scope(order.group):
            await self._run_job_grouped(job, order)

    async def _run_job_grouped(self, job: _Job, order: _JobOrder) -> None:
        spec = job.spec
        assert spec is not None
        spec = job.spec = self._group_effective_spec(
            spec, current_device_group())
        refusal = self._multi_group_handler_refusal(spec)
        if refusal:
            # pgw#778: this is now belt-and-braces — gate_functions withdraws
            # the function, so a dispatch can only arrive from a hub that had
            # not yet seen the withdrawal. RETRYABLE, never INVALID: nothing
            # about the CALLER's input is wrong, and INVALID meant the hub
            # neither re-routed nor charged the worker, so every request came
            # back blaming the caller.
            self.unavailable.setdefault(
                spec.name,
                ("multi_group_async_handler", refusal,
                 {"execution_groups": str(self.topology.execution_groups),
                  "degree": str(self.topology.degree)}))
            await self._finish(job, pb.JOB_STATUS_RETRYABLE, safe_message=refusal)
            return
        self._intent_transition(
            job.intent_id,
            pb.LIFECYCLE_INTENT_STATUS_RUNNING,
            pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
        )
        try:
            payload: Any = msgspec.msgpack.decode(order.payload, type=spec.payload_type)
        except (msgspec.ValidationError, msgspec.DecodeError) as exc:
            await self._finish(job, pb.JOB_STATUS_INVALID, safe_message=_sanitize(str(exc)))
            return
        try:
            # pgw#532: rebind declared Slots to the hub-resolved picks for
            # THIS dispatch (instance-per-pick). The derived spec drives the
            # whole job — pins, setup, adapters, ctx.slots — so every
            # downstream consumer sees the pick, never the code seed.
            spec = job.spec = self._dispatched_spec(spec, order.slots)
        except Exception as exc:
            status, msg = _map_exception(exc)
            await self._finish(job, status, safe_message=msg)
            return
        if spec.cls is not None:
            # Register the derived per-pick spec before fencing so the job is
            # a visible record owner and vacate cannot race the validated
            # incarnation.
            self._class_record(spec)
        try:
            order.fence(spec)
        except Exception as exc:
            status, msg = _map_exception(exc)
            await self._finish(job, status, safe_message=msg)
            return
        routed = list(spec.models)
        # Admission lease over this job's model refs for its WHOLE lifetime
        # (pgw#641 Stage 2, superseding the gw#409 whole-job executing() pin):
        # from admission on, no eviction/demotion path may victim these refs —
        # including refs whose entries do not exist yet (the executing() pin
        # no-op'd on those, leaving a freshly created entry demotable between
        # its track_vram and the execution-time pin) — and bytes for
        # not-yet-loaded refs are RESERVED so concurrent admissions cannot
        # book the same free VRAM and OOM each other mid-load. Lane refs are
        # NOT leased (gw#551): lane dispatch is handler-side, so leasing every
        # declared lane would make the idle sibling un-demotable and the used
        # lane un-promotable on an overcommitted card; the LaneResidencyGate pins
        # exactly the lane it executes, at call time.
        try:
            with self.store.residency.admit(
                typing.cast(
                    Mapping[str, int],
                    self._job_admission_sizes(spec, routed, order.snapshots)),
                # pgw#652: weights are not the whole cost of admitting a
                # request — a concurrent 1024^2 diffusion request also holds
                # GBs of latents/attention workspace. The claim is LEARNED
                # from this function's measured peaks (0 until measured), so
                # no endpoint declares it.
                activation_bytes=self.store.residency.activation_hint(
                    self._activation_key(spec)),
            ):
                await self._run_job_pinned(job, order, payload, routed)
        finally:
            # The whole-job pin is now gone. Only a measured increase that
            # satisfies a remembered requirement produces capacity progress.
            await self._observe_host_ram_progress([])

    async def _run_job_pinned(
        self, job: _Job, order: _JobOrder, payload: Any, routed: List[str]
    ) -> None:
        spec = job.spec
        assert spec is not None
        concurrency_at_start = len(self.in_flight_keys()) - 1

        snapshots = dict(order.snapshots)
        needs_gpu = (
            (order.accelerator == "cuda") if order.accelerator else spec.needs_gpu)
        gpu_index = order.gpu_index

        producer = spec.kind != "inference"
        source_info = _reserved_repo_info(payload, "source") if producer else {}
        destination_info = _reserved_repo_info(payload, "destination") if producer else {}
        # pgw#594/te#70: a second, wholly independent reserved model input
        # (e.g. a text-encoder repo separate from the primary `source` DiT).
        # Absent on every existing payload struct — stays {} and is a no-op.
        text_encoder_info = _reserved_repo_info(payload, "text_encoder") if producer else {}
        # pgw#684/te#121: a second repo to COMPARE against, not to build from —
        # a two-ref quality gate loads its reference from `source` and the arm
        # under test from `candidate`. Absent on every existing payload
        # struct — stays {} and is a no-op.
        candidate_info = _reserved_repo_info(payload, "candidate") if producer else {}
        # pgw#1242/te#185: a previously PUBLISHED checkpoint to CONTINUE from.
        # `ctx.save_checkpoint` already publishes; this is the door back in, so
        # a long training run can survive pod loss instead of restarting from
        # zero. Absent on every existing payload struct — stays {} and is a
        # no-op.
        resume_from_info = _reserved_repo_info(payload, "resume_from") if producer else {}

        # gw#453: arm repo-CAS checkpoint routing for producer jobs. Without
        # kind/destination_repo/job_id the ctx's _repo_job_upload_scope() is
        # None and save_checkpoint silently rides the media route (256 MiB
        # cap) instead of the job-bound checkpoint grant.
        execution_hints: Dict[str, Any] = {}
        if order.inline_output:
            execution_hints["output_format"] = "inline"
        job_id: Optional[str] = None
        # Producer-only ctx state (pgw#526): the reserved source/destination
        # structs and the hf token live on _PublisherMixin.__init__ — a plain
        # inference RequestContext takes none of these kwargs.
        producer_kwargs: Dict[str, Any] = {}
        if producer:
            execution_hints["kind"] = spec.kind
            dest_repo = _producer_destination_repo(payload, destination_info)
            if dest_repo:
                execution_hints["destination_repo"] = dest_repo
            # th#1987: a repo-CAS publish must name the release it attaches to,
            # and the caller states it in the reserved `destination` struct the
            # hub already validates (parseDestinationRelease). Carried beside
            # destination_repo so `ctx.save_checkpoint` has it without reaching
            # back into the payload.
            dest_release = str(destination_info.get("release") or "").strip()
            if dest_release:
                execution_hints["destination_release"] = dest_release
            job_id = _capability_job_id(order.capability_token)
            producer_kwargs = dict(
                source_info=source_info,
                destination_info=destination_info,
                text_encoder_info=text_encoder_info,
                candidate_info=candidate_info,
                resume_from_info=resume_from_info,
                hf_token=getattr(self._settings, "hf_token", "") or "",
            )

        ctx_cls = _CONTEXT_BY_KIND.get(spec.kind, RequestContext)
        ctx = ctx_cls(
            request_id=order.request_id,
            job_id=job_id,
            emitter=self._make_ctx_emitter(job),
            owner=order.org or None,
            invoker_id=order.invoker_id or None,
            file_api_base_url=self.file_base_url or None,
            worker_capability_token=order.capability_token or None,
            models={slot: so.ref for slot, so in order.slots.items()},
            loras={
                slot: tuple(
                    {"ref": a.ref, "weight": float(a.weight) or 1.0}
                    for a in advs
                )
                for slot, advs in order.adapters.items() if advs
            },
            **_resolve_slots_kwargs(spec, order.slots, order.adapters),
            execution_hints=execution_hints,
            **producer_kwargs,
        )
        job.ctx = ctx
        # th#1130 / pgw#652 Phase 0: let ctx.save_image defer its encode +
        # C2PA stamp + upload to the finalize tail, which this method drains
        # AFTER releasing the GPU permit. The handler's RETURN is the
        # terminality signal — no endpoint change, and an N-image loop cannot
        # release the permit early because save_image no longer releases
        # anything. Streaming handlers are excluded: they serialize items
        # MID-handler, so their outputs have no post-handler tail to ride.
        if spec.output_mode != "stream" and not spec.is_async_gen:
            ctx._arm_deferred_outputs()
        if job.cancel_requested:
            ctx._cancel()
        if order.capability_token and self.file_base_url:
            from .capability_renewal import renew_capability_while_running

            job.renew_task = asyncio.create_task(
                renew_capability_while_running(
                    file_base_url=self.file_base_url,
                    request_id=order.request_id,
                    attempt=order.attempt,
                    get_worker_jwt=self.worker_jwt_provider,
                    get_token=lambda: ctx._worker_capability_token or "",
                    set_token=lambda t: setattr(ctx, "_worker_capability_token", t),
                ),
                name=f"cap-renew-{order.request_id}",
            )

        try:
            # th#886 v4: canonical payload keeps the caller's opaque stored
            # refs; RunJob.input_assets is the ordered credential-free
            # manifest. Validate/materialize the whole payload (one resolver
            # POST for private refs) before any source/model acquisition or
            # tenant handler work; manifest drift and caller local_path
            # values fail closed.
            input_fetch_t0 = time.monotonic()
            await _to_thread_complete(
                materialize_input_assets,
                payload,
                order.request_id,
                attempt=order.attempt,
                manifest=order.input_manifest,
                file_base_url=self.file_base_url or "",
                capability_token=order.capability_token or "",
                cancel_check=lambda: ctx.cancelled,
            )
            # th#1111: pre-handler stage (outside runtime_ms).
            ctx._stages.record_pre("input_fetch", time.monotonic() - input_fetch_t0)
            ctx.raise_if_cancelled("canceled")
            if source_info:
                await self._materialize_source(ctx, source_info, snapshots)
            if text_encoder_info:
                await self._materialize_source(
                    ctx, text_encoder_info, snapshots,
                    set_path=ctx._set_text_encoder_path, field_name="text_encoder",
                )
            if candidate_info:
                await self._materialize_source(
                    ctx, candidate_info, snapshots,
                    set_path=ctx._set_candidate_path, field_name="candidate",
                )
            if resume_from_info:
                await self._materialize_source(
                    ctx, resume_from_info, snapshots,
                    set_path=ctx._set_resume_from_path, field_name="resume_from",
                )
            if producer:
                await self._materialize_datasets(ctx, payload)
            setup_intent = ""
            if spec.cls is not None:
                setup_intent = self._setup_intent(spec)
                self._intent_transition(
                    job.intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_WAITING,
                    pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
                    reason=pb.LIFECYCLE_WAIT_REASON_SINGLE_FLIGHT_OWNER,
                    blocker_intent_id=setup_intent,
                    detail=f"waiting for function {spec.name}",
                )
            instance = await self.ensure_setup(
                spec, snapshots, promote_slots=routed, arm=order.arm)
            if setup_intent:
                self._intent_transition(
                    job.intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                    pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                )
            # th#913/gw#596: the concrete lane actually serving this job.
            # th#1050: ctx.lane exposes the same post-degrade truth to the
            # handler (declared-lane endpoints branch on it).
            job.lane_report = order.lane_report
            job.compile_required = order.compile_required
            job.execution_lane = self._served_execution_lane(
                spec, instructed=job.lane_report)
            # pgw#789: the shape coordinate, taken from the EXECUTED payload
            # with endpoint defaults applied. runtime_terms carries these only
            # when the endpoint declares a runtime formula (and the hub drops
            # that map after scaling reads it), so a latency comparison had no
            # shape axis at all for most endpoints.
            job.shape = serving_mode_mod.shape_of(
                payload, self._effective_config(spec))
            ctx._set_execution_lane(job.execution_lane)
            # th#1087: effective declared-config values for this dispatch.
            effective_config = self._effective_config(
                spec, stamped=order.stamped_config)
            invocation_snapshot = None
            if spec.config:
                # Head-owned snapshotting (pgw#904): the legacy head encodes
                # values + generation; the Plan head answers with the spec's
                # own canonical ConfigSnapshot bytes (§4.16 — values, never a
                # generation pointer).
                invocation_snapshot = order.config_snapshot(
                    spec.name, effective_config)
            ctx._set_config(
                effective_config,
                snapshot=invocation_snapshot,
            )
            kwargs = await self._handler_kwargs(spec, snapshots)
            adapters = await self._prepare_adapters(order.adapters, spec, snapshots)
            ctx.raise_if_cancelled("canceled")
        except (asyncio.CancelledError, CanceledError):
            await self._finish(job, pb.JOB_STATUS_CANCELED, safe_message="canceled")
            return
        except Exception as exc:
            if isinstance(exc, HardwareUnmetError) and not isinstance(exc, InsufficientDiskError):
                # Self-disable the function on this worker; lifecycle emits
                # FnUnavailable and drops it from available_functions.
                # (Disk pressure is transient — GC frees space — so it only
                # fails the job RETRYABLE, never disables the function.)
                axes = exc.axes() if hasattr(exc, "axes") else {}
                self.unavailable[spec.name] = (
                    getattr(exc, "reason", "hardware_unmet"), _sanitize(str(exc)),
                    {str(k): str(v) for k, v in (axes or {}).items()},
                )
                self._on_state_change()
            status, msg = _map_exception(exc)
            logger.exception("setup/injection failed for %s", spec.name)
            await self._finish(job, status, safe_message=msg)
            return

        queue_ms = int((time.monotonic() - job.admitted_at) * 1000)
        lease: Optional[_GpuSlotLease] = None
        started = time.monotonic()
        alloc_at_start = 0
        try:
            # Pin-while-executing: the models (and adapter snapshots) this job
            # uses are not eviction candidates for its duration. Lane refs
            # excluded (gw#551): the LaneResidencyGate pins the one lane the handler
            # actually calls; pinning all of them here would deadlock the
            # gate's promote against its own job's pins.
            exec_refs = self._job_pin_refs(spec, routed)
            adapter_refs = [a.ref for group in adapters.values() for a in group]
            async with AsyncExitStack() as run_gate:
                # pgw#954 LOCK ORDER, worker-wide: instance gate -> GPU permit,
                # so no permit holder ever waits on a gate and the #382 lease's
                # mid-handler reacquire always finds the permit reachable. The
                # inverse order killed real jobs (pgw#738: 62922680 + d0cbf910,
                # one H100, 3h51m silent). Conforming acquirers: `_bg_turn`,
                # `_bg_turn_threaded` (gate only), `_exclusive_gpu` (permits
                # under the load-lock family, never an instance gate).
                #
                # pgw#647: SINGLE-FLIGHT per live instance — adapter attach and
                # the handler mutate the instance's materialized graph, so two
                # concurrent requests on one instance corrupt each other. Jobs
                # on DIFFERENT instances still run concurrently;
                # ``reentrant=True`` opts out.
                if spec.cls is not None and not spec.reentrant:
                    gate_t0 = time.monotonic()
                    rec_gate = self._classes[spec.instance_key]
                    await run_gate.enter_async_context(rec_gate.run_lock)
                    # pgw#677: exclude the shape-warm thread's compile from
                    # this request's whole mutation+forward window — the
                    # ungated overlap was the pgw#676 SIGSEGV race and the
                    # measured 8.6x mint-window degradation. Bounded: at
                    # most one in-flight compile.
                    await self._bg_locked(
                        rec_gate.turn_mutex, lambda: None)
                    run_gate.callback(rec_gate.turn_mutex.release)
                    gate_wait = time.monotonic() - gate_t0
                    if gate_wait >= 0.001:
                        # pgw#677: time queued behind the instance gate
                        # (typically a background mint/compile turn) is NOT
                        # this request's compute — its own pre-handler stage,
                        # so runtime_ms stops billing mint contention to the
                        # tenant (measured: 16.9s reported for 1.95s of real
                        # work). Now measured while holding no permit, so the
                        # card stays exploitable by other instances for it.
                        ctx._stages.record_pre(
                            "instance_gate_wait", gate_wait)
                if needs_gpu:
                    permit_t0 = time.monotonic()
                    # pgw#779: THIS job's group permit, not one of G
                    # interchangeable tickets. The group was stamped from the
                    # dispatch before anything ran, so a permit and a card are
                    # the same fact.
                    gpu_permit = self._gpu_permit_for_group(current_device_group())
                    await self._intent_await(
                        job.intent_id,
                        gpu_permit.acquire(),
                        operation=f"GPU permit for request {order.request_id}",
                        status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                        stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_GPU_SLOT,
                        reason=pb.LIFECYCLE_WAIT_REASON_GPU_SLOT,
                    )
                    permit_token = self._permits.take(
                        gpu_permit, f"request {order.request_id}")
                    # th#1111: the permit wait was in NO metric — it precedes
                    # the handler window, so runtime_ms never saw it.
                    ctx._stages.record_pre(
                        "gpu_permit_wait", time.monotonic() - permit_t0)
                    # pgw#1154: and the INVERSE wait — the card idle, waiting
                    # for a request — was in no metric either. Emitted only
                    # when a previous holder has actually released on this
                    # worker, so the number always means "gap after the job
                    # before me" and never "time since boot".
                    idle_before = self._permits.consume_idle(gpu_permit)
                    if idle_before is not None:
                        # `record_pre` drops non-positive values, and a ZERO
                        # bubble is the target state — the one reading that
                        # must never be silently indistinguishable from "this
                        # worker has no meter". Floored so the key is always
                        # emitted once a gap has actually been observed; it
                        # renders as 0 ms either way.
                        ctx._stages.record_pre(
                            "gpu_idle_before", max(idle_before, 1e-9))
                    self._loop = asyncio.get_running_loop()
                    lease = _GpuSlotLease(
                        gpu_permit, self._loop, self._permits,
                        f"request {order.request_id}", permit_token)
                    ctx._gpu_slot_lease = lease
                    # pgw#954: gate-holding parents may now yield. What still
                    # cannot is a job carrying per-request adapters — a
                    # follower on the shared pipeline would deactivate or
                    # replace this request's adapter state mid-handler, and no
                    # lock orders that away.
                    ctx._child_call_slot_yieldable = not adapters
                    # gw#516: the handler thread reports the terminal
                    # decode->finalize slot release so the hub sees the job as
                    # "finalizing" while its encode/upload tail runs slotless.
                    ctx._on_finalize_release = lambda: self._enter_finalize(job)
                    if job.ctx.cancelled:
                        raise CanceledError("canceled")
                    # pgw#513: reset the CUDA peak-allocator watermark now that
                    # this job exclusively owns gpu_index (jobs serialize under
                    # _gpu_semaphore) — peak_vram_bytes then measures THIS
                    # job's peak, not the process-lifetime high-water mark.
                    if torch is not None and cuda_ready():
                        try:
                            torch.cuda.reset_peak_memory_stats(gpu_index)
                        except Exception:
                            pass
                        # pgw#652: the baseline the peak is measured AGAINST.
                        # peak - baseline is this request's transient
                        # (activation) footprint, as opposed to the resident
                        # weights already allocated when it took the GPU.
                        alloc_at_start = cuda_allocated_bytes()
                # Last execution fence: no adapter mutation or tenant handler
                # has run yet. The repeated check catches a replacement between
                # scheduler assignment/intake and this GPU turn.
                order.fence(spec)
                # pgw#687: past here this job owns the permit/gate — it is no
                # longer refusable-in-place by the cancel-unwind quarantine.
                job.executing = True
                self._intent_transition(
                    job.intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                    pb.LIFECYCLE_INTENT_STAGE_READY,
                    detail="executing",
                )
                # The compute clock starts once BOTH admissions are paid; each
                # wait is its own `record_pre` stage above.
                started = time.monotonic()
                with self.store.residency.executing(*exec_refs, *adapter_refs):
                    active: List[Tuple[WireRef, Any]] = []
                    try:
                        for slot, prepared in adapters.items():
                            pipe = self._adapter_target(spec, slot)
                            ref = wire_ref(spec.models[slot])
                            await asyncio.to_thread(
                                self._adapters.activate, ref, pipe, prepared, order.request_id
                            )
                            active.append((ref, pipe))
                        # Explicit activation (gw#399): a slot this request
                        # names no adapters for must run bare even if a
                        # previous request's teardown failed and left adapters
                        # enabled.
                        for slot in spec.models:
                            if slot in adapters:
                                continue
                            ref = wire_ref(spec.models[slot])
                            if self._adapters.needs_deactivation(ref):
                                # pgw#678: the PIPELINE, not the lane handle.
                                pipe = self._slot_pipeline(spec, slot)
                                if pipe is not None:
                                    await asyncio.to_thread(
                                        self._adapters.deactivate, ref, pipe, order.request_id
                                    )
                        ctx.raise_if_cancelled("canceled")
                        # pgw#676: name the execution before the GPU touches
                        # it — a signal death mid-handler leaves this marker
                        # for the supervisor's post-mortem attribution.
                        from . import postmortem as postmortem_mod

                        inflight_token = postmortem_mod.note_inflight(
                            "request", spec.name,
                            request_id=str(order.request_id or ""))
                        try:
                            from . import compile_cache as compile_cache_mod

                            # pgw#680: tenant execution is THE serve window —
                            # compiled lanes run fail-on-recompile inside it
                            # (guard miss => eager + heal, never an inline
                            # compile). Warm/mint/adopt paths go through
                            # _invoke_warmup and never enter this window.
                            with compile_cache_mod.tenant_serve_window():
                                output = await self._execute(
                                    job, spec, instance, ctx, payload, kwargs,
                                    gpu_index=gpu_index)
                        except BaseException as exc:
                            # pgw#737: a tenant OOM while this worker was
                            # minting is the mint's fault, and it is fixable
                            # HERE — evict the capture, free the card, and
                            # re-run on a clean allocator. The request then
                            # SUCCEEDS instead of returning RETRYABLE for the
                            # hub to re-dispatch onto an identically loaded
                            # worker (th#1228: 5 attempts and a second H100
                            # bought for one deterministic OOM).
                            if await self._evict_mint_for_oom(spec, ctx, exc):
                                try:
                                    with compile_cache_mod.tenant_serve_window():
                                        output = await self._execute(
                                            job, spec, instance, ctx, payload,
                                            kwargs, gpu_index=gpu_index)
                                except BaseException as retry_exc:
                                    await self._quarantine_for_oom(
                                        spec, ctx, retry_exc)
                                    raise
                            else:
                                # A mid-inference CUDA OOM learns a per-ref
                                # floor, but the live object is quarantined.
                                # The hub retries only after ensure_setup
                                # reloads it cleanly at that rung.
                                await self._quarantine_for_oom(spec, ctx, exc)
                                raise
                        finally:
                            # Python-visible exits are not native crashes —
                            # only a signal death leaves the marker behind.
                            postmortem_mod.clear_inflight(inflight_token)
                    finally:
                        # Guaranteed-inactive on every exit (OK / cancel /
                        # deadline / handler error); attachments stay resident.
                        for ref, pipe in active:
                            await asyncio.to_thread(
                                self._adapters.deactivate, ref, pipe, order.request_id
                            )
            # The peak is read BEFORE the permit is released: the next job
            # resets the CUDA peak-allocator watermark when it takes the GPU,
            # and the finalize tail below runs concurrently with it.
            peak_vram = self._peak_vram_bytes(gpu_index)
            # Handler GPU work is done — free the slot before the deferred
            # encode/upload tail, the result-blob upload and the result send,
            # so the next job's compute starts now.
            overlapped = False
            released_at: Optional[float] = None
            if lease is not None:
                overlapped = not lease.yield_slot()
                released_at = lease.released_at
            self._intent_transition(
                job.intent_id,
                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
            )
            # th#1130: THE tail. Slotless by construction (the permit is gone
            # above), on a thread so the event loop keeps dispatching. Inside
            # the try, so a failing encode fails the request cleanly instead
            # of reporting OK with a hollow asset.
            if ctx._deferred.pending():
                from .video_encode import finalize_permit

                def _drain() -> int:
                    # gw#516 back-pressure: bound how many slotless CPU
                    # finalizes stack up, same permit the video path takes.
                    with finalize_permit():
                        return ctx._drain_deferred_outputs()

                drained = await asyncio.to_thread(_drain)
                logger.info(
                    "finalize tail: %d deferred output(s) encoded+uploaded "
                    "slotless for request %s", drained, order.request_id)
            handler_done = time.monotonic()
            # th#1111: the stage map's window must cover the tail it now
            # contains, so image_encode/credential_stamp/upload land in
            # total.tail and the map still closes against runtime_ms.
            ctx._stages.handler_close()
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index,
                                    output=output, execution_lane=job.execution_lane,
                                    runtime_terms=_runtime_term_values(spec, payload, ctx),
                                    peak_vram_bytes=peak_vram)
            # pgw#652: the request just told us what a concurrent one costs.
            # Only a completed run is evidence — a cancelled or OOM-killed job
            # never reached its real peak.
            if metrics.peak_vram_bytes > 0:
                self.store.residency.record_activation(
                    self._activation_key(spec),
                    metrics.peak_vram_bytes - alloc_at_start,
                )
            if lease is not None:
                if released_at is not None:
                    # gw#516 typed split of runtime_ms: how long the GPU slot
                    # was actually held vs the slotless finalize tail.
                    metrics.slot_held_ms = max(
                        0, int((released_at - started) * 1000))
                    metrics.finalize_wall_ms = max(
                        0, int((handler_done - released_at) * 1000))
                if released_at is not None and handler_done > released_at:
                    # The encode/upload tail ran slotless, overlapping the next
                    # request. `handoff` says who ended the GPU phase: the
                    # HANDLER at the decode->finalize signal (gw#476/gw#516
                    # write_video/write_image), or the EXECUTOR at handler
                    # return with a deferred tail behind it (th#1130).
                    logger.info(
                        "FINALIZE_OVERLAP fn=%s request=%s handoff=%s "
                        "slot_held_ms=%d handler_wall_ms=%d overlap_ms=%d",
                        spec.name, order.request_id,
                        "handler" if overlapped else "executor",
                        int((released_at - started) * 1000),
                        int((handler_done - started) * 1000),
                        int((handler_done - released_at) * 1000),
                    )
            if spec.output_mode == "stream":
                # gw#475: live deltas are droppable by contract (in-memory
                # ProgressHub only) — the terminal JobResult carries the
                # accumulated StreamResult so completed requests stay
                # retrievable after the live stream ends.
                inline: Optional[bytes] = None
                blob_ref: Optional[str] = None
                if output is not None:
                    inline, blob_ref = await self._serialize_output(
                        ctx, order.request_id, output)
                await self._finish(job, pb.JOB_STATUS_OK, inline=inline, blob_ref=blob_ref,
                                   metrics=metrics)
            else:
                inline, blob_ref = await self._serialize_output(
                    ctx, order.request_id, output)
                await self._finish(job, pb.JOB_STATUS_OK, inline=inline, blob_ref=blob_ref,
                                   metrics=metrics)
        except _ExecutionStalled as exc:
            # The process confessed a stall (liveness + progress-staleness,
            # never a clock). RETRYABLE: nothing about the caller's input is
            # wrong, and the hub re-routes onto a worker that is advancing.
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index,
                                    execution_lane=job.execution_lane)
            await self._finish(job, pb.JOB_STATUS_RETRYABLE,
                               safe_message=_sanitize(str(exc)), metrics=metrics)
        except BaseException as exc:

            if isinstance(exc, CompiledExecutionLaneUnavailableError):
                # pgw#672: a compiled lane failing at call time fails THIS
                # request, never the function — the guard wrapper has already
                # degraded the object to explicit eager and revoked the
                # compiled identity (tier flips on the wire). Disabling the
                # function here was half of the quarantine->disable->die loop.
                logger.error(
                    "compiled lane unavailable during %s; request failed, "
                    "function continues at eager tier: %s",
                    spec.name, exc,
                )
                self._on_state_change()
            status, msg = _map_exception(exc)
            if status == pb.JOB_STATUS_CANCELED:
                self._quarantine_after_cancel(spec, job)
            if status == pb.JOB_STATUS_FATAL:
                logger.exception("handler %s failed", spec.name)
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index,
                                    execution_lane=job.execution_lane)
            await self._finish(job, status, safe_message=msg, metrics=metrics)
        finally:
            if lease is not None:
                lease.yield_slot()
            # gw#516: result shipped (any terminal path) — the job leaves the
            # finalizing set the hub gates drain/retire on.
            self._exit_finalize(job)
            self._maybe_idle()

    async def _materialize_source(
        self,
        ctx: Any,
        info: Dict[str, Any],
        snapshots: Dict[WireRef, pb.Snapshot],
        *,
        set_path: Optional[Callable[[str], None]] = None,
        field_name: str = "source",
    ) -> None:
        """Reserved repo-field contract (#376, generalized pgw#594):
        materialize a reserved ``SourceRepo``-shaped payload field (default
        ``payload.source``, also used for ``payload.text_encoder``) locally
        before the handler runs. Same ModelStore path as model bindings —
        identical retry/classification and ModelEvent emission."""
        raw = str(info.get("ref") or "").strip()
        if not raw:
            raise ValidationError(f"payload.{field_name}.ref must be a non-empty repo ref")
        # pgw#1217: NORMALIZE WHERE IT ENTERS. This ref is client-supplied and
        # is then used as both the `snapshots` lookup key (that map is keyed in
        # normal form) and the residency key. Taken verbatim, a non-normal
        # spelling misses its snapshot and mints a SECOND residency identity for
        # one model — the th#736 mechanic `binding.rebind_pick` warns about.
        try:
            ref = normalize_model_ref(raw)
        except ValueError as exc:
            raise ValidationError(
                f"payload.{field_name}.ref {raw!r} is not a valid repo ref: {exc}"
            ) from exc
        path = await self.store.ensure_local(ref, snapshots.get(ref))
        (set_path or ctx._set_source_path)(str(path))

    async def _materialize_datasets(self, ctx: Any, payload: Any) -> None:
        """Reserved-datasets contract (gw#425): materialize every
        ``payload.datasets`` entry (DatasetRef) into a local dataset snapshot
        before the handler runs. Paths land in ``ctx.dataset_paths``."""
        datasets = getattr(payload, "datasets", None)
        if not datasets:
            return
        resolve = getattr(ctx, "resolve_dataset", None)
        if not callable(resolve):
            raise ValidationError(
                "payload.datasets requires a producer-kind endpoint "
                "(conversion/dataset/training)"
            )
        for entry in datasets:
            ref = str(getattr(entry, "ref", "") or "").strip()
            if not ref:
                raise ValidationError("payload.datasets entries need a non-empty ref")
            await asyncio.to_thread(resolve, ref)

    async def _handler_kwargs(
        self, spec: EndpointSpec, snapshots: Dict[WireRef, pb.Snapshot]
    ) -> Dict[str, Any]:
        """Per-call model injection: handler parameters (after ctx, payload)
        whose names match model slots receive the local snapshot path."""
        try:
            sig = typing.get_type_hints(spec.method)
        except Exception:
            sig = {}

        params = [
            p.name for p in _inspect.signature(spec.method).parameters.values()
            if p.name != "self"
        ][2:]
        setup_slots = set(self._setup_slots(spec))
        kwargs: Dict[str, Any] = {}
        for name in params:
            binding = spec.models.get(name)
            if binding is None or name in setup_slots:
                continue
            ref = wire_ref(binding)
            path = await self.store.ensure_local(ref, snapshots.get(ref), binding=binding)
            kwargs[name] = Path(path) if sig.get(name) is Path else str(path)
        return kwargs

    async def _prepare_adapters(
        self,
        adapters: Mapping[str, Tuple[dispatch.AdapterOrder, ...]],
        spec: EndpointSpec,
        snapshots: Dict[WireRef, pb.Snapshot],
    ) -> Dict[str, List[lora_util.PreparedAdapter]]:
        """Materialize + parse the job's per-slot LoRA overlays (gw#393).

        Downloads ride the normal ensure_local snapshot path (disk GC,
        ref-index, ModelEvents — so the hub learns adapter download bandwidth
        like any ref); parsed state dicts hit the digest-keyed RAM LRU.
        GPU-free: application happens later, under the job's GPU slot."""
        overlays = [(slot, list(advs)) for slot, advs in adapters.items() if advs]
        if not overlays:
            return {}
        total = sum(len(loras) for _, loras in overlays)
        if total > lora_util.MAX_LORAS_PER_REQUEST:
            raise ValidationError(
                f"too many lora adapters: {total} "
                f"(max {lora_util.MAX_LORAS_PER_REQUEST} per request)"
            )
        out: Dict[str, List[lora_util.PreparedAdapter]] = {}
        for slot, loras in overlays:
            if slot not in spec.models:
                raise ValidationError(f"lora overlay names unknown model slot {slot!r}")
            prepared: List[lora_util.PreparedAdapter] = []
            for overlay in loras:
                raw = str(overlay.ref or "").strip()
                if not raw:
                    raise ValidationError(f"lora overlay on slot {slot!r} has an empty ref")
                # pgw#1217, same boundary as `_materialize_source`: gw#491 made
                # one adapter mint one cache identity for the DIGEST spelling
                # (see below); this does it for the REF spelling, which it left
                # open.
                try:
                    ref = normalize_model_ref(raw)
                except ValueError as exc:
                    raise ValidationError(
                        f"lora overlay on slot {slot!r} has an invalid ref "
                        f"{raw!r}: {exc}"
                    ) from exc
                weight = lora_util.validate_overlay_weight(overlay.weight, ref=ref)
                t0 = time.monotonic()
                path = await self.store.ensure_local(ref, snapshots.get(ref))
                ensure_ms = int((time.monotonic() - t0) * 1000)
                snap = snapshots.get(ref)
                # gw#491: normalize to the bare-hex spelling — snap.digest may
                # carry an algo prefix ("sha256:<hex>") while path.name is the
                # bare hex; one adapter must never mint two cache identities.
                digest = (snap.digest if snap is not None else "") or path.name
                digest = digest.split(":", 1)[-1].strip().lower()
                cache_key = f"{ref}@{digest}"
                t1 = time.monotonic()
                state_dict = self._adapter_cache.get(cache_key)
                from_cache = state_dict is not None
                if state_dict is None:
                    file = lora_util.find_adapter_file(path, ref=ref)
                    state_dict = await asyncio.to_thread(
                        lora_util.load_adapter_state_dict, file, ref=ref
                    )
                    self._adapter_cache.put(cache_key, state_dict)
                parse_ms = int((time.monotonic() - t1) * 1000)
                prepared.append(lora_util.PreparedAdapter(
                    slot=slot, ref=ref, cache_key=cache_key,
                    name=lora_util.adapter_name(cache_key),
                    weight=weight, state_dict=state_dict,
                    from_cache=from_cache, ensure_ms=ensure_ms, parse_ms=parse_ms,
                ))
            out[slot] = prepared
        return out

    def _slot_pipeline(self, spec: EndpointSpec, slot: str) -> Any:
        """The worker-CONSTRUCTED pipeline object for ``slot``, or None.

        pgw#678: this is NOT ``residency.obj(ref)``. A shared-component lane
        books its EXCLUSIVE module set (an ``nn.ModuleDict``) as the residency
        entry so LRU demote/promote moves only lane-owned weights — and
        ``exclusive`` is non-empty exactly when some component does NOT ride
        the shared cache, which a th#980 ``components.*`` deploy override
        guarantees (the overridden component's bytes differ from the base's,
        so it is popped out of the share plan). The registry handle was then
        handed to ``LoraRegistry.activate`` as if it were the pipeline:
        ``branch_targets`` finds no denoiser on a ModuleDict, ``_split_adapters``
        never runs, and the residue meets ``isinstance(pipe,
        LoraCapablePipeline) is False`` -> "model slot does not support LoRA
        adapters" for every request (0/6 live on the sdxl turbo picks). The
        record keeps the pipeline identity separately; residency keeps the
        movement handle. Both facts are true — they are just not one object.
        """
        rec = self._classes.get(spec.instance_key)
        pipe = (rec.slot_pipelines.get(slot) if rec is not None else None)
        if pipe is not None:
            return pipe
        # Tenant-loaded slots have no worker-constructed pipeline; a
        # monolithic worker-loaded slot's residency entry IS the pipeline.
        return self.store.residency.obj(wire_ref(spec.models[slot]))

    def _adapter_target(self, spec: EndpointSpec, slot: str) -> Any:
        """The worker-managed pipeline object adapters for ``slot`` apply to."""
        pipe = self._slot_pipeline(spec, slot)
        if pipe is None:
            raise ValidationError(
                f"model slot {slot!r} has no worker-managed pipeline; "
                "lora overlays require a pipeline-injected setup slot"
            )
        return pipe

    async def _evict_mint_for_oom(
        self, spec: EndpointSpec, ctx: RequestContext, exc: BaseException,
    ) -> bool:
        """pgw#737: the survivable abort, from the tenant's side.

        A background self-mint is the ONE co-resident consumer this worker
        put on the card itself, and it is evictable. When a tenant request
        OOMs with a mint in flight, the honest recovery is not "tell the hub
        to try another worker" (identical load, deterministic OOM, one more
        pod bought) — it is: stop the mint, unwrap its targets, drop its
        branch buffers, empty the allocator, and re-run this request on the
        clean card. True = evicted, the caller re-runs; the mint stays gone
        for this process (its pre-budget declines the same card anyway).

        Deliberately narrow: only a CUDA OOM, only inference, only while
        nothing has been emitted or deferred — a replay must not duplicate
        output. Everything else keeps the quarantine + RETRYABLE path.
        """
        if not is_cuda_oom(exc) or getattr(ctx, "cancelled", False):
            return False
        if spec.kind != "inference" or spec.output_mode == "stream":
            return False
        if ctx._deferred.pending():
            return False
        rec = self._classes.get(spec.instance_key)
        if rec is None or rec.background_mint is None:
            return False
        logger.warning(
            "tenant request OOMed with a self-mint in flight for %s; "
            "evicting the mint and re-running on a clean allocator "
            "(pgw#737)", spec.name)
        await self.abandon_background_mint(
            rec, reason="tenant OOM — the mint loses, the request wins",
            code="tenant_oom", free_targets=True)
        activity_mod.emit_event(
            "self_mint_skipped",
            f"self-mint for {spec.name} evicted mid-flight: a tenant request "
            f"OOMed against its capture ({type(exc).__name__}); the mint is "
            "abandoned, the card freed and the request re-run eager on this "
            "same worker",
            phase="tenant_oom",
        )
        ctx.log(
            "DEGRADED_MODE=engaged fn=" + spec.name + ": a background "
            "compile capture was evicted after a CUDA OOM; this request is "
            "being re-run eager on the freed card.",
            level="warning",
        )
        self._on_state_change()
        return True

    async def _quarantine_for_oom(
        self, spec: EndpointSpec, ctx: RequestContext, exc: BaseException,
    ) -> None:
        """Quarantine an OOM'd instance and learn its next placement rung.

        Diffusers offload hooks are setup-time state. Attaching them to a
        fully resident pipeline after a mid-denoise OOM can leave CPU/CUDA
        tensors mixed while Residency still advertises VRAM. Do not reuse or
        retry that object in-process: mark its record stale, let the current
        OOM return RETRYABLE, then reload cleanly at the learned per-ref rung
        when the hub dispatches the retry.

        """
        if not is_cuda_oom(exc) or getattr(ctx, "cancelled", False):
            return
        if spec.kind != "inference":
            # Producer jobs (training/conversion) must surface RETRYABLE to
            # the hub — an in-process whole-job replay would redo hours of
            # work the hub can resume from a checkpoint instead.
            return
        if spec.output_mode == "stream":
            return  # chunks already emitted; a replay would duplicate them
        # pgw#748 §5.4: under sequence parallelism the degraded ladder is the
        # single most dangerous adaptive path there is. It picks an offload
        # rung from THIS card's measured free VRAM, so two ranks that OOM
        # differently take different rungs, execute different numbers of
        # collectives, and the group HANGS — or worse, agrees on the count and
        # silently produces wrong output. And CPU offload does not compose
        # with context parallelism at all (diffusers #12533: a shape error
        # after the first pipe call). So the group refuses as a group. Rank 0
        # decides; a rank that cannot honour the decision fails the whole
        # group; nothing ever adapts locally.
        rec_sp = self._classes.get(spec.instance_key)
        if rec_sp is not None and rec_sp.sp_runtime is not None:
            plan = self.group_plan_for(rec_sp)
            logger.error(
                "DEGRADED_MODE=refused fn=%s: CUDA OOM inside a degree-%d "
                "sequence-parallel group. The ladder would pick a rung from "
                "this rank's free VRAM alone (plan=%s); a per-rank rung either "
                "hangs the collective or corrupts the output silently, and CPU "
                "offload does not compose with context parallelism "
                "(diffusers #12533). Failing the GROUP.",
                spec.name, getattr(plan, "sp_degree", 0) or self.topology.degree,
                plan,
            )
            try:
                ctx.log(
                    f"DEGRADED_MODE=refused fn={spec.name}: a sequence-parallel "
                    "group cannot degrade one rank at a time; the request fails "
                    "and the pod is re-packable by the hub.",
                    level="error",
                )
            except Exception:
                pass
            rec_sp.stale = True
            return
        # Diffusers component models expose some pipeline offload methods too
        # (notably AutoencoderKL via ModelMixin). Exclude only that known
        # component base, then retain the capability check below so custom
        # duck-typed pipeline owners continue to work.
        diffusers_component_type: Any = ()
        try:
            from diffusers import ModelMixin

            diffusers_component_type = ModelMixin
        except ImportError:
            pass
        transitions: List[Tuple[WireRef, str, str, float]] = []
        for slot in spec.models:
            ref = wire_ref(spec.models[slot])
            # pgw#678: a shared-component lane's residency entry is an
            # nn.ModuleDict, which carries none of the offload methods below —
            # the OOM rung would silently skip every lane slot.
            obj = self._slot_pipeline(spec, slot)
            if obj is None:
                continue
            if diffusers_component_type and isinstance(obj, diffusers_component_type):
                continue
            if not any(callable(getattr(obj, name, None)) for name in (
                "enable_model_cpu_offload",
                "enable_group_offload",
                "enable_sequential_cpu_offload",
            )):
                continue
            before = low_vram_mode(obj)
            after_rung = rungspec.descend(before)
            after = after_rung.name if after_rung is not None else None
            if after is not None:
                transitions.append(
                    (ref, before, after, estimate_pipeline_size_gb(obj))
                )
        refused = await self._refuse_unfittable_offload(spec, transitions)
        if refused:
            transitions = []
        for ref, from_mode, to_mode, needed_gb in transitions:
            self._record_rung_transition(
                spec, ref=ref, phase="inference",
                from_rung=from_mode or "resident", to_rung=to_mode,
                run_mode=RUN_OFFLOAD, needed_gb=needed_gb,
                detail=f"CUDA OOM mid-inference ({type(exc).__name__}); "
                       "quarantining this instance for a clean offloaded reload",
            )
        flush_memory()
        rec = self._classes.get(spec.instance_key)
        if rec is not None and rec.ready:
            rec.stale = True
        if refused:
            try:
                ctx.log(
                    f"DEGRADED_MODE=refused fn={spec.name}: CUDA OOM, and the "
                    f"offloaded reload the ladder would run does not fit this "
                    f"pod's host RAM ({refused}). The function is disabled "
                    f"here and the hub re-places it; this worker does not "
                    f"attempt the reload.",
                    level="error",
                )
            except Exception:
                pass
            return
        if not transitions:
            # th#1867: A DESCENT THAT RUNS OUT MUST NAME ITS FLOOR. With the
            # proactive fit ladder deleted (an estimate deciding placement
            # before anything is measured — §4.33), this reactive walk is the
            # only ladder, so falling off its bottom must be a typed, visible
            # refusal naming OUR code — never a silent slide into a rung
            # nothing can run, which would convert a loud estimate-error into a
            # quiet execution-error.
            floors = {
                rungspec.descent_floor(low_vram_mode(obj))
                for obj in (self._slot_pipeline(spec, slot) for slot in spec.models)
                if obj is not None
            }
            floors.discard(None)
            if rungspec.FLOOR_CPU_RUNG_UNEXECUTABLE in floors:
                activity_mod.emit_event(
                    activity_mod.KIND_SERVE_DEGRADE,
                    detail=(
                        f"fn={spec.name}: the placement ladder descended to its "
                        f"last executable rung (sequential) and the next one is "
                        f"`cpu`, which THIS BUILD CANNOT EXECUTE — the reactive "
                        f"walk treats it as plan-time only (pgw#1212). This is a "
                        f"limitation of our worker, not of the card: §1.35 "
                        f"requires every model to run on every device, CPU "
                        f"included. The request returns retryable and the hub "
                        f"re-places it."
                    ),
                    phase=rungspec.FLOOR_CPU_RUNG_UNEXECUTABLE,
                )
            logger.warning(transition_line(
                event="engaged", fn=spec.name, phase="inference",
                free_gb=get_available_vram_gb(),
                detail="CUDA OOM with no worker-owned pipeline rung; "
                       "returning retryable without reusing the instance"))
        try:
            ctx.log(
                f"DEGRADED_MODE=engaged fn={spec.name}: CUDA OOM; quarantining "
                "this instance and reloading offloaded on retry.",
                level="warning",
            )
        except Exception:
            pass

    async def _refuse_unfittable_offload(
        self, spec: EndpointSpec,
        transitions: List[Tuple[WireRef, str, str, float]],
    ) -> str:
        """pgw#1063: price the offloaded reload the ladder is about to
        prescribe, and refuse the DEGRADE when the host cannot hold it.

        THREAT (§4.25): an offload rung keeps the whole weight set in host
        RAM. ie#615's H3 degrade re-staged a 105 GB set into a 233.76 GiB
        cgroup that was already holding the previous staging's anon — every
        fault went through direct reclaim, `read_bytes` reached 1.578 TB
        (a 15x re-read of a 105 GB set) over 37 minutes of billed H100, and
        the kernel OOM-killed the child. That death was arithmetically
        certain at minute zero, off numbers this worker already had.

        The observable is the same one pgw#752 refuses on: tree bytes on
        disk plus the staging floor against the cgroup-aware host TOTAL. A
        shortfall against the total is structural — no eviction, and no
        identical pod, changes it (th#1228) — so it reports as a hardware
        axis, the function self-disables here, and the hub places the work
        somewhere that can hold it. A shortfall against what is available
        RIGHT NOW keeps the rung but publishes the typed per-ref capacity
        block, so the retry is not re-admitted onto this worker while the
        quarantined instance's own staging is still resident — the window
        attempt 5 sat in for the whole 37 minutes.

        Returns the structural refusal summary, or "" when the ladder may
        proceed."""
        res = self.store.residency
        worst: Tuple[int, WireRef] = (0, WireRef(""))
        for ref, _from_mode, to_mode, _needed_gb in transitions:
            if not touches_host_ram(to_mode):
                continue
            local = res.local_path(ref)
            if local is None:
                continue
            tree = await asyncio.to_thread(disk_gc.tree_bytes, Path(local))
            if tree > worst[0]:
                worst = (tree, ref)
        incoming, ref = worst
        if incoming <= 0:
            return ""
        headroom = await asyncio.to_thread(res.host_ram_headroom, incoming)
        if headroom.sufficient:
            return ""
        if not headroom.structural:
            # It fits a host this size — just not this host RIGHT NOW, with
            # the quarantined instance's own staging still resident. That is
            # the ie#615 shape, and the thing that must not happen is the
            # RE-ADMISSION landing back here mid-reload (attempt 5 sat in
            # that window for the whole 37 minutes). Publishing the typed
            # per-ref capacity block is how this worker says so; it clears
            # itself the moment measured headroom covers the requirement.
            await self._record_host_ram_failure([ref], InsufficientHostRamError(
                spec.name,
                incoming_bytes=incoming,
                floor_bytes=headroom.floor_bytes,
                required_bytes=headroom.required_bytes,
                available_before_bytes=headroom.available_bytes,
                available_after_bytes=headroom.available_bytes,
                total_bytes=headroom.total_bytes,
            ))
            logger.warning(transition_line(
                event="engaged", fn=spec.name, model=ref, phase="inference",
                from_rung="resident", to_rung="model_offload",
                free_gb=get_available_vram_gb(),
                detail="the offloaded reload does not fit this host's CURRENT "
                       "headroom; the rung is learned but the ref is blocked "
                       "here until measured headroom covers it, so the retry "
                       "is not re-admitted into a reload that cannot fit "
                       "(pgw#1063)",
            ))
            return ""
        error = HostRamCapacityError(
            spec.name,
            incoming_bytes=incoming,
            floor_bytes=headroom.floor_bytes,
            required_bytes=headroom.required_bytes,
            available_before_bytes=headroom.available_bytes,
            available_after_bytes=headroom.available_bytes,
            total_bytes=headroom.total_bytes,
        )
        summary = (
            f"the {ref} weight set is "
            f"{incoming / float(1 << 30):.1f}GiB and an offloaded pipeline "
            f"keeps it in host RAM; required "
            f"{headroom.required_bytes / float(1 << 30):.1f}GiB against a "
            f"{headroom.total_bytes / float(1 << 30):.1f}GiB host total"
        )
        logger.error(transition_line(
            event="refused", fn=spec.name, model=ref, phase="inference",
            from_rung="resident", to_rung="model_offload",
            free_gb=get_available_vram_gb(),
            detail=f"the offloaded reload cannot fit its own staging: "
                   f"{summary}. Refusing the degrade instead of thrashing "
                   f"the host to a kernel OOM (pgw#1063).",
        ))
        await self._record_host_ram_failure([ref], error)
        return summary

    async def _execute(
        self,
        job: _Job,
        spec: EndpointSpec,
        instance: Any,
        ctx: RequestContext,
        payload: Any,
        kwargs: Dict[str, Any],
        *,
        gpu_index: int,
    ) -> Any:
        """Run the handler until it finishes — or until the process CONFESSES
        it is stalled.

        There is no wall deadline here, deliberately (pgw#904 part d, th#1457's
        `timeout_ms` deletion, the standing no-magic-timeouts rule): a clock
        cannot distinguish a slow-but-advancing handler from a wedged one, so a
        fixed duration names no threat (§4.24). The abort authority is
        ``progress.self_diagnosis()`` — non-None only when even the FRESHEST
        open counter is stale past its own per-phase window, the same typed
        ``self_stalled`` confession the activity beat reports to the hub.

        th#1779: the gate is given a source of evidence the handler cannot
        silence by saying nothing.
        Before, the only counter a serving request opened was ``infer:steps``,
        which advances only when the handler emits a ctx event — so an endpoint
        whose render is one long silent library call had a counter frozen at
        its opening log line, and the "no magic timeouts" gate degenerated into
        exactly the 300 s wall clock this docstring disclaims. Measured:
        minimax-h3 `reference-to-video` died `worker_retryable` at exactly
        300 s on four consecutive attempts while `generate` (126-229 s)
        squeaked under the same window. ``_HandlerEvidence`` samples the same
        process CPU + disk I/O signal ``activity.watchdog`` already trusts for
        wire-silent compiles, so a silent-but-working handler PROVES it is
        working and a genuinely wedged one still confesses on fact. The
        diagnosis itself stays registry-wide, as pgw#894 pinned it.
        """
        bound = spec.method if instance is None else getattr(instance, spec.attr_name)
        call_kwargs = {spec.ctx_param: ctx, spec.payload_param: payload, **kwargs}

        owner = f"request:{job.request_id}"
        loop = asyncio.get_running_loop()
        if spec.is_async_gen:
            coro = self._pump_async_gen(job, bound(**call_kwargs))
        elif spec.is_async:
            coro = bound(**call_kwargs)
        elif spec.output_mode == "stream":
            coro = asyncio.to_thread(self._pump_sync_gen, job, bound, call_kwargs, gpu_index, loop)
        else:
            coro = asyncio.to_thread(self._call_sync, job, bound, call_kwargs, gpu_index)

        job.exec_task = asyncio.ensure_future(coro)
        # th#1111: the handler window stage_ms reconciles against (the same
        # interval runtime_ms measures).
        ctx._stages.handler_open()
        try:
            # pgw#1265: the tenant forward is where "what does one forward cost
            # this card" is MEASURED. The adopt's headroom verdict has no other
            # source of truth — and this is the same forward, on the same card,
            # that a raised residency floor would have to make room for.
            with adopt_fit.forward_watermark(gpu_index), _HandlerEvidence(owner):
                while True:
                    try:
                        return await asyncio.wait_for(
                            asyncio.shield(job.exec_task), _STALL_POLL_S)
                    except asyncio.TimeoutError:
                        # pgw#894 pins this UNSCOPED on purpose: the loop's
                        # question is "is this process wedged", and many
                        # handlers register no counter of their own.
                        stalled = progress_mod.self_diagnosis()
                        if stalled is None:
                            continue  # advancing (or evidence-free): keep waiting
                        ctx._cancel()
                        job.exec_task.cancel()
                        if not spec.is_async:
                            self._reap_stuck_thread(job)
                        raise _ExecutionStalled(
                            f"self_stalled: counter {stalled.name!r} "
                            f"({stalled.unit}) has not advanced for "
                            f"{stalled.age_s:.0f}s (window {stalled.window_s:.0f}s)"
                        ) from None
        except asyncio.CancelledError:
            # CancelJob path: the exec task was cancelled underneath us.
            raise CanceledError("canceled")
        finally:
            ctx._stages.handler_close()

    @staticmethod
    def _call_sync(
        job: _Job, bound: Callable[..., Any], call_kwargs: Dict[str, Any], gpu_index: int,
    ) -> Any:
        if torch is not None and cuda_ready():
            try:
                torch.cuda.set_device(gpu_index)
            except Exception:
                pass
        try:
            return bound(**call_kwargs)
        finally:
            # th#1779: the ONLY honest report that the handler THREAD is gone.
            job.handler_thread_done.set()

    def _reap_stuck_thread(self, job: _Job) -> None:
        """Deadline fired but the sync handler thread may not die. If it's
        still running after the recycle grace, exit so the pod is recycled.

        th#1779: this watched ``job.exec_task`` — the task the caller had just
        CANCELLED one line earlier — so ``shield(...)`` re-raised that
        cancellation immediately, the ``except BaseException`` arm read it as
        "thread finished" and the reaper never fired once. A sync handler
        cannot be killed, so the abandoned thread kept denoising on the card
        while the hub re-dispatched the same request onto the same pod:
        measured across four attempts on one H100, reserved VRAM ratcheted
        59.0 -> 75.1 -> 75.8 -> 76.7 GiB with concurrent renders stacking up.
        The thread's own completion event is the only thing that can answer
        the question the reaper asks.
        """

        async def _watch() -> None:
            done = job.handler_thread_done
            deadline = time.monotonic() + _STUCK_THREAD_RECYCLE_S
            while not done.is_set():
                if time.monotonic() >= deadline:
                    logger.critical(
                        "handler thread for %s ignored deadline+cancel for %.0fs; "
                        "recycling worker process", job.request_id, _STUCK_THREAD_RECYCLE_S,
                    )
                    self._process_exit(70)
                    return
                await asyncio.sleep(_STUCK_THREAD_POLL_S)
            # Thread finished (with error or otherwise) — no recycle needed.

        asyncio.create_task(_watch(), name=f"reap-{job.request_id}")

    # ---- streaming ---------------------------------------------------------

    def _encode_chunk(self, item: Any) -> Optional[Tuple[bytes, str]]:
        # NOTE: keep in sync with api.streaming.StreamAccumulator.add (gw#475).
        if isinstance(item, Done):
            return None
        if isinstance(item, Error):
            raise ValidationError(getattr(item, "message", "") or "stream error")
        if isinstance(item, IncrementalTokenDelta):
            return item.text.encode("utf-8"), "text/plain"
        if isinstance(item, BatchItemDelta):
            # First-class multi-item delta: msgpack keeps `chunk` binary.
            return msgspec.msgpack.encode(item), "application/x-batch-item+msgpack"
        return msgspec.json.encode(item), "application/json"

    async def _emit_progress(self, job: _Job, seq: int, data: bytes, content_type: str) -> None:
        await self._send(pb.WorkerMessage(job_progress=pb.JobProgress(
            request_id=job.request_id, attempt=job.attempt, seq=seq,
            data=data, content_type=content_type)))

    def _make_ctx_emitter(self, job: _Job) -> Callable[[Dict[str, Any]], None]:
        """RequestContext emitter: ctx.progress/log/checkpoint events →
        JobProgress on the worker stream (best-effort, droppable by contract).
        Callable from any thread (handler thread, run_process reader)."""
        loop = asyncio.get_running_loop()

        async def _send_event(data: bytes) -> None:
            try:
                await self._emit_progress(job, next(job.seq), data, EVENT_CONTENT_TYPE)
            except Exception:
                logger.debug("ctx event send failed for %s", job.request_id, exc_info=True)

        # pgw#894: THIS REQUEST'S counter, registered under this request's own
        # scope. It used to be `activity.current().counter("infer:steps", ...)`
        # — a serving request feeding whatever activity happened to be current,
        # which on a pod running a background mint is the MINT. The hub
        # advances an activity's `UpdatedAt` from a counter-name change or
        # value increase (`worker_activity.go:323-338`), and that timestamp is
        # what its stall/condemnation path reads: measured on the standing
        # chaos hub, 28 lines reported `infer:steps` under `self_mint_compile`
        # and line 4542 declined a condemnation because that mint activity was
        # "0s ago". The counter still proves the PROCESS is alive — a
        # registry-wide `progress.freshest()` sees it, which is what the
        # in-call stall loop and the drain use — it just no longer answers a
        # question about work it is not doing.
        #
        # The old comment justified the credit with "warmup forwards run
        # GPU-bound with a quiet CPU". Warmup does not reach here: it builds
        # its context through `warmup.warm_context`, which passes no emitter
        # at all, and reports its own activity-owned `warmup:jobs` counter.
        steps = progress_mod.counter(
            "infer:steps", progress_mod.UNIT_STEPS,
            owner=f"request:{job.request_id}")

        def _emit(event: Dict[str, Any]) -> None:
            if job.finished:
                return
            steps.add(1)
            try:
                data = msgspec.json.encode(event)
            except Exception:
                logger.debug("unencodable ctx event dropped for %s", job.request_id)
                return
            try:
                asyncio.run_coroutine_threadsafe(_send_event(data), loop)
            except RuntimeError:
                pass  # loop closed — worker shutting down

        return _emit

    async def _pump_async_gen(self, job: _Job, agen: Any) -> Optional[StreamResult]:
        """Pump a streaming handler; returns the terminal StreamResult
        (gw#475: live deltas are droppable, the aggregate is the record)."""
        acc = StreamAccumulator()
        async for item in agen:
            if job.ctx is not None:
                job.ctx.raise_if_cancelled()
            enc = self._encode_chunk(item)
            if enc is None:
                break
            acc.add(item)
            await self._emit_progress(job, next(job.seq), enc[0], enc[1])
        return acc.result()

    def _pump_sync_gen(
        self,
        job: _Job,
        bound: Callable[..., Any],
        call_kwargs: Dict[str, Any],
        gpu_index: int,
        loop: asyncio.AbstractEventLoop,
    ) -> Optional[StreamResult]:
        if torch is not None and cuda_ready():
            try:
                torch.cuda.set_device(gpu_index)
            except Exception:
                pass
        acc = StreamAccumulator()
        try:
            for item in bound(**call_kwargs):
                if job.ctx is not None:
                    job.ctx.raise_if_cancelled()
                enc = self._encode_chunk(item)
                if enc is None:
                    break
                acc.add(item)
                fut = asyncio.run_coroutine_threadsafe(
                    self._emit_progress(job, next(job.seq), enc[0], enc[1]), loop
                )
                fut.result()  # backpressure: block the producer on queue overflow
        finally:
            job.handler_thread_done.set()  # th#1779, same contract as _call_sync
        return acc.result()

    # ---- results -----------------------------------------------------------

    async def _serialize_output(
        self, ctx: RequestContext, request_id: str, output: Any
    ) -> Tuple[Optional[bytes], Optional[str]]:
        # th#1130 safety net: msgpack reads struct fields straight off the C
        # layout, so an un-drained deferred asset would serialize as nulls.
        # _run_job_pinned always drains first; this catches any future path
        # that does not, loudly rather than by shipping a hollow asset.
        if ctx._deferred.pending():
            logger.error(
                "deferred outputs reached serialization un-drained for %s — "
                "materializing inline", request_id)
            await asyncio.to_thread(ctx._drain_deferred_outputs)
        data = msgspec.msgpack.encode(output)
        if len(data) <= INLINE_RESULT_MAX_BYTES:
            return data, None
        try:
            # pgw#767: the ENVELOPE, never ctx.save_bytes — the client's
            # `Prefer: bytes=inline` media hint must not decide whether the
            # transport blob this ref names actually exists.
            asset = await asyncio.to_thread(
                ctx._save_result_envelope, f"results/{request_id}.msgpack", data
            )
            ref = getattr(asset, "ref", "") or ""
            if not ref:
                raise RuntimeError("upload returned no ref")
            if getattr(asset, "inline_bytes", None):
                raise RuntimeError("result envelope was not uploaded")
            return None, ref
        except Exception as exc:
            logger.warning("result blob upload failed for %s: %s", request_id, exc)
            raise RetryableError("output upload failed") from exc

    @staticmethod
    def _peak_vram_bytes(gpu_index: int) -> int:
        """This job's CUDA peak-allocator high-water mark (reset when it took
        the GPU). 0 without torch/CUDA."""
        if torch is not None and cuda_ready():
            try:
                return int(torch.cuda.max_memory_allocated(gpu_index))
            except Exception:
                return 0
        return 0

    def _metrics(
        self, queue_ms: int, started: float, concurrency_at_start: int, gpu_index: int,
        output: Any = None, execution_lane: str = "",
        runtime_terms: "Optional[Dict[str, float]]" = None,
        peak_vram_bytes: "Optional[int]" = None,
    ) -> pb.JobMetrics:
        runtime_ms = int((time.monotonic() - started) * 1000)
        # rss_at_end_bytes (pgw#513): instantaneous RSS, honestly named — the
        # OS gives no per-process peak-RSS reset, so this is NOT a per-job
        # peak (unlike peak_vram_bytes below, reset at handler start).
        rss_at_end = 0
        try:
            import psutil

            rss_at_end = int(psutil.Process().memory_info().rss)
        except Exception:
            pass
        # th#1130: the caller reads the peak before releasing the GPU permit
        # (the next job resets the watermark), and passes it in.
        peak_vram = (
            self._peak_vram_bytes(gpu_index) if peak_vram_bytes is None
            else int(peak_vram_bytes)
        )
        duration_s, output_count = _scan_output_assets(output)
        usage = _output_token_usage(output)
        return pb.JobMetrics(
            runtime_ms=runtime_ms, queue_ms=queue_ms, rss_at_end_bytes=rss_at_end,
            peak_vram_bytes=peak_vram, concurrency_at_start=max(0, concurrency_at_start),
            output_media_duration_s=duration_s, output_count=output_count,
            input_tokens=usage.prompt_tokens if usage is not None else 0,
            input_cached_tokens=usage.cached_tokens if usage is not None else 0,
            output_tokens=usage.completion_tokens if usage is not None else 0,
            lane=execution_lane,
            # th#1051: declared-formula term features from the EXECUTED
            # payload (defaults applied); empty = no declared formula.
            runtime_terms=runtime_terms or {},
        )

    async def _send_result(
        self,
        request_id: str,
        attempt: int,
        status: "pb.JobStatus",
        *,
        inline: Optional[bytes] = None,
        blob_ref: Optional[str] = None,
        safe_message: str = "",
        metrics: Optional[pb.JobMetrics] = None,
        adjustments: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        result = pb.JobResult(request_id=request_id, attempt=attempt, status=status,
                              safe_message=safe_message)
        if inline is not None:
            result.inline = inline
        elif blob_ref:
            result.blob_ref = blob_ref
        if metrics is not None:
            result.metrics.CopyFrom(metrics)
        # pgw#654: caller-visible adjustment warnings ride the RESULT
        # ENVELOPE — the hub persists them on the request record and emits
        # them on its events stream; pod logs alone never reach a caller.
        for adj in adjustments or []:
            result.adjustments.add(
                field=str(adj.get("field", "")),
                requested=str(adj.get("requested", "")),
                applied=str(adj.get("applied", "")),
                reason=str(adj.get("reason", "")),
            )
        await self._send(pb.WorkerMessage(job_result=result))

    async def _finish(self, job: _Job, status: "pb.JobStatus", **kw: Any) -> None:
        if job.finished:
            return
        job.finished = True
        # pgw#894/pgw#962: a request's scope dies with the request. A counter
        # left open after its producer stopped is the min-age counter of work
        # nobody is doing, and it confesses for whoever asks next.
        progress_mod.counter(
            "infer:steps", progress_mod.UNIT_STEPS,
            owner=f"request:{job.request_id}").finish()
        # th#1111: stamp the per-stage breakdown on EVERY terminal path (ok,
        # deadline, cancel, error) — a slow request's stages are exactly the
        # ones worth seeing.
        metrics = kw.get("metrics")
        if isinstance(metrics, pb.JobMetrics) and job.ctx is not None:
            metrics.stage_ms.update(stage_ms_for_metrics(
                getattr(job.ctx, "_stages", None), metrics.runtime_ms))
        # pgw#789: stamp the th#1293 serving DIMENSIONS on EVERY terminal path,
        # for the same reason stage_ms is stamped here — a failed or
        # deadline-exceeded request's serving mode is exactly the one worth
        # seeing, and `_metrics` is called from three separate places.
        # Measured before this landed: 0 of 416 request_state rows on the chaos
        # stack carried serving_mode, so `/v1/admin/request-latency` could not
        # separate AOT from JIT from eager over any traffic at all.
        if isinstance(metrics, pb.JobMetrics):
            served = self._served_identity(job.spec, job)
            metrics.serving_mode = served.serving_mode
            metrics.served_cell_ref = served.served_cell_ref
            metrics.served_eager_fallback = served.served_eager_fallback
            metrics.fallback_reason = served.fallback_reason
            metrics.sm = served.sm
            metrics.steps, metrics.width, metrics.height = job.shape
            # ie#655: the lane is composed from the SAME ServedIdentity these
            # five fields come from, at the SAME instant — so `metrics.lane`
            # and `metrics.serving_mode` cannot disagree about eager. The
            # dispatch-time stamp below is a forecast for `ctx.lane` (the
            # handler needs a lane before it runs); the REPORT is here, where
            # a per-request eager fallback that happened DURING the handler is
            # finally knowable.
            if job.spec is not None:
                metrics.lane = self._served_execution_lane(
                    job.spec, instructed=job.lane_report, served=served)
                job.execution_lane = metrics.lane
                # th#1871 P1 (pgw#1225): the POSTURE, stamped from the same
                # ServedIdentity and at the same instant as the lane and the
                # serving mode — so the three cannot disagree about eager
                # (ie#655's rule). It rides every terminal path for the reason
                # stage_ms does: a degraded request's posture is exactly the one
                # worth having.
                self._stamp_posture(metrics, job.spec, served, metrics.lane,
                                    instructed=job.lane_report,
                                    compile_required=job.compile_required)
        terminal_status = (
            pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED
            if job.superseded
            else (
                pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED
                if status == pb.JOB_STATUS_OK
                else (
                    pb.LIFECYCLE_INTENT_STATUS_CANCELED
                    if status == pb.JOB_STATUS_CANCELED
                    else pb.LIFECYCLE_INTENT_STATUS_FAILED
                )
            )
        )
        if not job.superseded:
            self._intent_transition(
                job.intent_id,
                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
                detail=str(kw.get("safe_message", ""))[:512],
            )
        if job.renew_task is not None:
            job.renew_task.cancel()
            job.renew_task = None
        cleanup_input_assets(job.request_id, job.attempt)
        logger.info("job finished %s attempt=%d status=%s", job.request_id, job.attempt, status)
        if not job.superseded:
            adjustments = list(getattr(job.ctx, "_adjustments", ()) or ()) \
                if job.ctx is not None else []
            await self._send_result(
                job.request_id, job.attempt, status,
                adjustments=adjustments, **kw)
        self._intent_transition(
            job.intent_id,
            terminal_status,
            (
                pb.LIFECYCLE_INTENT_STAGE_READY
                if terminal_status == pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED
                else pb.LIFECYCLE_INTENT_STAGE_FINALIZING
            ),
            detail=str(kw.get("safe_message", ""))[:512],
        )
        # Keep finished records so a RunJob retransmission doesn't re-execute;
        # prune oldest finished entries beyond a small window.
        finished = [k for k, j in self.jobs.items() if j.finished]
        if len(finished) > 1024:
            for k in finished[: len(finished) - 1024]:
                self.jobs.pop(k, None)
        self._maybe_idle()
        # pgw#674: job completion is the rotation point — advance staging.
        self.preloader.poke()

    def _maybe_idle(self) -> None:
        # pgw#677: compile-turn quiescence keys off the last tenant
        # admission OR finish — arrivals cluster around completions.
        self._bg_last_tenant_activity = time.monotonic()
        if not self.in_flight_keys():
            self._idle.set()
            self._bg_quiet.set()


class _HandlerEvidence:
    """th#1779: loop-independent proof that a SERVING handler is working.

    ``activity.watchdog`` has done exactly this for wire-silent load/compile
    phases since gw#621; a serving request had no equivalent, so the only
    counter open under it was ``infer:steps`` — which advances on ctx events
    and therefore measures how CHATTY the endpoint is, not whether it is
    working. An endpoint whose render is one long silent library call froze
    that counter at its opening log line and was condemned at the family's
    300 s window: measured, minimax-h3 `reference-to-video` died
    `worker_retryable` at exactly 300 s on four consecutive attempts, while
    `generate` (126-229 s of the same silence) fit under the window and
    passed. That is a wall clock wearing a progress counter's name.

    The evidence is process+children CPU seconds plus process disk I/O MB —
    ``activity_mod.default_evidence``, unchanged. A GPU render burns CPU issuing
    and synchronising kernels; a deadlocked or wedged process burns neither,
    which is the distinction the gate is supposed to make. Registered under
    the REQUEST's scope so ``self_diagnosis(owner)`` answers about this
    request, so the counter dies with the handler (pgw#962) and cannot answer
    for work it is not doing (pgw#894). The in-call diagnosis stays
    registry-wide, which pgw#894 pinned deliberately.
    """

    def __init__(
        self,
        owner: str,
        *,
        interval_s: float = _HANDLER_EVIDENCE_INTERVAL_S,
        evidence: Optional[Callable[[], float]] = None,
    ) -> None:
        self._owner = owner
        self._interval = interval_s
        self._evidence = evidence or activity_mod.default_evidence
        self._stop = threading.Event()
        self._counter: Optional[progress_mod.Counter] = None
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        try:
            base = last = self._evidence()
        except Exception:
            base = last = 0.0
        while not self._stop.wait(self._interval):
            try:
                now = self._evidence()
            except Exception:
                continue
            if now - last >= activity_mod.EVIDENCE_EPS and self._counter is not None:
                last = now
                self._counter.set_done(now - base)

    def __enter__(self) -> "_HandlerEvidence":
        self._counter = progress_mod.counter(
            "evidence:handler", progress_mod.UNIT_EVIDENCE, owner=self._owner)
        self._thread = threading.Thread(
            target=self._run, name="handler-evidence", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._counter is not None:
            # A counter left open after its producer stopped is the min-age
            # counter of work nobody is doing (pgw#962).
            self._counter.finish()


class _ExecutionStalled(Exception):
    """The registry's own self_stalled confession, applied to the in-flight
    handler. Raised only on ``progress.self_diagnosis()`` — never a clock."""
