"""Job execution: intake, GPU semaphore, deadline + cancellation watchdog,
sync-on-thread / async-on-loop, JobProgress deltas, result send, and the
worker-side model seam (ensure-local, setup injection, declarative residency,
and compile-cache adoption).

One dispatch path for every endpoint kind. Everything runs on the single
asyncio loop; sync tenant code runs in threads via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import gc
import itertools
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import typing
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, cast


import msgspec

from . import activity as activity_mod
from . import boot_phases as boot_mod
from . import cpu_budget
from . import kernel_path
from . import mint_budget
from . import progress as progress_mod
from . import serving_mode as serving_mode_mod
from . import warmup
from . import worker_credential
from . import mint_goal as mint_goal_mod
from . import worker_goals
from .api.binding import ModelRef, wire_ref
from .convert.hub import HubPublishError
from .mint_process import MintSlot
from .api.errors import (
    ArtifactTransferError,
    CanceledError,
    ComponentSubstitutionError,
    GpuSlotUnreachable,
    IllegalCombination,
    ModelSlotIdentityError,
    RetryableError,
    ValidationError,
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
from .input_assets import cleanup_input_assets, manifest_from_run_job, materialize_input_assets
from .intent_registry import IntentRegistry
from .models import cozy_snapshot
from .models import disk_gc
from .models import disk_telemetry
from .models import loading
from .models import provision
from .models import residency as residency_mod
from .models import staging as staging_mod
from .models.memory import (
    aflush_memory,
    deeper_offload_mode,
    degraded_log_line,
    estimate_cuda_resident_gb,
    estimate_pipeline_size_gb,
    flush_memory,
    get_available_vram_gb,
    is_cuda_oom,
    low_vram_mode,
    next_offload_rung,
    release_unused_pinned_host_cache,
)
from .models.cache_paths import tensorhub_cas_dir, tensorhub_fill_source_dir
from .models.download import ensure_local, lookup_provider_for_ref
from .models.errors import MissingSnapshotError, UrlExpiredError
from .models.execution_lanes import ExecutionLaneUnavailableError
from .models.residency import Residency
from .topology import (
    ExecutionTopology,
    TopologyError,
    current_device_group,
    device_group_scope,
    pin_cuda_device_for_group,
)
from .pb import worker_scheduler_pb2 as pb
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
    from .models.hub_client import WorkerResolvedRepo
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
from .models.refs import DEFAULT_REF_TAG, parse_model_ref
from .models.hub_client import WorkerResolvedChunk, WorkerResolvedRepo, WorkerResolvedRepoFile
from . import cell_key, compile_cache
from .models.config_identity import CANONICAL_JSON_MAX_BYTES, canonical_json_digest
from .models.cozy_snapshot import _norm_rel_path
from .models.loading import safetensors_file_valid
from .models.volume_verify import snapshot_verify_targets, verify_files
from .models.cozy_snapshot import delete_blobs
from .compile_cache import CompiledExecutionLaneUnavailableError
from .preload import Preloader
from .api.binding import rebind_pick
from .models.hub_policy import FIT_INCOMPATIBLE, TensorhubWorkerCapabilities
from .models.serve_fit import RUN_CPU, RUN_OFFLOAD, plan_serve
from . import postmortem
from .models.serve_fit import demoted
from .models.serve_fit import load_rung_engaged
from .models.serve_fit import cast_dropped
from .models.loading import pipeline_weight_lane
from .models import execution_lanes as lanespec
from . import warmup as warmup_mod
from .api.decorators import ATTR as _DECL_ATTR
from . import compile_cache as _cc_execution_lane
from .parallel import ContextParallelUnavailable
from .parallel import GroupPlan
from .parallel.cp import w8a8_gemm_mode
from .parallel.runtime import BootPlan, SequenceRuntime, arm_sequence_gate
from .runtimes.server import RUNTIME_FACTORIES
from .models.loading import composition_compute_dtype
from .runtimes.server import ServerHandle
from .models.execution_lane_gate import ExecutionLaneGate, arm_execution_lane_gate
from .models.memory import rearm_offload
from . import fleet_cells
from . import aot_serve, shape_growth, trt_engine
from . import fleet_cells as fleet_cells_mod
from . import hot_swap
from . import mint_delegate
from . import hot_swap as hot_swap_mod

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
_STUCK_THREAD_RECYCLE_S = 30.0
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
_DOWNLOAD_RETRIES = 3
_PROGRESS_EVENT_MIN_INTERVAL_S = 5.0
# th#763: how long a cold tensorhub ref waits for the hub's re-minted
# snapshot after reporting missing_snapshot. The FAILED event triggers an
# immediate hub-side re-mint (resolve + DOWNLOAD push), so arrival is
# seconds; the bound only caps a hub that never answers.
_MISSING_SNAPSHOT_WAIT_S = 60.0
_GiB = 1024 ** 3
# Disk headroom preserved beyond a download's known size (#370).
_DISK_GC_MARGIN_BYTES = 2 * _GiB
# Refs used within the grace window are not disk-GC candidates.
_DISK_GC_GRACE_S = 300.0
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
_REDACTIONS = (
    re.compile(r"Bearer\s+[^\s\"'&]+"),
    re.compile(r"(?:X-Amz-[A-Za-z0-9-]+|Signature)=[^&\s\"']*"),
    # Absolute unix filesystem paths (/tmp/..., /app/..., /home/...): require
    # two segments so bare "/" and owner/repo-style refs survive, and no
    # scheme/word directly before the slash so URL paths inside https://...
    # stay intact. Pods are linux-only; no Windows drive-path variant.
    re.compile(r"(?<![\w:/])/(?:[\w.@+-]+/)+[\w.@+-]*"),
)


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


def _sanitize(message: str) -> str:
    out = str(message or "").strip()
    for pat in _REDACTIONS:
        out = pat.sub("[redacted]", out)
    return out[:1024]


def _reserved_repo_info(payload: Any, field_name: str) -> Dict[str, Any]:
    """``payload.source`` / ``payload.destination`` / ``payload.text_encoder``
    / ``payload.candidate`` as a plain dict ({} when absent). Producer payloads
    carry these reserved-name structs (#376, pgw#594, pgw#684). The set of
    names is hardcoded here; pgw#690 tracks making it declarative."""
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
    ``owner/repo[:tag][#flavor]`` grammar; this mints the binding the
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
        tag=th.tag or DEFAULT_REF_TAG,
        flavor=th.flavor or "",
    )


def _component_overrides(binding: Any) -> Tuple[Tuple[str, str], ...]:
    """(component, canonical ref) substitutions the binding carries (pgw#617)."""
    return tuple(getattr(binding, "component_overrides", ()) or ())


def _binding_wire_refs(binding: Any) -> List[str]:
    """The base wire ref plus every component-override ref (pgw#617): the
    full set of refs materializing this binding pins/downloads."""
    return [wire_ref(binding), *(ref for _, ref in _component_overrides(binding))]


def _snapshot_files_without_components(
    snapshot: "Optional[pb.Snapshot]", exclude: typing.Sequence[str],
) -> "List[pb.SnapshotFile]":
    """``snapshot.files`` minus every entry under an excluded ``<comp>/``
    subfolder (th#1330 B2). The one place the worker's byte accounting agrees
    with what the downloader will actually fetch."""
    files = list(snapshot.files) if snapshot is not None else []
    drop = {str(c).strip() for c in exclude if str(c or "").strip()}
    if not drop:
        return files
    kept = []
    for f in files:
        rel = str(f.path).strip().lstrip("/")
        top, sep, _ = rel.partition("/")
        if sep and top in drop:
            continue
        kept.append(f)
    return kept


def _snapshot_without_components(
    snapshot: "Optional[pb.Snapshot]", exclude: typing.Sequence[str],
) -> "Optional[pb.Snapshot]":
    """``snapshot`` re-stated over the narrowed file set — the manifest a
    verifier must use when the tree on disk was fetched with an exclusion."""
    if snapshot is None or not exclude:
        return snapshot
    return pb.Snapshot(
        digest=snapshot.digest,
        files=_snapshot_files_without_components(snapshot, exclude),
    )


def _alias_binding_matches(alias: "EndpointSpec", slot_key: str, ref: str) -> bool:
    """Does ``alias`` hold this load-time binding fact? ``slot_key`` is a
    slot name or ``<slot>.<component>`` override key (pgw#617)."""
    base, _, comp = slot_key.partition(".")
    binding = alias.models.get(base)
    if binding is None:
        return False
    if not comp:
        return wire_ref(binding).strip() == ref
    return (comp, ref) in _component_overrides(binding)


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


# ---------------------------------------------------------------------------
# Model seam: models.download (ensure-local) + models.residency (tier map),
# with ModelEvent emission. Single-loop, per-ref asyncio locks — no
# check-then-create races.
# ---------------------------------------------------------------------------


def _snapshot_to_resolved(snap: pb.Snapshot) -> "WorkerResolvedRepo":
    """pb.Snapshot -> the typed resolved-manifest struct (gw#497): the ONE
    wire-boundary conversion; everything downstream (ensure_local,
    ensure_snapshot_async) is typed — no dict laundering."""

    return WorkerResolvedRepo(
        snapshot_digest=snap.digest,
        files=[
            WorkerResolvedRepoFile(
                path=f.path,
                size_bytes=int(f.size_bytes),
                url=f.url or None,
                # th#1303 manifest v2: the algorithm-tagged digest and the
                # ordered chunk list. Dropping these here is what would make
                # every chunked snapshot look like a whole file with no URL.
                #
                # DIRECT FIELD ACCESS, deliberately — not `getattr(f, "digest",
                # "")`. These were read defensively at first, and the default
                # turned "the generated stub does not have this field" into
                # "the hub sent an empty value": the vendored proto WAS stale
                # (no `digest`/`chunks` at all), so every v2 snapshot arrived
                # blank on the production gRPC path and nothing said why. A
                # missing field must be an AttributeError at import-adjacent
                # code, not a silent empty string — same class as guarding a
                # digest check on the legacy field's truthiness.
                digest=f.digest or "",
                chunks=tuple(
                    WorkerResolvedChunk(
                        sha256=(c.sha256 or "").strip().lower(),
                        url=c.url,
                        length=int(c.len),
                    )
                    for c in f.chunks
                ),
                chunk_size_bytes=int(f.chunk_size_bytes or 0),
            )
            for f in snap.files
        ],
    )


#: Traced weight lanes a stored flavor MANDATES (fail-closed serving):
#: `#fp8-w8a8` -> "w8a8" (gw#534), `#nvfp4-w4a4` -> "w4a4" (gw#540).
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


def _cell_execution_lane_matches(
    ref: str,
    family: str,
    *,
    want_execution_lane: str,
    want_bucket: int,
    candidate_keys: typing.AbstractSet[str] = frozenset(),
) -> bool:
    """Whether an inductor cell ref serves this endpoint's graph family
    (gw#561): the declared rank bucket is half of the identity — a
    lora_bucket endpoint needs exactly a ``-lora<bucket>`` cell of its base
    lane, and a branchless endpoint must never fetch one (either mismatch is
    a guaranteed lane_drift that would shadow the right cell and serve
    eager). Key-flavored cells (th#883 pull-by-key, ``#ck1-…``) match only
    when their key is one this runtime computed for itself."""

    if not compile_cache.is_cache_ref(ref, family):
        return False
    _fam, flavor = compile_cache.parse_cell_ref(ref)
    if cell_key.is_key(flavor):
        return flavor in candidate_keys
    base, bucket = compile_cache.execution_lane_bucket(compile_cache.cell_execution_lane(ref))
    if bucket != int(want_bucket or 0):
        return False
    if want_execution_lane:
        return base == want_execution_lane
    return base not in _MANDATORY_EXECUTION_LANES


def _ref_mandatory_execution_lane(ref: str) -> str:
    """The traced weight lane one canonical Tensorhub model ref MANDATES:
    "w8a8" for `#fp8-w8a8` flavors, "w4a4" for `#nvfp4-w4a4`, "" otherwise."""

    try:
        parsed = parse_model_ref(ref).tensorhub
    except ValueError:
        return ""
    if parsed is None or parsed.owner == "root":
        return ""
    flavor = parsed.flavor or ""
    if flavor == "fp8-w8a8" or flavor.startswith("fp8-w8a8-"):
        return "w8a8"
    if flavor == "nvfp4-w4a4" or flavor.startswith("nvfp4-w4a4-"):
        return "w4a4"
    return ""


def _mandatory_execution_lane_of(refs: typing.Iterable[str]) -> str:
    """The single mandatory lane a binding set selects ("" when none)."""
    for ref in refs:
        execution_lane = _ref_mandatory_execution_lane(ref)
        if execution_lane:
            return execution_lane
    return ""


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


def _estimate_setup_need(
    per_ref: typing.Sequence[Tuple[int, int]],
    vram_gb: float,
) -> int:
    """Pre-load VRAM headroom estimate for one setup's refs (pgw#636).

    ``per_ref`` carries ``(vram_hint, snapshot_bytes)`` per ref: a prior
    MEASURED footprint wins; else the wire snapshot's byte total (an honest
    first-load footprint for stored-precision lanes — make_room's margin
    covers slack). Only when a ref has NEITHER fact does the declared
    ``vram_gb`` floor the total: the declaration is a placement MINIMUM
    ("a card with at least this much"), never a per-load reservation —
    reserving it wholesale for every never-seen checkpoint pick evicted the
    resident pipeline on 24 GB cards and pinned workers to one pipeline
    (the 2026-07-24 9.8/24 GB incident)."""
    needed = 0
    unknown = False
    for hint, snapshot_bytes in per_ref:
        size = hint if hint > 0 else max(0, int(snapshot_bytes))
        if size <= 0:
            unknown = True
        needed += size
    if unknown and vram_gb > 0:
        needed = max(needed, int(vram_gb * _GiB))
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


def _is_terminal_download_error(exc: BaseException) -> bool:
    if isinstance(exc, (UrlExpiredError, InsufficientDiskError, MissingSnapshotError)):
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        # requests.HTTPError carries the code on .response, not the exception.
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    return isinstance(exc, (ValueError, KeyError))


_RESIDENCY_STATE_TO_PB = {
    residency_mod.ON_DISK: pb.MODEL_STATE_ON_DISK,
    residency_mod.IN_RAM: pb.MODEL_STATE_IN_RAM,
    residency_mod.IN_VRAM: pb.MODEL_STATE_IN_VRAM,
    residency_mod.EVICTED: pb.MODEL_STATE_EVICTED,
}

_TIER_TO_PB = {
    residency_mod.Tier.VRAM: pb.RESIDENCY_TIER_VRAM,
    residency_mod.Tier.RAM: pb.RESIDENCY_TIER_RAM,
    residency_mod.Tier.DISK: pb.RESIDENCY_TIER_DISK,
}

_USE_RESIDENT_IDENTITY = object()
_ResidencyIdentity = Tuple[str, int]


@dataclass(frozen=True)
class _MaterializedLocal:
    path: Path
    identity: _ResidencyIdentity


class ModelStore:
    """The worker's model seam: ensure-local with retries, the residency map,
    and disk retention (#370). All tier transitions flow through
    :class:`~gen_worker.models.residency.Residency`, whose events this store
    forwards as wire ``ModelEvent``s."""

    def __init__(
        self,
        emit: Callable[[pb.WorkerMessage], Awaitable[None]],
        *,
        hf_home: str = "",
        hf_token: str = "",
        cache_dir: Optional[Path] = None,
        vram_budget_bytes: Optional[int] = None,
        disk_free_bytes_fn: Optional[Callable[[], int]] = None,
        fill_source_dir: Optional[Path] = None,
    ) -> None:
        self._emit = emit
        self._intent_registry: Optional[IntentRegistry] = None
        self._hf_home = hf_home or None
        self._hf_token = hf_token or None
        self._cache_dir = cache_dir or tensorhub_cas_dir()
        # th#850 managed-tier ruling (gw#599): endpoint-scoped datacenter-warm
        # fill source (RunPod volume mount), consulted before R2 on a blob
        # miss — resolved once at boot like _cache_dir; never the CAS root.
        # Same `or` shape as _cache_dir above: an explicit path (tests) wins,
        # otherwise resolve from env (production/tensorhub; unset -> None,
        # the cozy-local/no-volume degenerate case).
        self._fill_source_dir = fill_source_dir or tensorhub_fill_source_dir()
        # th#1063 visibility guard: a datacenter pod without a warm fill
        # source silently pulls everything from R2 with write-through off —
        # that state must be visible in the boot log, never inferred.
        if self._fill_source_dir is None and os.environ.get("RUNPOD_POD_ID") and (
            os.environ.get("RUNPOD_PROVIDER", "") != "local"
        ):
            configured = os.environ.get("TENSORHUB_FILL_SOURCE_DIR", "").strip()
            if configured:
                logger.warning(
                    "fill_source_disabled reason=not_a_mount configured=%s: "
                    "TENSORHUB_FILL_SOURCE_DIR is set but not a mounted volume; "
                    "all fills go to R2, write-through disabled (th#1063)",
                    configured,
                )
            else:
                logger.warning(
                    "fill_source_disabled reason=unset: datacenter pod booted with no "
                    "TENSORHUB_FILL_SOURCE_DIR (no endpoint volume attached); "
                    "all fills go to R2, write-through disabled (th#1063)"
                )
        elif self._fill_source_dir is not None:
            logger.info("fill_source_enabled dir=%s (volume-warm CAS fill tier)", self._fill_source_dir)
        # pgw#748 phase 1: ONE Residency registry per execution group, sharing
        # only the disk tier. VRAM is not fungible between cards, so a group's
        # LRU, leases and free-VRAM probe speak that group's devices and
        # nothing else — which is exactly what DeviceGroup's docstring has
        # promised since pgw#648. ``residency`` resolves the CURRENT group
        # (the executor stamps it per job), so every existing call site keeps
        # working and a single-group pod behaves byte-identically.
        self._vram_budget_bytes = vram_budget_bytes
        self._residency_by_group: Dict[int, Residency] = {}
        self._residency_groups: Dict[int, "residency_mod.DeviceGroup"] = {}
        self._residency_lock = threading.Lock()
        self.residency_topology: Optional[Any] = None
        self._residency_by_group[0] = Residency(
            on_event=self._on_residency_event, vram_budget_bytes=vram_budget_bytes,
        )
        self._locks: Dict[str, asyncio.Lock] = {}
        self._materialize_active: Dict[str, str] = {}
        self._materialize_intent_context: contextvars.ContextVar[str] = contextvars.ContextVar(
            "materialize_intent", default=""
        )
        self._bindings: Dict[str, Any] = {}
        # th#1330 B2: ref -> the component set last skipped for it, so the
        # typed event fires on transitions and not once per materialization.
        self._override_exclusions_reported: Dict[str, Tuple[str, ...]] = {}
        self.keep: list[str] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._index = disk_gc.RefIndex(self._cache_dir)
        self._disk_free = disk_free_bytes_fn or self._default_disk_free
        # Refs whose on-disk snapshot passed integrity verification THIS boot
        # (gw#408): a cached snapshot is re-verified on first use per process
        # so pod-churn corruption can never be trusted forever.
        self._verified: set[str] = set()
        # Last digest-carrying snapshot seen per ref (gw#465): companion-slot
        # setups may arrive snapshot-less; without memory of the hub's desired
        # state / RunJob snapshot they cannot materialize tensorhub refs. Stale
        # URLs self-heal: they fail url_expired and the hub re-mints.
        self._snapshots: Dict[str, pb.Snapshot] = {}
        # Current generation attached to each banked snapshot. A generation-
        # less bank inherits only from the exact current desired identity
        # below; historical desired generations are never resurrected.
        self._snapshot_generations: Dict[str, int] = {}
        # Current full-replacement desired identity per ref. This is bounded
        # by the active DesiredResidency set, not an unbounded digest history:
        # a priority RunJob may bank different bytes temporarily, while a
        # later generation-less bank of the still-desired digest recovers its
        # causal generation. Replacing desired state clears stale generations.
        self._desired_snapshot_identities: Dict[str, _ResidencyIdentity] = {}
        # Identity of the bytes that ACTUALLY produced the current residency.
        # This deliberately does not follow _snapshots when a tag moves.
        self._resident_identities: Dict[str, _ResidencyIdentity] = {}
        # A newer snapshot may coexist on disk while the prior snapshot's
        # pipeline is still in RAM/VRAM. Keep the disk identity separately
        # until record teardown makes it the highest residency tier.
        self._disk_identities: Dict[str, _ResidencyIdentity] = {}
        # pgw#628 (th#1070 residency v2): every applied HelloAck opens a new
        # republish epoch. The reconcile pass re-announces verified cached
        # identities the hub re-asked about even when unchanged — observations
        # are content-addressed and idempotent hub-side, and a force-resent
        # plan is exactly the hub saying "tell me again" (redrive/overdue
        # resends could otherwise never heal a lost success observation).
        # Job-path ensure_local calls within the same epoch stay deduped.
        self._residency_republish_epoch = 0
        self._identity_publish_epochs: Dict[str, int] = {}
        self._identity_lock = threading.RLock()
        # Cold-ref waiters (th#763): ensure_local blocks here until the
        # hub's re-minted DOWNLOAD banks a snapshot for the ref.
        self._snapshot_waiters: Dict[str, asyncio.Event] = {}
        # th#850 managed-tier ruling (gw#599): network_bytes for the NEXT
        # ON_DISK transition of this ref, handed off to
        # _on_residency_event so the one authoritative wire event Residency
        # emits carries it — set immediately before track_disk(), consumed
        # (popped) by _on_residency_event if it fires, cleared defensively
        # otherwise. Avoids a second, redundant ON_DISK event and avoids
        # widening EventFn's arity (Residency has other direct callers).
        self._pending_network_bytes: Dict[str, int] = {}
        # pgw#610/th#962 measured disk telemetry: generation bumps only when
        # the measured (quantized) shape changes, so the hub can fence
        # insufficient-disk failure clears on real capacity change.
        self._disk_report_lock = threading.Lock()
        self._disk_capacity_generation = 0
        self._last_disk_shape: Optional[bytes] = None
        # boothang fix: disk_usage_report() rides EVERY StateDelta build
        # (_state_delta() is a plain sync method called directly from many
        # call sites, some outside an await — never itself offloaded to a
        # thread). Measuring here means statvfs()/stat() on a real mount —
        # the provider-attached VOLUME fill-source is a network-backed
        # mount that can stall for minutes under load, exactly what a
        # self-mint's weight download + cell pack produce right before the
        # first post-publish delta. A stalled statvfs on the event loop
        # thread freezes the ENTIRE worker (including the th#965 heartbeat,
        # which shares the same loop): every StateDelta, RunJob dispatch,
        # and drain signal stops until the syscall returns. Cache the
        # measurement and refresh it off-loop (refresh_disk_usage_report,
        # driven by Lifecycle's TTL gate); disk_usage_report() only ever
        # reads the cache, so the hot state-delta path never blocks on I/O.
        self._cached_disk_usage_report = pb.DiskUsageReport()

    def _default_disk_free(self) -> int:
        p = Path(self._cache_dir)
        for candidate in (p, *p.parents):  # cache dir may not exist yet
            try:
                return int(shutil.disk_usage(candidate).free)
            except OSError:
                continue
        return 0

    # ---- events ------------------------------------------------------------

    def bind_loop(self) -> None:
        """Capture the running loop so residency events raised from worker
        threads (demote/promote via to_thread) still reach the wire."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def bind_intent_registry(self, registry: IntentRegistry) -> None:
        self._intent_registry = registry

    def _materialize_intent(self, ref: str) -> str:
        registry = self._intent_registry
        if registry is None:
            return ""
        return registry.ensure_local_intent(
            "materialize",
            ref,
            detail=f"materialize {ref}",
        )

    @contextmanager
    def materialize_intent(
        self,
        intent_id: str,
    ) -> typing.Iterator[None]:
        token = self._materialize_intent_context.set(intent_id)
        try:
            yield
        finally:
            self._materialize_intent_context.reset(token)

    async def _materialize_await(
        self,
        intent_id: str,
        awaitable: Awaitable[Any],
        *,
        operation: str,
        status: "pb.LifecycleIntentStatus",
        stage: "pb.LifecycleIntentStage",
        reason: "pb.LifecycleWaitReason" = pb.LIFECYCLE_WAIT_REASON_UNSPECIFIED,
        next_retry_at_unix_ms: int = 0,
        blocker_intent_id: str = "",
    ) -> Any:
        registry = self._intent_registry
        if registry is None:
            return await awaitable
        return await registry.reported_await(
            intent_id,
            awaitable,
            operation=operation,
            status=status,
            stage=stage,
            reason=reason,
            next_retry_at_unix_ms=next_retry_at_unix_ms,
            blocker_intent_id=blocker_intent_id,
        )

    def _on_residency_event(
        self, ref: str, state: str, vram_bytes: int, duration_ms: int = 0
    ) -> None:
        pb_state = _RESIDENCY_STATE_TO_PB.get(state)
        if pb_state is None:
            return
        kw: Dict[str, Any] = {}
        if state == residency_mod.IN_VRAM:
            kw["vram_bytes"] = int(vram_bytes)
        if duration_ms > 0:
            # Swap telemetry (gw#479): promote/demote wall time rides the
            # existing ModelEvent.duration_ms field.
            kw["duration_ms"] = int(duration_ms)
        if state == residency_mod.ON_DISK:
            with self._identity_lock:
                identity = self._disk_identities.get(
                    ref, self._resident_identities.get(ref, ("", 0))
                )
                if identity[0]:
                    self._resident_identities[ref] = identity
            pending_network_bytes = self._pending_network_bytes.pop(ref, None)
            if pending_network_bytes is not None:
                kw["network_bytes"] = int(pending_network_bytes)
        else:
            identity = self.resident_identity(ref)
        coro = self._event(ref, pb_state, identity=identity, **kw)
        if state == residency_mod.EVICTED:
            # Capture before removal so the eviction names the exact bytes it
            # removed; later events cannot inherit that stale identity.
            with self._identity_lock:
                self._resident_identities.pop(ref, None)
                self._disk_identities.pop(ref, None)
                self._identity_publish_epochs.pop(ref, None)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is not None and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            else:
                coro.close()
            return
        loop.create_task(coro)

    def model_event(
        self,
        ref: str,
        state: "pb.ModelState",
        *,
        identity: Any = _USE_RESIDENT_IDENTITY,
        **kw: Any,
    ) -> pb.ModelEvent:
        """Build one identity-fenced model event.

        Residency transitions and failures default to the identity of the
        resident bytes. Downloads pass their operation identity explicitly so
        a newly banked tag cannot relabel the old resident model.
        """
        if identity is _USE_RESIDENT_IDENTITY:
            identity = self.resident_identity(ref)
        digest, generation = identity or ("", 0)
        if digest:
            kw.setdefault("snapshot_digest", digest)
        if generation:
            kw.setdefault("residency_generation", int(generation))
        return pb.ModelEvent(ref=ref, state=state, **kw)

    async def _event(
        self,
        ref: str,
        state: "pb.ModelState",
        *,
        identity: Any = _USE_RESIDENT_IDENTITY,
        **kw: Any,
    ) -> None:
        await self._emit(pb.WorkerMessage(
            model_event=self.model_event(ref, state, identity=identity, **kw)
        ))

    # ---- per-group residency (pgw#748 phase 1) --------------------------------

    @property
    def residency(self) -> Residency:
        """The registry for the execution group this task is serving.

        The group is ambient because the device already is: every handler
        thread runs under ``torch.cuda.set_device(gpu_index)`` and every
        ``.to("cuda")`` in the load path follows the current device. This
        makes that ambient fact explicit and bookkept, instead of leaving G
        groups sharing one VRAM ledger they cannot all be true about.
        """
        return self.residency_for(current_device_group())

    def residency_for(self, group: int) -> Residency:
        g = int(group)
        existing = self._residency_by_group.get(g)
        if existing is not None:
            return existing
        with self._residency_lock:
            existing = self._residency_by_group.get(g)
            if existing is not None:
                return existing
            device_group = self._residency_groups.get(g)
            if device_group is None:
                # No topology delivered (or a group the topology does not
                # describe): fall back to the single-device group at that
                # ordinal rather than inventing a width.
                device_group = residency_mod.DeviceGroup(devices=(g,))
            reg = Residency(
                on_event=self._on_residency_event,
                vram_budget_bytes=self._vram_budget_bytes,
                device_group=device_group,
            )
            # Cross-group invariants that are wired once at boot on group 0.
            reg.pre_demote = self._residency_by_group[0].pre_demote
            self._residency_by_group[g] = reg
            logger.info(
                "residency registry armed for group %d on devices %s",
                g, list(device_group.devices),
            )
            return reg

    def all_residencies(self) -> List[Residency]:
        """Every armed group registry. Disk-facing questions (GC keep-sets,
        in-use, local paths) must union across these — the CAS is one tree
        with one page cache, shared by every group (§4.3)."""
        return list(self._residency_by_group.values())

    def bind_topology(self, topology: Any) -> None:
        """Install the delivered `G×D` packing: one registry per group, each
        accounting for exactly its own devices."""
        self.residency_topology = topology
        if topology is None:
            return
        with self._residency_lock:
            for ordinal in range(int(topology.execution_groups)):
                self._residency_groups[ordinal] = topology.group(ordinal)
            zero = self._residency_by_group.get(0)
            if zero is not None:
                # Group 0's registry predates the topology (it is created in
                # __init__ so a topology-less worker is never registry-less).
                # Retarget it rather than replace it: its entries and leases
                # are already the live bookkeeping.
                zero.device_group = self._residency_groups[0]
        # pgw#780 item 1: the pinned-host fair share was DEAD code — the pool's
        # per-group cap only engages once it knows G, and nothing in src/ ever
        # told it. Without this a G=4 degraded pod lets group 0 claim the whole
        # pinned budget (§4.3 caveat 2).
        staging_mod.pinned_pool().set_group_count(int(topology.execution_groups))
        # pgw#780 item 2: registries were created lazily on first dispatch, so
        # the boot disk re-track (which unions over all_residencies()) was a
        # no-op for groups 1..G-1 — their LRU/preserve/eviction views started
        # blind to the disk tier that was already there. Create every group's
        # registry NOW, before any boot walk unions over them.
        for ordinal in range(int(topology.execution_groups)):
            self.residency_for(ordinal)

    def disk_ref_in_use(self, ref: str) -> bool:
        """In-use across ALL groups (§4.3 caveat 3): one group's GC must never
        drop the pages another group is mmapping."""
        return any(reg.in_use(ref) for reg in self.all_residencies())

    def disk_local_path(self, ref: str) -> Optional[Path]:
        for reg in self.all_residencies():
            path = reg.local_path(ref)
            if path is not None:
                return path
        return None

    def disk_refs(self) -> List[str]:
        """Union of DISK-tier refs across groups."""
        seen: Dict[str, None] = {}
        for reg in self.all_residencies():
            for ref in reg.refs_in(residency_mod.Tier.DISK):
                seen.setdefault(ref, None)
        return list(seen)

    # ---- residency facade ----------------------------------------------------

    def residency_snapshot(self) -> List[pb.ModelResidency]:
        out: List[pb.ModelResidency] = []
        # pgw#776 / DPA-6: union across EVERY group's registry. This runs on
        # the event-loop thread (where _state_delta lives), whose contextvar
        # is always the default group — reading `self.residency` here meant a
        # G=4 pod reported 1/G of its resident set, and the hub's cache-aware
        # victims, keep-warm objectives and warm-preference routing all
        # decided on a quarter of the truth. Same union rule as disk_refs():
        # one row per ref at its BEST tier, vram summed across groups (the
        # pod's total VRAM commitment for that ref).
        merged: Dict[str, Tuple[residency_mod.Tier, int]] = {}
        rank = {
            residency_mod.Tier.VRAM: 2,
            residency_mod.Tier.RAM: 1,
            residency_mod.Tier.DISK: 0,
        }
        for reg in self.all_residencies():
            for ref, tier, vram in reg.snapshot():
                prev = merged.get(ref)
                if prev is None:
                    merged[ref] = (tier, int(vram))
                else:
                    best = tier if rank[tier] > rank[prev[0]] else prev[0]
                    merged[ref] = (best, prev[1] + int(vram))
        # Hold identity stable while emitting. Residency callbacks run only
        # after releasing their own lock, so this cannot invert lock order: a
        # transition either happens entirely before this snapshot, or its
        # identity update waits until the captured view is complete.
        with self._identity_lock:
            for ref, (tier, vram) in merged.items():
                # DISK is backed by the verified disk snapshot; RAM/VRAM is
                # backed by the loaded resident object. During stale A -> B
                # teardown those identities intentionally differ.
                resident = self._resident_identities.get(ref, ("", 0))
                identity = (
                    self._disk_identities.get(ref, resident)
                    if tier is residency_mod.Tier.DISK
                    else resident
                )
                digest, generation = identity
                out.append(pb.ModelResidency(
                    ref=ref,
                    tier=_TIER_TO_PB[tier],
                    vram_bytes=vram,
                    snapshot_digest=digest,
                    residency_generation=generation,
                ))
        return out

    def disk_usage_report(self) -> pb.DiskUsageReport:
        """Cached measured per-tier disk telemetry (pgw#610/th#962).

        Never measures directly — returns whatever
        :meth:`refresh_disk_usage_report` last computed. ``_state_delta()``
        calls this synchronously from many places (some with no event loop
        at all, e.g. the initial ``build_hello()``); it must never touch a
        filesystem. Empty/zeroed until the first refresh completes (boot's
        first StateDelta may ship no tiers — informational telemetry, never
        a dispatch gate on its own)."""
        return self._cached_disk_usage_report

    def _ref_blob_sizes(self, ref: str) -> Dict[str, int]:
        """CAS digest -> bytes for ``ref``'s banked snapshot, or ``{}`` when
        the worker has no manifest for it. The digest is the identity the CAS
        dedups on: ``blobs/`` is hardlinked into every snapshot tree, so a
        blob two refs share occupies the disk ONCE."""
        snap = self._snapshots.get(ref)
        if snap is None or not snap.files:
            return {}
        sizes: Dict[str, int] = {}
        for f in snap.files:
            # th#1303 S1: the tagged digest and nothing else. The legacy
            # `blake3` fallback was empty on every v2 entry, so this used to
            # bail to {} — sizes unknown — on exactly the manifests it was
            # written for. Zero entries is a REFUSAL ({}), never a silent 0.
            digest = str(getattr(f, "digest", "") or "").strip()
            if not digest:
                return {}
            sizes[digest.lower()] = int(f.size_bytes)
        return sizes

    def _reclaimable_entries(
        self, keep: set, entries: Dict[str, Any],
    ) -> List[Tuple[str, int]]:
        """(path, bytes) the disk GC could ACTUALLY free (th#1330 B4).

        The previous figure summed each evictable ref's whole indexed tree
        size, which over-reports twice: two evictable refs sharing a blob had
        it counted in both, and a blob an evictable ref shares with a RETAINED
        one is not reclaimable at all — ``sweep_orphan_blobs`` only unlinks
        blobs at ``st_nlink == 1``, so deleting that tree frees nothing.
        The hub sizes every capacity decision off this number.

        A ref with no banked manifest keeps its full indexed size: an unknown
        manifest is not a claim that the ref is free."""
        retained: set = set()
        for ref in self.disk_refs():
            if ref in keep or self.disk_ref_in_use(ref):
                retained.update(self._ref_blob_sizes(ref))
        counted: set = set()
        out: List[Tuple[str, int]] = []
        for ref in self.disk_refs():
            if ref in keep or self.disk_ref_in_use(ref):
                continue
            ent = entries.get(ref)
            if not ent:
                continue
            path = str(ent.get("path") or "")
            blobs = self._ref_blob_sizes(ref)
            if not blobs:
                out.append((path, int(ent.get("bytes") or 0)))
                continue
            freed = 0
            for digest, size in blobs.items():
                if digest in retained or digest in counted:
                    continue
                counted.add(digest)
                freed += size
            if freed > 0:
                out.append((path, freed))
        return out

    def _measure_disk_usage_report(self) -> pb.DiskUsageReport:
        """Blocking measurement — statvfs on the real mounts (CAS root =
        container tier; attached endpoint volume = volume tier; a shared
        NFS mount joins as TIER_NFS when the worker grows one) plus safely-
        reclaimable bytes: ref-index entries at DISK tier that are inactive
        AND not in the desired set — the disk-GC LRU's eligible set. Reuses
        ref-index bytes; never a tree rescan. capacity_generation bumps
        only on a measured shape change.

        Callers MUST run this off the event loop (``refresh_disk_usage_
        report``, or a thread pool in tests) — the attached VOLUME
        fill-source is a provider network mount that can stall for minutes
        under load; a blocking statvfs on the event loop thread freezes the
        whole worker, INCLUDING the th#965 heartbeat that shares the same
        loop (boothang: 0.40.7's post-seal_publish LTX hang)."""
        keep = set(self.keep)
        entries = self._index.entries()
        # pgw#748: the CAS is ONE tree with one page cache, hardlinked
        # across every group, so the preserve set is the UNION across groups —
        # dropping clean pages one group is done with would drop the pages a
        # sibling group is still mmapping (§4.3 caveat 3).
        reclaimable = self._reclaimable_entries(keep, entries)
        mounts = [disk_telemetry.MountSpec(
            tier=disk_telemetry.TIER_CONTAINER, path=str(self._cache_dir),
        )]
        if self._fill_source_dir is not None:
            mounts.append(disk_telemetry.MountSpec(
                tier=disk_telemetry.TIER_VOLUME, path=str(self._fill_source_dir),
            ))
        report = pb.DiskUsageReport(tiers=[
            pb.StorageTierUsage(
                tier=cast(Any, t.tier),  # proto enum value carried as int
                mount_path=t.mount_path,
                total_bytes=t.total_bytes, free_bytes=t.free_bytes,
                used_bytes=t.used_bytes, reclaimable_bytes=t.reclaimable_bytes,
            )
            for t in disk_telemetry.measure_tiers(mounts, reclaimable)
        ])
        shape = report.SerializeToString(deterministic=True)
        with self._disk_report_lock:
            if shape != self._last_disk_shape:
                self._last_disk_shape = shape
                self._disk_capacity_generation += 1
            report.capacity_generation = self._disk_capacity_generation
        return report

    async def refresh_disk_usage_report(self) -> pb.DiskUsageReport:
        """Off-loop refresh of the cached report (Lifecycle's TTL-gated
        refresh, driven off the heartbeat/state-delta path + once at boot).
        Never called from the hot StateDelta-build path — that path only
        reads the cache."""
        report = await asyncio.to_thread(self._measure_disk_usage_report)
        self._cached_disk_usage_report = report
        return report

    def local_path(self, ref: str) -> Optional[Path]:
        # Union across groups (pgw#748): the CAS is ONE hardlinked tree. A
        # group that has not yet booked this ref must still SEE the bytes a
        # sibling group already materialized.
        return self.disk_local_path(ref)

    def has_snapshot(self, ref: str) -> bool:
        """A digest-carrying snapshot for ``ref`` was seen this connection
        (gw#465): snapshot-less ops for it can still materialize the bytes."""
        return ref in self._snapshots

    def bank_snapshot(self, ref: str, snapshot: pb.Snapshot) -> None:
        """Make hub metadata available without starting a download."""
        if not ref or not snapshot.digest or not snapshot.files:
            return
        stored = pb.Snapshot()
        stored.CopyFrom(snapshot)
        with self._identity_lock:
            desired = self._desired_snapshot_identities.get(ref)
            generation = (
                desired[1]
                if desired is not None and desired[0] == stored.digest
                else 0
            )
            self._snapshots[ref] = stored
            self._snapshot_generations[ref] = max(0, int(generation))
        waiter = self._snapshot_waiters.get(ref)
        if waiter is not None:
            waiter.set()

    def replace_desired_snapshots(
        self, snapshots: Dict[str, pb.Snapshot], *, generation: int,
    ) -> None:
        """Atomically replace desired snapshot identity and bank its metadata.

        DesiredResidency is full-replacement state. Keeping this map separate
        from the last RunJob bank lets priority requests use older bytes
        without erasing the generation of bytes that remain desired, while a
        removal cannot resurrect an obsolete generation later.
        """
        accepted_generation = max(0, int(generation))
        stored: Dict[str, pb.Snapshot] = {}
        for ref, snapshot in snapshots.items():
            if not ref or not snapshot.digest or not snapshot.files:
                continue
            copy = pb.Snapshot()
            copy.CopyFrom(snapshot)
            stored[ref] = copy

        desired = {
            ref: (snapshot.digest, accepted_generation)
            for ref, snapshot in stored.items()
        }
        with self._identity_lock:
            self._residency_republish_epoch += 1
            self._desired_snapshot_identities = desired
            # Generations belong only to the current desired identity. Leave
            # actual resident identity untouched: those bytes may still be in
            # RAM/VRAM and must remain honestly observable until transitioned.
            for ref in self._snapshot_generations:
                self._snapshot_generations[ref] = 0
            for ref, snapshot in stored.items():
                self._snapshots[ref] = snapshot
                self._snapshot_generations[ref] = accepted_generation

        self._prune_banked_snapshots(stored)

        for ref in stored:
            waiter = self._snapshot_waiters.get(ref)
            if waiter is not None:
                waiter.set()

    def _prune_banked_snapshots(self, desired: Dict[str, pb.Snapshot]) -> None:
        """Drop banked manifests for refs that are neither desired, resident,
        in use, nor being materialized (th#1330 B5).

        ``_snapshots`` was append-only: a ref dropped from DesiredResidency
        kept its manifest forever, so a later bare ``ensure_local(ref)`` — a
        preload, a stale spec, a retry — could materialize OBSOLETE bytes off
        a manifest the hub stopped asking for, with no hub prompting and no
        way to notice. ``_verified``/``_snapshot_generations`` carried the same
        stale entries.

        The conditions are deliberately conservative: on disk, in use, or mid
        materialization all keep the manifest, so nothing in flight can lose
        the snapshot it is working from. A dropped ref that is wanted again
        goes through ``_await_hub_snapshot``, which is the correct path —
        the hub re-mints a manifest with LIVE presigned URLs."""
        try:
            resident = set(self.disk_refs())
        except Exception:  # pragma: no cover - residency not yet bound
            return
        active = set(self._materialize_active)
        keep = set(desired) | resident | active | set(self.keep)
        with self._identity_lock:
            stale = [
                ref for ref in list(self._snapshots)
                if ref not in keep and not self.disk_ref_in_use(ref)
            ]
            for ref in stale:
                self._snapshots.pop(ref, None)
                self._snapshot_generations.pop(ref, None)
                self._verified.discard(ref)
        if not stale:
            return
        logger.info(
            "dropped %d banked snapshot manifest(s) for refs that are neither "
            "desired nor resident (th#1330): %s",
            len(stale), ", ".join(sorted(stale)[:8]),
        )

    def snapshot_digest(self, ref: str, snapshot: Optional[pb.Snapshot] = None) -> str:
        candidate = snapshot
        if candidate is None:
            with self._identity_lock:
                candidate = self._snapshots.get(ref)
        return str(getattr(candidate, "digest", "") or "").strip()

    def resident_identity(self, ref: str) -> _ResidencyIdentity:
        with self._identity_lock:
            return self._resident_identities.get(ref, ("", 0))

    def _snapshot_identity(
        self, ref: str, snapshot: Optional[pb.Snapshot],
    ) -> _ResidencyIdentity:
        digest = self.snapshot_digest(ref, snapshot)
        if not digest:
            return ("", 0)
        with self._identity_lock:
            banked = self._snapshots.get(ref)
            generation = (
                self._snapshot_generations.get(ref, 0)
                if banked is not None and banked.digest == digest
                else 0
            )
        return (digest, generation)

    def _set_resident_identity(
        self, ref: str, identity: _ResidencyIdentity,
    ) -> bool:
        digest, generation = identity
        if not digest:
            return False
        exact = (str(digest).strip(), max(0, int(generation)))
        with self._identity_lock:
            changed = self._resident_identities.get(ref) != exact
            self._resident_identities[ref] = exact
        return changed

    def activate_disk_identity(self, ref: str) -> _ResidencyIdentity:
        """Make the verified disk snapshot the identity of a newly loaded
        RAM/VRAM instance immediately before its residency transition."""
        with self._identity_lock:
            identity = self._disk_identities.get(ref, ("", 0))
            if identity[0]:
                self._resident_identities[ref] = identity
            return identity

    async def _confirm_cached_identity(
        self, ref: str, identity: _ResidencyIdentity,
    ) -> None:
        """Publish exact identity when verified cached bytes satisfy the
        desired state without requiring a redundant download.

        pgw#628 (th#1070 residency v2): the emission is content-addressed and
        idempotent hub-side, so it is re-sent once per applied-HelloAck epoch
        even when the identity is unchanged — a re-received plan (redrive,
        overdue resend, reconnect) is the hub asking for a resync, and a
        worker that never re-announces can strand a lost success observation
        forever. Job-path calls within the same epoch remain deduped."""
        tier = self.residency.tier(ref)
        digest, _ = identity
        if not digest:
            return
        with self._identity_lock:
            self._disk_identities[ref] = identity
            current = self._resident_identities.get(ref, ("", 0))
        if tier is None:
            return
        # A newer tag may be on disk while an older pipeline remains loaded.
        # Do not relabel the loaded object; ensure_setup will vacate it before
        # serving the new snapshot.
        if tier in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM) and current[0] != digest:
            return
        changed = self._set_resident_identity(ref, identity)
        with self._identity_lock:
            epoch = self._residency_republish_epoch
            republish = self._identity_publish_epochs.get(ref) != epoch
            self._identity_publish_epochs[ref] = epoch
        if not changed and not republish:
            return
        state = {
            residency_mod.Tier.DISK: pb.MODEL_STATE_ON_DISK,
            residency_mod.Tier.RAM: pb.MODEL_STATE_IN_RAM,
            residency_mod.Tier.VRAM: pb.MODEL_STATE_IN_VRAM,
        }.get(tier)
        if state is None:
            return
        kw: Dict[str, Any] = {}
        if tier is residency_mod.Tier.VRAM:
            kw["vram_bytes"] = self.residency.vram_bytes(ref)
        await self._event(ref, state, identity=identity, **kw)

    def component_digests(self, ref: str, local_path: Optional[Path] = None) -> Dict[str, str]:
        """Per-component content identity of ``ref``'s snapshot (gw#479):
        ``{top_level_subfolder: content_set_digest}``. Weight/data files use
        the wire snapshot's per-file tagged digest; small JSON sidecars use
        CANONICAL digests read from ``local_path`` (save-era serialization —
        provenance stamps, explicit defaults, torch_dtype/dtype vocabulary —
        must not break sharing of byte-identical weights; see
        models/config_identity.py). Root-level files group under ``""``
        (never shared — model_index.json etc. differ per repo). Empty when
        no digest-carrying snapshot was seen — sharing stays off; weights
        are never hashed from disk."""

        snap = self._snapshots.get(ref)
        if snap is None:
            return {}
        groups: Dict[str, Dict[str, str]] = {}
        for f in snap.files:
            rel = str(f.path).strip().lstrip("/")
            # th#1303: this read `f.blake3`, which is EMPTY on every v2
            # entry, so every file of a v2 snapshot was skipped and component
            # sharing was silently OFF fleet-wide — the fail-CLOSED half of
            # the empty-guard class (the fail-open half is
            # `if want and got != want`). pgw#821 made it a dual-read; S1
            # retires the legacy mirror arm, leaving the tagged digest alone.
            digest = str(getattr(f, "digest", "") or "").strip()
            if not rel or not digest:
                continue
            comp, _, rest = rel.partition("/")
            if not rest:
                comp, rest = "", rel
            if (local_path is not None and comp
                    and rest.endswith(".json")
                    and int(f.size_bytes) <= CANONICAL_JSON_MAX_BYTES):
                canonical = canonical_json_digest(Path(local_path) / rel)
                if canonical:
                    digest = canonical
            groups.setdefault(comp, {})[rest] = digest
        return {c: residency_mod.content_set_digest(files)
                for c, files in groups.items()}

    def component_sizes(self, ref: str) -> Dict[str, int]:
        """Per-top-level-subfolder byte totals of ``ref``'s snapshot (gw#479):
        the make_room estimate for loading a subset of components."""
        snap = self._snapshots.get(ref)
        if snap is None:
            return {}
        sizes: Dict[str, int] = {}
        for f in snap.files:
            rel = str(f.path).strip().lstrip("/")
            if not rel:
                continue
            comp, _, rest = rel.partition("/")
            if not rest:
                comp = ""
            sizes[comp] = sizes.get(comp, 0) + int(f.size_bytes)
        return sizes

    # ---- disk retention (#370) ------------------------------------------------

    def rescan_disk(self) -> None:
        """Boot-time truth: re-register still-present downloads from the
        persisted ref index so Hello.models and GC see what disk holds.

        Also sweeps abandoned writer-unique CAS temp artifacts (th#850): on
        pod-local disk those died with the pod, but a CAS root pointed at a
        persistent volume keeps them until swept."""
        for ref, ent in self._index.entries().items():
            p = Path(str(ent.get("path") or ""))
            if p.exists():
                # Every group's registry learns the shared disk tier at boot.
                for reg in self.all_residencies():
                    if reg.tier(ref) is None:
                        reg.track_disk(ref, p)
            else:
                self._index.remove(ref)
        removed = disk_gc.sweep_stale_writer_temp(self._cache_dir)
        if removed:
            logger.info("disk-gc: swept %d abandoned writer temp artifact(s)", removed)

    def lru_disk_refs(self, *, exclude: Tuple[str, ...] = ()) -> List[str]:
        """Idle DISK refs in persisted last-use order, oldest first."""
        excluded = set(exclude)
        candidates = [
            (self._index.last_used(ref), ref)
            for ref in self.disk_refs()
            if ref not in excluded and not self.disk_ref_in_use(ref)
        ]
        candidates.sort()
        return [ref for _last_used, ref in candidates]

    def gc_disk(self, target_free_bytes: int, *, exclude: Tuple[str, ...] = ()) -> None:
        """Evict LRU disk-tier refs until free disk reaches the target.
        Non-keep refs go first (grace-honoring, then grace-ignoring); under
        keep-pressure the escape hatch evicts lowest-priority `keep` refs too
        (contract §7 — EVICTED is emitted so the hub re-downloads when demand
        returns).
        In-use / loaded refs are never touched."""
        keep = tuple(self.keep)
        keep_rank = {ref: index for index, ref in enumerate(keep)}
        for include_keep, honor_grace in (
            (False, True), (False, False), (True, False),
        ):
            for ref in self._gc_candidates(
                include_keep, honor_grace, exclude, keep, keep_rank
            ):
                if self._disk_free() >= target_free_bytes:
                    return
                self._evict_disk_ref(ref)

    def _gc_candidates(
        self,
        include_keep: bool,
        honor_grace: bool,
        exclude: Tuple[str, ...],
        keep: Tuple[str, ...],
        keep_rank: Dict[str, int],
    ) -> List[str]:
        """The evictable SET for one gc_disk pass: hard invariants only
        (never exclude/in-use — no policy ever overrides these), plus this
        pass's keep-membership/grace filter. Ordering within that set is a
        separate seam, see ``_disk_eviction_order``."""
        now = time.time()
        out: List[Tuple[float, str]] = []
        for ref in self.disk_refs():
            if ref in exclude or self.disk_ref_in_use(ref):
                continue
            if (ref in keep) != include_keep:
                continue
            last = self._index.last_used(ref)
            if honor_grace and (now - last) < _DISK_GC_GRACE_S:
                continue
            out.append((last, ref))
        return self._disk_eviction_order(out, include_keep, keep_rank)

    # th#850 managed-tier ruling (gw#599): the eviction POLICY (ranking one
    # pass's evictable set) is a distinct seam from the evictable SET itself
    # (``_gc_candidates`` above, which owns the hard never-evict invariants).
    # Default is the LRU-oldest-first/keep-priority-escape-hatch ordering
    # this store has always used — an instance may swap this attribute for a
    # scheduler-intent-aware policy without touching gc_disk's free-space
    # loop or the invariant filter. Building that policy is a follow-on
    # (Paul ruled the seam only here); the default below is exactly today's
    # behavior, byte-for-byte.
    @staticmethod
    def _default_disk_eviction_order(
        entries: List[Tuple[float, str]], include_keep: bool, keep_rank: Dict[str, int],
    ) -> List[str]:
        if include_keep:
            ordered = sorted(entries, key=lambda item: (-keep_rank[item[1]], item[0], item[1]))
        else:
            ordered = sorted(entries)
        return [ref for _, ref in ordered]

    _disk_eviction_order = _default_disk_eviction_order

    def _evict_disk_ref(self, ref: str) -> None:
        path = self.residency.local_path(ref) or self._index.path(ref)
        if not self.residency.evict(ref):  # refuses in-use entries; emits EVICTED
            return
        if path is not None:
            # th#1330 B4: snapshot trees are keyed by DIGEST, so two refs that
            # resolve to the same snapshot (a tag alias and its pin, the same
            # checkpoint reached under two spellings) share ONE directory.
            # rmtree-ing it here deleted the bytes a still-resident sibling ref
            # was pointing at. Drop only this ref's bookkeeping in that case;
            # the tree goes when its last holder does.
            sharer = self._other_ref_at_path(ref, path)
            if sharer:
                logger.info(
                    "disk-gc: keeping %s — %s still holds the same snapshot "
                    "tree", path, sharer,
                )
            else:
                disk_gc.delete_ref_bytes(ref, path, self._cache_dir)
                disk_gc.sweep_orphan_blobs(self._cache_dir)
        self._index.remove(ref)

    def _other_ref_at_path(self, ref: str, path: Path) -> str:
        """A still-tracked ref (any group) materialized at the same path."""
        target = str(path)
        for reg in self.all_residencies():
            for other in reg.refs_in(residency_mod.Tier.DISK):
                if other == ref:
                    continue
                other_path = reg.local_path(other)
                if other_path is not None and str(other_path) == target:
                    return other
        return ""

    async def _ensure_disk_headroom(
        self,
        ref: str,
        needed_bytes: int,
        identity: _ResidencyIdentity = ("", 0),
        *,
        intent_id: str = "",
    ) -> None:
        target = int(needed_bytes) + _DISK_GC_MARGIN_BYTES
        if self._disk_free() >= target:
            return
        await self._materialize_await(
            intent_id or self._materialize_intent(ref),
            asyncio.to_thread(self.gc_disk, target, exclude=(ref,)),
            operation=f"disk headroom for {ref}",
            status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
            stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_DISK_HEADROOM,
            reason=pb.LIFECYCLE_WAIT_REASON_DISK_HEADROOM,
        )
        free = self._disk_free()
        if free < target:
            await self._event(
                ref, pb.MODEL_STATE_FAILED,
                identity=identity, error="insufficient_disk",
            )
            raise InsufficientDiskError(
                f"need {needed_bytes} bytes for {ref}; {free} free after disk GC",
                available_bytes=free, required_bytes=needed_bytes,
                path=str(self._cache_dir),
            )

    # ---- ensure-local ----------------------------------------------------------

    def _lock(self, ref: str) -> asyncio.Lock:
        return self._locks.setdefault(ref, asyncio.Lock())

    def register_binding(self, ref: str, binding: Any) -> None:
        """Endpoint-spec binding for ``ref`` — supplies files/provider on
        download paths that only carry the bare ref (DesiredResidency or
        startup prefetch), so ``files=`` selections apply everywhere (#377)."""
        self._bindings.setdefault(ref, binding)

    def _override_excluded_components(
        self, ref: str, binding: Any, snapshot: Optional[pb.Snapshot],
    ) -> Tuple[str, ...]:
        """Base-composition subfolders this materialization must NOT fetch
        (th#1330 B2): the components a pgw#617 dispatch SUBSTITUTES.

        The override's own tree is materialized separately and handed to
        ``from_pretrained`` as a constructed object, so diffusers never reads
        the base's copy — it was downloaded and discarded (~1.64 GB per SDXL
        text-encoder override). The exclusion is derived only from the
        binding's ``component_overrides``, i.e. from the dispatch that is
        about to load, never from standing state.

        Only components the snapshot ACTUALLY carries as a subfolder are
        excluded, so the value is a fetch fact and not a guess — and a
        narrowed tree therefore keys on exactly what was left out."""
        overrides = _component_overrides(binding)
        if not overrides:
            return ()
        present = {
            str(f.path).strip().lstrip("/").partition("/")[0]
            for f in (snapshot.files if snapshot is not None else ())
            if "/" in str(f.path).strip().lstrip("/")
        }
        drop = tuple(sorted(
            {comp for comp, _ in overrides if comp in present}))
        if not drop:
            return ()
        saved = sum(
            int(f.size_bytes) for f in (snapshot.files if snapshot else ())
            if str(f.path).strip().lstrip("/").partition("/")[0] in drop
        )
        if self._override_exclusions_reported.get(ref) != drop:
            self._override_exclusions_reported[ref] = drop
            logger.info(
                "not fetching %s from %s: substituted by a component override "
                "(%d bytes skipped)", "/".join(drop), ref, saved,
            )
            activity_mod.emit_event(
                "component_fetch_skipped",
                f"base composition {ref} ships {'/'.join(drop)} that this "
                f"dispatch substitutes with a component override; skipping "
                f"{saved} bytes the load would discard (th#1330)",
                phase="skipped",
            )
        return drop

    async def _await_hub_snapshot(
        self,
        ref: str,
        *,
        intent_id: str = "",
    ) -> pb.Snapshot:
        """Cold tensorhub ref with no orchestrator-resolved snapshot: emit
        ``missing_snapshot`` (the hub refreshes desired state with fresh URLs
        on seeing it — connect_worker handleModelFailure) and block
        until that snapshot is banked (th#763). The bank site runs OUTSIDE
        the per-ref lock this coroutine holds, so the refreshed reconcile's
        ensure_local wakes us and then queues behind the lock. Raises
        :class:`MissingSnapshotError` when nothing arrives in
        ``_MISSING_SNAPSHOT_WAIT_S``."""
        snapshot = self._snapshots.get(ref)
        if snapshot is not None and snapshot.digest and snapshot.files:
            return snapshot
        waiter = self._snapshot_waiters.get(ref)
        if waiter is None:
            waiter = self._snapshot_waiters[ref] = asyncio.Event()
        await self._event(
            ref, pb.MODEL_STATE_FAILED,
            identity=("", 0), error="missing_snapshot",
        )
        logger.info("no snapshot for %s; waiting up to %.0fs for the hub re-mint",
                    ref, _MISSING_SNAPSHOT_WAIT_S)
        intent_id = intent_id or self._materialize_intent(ref)
        try:
            await self._materialize_await(
                intent_id,
                asyncio.wait_for(waiter.wait(), _MISSING_SNAPSHOT_WAIT_S),
                operation=f"snapshot resolution for {ref}",
                status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_SNAPSHOT,
                reason=pb.LIFECYCLE_WAIT_REASON_SNAPSHOT,
            )
        except asyncio.TimeoutError:
            raise MissingSnapshotError(
                f"tensorhub ref {ref!r} needs an orchestrator-resolved "
                f"snapshot; none arrived within {_MISSING_SNAPSHOT_WAIT_S:.0f}s "
                "of reporting missing_snapshot"
            ) from None
        finally:
            self._snapshot_waiters.pop(ref, None)
        snapshot = self._snapshots.get(ref)
        if snapshot is None or not snapshot.digest:
            raise MissingSnapshotError(
                f"tensorhub ref {ref!r} woke without a digest-carrying snapshot"
            )
        return snapshot

    async def ensure_local(
        self,
        ref: str,
        snapshot: Optional[pb.Snapshot] = None,
        *,
        binding: Any = None,
    ) -> Path:
        """Public path-only materialization API used by ordinary callers."""
        return (
            await self._materialize_local(
                ref,
                snapshot,
                binding=binding,
            )
        ).path

    async def _materialize_local(
        self,
        ref: str,
        snapshot: Optional[pb.Snapshot] = None,
        *,
        binding: Any = None,
        intent_id: str = "",
    ) -> _MaterializedLocal:
        """Materialize `ref` on disk. Transient failures retry with backoff;
        terminal (4xx-class) failures raise immediately. Emits ModelEvents.
        ``binding`` (when known) supplies provider + file-selection metadata;
        bare-ref callers fall back to the registered endpoint binding."""
        self.bind_loop()
        if binding is None:
            binding = self._bindings.get(ref)
        if snapshot is not None and snapshot.digest and snapshot.files:
            self.bank_snapshot(ref, snapshot)
        elif snapshot is None:
            snapshot = self._snapshots.get(ref)
        operation_identity = self._snapshot_identity(ref, snapshot)
        # th#1330 B2: the components this dispatch SUBSTITUTES are not fetched
        # from the base composition. The base is loaded with the override
        # object handed to `from_pretrained` (pgw#617 load-then-substitute),
        # so its own copy of that subfolder is downloaded and discarded.
        exclude_components = self._override_excluded_components(ref, binding, snapshot)
        registry = self._intent_registry
        scoped_intent_id = self._materialize_intent_context.get()
        command_owned = bool(intent_id or scoped_intent_id)
        intent_id = intent_id or scoped_intent_id
        blocker_intent_id = self._materialize_active.get(ref, "")
        if registry is None:
            intent_id = ""
        elif blocker_intent_id and not intent_id:
            task = asyncio.current_task()
            intent_id = registry.ensure_local_intent(
                "materialize-waiter",
                f"{ref}\0{id(task)}",
                detail=f"waiting to materialize {ref}",
            )
        elif not intent_id:
            intent_id = self._materialize_intent(ref)
        if registry is not None and not blocker_intent_id:
            self._materialize_active[ref] = intent_id
        failure_stage = pb.LIFECYCLE_INTENT_STAGE_WAIT_REF_LOCK
        acquired = False

        def complete(path: Path) -> _MaterializedLocal:
            # pgw#748: the bytes are pod-wide but each group keeps its own
            # ledger, so the group that asked must ALSO book the shared disk
            # entry — otherwise a group riding a sibling's materialization
            # never sees the ref in its own LRU, preserve set or eviction.
            if self.residency.tier(ref) is None:
                self.residency.track_disk(ref, path)
            if registry is not None:
                registry.transition(
                    intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                    pb.LIFECYCLE_INTENT_STAGE_ON_DISK,
                    actual_digest=operation_identity[0].encode(),
                )
            return _MaterializedLocal(path=path, identity=operation_identity)

        lock = self._lock(ref)
        try:
            await self._materialize_await(
                intent_id,
                lock.acquire(),
                operation=f"materialization ref lock for {ref}",
                status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_REF_LOCK,
                reason=pb.LIFECYCLE_WAIT_REASON_REF_LOCK,
                blocker_intent_id=blocker_intent_id,
            )
            acquired = True
            if registry is not None:
                self._materialize_active[ref] = intent_id
            failure_stage = pb.LIFECYCLE_INTENT_STAGE_VERIFYING
            if registry is not None:
                registry.transition(
                    intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                    pb.LIFECYCLE_INTENT_STAGE_VERIFYING,
                )
            # Union across groups: without this, group 1 loading a ref
            # group 0 already fetched sees `None` and re-runs the whole
            # download/verify — one pod, one copy, every group (pgw#748).
            cached = self.disk_local_path(ref)
            # A digest-carrying snapshot is authoritative: a cached
            # materialization of the SAME ref at a DIFFERENT digest is stale
            # (flavor re-published — e.g. compile-cache digest-change
            # re-adoption, e2e#117 live find #7) and must not short-circuit.
            want = ""
            if snapshot is not None and snapshot.digest:
                want = snapshot.digest.split(":", 1)[-1].strip().lower()
            # th#1330 B2: with an override exclusion the acceptable cached
            # names are the exclusion's own key OR the bare digest — the
            # latter is a SUPERSET (a complete tree already on disk serves an
            # excluded fetch for free, and is never narrowed retroactively).
            acceptable = {want}
            if want and exclude_components:
                acceptable.add(cozy_snapshot.snapshot_dir_key(
                    want, (), exclude_components))
            cached_partial = (
                cached is not None and want and cached.name != want
                and cached.name in acceptable
            )
            if cached is not None and cached.exists() and (not want or cached.name in acceptable):
                if ref in self._verified:
                    self._index.touch(ref)
                    await self._confirm_cached_identity(ref, operation_identity)
                    return complete(cached)
                # First use this boot: verify before trusting (gw#408). A
                # pod-churn-truncated snapshot used to fatal every load until
                # a manual delete; now it is quarantined + re-materialized.
                ok, bad = await asyncio.to_thread(
                    self._verify_snapshot_tree, cached,
                    _snapshot_without_components(snapshot, exclude_components)
                    if cached_partial else snapshot,
                )
                if ok:
                    self._verified.add(ref)
                    self._index.touch(ref)
                    await self._confirm_cached_identity(ref, operation_identity)
                    return complete(cached)
                logger.error(
                    "snapshot for %s failed first-use verification "
                    "(%d bad files); quarantining and re-materializing",
                    ref, len(bad),
                )
                # Quarantine emits EVICTED; the re-download below emits
                # DOWNLOADING/ON_DISK (or FAILED on a terminal error) — the
                # hub sees the true story, not a spurious FAILED.
                await asyncio.to_thread(self._quarantine_snapshot, ref, cached, bad)
                # fall through to a fresh download below
            if snapshot is None or not snapshot.digest:
                # Confident classification only (binding / boot provider
                # index) — unknown refs still flow to the download layer's
                # dispatch, which raises the same typed error terminally.
                prov = (getattr(binding, "source", None)
                        or lookup_provider_for_ref(ref, default=""))
                if prov == "tensorhub":
                    # The worker cannot resolve tensorhub-CAS refs itself
                    # (gw#465). Report missing_snapshot — the hub's re-mint
                    # trigger — then BLOCK until the re-minted DOWNLOAD
                    # banks a snapshot (th#763: a user request must never
                    # be the sacrificial cache warmer). No DOWNLOADING
                    # event, no retry burn; a hub that never answers raises
                    # the typed error (mapped RETRYABLE, never FATAL).
                    failure_stage = pb.LIFECYCLE_INTENT_STAGE_WAIT_SNAPSHOT
                    snapshot = await self._await_hub_snapshot(
                        ref,
                        intent_id=intent_id,
                    )
                    operation_identity = self._snapshot_identity(ref, snapshot)
            # th#1330 B2: every byte figure below (headroom gate, DOWNLOADING
            # totals, the boot weights span) counts what will actually be
            # fetched, so an override's skipped component never shows up as
            # bytes anybody planned for or reported.
            fetch_files = _snapshot_files_without_components(
                snapshot, exclude_components)
            if snapshot is not None and snapshot.files:
                # Sizes are known up front for tensorhub snapshots: gate on
                # disk headroom, GC-ing LRU refs first (#370).
                failure_stage = pb.LIFECYCLE_INTENT_STAGE_WAIT_DISK_HEADROOM
                await self._ensure_disk_headroom(
                    ref,
                    sum(int(f.size_bytes) for f in fetch_files),
                    operation_identity,
                    intent_id=intent_id,
                )
            last_progress = 0.0
            # th#850 managed-tier ruling (gw#599): opened before _progress so
            # its DOWNLOADING ticks can read the running total, and entered
            # once for the whole retry loop so it accumulates across
            # attempts. The hub (tensorhub th#850/PR#493) reads network_bytes
            # off the DOWNLOADING events' running value (mirrors
            # bytes_done/bytes_total), not just the terminal ON_DISK one —
            # both must carry it for the wire contract to actually work.
            net_scope = cozy_snapshot.NetworkBytesScope()

            # gw#621: per-ref bytes as a registry counter (visible on every
            # 10s beat while an activity is open); snapshot sizes make the
            # total known up front, so the wire never shows total=0 for
            # tensorhub refs.
            known_total = sum(int(f.size_bytes) for f in fetch_files)
            dl_counter = progress_mod.counter(
                f"download:{ref}", progress_mod.UNIT_BYTES, total=known_total)

            def _progress(done: int, total: Optional[int]) -> None:
                nonlocal last_progress
                dl_counter.set_done(float(done))
                if total:
                    dl_counter.set_total(float(total))
                now = time.monotonic()
                if now - last_progress < _PROGRESS_EVENT_MIN_INTERVAL_S:
                    return
                last_progress = now
                assert self._loop is not None
                asyncio.run_coroutine_threadsafe(
                    self._event(ref, pb.MODEL_STATE_DOWNLOADING,
                                identity=operation_identity,
                                bytes_done=int(done), bytes_total=int(total or 0),
                                network_bytes=net_scope.network_bytes),
                    self._loop,
                )

            await self._event(
                ref, pb.MODEL_STATE_DOWNLOADING, identity=operation_identity,
                bytes_total=known_total,
            )
            failure_stage = pb.LIFECYCLE_INTENT_STAGE_FETCHING
            if registry is not None:
                registry.transition(
                    intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                    pb.LIFECYCLE_INTENT_STAGE_FETCHING,
                    progress=pb.LifecycleProgress(
                        done=0,
                        total=float(known_total),
                        unit="bytes",
                    ),
                )
            # pgw#789: THE weights-fetch boot span. It lives here, not at a
            # caller, because this is the only layer that sees every
            # materialization path (startup prefetch, DesiredResidency
            # disk_refs, hot instances, RunJob delivery) AND the only layer
            # that knows where the bytes came from — net_scope separates a cold
            # R2 pull from a warm volume/CAS hit, which is the difference
            # between a 270s boot and a 40s one. Gated on `in_boot()` so a
            # steady-state materialization hours later does not land in the
            # boot ladder.
            fetch_span = (
                boot_mod.open_span(boot_mod.PHASE_WEIGHTS_FETCH, ref=ref)
                if boot_mod.in_boot() else None
            )
            fetch_exc: Optional[BaseException] = None
            fetch_bytes = 0
            try:
                delay = 1.0
                for attempt in range(1, _DOWNLOAD_RETRIES + 1):
                    try:
                        resolved = None
                        if snapshot is not None and snapshot.digest:
                            resolved = _snapshot_to_resolved(snapshot)
                        with net_scope:
                            path = await ensure_local(
                                ref,
                                provider=getattr(binding, "source", None),
                                snapshot=resolved,
                                cache_dir=self._cache_dir,
                                hf_home=self._hf_home,
                                hf_token=self._hf_token,
                                allow_patterns=tuple(getattr(binding, "files", ()) or ()),
                                components=tuple(getattr(binding, "components", ()) or ()),
                                exclude_components=exclude_components,
                                progress=_progress,
                                fill_source_dir=self._fill_source_dir,
                            )
                        tier_before = self.residency.tier(ref)
                        with self._identity_lock:
                            identity_changed = (
                                bool(operation_identity[0])
                                and self._disk_identities.get(ref) != operation_identity
                            )
                            if operation_identity[0]:
                                self._disk_identities[ref] = operation_identity
                                if tier_before in (None, residency_mod.Tier.DISK):
                                    self._resident_identities[ref] = operation_identity
                        # th#850 managed-tier ruling (gw#599): handed off to
                        # _on_residency_event so the ON_DISK event Residency
                        # emits for a genuinely fresh registration carries the
                        # bytes this materialization fetched over the network
                        # (0 included — pairs with the DOWNLOADING events'
                        # bytes_total for the "warm boot ⇒ ~0 R2 bytes" signal).
                        self._pending_network_bytes[ref] = net_scope.network_bytes
                        self.residency.track_disk(ref, path)
                        self._pending_network_bytes.pop(ref, None)  # defensive: unconsumed if no event fired
                        if tier_before is residency_mod.Tier.DISK and identity_changed:
                            # Residency suppresses same-tier event spam (track_disk
                            # above did not consume the pending value above). A
                            # digest move is nevertheless a semantic ON_DISK
                            # transition — carries network_bytes directly since
                            # this is our own explicit event, not Residency's.
                            await self._event(
                                ref, pb.MODEL_STATE_ON_DISK,
                                identity=operation_identity,
                                network_bytes=net_scope.network_bytes,
                            )
                        # tree_bytes stats every file — off-loop (gw#407: no
                        # multi-GB directory walks on the event loop).
                        size = await asyncio.to_thread(disk_gc.tree_bytes, path)
                        fetch_bytes = int(size)
                        self._index.record(ref, path, size)
                        # Fresh downloads were digest-verified by the downloader.
                        self._verified.add(ref)
                        return complete(path)
                    except Exception as exc:
                        terminal = _is_terminal_download_error(exc) or attempt >= _DOWNLOAD_RETRIES
                        if terminal:
                            vocab = self._error_vocab(exc)
                            if vocab == "download_failed":
                                # th#757: the generic bucket must carry the root
                                # cause — pods are often unreachable and the hub
                                # log is the only forensic surface (J24M run11:
                                # a starved request was undiagnosable hub-side).
                                vocab = f"download_failed: {_sanitize(f'{type(exc).__name__}: {exc}')[:200]}"
                            await self._event(
                                ref, pb.MODEL_STATE_FAILED,
                                identity=operation_identity, error=vocab,
                            )
                            raise
                        logger.warning(
                            "download of %s failed (attempt %d): %s; retrying in %.1fs",
                            ref,
                            attempt,
                            exc,
                            delay,
                        )
                        await self._materialize_await(
                            intent_id,
                            asyncio.sleep(delay),
                            operation=f"download retry backoff for {ref}",
                            status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                            stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_NETWORK_RETRY,
                            reason=pb.LIFECYCLE_WAIT_REASON_NETWORK_RETRY,
                            next_retry_at_unix_ms=(time.time_ns() // 1_000_000 + int(delay * 1000)),
                        )
                        if registry is not None:
                            registry.transition(
                                intent_id,
                                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                                pb.LIFECYCLE_INTENT_STAGE_FETCHING,
                            )
                        delay *= 4
                raise RuntimeError("unreachable")
            except BaseException as exc:
                fetch_exc = exc
                raise
            finally:
                dl_counter.finish()
                if fetch_span is not None:
                    net = int(net_scope.network_bytes)
                    if net > 0:
                        source = boot_mod.SOURCE_R2
                    elif known_total > 0:
                        # A CAS snapshot materialized with zero network bytes:
                        # every blob was already under blobs_root (local CAS) or
                        # came off the endpoint's warm datacenter volume.
                        source = (
                            boot_mod.SOURCE_VOLUME if self._fill_source_dir
                            else boot_mod.SOURCE_LOCAL
                        )
                    else:
                        # No snapshot: a provider-direct pull into the HF cache.
                        source = boot_mod.SOURCE_HF_CACHE
                    fetch_span.bytes_moved(net or known_total or fetch_bytes, source)
                    fetch_span.note(
                        f"ref={ref} net_bytes={net} manifest_bytes={known_total} "
                        f"tree_bytes={fetch_bytes} "
                        f"fill={'yes' if self._fill_source_dir else 'no'}"
                    )
                    fetch_span.close(fetch_exc)
        except asyncio.CancelledError:
            if registry is not None:
                if command_owned:
                    registry.transition(
                        intent_id,
                        pb.LIFECYCLE_INTENT_STATUS_WAITING,
                        pb.LIFECYCLE_INTENT_STAGE_WAIT_TENANT_IDLE,
                        reason=pb.LIFECYCLE_WAIT_REASON_TENANT_WORK,
                        detail="materialization preempted by tenant work",
                    )
                else:
                    registry.transition(
                        intent_id,
                        pb.LIFECYCLE_INTENT_STATUS_CANCELED,
                        failure_stage,
                        detail="materialization canceled",
                    )
            raise
        except BaseException as exc:
            if registry is not None:
                registry.transition(
                    intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_FAILED,
                    failure_stage,
                    error_code=(
                        pb.LIFECYCLE_ERROR_CODE_SNAPSHOT_IDENTITY_MISSING
                        if isinstance(exc, MissingSnapshotError)
                        else pb.LIFECYCLE_ERROR_CODE_UNSPECIFIED
                    ),
                    detail=_sanitize(str(exc))[:512],
                )
            raise
        finally:
            if self._materialize_active.get(ref) == intent_id:
                self._materialize_active.pop(ref, None)
            if acquired:
                lock.release()

    def activate_load_identity(
        self, ref: str, identity: _ResidencyIdentity,
    ) -> _ResidencyIdentity:
        """Promote the exact bytes used by one setup, never current disk state."""
        if identity[0]:
            self._set_resident_identity(ref, identity)
            return identity
        return self.activate_disk_identity(ref)

    # ---- snapshot integrity (gw#408) -------------------------------------------

    def _verify_snapshot_tree(
        self, path: Path, snapshot: Optional[pb.Snapshot]
    ) -> Tuple[bool, List[str]]:
        """Integrity of a materialized snapshot (worker thread; blocking IO).

        With a resolved manifest every regular file is checked against its
        declared size AND their CONTENT DIGEST, hashed under the algorithm the
        manifest named; files the manifest cannot cover (reassembled chunked
        originals, merged single-file checkpoints) plus manifest-less trees
        (hf/civitai) get the structural safetensors check (header parses +
        every declared tensor byte present). Returns ``(ok, bad_digests)`` —
        the digests name blobs to quarantine."""

        p = Path(path)
        bad: List[str] = []
        covered: set[Path] = set()
        files = list(snapshot.files) if snapshot is not None else []
        if files and p.is_dir():
            # pgw#769/#781 (th#1303): the hash algorithm comes from the DIGEST,
            # never from this call site. This used to read `f.blake3` and hash
            # with blake3 -- but under manifest v2 that field is EMPTY, so
            # `digest` was "" and BOTH the size and hash checks were skipped
            # (the legacy fallback is gone at S1). The tree
            # was then reported CLEAN WITHOUT BEING HASHED. On a volume shared
            # across releases and pods that is a security hole, not a cosmetic
            # gap, and it is the same false-clean shape as reading
            # manifest["files"] when the key is "entries": a verifier that
            # examines nothing looks exactly like one that passes.
            targets, skipped = snapshot_verify_targets(files, p)
            for rel in skipped:
                try:
                    covered.discard(p / _norm_rel_path(rel))
                except ValueError:
                    pass
            for t in targets:
                covered.add(t.path)
            if targets:
                rep = verify_files(targets, blobs_root=str(p.parent))
                bad.extend(rep.bad)
                for finding in rep.findings:
                    logger.warning("snapshot %s: %s", p.name, finding)
                # DENOMINATOR GUARD, and it applies only to an otherwise-CLEAN
                # report: a verdict that found nothing wrong is trustworthy only
                # if it actually read the bytes. `examined` must cover every
                # target handed in, and a clean run that neither hashed nor
                # memo-hit anything read nothing at all. (A report that already
                # names bad files is not vacuous -- it did its job, and folding
                # it in here would double-report the same digest.)
                vacuous = (
                    not rep.bad
                    and not rep.findings
                    and rep.hashed == 0
                    and rep.memo_hits == 0
                )
                if rep.examined != rep.expected or vacuous:
                    logger.error(
                        "snapshot %s verification is not trustworthy: examined=%d "
                        "expected=%d hashed=%d memo=%d bytes=%d -- treating as corrupt",
                        p.name, rep.examined, rep.expected, rep.hashed,
                        rep.memo_hits, rep.bytes_hashed,
                    )
                    already = set(bad)
                    bad.extend(t.ref for t in targets if t.ref not in already)
        try:
            candidates = [p] if p.is_file() else sorted(p.rglob("*.safetensors"))
        except OSError:
            candidates = []
        for st in candidates:
            if st in covered or st.suffix != ".safetensors":
                continue
            if not safetensors_file_valid(st):
                logger.warning("snapshot file %s structurally invalid (truncated?)", st)
                bad.append(str(st.relative_to(p)) if st != p else st.name)
        return (not bad, bad)

    def _quarantine_snapshot(self, ref: str, path: Path, bad: List[str]) -> None:
        """Evict + delete a corrupt materialization AND the corrupt blobs it
        was built from, so re-materialization re-downloads instead of
        re-linking the same bad bytes. Emits EVICTED via residency."""

        self._verified.discard(ref)
        self.residency.evict(ref, force=True)
        disk_gc.delete_ref_bytes(ref, Path(path), self._cache_dir)
        delete_blobs(self._cache_dir, [d for d in bad if "/" not in d and "." not in d])
        disk_gc.sweep_orphan_blobs(self._cache_dir)
        self._index.remove(ref)

    async def refetch_corrupt(
        self, ref: str, snapshot: Optional[pb.Snapshot] = None, *, binding: Any = None
    ) -> Optional[Path]:
        """Load-failure path (gw#408): a weights load failed with a
        corruption-shaped error — digest-verify the snapshot. A clean tree
        returns None (the failure is NOT corruption; caller re-raises); a
        dirty tree is quarantined and re-materialized, returning the fresh
        path for exactly one load retry."""
        path = self.residency.local_path(ref) or self._index.path(ref)
        if path is None:
            return None
        async with self._lock(ref):
            ok, bad = await asyncio.to_thread(self._verify_snapshot_tree, Path(path), snapshot)
            if ok:
                self._verified.add(ref)
                return None
            logger.error(
                "load failure traced to corrupt snapshot for %s (%d bad files); "
                "quarantining and re-materializing", ref, len(bad),
            )
            await asyncio.to_thread(self._quarantine_snapshot, ref, Path(path), bad)
        return await self.ensure_local(ref, snapshot, binding=binding)

    @staticmethod
    def _error_vocab(exc: BaseException) -> str:
        if isinstance(exc, MissingSnapshotError):
            return "missing_snapshot"
        if isinstance(exc, UrlExpiredError):
            return "url_expired"
        if isinstance(exc, InsufficientDiskError):
            return "insufficient_disk"
        text = str(exc).lower()
        if "expired" in text or "403" in text:
            return "url_expired"
        if "digest" in text or "hash" in text:
            return "digest_mismatch"
        if "no space" in text or "disk" in text:
            return "insufficient_disk"
        return "download_failed"


# pgw#686: base lanes speculated for pre-load pull-by-key cell lookups (no
# pipeline exists yet to probe). Must cover every base lane a loader can
# leave a pipeline on, or a cold worker can never pull the very cell its own
# boot would mint — the ie#546 burst published on "w8a8" while lookups
# speculated only ""/"fp8-hooks", so the armed cell was unreachable and all
# 9 workers re-minted. Verify-on-receipt remains the arming gate; a wrong
# speculation is only ever a benign extra lookup key.
#
# pgw#918: this was a SECOND authored copy of the lane vocabulary and it was
# missing "w4a4" and "svdq-native" — the identical ie#546 defect, unrepaired
# for two more lanes, in the constant whose own comment describes the
# incident. It is now the loader's own list, so a new lane cannot be stamped
# without appearing here.
_SPECULATIVE_CELL_BASE_EXECUTION_LANES: Tuple[str, ...] = loading.STAMPABLE_BASE_EXECUTION_LANES


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
    # th#883/gw#581 pull-by-key: the exact cell key this live object's
    # runtime computed for itself (gen_worker.cell_key), plus its axes for
    # the wire. "" when a required axis is unavailable (no cell identity).
    requested_cell_key: str = ""
    requested_cell_axes: Tuple[Tuple[str, str], ...] = ()
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
    # The operation that most recently certified the active cell. Boot-
    # attached cells have no ModelOp and therefore leave this empty rather
    # than fabricating causal failure evidence later.
    active_adoption_operation_id: str = ""
    # Runtime guard failures quarantine immutable cells on this exact
    # incarnation. Successful adoption of B must not clear an earlier failure
    # of A; only a newly minted target gets a fresh quarantine set.
    failed_compile_identities: set[Tuple[str, str]] = dc_field(
        default_factory=set)


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

    ``mint`` is either a finalized ``fleet_cells.SelfMint`` (artifact
    already packed, digest known) or a ``fleet_cells.PendingSelfMint``
    (gw#587 CORRECT FIX: armed for capture, not yet proven or packed — its
    ``target`` path exists on disk only once the warmup proof finalizes
    it, and its ``snapshot_digest`` is empty until then). Either way the
    ``ref`` is known immediately (computed from static axes, never the
    traced FX graph bytes), which is what the hub's self-attested dispatch
    fence needs at advertise time; the proof loop replaces this selection
    with the fully finalized one once it packs the proven capture.
    """
    if mint is not None:
        if (getattr(mint, "recipe", "") == "aot"
                and getattr(mint, "artifact", None) is None):
            # pgw#805: an AOT cell's key folds the COMBINED GRAPH HASH of the
            # exported class set, so it does not exist until the export
            # finishes. The dynamo pending's key is computable from static
            # axes and is therefore honest to advertise at arm time; an AOT
            # pending's is not, and advertising the dynamo-shaped handle would
            # publish a self-attested ref no artifact will ever carry. Nothing
            # is advertised until `adopt_delegated_mint` reads the real key
            # off the packed envelope.
            return delivered
        path = getattr(mint, "artifact", None)
        if path is None:
            path = mint.target
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
    #: pgw#844: per object, the aliases that proved SOME but not all of their
    #: declared graph classes on the EXPORTED lane -> the classes that stayed
    #: eager. Non-empty means "compiled for these shapes, eager for those",
    #: which is a serving posture, not a boot failure.
    partial_by_object: Dict[int, Dict[str, Tuple[str, ...]]] = dc_field(
        default_factory=dict)
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
    # call-time-owned (ExecutionLaneGate promotes + pins around each pipeline call);
    # the executor must neither whole-job-pin nor eagerly promote them, or
    # the idle sibling can never be LRU-swapped out.
    execution_lane_refs: set = dc_field(default_factory=set)
    # pgw#572: exact compile-capable objects owned by this READY record. The
    # IDs are minted after successful setup and cleared before vacate; they do
    # not derive from mutable refs, authored specs, or object memory addresses.
    compile_targets: Dict[str, _CompileTargetRecord] = dc_field(default_factory=dict)
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


class _MintDeclined(Exception):
    """pgw#737: the mint refused itself — it cannot capture on this card
    without taking the tenant down with it. NOT a failure: serving is eager,
    the cell stays absent, and a roomier config mints it later."""

    def __init__(
        self, reason: str, budget: "mint_budget.MintBudget",
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.budget = budget
        line = budget.line("mint_skipped", reason)
        super().__init__(f"{line}; {detail}" if detail else line)


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
    snapshots: Optional[Dict[str, "pb.Snapshot"]]
    # id(pipeline) -> fleet_cells.PendingSelfMint (same objects the arming
    # scope produced; shared captures keep their sharing structure).
    pendings: Dict[int, Any]
    # id(pipeline) -> the actual pipeline object (id() keys alone cannot
    # keep the object alive or recover it).
    pipes: Dict[int, Any]
    # id(pipeline) -> arm-time placeholder selection (claimed key ref,
    # digest empty until finalize) stashed out of the foreground install.
    selections: Dict[int, "_CompileArtifactSelection"]
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
    # composition, in ONE value per slot (pgw#974, `mint_process.MintSlot`).
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
    keys = sorted({str(getattr(p, "cell_key", "")) for p in bg.pendings.values()})
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


def _canonical_host_ram_refs(refs: typing.Iterable[str]) -> List[str]:
    """Keep only canonical model refs suitable for protocol evidence."""
    return list(dict.fromkeys(
        ref
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
    # gw#551: slots whose pipeline __call__ the ExecutionLaneGate wrapped. Only these
    # may become call-time-owned; an un-gateable pipeline (no instance
    # __call__) keeps the eager whole-job pin + promote path.
    gated_slots: set = dc_field(default_factory=set)
    # Actual worker-constructed pipelines whose declared compile targets
    # resolve. Kept separately because shared-lane residency may replace the
    # bookkeeping object with a ModuleDict while setup receives the pipeline.
    compile_objects: List[_CompileObjectCandidate] = dc_field(default_factory=list)
    # id(pipeline) -> exact attached artifact that successfully armed it.
    # Installed only after the setup warmup completes.
    active_compile_artifacts: Dict[int, _CompileArtifactSelection] = dc_field(
        default_factory=dict)
    trt_execution_before: Dict[int, int] = dc_field(default_factory=dict)
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
    # th#913/gw#596: the CONCRETE lane serving this job (stamped post-setup,
    # reported on JobMetrics.lane). "" = not yet determined.
    execution_lane: str = ""
    # pgw#789 (th#1293 dimensions): this request was served EAGER by a compiled
    # lane — a pgw#680 guard miss, a router heal/volatile verdict, or an
    # aot_serve ingress refusal. Set from the guard-miss callback, which fires
    # DURING the request and names it via postmortem.current_inflight_request().
    # Without it a fallback sample reports lane=...+compiled and silently
    # contaminates every compiled-vs-eager latency comparison with eager data.
    served_eager_fallback: bool = False
    fallback_reason: str = ""
    # pgw#789: (steps, width, height) of the EXECUTED payload, defaults
    # applied — the axes latency is a function of. Stamped beside `lane`,
    # where the resolved payload is in scope; 0 means "not applicable"
    # (non-spatial function), never "zero".
    shape: Tuple[int, int, int] = (0, 0, 0)
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

    __slots__ = ("depth", "_holds", "_next_token", "transitions")

    def __init__(self, depth: int) -> None:
        self.depth = max(1, int(depth))
        self._holds: Dict[int, Dict[int, _PermitHold]] = {}
        self._next_token = 0
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
        self._holds.setdefault(id(sem), {})[token] = _PermitHold(label, task)
        self.transitions += 1
        return token

    def drop(self, sem: asyncio.Semaphore, token: int) -> None:
        if self._holds.get(id(sem), {}).pop(token, None) is not None:
            self.transitions += 1

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
        # bootstrap, never inside per-instance endpoint setup. Largely moot
        # on the bf16 compute path (it affects residual fp32 matmuls only).
        if torch is not None and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
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
        # Compile-cache adoption mutates already-resident modules in place.
        # Serialize the whole operation (download through terminal evidence),
        # not only its GPU warmup, so two commands can never cross wraps or
        # let an older rollback mutate a newer adoption.
        self._compile_cache_adoption_lock = asyncio.Lock()
        self._compile_cache_adoption_active = ""
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
        self._host_ram_blocks: Dict[str, _HostRamBlock] = {}
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
        self._last_gpu_info: Optional[Dict[str, Any]] = None
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

    def _adoption_intent(self, op: pb.ModelOp) -> str:
        registry = self.intent_registry
        if registry is None:
            return ""
        return registry.intent_id(
            pb.DESIRED_INTENT_KIND_COMPILE_ADOPT,
            ref=op.ref,
        ) or registry.ensure_local_intent(
            "compile-adopt",
            op.operation_id or op.ref,
            detail=f"adopt compile artifact {op.ref}",
        )

    def _job_intent(self, run: pb.RunJob) -> str:
        registry = self.intent_registry
        if registry is None:
            return ""
        return registry.ensure_local_intent(
            "job",
            f"{run.request_id}\0{run.attempt}",
            function_name=run.function_name,
            detail=f"run request {run.request_id} attempt {run.attempt}",
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
                if self._record_in_use(rec):
                    return
                await self._vacate_record(rec)
        self._on_state_change()

    async def revalidate_snapshot_identity(
        self, ref: str, snapshot: Optional[pb.Snapshot],
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

    def gate_functions(self, gpu_info: Dict[str, Any]) -> None:
        """Run hardware gates; populate self.unavailable + self.serve_plans.

        th#683 P3 — the worker NEVER hard-refuses a function on the
        recommended-VRAM hint. Genuine incompatibilities (compute capability /
        missing quant library / a stored flavor outside its SM window) still
        gate a function off; everything else is an ADAPTIVE FIT: the function
        serves by the best available means (native -> runtime fp8 storage ->
        emergency 4-bit -> CPU/disk offload -> CPU-only) and records an honest
        advisory. Needing offload/CPU is NEVER a refusal (Paul's ruling
        2026-07-10: gen workers offload out of necessity, not preference —
        better to run degraded than not run). The only opt-out is the
        author's own ``Resources(strict_vram=True)`` for bindings that
        cannot tolerate CPU-resident weights. Every degraded serve is
        reported structurally (FnDegraded) so the orchestrator can move the
        release to a bigger card.
        """

        # Idempotent re-gate (gw#494): drop only the marks THIS gate made
        # last time; setup failures and other owners survive. Remember the
        # probe so apply_model_resolutions can re-run us.
        self._last_gpu_info = dict(gpu_info)
        for fn in self._gate_owned:
            self.unavailable.pop(fn, None)
        self._gate_owned = set()

        total_vram_gb = float(gpu_info.get("gpu_total_mem") or 0) / (1024 ** 3)
        free_vram_gb = float(gpu_info.get("gpu_free_mem") or gpu_info.get("gpu_total_mem") or 0) / (1024 ** 3)
        detected_sm = str(gpu_info.get("gpu_sm") or "")
        libs = {str(x) for x in (gpu_info.get("installed_libs") or [])}
        caps = TensorhubWorkerCapabilities(
            cuda_version=str(gpu_info.get("cuda_version") or ""),
            gpu_sm=int(detected_sm) if detected_sm.isdigit() else 0,
            torch_version=str(gpu_info.get("torch_version") or ""),
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
            # card class is the fit ladder's decision (sdxl runs fine in
            # fp16 on sm75); only stored-flavor SM windows (svdq/nvfp4,
            # via variant_fit) remain genuinely hard.
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

            # Adaptive serve-time fit for the VRAM / GPU-presence / stored-
            # flavor dimensions. The primary binding carries the flavor token
            # (#fp8 / #nvfp4 / #svdq-*) whose SM window variant_fit gates.
            primary = next(iter(spec.models.values()), None)
            plan = plan_serve(r, caps, free_vram_gb, binding=primary)
            self.serve_plans[name] = plan
            self._gate_serve_plans[name] = plan
            if not plan.serveable:
                if plan.run_mode in (RUN_CPU, RUN_OFFLOAD):
                    # The author's strict_vram opt-out of the CPU-touching
                    # rungs: on a GPU-less host that reads as no-CUDA, on a
                    # too-small card as a VRAM shortfall.
                    code = "cuda_unavailable" if plan.run_mode == RUN_CPU else "insufficient_vram"
                elif plan.fit == FIT_INCOMPATIBLE:
                    # A stored flavor outside its hardware window (fp8 /
                    # nvfp4 / svdq SM gates, quant stack pins).
                    code = "compute_capability_unmet"
                else:
                    code = "insufficient_vram"
                self.unavailable[name] = (
                    code, plan.reason,
                    {"detected_vram_gb": f"{total_vram_gb:.0f}",
                     "recommended_vram_gb": (
                         f"{r.vram_gb_hint:.0f}" if r.vram_gb_hint else "")})
                self._gate_owned.add(name)
                continue
            if plan.degraded:
                logger.warning(degraded_log_line(
                    event="planned", fn=name, phase="gate",
                    from_rung=plan.wanted, to_rung=plan.ran or plan.run_mode,
                    free_gb=free_vram_gb,
                    detail=f"~{plan.est_latency_multiplier:.1f}x latency: {plan.warning}",
                ))

    def _record_demotion(
        self,
        spec: EndpointSpec,
        *,
        ref: str,
        phase: str,
        from_rung: str,
        to_rung: str,
        needed_gb: float = 0.0,
        detail: str = "",
    ) -> None:
        """One ladder-demotion bookkeeper (gw#463): learned per-ref floor +
        updated ServePlan + loud DEGRADED_MODE warning + FnDegraded re-emit
        via the state-delta path."""

        if ref:
            self.degraded_floor[ref] = deeper_offload_mode(
                self.degraded_floor.get(ref, ""), to_rung)
        line = degraded_log_line(
            event="engaged", fn=spec.name, model=ref, phase=phase,
            from_rung=from_rung, to_rung=to_rung,
            needed_gb=needed_gb, free_gb=get_available_vram_gb(),
            detail=(detail or "CUDA OOM") + " — sticky for this worker until "
                   "reload; fix capacity/config, do not rely on this mode",
        )
        logger.warning(line)
        self.serve_plans[spec.name] = demoted(
            self.serve_plans.get(spec.name), detail=line, placement_mode=to_rung)
        self._on_state_change()

    def _record_adaptive_rung(self, spec: EndpointSpec, *, ref: str,
                              rung: str, detail: str) -> None:
        """gw#491: the load-time adaptive fit ladder engaged an emergency
        rung (runtime fp8 storage / nf4). Surface it exactly like the
        plan-time rungs — updated ServePlan + FnDegraded via the state-delta
        path — never as a log-line-only fallback."""

        logger.warning(
            "LOAD_RUNG_ENGAGED fn=%s model=%s rung=%s detail=%s",
            spec.name, ref, rung, detail)
        self.serve_plans[spec.name] = load_rung_engaged(
            self.serve_plans.get(spec.name), rung=rung, detail=detail)
        self._on_state_change()

    def _record_cast_drop(self, spec: EndpointSpec, *, ref: str,
                          wanted: str, detail: str, ran: str = "bf16") -> None:
        """th#737: a resolved cast (storage_dtype) cannot apply — the
        pipeline has no denoiser/cast surface. Serve at base precision but
        surface it STRUCTURALLY (FnDegraded wanted=fp8 ran=bf16 via the
        state-delta path), never as a silent log-line fallback: the recipe
        budgeted the cast's VRAM headroom."""

        logger.warning(
            "CAST_DROPPED fn=%s model=%s wanted=%s ran=%s detail=%s",
            spec.name, ref, wanted or "fp8", ran or "bf16", detail)
        self.serve_plans[spec.name] = cast_dropped(
            self.serve_plans.get(spec.name), wanted=wanted, detail=detail,
            ran=ran)
        self._on_state_change()

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
        requested_key = ""
        requested_axes: Tuple[Tuple[str, str], ...] = ()
        try:

            # pgw#686: the KEY lane resolves through the one shared brain
            # (compile_cache.cell_base_execution_lane — probe, then denoiser markers),
            # never the raw probe alone: the raw probe is blind to the w8a8
            # GEMM mode, so its key names a cell no mint ever publishes and
            # the fleet's armed cells are never adopted. The raw probe stays
            # authoritative for the wire lane descriptor below.
            key = cell_key.compute(
                str(getattr(cfg, "family", "") or ""),
                compile_cache.cell_base_execution_lane(target.pipeline), bucket,
                contract=cfg.contract_digest(),
                regional=bool(getattr(cfg, "regional", False)))
            requested_key, requested_axes = key.digest, key.axes
        except Exception:
            pass  # no computable identity on this runtime => no key
        with target.state_lock:
            target.pipeline_weight_lane = execution_lane
            target.lora_bucket = bucket
            target.contract_digest = contract_digest
            target.requested_cell_key = requested_key
            target.requested_cell_axes = requested_axes

    @staticmethod
    def _warn_cell_key_divergence(spec_name: str, target: "_CompileTargetRecord") -> None:
        """pgw#686 invariant: a SELF-MINTED active cell must carry exactly
        the key an identical worker would REQUEST (requested_cell_key) —
        anything else is fleet-wide silent zero-adoption: the published
        cell sits armed in the store while every cold pod re-mints (the
        ie#546 burst: 10 pods, 9 simultaneous mints, 0 adoptions). Loud
        and greppable; never fatal to serving."""
        with target.state_lock:
            active = str(target.active_compile_ref or "")
            self_mint = bool(target.active_self_mint)
            requested = str(target.requested_cell_key or "")
        if not (self_mint and active and requested):
            return
        if active.rpartition("#")[2] == requested:
            return
        logger.error(
            "cell_key_divergence: self-minted active cell %s is not this "
            "runtime's requested key %s for %s — the published cell can "
            "never be adopted by an identical worker; the lane/axis probes "
            "disagree (pgw#686)", active, requested, spec_name)

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
        failed identity is quarantined (per-target AND process-wide) so it is
        never re-adopted or re-minted this boot.
        """
        if rec.compile_targets.get(target.incarnation_id) is not target:
            raise RuntimeError("compiled target is no longer live")
        with target.state_lock:
            if not (
                target.active_compile_ref
                or target.active_compile_snapshot_digest
            ):
                return
            failed_ref = target.active_compile_ref
            failed_digest = target.active_compile_snapshot_digest
            operation_id = target.active_adoption_operation_id
            target.failed_compile_identities.add((failed_ref, failed_digest))
            target.active_compile_ref = ""
            target.active_compile_snapshot_digest = ""
            target.active_adoption_operation_id = ""

        compile_cache.record_cell_quarantined(failed_ref)
        logger.warning(
            "compile target %s runtime guard tripped; compiled proof revoked, "
            "serving degrades to explicit eager: %s",
            target.incarnation_id,
            detail,
        )
        self._signal_state_change_threadsafe()
        if operation_id:
            # State revocation above is synchronous and wins every local
            # dispatch race. The causal terminal event is delivered on the
            # executor loop and may arrive before or after the StateDelta.
            loop = self._loop
            if loop is None or loop.is_closed():
                raise RuntimeError(
                    "cannot deliver causal compile-runtime failure: "
                    "executor loop is unavailable"
                )
            event = pb.WorkerMessage(model_event=self._adoption_event(
                failed_ref,
                pb.MODEL_STATE_FAILED,
                failed_digest,
                operation_id,
                target.incarnation_id,
                error="adopt_failed:runtime_guard",
            ))

            def send_failure() -> None:
                async def deliver() -> None:
                    await self._send(event)

                task: asyncio.Task[None] = asyncio.create_task(
                    deliver(),
                    name=f"compile-runtime-failed-{target.incarnation_id}",
                )

                def log_delivery(done: asyncio.Task[None]) -> None:
                    if done.cancelled():
                        return
                    error = done.exception()
                    if error is not None:
                        logger.error(
                            "causal compile-runtime failure delivery failed",
                            exc_info=error,
                        )

                task.add_done_callback(log_delivery)

            loop.call_soon_threadsafe(send_failure)

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

        if trt_engine.set_guard_failure_callback(target.pipeline, callback):
            return True
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
        # target (telemetry only — no state mutation, no revocation). TRT
        # engines never dynamo-recompile, so only torch guards bind this.
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
    ) -> None:
        """pgw#824: record (and, once, confess) WHY this record has no cell.

        First token wins: the earliest honest cause outranks a later generic
        one. The typed event fires only on the transition, so a decline that
        happens per-object on a many-slot record coalesces to ONE row instead
        of N identical ones — counts, not silence, and not a flood.
        """
        token = str(token or "").strip()
        if not token or rec.eager_posture:
            return
        rec.eager_posture = token
        activity_mod.emit_event(
            "serve_eager_posture",
            f"fn={','.join(s.name for s in rec.specs) or '?'}: this instance "
            f"serves EAGER — {detail or token}. Every request it serves "
            f"reports fallback_reason={token}.",
            phase=token,
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
            permitted_names = (
                function_proofs[id(pipeline)]
                if id(pipeline) in function_proofs
                else contract_names
            )
            object_proven_by_custom_warmup = bool(
                spec.cls is not None
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
                    "%s compile target for %r has no proven active Forge "
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
            if active_ref and not self._bind_compile_guard(rec, target):
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
            self._warn_cell_key_divergence(spec.name, target)
            if target.active_compile_ref:
                # pgw#622: post-proof, novel request shapes serve eager while
                # the compiled path warms in the background; each completed
                # warm republishes the grown cell for the fleet.

                if hot_swap.enable(
                    pipeline,
                    on_warmed=hot_swap.Debounce(
                        self._shape_warm_republisher(spec, pipeline)),
                ):
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
        """Full-replace READY compile-target snapshot for StateDelta."""
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
                        requested_cell_key=target.requested_cell_key,
                        requested_cell_axes=dict(target.requested_cell_axes),
                    ))
        return sorted(out, key=lambda target: target.incarnation_id)

    def cell_lookups(self) -> List[pb.CellLookup]:
        """Full-replace pull-by-key lookup hints (th#883/gw#581).

        Live targets contribute their exact requested keys; compile-declared
        specs not yet live contribute pre-load CANDIDATE keys (the loader's
        resident-upgrade decision is unknown before load, so both plain
        lanes are candidates). Lookup hints only: the hub may attach store
        hits at boot; forge demand comes exclusively from live targets."""

        seen: set[Tuple[str, str]] = set()
        for rec in self._classes.values():
            live = [
                target for target in rec.compile_targets.values()
                if target.requested_cell_key
            ] if rec.ready else []
            for target in live:
                family = str(getattr(
                    target.spec.compile_cell(), "family", "") or "").strip()
                if family:
                    seen.add((family, target.requested_cell_key))
            if live:
                continue
            for spec in rec.specs:
                cfg = spec.compile_cell()
                family = str(getattr(cfg, "family", "") or "").strip()
                if cfg is None or not family:
                    continue
                bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
                want_execution_lane = self._mandatory_execution_lane_of_bound(
                    wire_ref(binding) for binding in spec.models.values()
                )
                execution_lanes = (
                    (want_execution_lane,) if want_execution_lane
                    else _SPECULATIVE_CELL_BASE_EXECUTION_LANES)
                for execution_lane in execution_lanes:
                    try:
                        digest = cell_key.compute(
                            family, execution_lane, bucket,
                            contract=cfg.contract_digest(),
                            regional=bool(getattr(cfg, "regional", False)),
                        ).digest
                    except Exception:
                        continue
                    seen.add((family, digest))
        return [
            pb.CellLookup(family=family, cell_key=key)
            for family, key in sorted(seen)
        ]

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
        """th#1059 twin (hub: ``mandatoryTracedLane``): the flavor token names
        the STORAGE format, not the execution. Mandatory-ness follows the
        hub-resolved EXECUTION lane whenever one is known for this ref —
        SDXL's mixed variant is ``#fp8-w8a8`` storage serving the w8a16
        upcast lane (plain graphs, never scaled_mm), while qwen's
        ``#fp8-w8a8`` executes real w8a8. Without lane evidence the flavor
        token remains the fallback; conflicting evidence fails closed to the
        mandatory reading.
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
        if known:
            return mandatory
        return _ref_mandatory_execution_lane(ref)

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
        if (
            target_active[0] != identity[1]
            or target_active[1] != identity[2]
            or target_active[2] != identity[3]
        ):
            raise RetryableError(
                "required_compile_identity_mismatch: active cell or execution "
                "contract changed"
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

    def _slot_dispatch_binding(
        self, spec: EndpointSpec, slot: str, run_ref: str
    ) -> ModelRef:
        """The binding a declared Slot materializes for THIS dispatch.

        Precedence (pgw#532): the hub-resolved pick from
        ``RunJob.models[slot]`` > the code-declared ``default_checkpoint``
        when it is itself a CAS ref. A hub-connected worker NEVER
        materializes a Slot's raw upstream default (mirror-first, gw#465):
        when neither source yields a CAS ref the dispatch fails RETRYABLE —
        the hub must resolve the slot to a ref this worker can load, not
        the worker self-fetching Civitai/HF.

        Identity gate (gw#583, the ie#518 silence): a FIXED slot — one whose
        declared ``Slot`` carries no ``selected_by=`` catalog, or a bare
        binding — has exactly one code-declared repo. A hub-resolved pick
        that differs only in tag/flavor is the ordinary case above; a pick
        naming a DIFFERENT REPO for a fixed slot is silent drift, not a
        legitimate choice, and refuses closed. ``selected_by=`` slots opt
        into hub-catalog picks explicitly — those are exempt by design.
        """
        declared = spec.models.get(slot)
        if run_ref:
            if (
                declared is not None
                and declared.source == "tensorhub"
                and run_ref == wire_ref(declared)
            ):
                return declared
            try:
                binding = self._hub_binding(run_ref)
            except ValueError:
                logger.warning(
                    "slot %r of %s: resolved_models ref %r is not a CAS ref; "
                    "falling back to the declared default", slot, spec.name, run_ref)
            else:
                catalog_slot = spec.slots.get(slot)
                fixed_repo = (
                    declared is not None
                    and declared.source == "tensorhub"
                    and not (catalog_slot is not None and catalog_slot.selected_by)
                )
                if fixed_repo and declared is not None and binding.path != declared.path:
                    raise ModelSlotIdentityError(
                        spec.name, slot,
                        declared_ref=wire_ref(declared), dispatched_ref=run_ref,
                    )
                return binding
        if declared is not None and declared.source == "tensorhub":
            return declared
        raise RetryableError(
            f"slot {slot!r} of {spec.name!r} has no loadable hub ref for this "
            f"request (resolved_models[{slot!r}]={run_ref!r}, declared "
            f"default source={getattr(declared, 'source', None)!r}); a "
            "hub-connected worker never fetches a Slot's raw upstream "
            "default (pgw#532/gw#465) — the hub must resolve the slot to a "
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
        """
        topo = self.topology
        if topo.degree > 1 and topo.parallel == "sequence":
            return (
                f"eager only at {topo}: compile/hot-swap/self-mint are "
                "disabled under context parallelism (pgw#775) — any forward "
                "outside the sequence gate would hang the degree-"
                f"{topo.degree} group"
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

    def _effective_spec(self, spec: EndpointSpec, run: "pb.RunJob") -> EndpointSpec:
        """The spec THIS dispatch runs (pgw#532): every declared Slot rebound
        to the hub-resolved pick in ``RunJob.models``. A pick that differs
        from the declared binding derives a NEW instance key — one resident
        instance per (class, resolved binding set), so ``setup()`` re-runs
        for the pick and setup-held state (``self.pipeline``) stays coherent
        per checkpoint while the LRU machinery evicts whole instances.
        Function-shaped (``cls=None``) specs rebind too — their slots inject
        via ``_handler_kwargs``, which reads the same ``spec.models``."""
        if not spec.slots:
            return spec
        run_refs = {
            b.slot: b.ref.strip() for b in run.models if b.slot and b.ref.strip()
        }
        # pgw#617 hierarchical bindings: per-component substitutions ride the
        # dispatch binding and become part of the derived spec's identity —
        # a component-only rebind derives a NEW instance (reload), a flat
        # binding (empty map) is byte-identical to the pre-#617 path.
        run_comps: Dict[str, Dict[str, str]] = {}
        for b in run.models:
            comps = {
                str(k).strip(): str(v).strip()
                for k, v in b.components.items()
                if str(k).strip() and str(v).strip()
            }
            if not comps:
                continue
            if b.slot not in spec.slots:
                logger.warning(
                    "component substitutions on %s slot %r ignored: not a "
                    "declared Slot", spec.name, b.slot)
                continue
            run_comps[b.slot] = comps
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
            binding = self._slot_dispatch_binding(
                spec, slot, run_refs.get(slot, ""))
            slot_comps = run_comps.get(slot) or {}
            if slot_comps:
                for comp, comp_ref in slot_comps.items():
                    try:
                        self._hub_binding(comp_ref)
                    except ValueError:
                        raise ComponentSubstitutionError(
                            spec.name, slot, comp,
                            detail=f"override ref {comp_ref!r} is not a "
                                   "tensorhub-CAS ref") from None
                binding = msgspec.structs.replace(
                    binding, component_overrides=tuple(sorted(slot_comps.items())))
            effective[slot] = binding
        if effective == spec.models:
            return spec
        return dc_replace(spec, models=effective)

    def _execution_lane_effective_spec(self, spec: EndpointSpec, execution_lane_str: str) -> EndpointSpec:
        """th#913/gw#596: rebind the spec's declared tensorhub models to the
        instructed lane. A family instruction expands through the hub's
        per-card resolution picks (or the local cast lane for fp8-w8a16); a
        full descriptor demands exactly that lane. An unserveable lane raises
        :class:`gen_worker.models.lanes.LaneUnavailableError` (typed refusal
        naming the lane — never a silent fallback). The rebound spec derives
        a new instance key, so warm workers keep both variants resident and
        cycle them via the gw#551 lane machinery."""

        raw = str(execution_lane_str or "").strip()
        if not raw:
            return spec
        try:
            req = lanespec.parse_execution_lane_spec(raw)
        except ValueError as exc:
            raise ValidationError(str(exc)) from None
        if req.is_zero:
            return spec
        # th#1050: a lane the endpoint DECLARES (handles=) is served by the
        # author's own code branching on ctx.lane — satisfiable with no
        # laddered rebind (custom loaders/kernels have nothing to rebind).
        declared_execution_lane = (
            req.execution_lane is not None
            and lanespec.execution_lane_body_id(req.execution_lane) in getattr(spec, "handles", ())
        )
        def pick_execution_lane_of(pick: "Optional[Tuple[Any, ...]]") -> "Optional[Any]":
            if pick is None or not pick[2]:
                return None
            try:
                return lanespec.parse_execution_lane(pick[2])
            except ValueError:
                return None

        # Only hub-LADDERED refs participate: a ref without a resolution pick
        # (ancillary VAE/encoder, flavor-pinned author override) keeps its
        # binding — the hub never laddered it, so no lane applies to it.
        declared = self._declared_models.get(spec.name) or {}
        effective = dict(spec.models)
        changed = False
        # bf16 is trivially serveable (the declared base IS bf16); quantized
        # lanes must find at least one laddered ref that can serve them —
        # unless the endpoint declares the lane (author code serves it).
        satisfied = req.family == lanespec.FAMILY_BF16 or declared_execution_lane
        want_w8a8 = req.execution_lane is not None and req.execution_lane.activation == lanespec.ACT_W8A8
        for slot, base_binding in declared.items():
            if slot not in effective:
                continue
            if getattr(base_binding, "source", "") != "tensorhub":
                continue
            base_ref = wire_ref(base_binding)
            pick = self._model_resolutions.get(base_ref)
            if pick is None:
                continue
            plane = pick_execution_lane_of(pick)
            new_binding = base_binding  # bf16 family: revert to the declared base
            if req.family == lanespec.FAMILY_BF16:
                satisfied = True
            elif req.family == lanespec.FAMILY_FP8:
                if plane is not None and lanespec.family_of(plane) == lanespec.FAMILY_FP8 and (
                    req.execution_lane is None or plane.activation == req.execution_lane.activation
                ):
                    resolved_ref, cast, _ = pick
                    new_binding = rebind_pick(
                        base_binding,
                        resolved_ref=(resolved_ref if resolved_ref != base_ref else ""),
                        cast=cast)
                    satisfied = True
                elif want_w8a8:
                    continue  # this ref cannot serve w8a8; refusal decided below
                else:
                    # family fp8 / explicit w8a16 without a stored fp8 pick:
                    # the local cast lane (per-layer upcast at inference).
                    new_binding = rebind_pick(base_binding, cast="fp8")
                    satisfied = True
            elif req.family == lanespec.FAMILY_4BIT:
                if plane is None or lanespec.family_of(plane) != lanespec.FAMILY_4BIT or (
                    req.execution_lane is not None and plane.weights != req.execution_lane.weights
                ):
                    continue
                resolved_ref, cast, _ = pick
                new_binding = rebind_pick(
                    base_binding,
                    resolved_ref=(resolved_ref if resolved_ref != base_ref else ""),
                    cast=cast)
                satisfied = True
            if wire_ref(effective[slot]) != wire_ref(new_binding) or (
                getattr(effective[slot], "storage_dtype", "")
                != getattr(new_binding, "storage_dtype", "")
            ):
                effective[slot] = new_binding
                self.store.register_binding(wire_ref(new_binding), new_binding)
                changed = True
        if not satisfied:
            raise lanespec.ExecutionLaneUnavailableError(
                raw, f"no laddered binding of {spec.name!r} can serve this lane "
                     "on this worker (flavor never resolved for its card)")
        if not changed:
            return spec
        derived = dc_replace(spec, models=effective)
        logger.info(
            "LANE_INSTRUCTION function=%s lane=%s rebound=%s",
            spec.name, raw,
            {s: wire_ref(b) for s, b in effective.items()
             if wire_ref(spec.models[s]) != wire_ref(b)})
        return derived

    def _effective_config(
        self, spec: EndpointSpec, run: Optional["pb.RunJob"] = None,
    ) -> Dict[str, Any]:
        """th#1087 effective declared-parameter values for one dispatch:
        declared defaults <- worker's current config store <- RunJob-stamped
        values (read-at-dispatch class; a stamped job keeps its values even
        if a gen bump lands mid-flight)."""
        if not spec.config:
            return {}
        values = {p.name: p.default for p in spec.config}
        declared = set(values)
        for name, v in self.runtime_config.parameters_for(spec.name).items():
            if name in declared:
                values[name] = v
        if run is not None:
            gen, stamped = extract_job_config(run)
            if stamped is not None:
                stamped = {
                    name: value
                    for name, value in stamped.items()
                    if name in declared
                }
                # Advance the worker store + snapshot file to this dispatch's
                # stamped values, so subprocesses read the latest on invoke.
                self.runtime_config.stamp_function(spec.name, stamped, gen)
            values.update(stamped or {})
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
                for target in rec.compile_targets.values():
                    with target.state_lock:
                        active = str(target.active_compile_ref or "")
                    if active:
                        ref, pipeline = active, target.pipeline
                        break
                if not ref:
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
        if not any(
            s.compile is not None and s.compile.family for s in rec.specs
        ):
            # Eager is this release's CONTRACT, not a degradation. Naming it
            # keeps the honest zero out of every defect-class count.
            return serving_mode_mod.POSTURE_NO_COMPILE_DECLARED
        if not rec.ready:
            return serving_mode_mod.POSTURE_ARM_PENDING
        return serving_mode_mod.POSTURE_UNCOMPILED

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

    def _served_execution_lane(self, spec: EndpointSpec, instructed: str = "") -> str:
        """The CONCRETE lane this spec's instance executes as, for
        JobMetrics.lane and ctx.lane reporting: the most-quantized pipeline
        binding's lane (table rank), with live compile state as a preference.
        Fixed-mode bodies override it; a declared (handles=) instruction owns
        the full lane outright."""

        compiled = False
        if spec.cls is not None:
            rec = self._classes.get(spec.instance_key)
            if rec is not None:
                compiled = any(
                    getattr(t, "active_compile_ref", "")
                    for t in rec.compile_targets.values())
        handled = self._handled_execution_lane_body(spec, instructed)
        if handled:
            return lanespec.execution_lane_id(lanespec.parse_execution_lane(instructed))
        # Report the most-quantized binding's lane: quantized lanes always
        # outrank bf16 (a bf16 VAE riding a w8a16 pipeline is still the
        # w8a16 lane), ties by table rank.
        ranked = {body: i for i, body in enumerate(lanespec.known_execution_lanes())}
        best = None
        best_key: Tuple[int, int] = (2, len(ranked) + 1)
        for binding in spec.models.values():
            execution_lane = lanespec.execution_lane_of_binding(
                getattr(binding, "flavor", "") or "",
                getattr(binding, "storage_dtype", "") or "",
                compiled)
            quant = 1 if lanespec.family_of(execution_lane) == lanespec.FAMILY_BF16 else 0
            key = (quant, ranked.get(lanespec.execution_lane_id(execution_lane), len(ranked)))
            if best is None or key < best_key:
                best, best_key = execution_lane, key
        if best is None:
            best = lanespec.ExecutionLane(
                weights=lanespec.WEIGHTS_BF16, activation=lanespec.ACT_W16A16,
                execution=lanespec.EXEC_COMPILED if compiled else lanespec.EXEC_EAGER)
        return lanespec.execution_lane_id(best)

    async def ensure_desired_instance(
        self,
        desired: "pb.DesiredInstance",
        snapshots: Dict[str, "pb.Snapshot"],
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
        snapshots: Dict[str, "pb.Snapshot"],
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

        run = pb.RunJob(function_name=desired.function_name, models=remapped)
        effective = self._effective_spec(spec, run)
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

    def _job_pin_refs(self, spec: EndpointSpec, slots: List[str]) -> List[str]:
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
                for r in _binding_wire_refs(spec.models[s])
                if r not in execution_lane_refs
            ]
            + shared_ids
        ))

    def _job_admission_sizes(
        self, spec: EndpointSpec, slots: List[str], run: "pb.RunJob",
    ) -> Dict[str, int]:
        """ref -> expected VRAM bytes for one job's admission lease (pgw#641
        Stage 2). Same ref set as :meth:`_job_pin_refs`; bytes follow the
        pgw#636 ask ladder — a prior MEASURED hint wins, else the dispatch's
        own snapshot byte total (honest for a never-seen pick), else the
        banked snapshot's total, else 0 (lease-protected, no reservation)."""
        res = self.store.residency
        run_snapshots = dict(run.snapshots) if run.snapshots else {}

        def _expect(ref: str) -> int:
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
        snapshots: Optional[Dict[str, pb.Snapshot]] = None,
        promote_slots: Optional[List[str]] = None,
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
                    r for slot in self._setup_slots(spec)
                    for r in _binding_wire_refs(spec.models[slot])
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
            if rec.ready and not rec.stale and spec.compile is not None:
                mandatory_execution_lane = self._mandatory_execution_lane_of_bound(
                    wire_ref(spec.models[slot])
                    for slot in self._setup_slots(spec)
                )

                try:
                    desired_cell = await self._fetch_compile_snapshot(
                        spec, snapshots)
                except compile_cache.CompiledExecutionLaneUnavailableError as exc:
                    if mandatory_execution_lane:
                        # Desired state no longer supplies a mandatory exact
                        # cell. Remove the old READY incarnation before
                        # reporting the state-driven failure; it must not keep
                        # serving under superseded scheduler evidence.
                        rec.stale = True
                        async with self._intent_lock(
                            intent_id,
                            self._load_lock,
                            operation=f"vacate stale setup for {spec.name}",
                            stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
                            reason=pb.LIFECYCLE_WAIT_REASON_LOAD_LOCK,
                            resume_stage=pb.LIFECYCLE_INTENT_STAGE_LOADING_HOST,
                        ):
                            await self._vacate_record(rec)
                        self._mark_compile_setup_unavailable(
                            rec, spec, str(exc))
                        self._on_state_change()
                        raise
                    desired_cell = None
                if desired_cell is not None:
                    live_targets = list(rec.compile_targets.values())
                    target_identities = []
                    for target in live_targets:
                        with target.state_lock:
                            target_identities.append((
                                target.active_compile_ref,
                                target.active_compile_snapshot_digest,
                                target.active_self_mint,
                            ))
                    identity_moved = not target_identities
                    digest_aligned = False
                    for active_ref, active_digest, is_self_mint in (
                            target_identities):
                        if active_ref != desired_cell.ref:
                            identity_moved = True
                            break
                        if active_digest == desired_cell.snapshot_digest:
                            continue
                        if not is_self_mint:
                            # Delivered target: a same-ref digest move is a
                            # genuine republish (mutable label cells) — the
                            # pre-gw#604 vacate/rebuild convergence stands
                            # (the rebuild loads a FRESH pipe, whose re-trace
                            # produces honest FX lookups).
                            identity_moved = True
                            break
                        # gw#604: for the worker's OWN self-mint, cell
                        # identity IS the key (gw#581), and the ref encodes
                        # it — the desired cell is the SAME cell this live
                        # object already proved, published, and serves; the
                        # digests differ only in transport FORM
                        # (self-attested tar digest vs the store's snapshot
                        # manifest digest, th#910 ruling). A passed proof on
                        # a live object stays valid for that object's
                        # lifetime; a warm-process re-proof cannot produce
                        # honest FX lookups (dynamo serves from in-memory
                        # code, hits stay 0 — the live fail-closed re-arm
                        # loop). NEVER vacate; align the advertised digest to
                        # the store's so gw#577 receipts/fences line up
                        # fleet-wide.
                        for target in live_targets:
                            with target.state_lock:
                                if (target.active_compile_ref
                                        == desired_cell.ref):
                                    target.active_compile_snapshot_digest = (
                                        desired_cell.snapshot_digest)
                        digest_aligned = True
                        logger.info(
                            "self-mint cell %s confirmed by the store; "
                            "advertised digest aligned %s -> %s (no re-arm)",
                            desired_cell.ref, active_digest,
                            desired_cell.snapshot_digest,
                        )
                    if identity_moved:
                        logger.info(
                            "desired compile identity moved for %s -> %s@%s; "
                            "vacating stale instance",
                            spec.name,
                            desired_cell.ref,
                            desired_cell.snapshot_digest,
                        )
                        rec.stale = True
                    elif digest_aligned:
                        self._on_state_change()
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
                    await self._vacate_record(rec)
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
        armed = next(
            (t.active_compile_ref for t in rec.compile_targets.values()
             if t.active_compile_ref), "")
        # th#1359: reaching here means this boot's mint disposition is FINAL
        # on every path (inline setup, adopted cell, eager-without-mint, and
        # the background mint's own `finally`). The mint-goal driver waits on
        # exactly that fact rather than inventing a second notion of "boot is
        # over" that could disagree with the one boot telemetry publishes.
        try:
            mint_goal_mod.note_disposition_final()
        except Exception:  # pragma: no cover - a latch never breaks a boot
            logger.debug("mint-goal disposition latch failed", exc_info=True)
        if armed or rec.background_mint is not None:
            return
        # pgw#805: a boot that DECLARED a compile target and ends with no
        # artifact and no mint in flight must say so. This is the terminal
        # backstop for the whole miss policy — the individual declines
        # (fleet_cells._fail_closed, mint_recipe, mint_delegate) each name
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
            phase="boot_ended_uncompiled",
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
                    "warmup",
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
        that selects graphs or kernels — precision lane (flavor /
        storage_dtype / dtype) and component overrides — and NEVER the
        checkpoint ref itself. Two fine-tunes of one family land on the same
        key by construction; a lane rebind or a component substitution
        derives a different one."""
        rows = tuple(
            (
                slot,
                getattr(b, "flavor", "") or "",
                getattr(b, "storage_dtype", "") or "",
                getattr(b, "dtype", "") or "",
                tuple(getattr(b, "component_overrides", ()) or ()),
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
        snapshots: Optional[Dict[str, pb.Snapshot]],
        *,
        proof_objects: typing.Iterable[Any] = (),
        cold_proof_ids: typing.Container[int] = (),
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
        # Tracing == some artifact is armed or minting on this setup; only
        # then does the full class x bucket cross-product buy anything (each
        # graph must trace into the capture / prove against the cell).
        tracing = bool(objects)
        memory = self._warm_contract_runs.setdefault(
            self._warm_contract_key(spec), set())
        armed_refs = tuple(armed_cell_refs)
        skip_ok = (
            allow_contract_skip
            and not cold_proof_ids
            and all(compile_cache.cell_proven_in_process(r) for r in armed_refs)
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
        # pgw#735: three compiled backends, three proofs. Dynamo proves by FX
        # cache hits, TRT by engine executions, an EXPORTED artifact by its own
        # invocations — an exported cell performs no FX lookup at all, so a
        # cache-hit requirement would score every honest .pt2 adoption as a
        # failure. Never synthesize a hit counter for it: this is the one path
        # whose whole job is to detect a lie about serving compiled.
        start_counts = {
            id(obj): (
                compile_cache.execution_count(obj),
                trt_engine.execution_count(obj),
                aot_serve.execution_count(obj),
            )
            for obj in objects
        }
        # pgw#654 coverage attribution: runs prove GRAPH CLASSES; an alias
        # is proven on an object once ALL of its graph classes proved there.
        proven_keys: Dict[int, set] = {}
        # pgw#844: which objects proved through the EXPORTED lane. An exported
        # artifact refuses an out-of-contract shape BY NAME and serves that one
        # call eager while staying armed, so a graph class it did not serve is
        # a per-shape posture, not a silent recompile — which is what lets the
        # attribution below be per-class for this lane and stay all-or-nothing
        # for dynamo, where an unproven class means an unannounced recompile.
        exported_proof_ids: set = set()

        async def _one(wj: Any, build: Any, mode: str, *, variant: bool) -> bool:
            """One warmup forward; False = OOM, stop warming."""
            before = {
                id(obj): (
                    compile_cache.execution_count(obj),
                    trt_engine.execution_count(obj),
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
                    if torch is not None and torch.cuda.is_available():
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
                calls_before, trt_before, aot_before = before[id(obj)]
                inductor_proven = (
                    compile_cache.execution_count(obj) > calls_before
                    and (
                        compile_cache.cache_hit_count(obj) > 0
                        or id(obj) in cold_proof_ids
                    )
                )
                trt_proven = trt_engine.execution_count(obj) > trt_before
                # pgw#735: an exported artifact proves itself by executing —
                # and by still being armed, so a call that ended in a revoked
                # (failed) artifact cannot count as proof.
                aot_proven = aot_serve.proven_since(obj, aot_before)
                if aot_proven:
                    exported_proof_ids.add(id(obj))
                if inductor_proven or trt_proven or aot_proven:
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
                and trt_engine.execution_count(obj) == start_counts[id(obj)][1]
                and aot_serve.execution_count(obj) == start_counts[id(obj)][2]
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
        for obj_id, proven in proven_keys.items():
            names = {
                name for name, keys in keys_by_name.items()
                if keys and keys <= proven
            }
            # pgw#844: …and on the EXPORTED lane an alias that proved SOME of
            # its graph classes is attributed too, with the rest named.
            #
            # The measured shape (attempt twelve, pod o0legpgj5olhic): a
            # regional sdxl cell armed all 72 entries, dispatched 1024x1024
            # correctly, and refused the other eight aspect buckets
            # `entry_ambiguous` because a transformer block sees H_lat*W_lat
            # and the entries are keyed on H and W separately. Those eight
            # classes went unproven, the all-or-nothing rule above attributed
            # NO alias, the target was omitted as `target_applicability_
            # incomplete`, and the boot ended `boot_ended_uncompiled` — so the
            # ONE bucket that was armed, correct and unambiguous served eager
            # too. One undispatchable shape cost the pod every shape.
            #
            # `boot_ended_uncompiled` must mean "nothing is dispatchable", not
            # "something wasn't". An exported artifact refuses a shape it
            # cannot serve BY NAME, counts it, emits `aot_ingress_refused`,
            # charges the request `fallback_reason=ingress_refused`, and stays
            # armed — so the degradation is per shape and fully visible, which
            # is exactly the fail-soft posture the compiled lane is built on.
            # Dynamo keeps the strict rule: there an unproven class means an
            # unannounced recompile at serve time, which is silent.
            partial: Dict[str, Tuple[str, ...]] = {}
            if obj_id in exported_proof_ids:
                for name, keys in keys_by_name.items():
                    if not keys or name in names or not (keys & proven):
                        continue
                    names.add(name)
                    partial[name] = tuple(sorted(
                        str(key) for key in (keys - proven)))
            if names:
                evidence.functions_by_object[obj_id] = names
            if partial:
                evidence.partial_by_object[obj_id] = partial
                for name, missing in sorted(partial.items()):
                    total = len(keys_by_name[name])
                    activity_mod.emit_event(
                        "compiled_shape_coverage",
                        f"fn={name}: {total - len(missing)}/{total} declared "
                        f"graph classes served COMPILED at boot; the compiled "
                        f"lane stays armed and these {len(missing)} class(es) "
                        f"serve EAGER per request (each one named at ingress, "
                        f"and every such request reports "
                        f"fallback_reason="
                        f"{serving_mode_mod.FALLBACK_INGRESS_REFUSED}): "
                        f"{list(missing[:8])!r}",
                        phase="partial_shape_coverage",
                    )
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
                ref in _binding_wire_refs(binding)
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
            released = await self._vacate_record(rec)
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
        snapshots: Optional[Dict[str, pb.Snapshot]],
        *,
        intent_id: str = "",
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
                setup_slots=setup_slots)
        finally:
            _cc_execution_lane._SETUP_EXEC_EXECUTION_LANE_PINNED.reset(_pin_token)
            _cc_execution_lane._SETUP_EXEC_EXECUTION_LANE.reset(_execution_lane_token)

    async def _setup_locked_inner(
        self, spec: EndpointSpec, rec: _ClassRecord,
        snapshots: Optional[Dict[str, pb.Snapshot]],
        *,
        intent_id: str = "",
        setup_slots: List[str],
    ) -> Any:
        assert spec.cls is not None
        # gw#494: residency keys for this setup are derived ONCE, here, in
        # resolved space; downloads, booking and the record's held_refs all
        # use these exact strings (a HelloAck rebind during an await below
        # cannot split download/booking/teardown identities).
        slot_refs: Dict[str, str] = {
            slot: wire_ref(spec.models[slot]) for slot in setup_slots
        }
        slot_identities: Dict[str, _ResidencyIdentity] = {}
        # pgw#974: ONE resolution per slot — the binding, the tree its bytes
        # were materialized into, and the pgw#617 component overrides that
        # complete the composition. Written by a single statement per slot, so
        # the three cannot drift apart or arrive without one another; the
        # plain-path and plain-override views the local loaders take are
        # DERIVED from it below, never maintained beside it.
        resolved_slots: Dict[str, MintSlot] = {}
        override_digests: Dict[str, str] = {}
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
            comps: Dict[str, str] = {}
            for comp, comp_ref in _component_overrides(binding):
                try:
                    comp_binding = self._hub_binding(comp_ref)
                except ValueError:
                    raise ComponentSubstitutionError(
                        spec.name, slot, comp,
                        detail=f"override ref {comp_ref!r} is not a "
                               "tensorhub-CAS ref") from None
                comp_mat = await self.store._materialize_local(
                    comp_ref, (snapshots or {}).get(comp_ref),
                    binding=comp_binding)
                comps[comp] = str(comp_mat.path)
                if comp_mat.identity[0]:
                    override_digests[comp_ref] = comp_mat.identity[0]
            resolved_slots[slot] = MintSlot(
                ref=binding, path=str(materialized.path),
                component_paths=comps)
        paths: Dict[str, str] = {
            slot: res.path for slot, res in resolved_slots.items()}
        component_paths: Dict[str, Dict[str, str]] = {
            slot: dict(res.component_paths)
            for slot, res in resolved_slots.items() if res.component_paths}
        eager_only = self._eager_only_reason()
        if eager_only and spec.compile is not None:
            logger.info("%s: %s", spec.name, eager_only)
        compile_selection = (
            None if eager_only
            else await self._fetch_compile_snapshot(spec, snapshots)
        )
        compile_artifact = compile_selection.path if compile_selection else None
        # pgw#947: the serving-kernel lane comes from the CELL, and it has to
        # be pinned BEFORE setup() — the linears are swapped at model load, so
        # a verdict read afterwards would arrive one whole pipeline too late.
        # No cell (eager boot, self-minting boot, pre-pgw#947 cell) is the
        # declared conservative default WITH a typed reason; there is no SM
        # allowlist and no per-boot probe to fall back on any more.
        # The verdict is EVIDENCE, not an instruction: cells are keyed on SM
        # and the lane is not a key axis, so a 96 GB card's winner can reach a
        # 32 GB card of the same SM. adopt() re-applies the fit constraint
        # against THIS device before pinning.
        kernel_path.adopt_from_artifact(
            compile_artifact, source=f"{spec.name} boot")
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
            rec.held_refs = sorted(
                set(slot_refs.values()) | set(override_digests)
                | {
                    comp_ref
                    for slot in setup_slots
                    for _, comp_ref in _component_overrides(spec.models[slot])
                }
            )
            rec.held_snapshot_digests = {
                slot_refs[slot]: identity[0]
                for slot, identity in slot_identities.items()
                if slot in slot_refs and identity[0]
            }
            rec.held_snapshot_digests.update(override_digests)
            # Override triples key as "<slot>.<component>" — part of the
            # composition's identity (compile-cell applicability, pgw#617).
            rec.held_bindings = sorted(
                [
                    (
                        slot,
                        ref,
                        rec.held_snapshot_digests.get(ref, ""),
                    )
                    for slot, ref in slot_refs.items()
                ]
                + [
                    (
                        f"{slot}.{comp}",
                        comp_ref,
                        rec.held_snapshot_digests.get(comp_ref, ""),
                    )
                    for slot in setup_slots
                    for comp, comp_ref in _component_overrides(spec.models[slot])
                ]
            )
            setup = getattr(instance, "setup", None)
            inj = _InjectionResult(kwargs={}, loaded={})

            vram_before = self._vram_allocated()
            if spec.runtime:
                rec.server = await self._boot_engine_server(spec, paths)
            if callable(setup):
                inj = await self._injection_kwargs(
                    spec, setup, paths, server=rec.server,
                    compile_selection=compile_selection,
                    snapshots=snapshots,
                    slot_identities=slot_identities,
                    component_paths=component_paths)
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
                    enable=self._arming_enable,
                )
                with arming_scope:
                    if asyncio.iscoroutinefunction(setup):
                        await setup(**inj.kwargs)
                    else:
                        await _to_thread_complete(setup, **inj.kwargs)
                # arm_compile() is the sole unambiguous ownership seam for a
                # self-loaded pipeline. Such a pipeline may be built from any
                # path-valued setup input, so freeze every self-loaded slot
                # into its applicability rather than guessing one later.
                self_loaded_slots = tuple(
                    slot for slot in setup_slots
                    if isinstance(inj.kwargs.get(slot), (str, Path))
                )
                scope_mints = arming_scope.self_mints
                for bug in arming_scope.selection_bugs.values():
                    # th#1031: the fleet policy already self-minted a working
                    # cell instead of aborting — still report the th#883
                    # invariant loudly.
                    await self._report_cell_selection_bug(
                        spec, compile_selection, bug)
                for pipe, armed in arming_scope.objects:
                    if not compile_cache.has_compile_target(pipe, spec.compile):
                        continue
                    inj.add_compile_object(pipe, self_loaded_slots)
                    mint = scope_mints.get(id(pipe))
                    selection = _selection_for(compile_selection, mint)
                    if getattr(mint, "delegated", False):
                        # pgw#784: see the slot path above — recorded, but
                        # never advertised as an active artifact.
                        inj.pending_self_mints[id(pipe)] = mint
                    elif armed and selection is not None:
                        inj.active_compile_artifacts[id(pipe)] = selection
                        if trt_engine.is_engine_ref(selection.ref):
                            inj.trt_execution_before[id(pipe)] = (
                                trt_engine.execution_count(pipe))
                        # gw#587 CORRECT FIX: a PendingSelfMint is not proven
                        # or packed yet — the warmup proof below finalizes it
                        # (pack + publish) only after confirming a real
                        # compiled call, never before.
                        if hasattr(mint, "capture_dir"):
                            inj.pending_self_mints[id(pipe)] = mint
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
            eager_first = self._eager_first_eligible(spec, inj)
            delegated_mints = _delegated_pendings(inj.pending_self_mints)
            if eager_first and delegated_mints:
                # pgw#784: _mint_budget_ok gates an IN-PROCESS capture that
                # will never exist here — nothing is armed on these pipes. The
                # child's own co-residency ask (its weights + one activation
                # set + inductor workspace + a CUDA context) is budgeted per
                # attempt by mint_delegate, against the card as it actually is
                # at spawn time rather than as it was at boot.
                pass
            elif eager_first and not self._mint_budget_ok(spec, inj):
                # pgw#737: THE gate. It sits here and not only in the driver
                # because arming + enabling the routers is already the first
                # allocation of the capture — the boot warm's own forwards
                # enqueue background compiles the instant the routers go
                # concurrent. A card that cannot hold the capture never gets
                # one armed: the targets go back to true eager, the cell
                # stays absent, and the boot follows the plain-eager shape
                # (never the foreground compile-then-serve mint, which is
                # strictly worse for the tenant).
                eager_first = False
            if delegated_mints and not eager_first:
                # pgw#784: nothing is armed on these pipes, so the foreground
                # compile-then-serve path below cannot drive them. Discard the
                # obligation and serve eager with the cell absent — the honest
                # miss policy — rather than run a warmup proof against an
                # unarmed pipeline. (fleet_cells.delegatable already refused to
                # delegate anything that MUST serve compiled, so this is the
                # custom-warmup / mixed-delivered-artifact remainder.)
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

                mint_selections: Dict[int, _CompileArtifactSelection] = {}
                mint_pipes: Dict[int, Any] = {}
                for candidate in inj.compile_objects:
                    pid = id(candidate.pipeline)
                    if pid not in inj.pending_self_mints:
                        continue
                    sel = inj.active_compile_artifacts.pop(pid, None)
                    if sel is None:
                        # pgw#784: a DELEGATED pending never entered
                        # active_compile_artifacts (nothing is armed on its
                        # pipe), but its claimed key ref is computable now from
                        # static axes, and th#910's self-attested dispatch
                        # fence wants it advertised while the child compiles.
                        sel = _selection_for(
                            None, inj.pending_self_mints.get(pid))
                    if sel is not None:
                        mint_selections[pid] = sel
                    mint_pipes[pid] = candidate.pipeline
                    hot_swap.enable(candidate.pipeline)
                    # pgw#677: the mint's own background compiles are the
                    # first consumers of the turn gate — wire it before the
                    # first seed can enqueue a warm job.
                    self._wire_turn_gate(rec, candidate.pipeline)
                rec.background_mint = _BackgroundMint(
                    spec=spec,
                    instance=instance,
                    snapshots=dict(snapshots) if snapshots else None,
                    pendings=dict(inj.pending_self_mints),
                    pipes=mint_pipes,
                    selections=mint_selections,
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
            # pgw#735: TRT engines and EXPORTED artifacts both prove
            # themselves by executing, not by an FX cache hit — only the
            # dynamo lane is scored by hits below.
            def _proves_by_fx(ref: str) -> bool:
                return not trt_engine.is_engine_ref(ref) and not \
                    aot_serve.is_aot_ref(ref)

            proves_inductor = any(
                _proves_by_fx(sel.ref)
                for sel in inj.active_compile_artifacts.values()
            )
            proof_before = {
                id(candidate.pipeline): (
                    compile_cache.execution_count(candidate.pipeline),
                    compile_cache.cache_miss_count(candidate.pipeline),
                    aot_serve.execution_count(candidate.pipeline),
                )
                for candidate in inj.compile_objects
                if proves_inductor
                and id(candidate.pipeline) in inj.active_compile_artifacts
                and _proves_by_fx(
                    inj.active_compile_artifacts[id(candidate.pipeline)].ref)
            }
            # Exported arms are proven separately: same fail-closed rule, its
            # own counter.
            aot_proof_before = {
                id(candidate.pipeline): aot_serve.execution_count(
                    candidate.pipeline)
                for candidate in inj.compile_objects
                if id(candidate.pipeline) in inj.active_compile_artifacts
                and aot_serve.is_aot_ref(
                    inj.active_compile_artifacts[id(candidate.pipeline)].ref)
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
            compile_seconds_before = (
                compile_cache.compile_wall_seconds() if proves_inductor else 0.0)
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
                            if _sel is not None and not trt_engine.is_engine_ref(
                                    _sel.ref):
                                compile_cache.reset_target_code(_cand.pipeline)
                        warmed, function_proofs, warm_aborted = await run_warmup()
                else:
                    warmed, function_proofs, warm_aborted = await run_warmup()
            compile_seconds = (
                compile_cache.compile_wall_seconds() - compile_seconds_before
                if proves_inductor else 0.0)
            # id(pipeline) -> (calls, cache_hits, cache_misses) observed across
            # this setup's warmup. Declared out here because pgw#923's adoption
            # report reads it whether or not this boot proved anything: a cell
            # that armed and then warmed to zero hits is exactly the adoption
            # the measurement lane exists to price.
            proof_by_obj: Dict[int, Tuple[int, int, int]] = {}
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
                proven = 0
                hits = 0
                misses = 0
                # gw#612: snapshot capture-sharing BEFORE the loop pops
                # pending entries. The publish decision needs to know, per
                # shared capture, whether EVERY sharer's graphs made it in.
                mint_by_id: Dict[int, Any] = {}
                mint_sharers: Dict[int, List[int]] = {}
                for _obj_id, _pending in inj.pending_self_mints.items():
                    mint_by_id[id(_pending)] = _pending
                    mint_sharers.setdefault(id(_pending), []).append(_obj_id)
                proven_mint_objs: set[int] = set()
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
                                compile_cache.record_cell_proven(proved_sel.ref)
                        else:
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
                    pending_mint = inj.pending_self_mints.get(id(pipe))
                    if not warmed or calls <= 0:
                        unexercised.append(candidate)
                    elif pending_mint is not None:
                        # gw#587 CORRECT FIX: a fresh self-mint capture has
                        # nothing pre-existing on disk to HIT against — its
                        # own real, successful compiled call (calls>0, no
                        # guard failure) IS the entire proof; requiring a
                        # disk hit here would fail every honest self-mint by
                        # construction. Pack the capture dir that call just
                        # populated and advertise the real digest — this is
                        # "prove-produces-the-mint": the published artifact
                        # is byte-derived from exactly this execution, never
                        # a second, separately-shaped compile. A pack
                        # failure never un-serves the request (the compiled
                        # callable is already live); it only forfeits
                        # advertising/publishing this boot's capture.

                        activity_mod.current_phase(
                            activity_mod.PHASE_SEAL_PUBLISH)
                        finalized = fleet_cells_mod.finalize_self_mint(
                            pipe, pending_mint,
                            expected_graphs=max(0, pipe_misses))
                        inj.pending_self_mints.pop(id(pipe), None)
                        if finalized is not None:
                            proven += 1
                            proven_mint_objs.add(id(pipe))
                            if callable(warmup):
                                function_proofs[id(pipe)] = {spec.name}
                            compile_cache.record_cell_proven(
                                str(finalized.ref))
                            inj.active_compile_artifacts[id(pipe)] = (
                                _CompileArtifactSelection(
                                    path=Path(finalized.artifact),
                                    ref=str(finalized.ref),
                                    snapshot_digest=str(
                                        finalized.snapshot_digest),
                                    self_mint=True))
                        else:
                            disproven.append(candidate)
                    elif pipe_hits > 0:
                        proven += 1
                        if callable(warmup):
                            function_proofs[id(pipe)] = {spec.name}
                        proved_sel = inj.active_compile_artifacts.get(id(pipe))
                        if proved_sel is not None:
                            compile_cache.record_cell_proven(proved_sel.ref)
                    elif (
                        pipe_misses <= 0
                        and (inmem_sel := inj.active_compile_artifacts.get(
                            id(pipe))) is not None
                        and compile_cache.cell_proven_in_process(inmem_sel.ref)
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
                unproven = list(disproven)
                if not proven:
                    unproven.extend(unexercised)
                    unexercised = []
                if unproven:

                    quant_execution_lane = any(
                        pipeline_weight_lane(
                            candidate.pipeline).startswith(_MANDATORY_EXECUTION_LANES)
                        for candidate in unproven
                    )
                    for candidate in unproven:
                        pipe = candidate.pipeline
                        function_proofs[id(pipe)] = set()
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
                            compile_cache.record_cell_quarantined(
                                failed_sel.ref)
                        failed_pending = inj.pending_self_mints.get(id(pipe))
                        if failed_pending is not None:
                            compile_cache.record_cell_quarantined(
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
                    # gw#608 FX-key forensics: this boot's recompiles already
                    # saved their entries (with embedded FxGraphHashDetails
                    # lines) into the live cache dir. The report ALWAYS
                    # carries the cache-state counts — fresh_keys>0 names the
                    # diverging key component (B1); fresh_keys=0 with
                    # same-key re-saves proves the keys MATCH and the miss is
                    # in torch's candidate-load path (B2), with the sibling
                    # guards/extern-libs diff and load probes naming the
                    # failing step. Store-served boots only (a minting boot
                    # has no seeded cell to diverge from).
                    # pgw#722 finding 2: FX forensics describe the dynamo
                    # lane only — a pure-exported disproof would report the
                    # SKIPPED delivered artifact's cache state, pure noise.
                    if proves_inductor and compile_selection is not None and not (
                        trt_engine.is_engine_ref(compile_selection.ref)
                    ):
                        try:
                            forensics = compile_cache.fx_cache_failure_report(
                                compile_selection.path)
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
                        # here -> cell_quarantined -> every declared function
                        # disabled -> pod retired -> the replacement re-mints
                        # the same key (5 cycles / 4 dead workers on the L4
                        # burst). A broken optimization must never kill a
                        # serving worker: withhold the unproven publish,
                        # quarantine the identity (above), and DEGRADE to
                        # explicit eager — serving_tier flips on the wire and
                        # the activity carries the confession; never silent
                        # (gw#586).
                        if mint_by_id:

                            for pending in mint_by_id.values():
                                fleet_cells_mod.withhold_self_mint_publish(
                                    pending,
                                    f"proof failed; degraded to eager "
                                    f"({detail})")
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
                    mandatory = pipeline_weight_lane(pipe).startswith(
                        _MANDATORY_EXECUTION_LANES)
                    if mandatory:
                        pending_mint = inj.pending_self_mints.pop(
                            id(pipe), None)
                        if pending_mint is not None:
                            finalized = pending_mint._state.get("minted")
                            if finalized is not None:
                                # A proven sibling finalized the SHARED
                                # capture (same key, one family cell) —
                                # advertise the finalized identity, not the
                                # arm-time placeholder.
                                inj.active_compile_artifacts[id(pipe)] = (
                                    _CompileArtifactSelection(
                                        path=Path(finalized.artifact),
                                        ref=str(finalized.ref),
                                        snapshot_digest=str(
                                            finalized.snapshot_digest),
                                        self_mint=True))
                            else:
                                # Its capture was never exercised and no
                                # sibling finalized it: there is no proven
                                # artifact to advertise. Drop the target
                                # loudly rather than advertise bytes
                                # nothing proved (the gw#586 shape).

                                fleet_cells_mod.abandon_self_mint(
                                    pending_mint)
                                logger.warning(
                                    "compile object (slots=%s) self-mint "
                                    "capture unexercised (calls=0) with no "
                                    "finalized sibling; dropping its "
                                    "compile target",
                                    sorted(candidate.slots))
                                function_proofs[id(pipe)] = set()
                                compile_cache.unwrap(pipe)
                                inj.active_compile_artifacts.pop(
                                    id(pipe), None)
                                continue
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
                    logger.warning(
                        "compile object (slots=%s) unproven (no warmup "
                        "modality, calls=0); serving eager",
                        sorted(candidate.slots))
                    function_proofs[id(pipe)] = set()
                    # pgw#722 finding 2: same exported-lane disarm as the
                    # unproven loop above.
                    if aot_serve.unwrap(pipe):
                        from .models import lora_lifted

                        lora_lifted.remove_lifted_lora_execution_lanes(pipe)
                    compile_cache.unwrap(pipe)
                    if spec.lora_bucket:
                        compile_cache.drop_lora_execution_lane(pipe)
                    inj.active_compile_artifacts.pop(id(pipe), None)
                    self._abandon_pending_mint(inj, pipe)
                if mint_by_id:
                    # gw#612 publish gate: a shared capture packs only the
                    # graphs the warmup compiled. Publish the family cell
                    # only when EVERY sharer proved into it; otherwise the
                    # store would gain a partial cell that fails gw#607's
                    # per-object adopt proof on every later boot (the
                    # gw#611 qwen hits=1/misses=1 release-breaker). The
                    # local mint keeps serving this process either way.

                    for pending in mint_by_id.values():
                        sharers = mint_sharers.get(id(pending), [])
                        gap = [
                            oid for oid in sharers
                            if oid not in proven_mint_objs
                        ]
                        if warm_aborted:
                            # pgw#677 reopen: a plan cut short (OOM backoff)
                            # can leave every OBJECT looking proven while
                            # whole graph CLASSES are missing — publishing
                            # that partial pack bricks every adopting boot.
                            fleet_cells_mod.withhold_self_mint_publish(
                                pending,
                                f"warm plan cut short ({warm_aborted}); "
                                "planned graphs are absent from the packed "
                                "cell")
                        elif gap:
                            fleet_cells_mod.withhold_self_mint_publish(
                                pending,
                                f"{len(gap)}/{len(sharers)} capture-sharing "
                                "compile object(s) were never exercised by "
                                "the warmup, so their graphs are absent "
                                "from the packed cell")
                        else:
                            fleet_cells_mod.publish_self_mint(pending)
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
                    ref = compile_selection.ref if compile_selection else ""
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
                            compile_cache.fx_cache_failure_report(
                                compile_selection.path))
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
            if compile_selection and trt_engine.is_engine_ref(compile_selection.ref):
                trt_candidates = [
                    candidate for candidate in inj.compile_objects
                    if id(candidate.pipeline) in inj.active_compile_artifacts
                ]
                unproven = [
                    candidate.pipeline for candidate in trt_candidates
                    if trt_engine.execution_count(candidate.pipeline)
                    <= inj.trt_execution_before.get(id(candidate.pipeline), 0)
                ]
                if callable(warmup):
                    unproven_ids = {id(pipe) for pipe in unproven}
                    for candidate in trt_candidates:
                        if id(candidate.pipeline) not in unproven_ids:
                            function_proofs[id(candidate.pipeline)] = {spec.name}
                if unproven:
                    for pipe in unproven:
                        function_proofs[id(pipe)] = set()
                        trt_engine.unwrap(pipe)
                        inj.active_compile_artifacts.pop(id(pipe), None)
                    logger.warning(
                        "attached TRT artifact did not execute during warmup; "
                        "serving eager"
                    )
            vram_delta = max(0, self._vram_allocated() - vram_before)
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
            # ExecutionLaneGate instead of being job-pinned + eagerly promoted.
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
        slot_refs: Dict[str, str],
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
        ref = slot_refs.get(slot, "")
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
        slot_refs: Optional[Dict[str, str]] = None,
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
        per_ref: Dict[str, Tuple[Any, int]] = {}
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
        cuda_host = torch is not None and torch.cuda.is_available()
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
    def _worker_loaded_slots(spec: EndpointSpec) -> set:
        """Setup slots the WORKER materializes in host RAM (class-typed
        annotations loaded via ``from_pretrained``). str/Path slots and
        engine runtimes (vllm/llama-server) stream weights themselves and
        must not be counted against the host-RAM admission gate."""
        if spec.cls is None or spec.runtime:
            return set()
        setup = getattr(spec.cls, "setup", None)
        if setup is None:
            return set()
        try:
            hints = typing.get_type_hints(setup)
        except Exception:
            return set()
        return {
            name for name, ann in hints.items()
            if isinstance(ann, type) and callable(getattr(ann, "from_pretrained", None))
        }

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
            satisfied: List[Tuple[str, _HostRamBlock]] = []
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
            events: List[Tuple[str, pb.ModelEvent]] = []
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

    async def _clear_host_ram_capacity(self, refs: List[str]) -> None:
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

    async def _ensure_host_ram_for(self, spec: EndpointSpec, paths: Dict[str, str]) -> None:
        """Owner-aware host-RAM admission (gw#407/pgw#541). ``from_pretrained``
        stages the full weight set in host RAM before placement; loading into
        a nearly-full host pushes it into reclaim-thrash that stalls the whole
        process — including gRPC keepalive acks — so the hub disconnects and
        requeues in a livelock (J17: 16 SDXL variants on a 31GB host).

        A warm pipeline is owned by both Residency and its endpoint
        ``_ClassRecord``. Clearing only the Residency reference reports
        ON_DISK while ``record.instance`` still owns every tensor. Evict
        record-owned victims through ``_vacate_record``; only ownerless
        entries may use ``release_to_disk`` directly. Re-probe observed RAM
        after every teardown and fail RETRYABLE if the real headroom still
        cannot cover the incoming bytes plus the derived floor.

        Only worker-loaded (pipeline-typed) slots count: tenant-owned and
        engine-runtime slots do not stage full weight sets in host RAM.

        Multi-slot setups stage SEQUENTIALLY under the load lock — each
        slot's weights move to VRAM (freeing host RAM) before the next slot
        loads — so the honest staging requirement is the LARGEST slot, not
        the sum (gw#479 live: two 28GiB fp8 lanes were refused as "56.2GiB
        incoming" on a 61GiB host that stages at most 28GiB at once)."""
        slots = self._worker_loaded_slots(spec)
        if not paths or not slots:
            return
        incoming = 0
        incoming_refs: List[str] = []
        for slot, p in paths.items():
            if slot in slots:
                slot_bytes = await asyncio.to_thread(disk_gc.tree_bytes, Path(p))
                ref = wire_ref(spec.models[slot])
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
        for ref in res.lru_ram_victims():
            # A previous record teardown may already have transitioned every
            # ref that appeared in the snapshot of LRU candidates.
            if res.tier(ref) is not residency_mod.Tier.RAM:
                continue
            owners = self._records_holding(ref)
            if len(owners) > 1:
                # A ref shared by several endpoint instances is not an
                # ownership key. Their unique refs drive record teardown.
                continue
            rec = owners[0] if owners else None
            if rec is not None:
                if self._record_in_use(rec, reclaim_ref=ref):
                    continue
                owned = [
                    held for held in self._record_refs(rec)
                    if res.tier(held) in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM)
                ]
                released = await self._vacate_record(rec)
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
            ],
            float(spec.resources.vram_gb_hint or 0),
        )
        if needed <= 0:
            return
        # CPU-only workers do not have a VRAM tier to admit against.
        if torch is None or not torch.cuda.is_available():
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
            owners = self._records_holding(ref)
            if len(owners) != 1:
                # Shared refs cannot identify which instance owns the
                # residency object; wait for a unique record-owned victim.
                continue
            rec = owners[0]
            if self._record_in_use(rec, reclaim_ref=ref):
                continue
            await self._vacate_record(rec)
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
            mode = deeper_offload_mode("" if mode == "auto" else mode, floor)
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

    async def _fetch_compile_snapshot(
        self, spec: EndpointSpec, snapshots: Optional[Dict[str, pb.Snapshot]]
    ) -> Optional[_CompileArtifactSelection]:
        """Tensorhub-delivered compiled artifact for this endpoint family.

        Plain acceleration remains optional and explicitly prefers a compatible
        TRT engine (#390) over an Inductor cell. W8A8 delivery is mandatory:
        setup fails retryably before pipeline/GPU load unless Tensorhub attaches
        one exact immutable Forge cell. Returns the selected ref/digest/path or
        ``None`` only for an ordinary eager-compatible lane.
        """
        ccell = spec.compile_cell()
        if ccell is None or not snapshots:
            return None
        family = ccell.family
        # The effective spec is already rebound to this RunJob's selected
        # checkpoints. Snapshot maps also contain attached cells and may carry
        # unrelated/prepositioned models, so they must not choose the lane.
        model_refs = [wire_ref(binding) for binding in spec.models.values()]
        want_execution_lane = self._mandatory_execution_lane_of_bound(model_refs)
        want_bucket = int(ccell.lora_bucket or 0)
        # th#883 pull-by-key: a key-flavored cell is selected only when its
        # key is one this runtime computed for itself (the same candidates
        # the worker advertises in cell_lookups).

        candidate_keys: set[str] = set()
        for execution_lane in (
                (want_execution_lane,) if want_execution_lane else _SPECULATIVE_CELL_BASE_EXECUTION_LANES):
            try:
                candidate_keys.add(cell_key.compute(
                    family, execution_lane, want_bucket,
                    contract=ccell.contract_digest(),
                    regional=bool(ccell.regional),
                ).digest)
            except Exception:
                continue
        if want_execution_lane:
            # TensorRT cells currently expose only their plain fp16 contract.
            # A Forge Inductor cell of the mandated lane is the sole artifact
            # proven to preserve the scaled_mm semantics (gw#534/gw#540).
            candidates = [
                (ref, snap) for ref, snap in snapshots.items()
                if _cell_execution_lane_matches(
                    ref, family, want_execution_lane=want_execution_lane, want_bucket=want_bucket,
                    candidate_keys=candidate_keys)
            ]
        else:
            trt_candidates = [
                (ref, snap) for ref, snap in snapshots.items()
                if trt_engine.is_engine_ref(ref, family)
            ] if not want_bucket else []
            inductor_candidates = [
                (ref, snap) for ref, snap in snapshots.items()
                if _cell_execution_lane_matches(
                    ref, family, want_execution_lane="", want_bucket=want_bucket,
                    candidate_keys=candidate_keys)
            ]
            # Explicit kind policy, then uniqueness within that kind. A map's
            # iteration order never chooses the artifact, while the existing
            # measured plain-lane TRT preference remains intact.
            candidates = trt_candidates or inductor_candidates
        # pgw#672: never re-select an identity whose serve/finalize proof
        # already failed in this process — one boot must not loop
        # adopt-fail on the same cell.
        quarantined = [
            ref for ref, _snap in candidates
            if compile_cache.cell_quarantined_in_process(ref)
        ]
        if quarantined:
            logger.warning(
                "skipping %d compiled-cell candidate(s) quarantined by a "
                "failed proof in this process: %s (pgw#672)",
                len(quarantined), ", ".join(sorted(quarantined)))
            candidates = [
                (ref, snap) for ref, snap in candidates
                if ref not in set(quarantined)
            ]
        candidates = sorted(candidates, key=lambda item: item[0])
        if want_execution_lane and not candidates:
            # gw#587: the fail-closed cell WAIT is retired. A mandatory-lane
            # key with no delivered cell proceeds to load and SELF-MINTS in
            # _enable_compiled (the boot warmup is the mint); the quantized
            # lane's typed refusal now fires only when the mint itself is
            # impossible (fleet_cells._fail_closed).
            logger.info(
                "no %s cell attached for family=%r lora_bucket=%d — "
                "proceeding to self-mint (gw#587)",
                want_execution_lane.upper(), family, want_bucket)
            return None
        if len(candidates) > 1:
            refs = ", ".join(ref for ref, _snap in candidates)
            detail = (
                "multiple compatible compiled artifacts were attached for "
                f"family={family!r} lane={want_execution_lane or 'plain'}: "
                f"{refs}; refusing map-order selection"
            )
            if want_execution_lane:
                # Mandated lanes have no eager-compatible fallback: setup's
                # lane gate must surface this as retryable before GPU load.
                raise compile_cache.CompiledExecutionLaneUnavailableError(detail)
            logger.warning("%s; serving eager", detail)
            return None
        if candidates:
            ref, snap = candidates[0]
            digest = str(snap.digest or "").strip()
            if not digest:
                detail = f"compiled-artifact snapshot {ref!r} has no immutable digest"
                if want_execution_lane:
                    raise compile_cache.CompiledExecutionLaneUnavailableError(detail)
                logger.warning("%s; serving eager", detail)
                return None
            try:
                local = await self.store.ensure_local(ref, snap)
                artifact = compile_cache.find_artifact(local)
                if artifact is None:
                    if want_execution_lane:
                        raise compile_cache.CompiledExecutionLaneUnavailableError(
                            f"{want_execution_lane.upper()} Forge snapshot {ref!r} "
                            "contains no artifact")
                    logger.warning(
                        "compiled-artifact snapshot %s contains no artifact; "
                        "serving eager", ref)
                    return None
                return _CompileArtifactSelection(
                    path=artifact, ref=ref, snapshot_digest=digest)
            except Exception as exc:
                if want_execution_lane and isinstance(
                    exc, compile_cache.CompiledExecutionLaneUnavailableError
                ):
                    raise
                if want_execution_lane:
                    raise compile_cache.CompiledExecutionLaneUnavailableError(
                        f"{want_execution_lane.upper()} Forge snapshot {ref!r} is "
                        f"unusable: {exc}") from exc
                logger.warning(
                    "compiled-artifact snapshot %s unusable (%s); serving eager", ref, exc
                )
        return None

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
        snapshots: Optional[Dict[str, pb.Snapshot]] = None,
        slot_identities: Optional[Dict[str, _ResidencyIdentity]] = None,
        component_paths: Optional[Dict[str, Dict[str, str]]] = None,
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
                overrides = dict((component_paths or {}).get(slot) or {})
                if overrides and not (
                    isinstance(ann, type)
                    and callable(getattr(ann, "from_pretrained", None))
                ):
                    # pgw#617: substitution requires a worker-loaded pipeline
                    # slot; a self-loading str/Path slot never sees components.
                    raise ComponentSubstitutionError(
                        spec.name, slot, sorted(overrides)[0],
                        detail="slot is not worker-loaded (no pipeline-class "
                               "annotation); components cannot substitute")
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
                    ref = wire_ref(binding) if binding is not None else ""
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
                    if overrides:
                        # pgw#617 load-then-substitute: each override component
                        # loads from its OWN materialized tree and rides the
                        # same from_pretrained components= injection as gw#479
                        # shared modules. Unknown names refuse typed at setup.
                        valid = self._model_index_components(path)
                        unknown = sorted(set(overrides) - valid)
                        if unknown:
                            raise ComponentSubstitutionError(
                                spec.name, slot, unknown[0],
                                detail=f"base composition {ref!r} declares "
                                       f"components {sorted(valid)}")
                        for comp in overrides:
                            # An overridden component never rides the shared
                            # cache: its bytes differ from the base's.
                            slot_share.pop(comp, None)
                        from .models.loading import load_component_override

                        for comp, comp_path in sorted(overrides.items()):
                            injected[comp] = await _to_thread_complete(
                                load_component_override, path, comp, comp_path,
                                dtype=str(getattr(binding, "dtype", "") or ""))
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
                    before = self._vram_allocated()
                    try:
                        sl = await _to_thread_complete(
                            provision.load_slot, ann, path, binding=binding,
                            slot=slot, ref=ref, mode=mode, components=injected,
                            declared_vram_gb=float(
                                spec.resources.vram_gb_hint or 0),
                            force_storage_dtype=(
                                "fp8" if slot in force_fp8_slots else ""),
                            strict_vram=bool(spec.resources.strict_vram),
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
                            declared_vram_gb=float(
                                spec.resources.vram_gb_hint or 0),
                            force_storage_dtype=(
                                "fp8" if slot in force_fp8_slots else ""),
                            strict_vram=bool(spec.resources.strict_vram),
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
                        self._record_cast_drop(
                            spec, ref=ref, wanted=sl.pre_drop_wanted,
                            ran=sl.ran, detail=sl.pre_drop_detail)
                    if sl.rung_detail:
                        self._record_adaptive_rung(
                            spec, ref=ref, rung=sl.rung, detail=sl.rung_detail)
                    elif sl.cast_fail_detail:
                        self._record_cast_drop(
                            spec, ref=ref, wanted=sl.cast_fail_wanted,
                            ran=sl.ran, detail=sl.cast_fail_detail)
                    placed = sl.placed
                    if placed.get("oom_demotions"):
                        self._record_demotion(
                            spec, ref=ref, phase="load",
                            from_rung=str(placed.get("requested_mode") or mode),
                            to_rung=str(placed.get("mode") or ""),
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
                        # Opt-in acceleration against a pre-built per-SKU artifact:
                        # a TRT engine (#390, refit with this pipeline's weights)
                        # or an inductor cache (#384). No verified artifact =>
                        # stays eager. ``compile_artifact`` is hub-attached (#569).

                        # pgw#677 reopen: stamp the hub-resolved execution
                        # lane on the pipe BEFORE arming, so the router's
                        # fail_closed and the eager-first eligibility both
                        # read the ONE serveability brain
                        # (compile_cache.mandatory_serving) instead of the
                        # weight-lane prefix. Never overwritten once set.
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
                                compile_selection,
                            )
                        except compile_cache.CompiledExecutionLaneUnavailableError as exc:
                            # Mandatory (w8a8/w4a4) lane: self-mint also hit a
                            # genuine impossibility (no CUDA/toolchain/target).
                            # When this refusal was chained from a caught
                            # cell_selection_bug (th#1031), report it — the
                            # lane refusal must not silently swallow the
                            # loud invariant event.
                            bug = exc.__cause__
                            if isinstance(bug, compile_cache.CellSelectionBugError):
                                await self._report_cell_selection_bug(
                                    spec, compile_selection, bug)
                            raise
                        armed = outcome.armed
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

                                if trt_engine.is_engine_ref(selection.ref):
                                    result.trt_execution_before[id(pipe)] = (
                                        trt_engine.execution_count(pipe))
                                # gw#587 CORRECT FIX: a PendingSelfMint is not
                                # proven or packed yet — the warmup proof
                                # finalizes it (pack + publish) only after
                                # confirming a real compiled call, never before.
                                if hasattr(pipe_mint, "capture_dir"):
                                    result.pending_self_mints[id(pipe)] = pipe_mint
                    delta = max(0, self._vram_allocated() - before)
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
                        if self._arm_execution_lane_gate(pipe, ref, spec=spec):
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
        ref: str,
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
        if self._arm_execution_lane_gate(pipe, ref):
            result.gated_slots.add(slot)
        return execution_lane_obj, execution_lane_bytes

    def _arm_execution_lane_gate(
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
        return arm_execution_lane_gate(pipe, ExecutionLaneGate(
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
        self._record_demotion(
            spec, ref=ref, phase="serve", from_rung="resident",
            to_rung="model_offload",
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
        dynamo router, which is every AOT and TRT arm by construction —
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
        elif trt_engine.is_armed(pipeline):
            arm = "trt"
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

    def _shape_warm_republisher(
        self, spec: EndpointSpec, pipeline: Any,
    ) -> Callable[[], None]:
        """Republish the grown cell after a background novel-shape warm
        (pgw#622). Runs on the Debounce thread, never the serving path."""
        cfg = spec.compile_cell()
        family = str(getattr(cfg, "family", "") or "")

        def republish() -> None:

            cache_dir = self.store._cache_dir
            live_root = (
                Path(cache_dir) if cache_dir
                else Path.home() / ".cache" / "gen-worker"
            ) / "compile-cache"
            fleet_cells.republish_after_shape_warm(
                pipeline, cfg, family, self._cell_publisher(), live_root)

        return republish

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
        # Any armed artifact that is NOT a pending self-mint (delivered
        # cell, TRT engine) keeps today's foreground proof for the whole
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
                # The child owns it; `_delegated_mint_run` resolves it and is
                # the one that must confess if it does not.
                continue
            family = str(getattr(pending, "family", "") or "")
            key = str(getattr(pending, "cell_key", "") or "")
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

    def _mint_budget_ok(
        self, spec: EndpointSpec, inj: "_InjectionResult",
    ) -> bool:
        """pgw#737: does this card have room to CAPTURE, on top of serving?

        False = decline: every pending self-mint is discarded, its target
        unwrapped to true eager and its branch lane dropped, so the boot
        continues as a plain eager boot with the cell absent. Loud (one
        structured ``mint_skipped`` line, logged and on the wire) and
        automatic — a roomier config, or a smaller-resident flavor, mints
        the same cell later."""

        pipes = [
            candidate.pipeline for candidate in inj.compile_objects
            if id(candidate.pipeline) in inj.pending_self_mints
        ]
        device = next(
            (dev for pipe in pipes
             if (dev := mint_budget.device_of(pipe)) is not None),
            None,
        )
        budget = mint_budget.probe(device)
        if budget.fits:
            return True
        line = budget.line("mint_skipped", "insufficient_vram")
        logger.warning("self-mint declined at boot for %s: %s", spec.name, line)
        activity_mod.emit_event(
            "self_mint_skipped",
            f"{line}; {spec.name} boots eager with no cell — the capture "
            "does not fit beside this model's own serving working set",
            phase="insufficient_vram",
        )
        for pipe in pipes:
            pending = inj.pending_self_mints.pop(id(pipe), None)
            inj.active_compile_artifacts.pop(id(pipe), None)
            if pending is not None:
                try:
                    fleet_cells_mod.abandon_self_mint(pending)
                except Exception:
                    logger.exception("declined mint capture cleanup failed")
            try:
                compile_cache.unwrap(pipe)
                if spec.lora_bucket:
                    compile_cache.drop_lora_execution_lane(pipe)
            except Exception:
                logger.exception("declined mint target unwrap failed")
        flush_memory()
        return False

    def serving_tiers(self) -> Dict[str, str]:
        """Per-function serving tier for the capability projection (th#1187
        wire contract): ``"compiled"`` when a READY record's compile target
        covering the function has a proven active artifact, ``"eager"``
        otherwise (including functions without a compile declaration —
        eager by construction). Never returns ``""``: the empty tier is
        reserved for pre-0.65 workers on the wire."""
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
                await self._background_mint_run(rec, bg, act)
                await self._await_publish_durable(act)
        except _MintDeclined as declined:
            # pgw#737: a declined mint is an OUTCOME, not a failure — the
            # activity terminates COMPLETED and the tier stays eager, so
            # nothing downstream classifies this worker as broken or
            # re-dispatches against it. The cell stays absent: a roomier
            # config (or a smaller-resident flavor) mints it later.
            self._abandon_mint_state(rec, bg, free_targets=True)
            logger.warning(
                "self-mint declined for %s: %s", bg.spec.name, declined)
            activity_mod.emit_event(
                "self_mint_skipped", str(declined), phase=declined.reason)
            act.completed()
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
        watching (every PRODUCTION forge pod, hub-launched from a publish or
        demand event) that window is unprotected, and a mint reaped there has
        paid its entire cost and produced nothing.

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

    async def _background_mint_run(
        self, rec: _ClassRecord, bg: "_BackgroundMint", act: Any,
    ) -> None:

        spec = bg.spec
        if _delegated_pendings(bg.pendings):
            # pgw#784: the compile leaves this interpreter. Everything below
            # runs the mint INSIDE the serving process, which is th#1299's
            # contract violation — long-running GIL-holding inductor Python on
            # the one asyncio task that carries both the 10s beat and eager
            # serving. It stays reachable only to red-verify that
            # (GEN_WORKER_MINT_IN_PROCESS=1).
            await self._delegated_mint_run(rec, bg, act)
            return
        # pgw#737: the capture's VRAM pre-budget, BEFORE any seed touches
        # the card. The boot warm has already run one real eager forward on
        # these shapes, so the peak high-water is a measured anchor, not a
        # guess. Not fitting is not an error — decline, serve eager, leave
        # the cell absent (never attempt-and-OOM: that is what took the
        # wan-2.2 tenant request down with 26 banked denoise steps).
        mint_device = next(
            (dev for pipe in bg.pipes.values()
             if (dev := mint_budget.device_of(pipe)) is not None),
            None,
        )
        budget = mint_budget.probe(mint_device)
        if not budget.fits:
            raise _MintDeclined("insufficient_vram", budget)
        jobs, _skips = self._warmup_plan(spec, rec)
        jobs, _mode = warmup_mod.select_runs(jobs, tracing=True)
        if not jobs:
            raise RuntimeError(
                "eager-first mint has no derived warm jobs to seed")
        routers: Dict[int, Any] = {}
        for pid, pipe in bg.pipes.items():
            router = hot_swap.router_of(pipe)
            if router is None:
                raise RuntimeError(
                    "pipeline lost its hot-swap router mid-mint")
            routers[pid] = router

        def _stats() -> Tuple[int, int, int]:
            warm = pending = failed = 0
            for router in routers.values():
                w, p, f = router.stats()
                warm, pending, failed = warm + w, pending + p, failed + f
            return warm, pending, failed

        def _checkpoint() -> None:
            if bg.abandon.is_set():
                raise _MintAbandoned()

        # pgw#672 honesty: a warm process may hold resident compiled code
        # for these class-shared targets from an earlier same-family arm —
        # the router's warm compiles would then capture NOTHING and the
        # finalize below would pack an empty cell. Reset so the background
        # compiles really trace into the pending capture.
        for pipe in bg.pipes.values():
            compile_cache.reset_target_code(pipe)
        exec_before = {
            pid: compile_cache.execution_count(pipe)
            for pid, pipe in bg.pipes.items()
        }
        miss_before = {
            pid: compile_cache.cache_miss_count(pipe)
            for pid, pipe in bg.pipes.items()
        }

        async def _forward(wj: Any, *, preemptible: bool = False) -> None:
            handler_kwargs = await self._handler_kwargs(
                wj.spec, bg.snapshots or {})
            with tempfile.TemporaryDirectory(prefix="gw-bgmint-") as tmp:
                payload = wj.build(tmp)
                if payload is None:
                    return
                ctx: RequestContext[Any] = warmup.warm_context(
                    wj.spec, request_id=f"bg-mint-{wj.spec.name}",
                    local_output_dir=tmp,
                    execution_lane=self._served_execution_lane(wj.spec),
                    config=self._effective_config(wj.spec),
                    # pgw#969: the in-process mint is a mint too — a slot that
                    # cannot resolve here must say which one it killed.
                    origin=_mint_origin(bg, wj.spec))
                if preemptible:
                    bg.seed_ctx = ctx
                    # Registration race: demand that arrived after the turn
                    # was granted but before this ctx existed must still
                    # preempt — check it here, not only at admission.
                    if not self._bg_quiet.is_set():
                        ctx._cancel()
                try:
                    # pgw#677: the seed window forces EAGER routing — a seed
                    # must never pay an inline Dynamo+Inductor compile while
                    # it holds the run gate (the measured 3.5-7 min units).
                    with hot_swap.mint_seed_window():
                        await self._invoke_warmup(
                            wj.spec, bg.instance, ctx, payload,
                            handler_kwargs)
                except CanceledError:
                    if preemptible and ctx.cancelled:
                        raise _SeedPreempted()
                    raise
                finally:
                    bg.seed_ctx = None

        async def _unit(wj: Any) -> bool:
            """One warm forward inside a background GPU turn (pgw#677):
            tenant work always wins the gate — the turn is granted when the
            worker is tenant-idle (or stolen under the minimum-progress
            rule), the forward routes EAGER by construction (seed window),
            and the actual graph compiles run on the router's warm thread
            in their own turns. False = preempted by a tenant arrival; the
            caller re-queues the unit."""
            _checkpoint()
            async with self._bg_turn(rec, "seed", abort=bg.abandon) as stole:
                _checkpoint()
                try:
                    # A stolen turn runs to completion (the steal already
                    # paid its debt); an idle-granted turn yields to any
                    # tenant arrival at the handler's next cancel poll.
                    await _forward(wj, preemptible=not stole)
                except _SeedPreempted:
                    return False
            return True

        async def _seed_pass(jobs_now: List[Any]) -> bool:
            """One full-plan seed pass; preempted units re-queue within the
            pass (a preemption is not plan progress — the pass semantics
            stay 'every unit ran to completion once'). False = the pass was
            CUT SHORT by an OOM backoff — the caller must never treat such
            a pass as converged (pgw#677 reopen: an OOM-truncated plan that
            converged silently is how a partial capture reached finalize at
            unit 8/18 with nothing publishable)."""
            pending_units = list(jobs_now)
            completed = 0
            while pending_units:
                wj = pending_units.pop(0)
                act.phase(
                    activity_mod.PHASE_WARMUP_FORWARD,
                    min(completed + 1, len(jobs_now)), len(jobs_now))
                try:
                    done = await _unit(wj)
                except _MintAbandoned:
                    raise
                except Exception as exc:
                    if not is_cuda_oom(exc):
                        raise
                    logger.warning(
                        "background mint seed %s OOMed (%s); backing off "
                        "this pass", wj.spec.name, exc)
                    activity_mod.emit_event(
                        "self_mint_abort",
                        f"seed pass cut short by CUDA OOM at unit "
                        f"{completed + 1}/{len(jobs_now)} "
                        f"({wj.spec.name}): {exc}; backing off and "
                        "retrying the pass",
                        phase="warmup_oom",
                    )
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return False
                if done:
                    completed += 1
                else:
                    pending_units.append(wj)
            return True

        # Phase 1 — SEED: run the full derived plan through the enabled
        # routers. Every novel signature serves EAGER here and enqueues its
        # background compile into the pending capture; passes repeat until
        # the signature vocabulary is stable (a full queue drops seeds — a
        # later pass re-enqueues them) and no compile is pending.
        seeding_pass = 0
        oom_passes = 0
        while True:
            seeding_pass += 1
            if seeding_pass > _MINT_SEED_MAX_PASSES:
                raise RuntimeError(
                    f"mint seeding did not converge after "
                    f"{_MINT_SEED_MAX_PASSES} passes")
            before = _stats()
            pass_ok = await _seed_pass(jobs)
            act.phase(activity_mod.PHASE_INDUCTOR_COMPILE)
            while True:
                _checkpoint()
                _warm, pending, _failed = _stats()
                if pending == 0:
                    break
                await asyncio.sleep(_MINT_POLL_INTERVAL_S)
            if not pass_ok:
                # pgw#677 reopen: an OOM-truncated pass is NEVER
                # convergence — stats can be stable precisely because the
                # remaining units never ran. Retry (bounded by the pass
                # cap); a persistent OOM ends the mint instead of
                # finalizing a partial capture.
                oom_passes += 1
                # pgw#737: re-budget on the allocator state the OOM just
                # measured (the pass emptied the cache on its way out). A
                # card that no longer has capture headroom must DECLINE
                # here — retrying the pass is how three more OOMs landed on
                # a live tenant request.
                budget = mint_budget.probe(mint_device)
                if not budget.fits:
                    raise _MintDeclined("insufficient_vram_after_oom", budget)
                if oom_passes >= _MINT_OOM_MAX_PASSES:
                    raise _MintDeclined(
                        f"oom_x{oom_passes}", budget,
                        "the warm plan OOMed on consecutive passes; "
                        "refusing to finalize a partial capture — this "
                        "process serves eager")
                continue
            oom_passes = 0
            after = _stats()
            if after == before:
                break
        _warm, _pending, failed_sigs = _stats()
        if failed_sigs:
            raise RuntimeError(
                f"{failed_sigs} signature(s) failed their background "
                "compile; the capture is incomplete")
        dropped = sum(r.seed_dropped for r in routers.values())
        if dropped:
            raise RuntimeError(
                f"{dropped} mint seed signature(s) could not enqueue their "
                "background compile (vocabulary overflow or dummy-batch "
                "failure); the capture would be incomplete")

        # Phase 2 — VERIFY: every signature is warm, so these forwards
        # route COMPILED through the guarded wrappers — the successful
        # compiled call IS the self-mint proof (gw#587: a fresh capture has
        # nothing on disk to HIT against). Real tenant traffic since the
        # swap counts as the same honest evidence.
        act.phase(activity_mod.PHASE_WARMUP_FORWARD, 0, len(jobs))
        def _unproven_pids() -> List[int]:
            return [
                pid for pid, pipe in bg.pipes.items()
                if compile_cache.execution_count(pipe) <= exec_before[pid]
            ]
        for wj in jobs:
            if not _unproven_pids():
                break
            while not await _unit(wj):
                _checkpoint()
        proven_pids = [
            pid for pid in bg.pipes if pid not in set(_unproven_pids())
        ]
        if not proven_pids:
            raise RuntimeError(
                "no compile object served a compiled call after the "
                "background warm; nothing to finalize")

        # Phase 3 — FINALIZE the proven captures (pack; digest computed from
        # the packed bytes), then decide publish per shared capture: a
        # family cell ships only when EVERY sharer proved into it (gw#612).
        act.phase(activity_mod.PHASE_SEAL_PUBLISH)
        finalized: Dict[int, Any] = {}
        for pid in proven_pids:
            pipe = bg.pipes[pid]
            pending_mint = bg.pendings.get(pid)
            if pending_mint is None:
                continue
            miss_delta = max(
                0, compile_cache.cache_miss_count(pipe) - miss_before[pid])
            outcome = fleet_cells_mod.finalize_self_mint(
                pipe, pending_mint, expected_graphs=miss_delta)
            if outcome is None:
                logger.warning(
                    "background mint pack failed for %s; that object stays "
                    "eager", spec.name)
                continue
            finalized[pid] = outcome
            compile_cache.record_cell_proven(str(outcome.ref))
        for pid in list(bg.pipes):
            if pid in finalized:
                continue
            # Unexercised or unpacked on a non-mandatory lane: today's miss
            # policy — unwrap and serve eager, never advertise unproven
            # bytes (gw#586).
            pipe = bg.pipes[pid]
            pending_mint = bg.pendings.get(pid)
            if pending_mint is not None:
                fleet_cells_mod.abandon_self_mint(pending_mint)
            compile_cache.unwrap(pipe)
            if spec.lora_bucket:
                compile_cache.drop_lora_execution_lane(pipe)
        sharers: Dict[int, List[int]] = {}
        mints: Dict[int, Any] = {}
        for pid, pending_mint in bg.pendings.items():
            mints[id(pending_mint)] = pending_mint
            sharers.setdefault(id(pending_mint), []).append(pid)
        for mint_id, pending_mint in mints.items():
            gap = [pid for pid in sharers[mint_id] if pid not in finalized]
            if gap:
                fleet_cells_mod.withhold_self_mint_publish(
                    pending_mint,
                    f"{len(gap)}/{len(sharers[mint_id])} capture-sharing "
                    "compile object(s) never proved into the background "
                    "capture")
            else:
                fleet_cells_mod.publish_self_mint(pending_mint)
        if not finalized:
            raise RuntimeError(
                "background mint finalization produced no advertisable "
                "cell; serving stays eager")

        # Phase 4 — HOT-SWAP the advertisement (shared with the delegated
        # route, pgw#784: an adopted cell is advertised the same way whichever
        # process built it).
        self._advertise_minted_cells(rec, bg, act, finalized)
        # Contract-keyed warm memory (pgw#654): the full plan executed in
        # this process — compiled — so later checkpoint instances of this
        # contract may inherit down to one verification run.
        memory = self._warm_contract_runs.setdefault(
            self._warm_contract_key(spec), set())
        memory.update(wj.graph_key for wj in jobs)
        logger.info(
            "background mint for %s armed: %d compile object(s) hot-swapped "
            "to compiled (tier flips in the next capability projection)",
            spec.name, len(finalized))

    def _advertise_minted_cells(
        self, rec: _ClassRecord, bg: "_BackgroundMint", act: Any,
        finalized: Dict[int, Any],
    ) -> None:
        """Activate a finalized self-mint identity on the live targets.

        State stays READY throughout — the tier flips eager->compiled in the
        next capability projection — and pgw#622 stays alive for post-mint
        novel shapes. Shared by the in-process and delegated routes (pgw#784):
        the artifact SOURCE differs, what it means to advertise one does not.
        """

        spec = bg.spec
        act.phase(activity_mod.PHASE_FINALIZE)
        # pgw#824: the eager posture is DISCHARGED — this record now serves
        # from a cell. Left behind, a stale token would misattribute a later,
        # unrelated un-arm (guard revocation) to whatever declined at boot.
        rec.eager_posture = ""
        for pid, outcome in finalized.items():
            pipe = bg.pipes[pid]
            for target in rec.compile_targets.values():
                if target.pipeline is not pipe:
                    continue
                with target.state_lock:
                    target.active_compile_ref = str(outcome.ref)
                    target.active_compile_snapshot_digest = str(
                        outcome.snapshot_digest)
                    target.active_self_mint = True
                # pgw#686: the mint stamped the pipe's lane; re-advertise so
                # the requested key names the key just published (the fleet
                # adopts by requested key — a stale advertisement leaves the
                # published cell unreachable), then assert the invariant.
                self._refresh_compile_target(target)
                self._warn_cell_key_divergence(spec.name, target)
                if not self._bind_compile_guard(rec, target):
                    with target.state_lock:
                        target.active_compile_ref = ""
                        target.active_compile_snapshot_digest = ""
                    logger.warning(
                        "compile target %s has no runtime guard revocation "
                        "signal; advertising eager", target.incarnation_id)
                    continue
            hot_swap.enable(
                pipe,
                on_warmed=hot_swap.Debounce(
                    self._shape_warm_republisher(spec, pipe)))

    async def _delegated_mint_run(
        self, rec: _ClassRecord, bg: "_BackgroundMint", act: Any,
    ) -> None:
        """pgw#784: build every owed cell in a CHILD PROCESS, then advertise.

        The delegated twin of ``_background_mint_run``, and far shorter,
        because the phases that used to live here — seed, drain the queued
        compiles, prove, pack — are the child's now. What stays is what only a
        serving worker can do: keep serving eager and beating while it happens,
        adopt the result through the DELIVERED-cell path, decide publish on
        gw#612's sibling-coverage rule, and advertise the identity.

        Raises exactly what ``_background_mint_run`` raises, so
        ``_background_mint``'s outcome handling is untouched: ``_MintDeclined``
        (an OUTCOME — tier stays eager, cell absent), ``_MintAbandoned``
        (adopt-on-arm / vacate / shutdown), or a plain ``Exception`` (a failed
        mint). Serving continues in every branch: the worker never dies with
        its mint.
        """

        spec = bg.spec
        # One child per DISTINCT pending: sibling pipes of one record whose
        # axes compute the same key share ONE cell (the qwen edit shape), and
        # the child mints their union exactly once.
        sharers: Dict[int, List[int]] = {}
        for pid, pending in bg.pendings.items():
            sharers.setdefault(id(pending), []).append(pid)
        if not sharers:
            raise RuntimeError("delegated mint has no pending cell to build")

        finalized: Dict[int, Any] = {}
        declined: Optional[_MintDeclined] = None
        # pgw#999: every classified refusal this run saw, so the terminal
        # RuntimeError names them instead of restating "no advertisable cell".
        declined_reasons: List[str] = []
        for pids in sharers.values():
            pending = bg.pendings[pids[0]]
            pipe = bg.pipes[pids[0]]
            result = await mint_delegate.build_cell(
                mint_delegate.MintTask(
                    pending=pending,
                    pipe=pipe,
                    function=spec.name,
                    modules=bg.modules or _mint_modules(spec),
                    slots=dict(bg.slots),
                    weight_lane=compile_cache.cell_base_execution_lane(pipe),
                    execution_lane=self._served_execution_lane(spec),
                    configs={spec.name: self._effective_config(spec)},
                    device=mint_budget.device_of(pipe),
                ),
                act=act, abandon=bg.abandon)
            if result.status == mint_delegate.ABANDONED:
                raise _MintAbandoned()
            if result.declined:
                # Remembered, not raised yet: another pending may still fit.
                declined = _MintDeclined(
                    "insufficient_vram",
                    result.budget or mint_budget.probe(),
                    result.detail)
                continue
            minted = result.minted
            if not result.ok or minted is None:
                logger.warning(
                    "delegated mint for %s produced no adoptable cell (%s); "
                    "that object stays eager", spec.name, result.detail)
                # pgw#815: resolve the obligation instead of dropping it —
                # a `continue` here left the pending with no terminus and no
                # wire trace whenever a SIBLING pending succeeded (the
                # `if not finalized: raise` below never fires then).
                fleet_cells_mod.abandon_self_mint(pending)
                # pgw#999: `phase` carries the CLASSIFIED reason when the
                # child's cell was built and then refused arming; it falls
                # back to the call-site token only when there is genuinely no
                # classification (no cell was produced at all).
                activity_mod.emit_event(
                    "self_mint_abort",
                    f"family={pending.family} key={pending.cell_key}: the "
                    f"delegated child produced no adoptable cell "
                    f"({result.detail or result.status}); this object stays "
                    f"eager and nothing is published",
                    phase=result.reason or "delegated_no_cell",
                )
                declined_reasons.append(result.reason or result.status)
                continue
            for pid in pids:
                finalized[pid] = minted
            compile_cache.record_cell_proven(str(minted.ref))

        if not finalized:
            if declined is not None:
                raise declined
            raise RuntimeError(
                "delegated mint produced no advertisable cell; serving stays "
                "eager"
                + (f" (refused: {', '.join(sorted(set(declined_reasons)))})"
                   if declined_reasons else ""))

        # Publish per shared cell on gw#612's rule: a family cell ships only
        # when EVERY sharer is covered by it — a partial pack bricks every
        # adopting boot at the gw#607 per-object proof.
        for pids in sharers.values():
            pending = bg.pendings[pids[0]]
            gap = [pid for pid in pids if pid not in finalized]
            if gap:
                fleet_cells_mod.withhold_self_mint_publish(
                    pending,
                    f"{len(gap)}/{len(pids)} cell-sharing compile object(s) "
                    "were not covered by the delegated mint")
            else:
                fleet_cells_mod.publish_self_mint(pending)

        self._advertise_minted_cells(rec, bg, act, finalized)
        logger.info(
            "delegated mint for %s armed: %d compile object(s) hot-swapped to "
            "compiled — this worker served eager and beat at its normal "
            "cadence for the whole mint (pgw#784)",
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
        bug_ref = compile_selection.ref if compile_selection is not None else ""
        bug_digest = (
            compile_selection.snapshot_digest
            if compile_selection is not None else "")
        logger.error("cell_selection_bug on %s (%s): %s", spec.name, bug_ref, exc)
        await self._send(pb.WorkerMessage(
            model_event=self.store.model_event(
                bug_ref,
                pb.MODEL_STATE_FAILED,
                identity=((bug_digest, 0) if bug_digest else None),
                error=f"cell_selection_bug: {str(exc)[:300]}",
            )
        ))

    def _enable_compiled(
        self, pipe: Any, cfg: Any, artifact: Optional[Path],
        delivered: Optional["_CompileArtifactSelection"] = None,
    ) -> "fleet_cells.ArmOutcome":
        """Arm the best available compiled path for a freshly loaded pipeline.

        gw#587: delivered cell first — a th#1031 ``cell_selection_bug``
        (self-requested cell fails contract_drift) is reported loudly but no
        longer fatal: this falls through to SELF-MINT exactly like an
        ordinary miss. The boot warmup compiles the real serving graphs
        once, serves compiled immediately, and publishes through the hub's
        attested gate so the next worker on this key is store-served. Eager
        fallback and the fail-closed cell wait are gone for reachable mints;
        genuine mint impossibilities keep the old miss policy (plain=eager,
        quantized=typed refusal).

        Returns the fleet ``ArmOutcome``; a ``self_mint`` result is recorded
        into ``active_compile_artifacts`` exactly like a delivered cell so
        the warmup proof runs and the target advertises the worker's own
        key ref (th#910 self-attested dispatch fence)."""

        return fleet_cells.enable_compiled(
            pipe, cfg, self.store._cache_dir, artifact,
            publisher=self._cell_publisher(),
            delivered_ref=delivered.ref if delivered else "",
            delivered_digest=delivered.snapshot_digest if delivered else "",
        )

    def _arming_enable(
        self, pipe: Any, cfg: Any, cache_dir: Optional[Path],
        artifact: Optional[Path],
    ) -> "fleet_cells.ArmOutcome":
        """ArmingScope adapter: a self-loaded pipeline's ``arm_compile()``
        gets the same fleet policy (delivered cell first, self-mint on miss)
        as a worker-loaded slot. ``cache_dir`` comes from the scope, which
        the executor constructed with its own store cache dir."""

        return fleet_cells.enable_compiled(
            pipe, cfg, cache_dir, artifact,
            publisher=self._cell_publisher(),
        )

    @staticmethod
    def _vram_allocated() -> int:
        if torch is not None and torch.cuda.is_available():
            try:
                return int(torch.cuda.memory_allocated())
            except Exception:
                return 0
        return 0

    def background_mint_tasks(self) -> List["asyncio.Task"]:
        """th#1359: every still-running background mint on this worker.

        A forge pod joins these before it calls itself done — a release may
        declare more than one compile family, and retiring on the first one to
        finish would abandon the rest, which is the exact failure (pgw#846
        attempt sixteen: `self_mint_abort/abandoned_shutdown`) this mode
        exists to make impossible.
        """
        tasks: List["asyncio.Task"] = []
        for rec in self._classes.values():
            bg = rec.background_mint
            task = getattr(bg, "task", None) if bg is not None else None
            if task is not None and not task.done():
                tasks.append(task)
        return tasks

    def declares_compile(self) -> bool:
        """Whether ANY discovered spec declares a compile family.

        A forge pod for a release that declares none has nothing to mint, and
        must say `nothing_owed` and retire rather than sit on a paid card.
        """
        return any(
            s.compile is not None and getattr(s.compile, "family", "")
            for s in self.specs.values()
        )

    async def shutdown_instances(self) -> None:
        for rec in self._classes.values():
            await self.abandon_background_mint(
                rec, reason="worker shutdown", code="shutdown")
            inst, rec.instance, rec.ready = rec.instance, None, False
            rec.compile_targets.clear()
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
        self._on_state_change()

    # ---- Compile-cache adoption -------------------------------------------

    async def handle_model_op(self, op: pb.ModelOp) -> None:
        """Handle the sole v3 ModelOp: hot adoption of a compile cache."""
        if op.op != pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE:
            return
        blocker_intent_id = self._compile_cache_adoption_active
        if self.intent_registry is not None and blocker_intent_id:
            intent_id = self.intent_registry.ensure_local_intent(
                "compile-adopt-waiter",
                op.operation_id or f"{op.ref}\0{id(asyncio.current_task())}",
                detail=f"waiting to adopt compile artifact {op.ref}",
            )
        else:
            intent_id = self._adoption_intent(op)
            self._compile_cache_adoption_active = intent_id
        self._intent_transition(
            intent_id,
            pb.LIFECYCLE_INTENT_STATUS_RUNNING,
            pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
        )
        try:
            async with self._intent_lock(
                intent_id,
                self._compile_cache_adoption_lock,
                operation=f"compile adoption single-flight for {op.ref}",
                stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
                reason=pb.LIFECYCLE_WAIT_REASON_SINGLE_FLIGHT_OWNER,
                resume_stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                blocker_intent_id=blocker_intent_id,
            ):
                self._compile_cache_adoption_active = intent_id
                await self._handle_compile_cache_adoption(
                    op,
                    intent_id=intent_id,
                )
        finally:
            if self._compile_cache_adoption_active == intent_id:
                self._compile_cache_adoption_active = ""

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
        that actually happens is a boot attach, and a boot attach reported
        itself in prose (`aot_adopt`, `duration_ms=0`) on a lane with no
        numbers in it. Two builders, one fact, and only the unmeasured builder
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
                # An arm with no candidate identity is not an adoption anyone
                # can attribute; recording it would add a row that answers
                # nothing. (The hub applies the same rule from its side.)
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

    def _adoption_event(
        self,
        ref: str,
        state: "pb.ModelState",
        snapshot_digest: str,
        operation_id: str,
        target_incarnation_id: str,
        **kw: Any,
    ) -> pb.ModelEvent:
        """Build terminal evidence for one orchestrator-minted adoption op."""
        identity = (snapshot_digest, 0) if snapshot_digest else None
        return self.store.model_event(
            ref,
            state,
            identity=identity,
            operation_id=operation_id,
            target_incarnation_id=target_incarnation_id,
            **kw,
        )

    async def _handle_compile_cache_adoption(
        self,
        op: pb.ModelOp,
        *,
        intent_id: str = "",
    ) -> None:
        self.store.bind_loop()
        ref = op.ref
        snap = op.snapshot if op.HasField("snapshot") else None
        snapshot_digest = snap.digest if snap is not None else ""
        operation_id = op.operation_id
        target_incarnation_id = op.target_incarnation_id
        if not operation_id.strip():
            await self._send(pb.WorkerMessage(
                model_event=self._adoption_event(
                    ref,
                    pb.MODEL_STATE_FAILED,
                    snapshot_digest,
                    operation_id,
                    target_incarnation_id,
                    error="adopt_failed:missing_operation_id",
                )
            ))
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail="missing operation_id",
            )
            return
        if not snapshot_digest.strip():
            # Adoption is one-shot evidence for one immutable artifact.  A
            # mutable ref (or the resident identity for that ref) cannot
            # identify which bytes this operation actually used.
            await self._send(pb.WorkerMessage(
                model_event=self._adoption_event(
                    ref,
                    pb.MODEL_STATE_FAILED,
                    snapshot_digest,
                    operation_id,
                    target_incarnation_id,
                    error="adopt_failed:missing_snapshot_digest",
                )
            ))
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail="missing snapshot digest",
            )
            return
        if not target_incarnation_id.strip():
            await self._send(pb.WorkerMessage(
                model_event=self._adoption_event(
                    ref,
                    pb.MODEL_STATE_FAILED,
                    snapshot_digest,
                    operation_id,
                    target_incarnation_id,
                    error="adopt_failed:missing_target_incarnation_id",
                )
            ))
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail="missing target incarnation id",
            )
            return
        try:
            await self._adopt_compile_cache(
                ref,
                snap,
                snapshot_digest,
                operation_id,
                target_incarnation_id,
                intent_id=intent_id,
            )
        except Exception as exc:
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_ADOPTING,
                detail=_sanitize(str(exc))[:512],
            )
            logger.warning("compile-cache adoption on %s failed: %s", ref, exc)
            await self._send(pb.WorkerMessage(
                model_event=self._adoption_event(
                    ref,
                    pb.MODEL_STATE_FAILED,
                    snapshot_digest,
                    operation_id,
                    target_incarnation_id,
                    error=(
                        f"adopt_failed:{type(exc).__name__.lower()}: "
                        f"{str(exc)[:300]}"
                    ),
                )
            ))

    async def _adopt_compile_cache(
        self,
        ref: str,
        snap: Optional[pb.Snapshot],
        snapshot_digest: str,
        operation_id: str,
        target_incarnation_id: str,
        *,
        intent_id: str = "",
    ) -> None:
        """Hot adoption (th#567): download+verify a compiled artifact and
        re-wrap the already-resident modules in place — weights untouched, no
        reload, one warmup. Handles BOTH cell kinds on the same rails: an
        inductor cache (#384: seed dirs + torch.compile) and a TRT engine
        (#390: deserialize + refit with the resident weights + module swap).
        ANY failure => stay eager and report ``adopt_failed:<reason>``;
        adoption must never degrade service.

        pgw#735: THREE cell kinds ride these rails now — the exported
        (``aot-inductor``) lane joins inductor and TRT, and proves adoption by
        its own artifact invocations rather than by an FX cache hit it can
        never produce."""

        t0 = time.monotonic()
        staged_artifact: Any = None
        # th#883: once the staged artifact's axes are proven to describe
        # exactly the key this target computed for itself, arm failures in
        # the selection/parity vocabulary are BY CONSTRUCTION bugs in the
        # one worker-owned brain and surface as their own loud event class.
        self_requested = False
        _SELECTION_BUG_REASONS = ("key_mismatch", "no_target", "lane_apply")

        async def fail(reason: str, detail: str = "") -> None:
            nonlocal staged_artifact
            if staged_artifact is not None:
                await asyncio.to_thread(staged_artifact.close)
                staged_artifact = None
            logger.warning("compile-cache adopt %s failed: %s %s", ref, reason, detail)
            # gw#577: terminal refusals carry the exact mismatch (axis +
            # cell-vs-runtime values) on the wire — pods expose no logs. The
            # th#875 transient vocabulary stays bare: the hub re-arm matcher
            # compares those four statuses EXACTLY.
            error = f"adopt_failed:{reason}"
            if (
                self_requested
                and reason in _SELECTION_BUG_REASONS
                # low_vram prep mode is dynamic placement, outside the key:
                # its drift is a legitimate miss, never the bug class.
                and "low_vram_mode" not in detail
            ):
                logger.error(
                    "cell_selection_bug on %s: %s %s", ref, reason, detail)
                error = f"cell_selection_bug:{reason}"
            if detail and reason not in (
                "model_in_use", "target_not_ready", "target_replaced", "download",
            ):
                error = f"{error}: {detail[:300]}"
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_ADOPTING,
                detail=_sanitize(error)[:512],
            )
            await self._send(
                pb.WorkerMessage(
                    model_event=self._adoption_event(
                        ref,
                        pb.MODEL_STATE_FAILED,
                        snapshot_digest,
                        operation_id,
                        target_incarnation_id,
                        error=error,
                    )
                )
            )

        family = compile_cache.family_from_ref(ref)
        is_trt = trt_engine.is_engine_ref(ref)
        # pgw#735: an EXPORTED cell is a third artifact kind, proven its own
        # way below. Without this it fails `bad_ref` before it is ever armed.
        is_aot = aot_serve.is_aot_ref(ref)
        if not family or not (
            is_trt or is_aot or compile_cache.is_cache_ref(ref)
        ):
            return await fail("bad_ref")
        found = self._compile_target(target_incarnation_id)
        if found is None:
            return await fail("target_not_ready")
        expected_rec, expected_target = found
        target_family = str(
            getattr(expected_target.spec.compile, "family", "") or ""
        ).strip()
        if target_family != family:
            return await fail("target_family_mismatch")
        with expected_target.state_lock:
            previous_ref = expected_target.active_compile_ref
            previous_digest = expected_target.active_compile_snapshot_digest
            cell_quarantined = (
                (ref, snapshot_digest)
                in expected_target.failed_compile_identities
            )
        if cell_quarantined:
            return await fail(
                "cell_quarantined",
                "this immutable cell already failed its runtime guard on "
                "the exact live target",
            )
        if previous_ref == ref and previous_digest == snapshot_digest:
            # Replayed/reconnected operation for the exact already-proven
            # artifact: acknowledge without another wrap or warmup, and retain
            # the latest causal operation identity for a later guard failure.
            with expected_target.state_lock:
                expected_target.active_adoption_operation_id = operation_id
            await self._send(
                pb.WorkerMessage(
                    model_event=self._adoption_event(
                        ref,
                        pb.MODEL_STATE_ADOPTED,
                        snapshot_digest,
                        operation_id,
                        target_incarnation_id,
                        duration_ms=0,
                    )
                )
            )
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                pb.LIFECYCLE_INTENT_STAGE_READY,
                actual_digest=snapshot_digest.encode(),
            )
            return
        if previous_ref:
            # Replacing any already-active wrapper is not transactional:
            # applying the new graph first unwraps the old one, and a failed
            # warmup cannot promise a lossless restore. Never report the old
            # artifact as active after removing it. Tensorhub vacates/reloads
            # this incarnation for same-ref republish or kind/ref changes.
            return await fail("active_replace_requires_reload")
        if self.in_flight_keys():
            # The hub schedules adoption idle-only; defensive — never touch
            # a module while any job is in flight.
            return await fail("model_in_use")
        # pgw#671 adopt-on-arm: a peer's upload armed this cell while our own
        # background mint was building — adopt, abandoning our build cleanly
        # (opportunistic adoption, never wait-for-peer). The router is also
        # suspended so the adoption's proof warmup keeps its sequential
        # semantics (an eager route would falsify the proof).
        await self.abandon_background_mint(
            expected_rec, reason=f"adopting peer cell {ref}",
            code="adopt_on_arm")
        adopt_router = hot_swap_mod.router_of(expected_target.pipeline)
        if adopt_router is not None:
            adopt_router.suspend()
        materialize_intent = self.store._materialize_intent(ref)
        self._intent_transition(
            intent_id,
            pb.LIFECYCLE_INTENT_STATUS_WAITING,
            pb.LIFECYCLE_INTENT_STAGE_WAIT_SNAPSHOT,
            reason=pb.LIFECYCLE_WAIT_REASON_SNAPSHOT,
            blocker_intent_id=materialize_intent,
        )
        try:
            local = await self.store.ensure_local(ref, snap)
        except Exception as exc:
            return await fail("download", str(exc))
        self._intent_transition(
            intent_id,
            pb.LIFECYCLE_INTENT_STATUS_RUNNING,
            pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
        )
        artifact = compile_cache.find_artifact(local)
        if artifact is None:
            return await fail("artifact_missing")
        if not is_trt and not is_aot:
            try:
                # Expensive extraction and runtime-key verification happen in
                # an isolated tree before taking model/GPU locks. Activation
                # and wrapper installation remain one serialized transaction.
                staged_artifact = await asyncio.to_thread(
                    compile_cache.stage_artifact,
                    artifact,
                    family,
                    self.store._cache_dir,
                )
            except compile_cache.AdoptError as exc:
                return await fail(exc.reason, str(exc))
            except Exception as exc:
                return await fail("artifact_invalid", str(exc))
            with expected_target.state_lock:
                want_key = expected_target.requested_cell_key
            if want_key and staged_artifact is not None:

                self_requested = not cell_key.mismatch(
                    staged_artifact.metadata, want_key)

        # Artifact work may take long enough for model juggling to replace the
        # object. Serialize the final check + mutation with setup/vacate, and
        # address only the exact incarnation observed before the download.
        async with self._intent_lock(
            intent_id,
            self._load_lock,
            operation=f"compile adoption load lock for {ref}",
            stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
            reason=pb.LIFECYCLE_WAIT_REASON_LOAD_LOCK,
            resume_stage=pb.LIFECYCLE_INTENT_STAGE_ADOPTING,
        ):
            current = self._compile_target(target_incarnation_id)
            if (
                current is None
                or current[0] is not expected_rec
                or current[1] is not expected_target
            ):
                return await fail("target_replaced")
            if self.in_flight_keys():
                return await fail("model_in_use")

            # A job landing mid-adoption queues behind every GPU permit;
            # process-global Inductor counters cannot tolerate another slot
            # compiling inside this exact target's proof window.
            async with self._exclusive_gpu(
                intent_id,
                resume_stage=pb.LIFECYCLE_INTENT_STAGE_ADOPTING,
            ):
                current = self._compile_target(target_incarnation_id)
                if (
                    current is None
                    or current[0] is not expected_rec
                    or current[1] is not expected_target
                ):
                    return await fail("target_replaced")

                rec, target = current
                spec = target.spec
                cfg = spec.compile_cell()
                assert cfg is not None
                obj = target.pipeline
                wrapped = False
                execution_lane_applied = False
                trt_before = trt_engine.execution_count(obj) if is_trt else 0
                aot_before = aot_serve.execution_count(obj) if is_aot else 0
                inductor_before = (0, 0, 0)

                async def rollback() -> None:
                    """Return a first-time failed adoption to honest eager."""
                    if is_trt and wrapped:
                        trt_engine.unwrap(obj)
                    if is_aot and wrapped:
                        aot_serve.unwrap(obj)
                    if wrapped:
                        if not is_trt and not is_aot:
                            compile_cache.unwrap(obj)
                    if execution_lane_applied:
                        compile_cache.drop_lora_execution_lane(obj)
                    live = self._compile_target(target_incarnation_id)
                    if live is not None and live[0] is rec and live[1] is target:
                        self._refresh_compile_target(target)
                        self._on_state_change()

                bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
                if bucket and not is_trt and not is_aot:
                    try:
                        compile_cache.apply_lora_execution_lane(obj, bucket)
                        execution_lane_applied = True
                    except Exception as exc:
                        await rollback()
                        return await fail("lane_apply", str(exc))

                if is_trt:
                    try:
                        await asyncio.to_thread(
                            trt_engine.load_and_wrap, obj, cfg,
                            artifact, self.store._cache_dir,
                        )
                        wrapped = True
                    except compile_cache.AdoptError as exc:
                        await rollback()
                        return await fail(exc.reason, str(exc))
                    except Exception as exc:
                        await rollback()
                        return await fail("artifact_invalid", str(exc))
                elif is_aot:
                    # pgw#734: HOT adoption of an exported cell. Boot arming
                    # already dispatches by kind in provision.enable_compiled;
                    # this path did not, so a .pt2 delivered to a RUNNING
                    # worker was handed to the dynamo stager and unpacked as an
                    # inductor cache tree. Same rails, same fail-closed
                    # classification — its own backend.
                    try:
                        await asyncio.to_thread(
                            aot_serve.load_and_wrap, obj, cfg,
                            artifact, self.store._cache_dir,
                        )
                        wrapped = True
                    except compile_cache.AdoptError as exc:
                        await rollback()
                        return await fail(exc.reason, str(exc))
                    except Exception as exc:
                        await rollback()
                        return await fail("artifact_invalid", str(exc))
                else:
                    assert staged_artifact is not None
                    try:
                        # Exact graph/lane parity is checked against this one
                        # live object, never every resident family member.
                        await asyncio.to_thread(
                            compile_cache.arm_staged_artifact,
                            obj,
                            cfg,
                            staged_artifact,
                        )
                        staged_artifact = None
                    except compile_cache.AdoptError as exc:
                        await rollback()
                        return await fail(exc.reason, str(exc))
                    except Exception as exc:
                        await rollback()
                        return await fail("artifact_invalid", str(exc))
                    wrapped = True

                if not is_trt and not is_aot:
                    inductor_before = (
                        compile_cache.execution_count(obj),
                        compile_cache.cache_hit_count(obj),
                        compile_cache.cache_miss_count(obj),
                    )
                warm_t0 = time.monotonic()
                warmup = getattr(rec.instance, "warmup", None)
                proven_function_names: set[str] = set()
                try:
                    if callable(warmup):
                        if asyncio.iscoroutinefunction(warmup):
                            await warmup()
                        else:
                            await asyncio.to_thread(warmup)
                        warmed = 1
                    else:
                        # Real FLUX/Z/SDXL endpoints use the decorator warmup
                        # contract rather than a custom instance method. Reuse
                        # the same production planner/invocation path as setup.
                        warmup_evidence = await self._run_synthesized_warmup(
                            spec,
                            rec,
                            rec.instance,
                            None,
                            proof_objects=(obj,),
                        )
                        warmed = warmup_evidence.count
                        proven_function_names.update(
                            warmup_evidence.functions_by_object.get(id(obj), set()))
                except Exception as exc:
                    await rollback()
                    return await fail("warmup", f"{type(exc).__name__}: {exc}")
                warmup_s = round(time.monotonic() - warm_t0, 3)
                # pgw#797 / th#1329: warmup AFTER arming a cell. Same quantity
                # the hub stores as `worker_activity_events.warmup_ms` on the
                # adopt event, recorded here as a boot row too so the ladder
                # and the adopt event answer "what does an armed cell still pay
                # on warmup" identically instead of by two derivations. Always
                # armed=1 by construction — this runs after the wrap.
                if boot_mod.in_boot() and warmed:
                    # pgw#924: `and warmed`. A warmup that ran no forward has
                    # no cost to report, and a zero-duration row here reads
                    # identically to "an armed cell warms instantly" — the
                    # exact reading 240 live rows of `duration_ms=0` invited.
                    boot_mod.mark(
                        boot_mod.PHASE_WARMUP,
                        duration_ms=int(round(warmup_s * 1000)),
                        function=spec.name,
                        ref=ref,
                        klass=boot_mod.CLASS_LOAD,
                        detail=f"armed=1 minting=0 forwards={warmed} adopt=1",
                    )
                hits = 0
                misses = 0

                if not is_trt and not is_aot:
                    calls = compile_cache.execution_count(obj) - inductor_before[0]
                    hits = compile_cache.cache_hit_count(obj) - inductor_before[1]
                    misses = compile_cache.cache_miss_count(obj) - inductor_before[2]
                    if not warmed:
                        await rollback()
                        return await fail(
                            "no_warmup",
                            "target defines no runnable warmup; cache hits unprovable")
                    if calls <= 0:
                        # gw#595: distinct from a genuine miss — the warmup
                        # has no modality that exercises this object at all,
                        # so the cell is unprovable on this target rather
                        # than disproven.
                        await rollback()
                        return await fail("no_warmup_modality", (
                            "no warmup modality exercises this object "
                            f"(calls=0, warmup={warmup_s}s); cell unprovable "
                            "on this target"))
                    if hits <= 0:
                        await rollback()
                        return await fail("cache_miss", (
                            "exact target warmup did not execute a cache-hit "
                            f"compiled graph (calls={calls}, hits={hits}, "
                            f"misses={misses}, warmup={warmup_s}s) — cell useless "
                            f"on this runtime, serving eager"))
                elif is_trt and trt_engine.execution_count(obj) <= trt_before:
                    await rollback()
                    return await fail(
                        "engine_not_executed",
                        "warmup did not execute the attached TRT engine",
                    )
                elif is_aot and not aot_serve.proven_since(obj, aot_before):
                    # pgw#735: the exported lane's own proof. An artifact that
                    # never ran, or that ran and then revoked (B1/B2 refusal),
                    # is NOT adopted — same fail-closed posture as a dynamo
                    # cell with zero cache hits, never a synthesized hit.
                    await rollback()
                    return await fail(
                        "artifact_not_executed", (
                            "warmup did not execute the attached exported "
                            f"artifact (calls={aot_serve.execution_count(obj) - aot_before}, "
                            f"armed={aot_serve.is_armed(obj)}, "
                            f"ingress_refusals={aot_serve.ingress_refusals(obj)})"
                        ),
                    )
                if callable(warmup):
                    # A custom object warmup has no sibling-handler identity.
                    proven_function_names.add(spec.name)
                advertised_function_names = set(target.function_names)
                if proven_function_names != advertised_function_names:
                    await rollback()
                    return await fail(
                        "function_alias_unproven",
                        "warmup proof does not equal the immutable advertised "
                        f"handler aliases (advertised={sorted(advertised_function_names)!r} "
                        f"proven={sorted(proven_function_names)!r})",
                    )

                current = self._compile_target(target_incarnation_id)
                if current is None or current[0] is not rec or current[1] is not target:
                    await rollback()
                    return await fail("target_replaced")
                self._refresh_compile_target(target)
                if not self._bind_compile_guard(rec, target):
                    await rollback()
                    return await fail(
                        "guard_unbound",
                        "compiled wrapper has no runtime revocation signal",
                    )
                with target.state_lock:
                    target.active_compile_ref = ref
                    target.active_compile_snapshot_digest = snapshot_digest
                    target.active_adoption_operation_id = operation_id
                self._on_state_change()

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "compile-cache adopt %s: adopted in %dms (fxgraph hits=%d misses=%d, "
            "warmup %.1fs)", ref, duration_ms, hits, misses, warmup_s)
        if misses:
            logger.warning(
                "compile-cache adopt %s: %d fxgraph misses during warmup — "
                "cell covers the declared shapes only partially", ref, misses)
        await self._send(pb.WorkerMessage(
            model_event=self._adoption_event(
                ref,
                pb.MODEL_STATE_ADOPTED,
                snapshot_digest,
                operation_id,
                target_incarnation_id,
                duration_ms=duration_ms,
                cache_hits=hits,
                cache_misses=misses,
                warmup_s=warmup_s,
            )
        ))
        self._intent_transition(
            intent_id,
            pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
            pb.LIFECYCLE_INTENT_STAGE_READY,
            actual_digest=snapshot_digest.encode(),
        )

    def _record_refs(self, rec: _ClassRecord) -> List[str]:
        """The wire refs a record's instance holds: the load-time booking
        keys when stamped (gw#494), else the current binding derivation
        (records that never completed a setup)."""
        if rec.held_refs:
            return list(rec.held_refs)
        return [wire_ref(b) for s in rec.specs for b in s.models.values()]

    def _records_holding(self, ref: str) -> List[_ClassRecord]:
        return [
            rec for rec in self._classes.values()
            if rec.ready and ref in self._record_refs(rec)
        ]

    def _record_in_use(
        self, rec: _ClassRecord, *, reclaim_ref: Optional[str] = None,
    ) -> bool:
        """Whether teardown would disturb live work.

        ``reclaim_ref`` narrows a pressure-driven teardown to the candidate
        that selected this record. A different held ref can be pinned by an
        incoming job before its own setup (the common SDXL VAE); that does not
        make this record's idle checkpoint active. ``_vacate_record`` leaves
        such a pinned ref resident because ``release_to_disk`` refuses it.
        Full-record invalidation omits the argument and remains conservative.

        A job on a rebound spec no longer references the record's held refs;
        membership of the job's spec in this record is the honest instance-use
        signal (gw#494).
        """
        for job in self.jobs.values():
            if job.finished or job.superseded or job.spec is None:
                continue
            if job.spec in rec.specs:
                return True
        refs = [reclaim_ref] if reclaim_ref is not None else self._record_refs(rec)
        for ref in refs:
            owners = self._records_holding(ref)
            if (len(owners) == 1 and owners[0] is rec
                    and self.store.residency.in_use(ref)):
                return True
        return False

    async def _vacate_record(self, rec: _ClassRecord) -> List[str]:
        """Tear an instance down and return refs whose owner was released."""
        # pgw#671: a departing instance takes its background mint with it —
        # stop the driver before any module teardown races a warm forward.
        await self.abandon_background_mint(
            rec, reason="instance vacate", code="vacate")
        held_refs = self._record_refs(rec)
        held_objects = rec.held_objects
        released_refs: List[str] = []
        old_obj: Any = None
        inst, rec.instance, rec.ready = rec.instance, None, False
        rec.compile_targets.clear()
        # The next full StateDelta must remove the old address before any
        # replacement can become READY. Do this synchronously before teardown
        # awaits; adoption's second validation then rejects the stale ID.
        self._on_state_change()
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
            tier_before = self.store.residency.tier(ref)
            old_obj = held_objects.get(ref)
            owners = self._records_holding(ref)
            if owners:
                # Residency keeps one representative object per wire ref. If
                # it points at the departing record, transfer it to a survivor
                # so the old pipeline can actually be collected. This is an
                # ownership handoff, not an ON_DISK transition.
                if old_obj is not None and self.store.residency.obj(ref) is old_obj:
                    replacement = next(
                        (owner.held_objects.get(ref) for owner in reversed(owners)
                         if owner.held_objects.get(ref) is not None),
                        None,
                    )
                    self.store.residency.replace_object(ref, replacement)
                if old_obj is not None:
                    released_refs.append(ref)
                continue
            if (
                tier_before in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM)
                and self.store.residency.release_to_disk(ref)
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
        self._close_sequence_group(rec)
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
                self.store.residency.release_shared(key)
            rec.shared_keys.clear()
        self._on_state_change()
        released_refs = list(dict.fromkeys(released_refs))
        await self._observe_host_ram_progress(released_refs, collect_host=True)
        return released_refs

    # ---- job intake --------------------------------------------------------

    async def handle_run_job(self, run: pb.RunJob) -> None:
        key = (run.request_id, run.attempt)
        existing = self.jobs.get(key)
        if existing is not None and not existing.superseded:
            if not existing.finished:
                await self._send(pb.WorkerMessage(job_accepted=pb.JobAccepted(
                    request_id=run.request_id, attempt=run.attempt)))
            return
        # Same request, different attempt: abort the old attempt silently.
        for (rid, att), job in list(self.jobs.items()):
            if rid == run.request_id and att != run.attempt and not job.finished:
                job.superseded = True
                self._intent_transition(
                    job.intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED,
                    pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
                    detail=f"superseded by attempt {run.attempt}",
                )
                job.cancel_requested = True
                if job.ctx is not None:
                    job.ctx._cancel()
                if job.exec_task is not None:
                    job.exec_task.cancel()
                self._arm_cancel_unwind_watch(job)

        intent_id = self._job_intent(run)
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
                f"request {run.request_id} attempt {run.attempt} for "
                f"{run.function_name!r} reached a worker holding no serve "
                f"goal — the hub placed tenant work on a mint-only pod "
                f"(pgw#930)",
                phase="goal_admission",
            )
            await self._send_result(
                run.request_id, run.attempt, pb.JOB_STATUS_RETRYABLE,
                safe_message="worker holds no serve goal",
            )
            return
        if self.draining:
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail="worker draining",
            )
            await self._send_result(
                run.request_id, run.attempt, pb.JOB_STATUS_RETRYABLE, safe_message="worker draining"
            )
            return
        spec = self.specs.get(run.function_name)
        if spec is None:
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail=f"unknown function {run.function_name!r}",
            )
            await self._send_result(
                run.request_id,
                run.attempt,
                pb.JOB_STATUS_INVALID,
                safe_message=f"unknown function {run.function_name!r}",
            )
            return
        if run.function_name in self.unavailable:
            reason, detail, _ = self.unavailable[run.function_name]
            self._intent_transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                detail=f"function unavailable: {reason}",
            )
            await self._send_result(
                run.request_id,
                run.attempt,
                pb.JOB_STATUS_RETRYABLE,
                safe_message=f"function unavailable: {reason}",
            )
            return

        job = _Job(
            request_id=run.request_id,
            attempt=run.attempt,
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
        logger.info("job admitted %s attempt=%d", run.request_id, run.attempt)
        await self._send(pb.WorkerMessage(job_accepted=pb.JobAccepted(
            request_id=run.request_id, attempt=run.attempt)))
        job.task = asyncio.create_task(
            self._supervise_job(job, run), name=f"job-{run.request_id}")
        # pgw#674: the serving set may have changed — re-derive what to
        # stage next while this job computes.
        self.preloader.poke()

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

    async def _supervise_job(self, job: _Job, run: pb.RunJob) -> None:
        """pgw#738 never-silent guarantee: a job task that ends WITHOUT having
        reported terminal state is reaped into one.

        The 62922680 face of this issue was 3h51m of `assigned` on a live
        heartbeat with a dead task — the worker is the only component
        positioned to know its own task died, and it stayed silent. Every
        escape from ``_run_job``'s own handlers lands here, as does a plain
        return that somehow skipped ``_finish``.
        """
        escaped: Optional[BaseException] = None
        try:
            await self._run_job(job, run)
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

    async def _run_job(self, job: _Job, run: pb.RunJob) -> None:
        spec = job.spec
        assert spec is not None
        # pgw#748 phase 1: stamp the execution group BEFORE anything reads
        # residency, admits, loads or sets a device. Contextvars propagate
        # into every coroutine and to_thread hop this job makes, so the whole
        # job — admission, staging, handler, teardown — speaks one group.
        try:
            group = self._dispatch_group(run)
        except DispatchGroupUnresolved as exc:
            # pgw#779: reported here (not raised out of the task) so the job
            # ends with a terminal state instead of going quiet.
            logger.error("refusing %s: %s", run.request_id, exc)
            await self._finish(
                job, pb.JOB_STATUS_RETRYABLE, safe_message=_sanitize(str(exc)))
            return
        with device_group_scope(group):
            await self._run_job_grouped(job, run)

    async def _run_job_grouped(self, job: _Job, run: pb.RunJob) -> None:
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
            payload: Any = msgspec.msgpack.decode(run.input_payload, type=spec.payload_type)
        except (msgspec.ValidationError, msgspec.DecodeError) as exc:
            await self._finish(job, pb.JOB_STATUS_INVALID, safe_message=_sanitize(str(exc)))
            return
        try:
            # pgw#532: rebind declared Slots to the hub-resolved picks for
            # THIS dispatch (instance-per-pick). The derived spec drives the
            # whole job — pins, setup, adapters, ctx.slots — so every
            # downstream consumer sees the pick, never the code seed.
            spec = job.spec = self._effective_spec(spec, run)
            # th#913/gw#596: honor a hub-resolved per-request lane. An
            # unserveable lane is a TYPED refusal naming it (INVALID) —
            # never a silent fallback.
            if run.lane:
                spec = job.spec = self._execution_lane_effective_spec(spec, run.lane)
        except ExecutionLaneUnavailableError as exc:
            await self._finish(job, pb.JOB_STATUS_INVALID, safe_message=_sanitize(str(exc)))
            return
        except Exception as exc:
            status, msg = _map_exception(exc)
            await self._finish(job, status, safe_message=msg)
            return
        for undeclared in _undeclared_model_slots(spec, run):
            logger.warning(
                "UNDECLARED_MODEL_SLOT function=%s slot=%s request_id=%s: "
                "dispatched model param not declared in @endpoint(models={...}) "
                "— ignored, not loaded", spec.name, undeclared, run.request_id)
        if spec.cls is not None:
            # Register the derived per-pick spec before fencing so the job is
            # a visible record owner and vacate cannot race the validated
            # incarnation.
            self._class_record(spec)
        try:
            self._validate_required_compile(spec, run)
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
        # lane un-promotable on an overcommitted card; the ExecutionLaneGate pins
        # exactly the lane it executes, at call time.
        try:
            with self.store.residency.admit(
                self._job_admission_sizes(spec, routed, run),
                # pgw#652: weights are not the whole cost of admitting a
                # request — a concurrent 1024^2 diffusion request also holds
                # GBs of latents/attention workspace. The claim is LEARNED
                # from this function's measured peaks (0 until measured), so
                # no endpoint declares it.
                activation_bytes=self.store.residency.activation_hint(
                    self._activation_key(spec)),
            ):
                await self._run_job_pinned(job, run, payload, routed)
        finally:
            # The whole-job pin is now gone. Only a measured increase that
            # satisfies a remembered requirement produces capacity progress.
            await self._observe_host_ram_progress([])

    async def _run_job_pinned(
        self, job: _Job, run: pb.RunJob, payload: Any, routed: List[str]
    ) -> None:
        spec = job.spec
        assert spec is not None
        concurrency_at_start = len(self.in_flight_keys()) - 1

        snapshots = dict(run.snapshots) if run.snapshots else {}
        compute = run.compute if run.HasField("compute") else None
        needs_gpu = (compute.accelerator == "cuda") if compute is not None else spec.needs_gpu
        gpu_index = int(compute.gpu_index) if compute is not None else 0
        timeout_ms = int(run.timeout_ms or 0) or int(spec.timeout_ms or 0)

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

        # gw#453: arm repo-CAS checkpoint routing for producer jobs. Without
        # kind/destination_repo/job_id the ctx's _repo_job_upload_scope() is
        # None and save_checkpoint silently rides the media route (256 MiB
        # cap) instead of the job-bound checkpoint grant.
        execution_hints: Dict[str, Any] = {}
        if run.output_mode == pb.OUTPUT_MODE_INLINE:
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
            job_id = _capability_job_id(run.capability_token)
            producer_kwargs = dict(
                source_info=source_info,
                destination_info=destination_info,
                text_encoder_info=text_encoder_info,
                candidate_info=candidate_info,
                hf_token=getattr(self._settings, "hf_token", "") or "",
            )

        ctx_cls = _CONTEXT_BY_KIND.get(spec.kind, RequestContext)
        ctx = ctx_cls(
            request_id=run.request_id,
            job_id=job_id,
            emitter=self._make_ctx_emitter(job),
            owner=run.org or None,
            invoker_id=run.invoker_id or None,
            timeout_ms=timeout_ms or None,
            file_api_base_url=self.file_base_url or None,
            worker_capability_token=run.capability_token or None,
            models={b.slot: b.ref for b in run.models},
            loras={
                b.slot: tuple(
                    {"ref": ov.ref, "weight": float(ov.weight) or 1.0} for ov in b.loras
                )
                for b in run.models if b.loras
            },
            **_resolve_slots_kwargs(spec, run),
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
        if run.capability_token and self.file_base_url:
            from .capability_renewal import renew_capability_while_running

            job.renew_task = asyncio.create_task(
                renew_capability_while_running(
                    file_base_url=self.file_base_url,
                    request_id=run.request_id,
                    attempt=run.attempt,
                    get_worker_jwt=self.worker_jwt_provider,
                    get_token=lambda: ctx._worker_capability_token or "",
                    set_token=lambda t: setattr(ctx, "_worker_capability_token", t),
                ),
                name=f"cap-renew-{run.request_id}",
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
                run.request_id,
                attempt=run.attempt,
                manifest=manifest_from_run_job(run.input_assets),
                file_base_url=self.file_base_url or "",
                capability_token=run.capability_token or "",
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
            instance = await self.ensure_setup(spec, snapshots, promote_slots=routed)
            if setup_intent:
                self._intent_transition(
                    job.intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                    pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                )
            # th#913/gw#596: the concrete lane actually serving this job.
            # th#1050: ctx.lane exposes the same post-degrade truth to the
            # handler (declared-lane endpoints branch on it).
            job.execution_lane = self._served_execution_lane(spec, instructed=run.lane)
            # pgw#789: the shape coordinate, taken from the EXECUTED payload
            # with endpoint defaults applied. runtime_terms carries these only
            # when the endpoint declares a runtime formula (and the hub drops
            # that map after scaling reads it), so a latency comparison had no
            # shape axis at all for most endpoints.
            job.shape = serving_mode_mod.shape_of(
                payload, self._effective_config(spec))
            ctx._set_execution_lane(job.execution_lane)
            # th#1087: effective declared-config values for this dispatch.
            effective_config = self._effective_config(spec, run)
            invocation_snapshot = None
            if spec.config:
                config_generation = int(
                    run.config_generation or self.runtime_config.generation
                )
                invocation_snapshot = self.runtime_config.invocation_snapshot(
                    spec.name,
                    effective_config,
                    config_generation,
                )
            ctx._set_config(
                effective_config,
                snapshot=invocation_snapshot,
            )
            kwargs = await self._handler_kwargs(spec, snapshots)
            adapters = await self._prepare_adapters(run, spec, snapshots)
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
            # excluded (gw#551): the ExecutionLaneGate pins the one lane the handler
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
                        operation=f"GPU permit for request {run.request_id}",
                        status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                        stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_GPU_SLOT,
                        reason=pb.LIFECYCLE_WAIT_REASON_GPU_SLOT,
                    )
                    permit_token = self._permits.take(
                        gpu_permit, f"request {run.request_id}")
                    # th#1111: the permit wait was in NO metric — it precedes
                    # the handler window, so runtime_ms never saw it.
                    ctx._stages.record_pre(
                        "gpu_permit_wait", time.monotonic() - permit_t0)
                    self._loop = asyncio.get_running_loop()
                    lease = _GpuSlotLease(
                        gpu_permit, self._loop, self._permits,
                        f"request {run.request_id}", permit_token)
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
                    if torch is not None and torch.cuda.is_available():
                        try:
                            torch.cuda.reset_peak_memory_stats(gpu_index)
                        except Exception:
                            pass
                        # pgw#652: the baseline the peak is measured AGAINST.
                        # peak - baseline is this request's transient
                        # (activation) footprint, as opposed to the resident
                        # weights already allocated when it took the GPU.
                        alloc_at_start = self._vram_allocated()
                # Last execution fence: no adapter mutation or tenant handler
                # has run yet. The repeated check catches a replacement between
                # scheduler assignment/intake and this GPU turn.
                self._validate_required_compile(spec, run)
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
                    active: List[Tuple[str, Any]] = []
                    try:
                        for slot, prepared in adapters.items():
                            pipe = self._adapter_target(spec, slot)
                            ref = wire_ref(spec.models[slot])
                            await asyncio.to_thread(
                                self._adapters.activate, ref, pipe, prepared, run.request_id
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
                                        self._adapters.deactivate, ref, pipe, run.request_id
                                    )
                        ctx.raise_if_cancelled("canceled")
                        # pgw#676: name the execution before the GPU touches
                        # it — a signal death mid-handler leaves this marker
                        # for the supervisor's post-mortem attribution.
                        from . import postmortem as postmortem_mod

                        inflight_token = postmortem_mod.note_inflight(
                            "request", spec.name,
                            request_id=str(run.request_id or ""))
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
                                    timeout_ms=timeout_ms, gpu_index=gpu_index)
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
                                            kwargs, timeout_ms=timeout_ms,
                                            gpu_index=gpu_index)
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
                                self._adapters.deactivate, ref, pipe, run.request_id
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
                    "slotless for request %s", drained, run.request_id)
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
                        spec.name, run.request_id,
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
                    inline, blob_ref = await self._serialize_output(ctx, run, output)
                await self._finish(job, pb.JOB_STATUS_OK, inline=inline, blob_ref=blob_ref,
                                   metrics=metrics)
            else:
                inline, blob_ref = await self._serialize_output(ctx, run, output)
                await self._finish(job, pb.JOB_STATUS_OK, inline=inline, blob_ref=blob_ref,
                                   metrics=metrics)
        except _DeadlineExceeded:
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index,
                                    execution_lane=job.execution_lane)
            await self._finish(job, pb.JOB_STATUS_FATAL, safe_message="deadline exceeded",
                               metrics=metrics)
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
        snapshots: Dict[str, pb.Snapshot],
        *,
        set_path: Optional[Callable[[str], None]] = None,
        field_name: str = "source",
    ) -> None:
        """Reserved repo-field contract (#376, generalized pgw#594):
        materialize a reserved ``SourceRepo``-shaped payload field (default
        ``payload.source``, also used for ``payload.text_encoder``) locally
        before the handler runs. Same ModelStore path as model bindings —
        identical retry/classification and ModelEvent emission."""
        ref = str(info.get("ref") or "").strip()
        if not ref:
            raise ValidationError(f"payload.{field_name}.ref must be a non-empty repo ref")
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
        self, spec: EndpointSpec, snapshots: Dict[str, pb.Snapshot]
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
        self, run: pb.RunJob, spec: EndpointSpec, snapshots: Dict[str, pb.Snapshot]
    ) -> Dict[str, List[lora_util.PreparedAdapter]]:
        """Materialize + parse the job's per-slot LoRA overlays (gw#393).

        Downloads ride the normal ensure_local snapshot path (disk GC,
        ref-index, ModelEvents — so the hub learns adapter download bandwidth
        like any ref); parsed state dicts hit the digest-keyed RAM LRU.
        GPU-free: application happens later, under the job's GPU slot."""
        overlays = [(b.slot, list(b.loras)) for b in run.models if b.loras]
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
                ref = str(overlay.ref or "").strip()
                if not ref:
                    raise ValidationError(f"lora overlay on slot {slot!r} has an empty ref")
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
        transitions: List[Tuple[str, str, str, float]] = []
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
            after = next_offload_rung(before)
            if after is not None:
                transitions.append(
                    (ref, before, after, estimate_pipeline_size_gb(obj))
                )
        for ref, from_mode, to_mode, needed_gb in transitions:
            self._record_demotion(
                spec, ref=ref, phase="inference",
                from_rung=from_mode or "resident", to_rung=to_mode,
                needed_gb=needed_gb,
                detail=f"CUDA OOM mid-inference ({type(exc).__name__}); "
                       "quarantining this instance for a clean offloaded reload",
            )
        flush_memory()
        rec = self._classes.get(spec.instance_key)
        if rec is not None and rec.ready:
            rec.stale = True
        if not transitions:
            logger.warning(degraded_log_line(
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

    async def _execute(
        self,
        job: _Job,
        spec: EndpointSpec,
        instance: Any,
        ctx: RequestContext,
        payload: Any,
        kwargs: Dict[str, Any],
        *,
        timeout_ms: int,
        gpu_index: int,
    ) -> Any:
        bound = spec.method if instance is None else getattr(instance, spec.attr_name)
        call_kwargs = {spec.ctx_param: ctx, spec.payload_param: payload, **kwargs}
        timeout_s = (timeout_ms / 1000.0) if timeout_ms > 0 else None

        loop = asyncio.get_running_loop()
        if spec.is_async_gen:
            coro = self._pump_async_gen(job, bound(**call_kwargs))
        elif spec.is_async:
            coro = bound(**call_kwargs)
        elif spec.output_mode == "stream":
            coro = asyncio.to_thread(self._pump_sync_gen, job, bound, call_kwargs, gpu_index, loop)
        else:
            coro = asyncio.to_thread(self._call_sync, bound, call_kwargs, gpu_index)

        job.exec_task = asyncio.ensure_future(coro)
        # th#1111: the handler window stage_ms reconciles against (the same
        # interval runtime_ms measures).
        ctx._stages.handler_open()
        try:
            return await asyncio.wait_for(asyncio.shield(job.exec_task), timeout_s)
        except asyncio.TimeoutError:
            ctx._cancel()
            job.exec_task.cancel()
            if not spec.is_async:
                self._reap_stuck_thread(job)
            raise _DeadlineExceeded()
        except asyncio.CancelledError:
            # CancelJob path: the exec task was cancelled underneath us.
            raise CanceledError("canceled")
        finally:
            ctx._stages.handler_close()

    @staticmethod
    def _call_sync(bound: Callable[..., Any], call_kwargs: Dict[str, Any], gpu_index: int) -> Any:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_index)
            except Exception:
                pass
        return bound(**call_kwargs)

    def _reap_stuck_thread(self, job: _Job) -> None:
        """Deadline fired but the sync handler thread may not die. If it's
        still running after the recycle grace, exit so the pod is recycled."""

        async def _watch() -> None:
            assert job.exec_task is not None
            try:
                await asyncio.wait_for(asyncio.shield(job.exec_task), _STUCK_THREAD_RECYCLE_S)
            except asyncio.TimeoutError:
                logger.critical(
                    "handler thread for %s ignored deadline+cancel for %.0fs; "
                    "recycling worker process", job.request_id, _STUCK_THREAD_RECYCLE_S,
                )
                self._process_exit(70)
            except BaseException:
                pass  # thread finished (with error) — no recycle needed

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

        def _emit(event: Dict[str, Any]) -> None:
            if job.finished:
                return
            # gw#621: a ctx event is real forward progress — feed the open
            # activity's step counter (warmup forwards run GPU-bound with a
            # quiet CPU; watchdog evidence alone would read stalled).
            act = activity_mod.current()
            if act is not None:
                act.counter("infer:steps", progress_mod.UNIT_STEPS).add(1)
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
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_index)
            except Exception:
                pass
        acc = StreamAccumulator()
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
        return acc.result()

    # ---- results -----------------------------------------------------------

    async def _serialize_output(
        self, ctx: RequestContext, run: pb.RunJob, output: Any
    ) -> Tuple[Optional[bytes], Optional[str]]:
        # th#1130 safety net: msgpack reads struct fields straight off the C
        # layout, so an un-drained deferred asset would serialize as nulls.
        # _run_job_pinned always drains first; this catches any future path
        # that does not, loudly rather than by shipping a hollow asset.
        if ctx._deferred.pending():
            logger.error(
                "deferred outputs reached serialization un-drained for %s — "
                "materializing inline", run.request_id)
            await asyncio.to_thread(ctx._drain_deferred_outputs)
        data = msgspec.msgpack.encode(output)
        if len(data) <= INLINE_RESULT_MAX_BYTES:
            return data, None
        try:
            asset = await asyncio.to_thread(
                ctx.save_bytes, f"results/{run.request_id}.msgpack", data
            )
            ref = getattr(asset, "ref", "") or ""
            if not ref:
                raise RuntimeError("upload returned no ref")
            return None, ref
        except Exception as exc:
            logger.warning("result blob upload failed for %s: %s", run.request_id, exc)
            raise RetryableError("output upload failed") from exc

    @staticmethod
    def _peak_vram_bytes(gpu_index: int) -> int:
        """This job's CUDA peak-allocator high-water mark (reset when it took
        the GPU). 0 without torch/CUDA."""
        if torch is not None and torch.cuda.is_available():
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


class _DeadlineExceeded(Exception):
    pass
