"""Job execution: intake, GPU semaphore, deadline + cancellation watchdog,
sync-on-thread / async-on-loop, JobProgress deltas, result send, and the
worker-side model seam (ensure-local, setup injection, declarative residency,
and compile-cache adoption).

One dispatch path for every endpoint kind. Everything runs on the single
asyncio loop; sync tenant code runs in threads via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


import msgspec

from .api.binding import ModelRef, wire_ref
from .api.errors import (
    ArtifactTransferError,
    CanceledError,
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
    HardwareUnmetError,
    InsufficientDiskError,
    InsufficientHostRamError,
)
from .input_assets import cleanup_input_assets, materialize_input_assets
from .models import disk_gc
from .models import provision
from .models import residency as residency_mod
from .models.memory import (
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
from .models.cache_paths import tensorhub_cas_dir
from .models.download import ensure_local, lookup_provider_for_ref
from .models.errors import MissingSnapshotError, UrlExpiredError
from .models.residency import Residency
from .pb import worker_scheduler_pb2 as pb
from .registry import EndpointSpec

if typing.TYPE_CHECKING:
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

_CONTEXT_BY_KIND: Dict[str, type] = {
    "inference": RequestContext,
    "conversion": ConversionContext,
    "dataset": DatasetContext,
    "training": TrainingContext,
}

logger = logging.getLogger(__name__)

INLINE_RESULT_MAX_BYTES = 64 * 1024


async def _to_thread_complete(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Join after cancellation: ``to_thread`` itself cannot be cancelled.

    Diffusers/Accelerate mutate process-global meta-device hooks while loading,
    so a surrounding model-load lock must outlive the worker thread.
    """
    work = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        try:
            await work
        except BaseException:
            pass
        raise


# ctx.progress/log/checkpoint events ride the JobProgress stream; the hub fans
# them to /v1/requests/:id/events SSE as output.delta envelopes whose
# payload.delta carries this JSON verbatim (th#640).
EVENT_CONTENT_TYPE = "application/x-request-event+json"
_CANCEL_GRACE_S = 5.0
_STUCK_THREAD_RECYCLE_S = 30.0
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


def _sanitize(message: str) -> str:
    out = str(message or "").strip()
    for pat in _REDACTIONS:
        out = pat.sub("[redacted]", out)
    return out[:1024]


def _reserved_repo_info(payload: Any, field_name: str) -> Dict[str, Any]:
    """``payload.source`` / ``payload.destination`` as a plain dict ({} when
    absent). Producer payloads carry these reserved-name structs (#376)."""
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


def _resolve_slots_kwargs(spec: EndpointSpec, run: "Optional[pb.RunJob]") -> Dict[str, Any]:
    """``ctx.slots`` resolution chain (pgw#520 / pgw#516): merge each
    Slot-declared slot's repo-metadata ``ModelBinding.inference_defaults``
    over its code fallback preset, then apply each riding lora's
    ``LoraOverlay.inference_defaults`` as a FIELD-LEVEL override, in lora
    order (pgw#516 composition rule — see ``api.slot._apply_lora_overrides``).
    Returns the ``resolved_slots=``/``slot_errors=`` kwargs for
    ``RequestContext.__init__`` — a slot that fails to resolve (no metadata +
    no fallback, or no ref) is deferred to a ``ctx.slots[name]`` access error
    instead of failing the whole dispatch."""
    if not spec.slots:
        return {"resolved_slots": {}, "slot_errors": {}}
    from .api.slot import resolve_slot

    run_models = list(run.models) if run is not None else []
    raw_defaults = {b.slot: b.inference_defaults for b in run_models if b.inference_defaults}
    lora_defaults = {
        b.slot: tuple(lo.inference_defaults for lo in b.loras if lo.inference_defaults)
        for b in run_models if b.loras
    }
    resolved: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for name, slot in spec.slots.items():
        try:
            resolved[name] = resolve_slot(
                name, slot,
                ref=spec.models.get(name),
                family=spec.slot_family.get(name, ""),
                raw_metadata_json=raw_defaults.get(name, ""),
                lora_metadata_json=lora_defaults.get(name, ()),
            )
        except ValueError as exc:
            errors[name] = str(exc)
    return {"resolved_slots": resolved, "slot_errors": errors}


def _hub_binding_for_wire_ref(ref: str) -> ModelRef:
    """A tensorhub-source binding for a hub-named wire ref (pgw#532).

    ``RunJob.models`` / desired-instance refs name hub-CAS repos in the canonical
    ``owner/repo[:tag][#flavor]`` grammar; this mints the binding the
    executor materializes them through (``ensure_local`` then follows the
    tensorhub lane: orchestrator snapshots or the th#763 missing_snapshot
    re-mint — never an upstream self-fetch). Raises ``ValueError`` when
    ``ref`` does not parse under that grammar (e.g. a raw upstream id the
    hub stamped for an unmirrored slot default)."""
    from .models.refs import parse_model_ref

    parsed = parse_model_ref(ref)
    th = parsed.tensorhub
    if th is None:  # pragma: no cover - parse_model_ref(tensorhub) guarantees it
        raise ValueError(f"{ref!r} is not a tensorhub ref")
    return ModelRef(
        source="tensorhub",
        path=f"{th.owner}/{th.repo}",
        tag=th.tag or "latest",
        flavor=th.flavor or "",
    )


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
    from .models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile

    return WorkerResolvedRepo(
        snapshot_digest=snap.digest,
        files=[
            WorkerResolvedRepoFile(
                path=f.path,
                size_bytes=int(f.size_bytes),
                blake3=f.blake3,
                url=f.url or None,
            )
            for f in snap.files
        ],
    )


def _cell_lane_matches(
    ref: str, family: str, *, wants_w8a8: bool, want_bucket: int
) -> bool:
    """Whether an inductor cell ref serves this endpoint's graph family
    (gw#561): the declared rank bucket is half of the identity — a
    lora_bucket endpoint needs exactly a ``-lora<bucket>`` cell of its base
    lane, and a branchless endpoint must never fetch one (either mismatch is
    a guaranteed lane_drift that would shadow the right cell and serve
    eager)."""
    from . import compile_cache

    if not compile_cache.is_cache_ref(ref, family):
        return False
    base, bucket = compile_cache.lane_bucket(compile_cache.cell_lane(ref))
    if bucket != int(want_bucket or 0):
        return False
    return base == "w8a8" if wants_w8a8 else base != "w8a8"


def _ref_wants_w8a8(ref: str) -> bool:
    """Whether one canonical Tensorhub model ref selects a W8A8 flavor."""
    from .models.refs import parse_model_ref

    try:
        parsed = parse_model_ref(ref).tensorhub
    except ValueError:
        return False
    if parsed is None or parsed.owner == "_system":
        return False
    flavor = parsed.flavor or ""
    return flavor == "fp8-w8a8" or flavor.startswith("fp8-w8a8-")


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


def _is_corrupt_load_error(exc: BaseException) -> bool:
    """Errors a truncated/corrupt snapshot produces at weights-load time
    (gw#408). Broad on purpose: the digest re-verify gate downstream
    separates real corruption from code bugs — a verified-clean tree
    re-raises the original error instead of quarantining."""
    import errno as _errno
    import struct as _struct

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
    ) -> None:
        self._emit = emit
        self._hf_home = hf_home or None
        self._hf_token = hf_token or None
        self._cache_dir = cache_dir or tensorhub_cas_dir()
        self.residency = Residency(
            on_event=self._on_residency_event, vram_budget_bytes=vram_budget_bytes,
        )
        self._locks: Dict[str, asyncio.Lock] = {}
        self._bindings: Dict[str, Any] = {}
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
        self._identity_lock = threading.RLock()
        # Cold-ref waiters (th#763): ensure_local blocks here until the
        # hub's re-minted DOWNLOAD banks a snapshot for the ref.
        self._snapshot_waiters: Dict[str, asyncio.Event] = {}

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
        else:
            identity = self.resident_identity(ref)
        coro = self._event(ref, pb_state, identity=identity, **kw)
        if state == residency_mod.EVICTED:
            # Capture before removal so the eviction names the exact bytes it
            # removed; later events cannot inherit that stale identity.
            with self._identity_lock:
                self._resident_identities.pop(ref, None)
                self._disk_identities.pop(ref, None)
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

    # ---- residency facade ----------------------------------------------------

    def residency_snapshot(self) -> List[pb.ModelResidency]:
        out: List[pb.ModelResidency] = []
        # Hold identity stable while Residency captures its tiers. Residency
        # callbacks run only after releasing their own lock, so this cannot
        # invert lock order: a transition either happens entirely before this
        # snapshot, or its identity update waits until the captured view is
        # complete.
        with self._identity_lock:
            for ref, tier, vram in self.residency.snapshot():
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

    def local_path(self, ref: str) -> Optional[Path]:
        return self.residency.local_path(ref)

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
            self._desired_snapshot_identities = desired
            # Generations belong only to the current desired identity. Leave
            # actual resident identity untouched: those bytes may still be in
            # RAM/VRAM and must remain honestly observable until transitioned.
            for ref in self._snapshot_generations:
                self._snapshot_generations[ref] = 0
            for ref, snapshot in stored.items():
                self._snapshots[ref] = snapshot
                self._snapshot_generations[ref] = accepted_generation

        for ref in stored:
            waiter = self._snapshot_waiters.get(ref)
            if waiter is not None:
                waiter.set()

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
        """Publish exact identity when verified cached bytes satisfy a newer
        desired generation without requiring a redundant download."""
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
        if not self._set_resident_identity(ref, identity):
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
        the wire snapshot's per-file blake3; small JSON sidecars use
        CANONICAL digests read from ``local_path`` (save-era serialization —
        provenance stamps, explicit defaults, torch_dtype/dtype vocabulary —
        must not break sharing of byte-identical weights; see
        models/config_identity.py). Root-level files group under ``""``
        (never shared — model_index.json etc. differ per repo). Empty when
        no digest-carrying snapshot was seen — sharing stays off; weights
        are never hashed from disk."""
        from .models.config_identity import CANONICAL_JSON_MAX_BYTES, canonical_json_digest

        snap = self._snapshots.get(ref)
        if snap is None:
            return {}
        groups: Dict[str, Dict[str, str]] = {}
        for f in snap.files:
            rel = str(f.path).strip().lstrip("/")
            if not rel or not f.blake3:
                continue
            comp, _, rest = rel.partition("/")
            if not rest:
                comp, rest = "", rel
            digest = str(f.blake3)
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
        persisted ref index so Hello.models and GC see what disk holds."""
        for ref, ent in self._index.entries().items():
            p = Path(str(ent.get("path") or ""))
            if p.exists():
                if self.residency.tier(ref) is None:
                    self.residency.track_disk(ref, p)
            else:
                self._index.remove(ref)

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
        now = time.time()
        out: List[Tuple[float, str]] = []
        for ref in self.residency.refs_in(residency_mod.Tier.DISK):
            if ref in exclude or self.residency.in_use(ref):
                continue
            if (ref in keep) != include_keep:
                continue
            last = self._index.last_used(ref)
            if honor_grace and (now - last) < _DISK_GC_GRACE_S:
                continue
            out.append((last, ref))
        if include_keep:
            out.sort(key=lambda item: (-keep_rank[item[1]], item[0], item[1]))
        else:
            out.sort()
        return [r for _, r in out]

    def _evict_disk_ref(self, ref: str) -> None:
        path = self.residency.local_path(ref) or self._index.path(ref)
        if not self.residency.evict(ref):  # refuses in-use entries; emits EVICTED
            return
        if path is not None:
            disk_gc.delete_ref_bytes(ref, path, self._cache_dir)
            disk_gc.sweep_orphan_blobs(self._cache_dir)
        self._index.remove(ref)

    async def _ensure_disk_headroom(
        self, ref: str, needed_bytes: int, identity: _ResidencyIdentity = ("", 0),
    ) -> None:
        target = int(needed_bytes) + _DISK_GC_MARGIN_BYTES
        if self._disk_free() >= target:
            return
        await asyncio.to_thread(self.gc_disk, target, exclude=(ref,))
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

    async def _await_hub_snapshot(self, ref: str) -> pb.Snapshot:
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
        try:
            await asyncio.wait_for(waiter.wait(), _MISSING_SNAPSHOT_WAIT_S)
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
        return (await self._materialize_local(
            ref, snapshot, binding=binding)).path

    async def _materialize_local(
        self,
        ref: str,
        snapshot: Optional[pb.Snapshot] = None,
        *,
        binding: Any = None,
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

        def complete(path: Path) -> _MaterializedLocal:
            return _MaterializedLocal(path=path, identity=operation_identity)

        async with self._lock(ref):
            cached = self.residency.local_path(ref)
            # A digest-carrying snapshot is authoritative: a cached
            # materialization of the SAME ref at a DIFFERENT digest is stale
            # (flavor re-published — e.g. compile-cache digest-change
            # re-adoption, e2e#117 live find #7) and must not short-circuit.
            want = ""
            if snapshot is not None and snapshot.digest:
                want = snapshot.digest.split(":", 1)[-1].strip().lower()
            if cached is not None and cached.exists() and (not want or cached.name == want):
                if ref in self._verified:
                    self._index.touch(ref)
                    await self._confirm_cached_identity(ref, operation_identity)
                    return complete(cached)
                # First use this boot: verify before trusting (gw#408). A
                # pod-churn-truncated snapshot used to fatal every load until
                # a manual delete; now it is quarantined + re-materialized.
                ok, bad = await asyncio.to_thread(
                    self._verify_snapshot_tree, cached, snapshot
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
                    snapshot = await self._await_hub_snapshot(ref)
                    operation_identity = self._snapshot_identity(ref, snapshot)
            if snapshot is not None and snapshot.files:
                # Sizes are known up front for tensorhub snapshots: gate on
                # disk headroom, GC-ing LRU refs first (#370).
                await self._ensure_disk_headroom(
                    ref,
                    sum(int(f.size_bytes) for f in snapshot.files),
                    operation_identity,
                )
            last_progress = 0.0

            def _progress(done: int, total: Optional[int]) -> None:
                nonlocal last_progress
                now = time.monotonic()
                if now - last_progress < _PROGRESS_EVENT_MIN_INTERVAL_S:
                    return
                last_progress = now
                assert self._loop is not None
                asyncio.run_coroutine_threadsafe(
                    self._event(ref, pb.MODEL_STATE_DOWNLOADING,
                                identity=operation_identity,
                                bytes_done=int(done), bytes_total=int(total or 0)),
                    self._loop,
                )

            await self._event(
                ref, pb.MODEL_STATE_DOWNLOADING, identity=operation_identity,
            )
            delay = 1.0
            for attempt in range(1, _DOWNLOAD_RETRIES + 1):
                try:
                    resolved = None
                    if snapshot is not None and snapshot.digest:
                        resolved = _snapshot_to_resolved(snapshot)
                    path = await ensure_local(
                        ref,
                        provider=getattr(binding, "source", None),
                        snapshot=resolved,
                        cache_dir=self._cache_dir,
                        hf_home=self._hf_home,
                        hf_token=self._hf_token,
                        allow_patterns=tuple(getattr(binding, "files", ()) or ()),
                        components=tuple(getattr(binding, "components", ()) or ()),
                        progress=_progress,
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
                    self.residency.track_disk(ref, path)
                    if tier_before is residency_mod.Tier.DISK and identity_changed:
                        # Residency suppresses same-tier event spam. A digest
                        # move is nevertheless a semantic ON_DISK transition.
                        await self._event(
                            ref, pb.MODEL_STATE_ON_DISK,
                            identity=operation_identity,
                        )
                    # tree_bytes stats every file — off-loop (gw#407: no
                    # multi-GB directory walks on the event loop).
                    size = await asyncio.to_thread(disk_gc.tree_bytes, path)
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
                    logger.warning("download of %s failed (attempt %d): %s; retrying in %.1fs",
                                   ref, attempt, exc, delay)
                    await asyncio.sleep(delay)
                    delay *= 4
            raise RuntimeError("unreachable")

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
        declared size AND blake3 digest; files the manifest cannot cover
        (reassembled chunked originals, merged single-file checkpoints) plus
        manifest-less trees (hf/civitai) get the structural safetensors check
        (header parses + every declared tensor byte present). Returns
        ``(ok, bad_digests)`` — the digests name blobs to quarantine."""
        from .models.cozy_cas import _blake3_file
        from .models.cozy_snapshot import _is_part_file, _is_parts_manifest, _norm_rel_path
        from .models.loading import safetensors_file_valid

        p = Path(path)
        bad: List[str] = []
        covered: set[Path] = set()
        files = list(snapshot.files) if snapshot is not None else []
        if files and p.is_dir():
            for f in files:
                if _is_parts_manifest(f.path) or _is_part_file(f.path):
                    continue  # not materialized: parts live only in blobs/
                try:
                    dst = p / _norm_rel_path(f.path)
                except ValueError:
                    continue
                covered.add(dst)
                digest = (f.blake3 or "").strip().lower()
                try:
                    if not dst.exists():
                        raise ValueError("missing")
                    if f.size_bytes and dst.stat().st_size != int(f.size_bytes):
                        raise ValueError("size mismatch")
                    if digest and _blake3_file(dst).lower() != digest:
                        raise ValueError("blake3 mismatch")
                except (OSError, ValueError) as exc:
                    logger.warning("snapshot file %s/%s corrupt: %s", p.name, f.path, exc)
                    bad.append(digest or f.path)
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
        from .models.cozy_snapshot import delete_blobs

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
        if "digest" in text or "blake3" in text or "hash" in text:
            return "digest_mismatch"
        if "no space" in text or "disk" in text:
            return "insufficient_disk"
        return "download_failed"


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
    """One immutable hub-attached cell selected before model setup."""

    path: Path
    ref: str
    snapshot_digest: str


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


@dataclass
class _ClassRecord:
    cls: type
    specs: List[EndpointSpec] = dc_field(default_factory=list)
    instance: Any = None
    server: Any = None  # ServerHandle for runtime="vllm"/"llama-server"
    ready: bool = False
    failed: Optional[str] = None
    lock: asyncio.Lock = dc_field(default_factory=asyncio.Lock)
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
    # gw#494: a resolution re-pick moved the specs' bindings away from
    # held_refs; the instance serves the OLD pick and must be vacated.
    stale: bool = False
    # gw#551: wire refs of lane-registered slots (gw#479). Lane residency is
    # call-time-owned (LaneGate promotes + pins around each pipeline call);
    # the executor must neither whole-job-pin nor eagerly promote them, or
    # the idle sibling can never be LRU-swapped out.
    lane_refs: set = dc_field(default_factory=set)
    # pgw#572: exact compile-capable objects owned by this READY record. The
    # IDs are minted after successful setup and cleared before vacate; they do
    # not derive from mutable refs, authored specs, or object memory addresses.
    compile_targets: Dict[str, _CompileTargetRecord] = dc_field(default_factory=dict)


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
    lane_slots: set = dc_field(default_factory=set)
    shared_keys: List[Any] = dc_field(default_factory=list)
    shared_bytes: int = 0
    # gw#551: slots whose pipeline __call__ the LaneGate wrapped. Only these
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
    ctx: Optional[RequestContext] = None
    task: Optional[asyncio.Task] = None
    exec_task: Optional[asyncio.Task] = None
    renew_task: Optional[asyncio.Task] = None
    finished: bool = False
    superseded: bool = False
    # gw#516: True while the job is past the decode->finalize handoff (GPU
    # slot terminally released, encode/upload tail running, result unshipped).
    finalizing: bool = False
    admitted_at: float = dc_field(default_factory=time.monotonic)
    # One JobProgress seq space per job, shared by stream chunks and ctx
    # events so interleaved sends stay monotonic. itertools.count.__next__
    # is atomic under the GIL — safe from handler threads.
    seq: "itertools.count[int]" = dc_field(default_factory=lambda: itertools.count(1))


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

    __slots__ = ("_sem", "_loop", "_lock", "_held", "released_at")

    def __init__(self, sem: asyncio.Semaphore, loop: asyncio.AbstractEventLoop) -> None:
        self._sem = sem
        self._loop = loop
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
            self._sem.release()
        else:
            self._loop.call_soon_threadsafe(self._sem.release)
        return True

    def reacquire(self) -> None:
        """Blocking re-acquire from a handler thread."""
        asyncio.run_coroutine_threadsafe(self._sem.acquire(), self._loop).result()
        with self._lock:
            self._held = True


class Executor:
    def __init__(
        self,
        specs: List[EndpointSpec],
        send: Callable[[pb.WorkerMessage], Awaitable[None]],
        *,
        settings: Any = None,
        store: Optional[ModelStore] = None,
        gpu_slots: int = 1,
    ) -> None:
        self.specs: Dict[str, EndpointSpec] = {s.name: s for s in specs}
        self._send = send
        self._settings = settings
        self.store = store or ModelStore(send)
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
        self._gpu_slots = max(1, gpu_slots)
        self._gpu_semaphore = asyncio.Semaphore(self._gpu_slots)
        # Model loads/promotions serialize so allocator-delta measurements
        # and free-VRAM reads don't cross-contaminate (#369).
        self._load_lock = asyncio.Lock()
        # Compile-cache adoption mutates already-resident modules in place.
        # Serialize the whole operation (download through terminal evidence),
        # not only its GPU warmup, so two commands can never cross wraps or
        # let an older rollback mutate a newer adoption.
        self._compile_cache_adoption_lock = asyncio.Lock()
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
        # Current worker JWT for hub HTTP calls (capability renewal). Worker
        # wiring points this at the transport's rotated credential.
        self.worker_jwt_provider: Callable[[], str] = (
            lambda: str(getattr(settings, "worker_jwt", "") or "").strip()
        )
        self.draining = False
        self.jobs: Dict[Tuple[str, int], _Job] = {}
        self._idle = asyncio.Event()
        self._idle.set()
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

    # ---- precision resolutions (th#697) -----------------------------------

    def apply_model_resolutions(self, resolutions: Dict[str, Tuple[str, str]]) -> None:
        """Rebind model slots to the hub's precision-ladder picks.

        ``resolutions`` maps a DECLARED wire ref to ``(resolved_ref, cast)``
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
        from .api.binding import rebind_pick

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
                    resolved_ref, cast = pick
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
        from .models.hub_policy import FIT_INCOMPATIBLE, TensorhubWorkerCapabilities
        from .models.serve_fit import RUN_CPU, RUN_OFFLOAD, plan_serve

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
        detected_cc = (float(detected_sm) / 10.0) if detected_sm.isdigit() else None
        libs = {str(x) for x in (gpu_info.get("installed_libs") or [])}
        caps = TensorhubWorkerCapabilities(
            cuda_version=str(gpu_info.get("cuda_version") or ""),
            gpu_sm=int(detected_sm) if detected_sm.isdigit() else 0,
            torch_version=str(gpu_info.get("torch_version") or ""),
            installed_libs=list(libs),
        )
        for name, spec in self.specs.items():
            r = spec.resources
            # Genuine hard incompatibilities keep their explicit reason codes —
            # no lever (offload/cpu/quant) makes them run on this silicon.
            if r.compute_capability is not None and detected_cc is not None \
                    and detected_cc < float(r.compute_capability):
                self.unavailable[name] = (
                    "compute_capability_unmet",
                    f"requires SM {r.compute_capability:.1f}, detected {detected_cc:.1f}",
                    {"detected_sm": f"{detected_cc:.1f}", "required_sm": f"{float(r.compute_capability):.1f}"})
                self._gate_owned.add(name)
                continue
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
                     "recommended_vram_gb": (f"{r.vram_gb:.0f}" if r.vram_gb else "")})
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
        from .models.serve_fit import demoted

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
        from .models.serve_fit import load_rung_engaged

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
        from .models.serve_fit import cast_dropped

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
            and self._classes[spec.instance_key].failed is None
        )

    @staticmethod
    def _refresh_compile_target(target: _CompileTargetRecord) -> None:
        """Refresh compatibility evidence after an in-place lane mutation."""
        from . import compile_cache
        from .models.loading import pipeline_weight_lane

        cfg = target.spec.compile
        assert cfg is not None
        contract_digest = compile_cache.execution_contract_digest(
            target.pipeline, cfg)
        lane = pipeline_weight_lane(target.pipeline)
        bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
        with target.state_lock:
            target.pipeline_weight_lane = lane
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
        A mandatory W8A8 object cannot serve eager; its next setup reloads it.
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
            mandatory_w8a8 = target.pipeline_weight_lane.startswith("w8a8")
            if mandatory_w8a8:
                # Keep the same incarnation visible with an empty active
                # identity until declarative reconciliation replaces it.
                # RequiredCompileExecution then fails closed locally while
                # Tensorhub can correlate the causal FAILED to this row.
                rec.stale = True
        if mandatory_w8a8:
            self._mark_compile_target_unavailable(rec, target, detail)
        logger.warning(
            "compile target %s runtime guard tripped; compiled proof revoked: %s",
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

    def _bind_compile_guard(
        self, rec: _ClassRecord, target: _CompileTargetRecord,
    ) -> bool:
        """Bind one live wrapper's first failure to exact target revocation."""
        from . import compile_cache, trt_engine

        def callback(detail: str) -> None:
            self._compile_guard_failed(rec, target, detail)

        if trt_engine.set_guard_failure_callback(target.pipeline, callback):
            return True
        return compile_cache.set_guard_failure_callback(target.pipeline, callback)

    def _install_compile_targets(
        self,
        rec: _ClassRecord,
        spec: EndpointSpec,
        objects: typing.Iterable[Any],
        active_artifacts: Optional[Dict[int, _CompileArtifactSelection]] = None,
        function_proofs: Optional[Dict[int, set[str]]] = None,
    ) -> None:
        """Mint one incarnation for every compile-capable object just set up."""
        from . import compile_cache

        cfg = spec.compile
        rec.compile_targets = {}
        if cfg is None:
            return
        # Production injection supplies object-scoped slot ownership. Keep
        # bare objects accepted for focused unit construction only, deriving
        # their ownership from the record's already-frozen held bindings.
        all_slots = {slot for slot, _ref, _digest in rec.held_bindings}
        candidates = [
            item if isinstance(item, _CompileObjectCandidate)
            else _CompileObjectCandidate(item, set(all_slots))
            for item in objects
        ]
        requested_w8a8 = any(
            _ref_wants_w8a8(wire_ref(spec.models[slot]))
            for slot in self._setup_slots(spec)
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
                if binding[0] in candidate.slots
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
                alias_cfg = alias.compile
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
                    slot not in alias.models
                    or wire_ref(alias.models[slot]).strip() != ref
                    for slot, ref, _digest in bindings
                ):
                    continue
                name = str(alias.name).strip()
                if name:
                    compatible_names.add(name)
            expected_names = compatible_names & required_names
            target.function_names = tuple(sorted(
                compatible_names & permitted_names))
            mandatory_w8a8 = target.pipeline_weight_lane.startswith("w8a8")
            candidate_requested_w8a8 = any(
                _ref_wants_w8a8(ref) for _slot, ref, _digest in bindings
            )
            if (
                (mandatory_w8a8 or candidate_requested_w8a8)
                and set(target.function_names) != expected_names
            ):
                raise compile_cache.CompiledLaneUnavailableError(
                    "mandatory W8A8 function proof incomplete "
                    f"(expected={sorted(expected_names)!r} "
                    f"proven={list(target.function_names)!r})"
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
                if mandatory_w8a8 or candidate_requested_w8a8:
                    raise compile_cache.CompiledLaneUnavailableError(detail)
                continue
            lane_error = compile_cache.compile_target_lane_error(
                target.pipeline_weight_lane, target.lora_bucket)
            if lane_error:
                if mandatory_w8a8 or candidate_requested_w8a8:
                    raise compile_cache.CompiledLaneUnavailableError(lane_error)
                logger.warning(
                    "compile target omitted for %s: %s", spec.name, lane_error)
                continue
            if candidate_requested_w8a8 and not mandatory_w8a8:
                raise compile_cache.CompiledLaneUnavailableError(
                    f"W8A8 binding for {spec.name!r} materialized a non-W8A8 "
                    f"pipeline lane {target.pipeline_weight_lane!r}"
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
                continue
            if mandatory_w8a8 and not active_ref:
                # A W8A8 object without a proven exact cell is not a READY
                # serving target. The loader normally raises earlier; keep
                # this final wire-state invariant fail closed too.
                raise compile_cache.CompiledLaneUnavailableError(
                    f"W8A8 compile target for {spec.name!r} has no proven "
                    "active Forge artifact"
                )
            with target.state_lock:
                target.active_compile_ref = active_ref
                target.active_compile_snapshot_digest = active_digest
            rec.compile_targets[incarnation_id] = target
            if active_ref and not self._bind_compile_guard(rec, target):
                # Production wrappers always expose one of the two guard
                # signals. A hand-built/custom wrapper without revocation
                # cannot be advertised as compiled. W8A8 remains fail-closed.
                with target.state_lock:
                    target.active_compile_ref = ""
                    target.active_compile_snapshot_digest = ""
                if mandatory_w8a8:
                    raise compile_cache.CompiledLaneUnavailableError(
                        f"W8A8 compile target for {spec.name!r} has no "
                        "runtime guard revocation signal"
                    )
                logger.warning(
                    "compile target %s has no runtime guard revocation signal; "
                    "advertising eager", incarnation_id,
                )
        if requested_w8a8 and not rec.compile_targets:
            raise compile_cache.CompiledLaneUnavailableError(
                f"W8A8 setup for {spec.name!r} produced no addressable "
                "compile-capable pipeline target"
            )

    def compile_targets(self) -> List[pb.CompileTarget]:
        """Full-replace READY compile-target snapshot for StateDelta."""
        out: List[pb.CompileTarget] = []
        for rec in self._classes.values():
            if not rec.ready:
                continue
            for target in rec.compile_targets.values():
                with target.state_lock:
                    cfg = target.spec.compile
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
        wants_w8a8 = any(
            _ref_wants_w8a8(wire_ref(spec.models[slot]))
            for slot in setup_slots
        )
        if not run.HasField("required_compile"):
            if wants_w8a8:
                raise RetryableError(
                    "required_compile_missing: W8A8 dispatch requires an "
                    "exact active compile incarnation"
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
            target_lane = target.pipeline_weight_lane
            target_functions = target.function_names
            target_active = (
                target.active_compile_ref,
                target.active_compile_snapshot_digest,
                target.contract_digest,
            )
            target_bindings = target.model_bindings
        if wants_w8a8 and not target_lane.startswith("w8a8"):
            raise RetryableError(
                "required_compile_lane_mismatch: W8A8 dispatch selected a "
                "non-W8A8 live pipeline"
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
                return self._hub_binding(run_ref)
            except ValueError:
                logger.warning(
                    "slot %r of %s: resolved_models ref %r is not a CAS ref; "
                    "falling back to the declared default", slot, spec.name, run_ref)
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
        effective = dict(spec.models)
        for slot in spec.slots:
            effective[slot] = self._slot_dispatch_binding(
                spec, slot, run_refs.get(slot, ""))
        if effective == spec.models:
            return spec
        return dc_replace(spec, models=effective)

    async def ensure_desired_instance(
        self,
        desired: "pb.DesiredInstance",
        snapshots: Dict[str, "pb.Snapshot"],
    ) -> None:
        """Best-effort warm of one declarative, fully bound instance."""
        spec = self.specs.get(desired.function_name)
        if spec is None:
            raise ValidationError(f"unknown function {desired.function_name!r}")
        if spec.cls is None:
            raise ValidationError(
                f"function {desired.function_name!r} has no persistent instance to warm"
            )

        pairs = [(m.slot.strip(), m.ref.strip()) for m in desired.models]
        bindings = dict(pairs)
        expected = set(spec.models)
        if any(not slot or not ref for slot, ref in pairs):
            raise ValidationError(
                f"desired instance {desired.function_name!r} has an empty slot or ref"
            )
        if len(bindings) != len(pairs) or set(bindings) != expected:
            raise ValidationError(
                f"desired instance {desired.function_name!r} must bind exactly "
                f"{sorted(expected)!r}; got {sorted(bindings)!r}"
            )

        run = pb.RunJob(function_name=desired.function_name, models=desired.models)
        effective = self._effective_spec(spec, run)
        resolved = {slot: wire_ref(binding) for slot, binding in effective.models.items()}
        if resolved != bindings:
            raise ValidationError(
                f"desired instance {desired.function_name!r} does not match the "
                "worker's resolved bindings"
            )
        try:
            await self.ensure_setup(effective, snapshots)
        except Exception as exc:
            # Host-RAM admission already emitted the precise largest staged
            # ref(s) that caused the capacity failure. Do not overwrite that
            # signal by failing smaller shared refs such as an SDXL VAE.
            if not isinstance(exc, InsufficientHostRamError):
                error = _model_failure_vocab(exc)
                for ref in dict.fromkeys(bindings.values()):
                    await self._send(pb.WorkerMessage(
                        model_event=self.store.model_event(
                            ref, pb.MODEL_STATE_FAILED, error=error,
                        )
                    ))
            raise

    def _job_pin_refs(self, spec: EndpointSpec, slots: List[str]) -> List[str]:
        """Wire refs a job pins for its whole lifetime: every routed slot
        EXCEPT lane refs (gw#551 — the LaneGate pins those around the actual
        pipeline call, so the idle sibling stays LRU-demotable)."""
        rec = self._classes.get(spec.instance_key) if spec.cls is not None else None
        lane_refs = rec.lane_refs if rec is not None else set()
        return [
            r for s in slots
            if (r := wire_ref(spec.models[s])) not in lane_refs
        ]

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
    async def _exclusive_gpu(self) -> typing.AsyncIterator[None]:
        """Hold every worker GPU permit for setup/adoption proof warmups.

        Inductor exposes process-global cache counters. Acquiring only one
        permit on a multi-slot worker would let another graph increment them
        inside this target's before/after window and falsely certify it.
        These maintenance paths run before a job holds a permit themselves.
        """
        acquired = 0
        try:
            for _ in range(self._gpu_slots):
                await self._gpu_semaphore.acquire()
                acquired += 1
            yield
        finally:
            for _ in range(acquired):
                self._gpu_semaphore.release()

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
        rec = self._class_record(spec)
        async with rec.lock:
            if rec.ready and not rec.stale:
                for slot in self._setup_slots(spec):
                    ref = wire_ref(spec.models[slot])
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
                mandatory_w8a8 = any(
                    _ref_wants_w8a8(wire_ref(spec.models[slot]))
                    for slot in self._setup_slots(spec)
                )
                from . import compile_cache

                try:
                    desired_cell = await self._fetch_compile_snapshot(
                        spec, snapshots)
                except compile_cache.CompiledLaneUnavailableError as exc:
                    if mandatory_w8a8:
                        # Desired state no longer supplies a mandatory exact
                        # cell. Remove the old READY incarnation before
                        # reporting the state-driven failure; it must not keep
                        # serving under superseded scheduler evidence.
                        rec.stale = True
                        async with self._load_lock:
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
                            ))
                    if not target_identities or any(
                        active_ref != desired_cell.ref
                        or active_digest != desired_cell.snapshot_digest
                        for active_ref, active_digest in target_identities
                    ):
                        logger.info(
                            "desired compile identity moved for %s -> %s@%s; "
                            "vacating stale instance",
                            spec.name,
                            desired_cell.ref,
                            desired_cell.snapshot_digest,
                        )
                        rec.stale = True
            if rec.ready and rec.stale:
                # gw#494: the instance was loaded for a superseded pick —
                # vacate (releasing its OLD-ref bookings) and set up fresh
                # with the current bindings.
                async with self._load_lock:
                    await self._vacate_record(rec)
            if rec.ready:
                await self._promote_setup_refs(spec, promote_slots, rec=rec)
                return rec.instance
            try:
                instance = await self._setup_locked(spec, rec, snapshots)
            except Exception as exc:
                # Honest failure (th#581): a function whose model download /
                # pipeline setup fails must surface a terminal per-function
                # error to the hub, not sit in loading_functions forever
                # while the worker reports READY.
                from .compile_cache import CompiledLaneUnavailableError

                if isinstance(exc, CompiledLaneUnavailableError):
                    self._mark_compile_setup_unavailable(rec, spec, str(exc))
                    self._on_state_change()
                self._mark_setup_failed(rec, exc)
                raise
            if rec.failed is not None:
                # Recovery (desired-state retry succeeded): lift the
                # per-function disable; the next StateDelta re-advertises.
                rec.failed = None
                for s in rec.specs:
                    self.unavailable.pop(s.name, None)
            rec.instance = instance
            rec.ready = True
            self._clear_recovered_compile_failures(rec)
            self._on_state_change()
            return instance

    def _warmup_plan(
        self, spec: EndpointSpec, rec: _ClassRecord,
    ) -> Tuple[list[Any], list[Any]]:
        """Return gw#470's authoritative per-handler warmup contract."""
        if spec.kind != "inference" or spec.cls is None:
            return [], []
        from . import warmup as warmup_mod
        from .api.decorators import ATTR as _DECL_ATTR

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

    def _compile_contract_names(
        self, spec: EndpointSpec, rec: _ClassRecord,
    ) -> set[str]:
        """Handler aliases this setup can attribute its warmup proof to."""
        if spec.cls is not None and callable(getattr(spec.cls, "warmup", None)):
            # A custom object-level warmup has no per-handler attribution.
            return {spec.name}
        return self._required_compile_names(spec, rec)

    def _required_compile_names(
        self, spec: EndpointSpec, rec: _ClassRecord,
    ) -> set[str]:
        """Non-skipped aliases that a mandatory compiled setup must prove."""
        jobs, _skips = self._warmup_plan(spec, rec)
        names = {job.spec.name for job in jobs}
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
    ) -> _WarmupEvidence:
        """Run the declared per-handler warmup contract pre-READY.

        In addition to the successful call count, record which exact compiled
        objects served each handler. A sibling handler is never certified by
        another handler's cache hit merely because both share config or an
        instance. Output remains local and discarded.
        """
        from . import compile_cache, trt_engine

        jobs, skips = self._warmup_plan(spec, rec)
        for skip in skips:
            logger.info("boot warmup skipped for %s: %s", skip.spec.name, skip.reason)
        objects = tuple({id(obj): obj for obj in proof_objects}.values())
        evidence = _WarmupEvidence()
        for wj in jobs:
            before = {
                id(obj): (
                    compile_cache.execution_count(obj),
                    trt_engine.execution_count(obj),
                )
                for obj in objects
            }
            handler_kwargs = await self._handler_kwargs(wj.spec, snapshots or {})
            t0 = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="gw-warmup-") as tmp:
                payload = wj.build(tmp)
                ctx = RequestContext(
                    request_id=f"boot-warmup-{wj.spec.name}",
                    local_output_dir=tmp,
                    models={slot: wire_ref(b) for slot, b in wj.spec.models.items()},
                    **_resolve_slots_kwargs(wj.spec, None),
                    boot_warmup=True,
                )
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
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return evidence
            evidence.count += 1
            for obj in objects:
                calls_before, trt_before = before[id(obj)]
                inductor_proven = (
                    compile_cache.execution_count(obj) > calls_before
                    and compile_cache.cache_hit_count(obj) > 0
                )
                trt_proven = trt_engine.execution_count(obj) > trt_before
                if inductor_proven or trt_proven:
                    evidence.functions_by_object.setdefault(id(obj), set()).add(
                        wj.spec.name)
            logger.info(
                "boot warmup %s (%s): %.1fs",
                wj.spec.name, "declared" if wj.declared else "synthesized",
                time.monotonic() - t0)
        return evidence

    async def _invoke_warmup(
        self, spec: EndpointSpec, instance: Any, ctx: "RequestContext",
        payload: Any, kwargs: Dict[str, Any],
    ) -> None:
        bound = getattr(instance, spec.attr_name)
        call_kwargs = {spec.ctx_param: ctx, spec.payload_param: payload, **kwargs}
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

    def _mark_setup_failed(self, rec: _ClassRecord, exc: BaseException) -> None:
        if isinstance(exc, (InsufficientDiskError, RetryableError, MissingSnapshotError)):
            # Transient pressure (disk GC frees space / warm-tier RAM drains /
            # the hub re-mints a snapshot): fail the op RETRYABLE, never
            # disable the function.
            return
        if isinstance(exc, HardwareUnmetError):
            reason = getattr(exc, "reason", "hardware_unmet")
            axes = {str(k): str(v) for k, v in (exc.axes() or {}).items()}
        else:
            reason, axes = "setup_failed", {}
        detail = _sanitize(f"{type(exc).__name__}: {exc}")
        rec.failed = detail
        for s in rec.specs:
            self.unavailable[s.name] = (reason, detail, axes)
        self._on_state_change()

    async def _setup_locked(
        self, spec: EndpointSpec, rec: _ClassRecord,
        snapshots: Optional[Dict[str, pb.Snapshot]],
    ) -> Any:
        assert spec.cls is not None  # guarded by ensure_setup
        setup_slots = self._setup_slots(spec)
        # gw#494: residency keys for this setup are derived ONCE, here, in
        # resolved space; downloads, booking and the record's held_refs all
        # use these exact strings (a HelloAck rebind during an await below
        # cannot split download/booking/teardown identities).
        slot_refs: Dict[str, str] = {
            slot: wire_ref(spec.models[slot]) for slot in setup_slots
        }
        slot_identities: Dict[str, _ResidencyIdentity] = {}
        paths: Dict[str, str] = {}
        for slot in setup_slots:
            binding = spec.models[slot]
            ref = slot_refs[slot]
            snap = (snapshots or {}).get(ref)
            materialized = await self.store._materialize_local(
                ref, snap, binding=binding)
            paths[slot] = str(materialized.path)
            slot_identities[slot] = materialized.identity
        compile_selection = await self._fetch_compile_snapshot(spec, snapshots)
        compile_artifact = compile_selection.path if compile_selection else None
        # Loads serialize: concurrent setups would cross-contaminate each
        # other's allocator deltas and place_pipeline's free-VRAM reads.
        async with self._load_lock:
            await self._make_room_for(spec, setup_slots)
            # VRAM make-room may demote the old pipeline into host RAM. Admit
            # the incoming load only AFTER that transition so the probe sees
            # the actual post-demotion pressure (pgw#541).
            await self._ensure_host_ram_for(spec, paths)
            instance = spec.cls()
            setup = getattr(instance, "setup", None)
            inj = _InjectionResult(kwargs={}, loaded={})
            from . import compile_cache, trt_engine

            vram_before = self._vram_allocated()
            if spec.runtime:
                rec.server = await self._boot_engine_server(spec, paths)
            if callable(setup):
                inj = await self._injection_kwargs(
                    spec, setup, paths, server=rec.server,
                    compile_selection=compile_selection,
                    snapshots=snapshots,
                    slot_identities=slot_identities)
                rec.shared_keys.extend(inj.shared_keys)
                # pgw#517: a self-loading (str/Path-slot) endpoint builds its
                # own pipeline inside setup() and the executor never sees it
                # to arm compile automatically (the branch above only fires
                # for class-annotated slots) — hold the arming scope open so
                # a `gen_worker.arm_compile(pipe)` call from inside setup()
                # reaches the same cache-artifact-gated policy. No-op when
                # spec.compile is None.
                arming_scope = provision.ArmingScope(
                    spec.compile, self.store._cache_dir, compile_artifact,
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
                for pipe, armed in arming_scope.objects:
                    if not compile_cache.has_compile_target(pipe, spec.compile):
                        continue
                    inj.add_compile_object(pipe, self_loaded_slots)
                    if armed and compile_selection is not None:
                        inj.active_compile_artifacts[id(pipe)] = compile_selection
                        if trt_engine.is_engine_ref(compile_selection.ref):
                            inj.trt_execution_before[id(pipe)] = (
                                trt_engine.execution_count(pipe))
            proves_inductor = bool(
                compile_selection
                and inj.active_compile_artifacts
                and not trt_engine.is_engine_ref(compile_selection.ref)
            )
            proof_before = {
                id(candidate.pipeline): (
                    compile_cache.execution_count(candidate.pipeline),
                    compile_cache.cache_miss_count(candidate.pipeline),
                )
                for candidate in inj.compile_objects
                if proves_inductor
                and id(candidate.pipeline) in inj.active_compile_artifacts
            }
            warmup = getattr(instance, "warmup", None)

            async def run_warmup() -> Tuple[int, Dict[int, set[str]]]:
                if callable(warmup):
                    warm_t0 = time.monotonic()
                    if asyncio.iscoroutinefunction(warmup):
                        await warmup()
                    else:
                        await _to_thread_complete(warmup)
                    if spec.compile is not None:
                        logger.info(
                            "compile-cache warmup %s completed in %.1fs",
                            spec.name, time.monotonic() - warm_t0)
                    return 1, {}

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
                )
                return evidence.count, evidence.functions_by_object

            if inj.active_compile_artifacts:
                # Cache-hit counters are process-global. Hold every GPU permit
                # so each exact guard window can belong to only this warmup.
                async with self._exclusive_gpu():
                    warmed, function_proofs = await run_warmup()
            else:
                warmed, function_proofs = await run_warmup()
            if proves_inductor:
                unproven: list[_CompileObjectCandidate] = []
                hits = 0
                misses = 0
                for candidate in inj.compile_objects:
                    pipe = candidate.pipeline
                    before = proof_before.get(id(pipe))
                    if before is None:
                        continue
                    calls = compile_cache.execution_count(pipe) - before[0]
                    pipe_hits = compile_cache.cache_hit_count(pipe)
                    pipe_misses = compile_cache.cache_miss_count(pipe) - before[1]
                    hits += max(0, pipe_hits)
                    misses += max(0, pipe_misses)
                    if not warmed or calls <= 0 or pipe_hits <= 0:
                        unproven.append(candidate)
                    elif callable(warmup):
                        function_proofs[id(pipe)] = {spec.name}
                if unproven:
                    from .models.loading import pipeline_weight_lane

                    w8a8 = any(
                        pipeline_weight_lane(candidate.pipeline).startswith("w8a8")
                        for candidate in unproven
                    )
                    for candidate in unproven:
                        pipe = candidate.pipeline
                        function_proofs[id(pipe)] = set()
                        compile_cache.unwrap(pipe)
                        if int(getattr(spec.compile, "lora_bucket", 0) or 0):
                            compile_cache.drop_lora_lane(pipe)
                        inj.active_compile_artifacts.pop(id(pipe), None)
                    detail = (
                        f"{len(unproven)} attached compile object(s) did not "
                        "serve their own warmup graph "
                        f"(warmups={warmed}, cache_hits={hits}, "
                        f"cache_misses={misses})"
                    )
                    if w8a8:
                        raise compile_cache.CompiledLaneUnavailableError(detail)
                    logger.warning("%s; serving eager", detail)
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
                lane_slots=inj.lane_slots, shared_bytes=inj.shared_bytes,
                slot_refs=slot_refs, slot_identities=slot_identities)
            rec.held_refs = sorted(set(slot_refs.values()))
            rec.held_snapshot_digests = {
                slot_refs[slot]: identity[0]
                for slot, identity in slot_identities.items()
                if slot in slot_refs and identity[0]
            }
            rec.held_bindings = sorted(
                (
                    slot,
                    ref,
                    rec.held_snapshot_digests.get(ref, ""),
                )
                for slot, ref in slot_refs.items()
            )
            # gw#551: call-time-owned refs. Any record holding 2+ worker-
            # constructed pipelines can overcommit VRAM (content-keyed lanes
            # AND monolithic siblings alike) — those swap per use via the
            # LaneGate instead of being job-pinned + eagerly promoted.
            pipe_slots = {s for s, (obj, _) in inj.loaded.items() if obj is not None}
            swap_owned = pipe_slots if len(pipe_slots) >= 2 else set(inj.lane_slots)
            swap_owned &= inj.gated_slots  # un-gateable pipes stay eager
            rec.lane_refs = {slot_refs[s] for s in swap_owned if s in slot_refs}
            rec.held_objects = {}
            for slot, ref in slot_refs.items():
                obj = inj.loaded.get(slot, (None, 0))[0]
                if obj is not None or ref not in rec.held_objects:
                    rec.held_objects[ref] = obj
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

    def _register_residency(
        self,
        spec: EndpointSpec,
        setup_slots: List[str],
        loaded: Dict[str, Tuple[Any, int]],
        total_delta: int,
        *,
        lane_slots: Optional[set] = None,
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
        lanes = lane_slots or set()
        refs = slot_refs or {}
        identities = slot_identities or {}
        per_ref: Dict[str, Tuple[Any, int]] = {}
        per_ref_identity: Dict[str, _ResidencyIdentity] = {}
        for slot in setup_slots:
            if slot in lanes:
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
        lane_bytes = sum(loaded[s][1] for s in lanes if s in loaded)
        residual = max(0, total_delta - sum(b for _, b in per_ref.values())
                       - lane_bytes - max(0, int(shared_bytes)))
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
        lane_refs = rec.lane_refs if rec is not None else set()
        refs = [
            r for s in setup_slots
            if (r := wire_ref(spec.models[s])) not in lane_refs
        ]
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
        self, refs: List[str], error: InsufficientHostRamError,
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

    async def _reclaim_released_file_cache(
        self, released_refs: List[str], incoming_paths: List[Path],
    ) -> int:
        """Advise only pressure-evicted immutable snapshots out of file cache.

        A DISK transition preserves model bytes, but recently-read clean pages
        can still fill the pod cgroup and make the following load look
        impossible.  Protect the incoming snapshot and every still-loaded or
        executing ref by inode; then advise only refs that truthfully reached
        DISK.  The caller always re-probes measured headroom afterward.

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
        for ref in dict.fromkeys(released_refs):
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
                if self._record_in_use(rec):
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
            advised = await self._reclaim_released_file_cache(
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

        error = InsufficientHostRamError(
            spec.name,
            incoming_bytes=incoming,
            floor_bytes=after.floor_bytes,
            required_bytes=after.required_bytes,
            available_before_bytes=before.available_bytes,
            available_after_bytes=after.available_bytes,
            evicted_refs=tuple(_canonical_host_ram_refs(evicted)),
        )
        # th#807: model-failure state is the scheduler's typed capacity seam.
        # Only the largest sequentially staged ref(s) caused this admission
        # decision; failing a smaller shared VAE would poison unrelated jobs.
        await self._record_host_ram_failure(incoming_refs, error)
        raise error

    async def _make_room_for(self, spec: EndpointSpec, setup_slots: List[str]) -> None:
        """Evict idle LRU pipelines before loading instead of degrading the
        new load down the offload ladder (#371). Estimate: per-ref vram_hint
        from a prior load, else the endpoint's declared vram_gb."""
        res = self.store.residency
        refs = [wire_ref(spec.models[s]) for s in setup_slots]
        hints = [res.vram_hint(r) for r in refs]
        needed = sum(hints)
        # A partially-known set is still an unknown load. Live SDXL exposed
        # the old sum-only bug: the fixed VAE had a small prior hint while a
        # never-seen checkpoint had none, so admission reserved only the VAE
        # and loaded a ~10GiB pipeline into an occupied 24GiB card.
        if any(h <= 0 for h in hints) and spec.resources.vram_gb:
            needed = max(needed, int(float(spec.resources.vram_gb) * _GiB))
        if needed <= 0:
            return
        # CPU-only workers do not have a VRAM tier to admit against.
        if torch is None or not torch.cuda.is_available():
            return
        if await asyncio.to_thread(res.make_room, needed):
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
            if self._record_in_use(rec):
                continue
            await self._vacate_record(rec)
            if await asyncio.to_thread(res.make_room, needed):
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
        from .models.serve_fit import RUN_OFFLOAD

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
        from .runtimes.server import RUNTIME_FACTORIES

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
        if spec.compile is None or not snapshots:
            return None
        from . import compile_cache, trt_engine
        family = getattr(spec.compile, "family", "") or ""
        # The effective spec is already rebound to this RunJob's selected
        # checkpoints. Snapshot maps also contain attached cells and may carry
        # unrelated/prepositioned models, so they must not choose the lane.
        model_refs = [wire_ref(binding) for binding in spec.models.values()]
        wants_w8a8 = any(_ref_wants_w8a8(ref) for ref in model_refs)
        want_bucket = int(getattr(spec.compile, "lora_bucket", 0) or 0)
        if wants_w8a8:
            # TensorRT cells currently expose only their plain fp16 contract.
            # The existing Forge's -w8a8 Inductor cell is the sole artifact
            # proven to preserve Fp8ScaledLinear/torch._scaled_mm semantics.
            candidates = [
                (ref, snap) for ref, snap in snapshots.items()
                if _cell_lane_matches(
                    ref, family, wants_w8a8=True, want_bucket=want_bucket)
            ]
        else:
            trt_candidates = [
                (ref, snap) for ref, snap in snapshots.items()
                if trt_engine.is_engine_ref(ref, family)
            ] if not want_bucket else []
            inductor_candidates = [
                (ref, snap) for ref, snap in snapshots.items()
                if _cell_lane_matches(
                    ref, family, wants_w8a8=False, want_bucket=want_bucket)
            ]
            # Explicit kind policy, then uniqueness within that kind. A map's
            # iteration order never chooses the artifact, while the existing
            # measured plain-lane TRT preference remains intact.
            candidates = trt_candidates or inductor_candidates
        candidates = sorted(candidates, key=lambda item: item[0])
        if wants_w8a8 and not candidates:
            raise compile_cache.CompiledLaneUnavailableError(
                f"no exact W8A8 Forge cell attached for family={family!r} "
                f"lora_bucket={want_bucket}"
            )
        if len(candidates) > 1:
            refs = ", ".join(ref for ref, _snap in candidates)
            detail = (
                "multiple compatible compiled artifacts were attached for "
                f"family={family!r} lane={'w8a8' if wants_w8a8 else 'plain'}: "
                f"{refs}; refusing map-order selection"
            )
            if wants_w8a8:
                # W8A8 has no eager-compatible fallback: setup's mandatory
                # lane gate must surface this as retryable before GPU load.
                raise compile_cache.CompiledLaneUnavailableError(detail)
            logger.warning("%s; serving eager", detail)
            return None
        if candidates:
            ref, snap = candidates[0]
            digest = str(snap.digest or "").strip()
            if not digest:
                detail = f"compiled-artifact snapshot {ref!r} has no immutable digest"
                if wants_w8a8:
                    raise compile_cache.CompiledLaneUnavailableError(detail)
                logger.warning("%s; serving eager", detail)
                return None
            try:
                local = await self.store.ensure_local(ref, snap)
                artifact = compile_cache.find_artifact(local)
                if artifact is None:
                    if wants_w8a8:
                        raise compile_cache.CompiledLaneUnavailableError(
                            f"W8A8 Forge snapshot {ref!r} contains no artifact")
                    logger.warning(
                        "compiled-artifact snapshot %s contains no artifact; "
                        "serving eager", ref)
                    return None
                return _CompileArtifactSelection(
                    path=artifact, ref=ref, snapshot_digest=digest)
            except Exception as exc:
                if wants_w8a8 and isinstance(
                    exc, compile_cache.CompiledLaneUnavailableError
                ):
                    raise
                if wants_w8a8:
                    raise compile_cache.CompiledLaneUnavailableError(
                        f"W8A8 Forge snapshot {ref!r} is unusable: {exc}") from exc
                logger.warning(
                    "compiled-artifact snapshot %s unusable (%s); serving eager", ref, exc
                )
        return None

    def _component_share_plan(
        self, spec: EndpointSpec, paths: Dict[str, str], hints: Dict[str, Any]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Content-keyed shared-component plan for a multi-lane record
        (gw#479): ``{slot: {component: LoadedComponentKey}}`` restricted to
        components whose CONTENT key appears under 2+ pipeline slots. None
        when the record has <2 pipeline slots, digests are unavailable, or
        nothing is byte-identical — loading then stays monolithic."""
        pipe_slots = [
            s for s in paths
            if isinstance(hints.get(s), type)
            and callable(getattr(hints[s], "from_pretrained", None))
        ]
        if len(pipe_slots) < 2:
            return None
        keys: Dict[str, Dict[str, Any]] = {}
        for slot in pipe_slots:
            binding = spec.models.get(slot)
            if binding is None:
                return None
            ref = wire_ref(binding)
            digests = self.store.component_digests(ref, local_path=Path(paths[slot]))
            keys[slot] = {
                comp: residency_mod.LoadedComponentKey.for_component(
                    content_digest=digest, component=comp, binding=binding,
                    label=f"{ref}/{comp}",
                )
                for comp, digest in digests.items() if comp
            }
        counts: Dict[Any, int] = {}
        for slot_keys in keys.values():
            for k in slot_keys.values():
                counts[k] = counts.get(k, 0) + 1
        plan = {
            slot: {c: k for c, k in slot_keys.items() if counts[k] >= 2}
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
        from .runtimes.server import ServerHandle

        try:
            hints = typing.get_type_hints(setup)
        except Exception:
            hints = {}
        kwargs: Dict[str, Any] = {}
        loaded: Dict[str, Tuple[Any, int]] = {}
        result = _InjectionResult(kwargs=kwargs, loaded=loaded)
        compile_artifact = compile_selection.path if compile_selection else None
        share_plan = self._component_share_plan(spec, paths, hints)
        if server is not None:
            for pname, ann in hints.items():
                if ann is ServerHandle:
                    kwargs[pname] = server
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
                        await _to_thread_complete(res.make_room, excl_bytes)
                before = self._vram_allocated()
                try:
                    sl = await _to_thread_complete(
                        provision.load_slot, ann, path, binding=binding,
                        slot=slot, ref=ref, mode=mode, components=injected,
                        declared_vram_gb=float(
                            getattr(spec.resources, "vram_gb", 0) or 0),
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
                            getattr(spec.resources, "vram_gb", 0) or 0),
                    )
                pipe = sl.obj
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
                    raise RetryableError(
                        f"lane {slot!r} of {spec.name} placed "
                        f"{placed.get('mode')!r}: shared-component lanes "
                        "require resident placement; retrying")
                if spec.compile is not None:
                    # Opt-in acceleration against a pre-built per-SKU artifact:
                    # a TRT engine (#390, refit with this pipeline's weights)
                    # or an inductor cache (#384). No verified artifact =>
                    # stays eager. ``compile_artifact`` is hub-attached (#569).
                    armed = await _to_thread_complete(
                        self._enable_compiled, pipe, spec.compile, compile_artifact,
                    )
                    from . import compile_cache

                    if compile_cache.has_compile_target(pipe, spec.compile):
                        result.add_compile_object(pipe, (slot,))
                        if armed and compile_selection is not None:
                            result.active_compile_artifacts[id(pipe)] = compile_selection
                            from . import trt_engine

                            if trt_engine.is_engine_ref(compile_selection.ref):
                                result.trt_execution_before[id(pipe)] = (
                                    trt_engine.execution_count(pipe))
                delta = max(0, self._vram_allocated() - before)
                if slot_share:
                    lane_obj, lane_bytes = self._register_lane(
                        slot,
                        ref,
                        pipe,
                        slot_share,
                        injected,
                        delta,
                        result,
                        (slot_identities or {}).get(slot, ("", 0)),
                    )
                    loaded[slot] = (lane_obj, lane_bytes)
                    result.lane_slots.add(slot)
                else:
                    loaded[slot] = (pipe, delta)
                    if self._arm_lane_gate(pipe, ref, spec=spec):
                        result.gated_slots.add(slot)
                kwargs[slot] = pipe
            else:
                kwargs[slot] = path
        return result

    def _register_lane(
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
        lane_obj: Any = nn.ModuleDict(exclusive) if exclusive else pipe
        lane_bytes = max(0, delta - fresh_bytes)
        result.shared_bytes += fresh_bytes
        logger.info(
            "lane %s (%s): exclusive %s (%.2f GiB), shared %s (%.2f GiB %s)",
            slot, ref, sorted(exclusive) or ["<none>"], lane_bytes / _GiB,
            sorted(slot_share), fresh_bytes / _GiB,
            "fresh" if fresh_bytes else "reused",
        )
        self.store.activate_load_identity(ref, load_identity)
        if lane_bytes > 0:
            res.track_vram(ref, lane_obj, vram_bytes=lane_bytes)
        elif int(estimate_cuda_resident_gb(lane_obj) * _GiB) > 0:
            res.track_vram(ref, lane_obj)
        else:
            res.track_ram(ref, lane_obj)
        if self._arm_lane_gate(pipe, ref):
            result.gated_slots.add(slot)
        return lane_obj, lane_bytes

    def _arm_lane_gate(
        self, pipe: Any, ref: str, spec: Optional[EndpointSpec] = None,
    ) -> bool:
        """gw#551: wrap a worker-constructed pipeline's ``__call__`` so a
        demoted/incomplete residency entry is promoted (pinned, idle sibling
        LRU-swapped out) before it executes — a cpu-resident lane must never
        run. No-op for offload-hooked pipelines (they own their placement).
        Monolithic pipelines (``spec`` given) additionally get the last-resort
        offload fallback; shared-component lanes never do (hooks on a shared
        module would poison sibling lanes)."""
        from .models.lane_gate import LaneGate, arm_lane_gate

        fallback = None
        if spec is not None:
            bound_spec = spec

            def fallback() -> bool:
                return self._serve_offload_fallback(bound_spec, pipe, ref)
        return arm_lane_gate(pipe, LaneGate(
            ref=ref, residency=self.store.residency, label=ref,
            retry_exc=RetryableError, offload_fallback=fallback,
        ))

    def _serve_offload_fallback(self, spec: EndpointSpec, pipe: Any, ref: str) -> bool:
        """Serve-time last resort (gw#551): promote could not fit even after
        LRU demotions — arm a coherent CPU-offload rung on the (cpu-resident)
        pipeline and rebook it honestly, instead of failing the request."""
        from .models.memory import rearm_offload

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

    def _enable_compiled(self, pipe: Any, cfg: Any, artifact: Optional[Path]) -> bool:
        """Arm the best available compiled path for a freshly loaded pipeline
        (shared with the local CLI — provision.enable_compiled)."""
        return provision.enable_compiled(pipe, cfg, self.store._cache_dir, artifact)

    @staticmethod
    def _vram_allocated() -> int:
        if torch is not None and torch.cuda.is_available():
            try:
                return int(torch.cuda.memory_allocated())
            except Exception:
                return 0
        return 0

    async def shutdown_instances(self) -> None:
        for rec in self._classes.values():
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
        async with self._compile_cache_adoption_lock:
            await self._handle_compile_cache_adoption(op)

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

    async def _handle_compile_cache_adoption(self, op: pb.ModelOp) -> None:
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
            return
        try:
            await self._adopt_compile_cache(
                ref, snap, snapshot_digest, operation_id, target_incarnation_id,
            )
        except Exception as exc:
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
    ) -> None:
        """Hot adoption (th#567): download+verify a compiled artifact and
        re-wrap the already-resident modules in place — weights untouched, no
        reload, one warmup. Handles BOTH cell kinds on the same rails: an
        inductor cache (#384: seed dirs + torch.compile) and a TRT engine
        (#390: deserialize + refit with the resident weights + module swap).
        ANY failure => stay eager and report ``adopt_failed:<reason>``;
        adoption must never degrade service."""
        from . import compile_cache, trt_engine

        t0 = time.monotonic()
        staged_artifact: Any = None

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
            if detail and reason not in (
                "model_in_use", "target_not_ready", "target_replaced", "download",
            ):
                error = f"{error}: {detail[:300]}"
            await self._send(pb.WorkerMessage(
                model_event=self._adoption_event(
                    ref,
                    pb.MODEL_STATE_FAILED,
                    snapshot_digest,
                    operation_id,
                    target_incarnation_id,
                    error=error,
                )
            ))

        family = compile_cache.family_from_ref(ref)
        is_trt = trt_engine.is_engine_ref(ref)
        if not family or not (is_trt or compile_cache.is_cache_ref(ref)):
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
            await self._send(pb.WorkerMessage(model_event=self._adoption_event(
                ref,
                pb.MODEL_STATE_ADOPTED,
                snapshot_digest,
                operation_id,
                target_incarnation_id,
                duration_ms=0,
            )))
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
        try:
            local = await self.store.ensure_local(ref, snap)
        except Exception as exc:
            return await fail("download", str(exc))
        artifact = compile_cache.find_artifact(local)
        if artifact is None:
            return await fail("artifact_missing")
        if not is_trt:
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

        # Artifact work may take long enough for model juggling to replace the
        # object. Serialize the final check + mutation with setup/vacate, and
        # address only the exact incarnation observed before the download.
        async with self._load_lock:
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
            async with self._exclusive_gpu():
                current = self._compile_target(target_incarnation_id)
                if (
                    current is None
                    or current[0] is not expected_rec
                    or current[1] is not expected_target
                ):
                    return await fail("target_replaced")

                rec, target = current
                spec = target.spec
                cfg = spec.compile
                assert cfg is not None
                obj = target.pipeline
                wrapped = False
                lane_applied = False
                trt_before = trt_engine.execution_count(obj) if is_trt else 0
                inductor_before = (0, 0, 0)

                async def rollback() -> None:
                    """Return a first-time failed adoption to honest eager."""
                    if is_trt and wrapped:
                        trt_engine.unwrap(obj)
                    if wrapped:
                        if not is_trt:
                            compile_cache.unwrap(obj)
                    if lane_applied:
                        compile_cache.drop_lora_lane(obj)
                    live = self._compile_target(target_incarnation_id)
                    if live is not None and live[0] is rec and live[1] is target:
                        self._refresh_compile_target(target)
                        self._on_state_change()

                bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
                if bucket and not is_trt:
                    try:
                        compile_cache.apply_lora_lane(obj, bucket)
                        lane_applied = True
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

                if not is_trt:
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
                hits = 0
                misses = 0

                if not is_trt:
                    calls = compile_cache.execution_count(obj) - inductor_before[0]
                    hits = compile_cache.cache_hit_count(obj) - inductor_before[1]
                    misses = compile_cache.cache_miss_count(obj) - inductor_before[2]
                    if not warmed:
                        await rollback()
                        return await fail(
                            "no_warmup",
                            "target defines no runnable warmup; cache hits unprovable")
                    if calls <= 0 or hits <= 0:
                        await rollback()
                        return await fail("cache_miss", (
                            "exact target warmup did not execute a cache-hit "
                            f"compiled graph (calls={calls}, hits={hits}, "
                            f"misses={misses}, warmup={warmup_s}s) — cell useless "
                            f"on this runtime, serving eager"))
                elif trt_engine.execution_count(obj) <= trt_before:
                    await rollback()
                    return await fail(
                        "engine_not_executed",
                        "warmup did not execute the attached TRT engine",
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

    def _record_in_use(self, rec: _ClassRecord) -> bool:
        # A job on a rebound spec no longer references the record's held
        # refs — membership of the job's spec in this record is the honest
        # "instance in use" signal (gw#494).
        for job in self.jobs.values():
            if job.finished or job.superseded or job.spec is None:
                continue
            if job.spec in rec.specs:
                return True
        for ref in self._record_refs(rec):
            owners = self._records_holding(ref)
            if (len(owners) == 1 and owners[0] is rec
                    and self.store.residency.in_use(ref)):
                return True
        return False

    async def _vacate_record(self, rec: _ClassRecord) -> List[str]:
        """Tear an instance down and return refs whose owner was released."""
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
        if torch is not None and torch.cuda.is_available():
            try:
                await asyncio.to_thread(torch.cuda.empty_cache)
            except Exception:
                pass
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
            if self.store.residency.release_to_disk(ref) and (
                old_obj is not None
                or tier_before in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM)
            ):
                released_refs.append(ref)
        rec.held_refs = []
        rec.held_snapshot_digests = {}
        rec.held_bindings = []
        rec.lane_refs = set()
        rec.held_objects = {}
        # Do not let this teardown frame itself retain a departing pipeline
        # while the cgroup probe decides whether capacity really progressed.
        old_obj = None
        replacement = None
        owners = []
        held_objects.clear()
        rec.stale = False
        if rec.shared_keys:
            # Drop this record's holds on content-keyed shared components
            # (gw#479); entries no other record references get evicted.
            for key in rec.shared_keys:
                self.store.residency.release_shared(key)
            rec.shared_keys.clear()
            self.store.residency.drain_shared()
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
                if job.ctx is not None:
                    job.ctx._cancel()
                if job.exec_task is not None:
                    job.exec_task.cancel()

        if self.draining:
            await self._send_result(run.request_id, run.attempt, pb.JOB_STATUS_RETRYABLE,
                                    safe_message="worker draining")
            return
        spec = self.specs.get(run.function_name)
        if spec is None:
            await self._send_result(run.request_id, run.attempt, pb.JOB_STATUS_INVALID,
                                    safe_message=f"unknown function {run.function_name!r}")
            return
        if run.function_name in self.unavailable:
            reason, detail, _ = self.unavailable[run.function_name]
            await self._send_result(run.request_id, run.attempt, pb.JOB_STATUS_RETRYABLE,
                                    safe_message=f"function unavailable: {reason}")
            return

        job = _Job(request_id=run.request_id, attempt=run.attempt, spec=spec)
        self.jobs[key] = job
        self._idle.clear()
        logger.info("job admitted %s attempt=%d", run.request_id, run.attempt)
        await self._send(pb.WorkerMessage(job_accepted=pb.JobAccepted(
            request_id=run.request_id, attempt=run.attempt)))
        job.task = asyncio.create_task(self._run_job(job, run), name=f"job-{run.request_id}")

    def handle_cancel(self, cancel: pb.CancelJob) -> None:
        job = self.jobs.get((cancel.request_id, cancel.attempt))
        if job is None or job.finished:
            return  # unknown pair or natural result already stands
        if job.ctx is not None:
            job.ctx._cancel()  # cooperative: sync handlers poll ctx
        if job.exec_task is not None and job.spec is not None and job.spec.is_async:
            job.exec_task.cancel()  # async handlers are cancelled on the loop

    async def wait_idle(self, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

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

    async def _run_job(self, job: _Job, run: pb.RunJob) -> None:
        spec = job.spec
        assert spec is not None
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
            self._validate_required_compile(spec, run)
        except Exception as exc:
            status, msg = _map_exception(exc)
            await self._finish(job, status, safe_message=msg)
            return
        routed = list(spec.models)
        # Pin this job's model refs for its WHOLE lifetime (gw#409): the gap
        # between ensure_setup's promote and the execution-time pin let a
        # concurrent job's make_room demote a just-promoted pipeline. Refs
        # without entries yet are no-ops; the inner pin still covers adapters.
        # Lane refs are NOT job-pinned (gw#551): lane dispatch is handler-side,
        # so pinning every declared lane would make the idle sibling
        # un-demotable and the used lane un-promotable on an overcommitted
        # card; the LaneGate pins exactly the lane it executes, at call time.
        try:
            with self.store.residency.executing(
                *self._job_pin_refs(spec, routed)
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
                hf_token=getattr(self._settings, "hf_token", "") or "",
            )

        ctx_cls = _CONTEXT_BY_KIND.get(spec.kind, RequestContext)
        ctx = ctx_cls(
            request_id=run.request_id,
            job_id=job_id,
            emitter=self._make_ctx_emitter(job),
            owner=run.tenant or None,
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
            if source_info:
                await self._materialize_source(ctx, source_info, snapshots)
            if producer:
                await self._materialize_datasets(ctx, payload)
            # Typed media inputs: URL-ref Assets (hub-approved remote media)
            # are downloaded and given a local_path before the handler runs.
            await asyncio.to_thread(materialize_input_assets, payload, run.request_id)
            instance = await self.ensure_setup(spec, snapshots, promote_slots=routed)
            kwargs = await self._handler_kwargs(spec, snapshots)
            adapters = await self._prepare_adapters(run, spec, snapshots)
        except asyncio.CancelledError:
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
        try:
            if needs_gpu:
                await self._gpu_semaphore.acquire()
                self._loop = asyncio.get_running_loop()
                lease = _GpuSlotLease(self._gpu_semaphore, self._loop)
                ctx._gpu_slot_lease = lease
                # gw#516: the handler thread reports the terminal
                # decode->finalize slot release so the hub sees the job as
                # "finalizing" while its encode/upload tail runs slotless.
                ctx._on_finalize_release = lambda: self._enter_finalize(job)
                if job.ctx.cancelled:
                    raise CanceledError("canceled")
                # pgw#513: reset the CUDA peak-allocator watermark now that
                # this job exclusively owns gpu_index (jobs serialize under
                # _gpu_semaphore) — peak_vram_bytes then measures THIS job's
                # peak, not the process-lifetime high-water mark.
                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats(gpu_index)
                    except Exception:
                        pass
            # Last execution fence: no adapter mutation or tenant handler has
            # run yet. The repeated check catches a replacement between
            # scheduler assignment/intake and this GPU turn.
            self._validate_required_compile(spec, run)
            started = time.monotonic()
            # Pin-while-executing: the models (and adapter snapshots) this job
            # uses are not eviction candidates for its duration. Lane refs
            # excluded (gw#551): the LaneGate pins the one lane the handler
            # actually calls; pinning all of them here would deadlock the
            # gate's promote against its own job's pins.
            exec_refs = self._job_pin_refs(spec, routed)
            adapter_refs = [a.ref for group in adapters.values() for a in group]
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
                    # Explicit activation (gw#399): a slot this request names
                    # no adapters for must run bare even if a previous
                    # request's teardown failed and left adapters enabled.
                    for slot in spec.models:
                        if slot in adapters:
                            continue
                        ref = wire_ref(spec.models[slot])
                        if self._adapters.needs_deactivation(ref):
                            pipe = self.store.residency.obj(ref)
                            if pipe is not None:
                                await asyncio.to_thread(
                                    self._adapters.deactivate, ref, pipe, run.request_id
                                )
                    try:
                        output = await self._execute(job, spec, instance, ctx, payload, kwargs,
                                                     timeout_ms=timeout_ms, gpu_index=gpu_index)
                    except BaseException as exc:
                        # A mid-inference CUDA OOM learns a per-ref floor, but
                        # the live object is quarantined. The hub retries only
                        # after ensure_setup reloads it cleanly at that rung.
                        await self._quarantine_for_oom(spec, ctx, exc)
                        raise
                finally:
                    # Guaranteed-inactive on every exit (OK / cancel /
                    # deadline / handler error); attachments stay resident.
                    for ref, pipe in active:
                        await asyncio.to_thread(
                            self._adapters.deactivate, ref, pipe, run.request_id
                        )
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index,
                                    output=output)
            handler_done = time.monotonic()
            # Handler GPU work is done — free the slot before result-blob
            # upload and result send so the next job's compute starts now.
            if lease is not None:
                overlapped = not lease.yield_slot()
                released_at = lease.released_at
                if released_at is not None:
                    # gw#516 typed split of runtime_ms: how long the GPU slot
                    # was actually held vs the slotless finalize tail.
                    metrics.slot_held_ms = max(
                        0, int((released_at - started) * 1000))
                    metrics.finalize_wall_ms = max(
                        0, int((handler_done - released_at) * 1000))
                if overlapped and released_at is not None:
                    # The handler released terminally at the decode->finalize
                    # handoff (gw#476/gw#516): the whole encode/upload tail
                    # overlapped the next request.
                    logger.info(
                        "FINALIZE_OVERLAP fn=%s request=%s slot_held_ms=%d "
                        "handler_wall_ms=%d overlap_ms=%d",
                        spec.name, run.request_id,
                        int((released_at - started) * 1000),
                        int((handler_done - started) * 1000),
                        int((handler_done - released_at) * 1000))
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
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index)
            await self._finish(job, pb.JOB_STATUS_FATAL, safe_message="deadline exceeded",
                               metrics=metrics)
        except BaseException as exc:
            from .compile_cache import CompiledLaneUnavailableError

            if isinstance(exc, CompiledLaneUnavailableError):
                # A seeded W8A8 graph failing at call time is a genuine lane
                # failure, not an invitation to retry eager under the same
                # advertised function. Remove this worker/function pair from
                # dispatch until a fresh setup with a compatible cell.
                found = None
                if run.HasField("required_compile"):
                    found = self._compile_target(
                        run.required_compile.target_incarnation_id)
                if found is not None and spec.name in found[1].function_names:
                    self._mark_compile_target_unavailable(
                        found[0], found[1], str(exc))
                elif spec.name not in self.unavailable:
                    # Defensive path for a custom compiled lane that raised
                    # without a live protocol target. It is intentionally not
                    # recovery-owned: no exact fresh target can prove it safe.
                    self.unavailable[spec.name] = (
                        "compile_cell_failed", _sanitize(str(exc)), {},
                    )
                self._on_state_change()
            status, msg = _map_exception(exc)
            if status == pb.JOB_STATUS_FATAL:
                logger.exception("handler %s failed", spec.name)
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index)
            await self._finish(job, status, safe_message=msg, metrics=metrics)
        finally:
            if lease is not None:
                lease.yield_slot()
            # gw#516: result shipped (any terminal path) — the job leaves the
            # finalizing set the hub gates drain/retire on.
            self._exit_finalize(job)
            self._maybe_idle()

    async def _materialize_source(
        self, ctx: Any, info: Dict[str, Any], snapshots: Dict[str, pb.Snapshot]
    ) -> None:
        """Reserved-source contract (#376): materialize ``payload.source``
        locally before the handler runs. Same ModelStore path as model
        bindings — identical retry/classification and ModelEvent emission."""
        ref = str(info.get("ref") or "").strip()
        if not ref:
            raise ValidationError("payload.source.ref must be a non-empty repo ref")
        path = await self.store.ensure_local(ref, snapshots.get(ref))
        ctx._set_source_path(str(path))

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
        import inspect as _inspect

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
                # carry an algo prefix ("blake3:<hex>") while path.name is the
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

    def _adapter_target(self, spec: EndpointSpec, slot: str) -> Any:
        """The worker-managed pipeline object adapters for ``slot`` apply to."""
        pipe = self.store.residency.obj(wire_ref(spec.models[slot]))
        if pipe is None:
            raise ValidationError(
                f"model slot {slot!r} has no worker-managed pipeline; "
                "lora overlays require a pipeline-injected setup slot"
            )
        return pipe

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
            obj = self.store.residency.obj(ref)
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
                os._exit(70)
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

    def _metrics(
        self, queue_ms: int, started: float, concurrency_at_start: int, gpu_index: int,
        output: Any = None,
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
        peak_vram = 0
        if torch is not None and torch.cuda.is_available():
            try:
                peak_vram = int(torch.cuda.max_memory_allocated(gpu_index))
            except Exception:
                pass
        duration_s, output_count = _scan_output_assets(output)
        usage = _output_token_usage(output)
        return pb.JobMetrics(
            runtime_ms=runtime_ms, queue_ms=queue_ms, rss_at_end_bytes=rss_at_end,
            peak_vram_bytes=peak_vram, concurrency_at_start=max(0, concurrency_at_start),
            output_media_duration_s=duration_s, output_count=output_count,
            input_tokens=usage.prompt_tokens if usage is not None else 0,
            input_cached_tokens=usage.cached_tokens if usage is not None else 0,
            output_tokens=usage.completion_tokens if usage is not None else 0,
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
    ) -> None:
        result = pb.JobResult(request_id=request_id, attempt=attempt, status=status,
                              safe_message=safe_message)
        if inline is not None:
            result.inline = inline
        elif blob_ref:
            result.blob_ref = blob_ref
        if metrics is not None:
            result.metrics.CopyFrom(metrics)
        await self._send(pb.WorkerMessage(job_result=result))

    async def _finish(self, job: _Job, status: "pb.JobStatus", **kw: Any) -> None:
        if job.finished:
            return
        job.finished = True
        if job.renew_task is not None:
            job.renew_task.cancel()
            job.renew_task = None
        cleanup_input_assets(job.request_id)
        logger.info("job finished %s attempt=%d status=%s", job.request_id, job.attempt, status)
        if not job.superseded:
            await self._send_result(job.request_id, job.attempt, status, **kw)
        # Keep finished records so a RunJob retransmission doesn't re-execute;
        # prune oldest finished entries beyond a small window.
        finished = [k for k, j in self.jobs.items() if j.finished]
        if len(finished) > 1024:
            for k in finished[: len(finished) - 1024]:
                self.jobs.pop(k, None)
        self._maybe_idle()

    def _maybe_idle(self) -> None:
        if not self.in_flight_keys():
            self._idle.set()


class _DeadlineExceeded(Exception):
    pass
