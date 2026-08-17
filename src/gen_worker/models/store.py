"""The on-disk/CAS model store: resolve -> materialize -> reference-count.

The longest-lived thing the worker owns — a pod's store outlives every job on
it. The job engine keeps ``executor.py``; nothing here reads a wire message.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
import shutil
import threading
import time
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, cast

from .. import activity as activity_mod
from .. import boot_phases as boot_mod
from .. import progress as progress_mod
from ..capability import InsufficientDiskError
from ..lifecycle_intents import IntentRegistry
from ..pb import worker_scheduler_pb2 as pb
from ..redact import sanitize as _sanitize
from ..topology import current_device_group
from . import cozy_snapshot, disk_gc, disk_telemetry, projection
from . import residency as residency_mod
from . import staging as staging_mod
from .cache_paths import tensorhub_cas_dir, tensorhub_fill_source_dir
from .config_identity import CANONICAL_JSON_MAX_BYTES, canonical_json_digest
from .cozy_snapshot import _norm_rel_path, delete_blobs
from .download import ensure_local, lookup_provider_for_ref
from .errors import MissingSnapshotError, UrlExpiredError
from .hub_client import (
    WorkerResolvedChunk,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)
from .loading import safetensors_file_valid
from .refs import WireRef
from .residency import Residency
from .volume_verify import (
    snapshot_verify_targets,
    split_projection_targets,
    verify_files,
    verify_projection,
)

logger = logging.getLogger(__name__)

__all__ = ["ModelStore"]

_GiB = 1024 ** 3
_DOWNLOAD_RETRIES = 3
_PROGRESS_EVENT_MIN_INTERVAL_S = 5.0
# How long a cold tensorhub ref waits for the hub's re-minted snapshot after
# reporting missing_snapshot. The FAILED event triggers an immediate hub-side
# re-mint (resolve + DOWNLOAD push), so arrival is seconds; the bound only
# caps a hub that never answers.
_MISSING_SNAPSHOT_WAIT_S = 60.0

# Disk headroom preserved beyond a download's known size.
_DISK_GC_MARGIN_BYTES = 2 * _GiB
# Refs used within the grace window are not disk-GC candidates.
_DISK_GC_GRACE_S = 300.0

# ---------------------------------------------------------------------------
# Model seam: models.download (ensure-local) + models.residency (tier map),
# with ModelEvent emission. Single-loop, per-ref asyncio locks.

def _snapshot_to_resolved(snap: pb.Snapshot) -> "WorkerResolvedRepo":
    """pb.Snapshot -> the typed resolved-manifest struct: the ONE
    wire-boundary conversion; everything downstream (ensure_local,
    ensure_snapshot_async) is typed — no dict laundering."""

    return WorkerResolvedRepo(
        snapshot_digest=snap.digest,
        files=[
            WorkerResolvedRepoFile(
                path=f.path,
                size_bytes=int(f.size_bytes),
                url=f.url or None,
                # The algorithm-tagged digest and the
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
            )
            for f in snap.files
        ],
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
        # endpoint-scoped datacenter-warm
        # fill source (RunPod volume mount), consulted before R2 on a blob
        # miss — resolved once at boot like _cache_dir; never the CAS root.
        # Same `or` shape as _cache_dir above: an explicit path (tests) wins,
        # otherwise resolve from env (production/tensorhub; unset -> None,
        # the cozy-local/no-volume degenerate case).
        self._fill_source_dir = fill_source_dir or tensorhub_fill_source_dir()
        # a datacenter pod without a warm fill
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
        # ONE Residency registry per execution group, sharing
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
        self.keep: list[str] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._index = disk_gc.RefIndex(self._cache_dir)
        self._disk_free = disk_free_bytes_fn or self._default_disk_free
        # Refs whose on-disk snapshot passed integrity verification THIS boot
        # : a cached snapshot is re-verified on first use per process
        # so pod-churn corruption can never be trusted forever.
        self._verified: set[str] = set()
        # Last digest-carrying snapshot seen per ref: companion-slot
        # setups may arrive snapshot-less; without memory of the hub's desired
        # state / RunJob snapshot they cannot materialize tensorhub refs. Stale
        # URLs self-heal: they fail url_expired and the hub re-mints.
        self._snapshots: Dict[WireRef, pb.Snapshot] = {}
        # Current generation attached to each banked snapshot. A generation-
        # less bank inherits only from the exact current desired identity
        # below; historical desired generations are never resurrected.
        self._snapshot_generations: Dict[WireRef, int] = {}
        # Current full-replacement desired identity per ref. This is bounded
        # by the active DesiredResidency set, not an unbounded digest history:
        # a priority RunJob may bank different bytes temporarily, while a
        # later generation-less bank of the still-desired digest recovers its
        # causal generation. Replacing desired state clears stale generations.
        self._desired_snapshot_identities: Dict[WireRef, _ResidencyIdentity] = {}
        # Identity of the bytes that ACTUALLY produced the current residency.
        # This deliberately does not follow _snapshots when a tag moves.
        self._resident_identities: Dict[str, _ResidencyIdentity] = {}
        # A newer snapshot may coexist on disk while the prior snapshot's
        # pipeline is still in RAM/VRAM. Keep the disk identity separately
        # until record teardown makes it the highest residency tier.
        self._disk_identities: Dict[str, _ResidencyIdentity] = {}
        # Every applied HelloAck opens a new
        # republish epoch. The reconcile pass re-announces verified cached
        # identities the hub re-asked about even when unchanged — observations
        # are content-addressed and idempotent hub-side, and a force-resent
        # plan is exactly the hub saying "tell me again" (redrive/overdue
        # resends could otherwise never heal a lost success observation).
        # Job-path ensure_local calls within the same epoch stay deduped.
        self._residency_republish_epoch = 0
        self._identity_publish_epochs: Dict[str, int] = {}
        self._identity_lock = threading.RLock()
        # Cold-ref waiters: ensure_local blocks here until the
        # hub's re-minted DOWNLOAD banks a snapshot for the ref.
        self._snapshot_waiters: Dict[str, asyncio.Event] = {}
        # network_bytes for the NEXT
        # ON_DISK transition of this ref, handed off to
        # _on_residency_event so the one authoritative wire event Residency
        # emits carries it — set immediately before track_disk(), consumed
        # (popped) by _on_residency_event if it fires, cleared defensively
        # otherwise. Avoids a second, redundant ON_DISK event and avoids
        # widening EventFn's arity (Residency has other direct callers).
        self._pending_network_bytes: Dict[str, int] = {}
        # generation bumps only when
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

    def _materialize_intent(self, ref: WireRef) -> str:
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
            # Swap telemetry: promote/demote wall time rides the
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
            identity = self.resident_identity(WireRef(ref))
        coro = self._event(WireRef(ref), pb_state, identity=identity, **kw)
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
        ref: WireRef,
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
        ref: WireRef,
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
        # The pinned-host fair share was DEAD code — the pool's
        # per-group cap only engages once it knows G, and nothing in src/ ever
        # told it. Without this a G=4 degraded pod lets group 0 claim the whole
        # pinned budget (§4.3 caveat 2).
        staging_mod.pinned_pool().set_group_count(int(topology.execution_groups))
        # registries were created lazily on first dispatch, so
        # the boot disk re-track (which unions over all_residencies()) was a
        # no-op for groups 1..G-1 — their LRU/preserve/eviction views started
        # blind to the disk tier that was already there. Create every group's
        # registry NOW, before any boot walk unions over them.
        for ordinal in range(int(topology.execution_groups)):
            self.residency_for(ordinal)

    def disk_ref_in_use(self, ref: WireRef) -> bool:
        """In-use across ALL groups (§4.3 caveat 3): one group's GC must never
        drop the pages another group is mmapping."""
        return any(reg.in_use(ref) for reg in self.all_residencies())

    def disk_local_path(self, ref: WireRef) -> Optional[Path]:
        for reg in self.all_residencies():
            path = reg.local_path(ref)
            if path is not None:
                return path
        return None

    def disk_refs(self) -> List[WireRef]:
        """Union of DISK-tier refs across groups.

        The tier lists hand back what normalized-ref callers wrote, so the
        cast restates a fact rather than making a new claim; `residency` is
        still keyed by plain `str` and is its own unit.
        """
        seen: Dict[WireRef, None] = {}
        for reg in self.all_residencies():
            for ref in reg.refs_in(residency_mod.Tier.DISK):
                seen.setdefault(WireRef(ref), None)
        return list(seen)

    # ---- residency facade ----------------------------------------------------

    def residency_snapshot(self) -> List[pb.ModelResidency]:
        out: List[pb.ModelResidency] = []
        # union across EVERY group's registry. This runs on
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
        """Cached measured per-tier disk telemetry.

        Never measures directly — returns whatever
        :meth:`refresh_disk_usage_report` last computed. ``_state_delta()``
        calls this synchronously from many places (some with no event loop
        at all, e.g. the initial ``build_hello()``); it must never touch a
        filesystem. Empty/zeroed until the first refresh completes (boot's
        first StateDelta may ship no tiers — informational telemetry, never
        a dispatch gate on its own)."""
        return self._cached_disk_usage_report

    def _reclaimable_entries(
        self, keep: Set[str], entries: Dict[str, Any],
    ) -> List[Tuple[str, int]]:
        """Materialized tree bytes the disk GC can certainly free.

        tensorfs owns content deduplication under ``objects/`` and materializes
        independent snapshot trees. Deleting an inactive tree therefore frees
        its indexed bytes even when another manifest references the same
        objects. Any newly unreachable objects are an additional conservative
        gain once tensorfs collects them; they are not counted here.
        """
        out: List[Tuple[str, int]] = []
        for ref in self.disk_refs():
            if ref in keep or self.disk_ref_in_use(ref):
                continue
            ent = entries.get(ref)
            if not ent:
                continue
            path = str(ent.get("path") or "")
            out.append((path, int(ent.get("bytes") or 0)))
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
        # Preserve decisions are unioned across groups: a sibling may still
        # mmap the same materialized tree even though one group is done.
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

    def local_path(self, ref: WireRef) -> Optional[Path]:
        # Union across groups (pgw#748): a group that has not yet booked this
        # ref must still see the tree a sibling group already materialized.
        return self.disk_local_path(ref)

    def has_snapshot(self, ref: WireRef) -> bool:
        """A digest-carrying snapshot for ``ref`` was seen this connection
: snapshot-less ops for it can still materialize the bytes."""
        return ref in self._snapshots

    def bank_snapshot(self, ref: WireRef, snapshot: pb.Snapshot) -> None:
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
        self, snapshots: Dict[WireRef, pb.Snapshot], *, generation: int,
    ) -> None:
        """Atomically replace desired snapshot identity and bank its metadata.

        DesiredResidency is full-replacement state. Keeping this map separate
        from the last RunJob bank lets priority requests use older bytes
        without erasing the generation of bytes that remain desired, while a
        removal cannot resurrect an obsolete generation later.
        """
        accepted_generation = max(0, int(generation))
        stored: Dict[WireRef, pb.Snapshot] = {}
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

    def _prune_banked_snapshots(self, desired: Dict[WireRef, pb.Snapshot]) -> None:
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

    def snapshot_digest(self, ref: WireRef, snapshot: Optional[pb.Snapshot] = None) -> str:
        candidate = snapshot
        if candidate is None:
            with self._identity_lock:
                candidate = self._snapshots.get(ref)
        return str(getattr(candidate, "digest", "") or "").strip()

    def resident_identity(self, ref: WireRef) -> _ResidencyIdentity:
        with self._identity_lock:
            return self._resident_identities.get(ref, ("", 0))

    def _snapshot_identity(
        self, ref: WireRef, snapshot: Optional[pb.Snapshot],
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
        self, ref: WireRef, identity: _ResidencyIdentity,
    ) -> bool:
        digest, generation = identity
        if not digest:
            return False
        exact = (str(digest).strip(), max(0, int(generation)))
        with self._identity_lock:
            changed = self._resident_identities.get(ref) != exact
            self._resident_identities[ref] = exact
        return changed

    def activate_disk_identity(self, ref: WireRef) -> _ResidencyIdentity:
        """Make the verified disk snapshot the identity of a newly loaded
        RAM/VRAM instance immediately before its residency transition."""
        with self._identity_lock:
            identity = self._disk_identities.get(ref, ("", 0))
            if identity[0]:
                self._resident_identities[ref] = identity
            return identity

    async def _confirm_cached_identity(
        self, ref: WireRef, identity: _ResidencyIdentity,
    ) -> None:
        """Publish exact identity when verified cached bytes satisfy the
        desired state without requiring a redundant download.

        the emission is content-addressed and
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

    def component_digests(self, ref: WireRef, local_path: Optional[Path] = None) -> Dict[str, str]:
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
            # This read `f.blake3`, which is EMPTY on every v2
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

    def component_sizes(self, ref: WireRef) -> Dict[str, int]:
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

        Also sweeps abandoned writer-unique CAS temp artifacts: on
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

    def lru_disk_refs(self, *, exclude: Tuple[str, ...] = ()) -> List[WireRef]:
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
    ) -> List[WireRef]:
        """The evictable SET for one gc_disk pass: hard invariants only
        (never exclude/in-use — no policy ever overrides these), plus this
        pass's keep-membership/grace filter. Ordering within that set is a
        separate seam, see ``_disk_eviction_order``."""
        now = time.time()
        out: List[Tuple[float, WireRef]] = []
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

    # The eviction POLICY (ranking one
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
        entries: List[Tuple[float, WireRef]], include_keep: bool,
        keep_rank: Dict[str, int],
    ) -> List[WireRef]:
        if include_keep:
            ordered = sorted(entries, key=lambda item: (-keep_rank[item[1]], item[0], item[1]))
        else:
            ordered = sorted(entries)
        return [ref for _, ref in ordered]

    _disk_eviction_order = _default_disk_eviction_order

    def _evict_disk_ref(self, ref: WireRef) -> None:
        path = self.residency.local_path(ref) or self._index.path(ref)
        if not self.residency.evict(ref):  # refuses in-use entries; emits EVICTED
            return
        if path is not None:
            # snapshot trees are keyed by DIGEST, so two refs that
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

    def _other_ref_at_path(self, ref: WireRef, path: Path) -> str:
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
        ref: WireRef,
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

    def _lock(self, ref: WireRef) -> asyncio.Lock:
        return self._locks.setdefault(ref, asyncio.Lock())

    def register_binding(self, ref: WireRef, binding: Any) -> None:
        """Endpoint-spec binding for ``ref`` — supplies files/provider on
        download paths that only carry the bare ref (DesiredResidency or
        startup prefetch), so ``files=`` selections apply everywhere (#377)."""
        self._bindings.setdefault(ref, binding)

    async def _await_hub_snapshot(
        self,
        ref: WireRef,
        *,
        intent_id: str = "",
    ) -> pb.Snapshot:
        """Cold tensorhub ref with no orchestrator-resolved snapshot: emit
        ``missing_snapshot`` (the hub refreshes desired state with fresh URLs
        on seeing it — connect_worker handleModelFailure) and block
        until that snapshot is banked. The bank site runs OUTSIDE
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
        ref: WireRef,
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
        ref: WireRef,
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
            # The bytes are pod-wide but each group keeps its own
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
            # download/verify — one pod, one copy, every group.
            cached = self.disk_local_path(ref)
            # A digest-carrying snapshot is authoritative: a cached
            # materialization of the SAME ref at a DIFFERENT digest is stale
            # (flavor re-published — e.g. compile-cache digest-change
            # re-adoption, e2e#117 live find #7) and must not short-circuit.
            want = ""
            if snapshot is not None and snapshot.digest:
                want = snapshot.digest.split(":", 1)[-1].strip().lower()
            # th#1941: the composed manifest digest IS the directory name, so
            # there is exactly one acceptable spelling per fetch identity.
            if cached is not None and cached.exists() and (not want or cached.name == want):
                if ref in self._verified:
                    self._index.touch(ref)
                    await self._confirm_cached_identity(ref, operation_identity)
                    return complete(cached)
                # First use this boot: verify before trusting. A
                # pod-churn-truncated snapshot used to fatal every load until
                # a manual delete; now it is quarantined + re-materialized.
                ok, bad = await asyncio.to_thread(
                    self._verify_snapshot_tree, cached, snapshot,
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
                    # . Report missing_snapshot — the hub's re-mint
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
            # Every byte figure below (headroom gate, DOWNLOADING totals, the
            # boot weights span) counts the resolved manifest's whole file
            # list — th#1941 made it exactly what gets fetched.
            fetch_files = list(snapshot.files) if snapshot is not None else []
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
            # opened before _progress so
            # its DOWNLOADING ticks can read the running total, and entered
            # once for the whole retry loop so it accumulates across
            # attempts. The hub (tensorhub th#850/PR#493) reads network_bytes
            # off the DOWNLOADING events' running value (mirrors
            # bytes_done/bytes_total), not just the terminal ON_DISK one —
            # both must carry it for the wire contract to actually work.
            net_scope = cozy_snapshot.NetworkBytesScope()

            # Per-ref bytes as a registry counter (visible on every
            # 10s beat while an activity is open); snapshot sizes make the
            # total known up front, so the wire never shows total=0 for
            # tensorhub refs.
            known_total = sum(int(f.size_bytes) for f in fetch_files)
            # owned by the activity this download is FOR when there
            # is one, so it advances that scope's clock and no other.
            dl_counter = activity_mod.scoped_counter(
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
            # THE weights-fetch boot span. It lives here, not at a
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
                        # name this span as the parent for the
                        # per-component rows opened inside the downloader.
                        # `open_span` cannot push the nesting stack itself
                        # (its close is in another frame), and a component row
                        # that lands top-level is counted twice.
                        with net_scope, (
                            boot_mod.parent_scope(fetch_span.ordinal)
                            if fetch_span is not None
                            else contextlib.nullcontext()
                        ):
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
                        # handed off to
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
                                # The generic bucket must carry the root
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
                        # every object was already in tensorfs (local CAS) or
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
        self, ref: WireRef, identity: _ResidencyIdentity,
    ) -> _ResidencyIdentity:
        """Promote the exact bytes used by one setup, never current disk state."""
        if identity[0]:
            self._set_resident_identity(ref, identity)
            return identity
        return self.activate_disk_identity(ref)

    # ---- snapshot integrity -------------------------------------------

    def _verify_snapshot_tree(
        self, path: Path, snapshot: Optional[pb.Snapshot]
    ) -> Tuple[bool, List[str]]:
        """Integrity of a snapshot tree (worker thread; blocking IO).

        With a resolved manifest every regular file holding real bytes is
        checked against its declared size AND its CONTENT DIGEST, hashed under
        the algorithm the manifest named; files the manifest cannot cover
        (reassembled chunked originals, merged single-file checkpoints) plus
        manifest-less trees (hf/civitai) get the structural safetensors check
        (header parses + every declared tensor byte present). Returns
        ``(ok, bad_digests)`` — the digests name blobs to quarantine.

        **A PROJECTED tree is verified structurally and is NEVER hashed here**
        (pgw#1308). Its files are pointer stubs and CAS symlinks that by
        construction do not hold the bytes their manifest entries name, so
        every byte-level check on them fails — and this function's caller
        deletes the tree AND its CAS blobs and re-downloads. On master that
        made a projected tree an infinite re-download: every boot, forever,
        presenting as "the CAS has stopped caching". The stub format's loud
        parse failure worked exactly as designed; it was read as corruption by
        a handler built to trust it. See :func:`volume_verify.verify_projection`
        for why structural is COMPLETE here rather than weaker."""

        p = Path(path)
        bad: List[str] = []
        covered: set[Path] = set()
        files = list(snapshot.files) if snapshot is not None else []
        if files and p.is_dir():
            # The hash algorithm comes from the DIGEST,
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
                # Projection artifacts are split off BEFORE hashing: a stub or
                # a CAS symlink does not hold the bytes its entry names, so
                # hashing it at its path is not a weaker check, it is a check
                # of the wrong thing that fails every time.
                projected, material = split_projection_targets(targets)
                rep = verify_files(material)
                if projected:
                    proj = verify_projection(projected)
                    rep.expected += proj.expected
                    rep.examined += proj.examined
                    rep.projected += proj.projected
                    rep.bad.extend(proj.bad)
                    rep.findings.extend(proj.findings)
                bad.extend(rep.bad)
                for finding in rep.findings:
                    logger.warning("snapshot %s: %s", p.name, finding)
                # DENOMINATOR GUARD, and it applies only to an otherwise-CLEAN
                # report: a verdict that found nothing wrong is trustworthy only
                # if it actually read the bytes. `examined` must cover every
                # target handed in, and a clean run that neither hashed a byte
                # nor checked a projection artifact read nothing at all. (A
                # report that already names bad files is not vacuous -- it did
                # its job, and folding it in here would double-report the same
                # digest.)
                vacuous = (
                    not rep.bad
                    and not rep.findings
                    and rep.hashed == 0
                    and rep.projected == 0
                )
                if rep.examined != rep.expected or vacuous:
                    logger.error(
                        "snapshot %s verification is not trustworthy: examined=%d "
                        "expected=%d hashed=%d projected=%d bytes=%d -- treating as corrupt",
                        p.name, rep.examined, rep.expected, rep.hashed,
                        rep.projected, rep.bytes_hashed,
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
            # A stub IS a valid projection artifact, not a truncated shard.
            # `safetensors_file_valid` correctly refuses it -- and concluding
            # "corrupt" from that correct refusal is the pgw#1308 defect.
            if projection.stub_at(st) is not None:
                continue
            if not safetensors_file_valid(st):
                logger.warning("snapshot file %s structurally invalid (truncated?)", st)
                bad.append(str(st.relative_to(p)) if st != p else st.name)
        return (not bad, bad)

    def _quarantine_snapshot(self, ref: WireRef, path: Path, bad: List[str]) -> None:
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
        self, ref: WireRef, snapshot: Optional[pb.Snapshot] = None, *, binding: Any = None
    ) -> Optional[Path]:
        """Load-failure path: a weights load failed with a
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
