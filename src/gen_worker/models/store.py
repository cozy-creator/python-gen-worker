"""The on-disk/CAS model store: resolve -> materialize -> reference-count."""

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
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, cast,
)

from .. import activity as activity_mod
from .. import boot_phases as boot_mod
from .. import progress as progress_mod
from .. import weight_position
from ..capability import InsufficientDiskError
from ..lifecycle_intents import IntentRegistry
from ..pb import worker_scheduler_pb2 as pb
from ..redact import sanitize as _sanitize
from ..topology import current_device_group
from ..wire_snapshots import resolved_repo_from_snapshot
from . import cozy_snapshot, disk_gc, disk_telemetry, fill_plan, projection
from .projection import SNAPSHOTS_DIR
from . import residency as residency_mod
from . import staging as staging_mod
from .cache_paths import tensorhub_cas_dir, tensorhub_fill_source_dir
from .config_identity import CANONICAL_JSON_MAX_BYTES, canonical_json_digest
from .cozy_snapshot import _norm_rel_path, delete_blobs, snapshot_dir_key
from .download import ensure_local, lookup_provider_for_ref
from .errors import MissingSnapshotError, UrlExpiredError
from .hub_client import WorkerResolvedRepo
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
_MISSING_SNAPSHOT_WAIT_S = 60.0

_DISK_GC_MARGIN_BYTES = 2 * _GiB
_DISK_GC_GRACE_S = 300.0

def _snapshot_to_resolved(snap: pb.Snapshot) -> "WorkerResolvedRepo":
    return cast("WorkerResolvedRepo", resolved_repo_from_snapshot(snap))

def _is_terminal_download_error(exc: BaseException) -> bool:
    if isinstance(exc, (UrlExpiredError, InsufficientDiskError, MissingSnapshotError)):
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
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

def _resident_position(ref: "WireRef", snapshot: Optional[pb.Snapshot]) -> None:
    total = (
        sum(int(f.size_bytes) for f in snapshot.files)
        if snapshot is not None else 0
    )
    weight_position.FetchPosition(ref, total_bytes=total).already_resident()

@dataclass(frozen=True)
class _MaterializedLocal:
    path: Path
    identity: _ResidencyIdentity

_ACTIVE_STORE: "Optional[ModelStore]" = None


def bind_active_store(store: "ModelStore") -> None:
    """Publish the process's store for paths that hold no reference to it."""
    global _ACTIVE_STORE
    _ACTIVE_STORE = store


def active_store() -> "Optional[ModelStore]":
    """The bound store, or None in a process that never made one (a fixture)."""
    return _ACTIVE_STORE


class ModelStore:
    """The worker's model seam: ensure-local with retries, the residency map, and disk retention (#370)."""

    def __init__(
        self,
        emit: Callable[[pb.WorkerMessage], Awaitable[None]],
        *,
        cache_dir: Optional[Path] = None,
        vram_budget_bytes: Optional[int] = None,
        disk_free_bytes_fn: Optional[Callable[[], int]] = None,
        fill_source_dir: Optional[Path] = None,
    ) -> None:
        self._emit = emit
        self._intent_registry: Optional[IntentRegistry] = None
        self._cache_dir = cache_dir or tensorhub_cas_dir()
        self._fill_source_dir = fill_source_dir or tensorhub_fill_source_dir()
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
        self._verified: set[str] = set()
        self._censused = False
        self._snapshots: Dict[WireRef, pb.Snapshot] = {}
        self._snapshot_generations: Dict[WireRef, int] = {}
        self._desired_snapshot_identities: Dict[WireRef, _ResidencyIdentity] = {}
        self._resident_identities: Dict[str, _ResidencyIdentity] = {}
        self._disk_identities: Dict[str, _ResidencyIdentity] = {}
        self._residency_republish_epoch = 0
        self._identity_publish_epochs: Dict[str, int] = {}
        self._identity_lock = threading.RLock()
        self._snapshot_waiters: Dict[str, asyncio.Event] = {}
        self._pending_network_bytes: Dict[str, int] = {}
        self._disk_report_lock = threading.Lock()
        self._disk_capacity_generation = 0
        self._last_disk_shape: Optional[bytes] = None
        self._cached_disk_usage_report = pb.DiskUsageReport()

    def _default_disk_free(self) -> int:
        p = Path(self._cache_dir)
        for candidate in (p, *p.parents):
            try:
                return int(shutil.disk_usage(candidate).free)
            except OSError:
                continue
        return 0

    def bind_loop(self) -> None:
        """Capture the running loop so residency events raised from worker threads (demote/promote via to_thread) still reach the wire."""
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
                logger.warning(
                    "model event DROPPED ref=%s state=%s: no event loop is "
                    "bound to this store yet (call bind_loop, or emit from "
                    "inside the loop)", ref, state,
                )
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
        """Build one identity-fenced model event."""
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

    @property
    def cache_dir(self) -> Path:
        """This pod's CAS root. Read-only; the store resolves it once at boot."""
        return Path(self._cache_dir)

    async def report_insufficient_disk(self, ref: WireRef, detail: str) -> None:
        """pgw#1612: tell the hub the SHAPE cannot fit, not that an attempt failed.

        `insufficient_disk` is the hub's existing model-failure reason with a
        whole migration/clear path behind it (drop the oldest resident non-hot
        disk goal, advance the capacity generation, clear the failures, re-send
        desired state). Reusing the token is the point: a second vocabulary
        would mean building that path twice.

        THE TOKEN IS SENT BARE, and that is a wire contract, not a style
        choice. `connect_worker.go:3737` reads `failureReason :=
        strings.TrimSpace(ev.GetError())` and then compares it for EXACT
        equality against `modelFailureReasonDisk` — so appending detail after a
        colon, the way `download_failed` does, would silently disable the whole
        migration path. Verified by reading the hub at HEAD, not from memory.

        The FACTS ride the activity stream instead, where they land in
        `worker_activity_events` and are readable off the wire: which mount ran
        out, its statvfs totals, and what the worker was doing. That is
        pgw#1620's lesson — a confession that only reaches a pod's stdout
        reaches nobody, because RunPod has no logs API.
        """
        try:
            activity_mod.emit_event(
                activity_mod.KIND_RESIDENCY_FAULT,
                f"{ref}: {_sanitize(detail)[:400]}",
                phase="insufficient_disk",
            )
        except Exception:  # noqa: BLE001 — reporting must not mask the failure
            logger.debug("insufficient_disk fact event dropped", exc_info=True)
        await self._event(ref, pb.MODEL_STATE_FAILED, error="insufficient_disk")

    @property
    def residency(self) -> Residency:
        """The registry for the execution group this task is serving."""
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
                device_group = residency_mod.DeviceGroup(devices=(g,))
            reg = Residency(
                on_event=self._on_residency_event,
                vram_budget_bytes=self._vram_budget_bytes,
                device_group=device_group,
            )
            reg.pre_demote = self._residency_by_group[0].pre_demote
            self._residency_by_group[g] = reg
            logger.info(
                "residency registry armed for group %d on devices %s",
                g, list(device_group.devices),
            )
            return reg

    def all_residencies(self) -> List[Residency]:
        """Every armed group registry."""
        return list(self._residency_by_group.values())

    def bind_topology(self, topology: Any) -> None:
        """Install the delivered `G×D` packing: one registry per group, each accounting for exactly its own devices."""
        self.residency_topology = topology
        if topology is None:
            return
        with self._residency_lock:
            for ordinal in range(int(topology.execution_groups)):
                self._residency_groups[ordinal] = topology.group(ordinal)
            zero = self._residency_by_group.get(0)
            if zero is not None:
                zero.device_group = self._residency_groups[0]
        staging_mod.pinned_pool().set_group_count(int(topology.execution_groups))
        for ordinal in range(int(topology.execution_groups)):
            self.residency_for(ordinal)

    def disk_ref_in_use(self, ref: WireRef) -> bool:
        """In-use across ALL groups (§4.3 caveat 3): one group's GC must never drop the pages another group is mmapping."""
        return any(reg.in_use(ref) for reg in self.all_residencies())

    def disk_local_path(self, ref: WireRef) -> Optional[Path]:
        for reg in self.all_residencies():
            path = reg.local_path(ref)
            if path is not None:
                return path
        return None

    def disk_refs(self) -> List[WireRef]:
        """Union of DISK-tier refs across groups."""
        seen: Dict[WireRef, None] = {}
        for reg in self.all_residencies():
            for ref in reg.refs_in(residency_mod.Tier.DISK):
                seen.setdefault(WireRef(ref), None)
        return list(seen)

    def residency_snapshot(self) -> List[pb.ModelResidency]:
        out: List[pb.ModelResidency] = []
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
        with self._identity_lock:
            for ref, (tier, vram) in merged.items():
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
        """Cached measured per-tier disk telemetry."""
        return self._cached_disk_usage_report

    def _reclaimable_entries(
        self, keep: Set[str], entries: Dict[str, Any],
    ) -> List[Tuple[str, int]]:
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
        keep = set(self.keep)
        entries = self._index.entries()
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
                tier=cast(Any, t.tier),
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
        """Off-loop refresh of the cached report (Lifecycle's TTL-gated refresh, driven off the heartbeat/state-delta path + once at boot)."""
        report = await asyncio.to_thread(self._measure_disk_usage_report)
        self._cached_disk_usage_report = report
        return report

    def local_path(self, ref: WireRef) -> Optional[Path]:
        return self.disk_local_path(ref)

    def has_snapshot(self, ref: WireRef) -> bool:
        """A digest-carrying snapshot for ``ref`` was seen this connection : snapshot-less ops for it can still materialize the bytes."""
        return ref in self._snapshots

    def banked_snapshot(self, ref: WireRef) -> Optional[pb.Snapshot]:
        """The snapshot identity this store holds for ``ref``, if any."""
        with self._identity_lock:
            return self._snapshots.get(ref)

    def banked_snapshot_for_tree(self, tree_name: str) -> Optional[pb.Snapshot]:
        """The banked snapshot whose DIGEST names ``tree_name``, if any."""
        want = str(tree_name)
        with self._identity_lock:
            banked = list(self._snapshots.values())
        for snapshot in banked:
            digest = str(getattr(snapshot, "digest", "") or "").strip()
            if not digest or not snapshot.files:
                continue
            bare = digest.split(":", 1)[-1]
            if want in (snapshot_dir_key(digest), snapshot_dir_key(bare)):
                return snapshot
        return None

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
        """Atomically replace desired snapshot identity and bank its metadata."""
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
        """Make the verified disk snapshot the identity of a newly loaded RAM/VRAM instance immediately before its residency transition."""
        with self._identity_lock:
            identity = self._disk_identities.get(ref, ("", 0))
            if identity[0]:
                self._resident_identities[ref] = identity
            return identity

    async def _confirm_cached_identity(
        self, ref: WireRef, identity: _ResidencyIdentity,
    ) -> None:
        tier = self.residency.tier(ref)
        digest, _ = identity
        if not digest:
            return
        with self._identity_lock:
            self._disk_identities[ref] = identity
            current = self._resident_identities.get(ref, ("", 0))
        if tier is None:
            return
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
        """Per-component content identity of ``ref``'s snapshot (gw#479): ``{top_level_subfolder: content_set_digest}``."""

        snap = self._snapshots.get(ref)
        if snap is None:
            return {}
        groups: Dict[str, Dict[str, str]] = {}
        for f in snap.files:
            rel = str(f.path).strip().lstrip("/")
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
        """Per-top-level-subfolder byte totals of ``ref``'s snapshot (gw#479): the make_room estimate for loading a subset of components."""
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

    def rescan_disk(self) -> None:
        """Boot-time truth: re-register still-present downloads from the persisted ref index so Hello.models and GC see what disk holds."""
        self._census_snapshot_pins()
        for ref, ent in self._index.entries().items():
            p = Path(str(ent.get("path") or ""))
            if p.exists():
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
        """Evict LRU disk-tier refs until free disk reaches the target."""
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
        if not self.residency.evict(ref):
            return
        if path is not None:
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
        target = str(path)
        for reg in self.all_residencies():
            for other in reg.refs_in(residency_mod.Tier.DISK):
                if other == ref:
                    continue
                other_path = reg.local_path(other)
                if other_path is not None and str(other_path) == target:
                    return other
        return ""

    def _mount_report(self) -> str:
        """This CAS root's mount and its real statvfs totals.

        pgw#1612: a refusal that does not name the mount is one the hub cannot
        act on and a lane has to re-derive weeks later. `disk_telemetry`
        already measures the real mount points; the refusal quotes it.
        """
        totals = disk_telemetry._statvfs_totals(str(self._cache_dir))
        if totals is None:
            return f"mount={self._cache_dir} statvfs=unreadable"
        total, free = totals
        return (
            f"mount={self._cache_dir} statvfs_total={total} "
            f"statvfs_free={free}"
        )

    async def _ensure_disk_headroom(
        self,
        ref: WireRef,
        plan: fill_plan.FillPlan,
        identity: _ResidencyIdentity = ("", 0),
        *,
        intent_id: str = "",
    ) -> None:
        """Gate on the fill's own PLAN. It cannot price anything else.

        pgw#1596 was a SHAPE bug, not an arithmetic one: this gate priced the
        REQUEST (the whole manifest) while the fill priced the DELTA (skip what
        `contains(digest, size)` already answers for). Two derivations of one
        quantity drifted and a 105 GB pull died 157 MB from the end on a disk
        that fit it fine — `need 104956706657 bytes; 65659441152 free after
        disk GC` on pod `6uneiwhdl7fz8u`, with ~86.2 GB of that same tree
        already resident.

        pgw#1631 promotes the fix from patch to construction. The argument is
        the plan the fill computed, so there is no manifest here to re-price
        and no second derivation to drift: `missing_bytes` is the arithmetic
        consequence of the same skip decision the fetch loop will make.

        NO GC DURING BOOT (pgw#1631). A boot that needs eviction to fit is a
        sizing bug upstream (th#2264), and evict-and-retry turns it into a slow
        expensive one — th#2246 burned two A100 pods and ~$1.72 on a shape that
        was doomed before it was paid for. At boot the gate refuses with a
        typed `insufficient_disk` naming the mount, its statvfs totals and the
        missing bytes, so the hub demotes the shape instead of re-buying it at
        the identical `container_disk_gb_requested`. Steady state keeps the LRU
        GC: there, eviction is what the tier is FOR.
        """
        remaining = plan.missing_bytes
        target = remaining + _DISK_GC_MARGIN_BYTES
        if self._disk_free() >= target:
            return

        booting = boot_mod.in_boot()
        if not booting:
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
            # The plan shows its working. The pre-pgw#1596 message named only
            # the whole tree, which is why an 86 GB-resident refusal read as a
            # 105 GB shortfall.
            after = "at boot (no GC: a boot that needs eviction to fit is a "
            after += "sizing bug upstream)" if booting else "after disk GC"
            raise InsufficientDiskError(
                f"need {remaining} more bytes for {ref} — {plan.describe()}; "
                f"{free} free {after}; {self._mount_report()}",
                available_bytes=free, required_bytes=remaining,
                path=str(self._cache_dir),
            )

    def _lock(self, ref: WireRef) -> asyncio.Lock:
        return self._locks.setdefault(ref, asyncio.Lock())

    def register_binding(self, ref: WireRef, binding: Any) -> None:
        """Endpoint-spec binding for ``ref`` — supplies files/provider on download paths that only carry the bare ref (DesiredResidency or startup prefetch), so ``files=`` selections apply everywhere (#377)."""
        self._bindings.setdefault(ref, binding)

    async def _await_hub_snapshot(
        self,
        ref: WireRef,
        *,
        intent_id: str = "",
    ) -> pb.Snapshot:
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

    def _census_snapshot_pins_once(self) -> None:
        if self._censused:
            return
        self._censused = True
        try:
            self._census_snapshot_pins()
        except Exception:  # noqa: BLE001 — a census never fails a residency answer
            logger.debug("snapshot census raised", exc_info=True)

    def _census_snapshot_pins(self) -> None:
        root = Path(self._cache_dir) / SNAPSHOTS_DIR
        unreadable_store = False
        try:
            trees = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            trees, unreadable_store = [], True
        if not trees and not unreadable_store:
            logger.info("snapshot census at boot: no trees on disk at %s", root)
        from . import projection as _projection

        unservable: List[str] = []
        packed: List[str] = []
        for tree in trees:
            try:
                pinned = _projection.resolve_projection(tree) is not None
                stubbed = _projection.stub_at_any(tree)
            except Exception as exc:  # noqa: BLE001 — a census never fails a boot
                logger.warning("snapshot census: %s unreadable: %s", tree.name, exc)
                continue
            if stubbed and not pinned:
                unservable.append(tree.name)
            short = tree.name.split(":")[-1][:12] or tree.name[:12]
            packed.append(f"{short}:{'P' if pinned else '-'}{'S' if stubbed else '-'}")
            (logger.error if (stubbed and not pinned) else logger.info)(
                "snapshot census at boot: %s pinned=%s projected=%s%s",
                tree.name, pinned, stubbed,
                "  <-- UNREADABLE by the streaming engine (pgw#1536)"
                if (stubbed and not pinned) else "",
            )

        try:
            from .. import activity as activity_mod

            shown = ",".join(packed[:40])
            more = "" if len(packed) <= 40 else f" (+{len(packed) - 40} more)"
            activity_mod.emit_event(
                activity_mod.KIND_SNAPSHOT_CENSUS,
                f"unservable={len(unservable)} of={len(trees)} "
                f"trees={shown}{more} store={root}",
                phase=(
                    "store_unreadable" if unreadable_store
                    else "unpinned_projected" if unservable
                    else "all_servable"
                ),
                step=len(unservable),
                total_steps=len(trees),
            )
        except Exception:  # noqa: BLE001 — telemetry never fails a boot
            logger.debug("snapshot census event dropped", exc_info=True)

    @staticmethod
    def _record_pin(path: Path, outcome: str) -> None:
        try:
            from . import projection as _projection

            _projection.record_pin_outcome(path.name, outcome)
        except Exception:  # noqa: BLE001 — telemetry never fails a load
            pass

    def ensure_pinned(
        self, ref: WireRef, tree: Path, snapshot: Optional[pb.Snapshot],
    ) -> bool:
        """Guarantee the tree's manifest pin exists."""
        path = Path(tree)
        if snapshot is None or not snapshot.files:
            snapshot = self.banked_snapshot(ref)
        if snapshot is None or not snapshot.files:
            logger.warning(
                "pin check SKIPPED for %s at %s: no digest-carrying snapshot "
                "is banked for this ref, so the manifest cannot be rebuilt. If "
                "this tree is projected it is unreadable by the streaming "
                "engine (pgw#1536)", ref, path,
            )
            self._record_pin(path, "no banked snapshot, manifest unrebuildable")
            return False
        try:
            from . import projection as _projection

            if _projection.resolve_projection(path) is not None:
                logger.debug("pin OK for %s at %s", ref, path)
                self._record_pin(path, "not needed: already pinned")
                return False
            if not _projection.stub_at_any(path):
                logger.debug(
                    "pin not required for %s at %s: tree is materialized, not "
                    "projected", ref, path,
                )
                self._record_pin(path, "not needed: tree is materialized")
                return False
        except Exception as exc:  # noqa: BLE001 — a probe must never fail a load
            logger.warning(
                "pin check FAILED to probe %s at %s: %s: %s (pgw#1536)",
                ref, path, type(exc).__name__, exc,
            )
            self._record_pin(path, f"probe failed: {type(exc).__name__}")
            return False
        try:
            from .cozy_snapshot import _manifest, _pin_manifest
            from .cache_paths import open_worker_cas

            resolved = resolved_repo_from_snapshot(snapshot)
            _pin_manifest(
                open_worker_cas(self._cache_dir), path.name, _manifest(resolved.files),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "could not repair the manifest pin for %s at %s: %s: %s — the "
                "streaming engine will decline this tree and the eager bridge "
                "will misreport it as a corrupt checkpoint (pgw#1526)",
                ref, path, type(exc).__name__, exc,
            )
            self._record_pin(path, f"ATTEMPTED and FAILED: {type(exc).__name__}")
            return False
        logger.warning(
            "REPAIRED the missing manifest pin `snapshot:%s` for %s at %s — the "
            "tree was materialized by a path that does not pin, which leaves it "
            "unreadable by the streaming engine. No bytes moved (pgw#1526)",
            path.name, ref, path,
        )
        self._record_pin(path, "REPAIRED, pin rewritten")
        return True

    async def announce_resident(
        self, ref: WireRef, snapshot: Optional[pb.Snapshot] = None,
    ) -> bool:
        self.bind_loop()
        self._census_snapshot_pins_once()
        if snapshot is None:
            snapshot = self._snapshots.get(ref)
        if snapshot is None or not snapshot.digest:
            return False
        digest = str(snapshot.digest).strip()
        bare = digest.split(":", 1)[-1]
        root = Path(self._cache_dir) / SNAPSHOTS_DIR
        tree: Optional[Path] = None
        for key in (snapshot_dir_key(digest), snapshot_dir_key(bare)):
            candidate = root / key
            if candidate.is_dir():
                tree = candidate
                break
        if tree is None:
            existing = self.disk_local_path(ref)
            if existing is None or not existing.is_dir():
                return False
            if existing.name not in (snapshot_dir_key(digest), snapshot_dir_key(bare)):
                return False
            tree = existing

        ok, bad = await asyncio.to_thread(self._verify_snapshot_tree, tree, snapshot)
        if not ok:
            logger.error(
                "residency REFUSED for %s at %s: the tree is present but does "
                "not match its manifest (%d bad file(s)) — quarantining and "
                "falling through to a fetch. This pod will NOT advertise these "
                "weights as resident; a tree that cannot be substantiated is "
                "not residency (pgw#1511)",
                ref, tree, len(bad),
            )
            await asyncio.to_thread(self._quarantine_snapshot, ref, tree, bad)
            return False

        from . import projection as _projection

        if _projection.stub_at_any(tree) and _projection.resolve_projection(tree) is None:
            if not self.ensure_pinned(ref, tree, snapshot):
                logger.error(
                    "residency REFUSED for %s at %s: the tree is intact but "
                    "its manifest pin is missing AND could not be repaired, so "
                    "the streaming engine cannot bind and the eager bridge "
                    "would read its pointer stubs as corrupt weights "
                    "(pgw#1526)",
                    ref, tree,
                )
                return False

        identity = self._snapshot_identity(ref, snapshot)
        with self._identity_lock:
            self._disk_identities[ref] = identity
        if self.residency.tier(ref) is None:
            self.residency.track_disk(ref, tree)
        else:
            await self._confirm_cached_identity(ref, identity)
        self._verified.add(ref)
        self._index.touch(ref)
        _resident_position(ref, snapshot)
        return True

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
            self.ensure_pinned(ref, path, snapshot)
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
            cached = self.disk_local_path(ref)
            want = ""
            if snapshot is not None and snapshot.digest:
                want = snapshot.digest.strip().lower()
            if cached is not None and cached.exists() and (
                not want or cached.name.lower() in (want, want.split(":", 1)[-1])
            ):
                if ref in self._verified:
                    self._index.touch(ref)
                    await self._confirm_cached_identity(ref, operation_identity)
                    _resident_position(ref, snapshot)
                    return complete(cached)
                ok, bad = await asyncio.to_thread(
                    self._verify_snapshot_tree, cached, snapshot,
                )
                if ok:
                    self._verified.add(ref)
                    self._index.touch(ref)
                    await self._confirm_cached_identity(ref, operation_identity)
                    _resident_position(ref, snapshot)
                    return complete(cached)
                logger.error(
                    "snapshot for %s failed first-use verification "
                    "(%d bad files); quarantining and re-materializing",
                    ref, len(bad),
                )
                await asyncio.to_thread(self._quarantine_snapshot, ref, cached, bad)
            if snapshot is None or not snapshot.digest:
                prov = (getattr(binding, "source", None)
                        or lookup_provider_for_ref(ref, default=""))
                if prov == "tensorhub":
                    failure_stage = pb.LIFECYCLE_INTENT_STAGE_WAIT_SNAPSHOT
                    snapshot = await self._await_hub_snapshot(
                        ref,
                        intent_id=intent_id,
                    )
                    operation_identity = self._snapshot_identity(ref, snapshot)
            fetch_files = list(snapshot.files) if snapshot is not None else []
            if snapshot is not None and snapshot.files:
                # pgw#1631: THE PLAN IS COMPUTED ONCE AND THE GATE IS HANDED IT.
                # The gate takes no manifest and no total, so it has nothing to
                # re-price — which is what makes a divergent precondition
                # unwritable rather than merely fixed (pgw#1596).
                failure_stage = pb.LIFECYCLE_INTENT_STAGE_WAIT_DISK_HEADROOM
                await self._ensure_disk_headroom(
                    ref,
                    await asyncio.to_thread(
                        fill_plan.plan_for_snapshot, self._cache_dir, fetch_files,
                    ),
                    operation_identity,
                    intent_id=intent_id,
                )
            last_progress = 0.0
            net_scope = cozy_snapshot.NetworkBytesScope()

            known_total = sum(int(f.size_bytes) for f in fetch_files)
            dl_counter = activity_mod.scoped_counter(
                f"download:{ref}", progress_mod.UNIT_BYTES, total=known_total)
            position = weight_position.FetchPosition(ref, total_bytes=known_total)

            resident_tier = self.residency.tier(ref)
            with self._identity_lock:
                banked_disk_identity = self._disk_identities.get(ref)
            already_resident = resident_tier is not None and (
                resident_tier in (residency_mod.Tier.RAM, residency_mod.Tier.VRAM)
                or (bool(operation_identity[0])
                    and banked_disk_identity is not None
                    and banked_disk_identity[0] == operation_identity[0])
            )
            download_open = False
            download_terminal = False

            def _open_download_record(done: int, total: int) -> None:
                nonlocal download_open
                download_open = True
                assert self._loop is not None
                asyncio.run_coroutine_threadsafe(
                    self._event(ref, pb.MODEL_STATE_DOWNLOADING,
                                identity=operation_identity,
                                bytes_done=int(done), bytes_total=int(total),
                                network_bytes=net_scope.network_bytes),
                    self._loop,
                )

            def _progress(done: int, total: Optional[int]) -> None:
                nonlocal last_progress
                position.progress(done, total)
                dl_counter.set_done(float(done))
                if total:
                    dl_counter.set_total(float(total))
                if not download_open:
                    if int(done) <= 0:
                        return
                    last_progress = time.monotonic()
                    _open_download_record(done, int(total or known_total))
                    return
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

            position.open()
            if already_resident:
                logger.info(
                    "model_download not opened for %s: already_resident "
                    "tier=%s digest=%s total_bytes=%d",
                    ref, getattr(resident_tier, "name", resident_tier),
                    operation_identity[0], known_total,
                )
            else:
                await self._event(
                    ref, pb.MODEL_STATE_DOWNLOADING, identity=operation_identity,
                    bytes_total=known_total,
                )
                download_open = True
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
                        self._pending_network_bytes[ref] = net_scope.network_bytes
                        self.residency.track_disk(ref, path)
                        self._pending_network_bytes.pop(ref, None)
                        if tier_before is None:
                            download_terminal = True
                        if tier_before is residency_mod.Tier.DISK and identity_changed:
                            await self._event(
                                ref, pb.MODEL_STATE_ON_DISK,
                                identity=operation_identity,
                                network_bytes=net_scope.network_bytes,
                            )
                            download_terminal = True
                        size = await asyncio.to_thread(disk_gc.tree_bytes, path)
                        fetch_bytes = int(size)
                        self._index.record(ref, path, size)
                        self._verified.add(ref)
                        return complete(path)
                    except Exception as exc:
                        terminal = _is_terminal_download_error(exc) or attempt >= _DOWNLOAD_RETRIES
                        if terminal:
                            vocab = self._error_vocab(exc)
                            if vocab == "download_failed":
                                vocab = f"download_failed: {_sanitize(f'{type(exc).__name__}: {exc}')[:200]}"
                            await self._event(
                                ref, pb.MODEL_STATE_FAILED,
                                identity=operation_identity, error=vocab,
                            )
                            download_terminal = True
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
                position.close(
                    ok=fetch_exc is None,
                    resident=already_resident and not download_open,
                )
                if download_open and not download_terminal:
                    download_terminal = True
                    if fetch_exc is None:
                        terminal_event = self._event(
                            ref, pb.MODEL_STATE_ON_DISK,
                            identity=operation_identity,
                            network_bytes=net_scope.network_bytes,
                        )
                    else:
                        terminal_event = self._event(
                            ref, pb.MODEL_STATE_FAILED,
                            identity=operation_identity,
                            error=(
                                "download_canceled"
                                if isinstance(fetch_exc, asyncio.CancelledError)
                                else self._error_vocab(fetch_exc)
                            ),
                        )
                    if isinstance(fetch_exc, BaseException) and not isinstance(
                        fetch_exc, Exception
                    ):
                        asyncio.ensure_future(terminal_event)
                    else:
                        await terminal_event
                if fetch_span is not None:
                    net = int(net_scope.network_bytes)
                    if net > 0:
                        source = boot_mod.SOURCE_R2
                    elif known_total > 0:
                        source = (
                            boot_mod.SOURCE_VOLUME if self._fill_source_dir
                            else boot_mod.SOURCE_LOCAL
                        )
                    else:
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

    def _verify_snapshot_tree(
        self, path: Path, snapshot: Optional[pb.Snapshot]
    ) -> Tuple[bool, List[str]]:

        p = Path(path)
        bad: List[str] = []
        covered: set[Path] = set()
        files = list(snapshot.files) if snapshot is not None else []
        if files and p.is_dir():
            targets, skipped = snapshot_verify_targets(files, p)
            for rel in skipped:
                try:
                    covered.discard(p / _norm_rel_path(rel))
                except ValueError:
                    pass
            for t in targets:
                covered.add(t.path)
            if targets:
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
            if projection.stub_at(st) is not None:
                continue
            if not safetensors_file_valid(st):
                logger.warning("snapshot file %s structurally invalid (truncated?)", st)
                bad.append(str(st.relative_to(p)) if st != p else st.name)
        return (not bad, bad)

    def _quarantine_snapshot(self, ref: WireRef, path: Path, bad: List[str]) -> None:

        self._verified.discard(ref)
        self.residency.evict(ref, force=True)
        disk_gc.delete_ref_bytes(ref, Path(path), self._cache_dir)
        delete_blobs(self._cache_dir, [d for d in bad if "/" not in d and "." not in d])
        disk_gc.sweep_orphan_blobs(self._cache_dir)
        self._index.remove(ref)

    async def refetch_corrupt(
        self, ref: WireRef, snapshot: Optional[pb.Snapshot] = None, *, binding: Any = None
    ) -> Optional[Path]:
        """Load-failure path: a weights load failed with a corruption-shaped error — digest-verify the snapshot."""
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
