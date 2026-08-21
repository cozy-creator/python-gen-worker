"""Model residency: LRU VRAM/RAM/disk tiers + shared-component cache."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .. import activity as activity_mod
from .memory import (
    device_mismatches,
    effective_ram_floor_gb,
    estimate_cuda_resident_gb,
    estimate_pipeline_size_gb,
    flush_memory,
    get_available_ram_gb,
    get_total_ram_gb,
    log_ram_budget_once,
    repair_device_placement,
)
from .pinned_swap import swap_object
from .pinned_swap import cached_swap_bytes
from .. import hostfacts

logger = logging.getLogger(__name__)

_GiB = 1024 ** 3
_VRAM_MARGIN_BYTES = 2 * _GiB
def _effective_ram_floor_gb() -> float:
    return effective_ram_floor_gb(get_total_ram_gb())

ON_DISK = "on_disk"
IN_RAM = "in_ram"
IN_VRAM = "in_vram"
EVICTED = "evicted"

EventFn = Callable[[str, str, int, int], None]


class Tier(str, Enum):
    VRAM = "VRAM"
    RAM = "RAM"
    DISK = "DISK"


@dataclass(frozen=True)
class HostRamHeadroom:
    """One observed host-RAM admission decision."""

    available_bytes: int
    floor_bytes: int
    required_bytes: int
    total_bytes: int = 0

    @property
    def sufficient(self) -> bool:
        return self.available_bytes >= self.required_bytes

    @property
    def structural(self) -> bool:
        """The requirement exceeds the whole host — no eviction can help, and no identically-sized pod ever will either."""
        return self.total_bytes > 0 and self.required_bytes > self.total_bytes


REPLICATED = "replicated"
SHARDED = "sharded"
_PLACEMENT_MODES = (REPLICATED, SHARDED)


@dataclass(frozen=True)
class DeviceGroup:
    """The unit of placement (see WORKER-RESIDENCY-DESIGN "Multi-GPU")."""

    devices: Tuple[int, ...] = (0,)
    placement_mode: str = REPLICATED

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("DeviceGroup needs at least one device")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f"DeviceGroup devices must be unique: {self.devices}")
        if any(int(d) < 0 for d in self.devices):
            raise ValueError(f"DeviceGroup devices must be >= 0: {self.devices}")
        if self.placement_mode not in _PLACEMENT_MODES:
            raise ValueError(
                f"DeviceGroup placement_mode must be one of {_PLACEMENT_MODES}: "
                f"{self.placement_mode!r}"
            )

    @property
    def primary(self) -> int:
        return self.devices[0]

    @property
    def replicated(self) -> bool:
        return self.placement_mode == REPLICATED

    def _per_device_free_bytes(self) -> List[int]:
        count = hostfacts.device_count()
        if not count:
            return []
        out: List[int] = []
        for d in self.devices:
            free = hostfacts.free_vram_bytes(d) if 0 <= int(d) < count else None
            out.append(int(free or 0))
        return out

    def free_vram_bytes(self) -> int:
        """Free VRAM budget for THIS group under its placement mode: the MIN across members when replicated, the sum when sharded."""
        try:
            per_device = self._per_device_free_bytes()
            if not per_device:
                return 0
            return sum(per_device) if not self.replicated else min(per_device)
        except Exception:
            return 0


@dataclass
class _Entry:
    ref: str
    tier: Tier
    path: Optional[Path] = None
    obj: Any = None
    vram_bytes: int = 0
    vram_hint: int = 0
    pinned: bool = False
    refcount: int = 0
    holders: int = 0
    last_used: float = field(default_factory=time.monotonic)
    promote_count: int = 0
    demote_count: int = 0
    last_promote_ms: int = 0
    last_demote_ms: int = 0

    @property
    def movable(self) -> bool:
        """True when the registry can actually move this object between devices (``.to()``); offload-hooked pipelines own their placement."""
        return (
            self.obj is not None
            and callable(getattr(self.obj, "to", None))
            and not _obj_manages_own_device(self.obj)
        )


def _default_free_vram_bytes(group: Optional[DeviceGroup] = None) -> int:
    return (group or DeviceGroup()).free_vram_bytes()


def _obj_manages_own_device(obj: Any) -> bool:
    return getattr(obj, "_cozy_low_vram_mode", None) in (
        "partial_resident", "model_offload", "group_offload", "sequential",
    )


def _obj_offload_hooked(obj: Any) -> bool:
    if _obj_manages_own_device(obj):
        return True
    try:
        from .stream_residency import stream_residency_active

        return bool(stream_residency_active(obj))
    except Exception:
        return False


def _move_obj(obj: Any, device: str) -> None:
    if obj is None or _obj_manages_own_device(obj):
        return

    if swap_object(obj, device):
        return
    to = getattr(obj, "to", None)
    if callable(to):
        to(device)


def move_verified(
    obj: Any,
    device: str,
    *,
    label: str = "",
    move_fn: Callable[[Any, str], None] = _move_obj,
) -> bool:
    """Move ``obj`` to ``device`` and PROVE it landed."""
    name = label or type(obj).__name__
    restore = "cpu" if device != "cpu" else "cuda"
    try:
        move_fn(obj, device)
        missed = device_mismatches(obj, device)
        if missed:
            logger.warning(
                "residency: .to(%s) on %s left %d tensors behind (e.g. %s); repairing",
                device, name, len(missed), missed[:3],
            )
            missed = repair_device_placement(obj, device)
        if not missed:
            return True
        logger.error(
            "residency: move of %s to %s incomplete after repair (%s); rolling back",
            name, device, missed[:5],
        )
    except Exception as exc:
        logger.error(
            "residency: .to(%s) failed for %s: %s; rolling back", device, name, exc,
        )
    try:
        move_fn(obj, restore)
        left = repair_device_placement(obj, restore)
        if left:
            logger.critical(
                "residency: rollback of %s to %s ALSO incomplete (%s) — "
                "object is mixed-device and unusable",
                name, restore, left[:5],
            )
            activity_mod.emit_event(
                activity_mod.KIND_RESIDENCY_FAULT,
                f"ref={name} wanted={device} "
                f"restore={restore}: move AND rollback both incomplete "
                f"(e.g. {left[:3]}); object is mixed-device and unusable",
                phase="mixed_device_unusable",
            )
    except Exception as exc:
        logger.exception("residency: rollback .to(%s) failed for %s", restore, name)
        activity_mod.emit_event(
            activity_mod.KIND_RESIDENCY_FAULT,
            f"ref={name} wanted={device} "
            f"restore={restore}: rollback move raised; object is likely "
            f"mixed-device and unusable: {type(exc).__name__}: {exc}",
            phase="mixed_device_unusable",
        )
    flush_memory()
    return False


class Lease:
    """Admission lease: the set of refs one job needs, taken BEFORE the job starts and held for its whole lifetime."""

    __slots__ = ("_registry", "refs", "activation", "_released")

    def __init__(
        self, registry: "Residency", refs: Tuple[str, ...], activation: int = 0,
    ) -> None:
        self._registry = registry
        self.refs = refs
        self.activation = max(0, int(activation))
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._registry._release_lease(self)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


class Residency:
    """LRU-tiered model registry."""

    def __init__(
        self,
        *,
        on_event: Optional[EventFn] = None,
        vram_budget_bytes: Optional[int] = None,
        free_vram_bytes_fn: Optional[Callable[[], int]] = None,
        move_fn: Callable[[Any, str], None] = _move_obj,
        device_group: Optional[DeviceGroup] = None,
    ) -> None:
        self._on_event = on_event
        self._vram_budget = vram_budget_bytes
        self._free_vram_fn = free_vram_bytes_fn
        self._move = move_fn
        self.device_group = device_group or DeviceGroup()
        self.pre_demote: Optional[Callable[[str, Any], None]] = None
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.RLock()
        self._shared_hits = 0
        self._shared_misses = 0
        self._leases: Dict[int, Lease] = {}
        self._ref_reservations: Dict[str, Dict[int, int]] = {}
        self._activation: Dict[str, int] = {}
        log_ram_budget_once(floor_gb=_effective_ram_floor_gb())

    def _emit(self, ref: str, state: str, vram_bytes: int = 0, duration_ms: int = 0) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(ref, state, int(vram_bytes), int(duration_ms))
        except Exception as exc:
            logger.exception("residency event callback failed for %s", ref)
            activity_mod.emit_event(
                activity_mod.KIND_RESIDENCY_FAULT,
                f"ref={ref} state={state}: residency event callback failed "
                f"(hub residency view may be stale): "
                f"{type(exc).__name__}: {exc}",
                phase="event_callback_failed",
            )

    def free_vram_bytes(self) -> int:
        if self._vram_budget is not None:
            with self._lock:
                used = sum(e.vram_bytes for e in self._entries.values() if e.tier is Tier.VRAM)
            return max(0, int(self._vram_budget) - used)
        if self._free_vram_fn is not None:
            return int(self._free_vram_fn())
        return _default_free_vram_bytes(self.device_group)

    def host_ram_headroom(self, needed_bytes: int) -> HostRamHeadroom:
        """Observed capacity for one incoming host-staged model load."""
        floor = int(_effective_ram_floor_gb() * _GiB)
        return HostRamHeadroom(
            available_bytes=int(get_available_ram_gb() * _GiB),
            floor_bytes=floor,
            required_bytes=max(0, int(needed_bytes)) + floor,
            total_bytes=int(get_total_ram_gb() * _GiB),
        )

    def tier(self, ref: str) -> Optional[Tier]:
        with self._lock:
            e = self._entries.get(ref)
            return e.tier if e else None

    def local_path(self, ref: str) -> Optional[Path]:
        with self._lock:
            e = self._entries.get(ref)
            return e.path if e else None

    def obj(self, ref: str) -> Any:
        with self._lock:
            e = self._entries.get(ref)
            return e.obj if e else None

    def replace_object(self, ref: str, obj: Any) -> bool:
        """Replace a resident ref's bookkeeping object without a state event."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None:
                return False
            e.obj = obj
            return True

    def vram_bytes(self, ref: str) -> int:
        with self._lock:
            e = self._entries.get(ref)
            return e.vram_bytes if e else 0

    def vram_hint(self, ref: str) -> int:
        """Last measured VRAM footprint (survives demotion) — the load-size estimate for make_room before a re-load/promotion."""
        with self._lock:
            e = self._entries.get(ref)
            return e.vram_hint if e else 0

    def movable(self, ref: str) -> bool:
        with self._lock:
            e = self._entries.get(ref)
            return bool(e and e.movable)

    def refs_in(self, tier: Tier) -> List[str]:
        with self._lock:
            return [r for r, e in self._entries.items() if e.tier is tier]

    def snapshot(self) -> List[Tuple[str, Tier, int]]:
        """(ref, tier, vram_bytes) for every tracked entry (Hello.models)."""
        with self._lock:
            return [(r, e.tier, e.vram_bytes) for r, e in self._entries.items()]

    def touch(self, ref: str) -> None:
        with self._lock:
            e = self._entries.get(ref)
            if e:
                e.last_used = time.monotonic()

    def track_disk(self, ref: str, path: Path) -> None:
        """Register an on-disk snapshot."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None:
                self._entries[ref] = _Entry(ref=ref, tier=Tier.DISK, path=Path(path))
            else:
                e.path = Path(path)
                e.last_used = time.monotonic()
                return
        self._emit(ref, ON_DISK)

    def track_ram(self, ref: str, obj: Any = None, *, path: Optional[Path] = None) -> None:
        """Register a loaded-but-not-VRAM object (CPU-only hosts, warm tier)."""
        with self._lock:
            e = self._entries.setdefault(ref, _Entry(ref=ref, tier=Tier.RAM))
            e.tier = Tier.RAM
            if obj is not None:
                e.obj = obj
            if path is not None:
                e.path = Path(path)
            e.vram_bytes = 0
            e.last_used = time.monotonic()
        self._emit(ref, IN_RAM)

    def track_vram(
        self,
        ref: str,
        obj: Any = None,
        *,
        vram_bytes: int = 0,
        path: Optional[Path] = None,
        pinned: bool = False,
    ) -> None:
        """Register a VRAM-resident object with its MEASURED footprint (``torch.cuda.memory_allocated`` delta across the load, or :func:`~gen_worker.models.memory.estimate_cuda_resident_gb`)."""
        if obj is not None and _obj_offload_hooked(obj):
            logger.info(
                "residency: %s is offload-hooked; booking RAM tier "
                "(VRAM unmeasurable under offload hooks)", ref,
            )
            hint = int(estimate_pipeline_size_gb(obj) * _GiB)
            with self._lock:
                e = self._entries.setdefault(ref, _Entry(ref=ref, tier=Tier.RAM))
                e.tier = Tier.RAM
                e.obj = obj
                if path is not None:
                    e.path = Path(path)
                e.vram_bytes = 0
                e.vram_hint = max(e.vram_hint, hint)
                if pinned:
                    e.pinned = True
                e.last_used = time.monotonic()
            self._emit(ref, IN_RAM)
            return
        measured = int(vram_bytes)
        if measured <= 0 and obj is not None:
            measured = int(estimate_cuda_resident_gb(obj) * _GiB)
        with self._lock:
            e = self._entries.setdefault(ref, _Entry(ref=ref, tier=Tier.VRAM))
            e.tier = Tier.VRAM
            if obj is not None:
                e.obj = obj
            if path is not None:
                e.path = Path(path)
            e.vram_bytes = max(0, measured)
            e.vram_hint = max(e.vram_hint, e.vram_bytes)
            if pinned:
                e.pinned = True
            e.last_used = time.monotonic()
            self._consume_reservation_locked(ref)
        self._emit(ref, IN_VRAM, max(0, measured))

    def demote(self, ref: str) -> bool:
        """VRAM -> RAM warm tier."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None or e.tier is not Tier.VRAM or e.pinned or e.refcount > 0:
                return False
            if self._leased_locked(ref):
                return False
            if not e.movable:
                return False
            need_gb = float(e.vram_hint or e.vram_bytes) / _GiB
            if need_gb <= 0.0:
                need_gb = estimate_pipeline_size_gb(e.obj)
            need_gb = max(0.0, need_gb - cached_swap_bytes(e.obj) / _GiB)
            if get_available_ram_gb() - need_gb < _effective_ram_floor_gb():
                logger.info(
                    "residency: refusing VRAM->RAM demote of %s (~%.1fGiB into "
                    "%.1fGiB available; floor %.1fGiB)",
                    ref, need_gb, get_available_ram_gb(), _effective_ram_floor_gb(),
                )
                return False
            t0 = time.monotonic()
            if self.pre_demote is not None:
                try:
                    self.pre_demote(ref, e.obj)
                except Exception:
                    logger.exception("pre_demote hook failed for %s", ref)
            if not self._move_verified(e.obj, "cpu", ref=ref):
                return False
            e.tier = Tier.RAM
            e.vram_bytes = 0
            e.demote_count += 1
            e.last_demote_ms = int((time.monotonic() - t0) * 1000)
            duration_ms = e.last_demote_ms
        flush_memory()
        self._emit(ref, IN_RAM, duration_ms=duration_ms)
        return True

    def _move_verified(self, obj: Any, device: str, *, ref: str = "") -> bool:
        return move_verified(obj, device, label=ref, move_fn=self._move)

    @property
    def vram_device(self) -> str:
        """Where THIS registry's promotions land: ``cuda`` (thread-current) for the default group, an explicit ``cuda:N`` once a topology says the group owns card N — a group-1 instance must never load onto c..."""
        primary = int(self.device_group.primary)
        return "cuda" if primary == 0 else f"cuda:{primary}"

    def promote(self, ref: str, device: str = "") -> bool:
        """RAM -> VRAM (makes room first)."""
        device = device or self.vram_device
        with self._lock:
            e = self._entries.get(ref)
            if e is None or not e.movable:
                return False
            hint = e.vram_hint
            obj = e.obj
            already_vram = e.tier is Tier.VRAM
            if already_vram:
                e.last_used = time.monotonic()
        if already_vram:
            missed = device_mismatches(obj, device)
            if not missed:
                return True
            logger.warning(
                "residency: VRAM-tier %s holds %d off-device tensors (e.g. %s); repairing",
                ref, len(missed), missed[:3],
            )
            if not repair_device_placement(obj, device):
                return True
            with self._lock:
                e = self._entries.get(ref)
                if e is None:
                    return False
                evicted = self._move_verified(e.obj, "cpu", ref=ref)
                if not evicted:
                    logger.critical(
                        "residency: %s could not be evicted to CPU; it stays "
                        "booked in VRAM (%d bytes) and is refused for serving",
                        ref, e.vram_bytes)
                    activity_mod.emit_event(
                        activity_mod.KIND_RESIDENCY_FAULT,
                        f"ref={ref}: a mixed-device VRAM entry could not be "
                        f"repaired OR evicted to CPU. It stays on the card and "
                        f"stays BOOKED at {e.vram_bytes} bytes — booking it as "
                        f"RAM/0 would hand `make_room` headroom that does not "
                        f"exist and OOM an unrelated promote later. This ref is "
                        f"refused for serving until it is reloaded",
                        phase="eviction_failed_still_resident",
                    )
                    return False
                e.tier = Tier.RAM
                e.vram_bytes = 0
            flush_memory()
            self._emit(ref, IN_RAM)
            return False
        if hint <= 0:
            hint = int(estimate_pipeline_size_gb(obj) * _GiB)
        t0 = time.monotonic()
        if not self.make_room(hint, for_refs=(ref,)):
            need = int(estimate_pipeline_size_gb(obj) * _GiB)
            if need > 0 and self.free_vram_bytes() < need:
                logger.info(
                    "residency: promote of %s refused (weights %.1fGiB, free "
                    "%.1fGiB after make_room)",
                    ref, need / _GiB, self.free_vram_bytes() / _GiB,
                )
                return False
        with self._lock:
            e = self._entries.get(ref)
            if e is None or not e.movable:
                return False
            if not self._move_verified(e.obj, device, ref=ref):
                return False
            e.tier = Tier.VRAM
            e.vram_bytes = int(estimate_cuda_resident_gb(e.obj) * _GiB) or hint
            e.vram_hint = max(e.vram_hint, e.vram_bytes)
            e.last_used = time.monotonic()
            e.promote_count += 1
            e.last_promote_ms = int((time.monotonic() - t0) * 1000)
            measured = e.vram_bytes
            duration_ms = e.last_promote_ms
        self._emit(ref, IN_VRAM, measured, duration_ms=duration_ms)
        return True

    def release_to_disk(self, ref: str) -> bool:
        """Drop the loaded object entirely; disk snapshot kept."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None:
                return False
            if e.refcount > 0 or e.holders > 0 or self._leased_locked(ref):
                return False
            was_loaded = e.tier in (Tier.VRAM, Tier.RAM)
            e.obj = None
            e.vram_bytes = 0
            if e.path is not None:
                e.tier = Tier.DISK
                state = ON_DISK
            else:
                del self._entries[ref]
                state = EVICTED
        if was_loaded:
            flush_memory()
        self._emit(ref, state)
        return True

    def evict(self, ref: str, *, force: bool = False) -> bool:
        """Remove the entry entirely (fully gone -> EVICTED)."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None:
                return False
            if (e.refcount > 0 or e.holders > 0 or self._leased_locked(ref)) and not force:
                return False
            was_loaded = e.tier in (Tier.VRAM, Tier.RAM)
            del self._entries[ref]
        if was_loaded:
            flush_memory()
        self._emit(ref, EVICTED)
        return True

    @contextmanager
    def executing(self, *refs: str) -> Iterator[None]:
        """Pin-while-executing: entries named here are not eviction candidates for the duration (cross-pipeline eviction never yanks a model that a handler is actively using)."""
        with self._lock:
            for ref in refs:
                e = self._entries.get(ref)
                if e:
                    e.refcount += 1
                    e.last_used = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                for ref in refs:
                    e = self._entries.get(ref)
                    if e and e.refcount > 0:
                        e.refcount -= 1

    def in_use(self, ref: str) -> bool:
        with self._lock:
            e = self._entries.get(ref)
            return bool(e and e.refcount > 0) or self._leased_locked(ref)

    def _leased_locked(self, ref: str) -> bool:
        return bool(self._ref_leases.get(ref))

    @property
    def _ref_leases(self) -> Dict[str, set]:
        view: Dict[str, set] = {}
        for lid, lease in self._leases.items():
            for ref in lease.refs:
                view.setdefault(ref, set()).add(lid)
        return view

    def _outstanding_reserved_bytes(self, exclude_refs: frozenset = frozenset()) -> int:
        total = 0
        for ref, by_lease in self._ref_reservations.items():
            if ref in exclude_refs or not by_lease:
                continue
            total += max(by_lease.values())
        return total

    def _outstanding_activation_bytes(self, exclude_lease_id: int = 0) -> int:
        return sum(
            lease.activation for lid, lease in self._leases.items()
            if lid != exclude_lease_id
        )

    def record_activation(self, key: str, observed_bytes: int) -> int:
        """Learn one request's measured transient VRAM (peak allocated minus what was already allocated when the handler took the GPU)."""
        key = str(key or "")
        if not key:
            return 0
        observed = max(0, int(observed_bytes))
        with self._lock:
            prev = self._activation.get(key, 0)
            hint = max(observed, prev - prev // 8)
            self._activation[key] = hint
            return hint

    def activation_hint(self, key: str) -> int:
        """Learned transient VRAM for ``key``; 0 until it has been measured."""
        with self._lock:
            return self._activation.get(str(key or ""), 0)

    def admit(self, sizes: Mapping[str, int], *, activation_bytes: int = 0) -> Lease:
        """Take an admission lease over one job's refs."""
        with self._lock:
            lease = Lease(
                self,
                tuple(dict.fromkeys(str(r) for r in sizes if str(r))),
                activation=activation_bytes,
            )
            lid = id(lease)
            self._leases[lid] = lease
            for ref in lease.refs:
                expect = max(0, int(sizes.get(ref, 0) or 0))
                e = self._entries.get(ref)
                if e is not None and e.tier is Tier.VRAM:
                    continue
                if e is not None and expect <= 0:
                    expect = e.vram_hint
                if expect > 0:
                    self._ref_reservations.setdefault(ref, {})[lid] = expect
                if e is not None:
                    e.last_used = time.monotonic()
            return lease

    def _release_lease(self, lease: Lease) -> None:
        with self._lock:
            lid = id(lease)
            self._leases.pop(lid, None)
            for ref in lease.refs:
                by_lease = self._ref_reservations.get(ref)
                if by_lease is not None:
                    by_lease.pop(lid, None)
                    if not by_lease:
                        del self._ref_reservations[ref]

    def _consume_reservation_locked(self, ref: str) -> None:
        self._ref_reservations.pop(ref, None)

    def fits(self, sizes: Mapping[str, int], *, activation_bytes: int = 0) -> bool:
        """Cheap honest admission query: could this ref set be served now — counting measured free VRAM, minus other jobs' outstanding weight and activation claims, plus what LRU demotion of unprotected entri..."""
        with self._lock:
            needed = max(0, int(activation_bytes))
            for ref, expect in sizes.items():
                e = self._entries.get(str(ref))
                if e is not None and e.tier is Tier.VRAM:
                    continue
                size = max(0, int(expect or 0))
                if size <= 0 and e is not None:
                    size = e.vram_hint
                needed += size
            if needed <= 0:
                return True
            reserved = self._outstanding_reserved_bytes(
                exclude_refs=frozenset(str(r) for r in sizes))
            reserved += self._outstanding_activation_bytes()
            ref_leases = self._ref_leases
            reclaimable = sum(
                e.vram_bytes for e in self._entries.values()
                if e.tier is Tier.VRAM and e.movable and not e.pinned
                and e.refcount <= 0 and not ref_leases.get(e.ref)
            )
        available = self.free_vram_bytes() - reserved + reclaimable
        return needed + _VRAM_MARGIN_BYTES <= available

    def leased_refs(self) -> List[str]:
        with self._lock:
            return sorted(self._ref_leases)

    def lru_vram_victims(self) -> List[str]:
        """Evictable VRAM refs, LRU first (pinned/executing excluded)."""
        with self._lock:
            ref_leases = self._ref_leases
            candidates = [
                e for e in self._entries.values()
                if e.tier is Tier.VRAM and not e.pinned and e.refcount <= 0
                and not ref_leases.get(e.ref)
            ]
            candidates.sort(key=lambda e: (1 if e.holders >= 2 else 0, e.last_used))
            return [e.ref for e in candidates]

    def make_room(
        self, needed_bytes: int, *, for_refs: Iterable[str] = (),
    ) -> bool:
        """Demote LRU VRAM entries until measured free VRAM covers ``needed_bytes`` + margin."""
        exclude = frozenset(str(r) for r in for_refs)

        def _headroom() -> int:
            with self._lock:
                reserved = self._outstanding_reserved_bytes(exclude_refs=exclude)
                reserved += sum(
                    lease.activation for lease in self._leases.values()
                    if not (exclude and exclude.intersection(lease.refs))
                )
            return self.free_vram_bytes() - reserved

        target = int(needed_bytes) + _VRAM_MARGIN_BYTES
        if _headroom() >= target:
            return True
        for ref in self.lru_vram_victims():
            if not self.demote(ref):
                continue
            logger.info("residency: demoted LRU %s for %d bytes headroom", ref, needed_bytes)
            if _headroom() >= target:
                return True
        return _headroom() >= target

    def lru_ram_victims(self) -> List[str]:
        """Droppable warm RAM-tier refs, LRU first (pinned/executing excluded)."""
        with self._lock:
            ref_leases = self._ref_leases
            candidates = [
                e for e in self._entries.values()
                if e.tier is Tier.RAM and e.obj is not None
                and not e.pinned and e.refcount <= 0 and e.holders <= 0
                and not ref_leases.get(e.ref)
            ]
            candidates.sort(key=lambda e: e.last_used)
            return [e.ref for e in candidates]

    def acquire_shared(
        self,
        key: "LoadedComponentKey",
        loader: Callable[[], Any],
        *,
        vram_bytes: int = 0,
        pin: bool = False,
    ) -> Any:
        """Load-once-or-reuse a shared immutable component set."""
        ref = key.cache_id()
        with self._lock:
            e = self._entries.get(ref)
            if e is not None and e.obj is not None:
                self._shared_hits += 1
                e.holders += 1
                e.last_used = time.monotonic()
                return e.obj
            self._shared_misses += 1
            obj = loader()
            measured = int(vram_bytes)
            if measured <= 0:
                measured = int(estimate_cuda_resident_gb(obj) * _GiB)
            e = _Entry(
                ref=ref,
                tier=Tier.VRAM if measured > 0 else Tier.RAM,
                obj=obj,
                vram_bytes=measured,
                vram_hint=measured,
                pinned=pin,
                holders=1,
            )
            self._entries[ref] = e
            state, vb = (IN_VRAM, measured) if e.tier is Tier.VRAM else (IN_RAM, 0)
        self._emit(ref, state, vb)
        return obj

    def release_shared(self, key: "LoadedComponentKey") -> int:
        """Drop one shared hold; returns the new holder count."""
        with self._lock:
            e = self._entries.get(key.cache_id())
            if e is None:
                return 0
            if e.holders > 0:
                e.holders -= 1
            return e.holders

    def shared_refcount(self, key: "LoadedComponentKey") -> int:
        with self._lock:
            e = self._entries.get(key.cache_id())
            return e.holders if e else 0

    def shared_obj(self, key: "LoadedComponentKey") -> Any:
        return self.obj(key.cache_id())

    def shared_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": self._shared_hits,
                "misses": self._shared_misses,
                "entries": [
                    {"ref": e.ref, "tier": e.tier.value, "refcount": e.refcount,
                     "holders": e.holders, "vram_bytes": e.vram_bytes,
                     "pinned": e.pinned}
                    for e in self._entries.values() if e.ref.startswith("shared::")
                ],
            }

    def transition_stats(self) -> Dict[str, Dict[str, int]]:
        """Per-ref swap telemetry: promote/demote counts + last wall durations for every entry that has ever transitioned."""
        with self._lock:
            return {
                e.ref: {
                    "promotes": e.promote_count,
                    "demotes": e.demote_count,
                    "last_promote_ms": e.last_promote_ms,
                    "last_demote_ms": e.last_demote_ms,
                }
                for e in self._entries.values()
                if e.promote_count or e.demote_count
            }

    def drain_shared(self, *, force: bool = False) -> int:
        """Evict shared entries with no holders (or everything when ``force``)."""
        with self._lock:
            victims = [
                r for r, e in self._entries.items()
                if r.startswith("shared::")
                and (force or (e.holders <= 0 and e.refcount <= 0))
            ]
        freed = 0
        for r in victims:
            if self.evict(r, force=force):
                freed += 1
        return freed


def _digest(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.strip()
    elif isinstance(value, dict):
        raw = repr(sorted((str(k), repr(v)) for k, v in value.items()))
    else:
        raw = repr(value)
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def content_set_digest(files: Any) -> str:
    """Digest of a component's sorted ``(relative_path, blake3)`` pairs — the CONTENT identity of the file set."""
    rows = sorted(f"{str(p)}\x1f{str(d)}" for p, d in dict(files).items())
    if not rows:
        return ""
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class LoadedComponentKey:
    """Canonical identity of a loadable immutable component set, keyed by CONTENT."""

    content_digest: str = ""
    dtype: str = ""
    quantization: str = ""
    quant_config_digest: str = ""
    device_id: int = 0
    placement: str = "full"
    component_set: str = ""
    adapter_id: str = ""
    label: str = field(default="", compare=False)

    @classmethod
    def for_component(
        cls,
        *,
        content_digest: str,
        component: str = "",
        binding: Any = None,
        dtype: str = "",
        quantization: str = "",
        quant_config: Any = None,
        device_id: int = 0,
        placement: str = "full",
        adapter_id: str = "",
        label: str = "",
    ) -> "LoadedComponentKey":
        """Key for one component of a bound snapshot: content digest + the binding's load-affecting facts (dtype, storage_dtype)."""
        if binding is not None:
            dtype = dtype or str(getattr(binding, "dtype", "") or "")
            quantization = quantization or str(getattr(binding, "storage_dtype", "") or "")
            if not label:
                ref = str(getattr(binding, "path", "") or "")
                label = f"{ref}/{component}" if ref else component
        return cls(
            content_digest=str(content_digest or "").strip(),
            dtype=str(dtype or "").strip().lower(),
            quantization=str(quantization or "").strip().lower(),
            quant_config_digest=_digest(quant_config),
            device_id=int(device_id),
            placement=str(placement or "full").strip() or "full",
            component_set=str(component or "").strip(),
            adapter_id=str(adapter_id or "").strip(),
            label=str(label or component or "").strip(),
        )

    def cache_id(self) -> str:
        fields = (
            self.content_digest, self.dtype, self.quantization,
            self.quant_config_digest, str(self.device_id), self.placement,
            self.component_set, self.adapter_id,
        )
        digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:16]
        readable = (self.component_set or "?").replace("/", "--")[:48]
        return f"shared::{readable}::dev{self.device_id}::{digest}"


__all__ = [
    "Residency",
    "Tier",
    "DeviceGroup",
    "Lease",
    "HostRamHeadroom",
    "LoadedComponentKey",
    "content_set_digest",
    "move_verified",
    "ON_DISK", "IN_RAM", "IN_VRAM", "EVICTED",
]
