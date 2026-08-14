"""Model residency: LRU VRAM/RAM/disk tiers + shared-component cache.

One registry for everything loaded or on disk, keyed once by canonical ref
string. Public API only — no private reach-ins, no tier strings smuggled
through model ids. Emits residency transitions through an ``on_event``
callback (the worker maps them to wire ``ModelEvent``s; the local CLI ignores
them).

Eviction is driven by FREE VRAM, never total capacity: ``make_room(needed)``
demotes LRU entries until measured free VRAM covers the request. An explicit
``vram_budget_bytes`` (shared-GPU slice, tests) replaces the probe with
``budget - tracked``.

Multi-pipeline endpoints (e.g. several variants of one family on one GPU) use
the same registry: register each pipeline under its ref, wrap handler
execution in :meth:`executing` (pin-while-executing), and call
:meth:`make_room` before promoting — cross-pipeline eviction picks the LRU
non-pinned, non-executing victim.
"""

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

logger = logging.getLogger(__name__)

_GiB = 1024 ** 3
# Free-VRAM slack preserved beyond the requested headroom (activations).
_VRAM_MARGIN_BYTES = 2 * _GiB
# Host-RAM floor below which the warm RAM tier is refused; demote() then fails
# and the owner tears down instead. The floor and its rationale are owned by
# `memory.effective_ram_floor_gb`.
def _effective_ram_floor_gb() -> float:
    return effective_ram_floor_gb(get_total_ram_gb())

# Residency event states (mirrors the wire ModelEvent vocabulary).
ON_DISK = "on_disk"
IN_RAM = "in_ram"
IN_VRAM = "in_vram"
EVICTED = "evicted"

EventFn = Callable[[str, str, int, int], None]  # (ref, state, vram_bytes, duration_ms)


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
        """The requirement exceeds the whole host — no eviction can help, and
        no identically-sized pod ever will either."""
        return self.total_bytes > 0 and self.required_bytes > self.total_bytes


REPLICATED = "replicated"
SHARDED = "sharded"
_PLACEMENT_MODES = (REPLICATED, SHARDED)


@dataclass(frozen=True)
class DeviceGroup:
    """The unit of placement (see WORKER-RESIDENCY-DESIGN "Multi-GPU").

    A device-group is the set of CUDA devices one materialization may span
    (size 1 today; a TP mesh later). VRAM accounting is PER GROUP and never
    summed across groups — VRAM is not fungible between cards: a 3x24GB pod
    has three 24GB pools, not one 72GB pool. One :class:`Residency` registry
    accounts for exactly one group's pool; the future multi-group agent owns
    one registry per executor/device-group, sharing only the disk tier.

    ``placement_mode`` says how a materialization occupies the group, and it
    decides the arithmetic:

    - ``replicated`` (default) — every member holds a FULL copy of the
      weights. Sequence/context parallelism is this: activations shard,
      weights do not. The group's budget is therefore the **smallest**
      member's free pool, never the sum. A 2x24GB replicated group that
      summed would report 48GB and admit a 30GB model fitting on neither card.
    - ``sharded`` — the weights themselves are split across members (a
      future TP/pipeline mesh), so the pool genuinely IS the sum.

    A single-device group is identical under both; the distinction only
    starts paying at degree >= 2.
    """

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
        """Measured free bytes for each member, in declaration order. A
        device the host does not actually have contributes 0 (a group is a
        plan; the probe reports physics) — which under ``replicated`` makes
        the whole group unusable, correctly: you cannot replicate onto a
        card that is not there."""
        import torch

        if not torch.cuda.is_available():
            return []
        count = int(torch.cuda.device_count())
        out: List[int] = []
        for d in self.devices:
            out.append(
                int(torch.cuda.mem_get_info(d)[0]) if 0 <= int(d) < count else 0
            )
        return out

    def free_vram_bytes(self) -> int:
        """Free VRAM budget for THIS group under its placement mode: the
        MIN across members when replicated, the sum when sharded."""
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
    vram_hint: int = 0           # last measured footprint (survives demotion)
    pinned: bool = False
    refcount: int = 0            # live executions (pin-while-executing)
    # Records referencing a shared component. Holders do NOT block
    # VRAM->RAM demotion (an idle pick's component may swap out; the owner
    # promotes it back before executing) but DO block evict/release_to_disk:
    # the registry must never drop its handle on a module that live
    # pipelines still alias.
    holders: int = 0
    last_used: float = field(default_factory=time.monotonic)
    # Swap telemetry: tier-transition counts + last durations.
    promote_count: int = 0
    demote_count: int = 0
    last_promote_ms: int = 0
    last_demote_ms: int = 0

    @property
    def movable(self) -> bool:
        """True when the registry can actually move this object between
        devices (``.to()``); offload-hooked pipelines own their placement."""
        return (
            self.obj is not None
            and callable(getattr(self.obj, "to", None))
            and not _obj_manages_own_device(self.obj)
        )


def _default_free_vram_bytes(group: Optional[DeviceGroup] = None) -> int:
    """Free VRAM of ONE device-group. Never an all-device SUM: a 3x24GB pod
    summing to 72GB free admits a 30GB model that fits on no single card."""
    return (group or DeviceGroup()).free_vram_bytes()


def _obj_manages_own_device(obj: Any) -> bool:
    """diffusers offload hooks own placement; manual .to() would break them."""
    return getattr(obj, "_cozy_low_vram_mode", None) in (
        "model_offload", "group_offload", "sequential",
    )


def _obj_offload_hooked(obj: Any) -> bool:
    """Any offload arming that parks weights in host RAM: the diffusers
    CPU-offload modes plus the block-window rung."""
    if _obj_manages_own_device(obj):
        return True
    try:
        from .loading import block_offload_active

        return bool(block_offload_active(obj))
    except Exception:
        return False


def _move_obj(obj: Any, device: str) -> None:
    """Move an object between devices: the pinned swap cache when it applies
    (full-PCIe H2D promotes, pointer-swap demotes), else whole-object
    ``.to(device)``. Raises on failure — the caller
    (:meth:`Residency._move_verified`) owns rollback; swallowing a mid-move
    CUDA OOM here books a half-moved pipeline as resident."""
    if obj is None or _obj_manages_own_device(obj):
        return

    if swap_object(obj, device):
        return
    to = getattr(obj, "to", None)
    if callable(to):
        to(device)


class Lease:
    """Admission lease: the set of refs one job needs, taken BEFORE the job
    starts and held for its whole lifetime.

    While live, every named ref is excluded from eviction/demotion victim
    selection — including refs whose entries do not exist yet, which closes
    the window where a freshly created entry could be demoted between its
    ``track_vram`` and the execution-time pin. Refs not yet VRAM-resident
    additionally carry a byte RESERVATION so concurrent admissions cannot
    double-book the same free bytes; a reservation is consumed the moment
    the ref actually books VRAM, and any unconsumed remainder dies with the
    lease.

    A lease also carries an ACTIVATION reservation: the transient
    VRAM a running request allocates on top of its weights. Unlike weight
    reservations — which are MAXed per ref because two jobs needing the same
    checkpoint share one future load — activation SUMS across live leases:
    every concurrent request allocates its own latents and attention
    workspace. It is never consumed, only released, because it is transient
    by definition."""

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
    """LRU-tiered model registry. Thread-safe."""

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
        # The one device-group whose VRAM pool this registry accounts for.
        # Admission, make_room and the free probe all speak this group;
        # nothing here ever sums VRAM across groups.
        self.device_group = device_group or DeviceGroup()
        # Called with (ref, obj) before a VRAM->RAM demotion moves the object
        # (executor wires adapter detach here). Must never raise.
        self.pre_demote: Optional[Callable[[str, Any], None]] = None
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.RLock()
        self._shared_hits = 0
        self._shared_misses = 0
        # Admission leases: live lease objects, plus a per-ref map of
        # outstanding byte reservations (ref -> {lease id -> reserved bytes}).
        # The outstanding claim for a ref is the MAX across leases (two jobs
        # cold-needing the same ref share ONE future load), consumed by the
        # ref's actual track_vram booking.
        self._leases: Dict[int, Lease] = {}
        self._ref_reservations: Dict[str, Dict[int, int]] = {}
        # Learned activation footprints: key -> observed transient VRAM
        # high-water. Populated ONLY by measurement (record_activation); an
        # unmeasured key claims nothing. No endpoint knob feeds this.
        self._activation: Dict[str, int] = {}
        log_ram_budget_once(floor_gb=_effective_ram_floor_gb())

    # ---- events -------------------------------------------------------------

    def _emit(self, ref: str, state: str, vram_bytes: int = 0, duration_ms: int = 0) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(ref, state, int(vram_bytes), int(duration_ms))
        except Exception as exc:
            logger.exception("residency event callback failed for %s", ref)
            # The hub's residency view just silently diverged; the activity
            # stream is an independent channel, so confess there.
            activity_mod.emit_event(
                activity_mod.KIND_RESIDENCY_FAULT,
                f"ref={ref} state={state}: residency event callback failed "
                f"(hub residency view may be stale): "
                f"{type(exc).__name__}: {exc}",
                phase="event_callback_failed",
            )

    # ---- probes ---------------------------------------------------------------

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

    # ---- queries ---------------------------------------------------------------

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
        """Replace a resident ref's bookkeeping object without a state event.

        Record-owner handoff can change the representative object while the
        same logical ref remains resident. In particular, ``None`` releases a
        departed typed object when the surviving owner is tenant-loaded.
        """
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
        """Last measured VRAM footprint (survives demotion) — the load-size
        estimate for make_room before a re-load/promotion."""
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

    # ---- registration / transitions ---------------------------------------------

    def touch(self, ref: str) -> None:
        with self._lock:
            e = self._entries.get(ref)
            if e:
                e.last_used = time.monotonic()

    def track_disk(self, ref: str, path: Path) -> None:
        """Register an on-disk snapshot.

        Updating the disk path of a RAM/VRAM entry is not a tier transition:
        the loaded object remains the highest residency until its owner
        releases it. ``release_to_disk`` emits the later honest demotion.
        """
        with self._lock:
            e = self._entries.get(ref)
            if e is None:
                self._entries[ref] = _Entry(ref=ref, tier=Tier.DISK, path=Path(path))
            else:
                e.path = Path(path)
                e.last_used = time.monotonic()
                return  # same highest tier, no event
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
        """Register a VRAM-resident object with its MEASURED footprint
        (``torch.cuda.memory_allocated`` delta across the load, or
        :func:`~gen_worker.models.memory.estimate_cuda_resident_gb`).

        Offload-hooked pipelines (diffusers CPU-offload modes, block-window
        offload) are booked in the RAM tier instead: their weights rest in host
        RAM and the allocator delta across such a load is noise, so a VRAM
        booking would be a lie in both tier and size."""
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
            # The booked bytes are now IN the measured pool — any admission
            # reservation for this ref is satisfied, not outstanding.
            self._consume_reservation_locked(ref)
        self._emit(ref, IN_VRAM, max(0, measured))

    def demote(self, ref: str) -> bool:
        """VRAM -> RAM warm tier. Only performs transitions it can actually
        execute: the entry must hold a movable object and host RAM must have
        headroom — otherwise it refuses (False) and the OWNER of the memory
        (executor record teardown) must free it and book the result. Never
        books a state it didn't produce. Refuses pinned / executing entries."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None or e.tier is not Tier.VRAM or e.pinned or e.refcount > 0:
                return False
            if self._leased_locked(ref):
                return False  # an admitted job needs it
            if not e.movable:
                return False
            # Size-aware RAM floor: demoting a pipeline of size X eats ~X host
            # RAM, and landing it must still leave the floor or the host
            # thrashes into the keepalive-stall livelock. Bytes already staged
            # in the pinned swap cache are already resident host RAM, so only
            # the uncached remainder is a fresh demand.
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
                return False  # entry stays VRAM; object restored to cuda
            e.tier = Tier.RAM
            e.vram_bytes = 0
            e.demote_count += 1
            e.last_demote_ms = int((time.monotonic() - t0) * 1000)
            duration_ms = e.last_demote_ms
        flush_memory()
        self._emit(ref, IN_RAM, duration_ms=duration_ms)
        return True

    def _move_verified(self, obj: Any, device: str, *, ref: str = "") -> bool:
        """Move + paranoid completeness walk: after ``.to(device)``,
        every module parameter/buffer must actually be on ``device`` — a move
        that raised (mid-move CUDA OOM) or skipped tensors gets one targeted
        repair pass, and an unrepairable object is rolled back to the other
        side. Residency NEVER books a mixed-device pipeline: mixed devices
        fatal mid-denoise ("Expected all tensors to be on the same device")."""
        restore = "cpu" if device != "cpu" else "cuda"
        try:
            self._move(obj, device)
            missed = device_mismatches(obj, device)
            if missed:
                logger.warning(
                    "residency: .to(%s) on %s left %d tensors behind (e.g. %s); repairing",
                    device, ref or type(obj).__name__, len(missed), missed[:3],
                )
                missed = repair_device_placement(obj, device)
            if not missed:
                return True
            logger.error(
                "residency: move of %s to %s incomplete after repair (%s); rolling back",
                ref or type(obj).__name__, device, missed[:5],
            )
        except Exception as exc:
            logger.error(
                "residency: .to(%s) failed for %s: %s; rolling back",
                device, ref or type(obj).__name__, exc,
            )
        try:
            self._move(obj, restore)
            left = repair_device_placement(obj, restore)
            if left:
                logger.critical(
                    "residency: rollback of %s to %s ALSO incomplete (%s) — "
                    "object is mixed-device and unusable",
                    ref or type(obj).__name__, restore, left[:5],
                )
                # The next forward on this object fatals mid-denoise
                # ("Expected all tensors to be on the same device"), so the
                # hub must see the cause, not only the downstream job error.
                activity_mod.emit_event(
                    activity_mod.KIND_RESIDENCY_FAULT,
                    f"ref={ref or type(obj).__name__} wanted={device} "
                    f"restore={restore}: move AND rollback both incomplete "
                    f"(e.g. {left[:3]}); object is mixed-device and unusable",
                    phase="mixed_device_unusable",
                )
        except Exception as exc:
            logger.exception("residency: rollback .to(%s) failed for %s", restore, ref)
            activity_mod.emit_event(
                activity_mod.KIND_RESIDENCY_FAULT,
                f"ref={ref or type(obj).__name__} wanted={device} "
                f"restore={restore}: rollback move raised; object is likely "
                f"mixed-device and unusable: {type(exc).__name__}: {exc}",
                phase="mixed_device_unusable",
            )
        flush_memory()
        return False

    @property
    def vram_device(self) -> str:
        """Where THIS registry's promotions land: ``cuda`` (thread-current) for
        the default group, an explicit ``cuda:N`` once a topology says the
        group owns card N — a group-1 instance must never load onto card 0
        merely because the loading thread's current device said so."""
        primary = int(self.device_group.primary)
        return "cuda" if primary == 0 else f"cuda:{primary}"

    def promote(self, ref: str, device: str = "") -> bool:
        """RAM -> VRAM (makes room first). True when resident afterward —
        i.e. every tensor verified on ``device``; a failed/partial move is
        rolled back to CPU and refused instead of booked."""
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
            # Paranoid fast path: a VRAM-booked entry must actually be device-
            # complete (a crashed rollback / out-of-band .to() must not serve
            # a mixed-device pipeline). The clean-case walk is tensor metadata
            # only — no data movement.
            missed = device_mismatches(obj, device)
            if not missed:
                return True
            logger.warning(
                "residency: VRAM-tier %s holds %d off-device tensors (e.g. %s); repairing",
                ref, len(missed), missed[:3],
            )
            if not repair_device_placement(obj, device):
                return True
            # Unrepairable: book the truth and refuse.
            with self._lock:
                e = self._entries.get(ref)
                if e is None:
                    return False
                evicted = self._move_verified(e.obj, "cpu", ref=ref)
                if not evicted:
                    # The booking below must NOT be unconditional: a failed
                    # eviction leaves the object on CUDA (`_move_verified`
                    # rolled it back), so writing `tier=RAM, vram_bytes=0`
                    # would make `make_room` hand out headroom that does not
                    # exist and OOM an unrelated `promote()` later. Book the
                    # truth in BOTH branches and still refuse the promotion.
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
            # Never-measured entry: estimate from weights so make_room asks for
            # real headroom instead of 0 — a 0-byte ask promotes multi-GB
            # pipelines into whatever is free and OOMs mid-move.
            hint = int(estimate_pipeline_size_gb(obj) * _GiB)
        t0 = time.monotonic()  # swap wall incl. the make_room demote
        if not self.make_room(hint, for_refs=(ref,)):
            # Refuse FAST instead of paying a doomed multi-GB partial move +
            # rollback. Gate on the hard physical minimum — actual
            # weight bytes — never the bookkeeping hint (which can inflate
            # from residual shares, and is faked by budget-mode tests).
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
                return False  # entry stays RAM; object restored to cpu
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
        """Drop the loaded object entirely; disk snapshot kept.
        Refuses executing/shared-held entries."""
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
        """Remove the entry entirely (fully gone -> EVICTED). Refuses
        executing/shared-held entries unless ``force``. Does not delete disk
        files — callers own filesystem GC."""
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

    # ---- protection ----------------------------------------------------------

    @contextmanager
    def executing(self, *refs: str) -> Iterator[None]:
        """Pin-while-executing: entries named here are not eviction candidates
        for the duration (cross-pipeline eviction never yanks a model that a
        handler is actively using)."""
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

    # ---- admission leases ------------------------------------------------------

    def _leased_locked(self, ref: str) -> bool:
        return bool(self._ref_leases.get(ref))

    @property
    def _ref_leases(self) -> Dict[str, set]:
        # Derived view: ref -> live lease ids naming it. Small (few in-flight
        # jobs x few refs); rebuilt on demand to keep one source of truth.
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
        """Transient VRAM every OTHER live request will allocate. SUMMED, not
        maxed: concurrency costs activation per request."""
        return sum(
            lease.activation for lid, lease in self._leases.items()
            if lid != exclude_lease_id
        )

    def record_activation(self, key: str, observed_bytes: int) -> int:
        """Learn one request's measured transient VRAM (peak allocated minus
        what was already allocated when the handler took the GPU).

        The stored hint is a DECAYING high-water: it jumps to a larger
        observation immediately (under-reserving is what OOMs) and bleeds off
        ~12% per subsequent request otherwise, so one 4096^2 outlier does not
        permanently tax residency depth for a stream of 512^2 requests. Purely
        observational — there is no declared value to fall back to."""
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
        """Take an admission lease over one job's refs.

        ``sizes`` maps ref -> expected VRAM bytes (0 = unknown; the ref is
        still lease-protected, it just books no reservation).
        ``activation_bytes`` is the LEARNED transient footprint of this
        request — 0 until measured — reserved for the lease's whole life so
        concurrent requests cannot each assume the same headroom. Admission
        never REFUSES here (the adaptive-fit ladder downstream absorbs genuine
        overcommit), but from this moment on (a) no eviction/demotion path may
        pick any named ref as a victim, and (b) concurrent admissions see the
        not-yet-loaded bytes as claimed (:meth:`fits`, :meth:`make_room`), so
        two jobs can no longer book the same free bytes and OOM each other
        mid-load."""
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
                    continue  # already booked in the pool; nothing to claim
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
        """The ref's bytes just became real (track_vram) — its future claim
        is no longer outstanding for ANY lease."""
        self._ref_reservations.pop(ref, None)

    def fits(self, sizes: Mapping[str, int], *, activation_bytes: int = 0) -> bool:
        """Cheap honest admission query: could this ref set be served now —
        counting measured free VRAM, minus other jobs' outstanding weight and
        activation claims, plus what LRU demotion of unprotected entries could
        reclaim. ``activation_bytes`` is this request's own transient
        footprint, which must fit alongside its weights. Read only; takes
        nothing."""
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

    # ---- pressure -------------------------------------------------------------

    def lru_vram_victims(self) -> List[str]:
        """Evictable VRAM refs, LRU first (pinned/executing excluded).

        Genuinely shared components (2+ holders — e.g. a TE/VAE aliased by
        several resident picks) sort LAST: swapping one out costs every sibling
        a re-promote, so exclusive entries go first."""
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
        """Demote LRU VRAM entries until measured free VRAM covers
        ``needed_bytes`` + margin. True when the headroom was reached.
        Only movable entries are demoted here; when this returns False the
        caller (executor) tears down non-movable LRU victims itself.

        Free bytes CLAIMED by other admissions' outstanding reservations do not
        count as available; ``for_refs`` names the refs this call is making
        room FOR, whose own reservations are the very demand being satisfied
        and are therefore excluded. The same exclusion identifies the CALLING
        lease, whose activation claim is likewise not subtracted — it is either
        already allocated (and thus already visible in the measured free bytes)
        or it is this job's own future demand. Other in-flight requests'
        activation IS subtracted: their latents and attention workspace are
        real bytes this load must not eat."""
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

    # ---- shared components -----------------------------------------------------

    def acquire_shared(
        self,
        key: "LoadedComponentKey",
        loader: Callable[[], Any],
        *,
        vram_bytes: int = 0,
        pin: bool = False,
    ) -> Any:
        """Load-once-or-reuse a shared immutable component set. The entry is
        registered ONCE (VRAM counted once) under ``key.cache_id()`` and held
        (``holders``) so evict/release can never drop the registry's handle
        while any pipeline aliases the module. Holders do NOT block VRAM->RAM
        demotion: an idle pick's component may swap out under real
        pressure; owners re-promote before executing (pin + promote path)."""
        ref = key.cache_id()
        with self._lock:
            e = self._entries.get(ref)
            if e is not None and e.obj is not None:
                self._shared_hits += 1
                e.holders += 1
                e.last_used = time.monotonic()
                return e.obj
            # MISS: load under the lock so a concurrent acquire of the same key
            # can't double-load (loads are GPU-semaphore-serialized anyway).
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
        """Drop one shared hold; returns the new holder count. The last
        release makes the entry an ordinary LRU candidate — it is NOT freed
        eagerly: a hot GPU keeps components resident for the next pick, and
        real pressure reclaims them through make_room."""
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
        """Per-ref swap telemetry: promote/demote counts + last wall
        durations for every entry that has ever transitioned."""
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


# ---------------------------------------------------------------------------
# Shared-component identity
# ---------------------------------------------------------------------------


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
    """Digest of a component's sorted ``(relative_path, blake3)`` pairs — the
    CONTENT identity of the file set. Content-addressed = immutable:
    a tag moving to new bytes changes file digests, hence this digest."""
    rows = sorted(f"{str(p)}\x1f{str(d)}" for p, d in dict(files).items())
    if not rows:
        return ""
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class LoadedComponentKey:
    """Canonical identity of a loadable immutable component set, keyed by
    CONTENT. Two bindings share one loaded entry IFF the bytes and
    every load-affecting fact are equal: the file-set content digest plus
    dtype, quant scheme+config, GPU index, placement mode, component name and
    adapter overlays. ref/revision are NOT identity — byte-identical
    components mirrored under different refs share one in-memory copy; the
    readable ref survives only in the ``label`` (cache_id display)."""

    content_digest: str = ""      # content_set_digest of the component's files
    dtype: str = ""
    quantization: str = ""        # quant scheme / storage_dtype
    quant_config_digest: str = ""
    device_id: int = 0
    placement: str = "full"
    component_set: str = ""       # component name (e.g. "text_encoder")
    adapter_id: str = ""
    label: str = field(default="", compare=False)  # readable, non-identity

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
        """Key for one component of a bound snapshot: content digest + the
        binding's load-affecting facts (dtype, storage_dtype)."""
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
        # Readable part comes from IDENTITY fields only: equal keys MUST map
        # to one cache entry even when their ref labels differ (that is the
        # entire point of content keying).
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
    "ON_DISK", "IN_RAM", "IN_VRAM", "EVICTED",
]
