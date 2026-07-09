"""Model residency: LRU VRAM/RAM/disk tiers + shared-component cache (#366).

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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .memory import (
    device_mismatches,
    estimate_cuda_resident_gb,
    estimate_pipeline_size_gb,
    flush_memory,
    get_available_ram_gb,
    repair_device_placement,
)

logger = logging.getLogger(__name__)

_GiB = 1024 ** 3
# Free-VRAM slack preserved beyond the requested headroom (activations).
_VRAM_MARGIN_BYTES = 2 * _GiB
# Host-RAM floor below which the warm RAM tier is refused (don't push the
# host into swap); demote() then fails and the owner tears down instead.
_RAM_FLOOR_GB = 8.0

# Residency event states (mirrors the wire ModelEvent vocabulary).
ON_DISK = "on_disk"
IN_RAM = "in_ram"
IN_VRAM = "in_vram"
EVICTED = "evicted"

EventFn = Callable[[str, str, int], None]  # (ref, state, vram_bytes)


class Tier(str, Enum):
    VRAM = "VRAM"
    RAM = "RAM"
    DISK = "DISK"


@dataclass
class _Entry:
    ref: str
    tier: Tier
    path: Optional[Path] = None
    obj: Any = None
    vram_bytes: int = 0
    vram_hint: int = 0           # last measured footprint (survives demotion)
    pinned: bool = False
    refcount: int = 0            # live executions / shared holders
    last_used: float = field(default_factory=time.monotonic)

    @property
    def movable(self) -> bool:
        """True when the registry can actually move this object between
        devices (``.to()``); offload-hooked pipelines own their placement."""
        return (
            self.obj is not None
            and callable(getattr(self.obj, "to", None))
            and not _obj_manages_own_device(self.obj)
        )


def _default_free_vram_bytes() -> int:
    """Free VRAM summed across ALL CUDA devices (multi-GPU workers spread
    jobs across cards; a device-0-only probe under-reports)."""
    try:
        import torch

        if torch.cuda.is_available():
            return sum(
                int(torch.cuda.mem_get_info(i)[0])
                for i in range(torch.cuda.device_count())
            )
    except Exception:
        pass
    return 0


def _obj_manages_own_device(obj: Any) -> bool:
    """diffusers offload hooks own placement; manual .to() would break them."""
    return getattr(obj, "_cozy_low_vram_mode", None) in (
        "model_offload", "group_offload", "sequential",
    )


def _move_obj(obj: Any, device: str) -> None:
    """Whole-object ``.to(device)``. Raises on failure — the caller
    (:meth:`Residency._move_verified`) owns rollback; swallowing a mid-move
    CUDA OOM here used to book a half-moved pipeline as resident (gw#409)."""
    if obj is None or _obj_manages_own_device(obj):
        return
    to = getattr(obj, "to", None)
    if callable(to):
        to(device)


class Residency:
    """LRU-tiered model registry. Thread-safe."""

    def __init__(
        self,
        *,
        on_event: Optional[EventFn] = None,
        vram_budget_bytes: Optional[int] = None,
        free_vram_bytes_fn: Optional[Callable[[], int]] = None,
        move_fn: Callable[[Any, str], None] = _move_obj,
    ) -> None:
        self._on_event = on_event
        self._vram_budget = vram_budget_bytes
        self._free_vram_fn = free_vram_bytes_fn
        self._move = move_fn
        # Called with (ref, obj) before a VRAM->RAM demotion moves the object
        # (executor wires adapter detach here, gw#399). Must never raise.
        self.pre_demote: Optional[Callable[[str, Any], None]] = None
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.RLock()
        self._shared_hits = 0
        self._shared_misses = 0

    # ---- events -------------------------------------------------------------

    def _emit(self, ref: str, state: str, vram_bytes: int = 0) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(ref, state, int(vram_bytes))
        except Exception:
            logger.exception("residency event callback failed for %s", ref)

    # ---- probes ---------------------------------------------------------------

    def free_vram_bytes(self) -> int:
        if self._vram_budget is not None:
            with self._lock:
                used = sum(e.vram_bytes for e in self._entries.values() if e.tier is Tier.VRAM)
            return max(0, int(self._vram_budget) - used)
        if self._free_vram_fn is not None:
            return int(self._free_vram_fn())
        return _default_free_vram_bytes()

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
        """Register (or demote-to) an on-disk snapshot."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None:
                self._entries[ref] = _Entry(ref=ref, tier=Tier.DISK, path=Path(path))
            else:
                e.path = Path(path)
                if e.tier is Tier.DISK:
                    e.last_used = time.monotonic()
                    return  # no transition, no event
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
        :func:`~gen_worker.models.memory.estimate_cuda_resident_gb`)."""
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
            if not e.movable or get_available_ram_gb() < _RAM_FLOOR_GB:
                return False
            if self.pre_demote is not None:
                try:
                    self.pre_demote(ref, e.obj)
                except Exception:
                    logger.exception("pre_demote hook failed for %s", ref)
            if not self._move_verified(e.obj, "cpu", ref=ref):
                return False  # entry stays VRAM; object restored to cuda
            e.tier = Tier.RAM
            e.vram_bytes = 0
        flush_memory()
        self._emit(ref, IN_RAM)
        return True

    def _move_verified(self, obj: Any, device: str, *, ref: str = "") -> bool:
        """Move + paranoid completeness walk (gw#409): after ``.to(device)``,
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
        except Exception:
            logger.exception("residency: rollback .to(%s) failed for %s", restore, ref)
        flush_memory()
        return False

    def promote(self, ref: str, device: str = "cuda") -> bool:
        """RAM -> VRAM (makes room first). True when resident afterward —
        i.e. every tensor verified on ``device``; a failed/partial move is
        rolled back to CPU and refused instead of booked (gw#409)."""
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
            # Unrepairable: book the truth (RAM) and refuse.
            with self._lock:
                e = self._entries.get(ref)
                if e is None:
                    return False
                if not self._move_verified(e.obj, "cpu", ref=ref):
                    logger.critical("residency: %s stuck mixed-device", ref)
                e.tier = Tier.RAM
                e.vram_bytes = 0
            flush_memory()
            self._emit(ref, IN_RAM)
            return False
        if hint <= 0:
            # Never-measured entry: estimate from weights so make_room asks
            # for real headroom instead of 0 (a 0-byte ask promoted 6.9GB
            # pipelines into ~2GB free and OOMed mid-move, gw#409).
            hint = int(estimate_pipeline_size_gb(obj) * _GiB)
        self.make_room(hint)
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
            measured = e.vram_bytes
        self._emit(ref, IN_VRAM, measured)
        return True

    def release_to_disk(self, ref: str) -> bool:
        """Drop the loaded object entirely (UNLOAD); disk snapshot kept.
        Refuses executing/shared-held entries."""
        with self._lock:
            e = self._entries.get(ref)
            if e is None:
                return False
            if e.refcount > 0:
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
            if e.refcount > 0 and not force:
                return False
            was_loaded = e.tier in (Tier.VRAM, Tier.RAM)
            del self._entries[ref]
        if was_loaded:
            flush_memory()
        self._emit(ref, EVICTED)
        return True

    # ---- protection ----------------------------------------------------------

    def pin(self, ref: str) -> None:
        with self._lock:
            e = self._entries.get(ref)
            if e:
                e.pinned = True

    def unpin(self, ref: str) -> None:
        with self._lock:
            e = self._entries.get(ref)
            if e:
                e.pinned = False

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
            return bool(e and e.refcount > 0)

    # ---- pressure -------------------------------------------------------------

    def lru_vram_victims(self) -> List[str]:
        """Evictable VRAM refs, LRU first (pinned/executing excluded)."""
        with self._lock:
            candidates = [
                e for e in self._entries.values()
                if e.tier is Tier.VRAM and not e.pinned and e.refcount <= 0
            ]
            candidates.sort(key=lambda e: e.last_used)
            return [e.ref for e in candidates]

    def make_room(self, needed_bytes: int) -> bool:
        """Demote LRU VRAM entries until measured free VRAM covers
        ``needed_bytes`` + margin. True when the headroom was reached.
        Only movable entries are demoted here; when this returns False the
        caller (executor) tears down non-movable LRU victims itself."""
        target = int(needed_bytes) + _VRAM_MARGIN_BYTES
        if self.free_vram_bytes() >= target:
            return True
        for ref in self.lru_vram_victims():
            if not self.demote(ref):
                continue
            logger.info("residency: demoted LRU %s for %d bytes headroom", ref, needed_bytes)
            if self.free_vram_bytes() >= target:
                return True
        return self.free_vram_bytes() >= target

    # ---- shared components (#335, folded in) -----------------------------------

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
        (refcount) so eviction never reclaims it while any pipeline uses it."""
        ref = key.cache_id()
        with self._lock:
            e = self._entries.get(ref)
            if e is not None and e.obj is not None:
                self._shared_hits += 1
                e.refcount += 1
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
                pinned=pin,
                refcount=1,
            )
            self._entries[ref] = e
            state, vb = (IN_VRAM, measured) if e.tier is Tier.VRAM else (IN_RAM, 0)
        self._emit(ref, state, vb)
        return obj

    def release_shared(self, key: "LoadedComponentKey") -> int:
        """Drop one shared reference; returns the new count. The last release
        makes the entry an LRU candidate — it is not freed eagerly."""
        with self._lock:
            e = self._entries.get(key.cache_id())
            if e is None:
                return 0
            if e.refcount > 0:
                e.refcount -= 1
            return e.refcount

    def shared_refcount(self, key: "LoadedComponentKey") -> int:
        with self._lock:
            e = self._entries.get(key.cache_id())
            return e.refcount if e else 0

    def shared_obj(self, key: "LoadedComponentKey") -> Any:
        return self.obj(key.cache_id())

    def shared_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": self._shared_hits,
                "misses": self._shared_misses,
                "entries": [
                    {"ref": e.ref, "tier": e.tier.value, "refcount": e.refcount,
                     "vram_bytes": e.vram_bytes, "pinned": e.pinned}
                    for e in self._entries.values() if e.ref.startswith("shared::")
                ],
            }

    def drain_shared(self, *, force: bool = False) -> int:
        """Evict shared entries with refcount 0 (or everything when ``force``)."""
        with self._lock:
            victims = [
                r for r, e in self._entries.items()
                if r.startswith("shared::") and (force or e.refcount <= 0)
            ]
        freed = 0
        for r in victims:
            if self.evict(r, force=force):
                freed += 1
        return freed


# ---------------------------------------------------------------------------
# Shared-component identity (#335)
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


@dataclass(frozen=True)
class LoadedComponentKey:
    """Canonical identity of a loadable immutable component set. Two bindings
    share one loaded entry IFF every field is equal — the fields cover every
    dimension that would make the loaded bytes unshareable (provider/ref,
    revision/snapshot, dtype, quant scheme+config, GPU index, placement mode,
    subfolder, component-set/pipeline-class identity, adapter overlays)."""

    provider: str = "tensorhub"
    ref: str = ""
    revision: str = ""
    dtype: str = ""
    quantization: str = ""
    quant_config_digest: str = ""
    device_id: int = 0
    placement: str = "full"
    subfolder: str = ""
    component_set: str = ""
    adapter_id: str = ""

    @classmethod
    def from_binding(
        cls,
        binding: Any,
        *,
        device_id: int = 0,
        placement: str = "full",
        component_set: str = "",
        snapshot_digest: str = "",
        quantization: str = "",
        quant_config: Any = None,
        adapter_id: str = "",
    ) -> "LoadedComponentKey":
        provider = str(getattr(binding, "provider", "tensorhub") or "tensorhub").strip()
        ref = str(getattr(binding, "ref", "") or "").strip()
        dtype = str(getattr(binding, "dtype", "") or "").strip().lower()
        revision = str(getattr(binding, "revision", "") or "").strip() or str(snapshot_digest or "").strip()
        subfolder = str(getattr(binding, "subfolder", "") or "").strip()
        return cls(
            provider=provider,
            ref=ref,
            revision=revision,
            dtype=dtype,
            quantization=str(quantization or "").strip().lower(),
            quant_config_digest=_digest(quant_config),
            device_id=int(device_id),
            placement=str(placement or "full").strip() or "full",
            subfolder=subfolder,
            component_set=str(component_set or "").strip(),
            adapter_id=str(adapter_id or "").strip(),
        )

    def cache_id(self) -> str:
        fields = (
            self.provider, self.ref, self.revision, self.dtype,
            self.quantization, self.quant_config_digest, str(self.device_id),
            self.placement, self.subfolder, self.component_set, self.adapter_id,
        )
        digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:16]
        readable = (self.ref or "?").replace("/", "--")[:48]
        return f"shared::{readable}::dev{self.device_id}::{digest}"


def build_function_owned_pipeline(
    shared: Any,
    pipeline_cls: Optional[Any] = None,
    **extra_components: Any,
) -> Any:
    """Build a *function-owned* pipeline over SHARED immutable components:
    own scheduler/mutable state, same heavy modules (same CUDA storages).
    Tries ``pipeline_cls.from_pipe(shared)`` then ``pipeline_cls(**components)``."""
    cls = pipeline_cls or type(shared)
    from_pipe = getattr(cls, "from_pipe", None)
    if callable(from_pipe):
        return from_pipe(shared, **extra_components)
    comps = getattr(shared, "components", None)
    if isinstance(comps, dict):
        return cls(**{**comps, **extra_components})
    raise TypeError(
        f"cannot build a function-owned pipeline from {type(shared).__name__}: "
        "no from_pipe() and no .components dict to re-assemble"
    )


__all__ = [
    "Residency",
    "Tier",
    "LoadedComponentKey",
    "build_function_owned_pipeline",
    "ON_DISK", "IN_RAM", "IN_VRAM", "EVICTED",
]
