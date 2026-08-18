"""Model residency: admission BEFORE allocation, two tiers, serialized loads.

Paul's pgw#1372 residency ruling (2026-08-18, verbatim framing: *"this should
respect whatever the wishes of the worker are... we don't want random OOMs
because 3 checkpoints were loaded at the same time"*):

1. **Admission before allocation.** Placement is the WORKER's decision —
   author code is device-neutral (no ``.to()``/``torch_dtype`` in the
   contract file). Before constructing an instance, its EXACT weight byte
   size (the tensorfs manifest per lane — known ahead of load) plus an
   activation-headroom estimate (per model type / resolution class) is
   reserved against the VRAM budget; LRU residents are evicted until it
   fits. A model that can NEVER fit refuses typed at admission — a CUDA OOM
   mid-load or at first request is a design bug, not bad luck.
2. **Loads serialized per GPU.** Concurrent instance loads must not race the
   VRAM budget or fragment pinned staging: one load gate.
3. **Two-tier residency.** VRAM -> RAM-resident (host-staged weights;
   re-promotion is an H2D copy, no disk walk) -> dropped to the chunk store.
   Demotion/promotion happen only BETWEEN requests — single-flight per
   instance guarantees the window — and the author never observes a tier
   move.

The SDK-core lane's ``Model`` wrapper (pgw#1382) sits ABOVE this engine: the
worker resolves ``(checkpoint x lane)`` to a :class:`ModelBackend` factory
and calls :meth:`ResidencyManager.lease` around every entrypoint invocation.
The manager knows bytes, tiers and order — never author code.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, Protocol, Tuple

logger = logging.getLogger(__name__)

#: One resident identity: (checkpoint ref, lane contract handle).
InstanceKey = Tuple[str, str]


class ResidencyError(RuntimeError):
    """A residency decision could not be made; the message names the bytes."""


class NeverFits(ResidencyError):
    """Typed admission refusal: this instance exceeds the whole budget —
    no eviction schedule can ever make it fit."""


class ModelBackend(Protocol):
    """What one resident instance must provide (the pgw#1382 seam).

    The SDK-core Model wrapper implements these over the author's own
    ``load``/``unload`` hooks and the native tensor loader; the manager only
    ORDERS the moves. Every method runs with the instance idle — the manager
    guarantees the between-requests window — and under the worker's gates.
    """

    def load(self) -> None:
        """Chunk store -> VRAM. First residency only; serialized per GPU."""
        ...

    def demote_to_host(self) -> None:
        """VRAM -> host-staged weights (eviction to the warm tier)."""
        ...

    def promote_to_device(self) -> None:
        """Host -> VRAM (an H2D copy; never a disk walk)."""
        ...

    def drop(self) -> None:
        """Release both tiers; the chunk store is the only remaining copy.
        The author's ``unload`` hook fires inside (best-effort tidiness,
        never correctness)."""
        ...


class InstanceSizer(Protocol):
    """Byte facts, known AHEAD of any allocation."""

    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        """Exact weight bytes for (checkpoint, lane) — the tensorfs manifest
        for the lane's layout contract states this without loading."""
        ...

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        """The serving-time activation estimate (per model type / resolution
        class) reserved alongside the weights."""
        ...


class Tier(StrEnum):
    VRAM = "vram"
    HOST = "host"
    ABSENT = "absent"


@dataclass
class _Slot:
    key: InstanceKey
    backend: Any = None
    weight_bytes: int = 0
    headroom_bytes: int = 0
    #: Where the BYTES are. `placing` below is the reservation that precedes
    #: them — admission-before-allocation is exactly this distinction.
    tier: Tier = Tier.ABSENT
    #: True from the moment the VRAM reservation is claimed until the bytes
    #: are on-device (or the placement failed). Counted like VRAM residency.
    placing: bool = False
    #: LRU clock value at last release; larger = more recently used.
    last_used: int = 0
    #: Requests currently holding or awaiting a lease on this instance.
    #: A slot with inflight > 0 is never a tier-move candidate.
    inflight: int = 0
    #: Single-flight per model instance: one entrypoint at a time.
    gate: threading.Lock = field(default_factory=threading.Lock)

    @property
    def vram_footprint(self) -> int:
        return self.weight_bytes + self.headroom_bytes


class Lease:
    """One request's hold on a resident instance (single-flight).

    Constructed only by :meth:`ResidencyManager.lease`; ``backend`` is the
    live instance, on-device for the whole lease. Context-manager exit
    releases the instance and stamps LRU recency.
    """

    def __init__(self, manager: "ResidencyManager", slot: _Slot) -> None:
        self._manager = manager
        self._slot = slot
        self.backend = slot.backend
        self._released = False

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager._release(self._slot)


class ResidencyManager:
    """The worker's placement authority for model instances on ONE device.

    All byte accounting is RESERVATIONS (weights + headroom), never live
    allocator readings: the point is deciding *before* allocating.
    """

    def __init__(
        self,
        vram_budget_bytes: int,
        sizer: InstanceSizer,
        *,
        host_budget_bytes: int = 0,
    ) -> None:
        if vram_budget_bytes <= 0:
            raise ResidencyError("vram_budget_bytes must be positive")
        self.vram_budget_bytes = int(vram_budget_bytes)
        self.host_budget_bytes = int(host_budget_bytes)
        self._sizer = sizer
        self._slots: Dict[InstanceKey, _Slot] = {}
        self._state = threading.Condition(threading.Lock())
        #: Loads and promotions serialized per GPU — ruling clause 2.
        self._load_gate = threading.Lock()
        self._clock = 0

    # -- byte accounting (self._state held) ---------------------------------

    def _vram_reserved(self) -> int:
        return sum(
            s.vram_footprint
            for s in self._slots.values()
            if s.tier is Tier.VRAM or s.placing
        )

    def _host_reserved(self) -> int:
        return sum(s.weight_bytes for s in self._slots.values() if s.tier is Tier.HOST)

    # -- the one public operation -------------------------------------------

    def lease(
        self,
        checkpoint_ref: str,
        lane: str,
        factory: Callable[[], ModelBackend],
    ) -> Lease:
        """Admit, place and single-flight one request's model instance.

        Blocks while the fit depends only on in-flight work draining
        (progress, not a clock: every release re-evaluates). Raises
        :class:`NeverFits` at ADMISSION when weights + headroom exceed the
        whole budget — before ``factory`` runs, before any byte moves.
        """
        key: InstanceKey = (str(checkpoint_ref), str(lane))
        weight = int(self._sizer.resident_bytes(*key))
        headroom = int(self._sizer.activation_headroom_bytes(*key))
        if weight <= 0:
            raise ResidencyError(
                f"{key}: the lane manifest states {weight} weight bytes; "
                f"admission needs the real size"
            )
        if weight + headroom > self.vram_budget_bytes:
            raise NeverFits(
                f"{key}: needs {weight + headroom} bytes resident "
                f"({weight} weights + {headroom} activation headroom) and the "
                f"whole VRAM budget is {self.vram_budget_bytes}; no eviction "
                f"schedule can fit it — refuse at admission, never OOM mid-load"
            )

        with self._state:
            slot = self._slots.get(key)
            if slot is None:
                slot = _Slot(key=key, weight_bytes=weight, headroom_bytes=headroom)
                self._slots[key] = slot
            slot.inflight += 1  # from here on, no tier machinery touches it
            try:
                if slot.tier is not Tier.VRAM and not slot.placing:
                    # Admission: evict LRU until the reservation fits, then
                    # CLAIM it — atomically under the state lock, so
                    # concurrent admissions serialize on the same arithmetic.
                    self._make_room(slot)
                    slot.placing = True
            except BaseException:
                slot.inflight -= 1
                self._forget_if_dead(slot)
                self._state.notify_all()
                raise

        # Materialize outside the state lock, under the serialized load gate.
        try:
            with self._load_gate:
                with self._state:
                    tier = slot.tier
                if tier is not Tier.VRAM:
                    if tier is Tier.HOST:
                        slot.backend.promote_to_device()  # H2D, no disk walk
                    else:
                        slot.backend = factory()
                        slot.backend.load()
                    with self._state:
                        slot.tier = Tier.VRAM
                        slot.placing = False
        except BaseException:
            with self._state:
                slot.placing = False
                slot.inflight -= 1
                self._forget_if_dead(slot)
                self._state.notify_all()
            raise

        # Single-flight per instance: the request owns the slot until release.
        slot.gate.acquire()
        return Lease(self, slot)

    def _release(self, slot: _Slot) -> None:
        slot.gate.release()
        with self._state:
            slot.inflight -= 1
            self._clock += 1
            slot.last_used = self._clock
            # A release is the progress event every blocked admission waits on.
            self._state.notify_all()

    # -- placement (self._state held) ---------------------------------------

    def _make_room(self, incoming: _Slot) -> None:
        """Evict LRU until the incoming reservation fits.

        Tier moves touch IDLE slots only; when the only obstacle is
        in-flight work, block until a release re-evaluates the fit —
        single-flight defines drained, and drained instances are evictable.
        """
        need = incoming.vram_footprint
        while True:
            free = self.vram_budget_bytes - self._vram_reserved()
            if free >= need:
                return
            idle = sorted(
                (
                    s
                    for s in self._slots.values()
                    if s.tier is Tier.VRAM and s.inflight == 0 and not s.placing
                ),
                key=lambda s: s.last_used,
            )
            if idle:
                self._demote(idle[0])
                continue
            busy = sum(
                s.vram_footprint
                for s in self._slots.values()
                if (s.tier is Tier.VRAM or s.placing)
                and s.inflight > 0
                and s is not incoming
            )
            if busy == 0:
                raise ResidencyError(
                    f"{incoming.key}: needs {need} bytes, {free} free, and "
                    f"nothing resident is evictable or draining — the budget "
                    f"arithmetic is inconsistent"
                )
            self._state.wait()

    def _demote(self, slot: _Slot) -> None:
        """VRAM -> HOST (or straight to the chunk store when the host tier
        cannot take it). Idle slots only; host overflow drops host-LRU."""
        if self.host_budget_bytes >= slot.weight_bytes:
            while self._host_reserved() + slot.weight_bytes > self.host_budget_bytes:
                host_lru = min(
                    (s for s in self._slots.values() if s.tier is Tier.HOST),
                    key=lambda s: s.last_used,
                )
                self._drop(host_lru)
            slot.backend.demote_to_host()
            slot.tier = Tier.HOST
            logger.info(
                "residency: %s demoted to host (%d bytes)", slot.key, slot.weight_bytes
            )
        else:
            self._drop(slot)

    def _drop(self, slot: _Slot) -> None:
        slot.backend.drop()
        slot.backend = None
        slot.tier = Tier.ABSENT
        logger.info("residency: %s dropped to the chunk store", slot.key)

    def _forget_if_dead(self, slot: _Slot) -> None:
        if slot.tier is Tier.ABSENT and slot.inflight == 0 and slot.backend is None:
            self._slots.pop(slot.key, None)

    # -- read side ----------------------------------------------------------

    def tier_of(self, checkpoint_ref: str, lane: str) -> Tier:
        with self._state:
            slot = self._slots.get((str(checkpoint_ref), str(lane)))
            return slot.tier if slot is not None else Tier.ABSENT

    def reserved_bytes(self) -> Tuple[int, int]:
        """(vram reservations, host reservations) — the worker's own view."""
        with self._state:
            return self._vram_reserved(), self._host_reserved()


__all__ = [
    "InstanceKey",
    "InstanceSizer",
    "Lease",
    "ModelBackend",
    "NeverFits",
    "ResidencyError",
    "ResidencyManager",
    "Tier",
]
