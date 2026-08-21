"""Model residency: admission BEFORE allocation, two tiers, serialized loads."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, Protocol, Tuple

logger = logging.getLogger(__name__)

InstanceKey = Tuple[str, str]


class ResidencyError(RuntimeError):
    """A residency decision could not be made; the message names the bytes."""


class NeverFits(ResidencyError):
    """Typed admission refusal: this instance exceeds the whole budget — no eviction schedule can ever make it fit."""


@dataclass(frozen=True)
class Charge:

    weight_bytes: int
    headroom_bytes: int
    basis: str

    @property
    def total(self) -> int:
        return self.weight_bytes + self.headroom_bytes


def admission_charge(weight_bytes: int, headroom_bytes: int) -> Charge:
    """The bytes to reserve for one instance: the tree's weights + headroom."""

    weights, headroom = int(weight_bytes), int(headroom_bytes)
    return Charge(
        weights,
        headroom,
        f"charged from the STORED TREE: {weights} weight bytes at the "
        f"checkpoint's on-disk precision + {headroom} activation headroom. "
        f"This is an UPPER BOUND (pgw#1599): a lane that holds less resident "
        f"— a setup()-time quantize, an offloaded or unused component — is "
        f"charged for bytes it never puts on the card, and the fix is a lane "
        f"whose CONTRACT states the precision the weights land at, never a "
        f"hand-written floor",
    )


class ModelBackend(Protocol):

    def load(self) -> None:
        """Chunk store -> VRAM."""
        ...

    def demote_to_host(self) -> None:
        """VRAM -> host-staged weights (eviction to the warm tier)."""
        ...

    def promote_to_device(self) -> None:
        """Host -> VRAM (an H2D copy; never a disk walk)."""
        ...

    def drop(self) -> None:
        """Release both tiers; the chunk store is the only remaining copy."""
        ...


class InstanceSizer(Protocol):
    """Byte facts, known AHEAD of any allocation."""

    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        """Weight bytes THIS LANE puts on the card, without loading."""
        ...

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        """The serving-time activation estimate (per model type / resolution class) reserved alongside the weights."""
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
    tier: Tier = Tier.ABSENT
    placing: bool = False
    last_used: int = 0
    inflight: int = 0
    gate: threading.Lock = field(default_factory=threading.Lock)

    @property
    def vram_footprint(self) -> int:
        return self.weight_bytes + self.headroom_bytes


class Lease:
    """One request's hold on a resident instance (single-flight)."""

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
    """The worker's placement authority for model instances on ONE device."""

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
        self._load_gate = threading.Lock()
        self._clock = 0
        self._confessed: set[InstanceKey] = set()

    def _confess(self, key: InstanceKey, charge: "Charge") -> None:
        if key in self._confessed:
            return
        self._confessed.add(key)
        logger.info(
            "ADMISSION %s: %d bytes against a %d byte budget — %s",
            key, charge.total, self.vram_budget_bytes, charge.basis,
        )

    def _vram_reserved(self) -> int:
        return sum(
            s.vram_footprint
            for s in self._slots.values()
            if s.tier is Tier.VRAM or s.placing
        )

    def _host_reserved(self) -> int:
        return sum(s.weight_bytes for s in self._slots.values() if s.tier is Tier.HOST)

    def lease(
        self,
        checkpoint_ref: str,
        lane: str,
        factory: Callable[[], ModelBackend],
    ) -> Lease:
        """Admit, place and single-flight one request's model instance."""
        key: InstanceKey = (str(checkpoint_ref), str(lane))
        weight = int(self._sizer.resident_bytes(*key))
        if weight <= 0:
            raise ResidencyError(
                f"{key}: the lane manifest states {weight} weight bytes; "
                f"admission needs the real size"
            )
        charge = admission_charge(
            weight, int(self._sizer.activation_headroom_bytes(*key))
        )
        weight, headroom = charge.weight_bytes, charge.headroom_bytes
        self._confess(key, charge)
        if charge.total > self.vram_budget_bytes:
            raise NeverFits(
                f"{key}: needs {charge.total} bytes resident "
                f"({weight} weights + {headroom} activation headroom) and the "
                f"whole VRAM budget is {self.vram_budget_bytes}; no eviction "
                f"schedule can fit it — refuse at admission, never OOM "
                f"mid-load. Basis: {charge.basis}"
            )

        with self._state:
            slot = self._slots.get(key)
            if slot is None:
                slot = _Slot(key=key, weight_bytes=weight, headroom_bytes=headroom)
                self._slots[key] = slot
            slot.inflight += 1
            try:
                if slot.tier is not Tier.VRAM and not slot.placing:
                    self._make_room(slot)
                    slot.placing = True
            except BaseException:
                slot.inflight -= 1
                self._forget_if_dead(slot)
                self._state.notify_all()
                raise

        try:
            with self._load_gate:
                with self._state:
                    tier = slot.tier
                if tier is not Tier.VRAM:
                    if tier is Tier.HOST:
                        slot.backend.promote_to_device()
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

        slot.gate.acquire()
        return Lease(self, slot)

    def _release(self, slot: _Slot) -> None:
        slot.gate.release()
        with self._state:
            slot.inflight -= 1
            self._clock += 1
            slot.last_used = self._clock
            self._state.notify_all()

    def _make_room(self, incoming: _Slot) -> None:
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

    def tier_of(self, checkpoint_ref: str, lane: str) -> Tier:
        with self._state:
            slot = self._slots.get((str(checkpoint_ref), str(lane)))
            return slot.tier if slot is not None else Tier.ABSENT

    def weight_budget_bytes(self, checkpoint_ref: str, lane: str) -> int:
        """The DEVICE bytes this instance's weights are admitted for."""
        with self._state:
            slot = self._slots.get((str(checkpoint_ref), str(lane)))
            if slot is not None and slot.weight_bytes > 0:
                return int(slot.weight_bytes)
        try:
            return int(self._sizer.resident_bytes(str(checkpoint_ref), str(lane)))
        except Exception:  # noqa: BLE001 — a sizer that cannot answer means
            logger.debug(
                "weight_budget_bytes(%r, %r) unreadable", checkpoint_ref, lane,
                exc_info=True,
            )
            return 0

    def reserved_bytes(self) -> Tuple[int, int]:
        """(vram reservations, host reservations) — the worker's own view."""
        with self._state:
            return self._vram_reserved(), self._host_reserved()


__all__ = [
    "Charge",
    "InstanceKey",
    "InstanceSizer",
    "Lease",
    "ModelBackend",
    "NeverFits",
    "ResidencyError",
    "ResidencyManager",
    "Tier",
    "admission_charge",
]
