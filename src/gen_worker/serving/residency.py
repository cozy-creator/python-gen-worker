"""Model residency: admission BEFORE allocation, two tiers, serialized loads.

Paul's pgw#1372 residency ruling (2026-08-18, verbatim framing: *"this should
respect whatever the wishes of the worker are... we don't want random OOMs
because 3 checkpoints were loaded at the same time"*):

1. **Admission before allocation.** Placement is the WORKER's decision —
   author code is device-neutral (no ``.to()``/``torch_dtype`` in the
   contract file). Before constructing an instance, its weight byte size
   plus an activation-headroom estimate is reserved against the VRAM budget;
   LRU residents are evicted until it fits. A model that can NEVER fit
   refuses typed at admission — a CUDA OOM mid-load or at first request is a
   design bug, not bad luck.

   pgw#1590 AMENDS WHERE THAT SIZE COMES FROM. This clause used to say "its
   EXACT weight byte size (the tensorfs manifest per lane)". The manifest is
   not exact and is not per lane: it is the whole tree at its STORED
   precision, an upper bound on residency that cannot see a setup()-time
   `quantize_()` or an offloaded component. So the lane's declared VRAM floor
   caps the charge when it is smaller — see :func:`admission_charge`.
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


@dataclass(frozen=True)
class Charge:
    """What admission bills one instance for, and WHY (pgw#1590).

    ``basis`` is not decoration: it is the sentence the refusal and the
    confession both carry, so an operator reading either can tell a number
    derived from the tree from a number the author declared.
    """

    weight_bytes: int
    headroom_bytes: int
    basis: str

    @property
    def total(self) -> int:
        return self.weight_bytes + self.headroom_bytes


def admission_charge(
    tree_weight_bytes: int, tree_headroom_bytes: int, declared_vram_bytes: int
) -> Charge:
    """The bytes to reserve for one instance: the smaller of what the LANE
    DECLARES it needs of a card and what the STORED TREE implies.

    pgw#1590. The tree-derived number is an upper bound on residency and
    nothing more. It reads a checkpoint's on-disk precision and cannot see
    what the lane's load path does with those bytes — a ``setup()``-time
    ``quantize_()`` to w8a8, a text encoder that is offloaded and never lands
    on the card, a component the lane does not touch at all. minimax-h3 is all
    three at once: a 133 GB bf16 DiT that serves as ~66 GB of fp8, inside a
    144 GB repo of which the DiT lane loads one part. Charged from the tree it
    "needs" 180 GB and no card exists; charged from its own declaration it
    needs 78 GiB, which is the card the hub already bought for it, and which
    it demonstrably served on.

    So a declared floor CAPS the charge, and only ever downward:

    * **it can never make admission stricter.** ``min`` — a lane whose floor is
      generous relative to its tree keeps the tree number, so no lane that is
      admitted today is refused tomorrow.
    * **it is not optimism.** The floor is a DECLARATION in the author's own
      class header (``lanes={contract: "vram78g"}``), statically extractable,
      and already the number the hub filters placement on. Charging more than
      it is the incoherent position: it says "the platform placed me on a card
      sized by this floor, and I refuse to run because I believe I need more
      than the floor". An endpoint whose floor is below its true residency is
      broken at PLACEMENT, before admission ever sees it, and that failure has
      a named owner and a loud warning path (``placement.warn_if_degraded``).
    * **an undeclared floor changes nothing.** No declaration, no cap, and the
      conservative whole-tree charge stands. A guess is never allowed to be the
      optimistic input — an OOM on a rented card is worse than a refusal.

    A capped charge carries its whole footprint in ``weight_bytes`` with zero
    headroom, because that is what the declaration says: ``vram78g`` is what
    the lane needs OF A CARD, activations included, not a weights subtotal to
    add 25% to.
    """

    tree = Charge(
        int(tree_weight_bytes),
        int(tree_headroom_bytes),
        f"charged from the STORED TREE: {int(tree_weight_bytes)} weight bytes at "
        f"the checkpoint's on-disk precision + {int(tree_headroom_bytes)} "
        f"activation headroom",
    )
    declared = int(declared_vram_bytes)
    if declared <= 0:
        return Charge(
            tree.weight_bytes,
            tree.headroom_bytes,
            tree.basis
            + ". This lane DECLARES NO VRAM FLOOR, so admission has nothing "
            "but the tree to go on and charges all of it. If the lane's load "
            "path holds less than the tree resident — a setup()-time "
            "quantize, an offloaded or unused component — declare it in the "
            "class header (`lanes={contract: \"vramNNg\"}`) and admission "
            "charges the declaration instead",
        )
    if declared >= tree.total:
        return Charge(
            tree.weight_bytes,
            tree.headroom_bytes,
            tree.basis
            + f". The lane's declared floor ({declared} bytes) is not smaller, "
            f"so it does not cap this charge",
        )
    return Charge(
        declared,
        0,
        f"charged from the LANE'S OWN DECLARATION: {declared} bytes "
        f"(`lanes={{contract: \"vram...g\"}}`), the card this lane states it "
        f"needs — activations included. The stored tree implies {tree.total} "
        f"bytes ({tree.weight_bytes} + {tree.headroom_bytes} headroom), an "
        f"on-disk-precision upper bound this lane's load path does not hold "
        f"resident",
    )


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
        """Weight bytes for (checkpoint, lane) from the tensorfs manifest,
        without loading.

        AN UPPER BOUND, not an exact resident cost (pgw#1590 — this docstring
        used to say "exact" and the production sizer answers with the whole
        tree at its stored precision). A manifest cannot see a setup()-time
        `quantize_()`, an offloaded component, or a component the lane never
        touches; only the lane's own declaration can, and
        :func:`admission_charge` is where the two meet."""
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
        #: Keys whose admission basis has already been said out loud. Once per
        #: instance, not once per request — the basis cannot change between
        #: two requests for the same key.
        self._confessed: set[InstanceKey] = set()

    def _confess(self, key: InstanceKey, charge: "Charge") -> None:
        """Say what admission is about to decide FROM, before it decides.

        pgw#1586's rule, applied to the rung above the ladder: a decision that
        does not state the numbers it saw is a decision nobody can audit later
        — which is exactly how pgw#1590's refusal read as a hardware fact
        instead of a sizing bug for six days.
        """
        if key in self._confessed:
            return
        self._confessed.add(key)
        logger.info(
            "ADMISSION %s: %d bytes against a %d byte budget — %s",
            key, charge.total, self.vram_budget_bytes, charge.basis,
        )

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
        *,
        declared_vram_bytes: int = 0,
    ) -> Lease:
        """Admit, place and single-flight one request's model instance.

        Blocks while the fit depends only on in-flight work draining
        (progress, not a clock: every release re-evaluates). Raises
        :class:`NeverFits` at ADMISSION when the charge exceeds the whole
        budget — before ``factory`` runs, before any byte moves.

        ``declared_vram_bytes`` is the lane's own floor declaration
        (``placement.declared_vram_bytes``), passed by the caller that holds
        the model class. It CAPS the sizer's tree-derived charge and can never
        raise it — see :func:`admission_charge` for why the declaration is the
        more trustworthy of the two numbers.
        """
        key: InstanceKey = (str(checkpoint_ref), str(lane))
        tree_weight = int(self._sizer.resident_bytes(*key))
        if tree_weight <= 0:
            raise ResidencyError(
                f"{key}: the lane manifest states {tree_weight} weight bytes; "
                f"admission needs the real size"
            )
        charge = admission_charge(
            tree_weight,
            int(self._sizer.activation_headroom_bytes(*key)),
            int(declared_vram_bytes),
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

    def weight_budget_bytes(self, checkpoint_ref: str, lane: str) -> int:
        """The DEVICE bytes this instance's weights are admitted for.

        pgw#1497's admission-first seam, and the reason it is a method here
        rather than a number the loader works out: this is the same figure
        ``lease`` reserves against the budget before anything is allocated, so
        the rung that sizes a resident set from it can never disagree with the
        reservation that let the instance in. It is available BEFORE the
        factory runs, which is when the load context that carries it is built.

        pgw#1590: THE ADMITTED CHARGE, not the sizer's raw tree number. Once a
        declared lane floor can cap the reservation, re-asking the sizer here
        would hand the streaming rung a budget the manager never reserved —
        the exact disagreement this method exists to make impossible. The slot
        is already in the ledger by the time a load context is built; the
        sizer is only the answer before any lease has run.
        """
        with self._state:
            slot = self._slots.get((str(checkpoint_ref), str(lane)))
            if slot is not None and slot.weight_bytes > 0:
                return int(slot.weight_bytes)
        try:
            return int(self._sizer.resident_bytes(str(checkpoint_ref), str(lane)))
        except Exception:  # noqa: BLE001 — a sizer that cannot answer means
            # "no lease number", and the rung refuses on 0 rather than
            # inventing one. Never a failed load.
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
