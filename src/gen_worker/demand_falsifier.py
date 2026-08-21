"""Predicted-vs-measured, banked every serve. THE FALSIFIER, AND IT SHIPS FIRST.

pgw#1600 §3 / pgw#1598 §3. A demand formula is a claim about a card, and a
claim nothing checks is a number that looks computed. So the instrument lands
BEFORE any consumer: every serve evaluates the lane's declared formula at the
request's own shape, measures what the request actually cost, and banks the
pair. `demand_miss` (measured > predicted) is a COUNTED, hub-visible event, per
(lane x regime).

**This module decides nothing.** It is not consulted by admission, by
placement, or by the oom ladder, and `tests/test_demand_no_enforcement_pgw1600.py`
asserts that absence mechanically rather than in prose — pgw#1600 acceptance
(d). Enforcement is pgw#1601's (the mint-time stamp) and pgw#1602's (the
grant); wiring this number into either is a different issue's commit.

## Why two measured numbers, and which one the verdict uses

The request arena is spent in two places and only one of them is the torch
allocator (pgw#1598 §1, upstream-cited at `aoti_runtime/model_base.h:345`):

* ``allocated`` — the allocator's peak over the handler, minus what was
  already allocated when the handler took the card. What EAGER activations
  come out of.
* ``out_of_allocator`` — the CUDA context, cuDNN/cuBLAS workspaces, and
  AOTI's own ``cudaMalloc``'d first-call pool. Invisible to
  ``max_memory_allocated`` and unavailable to the caching allocator.

tcg#80's sm_89 acceptance run is the reference decomposition: at denoise, sdxl
compiled UNet-only measured 4907 MiB allocated + 1155 MiB out-of-allocator =
6649 MiB at the driver, on a COLD-DAEMON basis. So the regime picks the basis:
eager is judged on ``allocated``; compiled is judged on the sum, because that
is the budget a compiled admit actually has.

## Regime, and the rule that outranks convenience

Samples NEVER pool across regimes or lanes (pgw#1586, pgw#1600). An unknown
regime pools with nothing. On the compiled path a miss is a P0 defect of the
stamp and is emitted as one, never as a statistic (pgw#1601 acceptance (d)).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import msgspec

from . import activity as activity_mod
from .demand import Basis, Demand, RequestShape

logger = logging.getLogger(__name__)

__all__ = [
    "KIND_DEMAND_MISS",
    "Banked",
    "DemandObservation",
    "MeasuredArena",
    "bank",
    "banked",
    "measure_request_arena",
    "observe",
    "reset_banked",
]

KIND_DEMAND_MISS = "demand_miss"

_MIB = 1 << 20


class MeasuredArena(msgspec.Struct, frozen=True, kw_only=True):
    """What one request actually cost, decomposed the way it is SPENT."""

    #: The torch allocator's PEAK over the handler, minus what was already
    #: allocated when it took the card. What eager activations come out of.
    allocated_bytes: int = 0
    #: Growth in what the process holds on the card OUTSIDE the caching
    #: allocator: CUDA context, cuDNN/cuBLAS workspaces, AOTI's own
    #: ``cudaMalloc``'d pool.
    out_of_allocator_bytes: int = 0
    #: Growth in ``total - driver_free`` across the handler. Includes
    #: allocator CACHE, which driver-free counts as gone and which a COMPILED
    #: call cannot spend at all (pgw#1627) — so it belongs in the compiled
    #: basis and not in the eager one.
    driver_growth_bytes: int = 0
    #: False when there was no card to read (CPU-only test, torch-free
    #: install). An unmeasured request banks NOTHING — a zero measurement
    #: would read as "predicted generously" and quietly prove the formula
    #: right.
    measured: bool = False

    @property
    def driver_bytes(self) -> int:
        """The compiled basis, and it is a LOWER BOUND on the true peak.

        Two views of the same window, neither complete on its own: the
        reconstructed peak (``allocated`` peak + out-of-allocator growth)
        misses cache the allocator took and did not give back; the driver
        growth misses a transient peak that was freed before the closing
        sample. The larger of the two is the strongest bound available
        without continuous sampling, which is what tcg#80 had to do on the
        card and what a serve path cannot afford per request.
        """

        return max(
            self.allocated_bytes + self.out_of_allocator_bytes,
            self.driver_growth_bytes,
        )

    def for_regime(self, regime: str) -> int:
        return self.driver_bytes if regime == "compiled" else self.allocated_bytes


class DemandObservation(msgspec.Struct, frozen=True, kw_only=True):
    """One serve's predicted-vs-measured pair, with everything to re-derive it."""

    lane: str
    regime: str
    shape: RequestShape
    predicted_bytes: int
    basis: Basis
    measured: MeasuredArena

    @property
    def measured_bytes(self) -> int:
        return self.measured.for_regime(self.regime)

    @property
    def is_miss(self) -> bool:
        return self.measured.measured and self.measured_bytes > self.predicted_bytes

    @property
    def key(self) -> tuple[str, str]:
        return (self.lane, self.regime)


class Banked(msgspec.Struct, frozen=True, kw_only=True):
    """The counted state for one (lane x regime). Hub-visible via the event."""

    lane: str
    regime: str
    served: int = 0
    misses: int = 0
    worst_miss_bytes: int = 0
    worst_predicted_bytes: int = 0
    worst_measured_bytes: int = 0
    basis: Basis = Basis.UNCALIBRATED

    def as_document(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "regime": self.regime,
            "served": self.served,
            "demand_miss": self.misses,
            "worst_miss_bytes": self.worst_miss_bytes,
            "worst_predicted_bytes": self.worst_predicted_bytes,
            "worst_measured_bytes": self.worst_measured_bytes,
            "basis": str(self.basis),
        }


_lock = threading.Lock()
_banked: dict[tuple[str, str], Banked] = {}


def banked() -> tuple[Banked, ...]:
    """Every counted (lane x regime), in a stable order."""

    with _lock:
        return tuple(_banked[key] for key in sorted(_banked))


def reset_banked() -> None:
    """Test hook."""

    with _lock:
        _banked.clear()


def bank(observation: DemandObservation) -> Banked:
    """Record one serve. Emits `demand_miss` when the formula was WRONG LOW."""

    with _lock:
        held = _banked.get(observation.key) or Banked(
            lane=observation.lane, regime=observation.regime,
        )
        miss = observation.is_miss
        over = (
            observation.measured_bytes - observation.predicted_bytes if miss else 0
        )
        updated = Banked(
            lane=held.lane,
            regime=held.regime,
            served=held.served + (1 if observation.measured.measured else 0),
            misses=held.misses + (1 if miss else 0),
            worst_miss_bytes=max(held.worst_miss_bytes, over),
            worst_predicted_bytes=(
                observation.predicted_bytes if over > held.worst_miss_bytes
                else held.worst_predicted_bytes
            ),
            worst_measured_bytes=(
                observation.measured_bytes if over > held.worst_miss_bytes
                else held.worst_measured_bytes
            ),
            basis=observation.basis,
        )
        _banked[observation.key] = updated
    if miss:
        _confess(observation, updated)
    return updated


def _confess(observation: DemandObservation, state: Banked) -> None:
    """One typed, countable event per miss.

    ``step``/``total_steps`` carry (misses, served) so the hub can read a RATE
    off the event alone — the pgw#1597-plan-3 leak-rate shape — without
    joining anything. ``phase`` carries the key, so a hub reader groups by
    (lane x regime) without parsing the sentence.
    """

    severity = "p0_stamp_defect" if observation.regime == "compiled" else "statistic"
    detail = (
        f"lane={observation.lane} regime={observation.regime} "
        f"predicted={observation.predicted_bytes / _MIB:.0f} MiB "
        f"measured={observation.measured_bytes / _MIB:.0f} MiB "
        f"(allocated={observation.measured.allocated_bytes / _MIB:.0f} + "
        f"out_of_allocator="
        f"{observation.measured.out_of_allocator_bytes / _MIB:.0f}) "
        f"over={(observation.measured_bytes - observation.predicted_bytes) / _MIB:.0f} MiB "
        f"shape={observation.shape.as_document()} "
        f"coefficient_basis={observation.basis} severity={severity} "
        f"misses={state.misses}/{state.served}"
    )
    if observation.regime == "compiled":
        # pgw#1601 acceptance (d): on the compiled path admission is the ONLY
        # safety mechanism, so a stamp the card disagreed with is a defect of
        # the stamp — not a sample to average away.
        logger.error("demand_miss (P0, compiled): %s", detail)
    else:
        logger.warning("demand_miss: %s", detail)
    activity_mod.emit_event(
        KIND_DEMAND_MISS,
        detail,
        phase=f"{observation.lane}|{observation.regime}|{severity}",
        step=state.misses,
        total_steps=state.served,
    )


def _arena_now() -> Optional[tuple[int, int, int]]:
    """``(allocated, reserved, total - driver_free)``, or None with no card."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        driver_free, total = torch.cuda.mem_get_info()
        return allocated, reserved, int(total) - int(driver_free)
    except Exception:  # noqa: BLE001 — a falsifier must never fail a request
        return None


@contextmanager
def measure_request_arena() -> Iterator[list[MeasuredArena]]:
    """Measure one handler's request arena, decomposed. Never raises.

    Yields a one-element list the caller reads AFTER the block. The
    allocator peak is reset on entry so the number is THIS request's, and the
    out-of-allocator term is the growth in what the process holds on the card
    outside the caching allocator — where AOTI's first-call pool lands.
    """

    out: list[MeasuredArena] = [MeasuredArena()]
    before = _arena_now()
    if before is not None:
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            before = None
    try:
        yield out
    finally:
        after = _arena_now()
        if before is not None and after is not None:
            try:
                import torch

                peak = int(torch.cuda.max_memory_allocated())
            except Exception:  # noqa: BLE001
                peak = after[0]
            allocated_before, reserved_before, held_before = before
            _, reserved_after, held_after = after
            outside_before = max(0, held_before - reserved_before)
            outside_after = max(0, held_after - reserved_after)
            out[0] = MeasuredArena(
                allocated_bytes=max(0, peak - allocated_before),
                out_of_allocator_bytes=max(0, outside_after - outside_before),
                driver_growth_bytes=max(0, held_after - held_before),
                measured=True,
            )


def observe(
    *,
    lane: str,
    regime: str,
    demand: Optional[Demand],
    shape: RequestShape,
    measured: MeasuredArena,
) -> Optional[DemandObservation]:
    """Build and bank one observation. Returns None when there is nothing to
    falsify (no declared formula, or nothing measured)."""

    if demand is None or not measured.measured:
        return None
    try:
        predicted = demand.evaluate(shape)
    except Exception:  # noqa: BLE001 — an out-of-domain shape is not a failed request
        logger.debug("demand falsifier: shape out of domain", exc_info=True)
        return None
    observation = DemandObservation(
        lane=lane or "unknown",
        regime=regime if regime in ("eager", "compiled") else f"unknown({regime})",
        shape=shape,
        predicted_bytes=predicted,
        basis=demand.weakest_basis(),
        measured=measured,
    )
    bank(observation)
    return observation
