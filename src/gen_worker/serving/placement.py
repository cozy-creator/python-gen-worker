"""Machine-floor fit at residency-admit: warn loudly, then run anyway.

Paul, 2026-08-18, verbatim: *"the way I want it to work is that we should be
able to place any model on any machine; if the machine is a poor match, it will
complain and warn loudly when it enters degraded mode, but still run anyway.
That way cozy-local users can still run these models, even if they run really
slow."*

So a declared floor INFORMS, it never PERMITS. The same number is read three
times, for three different purposes:

* **route-1 buying filter** — the hub filters placement on it, so we do not
  rent a pod that will serve this model badly;
* **route-2 warning** — a private deployment is told its own hardware is under
  the floor before it wonders why generation takes ten minutes;
* **worker degraded mode** — THIS module. The floor is unmet, the load happens
  anyway, and the fact is loud on both channels rather than inferred later
  from a latency graph.

This generalizes pgw#1312's offload ruling from "no env var gates CPU offload"
to "no declared floor gates anything either". There is no refusal here and no
environment variable to consult; the residency ladder does its job and the
operator is told what it is about to cost.

What this is NOT: a kernel check. Which attention kernel a serving picks
(sage2, flash-*) is a serve-recipe fallback arm with its own degrade path, not
a placement floor — see ``DTYPE_MIN_SM``'s header.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

#: Wire-shared activity kind (tensorhub
#: `internal/orchestrator/grpc/worker_activity.go`). One COMPLETED event per
#: unmet floor per residency-admit, so "how often did we serve degraded, and on
#: what" is a COUNT over `worker_activity_events`, not a log-scrape.
ACTIVITY_KIND = "placement_degraded"

_BYTES_PER_GIB = float(1024 ** 3)


@dataclass(frozen=True)
class DeviceFacts:
    """What the machine actually is, as the floors are spelled."""

    name: str
    vram_gib: float
    sm: int


@dataclass(frozen=True)
class Shortfall:
    """One declared floor this machine does not meet."""

    term: str
    lane: str
    declared: Any
    actual: Any
    device: str

    @property
    def message(self) -> str:
        if self.term == "min_vram_gb":
            return (
                f"DEGRADED PLACEMENT: {self.device} has {self.actual:.1f} GiB "
                f"of VRAM, but lane {self.lane} declares a floor of "
                f"{self.declared:.1f} GiB. Running anyway — the load asks the "
                f"offload ladder whether this pipeline fits BEFORE placing it "
                f"and engages a host-offload rung when it does not, naming "
                f"that rung in the log. Expect a HEAVY slowdown (minutes "
                f"where seconds are normal) rather than a failure; if no rung "
                f"is named, nothing was offloaded and it fit. "
                f"A declared floor is a buying hint and a warning, never "
                f"permission: any model runs on any machine."
            )
        if self.term == "min_sm":
            capability = (
                f"is compute capability sm_{self.actual}"
                if self.actual
                else "reports no CUDA compute capability at all"
            )
            return (
                f"DEGRADED PLACEMENT: {self.device} {capability}, but lane "
                f"{self.lane} needs sm_{self.declared}+ for its own load "
                f"dtype's arithmetic. "
                f"Running anyway — where torch has a fallback the kernels "
                f"will emulate or upcast and be MUCH slower; where it has "
                f"none this load will fail in the kernel rather than here. "
                f"A declared floor is a buying hint and a warning, never "
                f"permission: any model runs on any machine."
            )
        return (  # pragma: no cover - the vocabulary above is closed
            f"DEGRADED PLACEMENT: {self.device} does not meet {self.term} "
            f"{self.declared!r} declared by lane {self.lane} (actual "
            f"{self.actual!r}). Running anyway."
        )


def device_facts(index: int = 0) -> DeviceFacts:
    """This worker's actual card. Never raises — a machine that will not
    describe itself reads as the emptiest possible one, which makes every
    declared floor unmet and therefore LOUD. Silence is the one answer a
    placement report must never give.

    A CPU-only host (cozy-local) is exactly that case and is not special-cased:
    0 GiB of VRAM is under every floor, so it warns, and then it runs.
    """
    from ..hostfacts import cuda_ready

    try:
        # `cuda_ready()` is the ONE `torch.cuda.is_available()` site in `src/`
        # (fenced by tests/test_hostfacts_pgw896.py), and it is the right
        # question here: this is a CAPABILITY check — "may I read a card?" —
        # for which degrading on a device that will not answer is correct,
        # because an unreadable device is under every floor anyway.
        if not cuda_ready():
            return DeviceFacts(name="cpu (no CUDA device)", vram_gib=0.0, sm=0)
        import torch

        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        return DeviceFacts(
            name=str(properties.name),
            vram_gib=float(properties.total_memory) / _BYTES_PER_GIB,
            sm=int(major) * 10 + int(minor),
        )
    except Exception:  # noqa: BLE001
        logger.debug("device_facts: unreadable device", exc_info=True)
        return DeviceFacts(name="unknown device", vram_gib=0.0, sm=0)


def serving_device() -> str:
    """The device this worker serves on, PROBED — never assumed (pgw#1452).

    The host used to default to the literal string ``"cuda"``. That is an
    enumeration of what a pod usually is, not a measurement of what this
    machine is, and it reaches two places at once: ``engine_for(device=)`` on
    the streaming arm and ``_placed`` on the eager bridge. On a host with no
    card it turns a boot into a bare ``torch`` device error rather than a
    sentence naming the fallback.

    So it is asked the strong way. ``hostfacts.cuda_state`` is the
    probe-BY-ALLOCATION verdict (allocate, op, synchronize, free) rather than
    ``is_available()``, which is why a card that is present but will not answer
    reads as ``device_unreadable`` and takes the CPU arm with its probe class
    named — instead of failing on the first tensor with no explanation.

    The fallback is LOUD by design, exactly as ``derive._trace_device``'s is:
    cozy-local on a CPU-only box is a supported way to run (Paul, 2026-08-18:
    *"still run anyway"*), and serving on the wrong processor silently is the
    defect this replaces.
    """

    from ..hostfacts import cuda_state

    state = cuda_state()
    if state.present:
        return "cuda"
    logger.warning(
        "PLACEMENT: no usable CUDA device on this host (%s%s), so this "
        "endpoint loads and serves on the CPU. Every timing taken here is a "
        "CPU timing (pgw#1452).",
        state.state,
        f", {state.probe_class}: {state.detail}" if state.probe_class else "",
    )
    return "cpu"


def shortfalls(
    model_cls: type, lane: Any, *, facts: Optional[DeviceFacts] = None
) -> Tuple[Shortfall, ...]:
    """Every floor lane ``lane`` of ``model_cls`` declares that this machine
    does not meet. Empty for an undeclared floor, an eager-permanent model, or
    a machine that is simply big enough.
    """
    from .model import lane_handle, model_requires

    if lane is None:
        return ()  # eager-permanent: no lane, so no lane floor
    requirements = model_requires(model_cls).get(lane_handle(lane))
    if requirements is None:
        return ()
    terms = requirements.min_terms()
    if facts is None:
        facts = device_facts()

    out: list[Shortfall] = []
    # `min_sm` first: a card that cannot do the arithmetic at all is the more
    # fundamental complaint, and an operator reading two lines should see it
    # before the size one.
    if terms.min_sm and facts.sm < terms.min_sm:
        out.append(Shortfall("min_sm", lane_handle(lane), terms.min_sm,
                             facts.sm, facts.name))
    if terms.min_vram_gb and facts.vram_gib < terms.min_vram_gb:
        out.append(Shortfall("min_vram_gb", lane_handle(lane),
                             float(terms.min_vram_gb), facts.vram_gib,
                             facts.name))
    return tuple(out)


def warn_if_degraded(
    model_cls: type, lane: Any, *, facts: Optional[DeviceFacts] = None
) -> Tuple[str, ...]:
    """Report every unmet floor on BOTH channels, and return the messages so
    the caller can also make them caller-visible.

    Operator-visible: one COMPLETED ``placement_degraded`` activity event per
    unmet floor (countable, per-pod, on the channel every other worker fact
    already lands on) plus a WARNING on the logger, which on cozy-local IS the
    progress UI. Caller-visible: the returned messages, which the serve path
    hands to ``ctx.warn`` so they ride the response envelope's adjustment
    ledger on every request this instance serves.

    Never raises and never refuses — best-effort reporting around a load that
    is going to happen either way.
    """
    try:
        found = shortfalls(model_cls, lane, facts=facts)
    except Exception:  # noqa: BLE001
        logger.debug("placement floor check failed", exc_info=True)
        return ()
    messages: list[str] = []
    for shortfall in found:
        messages.append(shortfall.message)
        logger.warning("%s", shortfall.message)
        try:
            from ..activity import emit_event

            emit_event(
                ACTIVITY_KIND,
                shortfall.message,
                phase=shortfall.term,
                family=f"{model_cls.__name__}/{shortfall.lane}",
            )
        except Exception:  # noqa: BLE001
            logger.debug("placement_degraded event not emitted", exc_info=True)
    return tuple(messages)


__all__ = [
    "ACTIVITY_KIND",
    "DeviceFacts",
    "Shortfall",
    "device_facts",
    "serving_device",
    "shortfalls",
    "warn_if_degraded",
]
