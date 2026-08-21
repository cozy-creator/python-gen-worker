"""THE degrade ladder — one ordered Rung, one walk, one price; serve_fit.RUN_* (hub wire), memory's placement modes and the executor's load-rung strings are all PROJECTIONS of it. No rung quantizes at runtime: a quant rung is an AOT artifact the ladder SELECTS, never one the loader manufactures. off/vae_only/auto are not rungs — they are resident-placement flavors inside the native rung. Wire contract: ServePlan.ran/run_mode carry ONLY the RUN_MODES vocabulary, and tensorhub matches FnDegraded.ran EXACTLY — a decorated value like "offload:model_offload" silently misses the hub's arm, so placement detail travels in `reason`, never in `ran`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rung:
    """One step of the degrade ladder."""

    name: str
    placement: str
    storage: str
    run_mode: str
    latency: float
    touches_host_ram: bool
    admission_only: bool = False
    parks_named_components_only: bool = False


RUN_NATIVE = "native"
RUN_FP8_STORAGE = "fp8_storage"
RUN_OFFLOAD = "offload"
RUN_CPU = "cpu"

NATIVE = Rung("native", "", "", RUN_NATIVE, 1.0, False)
FP8_STORAGE = Rung("fp8_storage", "", "fp8", RUN_FP8_STORAGE, 1.05, False)
PARTIAL_RESIDENT = Rung(
    "partial_resident", "partial_resident", "", RUN_OFFLOAD, 1.3, True,
    admission_only=True, parks_named_components_only=True,
)
MODEL_OFFLOAD = Rung("model_offload", "model_offload", "", RUN_OFFLOAD, 2.5, True)
PARTIAL_STREAM = Rung(
    "partial_stream", "partial_stream", "", RUN_OFFLOAD, 2.8, True,
    admission_only=True,
)
GROUP_OFFLOAD = Rung("group_offload", "group_offload", "", RUN_OFFLOAD, 3.0, True)
SEQUENTIAL = Rung("sequential", "sequential", "", RUN_OFFLOAD, 4.0, True)
CPU = Rung("cpu", "cpu", "", RUN_CPU, 40.0, True)

LADDER: tuple[Rung, ...] = (
    NATIVE, FP8_STORAGE, PARTIAL_RESIDENT, MODEL_OFFLOAD, PARTIAL_STREAM,
    GROUP_OFFLOAD, SEQUENTIAL, CPU,
)

PLACEMENT_LADDER: tuple[str, ...] = tuple(
    r.name for r in LADDER if r.touches_host_ram
)

_BY_NAME = {r.name: r for r in LADDER}


FLOOR_LADDER_EXHAUSTED = "placement_ladder_exhausted"

def _walk(current: Optional[str]) -> tuple[Optional[Rung], Optional[str]]:
    cur = str(current or "")
    if cur == CPU.name:
        return None, FLOOR_LADDER_EXHAUSTED
    if cur not in PLACEMENT_LADDER:
        nxt: Optional[Rung] = MODEL_OFFLOAD
    else:
        idx = LADDER.index(_BY_NAME[cur]) + 1
        nxt = LADDER[idx] if idx < len(LADDER) else None
    if nxt is None:
        return None, FLOOR_LADDER_EXHAUSTED
    return nxt, None


def descend(current: Optional[str]) -> Optional[Rung]:
    """The next rung down from ``current``; None only when standing on ``cpu``."""
    return _walk(current)[0]


def descent_floor(current: Optional[str]) -> Optional[str]:
    """Why ``descend`` returned None — the typed floor, or None if it did not."""
    return _walk(current)[1]


def run_mode_of(name: Optional[str]) -> str:
    """The hub-wire ``RUN_*`` projection of a rung NAME ("" when not a rung)."""
    r = _BY_NAME.get(str(name or ""))
    return r.run_mode if r is not None else ""


def price(mode_or_rung: str) -> float:
    """Honest latency multiplier vs a native run."""
    exact = _BY_NAME.get(str(mode_or_rung or ""))
    if exact is not None:
        return exact.latency
    for r in LADDER:
        if r.run_mode == mode_or_rung and not r.admission_only:
            return r.latency
    return 1.0


def touches_host_ram(mode: Optional[str]) -> bool:
    return str(mode or "") in PLACEMENT_LADDER


def moves_every_component(mode: Optional[str]) -> bool:
    r = _BY_NAME.get(str(mode or ""))
    return bool(r and r.touches_host_ram and not r.parks_named_components_only)


def floor_of(a: Optional[str], b: Optional[str]) -> str:
    """The more-degraded of two placement tokens ('' / non-ladder = shallowest)."""
    def rank(m: Optional[str]) -> int:
        token = str(m or "")
        return PLACEMENT_LADDER.index(token) + 1 if token in PLACEMENT_LADDER else 0
    a_s, b_s = str(a or ""), str(b or "")
    return a_s if rank(a_s) >= rank(b_s) else b_s


def transition_line(
    *,
    event: str,
    fn: str = "",
    model: str = "",
    phase: str = "",
    from_rung: str = "",
    to_rung: str = "",
    needed_gb: float = 0.0,
    free_gb: float = 0.0,
    detail: str = "",
) -> str:
    parts = [f"DEGRADED_MODE={event}"]
    if fn:
        parts.append(f"fn={fn}")
    if model:
        parts.append(f"model={model}")
    if phase:
        parts.append(f"phase={phase}")
    if from_rung or to_rung:
        parts.append(f"rung={from_rung or '?'}->{to_rung or '?'}")
    if needed_gb > 0:
        parts.append(f"needed_gb={needed_gb:.1f}")
    if free_gb > 0:
        parts.append(f"free_gb={free_gb:.1f}")
    line = " ".join(parts)
    return f"{line}: {detail}" if detail else line
