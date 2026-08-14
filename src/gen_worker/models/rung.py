"""THE degrade ladder (pgw#1206 A2) — one ordered ``Rung``, one walk, one price.

Degrade-don't-OOM (Paul, 2026-07-10) used to span three vocabularies in five
files, joined by f-strings: plan-time ``serve_fit.RUN_*`` (hub wire), placement
modes in ``memory`` (``model_offload``/``group_offload``/``sequential``), and
bare ``"fp8"`` load-rung strings through the executor's bookkeepers. This
module is the single ordered ladder they all project from.

**No rung quantizes at runtime** (Paul, 2026-08-13, pgw#1206 D: *"We shouldn't
be doing runtime quants; if we're really memory-constrained then we should be
fetching the quant we need"*). The bnb-nf4 emergency rung sat between
``FP8_STORAGE`` and ``MODEL_OFFLOAD`` and is deleted: a quant rung is an AOT
artifact the ladder SELECTS, never one the loader manufactures. Deleting a
rung is one line here because A2 made this the only ordering.

``off``/``vae_only``/``auto`` are NOT rungs — they are resident-placement
flavors ``memory.select_auto_mode`` picks *inside* the native rung.

The wire contract: ``ServePlan.ran``/``run_mode`` carry ONLY the
``RUN_MODES`` vocabulary. tensorhub matches ``FnDegraded.ran`` EXACTLY
(``degradation_reschedule.go``: ``case "offload","cpu","emergency_quant"`` —
we no longer produce the third, and the hub keeps accepting it),
so a decorated value like ``offload:model_offload`` silently misses the
VRAM-driven-drain arm — placement detail travels in ``reason``, never in
``ran``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rung:
    """One step of the degrade ladder.

    ``placement`` is the ``memory`` placement mode this rung implies ("" for
    rungs that keep full residency); ``storage`` the runtime weight-storage
    transform ("" or "fp8"); ``run_mode`` the hub-wire projection
    (``serve_fit.RUN_*``); ``latency`` the honest price vs a native run;
    ``touches_host_ram`` whether weights become host-resident (the pgw#1063
    host-RAM pricing trigger; it was also the ``strict_vram`` opt-out boundary
    until th#1867 deleted that declaration)."""

    name: str
    placement: str
    storage: str
    run_mode: str
    latency: float
    touches_host_ram: bool


# Wire run modes (Go-mirrored: tensorhub profiling.RunMode). Values are the
# contract; tensorhub matches them EXACTLY.
RUN_NATIVE = "native"
RUN_FP8_STORAGE = "fp8_storage"
RUN_OFFLOAD = "offload"
RUN_CPU = "cpu"

NATIVE = Rung("native", "", "", RUN_NATIVE, 1.0, False)
FP8_STORAGE = Rung("fp8_storage", "", "fp8", RUN_FP8_STORAGE, 1.05, False)
MODEL_OFFLOAD = Rung("model_offload", "model_offload", "", RUN_OFFLOAD, 2.5, True)
GROUP_OFFLOAD = Rung("group_offload", "group_offload", "", RUN_OFFLOAD, 3.0, True)
SEQUENTIAL = Rung("sequential", "sequential", "", RUN_OFFLOAD, 4.0, True)
CPU = Rung("cpu", "cpu", "", RUN_CPU, 40.0, False)

#: The ONE ordering, best first. ``descend`` walks it; nothing else may.
LADDER: tuple[Rung, ...] = (
    NATIVE, FP8_STORAGE, MODEL_OFFLOAD, GROUP_OFFLOAD, SEQUENTIAL, CPU,
)

#: The reactive placement tail, shallowest first (gw#463): a load-time CUDA
#: OOM rolls back and retries one rung lower; a mid-inference OOM records the
#: next rung and applies it only during a clean reload.
PLACEMENT_LADDER: tuple[str, ...] = tuple(
    r.name for r in LADDER if r.touches_host_ram
)

_BY_NAME = {r.name: r for r in LADDER}


#: Stopped one rung above ``cpu``: it is declared in LADDER and priced, and
#: this build cannot execute it — the reactive walk treats it as plan-time
#: only (pgw#1212). Names OUR code, never the card.
FLOOR_CPU_RUNG_UNEXECUTABLE = "cpu_rung_unexecutable"

#: Standing on the last rung the ladder declares. Nothing is below it.
FLOOR_LADDER_EXHAUSTED = "placement_ladder_exhausted"

# th#1867 deleted FLOOR_STRICT_VRAM_TRUNCATED with the declaration that
# produced it. `Resources(strict_vram=True)` was the author asserting a card
# requirement in softer words (§2.4 ruling 4), and it truncated this walk
# before the first host-RAM-touching rung — turning "run slower" into "do not
# run". The walk now ends only where the LADDER ends, which is a fact about
# our code, not a declaration about a card.


def _walk(current: Optional[str]) -> tuple[Optional[Rung], Optional[str]]:
    """The ONE verdict :func:`descend` and :func:`descent_floor` both read.

    th#1867 §3.0: a diagnosis and an action that answer the same question from
    two implementations drift into disagreeing about what was said. Exactly one
    of the two returns is ever non-None.
    """
    cur = str(current or "")
    if cur == CPU.name:
        # The bottom rung itself. The walk must not climb back into the
        # placement tail, which the resident-token arm below would do.
        return None, FLOOR_LADDER_EXHAUSTED
    if cur not in PLACEMENT_LADDER:
        nxt: Optional[Rung] = MODEL_OFFLOAD
    else:
        idx = LADDER.index(_BY_NAME[cur]) + 1
        nxt = LADDER[idx] if idx < len(LADDER) else None
    if nxt is None:
        return None, FLOOR_LADDER_EXHAUSTED
    if nxt is CPU:
        return None, FLOOR_CPU_RUNG_UNEXECUTABLE
    return nxt, None


def descend(current: Optional[str]) -> Optional[Rung]:
    """The next rung down from ``current``; None when the ladder ends.

    The reactive OOM path descends only through the PLACEMENT tail (a storage
    transform cannot be applied to an already-loaded object), so from any
    resident token the next reactive rung is ``model_offload``.
    """
    return _walk(current)[0]


def descent_floor(current: Optional[str]) -> Optional[str]:
    """Why ``descend`` returned None — the typed floor, or None if it did not.

    th#1867: A DESCENT THAT RUNS OUT MUST NAME ITS FLOOR. The proactive fit
    ladder (``Resources.vram_gb_hint``, an ESTIMATE deciding placement before
    anything is measured — §4.33) is now DELETED, so this reactive walk is the
    only ladder and its bottom rung has stopped being theoretical. The failure
    mode that must not happen is the descent silently falling into a rung
    nothing can run: that converts a loud estimate-error into a quiet
    execution-error, which is the trade §1.35 and §1.36 keep rejecting.

    The refusal this feeds names OUR CODE ("this build cannot execute a CPU
    rung"), never the card — the legitimate species under §1.35's second
    amendment. It also makes pgw#1212's gap visible in production rather than
    latent, which is the fastest way it gets prioritised on evidence.
    """
    return _walk(current)[1]


def price(run_mode: str) -> float:
    """Honest latency multiplier vs a native run, by wire run mode. Coarse
    order-of-magnitude guidance (the hub's measured fit-matrix latency is
    authoritative when available); monotonic down the ladder."""
    for r in LADDER:
        if r.run_mode == run_mode:
            return r.latency
    return 1.0


def touches_host_ram(mode: Optional[str]) -> bool:
    """True when this placement token leaves weights RESIDENT IN HOST RAM
    (pgw#1063): that is what offloading IS — the whole tree lives on the host
    and streams to the card per forward, so host-RAM accounting must charge
    the full tree."""
    return str(mode or "") in PLACEMENT_LADDER


def floor_of(a: Optional[str], b: Optional[str]) -> str:
    """The more-degraded of two placement tokens ('' / non-ladder =
    shallowest). The learned per-ref floor (gw#463) only ever deepens."""
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
    """The ONE degraded-mode log format (gw#463; quality bar: ie#369's ltx
    DEGRADED_MODE lines). event: planned | engaged | serving | refused."""
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
