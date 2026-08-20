"""THE degrade ladder (pgw#1206 A2) — one ordered ``Rung``, one walk, one price.

Degrade-don't-OOM (Paul, 2026-07-10) as ONE ordered ladder. Plan-time
``serve_fit.RUN_*`` (hub wire), ``memory``'s placement modes and the load-rung
strings the executor's bookkeepers carry are all PROJECTIONS of it, never three
vocabularies joined by f-strings.

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
    #: pgw#1497. True for a rung only the ADMISSION path can select, because
    #: its budget comes from a residency lease. Neither ``select_auto_mode``
    #: (a proactive decider, no lease) nor the reactive OOM descent (no lease
    #: at the moment it fires) may produce one, and a run-mode-only
    #: :func:`price` does not describe one — a caller standing on such a rung
    #: passes its NAME.
    admission_only: bool = False
    #: pgw#1587. True when the rung moves only the components it NAMED at
    #: admission and leaves every other one device-resident for the life of
    #: the load. This is the fact `ctx.compile` needs and `touches_host_ram`
    #: cannot answer: a compiled graph binds its constants to device pointers,
    #: so it may be armed on a target whose weights do not move — but not on
    #: one accelerate onloads and frees per forward. `touches_host_ram` is
    #: about ACCOUNTING (host RAM is charged); this is about STABILITY.
    parks_named_components_only: bool = False


# Wire run modes (Go-mirrored: tensorhub profiling.RunMode). Values are the
# contract; tensorhub matches them EXACTLY.
RUN_NATIVE = "native"
RUN_FP8_STORAGE = "fp8_storage"
RUN_OFFLOAD = "offload"
RUN_CPU = "cpu"

NATIVE = Rung("native", "", "", RUN_NATIVE, 1.0, False)
FP8_STORAGE = Rung("fp8_storage", "", "fp8", RUN_FP8_STORAGE, 1.05, False)
# pgw#1577 — COMPONENT-granular residency: evict the minimum-BYTE subset of
# components that clears the budget and keep everything else, denoiser first,
# on the card between requests. It sits ABOVE `model_offload` because it is
# strictly cheaper on the same placement decision: same fit test, a subset of
# the traffic. Measured cause on the campaign card (SDXL, RTX 4070 Laptop):
# `model_offload` moves 13 GiB per request — the whole pipeline out and back —
# to reclaim 1.2 GiB, at ~2.7 GB/s effective, which is ~4.8 s of the ~7.0 s
# fixed per-request cost. This rung moves 1.64 GiB, one way, from pinned host
# RAM, because a weight the host still holds needs no copy back.
#
# ADMISSION-ONLY, for the same reason `partial_stream` is: the resident set is
# chosen from free VRAM and component sizes at LOAD, and the reactive OOM
# descent has neither in hand when it fires. `_walk` therefore sends any
# resident token to `model_offload`, exactly as before, and `price(RUN_OFFLOAD)`
# keeps answering with `model_offload`'s coarse number rather than this rung's.
#
# `parks_named_components_only`: the plan NAMES its offload set and forces the
# denoiser resident, so every component outside that set keeps a stable device
# pointer for the life of the load. That is what makes this the one offload
# rung a COMPILED graph can be armed under (pgw#1587, Paul: *"For SDXL in
# particular, we need to offload the text encoders to free up room for the
# Unet, and then it works, during inference. This doesn't conflict with
# compilation however because [we] are only running the compiled UNet."*).
PARTIAL_RESIDENT = Rung(
    "partial_resident", "partial_resident", "", RUN_OFFLOAD, 1.3, True,
    admission_only=True, parks_named_components_only=True,
)
MODEL_OFFLOAD = Rung("model_offload", "model_offload", "", RUN_OFFLOAD, 2.5, True)
# pgw#1497 — per-LEAF-MODULE budgeted residency, the tail cast per forward from
# pinned host RAM (`models.stream_residency`). The only rung with a BUDGET: the
# others move a whole component (model_offload), every leaf unconditionally
# (sequential) or a fixed group (group_offload), and none of them can be asked
# for a number.
#
# ITS POSITION IS MEASURED, AND IT IS NOT WHERE THE ISSUE PREDICTED.
# pgw#1497 specified it "between fp8_storage and model_offload". On the card
# that is false. sd1.5, 512^2, 25 steps, CFG, fp16, eager, one config, RTX
# 4070, best of 2 timed runs after a warmup:
#
#   rung                        ms/step   x native   peak VRAM
#   resident                      119.6      1.00      2.81 GB
#   model_offload                 187.2      1.57      1.88 GB
#   partial_stream @50% budget    228.9      1.91      1.89 GB
#   partial_stream @25% budget    377.7      3.16      1.35 GB
#   partial_stream @5%  budget    426.4      3.56      1.04 GB
#   group_offload                 550.2      4.60      0.82 GB
#   sequential                    903.6      7.55      0.65 GB
#
# At EQUAL peak VRAM (1.89 vs 1.88 GB) model_offload is faster — 1.57x against
# 1.91x — because its offload tax is per-CALL (measured: 1.69 s per generation,
# fixed) while streaming is per-STEP. So this rung does not belong above it.
# What it does that model_offload cannot is go LOWER: model_offload is
# whole-component granular and bottoms out near 1.9 GB here, and this rung
# keeps serving down to 1.04 GB at 3.56x — still 1.3x faster than
# group_offload and 2.1x faster than sequential, the only other rungs that
# reach that floor. Hence: below model_offload, above group_offload.
#
# The PRICE is that interval read off the measurements. At the 25% budget —
# the regime it is actually selected in, a budget model_offload cannot meet —
# it sits 52% of the way from model_offload to group_offload on the measured
# scale ((3.16-1.57)/(4.60-1.57)), which maps onto the declared [2.5, 3.0]
# interval as 2.76. Filed as 2.8.
#
# CAVEAT, recorded because it is the honest reading of the same run: the
# ladder's OTHER declared prices are stale against this card (model_offload
# 2.5 declared / 1.57 measured, group_offload 3.0 / 4.60, sequential 4.0 /
# 7.55). The ORDER is right and the magnitudes are not. Re-deriving them is a
# separate issue; interpolating into the declared interval keeps this rung
# consistent with its neighbours rather than correct against a scale nothing
# else uses.
#
# ADMISSION-FIRST, its defining constraint: the budget is the residency
# lease's, never an activation estimate. `select_auto_mode` NEVER returns it —
# a proactive decider has no lease to read — and `apply_low_vram_config`
# REFUSES it without an explicit `stream_budget_bytes`. The mechanism ported
# here is the one ComfyUI drives from hand-fitted per-architecture activation
# lambdas; taking the mechanism without the estimator is the whole point. The
# reactive descent does not enter it either (`_walk` sends any resident token
# to `model_offload`): a load-time OOM has no lease in hand when it fires.
PARTIAL_STREAM = Rung(
    "partial_stream", "partial_stream", "", RUN_OFFLOAD, 2.8, True,
    admission_only=True,
)
GROUP_OFFLOAD = Rung("group_offload", "group_offload", "", RUN_OFFLOAD, 3.0, True)
SEQUENTIAL = Rung("sequential", "sequential", "", RUN_OFFLOAD, 4.0, True)
# touches_host_ram: a CPU-placed pipeline keeps its WHOLE tree in host RAM —
# that IS the rung. Declaring otherwise handed a CPU-rung load the pgw#1063
# per-component staging discount, which is an admission lie by the same
# arithmetic that made ie#615 a certain kernel OOM.
CPU = Rung("cpu", "cpu", "", RUN_CPU, 40.0, True)

#: The ONE ordering, best first. ``descend`` walks it; nothing else may.
LADDER: tuple[Rung, ...] = (
    NATIVE, FP8_STORAGE, PARTIAL_RESIDENT, MODEL_OFFLOAD, PARTIAL_STREAM,
    GROUP_OFFLOAD, SEQUENTIAL, CPU,
)

#: The reactive placement tail, shallowest first (gw#463): a load-time CUDA
#: OOM rolls back and retries one rung lower; a mid-inference OOM records the
#: next rung and applies it only during a clean reload. It ends at ``cpu``
#: (pgw#1315) — the tail is every rung whose weights live on the host, and the
#: bottom one is where the always-runs guarantee is actually kept.
PLACEMENT_LADDER: tuple[str, ...] = tuple(
    r.name for r in LADDER if r.touches_host_ram
)

_BY_NAME = {r.name: r for r in LADDER}


# pgw#1315: there is NO floor above ``cpu``. A reactive walk that stops one
# rung short makes the always-runs guarantee true at plan time and false on a
# descent that reaches the bottom; the fix is to execute the rung
# (``memory.apply_low_vram_config`` mode ``cpu``), not to soften the wording.
# A floor whose cause is gone must not be left pointing somewhere else.

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
    return nxt, None


def descend(current: Optional[str]) -> Optional[Rung]:
    """The next rung down from ``current``; None only when standing on ``cpu``.

    The reactive OOM path descends only through the PLACEMENT tail (a storage
    transform cannot be applied to an already-loaded object), so from any
    resident token the next reactive rung is ``model_offload`` and the walk ends
    at ``cpu``, which serves.
    """
    return _walk(current)[0]


def descent_floor(current: Optional[str]) -> Optional[str]:
    """Why ``descend`` returned None — the typed floor, or None if it did not.

    th#1867: A DESCENT THAT RUNS OUT MUST NAME ITS FLOOR. The proactive fit
    ladder (``Resources.vram_gb_hint``, an ESTIMATE deciding placement before
    anything is measured — §4.33) is now DELETED, so this reactive walk is the
    only ladder.

    pgw#1315 left exactly ONE floor. A descent now runs to ``cpu`` and serves
    there, so the only way out of rungs is to be standing on the bottom one
    already — a fact about where we are, not a refusal to go further. Nothing
    here may ever again report that a declared rung cannot be executed: the
    answer to that is to execute it.
    """
    return _walk(current)[1]


def run_mode_of(name: Optional[str]) -> str:
    """The hub-wire ``RUN_*`` projection of a rung NAME ("" when not a rung).

    tensorhub matches ``FnDegraded.ran`` against its RunMode vocabulary
    EXACTLY, and ``cpu`` is a member of it in its own right. Reporting a
    descent onto the CPU rung under the offload tail's token is therefore a
    wrong measurement, not a coarse one. Read the rung's own projection; never
    assume the tail's (pgw#1315).
    """
    r = _BY_NAME.get(str(name or ""))
    return r.run_mode if r is not None else ""


def price(mode_or_rung: str) -> float:
    """Honest latency multiplier vs a native run.

    Takes either a RUNG NAME — the exact price of the rung the caller is
    standing on — or a wire run mode, which is coarse order-of-magnitude
    guidance (the hub's measured fit-matrix latency is authoritative when
    available); monotonic down the ladder.

    The run-mode scan skips ``admission_only`` rungs. Three rungs now project
    onto ``RUN_OFFLOAD`` and they span the ladder, so the run-mode answer has
    to name which one it describes: it describes the shallowest rung a
    PROACTIVE walk can land on, because that is the only kind of walk a caller
    holding nothing but a run mode has taken. A caller who knows it is on
    ``partial_stream`` passes that name and gets that rung's own number.
    """
    exact = _BY_NAME.get(str(mode_or_rung or ""))
    if exact is not None:
        return exact.latency
    for r in LADDER:
        if r.run_mode == mode_or_rung and not r.admission_only:
            return r.latency
    return 1.0


def touches_host_ram(mode: Optional[str]) -> bool:
    """True when this placement token leaves weights RESIDENT IN HOST RAM
    (pgw#1063): that is what offloading IS — the whole tree lives on the host
    and streams to the card per forward, so host-RAM accounting must charge
    the full tree."""
    return str(mode or "") in PLACEMENT_LADDER


def moves_every_component(mode: Optional[str]) -> bool:
    """True when this placement token moves EVERY component's weights between
    host and device per forward (pgw#1587).

    The question `ctx.compile` has to ask, and the one ``touches_host_ram``
    was standing in for. Under such a rung no compiled graph can be armed on
    anything, because the device pointers it binds are freed after each
    forward. Under a rung that parks only the components it named, the answer
    is per-TARGET and the plan holds it — see
    :func:`gen_worker.models.partial_resident.parks_module`.
    """
    r = _BY_NAME.get(str(mode or ""))
    return bool(r and r.touches_host_ram and not r.parks_named_components_only)


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
