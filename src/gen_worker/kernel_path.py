"""Measured serving-kernel lane selection.

The svdq serving path has TWO independent kernel choices, and they cannot be
one switch:

    linear      W4A4 matmuls: ``fused`` (our triton kernels) or ``baseline``
                (the open unfused chain). A THROUGHPUT question.
    modulation  W4A16 AdaLN modulation: ``packed`` (4-bit resident,
                dequantised in-kernel) or ``dense`` (decoded to bf16 at
                load). A RESIDENCY question — 22.8 -> 13.3 GB on B200.

Which COMBINATION wins on a card is a per-card FACT, not a derivable one: a
custom op is opaque to inductor, so on sm_120 our fusion beats what inductor
can do with the open chain and on sm_100 it loses to it by 19%, while the
packed modulation is worth having on both. Deliberately NOT a hand-maintained
SM tuple per axis: a single switch cannot express "baseline linears, packed
modulation", so sm_100 would have to give up either 9.5 GB of residency or
19% of its step time.

The verdict is a measurement taken on the card the cell is being minted for,
recorded INTO the cell, and adopted by serving:

    mint      probe(candidates, ...) -> Verdict     (every buildable
              combination built, compiled, and timed on the target card, in
              the mint process)
    envelope  metadata.json["kernel_lane"] — the DISCRETE verdict, plus each
              candidate's QUANTIZED peak and the fallback order
    result    metadata["kernel_lane_evidence"] — the numbers, published with
              the checkpoint, never packed (the double-mint byte-compare
              forbids wall clocks inside the artifact)
    serve     adopt(meta) -> re-apply the fit rule on THIS card -> pin(); the
              load-time swap reads the pin, each axis projecting its own
              value out of it

CELL KEYS ARE KEYED ON SM, AND THE LANE IS DELIBERATELY NOT A KEY AXIS (that
would fork the namespace and halve reuse), so one SM class is many cards (a
96 GB RTX PRO 6000 and a 32 GB RTX 5090 share a key) and a recorded verdict is
EVIDENCE, not an instruction: serving RE-APPLIES the fit constraint against its
own detected total before it adopts, and falls to the fastest candidate that
does fit here with a typed reason. That is why each candidate's peak rides the
packed envelope beside the winner — bytes are discrete, wall clocks are not, so
the fit half of the rule can be re-applied by a worker that will never see the
timings.

A lane is therefore a COMBINATION, written ``"<linear>+<modulation>"``
(``"baseline+packed"``, ``"fused+dense"``, ...). Measuring the combinations
rather than each axis alone assumes no independence between them, and is what
lets ONE rule price a residency win and a throughput win against each other
instead of hard-coding which one matters.

THE SELECTION RULE: **fit-constrained speed maximization.** Among lanes whose
measured peak VRAM fits the target card with headroom, pick the FASTEST. VRAM is
a CONSTRAINT, not an objective — it breaks a tie only when two lanes are within
the speed noise margin. The packed modulation is speed-NEUTRAL, so it wins its
axis on exactly that tiebreak, the smaller peak. On a 32 GB card the fit
constraint does real work and can exclude a faster lane outright.
"""

from __future__ import annotations

import logging
import time
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple)

import msgspec

from . import artifact_meta

logger = logging.getLogger(__name__)

# --- vocabulary -------------------------------------------------------------

# The two axes. Each is decided independently at serving time (a card can want
# baseline linears AND packed modulation — sm_100 does), but they are MEASURED
# together, as combinations, so the rule never has to assume they do not
# interact.
AXIS_LINEAR = "linear"
AXIS_MODULATION = "modulation"
AXES = (AXIS_LINEAR, AXIS_MODULATION)

LINEAR_BASELINE = "baseline"
LINEAR_FUSED = "fused"
LINEAR_EXECUTION_LANES = (LINEAR_BASELINE, LINEAR_FUSED)

MOD_DENSE = "dense"
MOD_PACKED = "packed"
MOD_EXECUTION_LANES = (MOD_DENSE, MOD_PACKED)

# A lane NAME is the combination, so one string pins both axes, one verdict
# ranks them together, and one grep finds a decision in a log.
SEP = "+"


def execution_lane_of(linear: str, modulation: str) -> str:
    """The combination's canonical name."""
    return f"{linear}{SEP}{modulation}"


def split_execution_lane(execution_lane: str) -> Tuple[str, str]:
    """``(linear, modulation)`` for a combination name."""
    linear, _, modulation = str(execution_lane).partition(SEP)
    return linear, modulation


def linear_of(execution_lane: str) -> str:
    """The linear-axis value a combination names."""
    return split_execution_lane(execution_lane)[0]


def modulation_of(execution_lane: str) -> str:
    """The modulation-axis value a combination names."""
    return split_execution_lane(execution_lane)[1]


EXECUTION_LANES = tuple(
    execution_lane_of(linear, modulation)
    for linear in LINEAR_EXECUTION_LANES for modulation in MOD_EXECUTION_LANES)

# What a worker runs when no cell says otherwise: the pair that exists on
# every card and is a pessimisation on none of them.
DEFAULT_EXECUTION_LANE = execution_lane_of(LINEAR_BASELINE, MOD_DENSE)

# Envelope key (discrete facts, packed into metadata.json) and result key
# (measurements, published with the checkpoint, NEVER packed).
META_KEY = "kernel_lane"
EVIDENCE_KEY = "kernel_lane_evidence"
SCHEMA = 1

# --- the policy constants ---------------------------------------------------

# A candidate must beat the incumbent by this much to be declared the SPEED
# winner. Below it the two are a tie and the smaller peak wins, so ordinary
# measurement noise can never flip a recorded verdict between two mints on
# one card. `_gemm_profitable` uses 1.10x on a 4096^3 microbenchmark; a
# whole-graph step timed over `BENCH_ITERS` medians is far quieter, and every
# decision this mechanism exists to make is 7-34% wide.
MARGIN_FRACTION = 0.05

# "Fits" = the measured peak, plus an allowance for the shapes the mint did
# NOT measure. The mint times ONE representative shape; a tenant may ask for a
# larger resolution or a longer prompt, and the allocator fragments. The
# strict_vram rule stands (declare the honest peak ask — a marketing-GB
# declaration is release-broken by one GiB), so the allowance is explicit and
# the term that bound a verdict is recorded with it.
ACTIVATION_SPIKE_FRACTION = 0.20
FRAGMENTATION_HEADROOM_BYTES = 1 << 30  # 1 GiB

# Peaks recorded in the PACKED envelope are quantized (rounded UP). A raw
# `max_memory_allocated()` is a MEASUREMENT, and an unrounded measurement in
# metadata.json is precisely what the double-mint byte-compare forbids —
# an autotuned kernel picking a different workspace on the second mint would
# move the number and strand the cell. Rounding to a coarse grain makes the
# recorded byte count reproducible for the same reason `MARGIN_FRACTION`
# makes the winner reproducible, and rounding UP can only make the fit test
# stricter, never optimistic (the strict_vram rule).
PEAK_QUANTUM_BYTES = 1 << 28  # 256 MiB

# Fixed measurement protocol — part of the verdict's determinism, recorded
# with it so a re-measurement is comparable.
BENCH_WARMUP = 3
BENCH_ITERS = 9  # odd: a true median, no interpolation

# Capability floor for the fused lane: block-scaled MMA (`dot_scaled` lowering
# to kind::mxf4nvf4 / tcgen05) is Blackwell silicon. This is a CAPABILITY
# precondition — can the candidate be built at all — not a performance
# allowlist. Whether it WINS on a qualifying card is what gets measured.
FUSED_MIN_SM = 100

# Why a lane is what it is, when no verdict decided it. Typed so a serving
# worker's degrade is greppable and never a silent fall-through.
REASON_ABSENT = "kernel_lane_verdict_absent"
REASON_UNREADABLE = "kernel_lane_verdict_unreadable"
REASON_UNKNOWN_EXECUTION_LANE = "kernel_lane_verdict_unknown_lane"
REASON_NO_CELL = "kernel_lane_no_cell"
REASON_ADOPTED = "kernel_lane_verdict_adopted"
# The recorded winner does not fit THIS card (same SM class, smaller card):
# the rule was re-applied locally and a different lane was pinned.
REASON_REFIT_LOCAL = "kernel_lane_refit_local"
# Nothing the cell measured fits this card. The smallest peak is pinned and
# says so — obeying a verdict into an OOM is not conservatism.
REASON_REFIT_NO_FIT = "kernel_lane_refit_no_fit"
# The fit could not be re-applied here (no per-candidate peaks recorded, or
# no detectable device total). Adopted, and marked as unverified across cards.
REASON_FIT_UNVERIFIED = "kernel_lane_fit_unverified"

# Which term bound a verdict.
BIND_SPEED = "speed"
BIND_FIT = "fit"
BIND_VRAM_TIEBREAK = "vram_tiebreak"
BIND_SOLE_CANDIDATE = "sole_candidate"
BIND_NO_FIT = "no_fit"
#: The mint held no weights, so no whole-model A/B was run.
BIND_UNMEASURED = "unmeasured"


class ExecutionLaneProbeError(RuntimeError):
    """A candidate could not be built or measured. Never fatal: the candidate
    drops out of the ranking with its reason recorded."""


# --- value types ------------------------------------------------------------


def required_for(peak_bytes: int) -> int:
    """A measured peak's ASK: the peak plus the stated allowance. One
    formula, used by the mint and re-used verbatim by every serving worker
    that re-applies the constraint on its own card."""
    return int(int(peak_bytes or 0) * (1.0 + ACTIVATION_SPIKE_FRACTION)) \
        + FRAGMENTATION_HEADROOM_BYTES


def fits_bytes(peak_bytes: int, device_total_bytes: int) -> bool:
    """Does a peak fit a card with the stated allowance? A total of 0 is
    "unknown" and constrains nothing."""
    if not device_total_bytes:
        return True
    return required_for(peak_bytes) <= int(device_total_bytes)


def quantize_peak(peak_bytes: int) -> int:
    """A peak rounded UP to ``PEAK_QUANTUM_BYTES`` — the form that may ride
    the packed envelope without moving between two mints."""
    n = int(peak_bytes or 0)
    if n <= 0:
        return 0
    return -(-n // PEAK_QUANTUM_BYTES) * PEAK_QUANTUM_BYTES


class Measurement(msgspec.Struct, frozen=True, kw_only=True):
    """One candidate lane, measured on the target card."""

    execution_lane: str
    ms_per_step: float = 0.0
    peak_bytes: int = 0
    samples_ms: Tuple[float, ...] = ()
    unavailable: str = ""  # typed reason it could not be measured

    @property
    def usable(self) -> bool:
        return not self.unavailable and self.ms_per_step > 0.0

    def required_bytes(self) -> int:
        """The peak plus the stated allowance for the shapes the mint did not
        measure (activation spikes, resolution variance, fragmentation)."""
        return required_for(self.peak_bytes)


class Verdict(msgspec.Struct, frozen=True, kw_only=True):
    """Which lane won on this card, why, and on what evidence."""

    winner: str
    binding: str
    rule: str = "fit_constrained_speed"
    schema: int = SCHEMA
    margin_fraction: float = MARGIN_FRACTION
    activation_spike_fraction: float = ACTIVATION_SPIKE_FRACTION
    fragmentation_headroom_bytes: int = FRAGMENTATION_HEADROOM_BYTES
    device_total_bytes: int = 0
    device_name: str = ""
    sm: str = ""
    warmup: int = BENCH_WARMUP
    iters: int = BENCH_ITERS
    measured_at: float = 0.0
    measurements: Tuple[Measurement, ...] = ()
    detail: str = ""

    def measurement(self, execution_lane: str) -> Optional[Measurement]:
        for row in self.measurements:
            if row.execution_lane == execution_lane:
                return row
        return None


# --- the rule ---------------------------------------------------------------


def fits(row: Measurement, device_total_bytes: int) -> bool:
    """Does this lane fit the card with the stated allowance? A card whose
    total is unknown (0) constrains nothing — the mint measured it there, so
    it ran there."""
    return fits_bytes(row.peak_bytes, device_total_bytes)


def select(
    measurements: Sequence[Measurement],
    *,
    device_total_bytes: int = 0,
    device_name: str = "",
    sm: str = "",
) -> Verdict:
    """FIT-CONSTRAINED SPEED MAXIMIZATION.

    1. Drop candidates that could not be measured.
    2. Drop candidates whose measured peak + allowance does not fit the card.
    3. Among the rest take the FASTEST; a rival within ``MARGIN_FRACTION`` of
       it is a TIE and the smaller peak wins.

    Nothing fits, or nothing measured -> the smallest-peak usable candidate
    with ``binding="no_fit"`` (loud, and the serving degrade path owns what
    happens next), or the declared default with no measurements at all.
    """
    rows = tuple(measurements)
    usable = tuple(r for r in rows if r.usable)
    common: Dict[str, Any] = {
        "device_total_bytes": int(device_total_bytes or 0),
        "device_name": str(device_name or ""),
        "sm": str(sm or ""),
        "measurements": rows,
        "measured_at": time.time(),
    }
    if not usable:
        gaps = "; ".join(
            f"{r.execution_lane}: {r.unavailable or 'no timing'}" for r in rows)
        return Verdict(
            winner=DEFAULT_EXECUTION_LANE, binding=BIND_NO_FIT,
            detail=f"no candidate measured ({gaps or 'no candidates'})",
            **common)

    fitting = tuple(r for r in usable if fits(r, device_total_bytes))
    if not fitting:
        smallest = min(usable, key=lambda r: (r.peak_bytes, r.execution_lane))
        return Verdict(
            winner=smallest.execution_lane, binding=BIND_NO_FIT,
            detail=(
                f"no candidate fits {device_total_bytes} B with the stated "
                f"allowance; smallest peak wins ({smallest.execution_lane}, "
                f"{smallest.peak_bytes} B peak, "
                f"{smallest.required_bytes()} B required)"),
            **common)

    excluded = tuple(r.execution_lane for r in usable if r not in fitting)
    if len(fitting) == 1:
        only = fitting[0]
        binding = BIND_FIT if excluded else BIND_SOLE_CANDIDATE
        detail = (
            f"{only.execution_lane} is the only lane that fits; excluded "
            f"{sorted(excluded)!r}" if excluded
            else f"{only.execution_lane} is the only candidate")
        return Verdict(winner=only.execution_lane, binding=binding, detail=detail,
                       **common)

    ranked = sorted(fitting, key=lambda r: (r.ms_per_step, r.execution_lane))
    best, rival = ranked[0], ranked[1]
    gap = (rival.ms_per_step - best.ms_per_step) / max(best.ms_per_step, 1e-9)
    if gap >= MARGIN_FRACTION:
        return Verdict(
            winner=best.execution_lane, binding=BIND_SPEED,
            detail=(
                f"{best.execution_lane} {best.ms_per_step:.1f} ms/step beats "
                f"{rival.execution_lane} {rival.ms_per_step:.1f} by {gap * 100:.1f}% "
                f"(margin {MARGIN_FRACTION * 100:.0f}%)"
                + (f"; excluded on fit {sorted(excluded)!r}" if excluded
                   else "")),
            **common)
    # Within the noise margin: VRAM breaks the tie, and only here.
    tied = tuple(
        r for r in fitting
        if (r.ms_per_step - best.ms_per_step)
        / max(best.ms_per_step, 1e-9) < MARGIN_FRACTION)
    winner = min(tied, key=lambda r: (r.peak_bytes, r.execution_lane))
    return Verdict(
        winner=winner.execution_lane, binding=BIND_VRAM_TIEBREAK,
        detail=(
            f"{[r.execution_lane for r in tied]!r} within {MARGIN_FRACTION * 100:.0f}% "
            f"({best.ms_per_step:.1f}-{rival.ms_per_step:.1f} ms/step); "
            f"smaller peak wins ({winner.execution_lane}, {winner.peak_bytes} B)"),
        **common)


def unmeasured(execution_lane: str, detail: str) -> Verdict:
    """The verdict for a mint that COULD have benchmarked and deliberately did
    not.

    The lane A/B is a whole-model benchmark: it loads one full pipeline per
    candidate and times a real step, so it needs weight-scale residency —
    which is exactly the property a structure-only mint exists to keep. A
    cell with no verdict is the documented conservative-default case on the
    serving side, and below Blackwell the A/B has no candidates to compare
    anyway (`fused_candidate_gap`), so this costs nothing that has a consumer
    today. Recorded as a TYPED verdict with its reason rather than as an
    absence, so the choice is one lookup instead of a re-derivation.
    """
    total, name, sm = device_facts()
    return Verdict(
        winner=execution_lane, binding=BIND_UNMEASURED, detail=detail,
        device_total_bytes=total, device_name=name, sm=sm,
        measured_at=time.time())


def sole(execution_lane: str, detail: str) -> Verdict:
    """The verdict for a card that can only build ONE lane. No benchmark is
    run: with nothing to compare against, a measurement would buy a compile
    and decide nothing. The cell still records a real, typed verdict rather
    than the absence a serving worker has to guess about."""
    total, name, sm = device_facts()
    return Verdict(
        winner=execution_lane, binding=BIND_SOLE_CANDIDATE, detail=detail,
        device_total_bytes=total, device_name=name, sm=sm,
        measured_at=time.time())


# --- measurement ------------------------------------------------------------


def device_facts() -> Tuple[int, str, str]:
    """``(total_bytes, name, sm)`` for the card this process is on, honestly
    detected (an H100-80GB reports 79.19 GiB — declare that, not the
    marketing number). ``(0, "", "")`` off-GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, "", ""
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        major, minor = torch.cuda.get_device_capability()
        return int(props.total_memory), str(props.name), f"sm_{major}{minor}"
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return 0, "", ""


def measure(execution_lane: str, step: Callable[[], Any]) -> Measurement:
    """Time one candidate's representative step and record its device peak.

    ``step`` must run ONE forward of the graph in its production posture
    (compiled), already built. Fixed warmup + median over ``BENCH_ITERS``
    CUDA-event timings — the same protocol `_gemm_profitable` uses, so two
    mints on one card measure the same thing.
    """
    import torch

    try:
        for _ in range(BENCH_WARMUP):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(BENCH_ITERS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            samples.append(round(float(start.elapsed_time(end)), 3))
        peak = int(torch.cuda.max_memory_allocated())
    except Exception as exc:  # noqa: BLE001 — a candidate that cannot be
        # measured drops out of the ranking; it never fails the mint.
        return Measurement(
            execution_lane=execution_lane, unavailable=f"{type(exc).__name__}: {exc}")
    ordered = sorted(samples)
    return Measurement(
        execution_lane=execution_lane, ms_per_step=ordered[len(ordered) // 2], peak_bytes=peak,
        samples_ms=tuple(samples))


def probe(
    candidates: Sequence[str],
    build: Callable[[str], Callable[[], Any]],
    *,
    device_total_bytes: int = 0,
    device_name: str = "",
    sm: str = "",
) -> Verdict:
    """Build, measure and rank every candidate lane on THIS card.

    ``build(lane)`` returns a zero-argument callable that runs one
    representative step with that lane armed — the mint's job, because only
    the loader can swap the linears. A builder that raises drops its
    candidate with the reason recorded; it never fails the mint.
    """
    if not device_total_bytes and not device_name and not sm:
        device_total_bytes, device_name, sm = device_facts()
    rows = []
    for execution_lane in candidates:
        t0 = time.monotonic()
        try:
            step = build(execution_lane)
        except Exception as exc:  # noqa: BLE001
            logger.warning("kernel-lane probe: %s could not be built — %s",
                           execution_lane, exc)
            rows.append(Measurement(
                execution_lane=execution_lane, unavailable=f"build: {type(exc).__name__}: {exc}"))
            continue
        row = measure(execution_lane, step)
        rows.append(row)
        logger.info(
            "kernel-lane probe: %s -> %s (%.1f s to build+measure)",
            execution_lane,
            (f"{row.ms_per_step:.1f} ms/step, {row.peak_bytes / 1e9:.1f} GB "
             f"peak" if row.usable else f"UNAVAILABLE {row.unavailable}"),
            time.monotonic() - t0)
    verdict = select(
        rows, device_total_bytes=device_total_bytes,
        device_name=device_name, sm=sm)
    logger.info("kernel-lane verdict: %s (%s) — %s",
                verdict.winner, verdict.binding, verdict.detail)
    return verdict


def fused_candidate_gap(sm: int) -> str:
    """Why the fused LINEAR is not even a CANDIDATE here, or ''.

    Capability only — silicon plus the numerics self-check. Whether it is
    the faster lane on a qualifying card is what :func:`probe` measures, and
    is deliberately NOT asked here.
    """
    if sm < FUSED_MIN_SM:
        return (f"fused svdq kernels need Blackwell block-scaled MMA "
                f"(sm_{FUSED_MIN_SM}+); this GPU is sm_{sm}")
    try:
        from .models.svdq_fused import fused_self_check
    except Exception as exc:  # noqa: BLE001
        return f"fused kernels are not importable: {exc}"
    try:
        return str(fused_self_check() or "")
    except Exception as exc:  # noqa: BLE001
        return f"fused self-check raised: {type(exc).__name__}: {exc}"


def packed_candidate_gap() -> str:
    """Why the PACKED modulation is not a candidate here, or ''.

    No SM term at all, deliberately: the W4A16 dequant-GEMM is ordinary
    triton with no block-scaled-MMA dependency, so it builds on any CUDA card
    triton supports. The measurement decides the rest.
    """
    try:
        from .models.svdq_awq_packed import awq_packed_self_check
    except Exception as exc:  # noqa: BLE001
        return f"packed modulation kernels are not importable: {exc}"
    try:
        return str(awq_packed_self_check() or "")
    except Exception as exc:  # noqa: BLE001
        return (f"packed modulation self-check raised: "
                f"{type(exc).__name__}: {exc}")


def candidate_axes() -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, str]]:
    """``({axis: buildable values}, {axis: why the armed value is not one})``.

    The gaps are CAPABILITY answers only. An axis with one value needs no
    measurement on that axis; an axis with two is what :func:`probe` prices.
    """
    linear = [LINEAR_BASELINE]
    modulation = [MOD_DENSE]
    gaps: Dict[str, str] = {}
    total, _name, sm_text = device_facts()
    if not total:
        gaps[AXIS_LINEAR] = gaps[AXIS_MODULATION] = (
            "no CUDA device — native kernels cannot be built here")
        return ({AXIS_LINEAR: tuple(linear),
                 AXIS_MODULATION: tuple(modulation)}, gaps)
    try:
        sm = int(str(sm_text or "sm_0").removeprefix("sm_"))
    except ValueError:
        sm = 0
    gap = fused_candidate_gap(sm)
    if gap:
        gaps[AXIS_LINEAR] = gap
        logger.info("kernel-lane: the fused linear is not a candidate "
                    "here — %s", gap)
    else:
        linear.append(LINEAR_FUSED)
    gap = packed_candidate_gap()
    if gap:
        gaps[AXIS_MODULATION] = gap
        logger.info("kernel-lane: the packed modulation is not a candidate "
                    "here — %s", gap)
    else:
        modulation.append(MOD_PACKED)
    return ({AXIS_LINEAR: tuple(linear),
             AXIS_MODULATION: tuple(modulation)}, gaps)


def candidates_here() -> Tuple[str, ...]:
    """Every lane COMBINATION that can be BUILT on this card, cheapest-first.

    The cross product, not one axis at a time: measuring the pairs assumes
    nothing about whether the two kernels interact, and it is what lets the
    one rule in :func:`select` weigh a residency win against a throughput
    win. On a card where neither armed value can be built this is exactly one
    combination and the mint skips the benchmark entirely
    (``mint_child.lane_verdict_for``), so the cost lands only where there is
    actually a choice to make.
    """
    axes, _gaps = candidate_axes()
    return tuple(
        execution_lane_of(linear, modulation)
        for linear in axes[AXIS_LINEAR]
        for modulation in axes[AXIS_MODULATION])


# --- recording and reading --------------------------------------------------


def refit_order(
    measurements: Sequence[Measurement], *, winner: str = "",
) -> Tuple[str, ...]:
    """The lanes best-first, as a DISCRETE fact a second mint reproduces.

    A serving worker that has to drop the recorded winner needs to know what
    to fall to, and it has no timings — so the ORDER travels instead. It is
    discretized exactly the way the winner is: lanes within
    ``MARGIN_FRACTION`` of the fastest remaining lane are one tie class,
    ordered inside the class by quantized peak then name. Jitter therefore
    has to cross the margin to move the order, which is the same bar that
    already has to be cleared to move the winner.

    The recorded ``winner`` is forced to the front so the order and the
    verdict can never disagree about what won on the minting card.
    """
    usable = [r for r in measurements if r.usable]
    ranked: List[str] = []
    remaining = sorted(usable, key=lambda r: (r.ms_per_step, r.execution_lane))
    while remaining:
        head = remaining[0]
        tie = [
            r for r in remaining
            if (r.ms_per_step - head.ms_per_step)
            / max(head.ms_per_step, 1e-9) < MARGIN_FRACTION
        ]
        ranked.extend(
            r.execution_lane for r in sorted(
                tie, key=lambda r: (quantize_peak(r.peak_bytes), r.execution_lane)))
        remaining = [r for r in remaining if r not in tie]
    if winner in ranked:
        ranked = [winner] + [execution_lane for execution_lane in ranked if execution_lane != winner]
    return tuple(ranked)


def fit_block(verdict: Verdict) -> Dict[str, Any]:
    """Everything a DIFFERENT card of the same SM class needs to re-apply the
    fit half of the rule: each measured candidate's quantized peak, and the
    order to fall through when the recorded winner does not fit locally.

    All discrete, all reproducible, no wall clocks — so it can ride the
    packed envelope. ``{}`` when nothing was measured (a ``sole_candidate``
    verdict), which is the honest signal that the fit cannot be re-applied.
    """
    peaks = {
        r.execution_lane: quantize_peak(r.peak_bytes)
        for r in sorted(verdict.measurements, key=lambda r: r.execution_lane)
        if r.usable
    }
    if not peaks:
        return {}
    return {
        "quantum_bytes": PEAK_QUANTUM_BYTES,
        "activation_spike_fraction": float(verdict.activation_spike_fraction),
        "fragmentation_headroom_bytes":
            int(verdict.fragmentation_headroom_bytes),
        "peaks": peaks,
        "order": list(refit_order(verdict.measurements,
                                  winner=verdict.winner)),
    }


def envelope_block(verdict: Verdict) -> Dict[str, Any]:
    """The DISCRETE verdict, for ``metadata.json`` inside the packed cell.

    Deliberately carries no wall clocks: the double-mint byte-compare requires
    the artifact to be reproducible, and milliseconds are not. Peak
    BYTES are a different kind of number — discrete, quantized here, and
    load-bearing for a card that shares this cell's SM but not its memory —
    so they ride the envelope while the timings stay in
    :func:`evidence_block`, which rides the published checkpoint metadata
    beside ``mint_phases``.
    """
    block = {
        "schema": int(verdict.schema),
        "winner": str(verdict.winner),
        "rule": str(verdict.rule),
        "binding": str(verdict.binding),
        "margin_fraction": float(verdict.margin_fraction),
        "candidates": sorted(r.execution_lane for r in verdict.measurements),
    }
    fit = fit_block(verdict)
    if fit:
        block["fit"] = fit
    return block


def evidence_block(verdict: Verdict) -> Dict[str, Any]:
    """Every number behind the verdict — published with the checkpoint, never
    packed. This is what makes a verdict auditable and a flip explicable."""
    return {
        "schema": int(verdict.schema),
        "winner": str(verdict.winner),
        "rule": str(verdict.rule),
        "binding": str(verdict.binding),
        "detail": str(verdict.detail),
        "margin_fraction": float(verdict.margin_fraction),
        "activation_spike_fraction": float(verdict.activation_spike_fraction),
        "fragmentation_headroom_bytes":
            int(verdict.fragmentation_headroom_bytes),
        "device_total_bytes": int(verdict.device_total_bytes),
        "device_name": str(verdict.device_name),
        "sm": str(verdict.sm),
        "warmup": int(verdict.warmup),
        "iters": int(verdict.iters),
        "measured_at": float(verdict.measured_at),
        "candidates": [
            {
                "lane": r.execution_lane,
                "ms_per_step": float(r.ms_per_step),
                "peak_bytes": int(r.peak_bytes),
                "required_bytes": int(r.required_bytes()),
                "fits": fits(r, verdict.device_total_bytes),
                "samples_ms": list(r.samples_ms),
                "unavailable": r.unavailable,
            }
            for r in verdict.measurements
        ],
    }


def verdict_from_evidence(block: Mapping[str, Any]) -> Verdict:
    """Rebuild a :class:`Verdict` from :func:`evidence_block` — the audit
    round-trip, and how a recorded campaign is replayed against the rule."""
    rows = tuple(
        Measurement(
            execution_lane=str(c.get("lane") or ""),
            ms_per_step=float(c.get("ms_per_step") or 0.0),
            peak_bytes=int(c.get("peak_bytes") or 0),
            samples_ms=tuple(float(s) for s in (c.get("samples_ms") or ())),
            unavailable=str(c.get("unavailable") or ""),
        )
        for c in (block.get("candidates") or ())
    )
    return Verdict(
        winner=str(block.get("winner") or DEFAULT_EXECUTION_LANE),
        binding=str(block.get("binding") or ""),
        rule=str(block.get("rule") or "fit_constrained_speed"),
        schema=int(block.get("schema") or SCHEMA),
        margin_fraction=float(block.get("margin_fraction") or MARGIN_FRACTION),
        activation_spike_fraction=float(
            block.get("activation_spike_fraction")
            or ACTIVATION_SPIKE_FRACTION),
        fragmentation_headroom_bytes=int(
            block.get("fragmentation_headroom_bytes")
            or FRAGMENTATION_HEADROOM_BYTES),
        device_total_bytes=int(block.get("device_total_bytes") or 0),
        device_name=str(block.get("device_name") or ""),
        sm=str(block.get("sm") or ""),
        warmup=int(block.get("warmup") or BENCH_WARMUP),
        iters=int(block.get("iters") or BENCH_ITERS),
        measured_at=float(block.get("measured_at") or 0.0),
        measurements=rows,
        detail=str(block.get("detail") or ""),
    )


def execution_lane_from_metadata(meta: Mapping[str, Any]) -> Tuple[str, str]:
    """``(lane, reason)`` a cell's envelope states.

    A cell minted before this mechanism records nothing — that is the
    DECLARED default with a typed reason, never a silent fall-through. An
    unreadable or unknown verdict is the same: conservative, and it says so.
    """
    block = meta.get(META_KEY) if isinstance(meta, Mapping) else None
    if block is None:
        return DEFAULT_EXECUTION_LANE, (
            f"{REASON_ABSENT}: cell records no kernel-lane verdict "
            f"(pre-pgw#947 cell); serving the declared default "
            f"{DEFAULT_EXECUTION_LANE!r}")
    if not isinstance(block, Mapping):
        return DEFAULT_EXECUTION_LANE, (
            f"{REASON_UNREADABLE}: cell's {META_KEY!r} is "
            f"{type(block).__name__}, not a block; serving {DEFAULT_EXECUTION_LANE!r}")
    winner = str(block.get("winner") or "")
    if winner not in EXECUTION_LANES:
        return DEFAULT_EXECUTION_LANE, (
            f"{REASON_UNKNOWN_EXECUTION_LANE}: cell names lane {winner!r}, which this "
            f"worker does not implement ({list(EXECUTION_LANES)!r}); serving "
            f"{DEFAULT_EXECUTION_LANE!r}")
    return winner, (
        f"{REASON_ADOPTED}: cell verdict {winner!r} "
        f"(binding={block.get('binding') or '?'}, "
        f"rule={block.get('rule') or '?'})")


# --- re-applying the rule on THIS card --------------------------------------


def recorded_fit(
    meta: Mapping[str, Any],
) -> Tuple[Dict[str, int], Tuple[str, ...], Tuple[Measurement, ...], str]:
    """``(peaks, order, measurements, provenance)`` for the local re-fit.

    Two channels, ranked by how much they carry:

    * ``kernel_lane_evidence`` — the full record (ms/step AND peak bytes).
      Present when the caller has the PUBLISHED checkpoint metadata, and
      enough to re-run :func:`select` outright.
    * the packed envelope's ``fit`` block — quantized peaks and the fallback
      order, and nothing else. This is what a serving worker actually holds,
      because it reads ``metadata.json`` out of the delivered cell.

    Empty when neither is there (a cell minted before this block existed, or
    a ``sole_candidate`` verdict that measured nothing) — the caller must
    then say so rather than pretend the fit was checked.
    """
    evidence = meta.get(EVIDENCE_KEY)
    if isinstance(evidence, Mapping):
        rows = tuple(
            r for r in verdict_from_evidence(evidence).measurements
            if r.usable and r.execution_lane in EXECUTION_LANES)
        if rows:
            return (
                {r.execution_lane: r.peak_bytes for r in rows},
                refit_order(rows, winner=str(evidence.get("winner") or "")),
                rows, EVIDENCE_KEY)
    block = meta.get(META_KEY)
    fit = block.get("fit") if isinstance(block, Mapping) else None
    if isinstance(fit, Mapping):
        peaks = {
            str(execution_lane): int(peak)
            for execution_lane, peak in (fit.get("peaks") or {}).items()
            if str(execution_lane) in EXECUTION_LANES and int(peak or 0) > 0
        }
        if peaks:
            order = tuple(
                str(execution_lane) for execution_lane in (fit.get("order") or ())
                if str(execution_lane) in peaks)
            return peaks, order or tuple(sorted(peaks)), (), META_KEY
    return {}, (), (), ""


def refit(
    execution_lane: str, reason: str, meta: Mapping[str, Any],
) -> Tuple[str, str]:
    """Re-apply the fit constraint against THIS card and return the lane to
    pin with the reason it is pinned.

    Cells are keyed on SM and the lane is not a key axis, so the card that
    minted this verdict may be much larger than the card serving it — a
    96 GB RTX PRO 6000 and a 32 GB RTX 5090 are one cell key. The recorded
    winner is therefore checked against this device's honestly detected
    total before it is obeyed:

    * it fits -> the recorded verdict, unchanged (the fast path);
    * it does not -> the fastest RECORDED candidate that does fit here,
      pinned with ``kernel_lane_refit_local``;
    * nothing fits -> the smallest recorded peak, pinned with
      ``kernel_lane_refit_no_fit``. Never the declared default: the default
      carries the DENSE modulation, which is the LARGER residency, so
      "fall back to the default" on a card that ran out of memory would
      choose the biggest lane of all;
    * the fit is not re-applicable here (no recorded peaks, or no detectable
      device total) -> the recorded verdict, marked
      ``kernel_lane_fit_unverified``.
    """
    total, name, sm = device_facts()
    peaks, order, rows, provenance = recorded_fit(meta)
    device = f"{name or 'this device'} ({sm or 'sm unknown'})"
    if not total:
        return execution_lane, (
            f"{reason}; {REASON_FIT_UNVERIFIED}: this process cannot detect a "
            f"device total, so the cell's fit constraint could not be "
            f"re-applied here")
    if execution_lane not in peaks:
        return execution_lane, (
            f"{reason}; {REASON_FIT_UNVERIFIED}: the cell records no measured "
            f"peak for {execution_lane!r}, so its fit could not be re-applied on "
            f"{device}, {total} B — adopting it unverified across cards "
            f"(cells are keyed on SM, not on card memory)")
    if fits_bytes(peaks[execution_lane], total):
        return execution_lane, (
            f"{reason}; re-checked here: {required_for(peaks[execution_lane])} B "
            f"required of {total} B on {device} [{provenance}]")
    if rows:
        # The full record is present: re-run THE rule, not a reduction of it.
        local = select(rows, device_total_bytes=total, device_name=name, sm=sm)
        winner, binding, detail = local.winner, local.binding, local.detail
    else:
        fitting = tuple(
            cand for cand in order if fits_bytes(peaks[cand], total))
        if fitting:
            winner, binding = fitting[0], BIND_FIT
            detail = (
                f"{winner!r} is the fastest recorded candidate that fits "
                f"({required_for(peaks[winner])} B required); order "
                f"{list(order)!r}")
        else:
            winner = min(peaks, key=lambda cand: (peaks[cand], cand))
            binding, detail = BIND_NO_FIT, (
                f"no recorded candidate fits {total} B; the smallest peak "
                f"wins ({winner!r}, {required_for(peaks[winner])} B required)")
    tag = REASON_REFIT_LOCAL if binding != BIND_NO_FIT else REASON_REFIT_NO_FIT
    return winner, (
        f"{tag}: the cell's verdict {execution_lane!r} asks "
        f"{required_for(peaks[execution_lane])} B and {device} has {total} B, so it "
        f"does not fit here; re-applied the rule locally -> {winner!r} "
        f"(binding={binding}) — {detail} [{provenance}]")


# --- the process pin --------------------------------------------------------

_PIN: Optional[str] = None
_PIN_REASON: str = ""


def pin(execution_lane: str, reason: str) -> None:
    """Pin the lane THIS process loads on, with the reason it is pinned.

    Set by the mint (once per candidate while probing, then to the winner)
    and by the executor from the delivered cell before ``setup()`` runs —
    the swap happens at model load, so the pin must precede it.
    """
    global _PIN, _PIN_REASON

    if execution_lane not in EXECUTION_LANES:
        raise ExecutionLaneProbeError(f"unknown kernel lane {execution_lane!r} (have {EXECUTION_LANES!r})")
    _PIN, _PIN_REASON = execution_lane, str(reason or "")
    logger.info("kernel-lane: pinned %s — %s", execution_lane, _PIN_REASON)


def pinned() -> Tuple[Optional[str], str]:
    """``(lane, reason)`` or ``(None, "")`` when nothing has pinned one."""
    return _PIN, _PIN_REASON


def clear() -> None:
    """Forget the pin (mint between candidates; tests)."""
    global _PIN, _PIN_REASON

    _PIN, _PIN_REASON = None, ""


def adopt(meta: Optional[Mapping[str, Any]], *, source: str = "") -> str:
    """Adopt the lane a delivered cell's metadata states, RE-CHECKED against
    this card, and return it.

    ``None`` (no cell for this boot) is the declared default with its own
    typed reason — an eager or self-minting boot has no verdict yet and must
    say so rather than guessing that a hand-written tuple was right.

    A verdict that IS present is not obeyed on sight. The cell key is keyed
    on SM and the lane is not one of its axes, so the minting card and this
    one can differ by 64 GB of memory; :func:`refit` re-applies the fit
    constraint here before the lane is pinned.
    """
    if meta is None:
        execution_lane, reason = DEFAULT_EXECUTION_LANE, (
            f"{REASON_NO_CELL}: no compiled cell delivered for this load; "
            f"serving the declared default {DEFAULT_EXECUTION_LANE!r}")
    else:
        execution_lane, reason = execution_lane_from_metadata(meta)
        if reason.startswith(REASON_ADOPTED):
            execution_lane, reason = refit(execution_lane, reason, meta)
    if source:
        reason = f"{reason} [{source}]"
    pin(execution_lane, reason)
    return execution_lane


def adopt_from_artifact(artifact: Any, *, source: str = "") -> str:
    """Adopt the verdict recorded in a packed cell on disk.

    Reads ``metadata.json`` out of the tar without unpacking it (both the
    exported and the inductor-cache artifact kinds put it at the root). Any
    read failure is the conservative default WITH the failure named — a
    serving worker must never guess a lane.
    """
    if artifact is None:
        return adopt(None, source=source)
    try:
        return adopt(
            artifact_meta.read_metadata(artifact),
            source=source or str(artifact))
    except Exception as exc:  # noqa: BLE001 — never fails a load
        pin(DEFAULT_EXECUTION_LANE, (
            f"{REASON_UNREADABLE}: cannot read the verdict from "
            f"{artifact} ({type(exc).__name__}: {exc}); serving "
            f"{DEFAULT_EXECUTION_LANE!r}"
            + (f" [{source}]" if source else "")))
        return DEFAULT_EXECUTION_LANE


__all__ = [
    "ACTIVATION_SPIKE_FRACTION",
    "AXES",
    "AXIS_LINEAR",
    "AXIS_MODULATION",
    "BENCH_ITERS",
    "BENCH_WARMUP",
    "BIND_FIT",
    "BIND_NO_FIT",
    "BIND_SOLE_CANDIDATE",
    "BIND_SPEED",
    "BIND_VRAM_TIEBREAK",
    "DEFAULT_EXECUTION_LANE",
    "EVIDENCE_KEY",
    "FRAGMENTATION_HEADROOM_BYTES",
    "FUSED_MIN_SM",
    "EXECUTION_LANES",
    "LINEAR_BASELINE",
    "LINEAR_FUSED",
    "LINEAR_EXECUTION_LANES",
    "MARGIN_FRACTION",
    "META_KEY",
    "MOD_DENSE",
    "MOD_EXECUTION_LANES",
    "MOD_PACKED",
    "PEAK_QUANTUM_BYTES",
    "REASON_ABSENT",
    "REASON_ADOPTED",
    "REASON_FIT_UNVERIFIED",
    "REASON_NO_CELL",
    "REASON_REFIT_LOCAL",
    "REASON_REFIT_NO_FIT",
    "REASON_UNKNOWN_EXECUTION_LANE",
    "REASON_UNREADABLE",
    "SCHEMA",
    "SEP",
    "ExecutionLaneProbeError",
    "Measurement",
    "Verdict",
    "adopt",
    "adopt_from_artifact",
    "candidate_axes",
    "candidates_here",
    "clear",
    "device_facts",
    "envelope_block",
    "evidence_block",
    "fit_block",
    "fits",
    "fits_bytes",
    "fused_candidate_gap",
    "execution_lane_from_metadata",
    "execution_lane_of",
    "linear_of",
    "measure",
    "modulation_of",
    "packed_candidate_gap",
    "pin",
    "pinned",
    "probe",
    "quantize_peak",
    "recorded_fit",
    "refit",
    "refit_order",
    "required_for",
    "select",
    "sole",
    "split_execution_lane",
    "verdict_from_evidence",
]
