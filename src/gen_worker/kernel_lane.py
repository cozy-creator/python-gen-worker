"""Measured serving-kernel lane selection (pgw#946).

The svdq serving path can run its linears through hand-fused native kernels
or through the baseline unfused chain. Which one is FASTER is a per-card
FACT, not a derivable one: a custom op is opaque to inductor, so on sm_120
our fusion beats what inductor can do with the open chain and on sm_100 it
loses to it. That was a hand-maintained SM tuple, informed by ~$12 benchmark
campaigns per card and one human edit per new card class.

This module replaces the tuple with a measurement taken on the card the cell
is being minted for, recorded INTO the cell, and adopted by serving:

    mint      probe(candidates, ...) -> Verdict     (both lanes built,
              compiled, and timed on the target card, in the pgw#784 mint
              process)
    envelope  metadata.json["kernel_lane"] — the DISCRETE verdict only
    result    metadata["kernel_lane_evidence"] — the numbers, published with
              the checkpoint, never packed (the #699 double-mint byte-compare
              forbids wall clocks inside the artifact)
    serve     adopt(meta) -> pin(); the load-time swap reads the pin

THE SELECTION RULE (Paul, 2026-08-03): **fit-constrained speed
maximization.** Among lanes whose measured peak VRAM fits the target card
with headroom, pick the FASTEST. VRAM is a CONSTRAINT, not an objective — it
breaks a tie only when two lanes are within the speed noise margin. On a
B200 that means the baseline lane (228 ms/step, 44.1 GB) over the fused lane
(350 ms/step, 35.1 GB): the card has the room. On a 32 GB 5090 the fit
constraint does real work and can exclude a faster lane outright.

`models/w4a4.py::_gemm_profitable` is the in-tree precedent — it
micro-benchmarks fp4 against bf16 at boot and refuses to arm unless it wins
by a real margin. This is that mechanism one level up: whole compiled graphs
instead of one GEMM, paid once at mint instead of on every boot, and
RECORDED instead of re-derived.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)

# --- vocabulary -------------------------------------------------------------

LANE_BASELINE = "baseline"
LANE_FUSED = "fused"
LANES = (LANE_BASELINE, LANE_FUSED)

# What a worker runs when no cell says otherwise. The unfused chain is the
# one that exists on every card and is never a pessimisation on any of them.
DEFAULT_LANE = LANE_BASELINE

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
REASON_UNKNOWN_LANE = "kernel_lane_verdict_unknown_lane"
REASON_NO_CELL = "kernel_lane_no_cell"
REASON_ADOPTED = "kernel_lane_verdict_adopted"

# Which term bound a verdict.
BIND_SPEED = "speed"
BIND_FIT = "fit"
BIND_VRAM_TIEBREAK = "vram_tiebreak"
BIND_SOLE_CANDIDATE = "sole_candidate"
BIND_NO_FIT = "no_fit"


class LaneProbeError(RuntimeError):
    """A candidate could not be built or measured. Never fatal: the candidate
    drops out of the ranking with its reason recorded."""


# --- value types ------------------------------------------------------------


class Measurement(msgspec.Struct, frozen=True, kw_only=True):
    """One candidate lane, measured on the target card."""

    lane: str
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
        return int(self.peak_bytes * (1.0 + ACTIVATION_SPIKE_FRACTION)) \
            + FRAGMENTATION_HEADROOM_BYTES


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

    def measurement(self, lane: str) -> Optional[Measurement]:
        for row in self.measurements:
            if row.lane == lane:
                return row
        return None


# --- the rule ---------------------------------------------------------------


def fits(row: Measurement, device_total_bytes: int) -> bool:
    """Does this lane fit the card with the stated allowance? A card whose
    total is unknown (0) constrains nothing — the mint measured it there, so
    it ran there."""
    if not device_total_bytes:
        return True
    return row.required_bytes() <= int(device_total_bytes)


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
            f"{r.lane}: {r.unavailable or 'no timing'}" for r in rows)
        return Verdict(
            winner=DEFAULT_LANE, binding=BIND_NO_FIT,
            detail=f"no candidate measured ({gaps or 'no candidates'})",
            **common)

    fitting = tuple(r for r in usable if fits(r, device_total_bytes))
    if not fitting:
        smallest = min(usable, key=lambda r: (r.peak_bytes, r.lane))
        return Verdict(
            winner=smallest.lane, binding=BIND_NO_FIT,
            detail=(
                f"no candidate fits {device_total_bytes} B with the stated "
                f"allowance; smallest peak wins ({smallest.lane}, "
                f"{smallest.peak_bytes} B peak, "
                f"{smallest.required_bytes()} B required)"),
            **common)

    excluded = tuple(r.lane for r in usable if r not in fitting)
    if len(fitting) == 1:
        only = fitting[0]
        binding = BIND_FIT if excluded else BIND_SOLE_CANDIDATE
        detail = (
            f"{only.lane} is the only lane that fits; excluded "
            f"{sorted(excluded)!r}" if excluded
            else f"{only.lane} is the only candidate")
        return Verdict(winner=only.lane, binding=binding, detail=detail,
                       **common)

    ranked = sorted(fitting, key=lambda r: (r.ms_per_step, r.lane))
    best, rival = ranked[0], ranked[1]
    gap = (rival.ms_per_step - best.ms_per_step) / max(best.ms_per_step, 1e-9)
    if gap >= MARGIN_FRACTION:
        return Verdict(
            winner=best.lane, binding=BIND_SPEED,
            detail=(
                f"{best.lane} {best.ms_per_step:.1f} ms/step beats "
                f"{rival.lane} {rival.ms_per_step:.1f} by {gap * 100:.1f}% "
                f"(margin {MARGIN_FRACTION * 100:.0f}%)"
                + (f"; excluded on fit {sorted(excluded)!r}" if excluded
                   else "")),
            **common)
    # Within the noise margin: VRAM breaks the tie, and only here.
    tied = tuple(
        r for r in fitting
        if (r.ms_per_step - best.ms_per_step)
        / max(best.ms_per_step, 1e-9) < MARGIN_FRACTION)
    winner = min(tied, key=lambda r: (r.peak_bytes, r.lane))
    return Verdict(
        winner=winner.lane, binding=BIND_VRAM_TIEBREAK,
        detail=(
            f"{[r.lane for r in tied]!r} within {MARGIN_FRACTION * 100:.0f}% "
            f"({best.ms_per_step:.1f}-{rival.ms_per_step:.1f} ms/step); "
            f"smaller peak wins ({winner.lane}, {winner.peak_bytes} B)"),
        **common)


def sole(lane: str, detail: str) -> Verdict:
    """The verdict for a card that can only build ONE lane. No benchmark is
    run: with nothing to compare against, a measurement would buy a compile
    and decide nothing. The cell still records a real, typed verdict rather
    than the absence a serving worker has to guess about."""
    total, name, sm = device_facts()
    return Verdict(
        winner=lane, binding=BIND_SOLE_CANDIDATE, detail=detail,
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


def measure(lane: str, step: Callable[[], Any]) -> Measurement:
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
            lane=lane, unavailable=f"{type(exc).__name__}: {exc}")
    ordered = sorted(samples)
    return Measurement(
        lane=lane, ms_per_step=ordered[len(ordered) // 2], peak_bytes=peak,
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
    for lane in candidates:
        t0 = time.monotonic()
        try:
            step = build(lane)
        except Exception as exc:  # noqa: BLE001
            logger.warning("kernel-lane probe: %s could not be built — %s",
                           lane, exc)
            rows.append(Measurement(
                lane=lane, unavailable=f"build: {type(exc).__name__}: {exc}"))
            continue
        row = measure(lane, step)
        rows.append(row)
        logger.info(
            "kernel-lane probe: %s -> %s (%.1f s to build+measure)",
            lane,
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
    """Why the fused lane is not even a CANDIDATE here, or ''.

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


def candidates_here() -> Tuple[str, ...]:
    """The lanes that can be BUILT on this card, cheapest-first. Always
    includes the baseline: it exists everywhere."""
    lanes = [LANE_BASELINE]
    total, _name, sm_text = device_facts()
    if not total:
        return tuple(lanes)
    try:
        sm = int(str(sm_text or "sm_0").removeprefix("sm_"))
    except ValueError:
        return tuple(lanes)
    gap = fused_candidate_gap(sm)
    if gap:
        logger.info("kernel-lane: fused is not a candidate here — %s", gap)
    else:
        lanes.append(LANE_FUSED)
    return tuple(lanes)


# --- recording and reading --------------------------------------------------


def envelope_block(verdict: Verdict) -> Dict[str, Any]:
    """The DISCRETE verdict, for ``metadata.json`` inside the packed cell.

    Deliberately carries no wall clocks and no measured numbers: the #699
    double-mint byte-compare requires the artifact to be reproducible, and
    milliseconds are not. What a serving worker needs is the winner; what an
    auditor needs is :func:`evidence_block`, which rides the published
    checkpoint metadata beside ``mint_phases``.
    """
    return {
        "schema": int(verdict.schema),
        "winner": str(verdict.winner),
        "rule": str(verdict.rule),
        "binding": str(verdict.binding),
        "margin_fraction": float(verdict.margin_fraction),
        "candidates": sorted(r.lane for r in verdict.measurements),
    }


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
                "lane": r.lane,
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
            lane=str(c.get("lane") or ""),
            ms_per_step=float(c.get("ms_per_step") or 0.0),
            peak_bytes=int(c.get("peak_bytes") or 0),
            samples_ms=tuple(float(s) for s in (c.get("samples_ms") or ())),
            unavailable=str(c.get("unavailable") or ""),
        )
        for c in (block.get("candidates") or ())
    )
    return Verdict(
        winner=str(block.get("winner") or DEFAULT_LANE),
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


def lane_from_metadata(meta: Mapping[str, Any]) -> Tuple[str, str]:
    """``(lane, reason)`` a cell's envelope states.

    A cell minted before this mechanism records nothing — that is the
    DECLARED default with a typed reason, never a silent fall-through. An
    unreadable or unknown verdict is the same: conservative, and it says so.
    """
    block = meta.get(META_KEY) if isinstance(meta, Mapping) else None
    if block is None:
        return DEFAULT_LANE, (
            f"{REASON_ABSENT}: cell records no kernel-lane verdict "
            f"(pre-pgw#946 cell); serving the declared default "
            f"{DEFAULT_LANE!r}")
    if not isinstance(block, Mapping):
        return DEFAULT_LANE, (
            f"{REASON_UNREADABLE}: cell's {META_KEY!r} is "
            f"{type(block).__name__}, not a block; serving {DEFAULT_LANE!r}")
    winner = str(block.get("winner") or "")
    if winner not in LANES:
        return DEFAULT_LANE, (
            f"{REASON_UNKNOWN_LANE}: cell names lane {winner!r}, which this "
            f"worker does not implement ({list(LANES)!r}); serving "
            f"{DEFAULT_LANE!r}")
    return winner, (
        f"{REASON_ADOPTED}: cell verdict {winner!r} "
        f"(binding={block.get('binding') or '?'}, "
        f"rule={block.get('rule') or '?'})")


# --- the process pin --------------------------------------------------------

_PIN: Optional[str] = None
_PIN_REASON: str = ""


def pin(lane: str, reason: str) -> None:
    """Pin the lane THIS process loads on, with the reason it is pinned.

    Set by the mint (once per candidate while probing, then to the winner)
    and by the executor from the delivered cell before ``setup()`` runs —
    the swap happens at model load, so the pin must precede it.
    """
    global _PIN, _PIN_REASON

    if lane not in LANES:
        raise LaneProbeError(f"unknown kernel lane {lane!r} (have {LANES!r})")
    _PIN, _PIN_REASON = lane, str(reason or "")
    logger.info("kernel-lane: pinned %s — %s", lane, _PIN_REASON)


def pinned() -> Tuple[Optional[str], str]:
    """``(lane, reason)`` or ``(None, "")`` when nothing has pinned one."""
    return _PIN, _PIN_REASON


def clear() -> None:
    """Forget the pin (mint between candidates; tests)."""
    global _PIN, _PIN_REASON

    _PIN, _PIN_REASON = None, ""


def adopt(meta: Optional[Mapping[str, Any]], *, source: str = "") -> str:
    """Adopt the lane a delivered cell's metadata states, and return it.

    ``None`` (no cell for this boot) is the declared default with its own
    typed reason — an eager or self-minting boot has no verdict yet and must
    say so rather than guessing that a hand-written tuple was right.
    """
    if meta is None:
        lane, reason = DEFAULT_LANE, (
            f"{REASON_NO_CELL}: no compiled cell delivered for this load; "
            f"serving the declared default {DEFAULT_LANE!r}")
    else:
        lane, reason = lane_from_metadata(meta)
    if source:
        reason = f"{reason} [{source}]"
    pin(lane, reason)
    return lane


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
        import json
        import tarfile

        with tarfile.open(str(artifact), "r:*") as tar:
            for member in tar:
                if member.name == "metadata.json" and member.isfile():
                    fh = tar.extractfile(member)
                    if fh is None:
                        break
                    meta = json.loads(fh.read().decode())
                    return adopt(meta, source=source or str(artifact))
        raise LaneProbeError("artifact has no metadata.json")
    except Exception as exc:  # noqa: BLE001 — never fails a load
        pin(DEFAULT_LANE, (
            f"{REASON_UNREADABLE}: cannot read the verdict from "
            f"{artifact} ({type(exc).__name__}: {exc}); serving "
            f"{DEFAULT_LANE!r}"
            + (f" [{source}]" if source else "")))
        return DEFAULT_LANE


__all__ = [
    "ACTIVATION_SPIKE_FRACTION",
    "BENCH_ITERS",
    "BENCH_WARMUP",
    "BIND_FIT",
    "BIND_NO_FIT",
    "BIND_SOLE_CANDIDATE",
    "BIND_SPEED",
    "BIND_VRAM_TIEBREAK",
    "DEFAULT_LANE",
    "EVIDENCE_KEY",
    "FRAGMENTATION_HEADROOM_BYTES",
    "FUSED_MIN_SM",
    "LANES",
    "LANE_BASELINE",
    "LANE_FUSED",
    "MARGIN_FRACTION",
    "META_KEY",
    "REASON_ABSENT",
    "REASON_ADOPTED",
    "REASON_NO_CELL",
    "REASON_UNKNOWN_LANE",
    "REASON_UNREADABLE",
    "SCHEMA",
    "LaneProbeError",
    "Measurement",
    "Verdict",
    "adopt",
    "adopt_from_artifact",
    "candidates_here",
    "clear",
    "device_facts",
    "envelope_block",
    "evidence_block",
    "fits",
    "fused_candidate_gap",
    "lane_from_metadata",
    "measure",
    "pin",
    "pinned",
    "probe",
    "select",
    "sole",
    "verdict_from_evidence",
]
