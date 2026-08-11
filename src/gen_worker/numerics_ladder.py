"""The verdict ladder, shared (pgw#817, lifting pgw#800; requested by pgw#814).

pgw#800 built ONE fail-closed fidelity gate — cosine as the gated quantity,
``HEALTHY / DEGRADED / DESTROYED``, a typed refusal to the caller AND a typed
activity event to the hub — and calibrated it for adapter deltas. pgw#814 then
measured a second, unrelated population through the same ladder (a compiled
artifact's output against eager's) and found the ladder itself transfers
exactly while the low-rank machinery around it does not:

* ``_gram_cosine`` / ``evaluate_branch`` / ``evaluate_fuse`` exploit ``D = BA``
  to get ``tr(D'^T D)`` from two ``r x r`` Grams. Comparing two OUTPUT tensors
  is ``O(n)`` and needs none of it.
* ``TargetGrid`` is a weight-grid concept with no output-comparison meaning.
* ``FIDELITY_FLOOR`` / ``FIDELITY_WARN`` are calibrated for adapter deltas and
  must NOT be inherited by an output comparison (pgw#814's explicit warning).

So this module owns the parts that are population-independent — the verdict
ladder, the aggregate rule, the evidence formatting and the gate shape — and
each caller brings its own :class:`Thresholds` and its own evaluator.
:mod:`gen_worker.models.adapter_fidelity` is one caller; the compiled-cell
assembled-vs-eager population (whose calibrated defaults live below, moved
here from the retired regional module by pgw#846) is the other.

**The aggregate rule, kept verbatim from pgw#800 because it is the whole
point:** one norm-weighted number over every row, never a per-row median. A
median lets a handful of destroyed high-norm rows hide behind many intact
low-norm ones, which is precisely the shape of an artifact that diverges in
its last blocks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from . import activity as activity_mod

logger = logging.getLogger(__name__)

#: The ladder's three rungs. Ordered worst-first by :data:`_ORDER`.
VERDICT_HEALTHY = "healthy"
VERDICT_DEGRADED = "degraded"
VERDICT_DESTROYED = "destroyed"

_ORDER = {VERDICT_HEALTHY: 0, VERDICT_DEGRADED: 1, VERDICT_DESTROYED: 2}

#: Typed event phases. A DEGRADED subject is served and confesses; a DESTROYED
#: subject is refused and confesses. Both reach the hub, because the error
#: reaches only the caller of one request while a fleet-wide rate is only
#: countable from activity records (pgw#800).
PHASE_DEGRADED = "degraded"
PHASE_REFUSED = "refused"

#: pgw#1141 (§4.31/§4.32) — an armed exported cell took no warm dispatch this
#: boot, which is the NORMAL state of every adopted cell (the arm precedes
#: setup) and is no longer a verdict about it. Carried on this kind so one
#: query (`?kind=cell_numerics`) still answers what happened to every cell that
#: armed on a pod, and emitted because an unannounced posture is
#: indistinguishable from a gate that never ran.
PHASE_ARMED_UNDISPATCHED = "armed_undispatched"


def worst_of(*verdicts: str) -> str:
    """The worst rung among ``verdicts`` (unknown values read as HEALTHY)."""
    return max(
        (v for v in verdicts if v), key=lambda v: _ORDER.get(v, 0),
        default=VERDICT_HEALTHY)


@dataclass(frozen=True)
class Thresholds:
    """One population's calibration of the ladder.

    ``floor`` and ``warn`` are cosine bounds: below ``floor`` the subject is
    DESTROYED (typed refusal), in ``[floor, warn)`` DEGRADED (served, typed
    event), at or above ``warn`` HEALTHY (silent).

    ``retention_floor`` gates the SECOND number, and it is not decoration.
    Cosine is scale-invariant, so a compiled artifact that reproduces eager's
    direction perfectly at 0.9x the magnitude scores a perfect cosine while
    serving a systematically dimmer image. The band is symmetric in the log:
    ``retention`` outside ``[retention_floor, 1 / retention_floor]`` is at
    least DEGRADED. ``0.0`` disables it (the adapter population gates cosine
    only — pgw#800 measured retention 15.3 on a DESTROYED adapter, so there
    the number is evidence, not a bound).
    """

    floor: float
    warn: float
    retention_floor: float = 0.0
    #: Where these numbers came from. Carried into every refusal so the
    #: reader can argue with the calibration instead of guessing at it.
    label: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.floor <= self.warn <= 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 <= floor <= warn <= 1, got "
                f"floor={self.floor!r} warn={self.warn!r}")
        if not 0.0 <= self.retention_floor <= 1.0:
            raise ValueError(
                f"retention_floor must be in [0, 1], got "
                f"{self.retention_floor!r}")

    def verdict(self, cosine: float, retention: float = 1.0) -> str:
        """The rung ``(cosine, retention)`` lands on — the WORST of the two
        bounds, never an average of them."""
        by_cosine = VERDICT_HEALTHY
        if cosine < self.floor:
            by_cosine = VERDICT_DESTROYED
        elif cosine < self.warn:
            by_cosine = VERDICT_DEGRADED
        by_retention = VERDICT_HEALTHY
        if self.retention_floor > 0.0 and retention > 0.0:
            lo, hi = self.retention_floor, 1.0 / self.retention_floor
            if not lo <= retention <= hi:
                by_retention = VERDICT_DEGRADED
        elif self.retention_floor > 0.0:
            # A zero-norm subject against a non-zero reference is not
            # "slightly off" — it is the output missing entirely.
            by_retention = VERDICT_DESTROYED
        return worst_of(by_cosine, by_retention)

    def __str__(self) -> str:
        tail = (f" retention_floor={self.retention_floor:g}"
                if self.retention_floor else "")
        head = f"{self.label} " if self.label else ""
        return f"{head}floor={self.floor:g} warn={self.warn:g}{tail}"


@dataclass(frozen=True)
class RowStat:
    """One comparable unit's own numbers — an output tensor, a module's delta.

    Carried as evidence so a refusal names WHICH unit parted from the
    reference, which is the difference between an actionable hub record and a
    number nobody can act on.
    """

    name: str
    elements: int
    cosine: float
    retention: float
    #: ``|candidate - reference| / |reference|``. Reported because it is what
    #: the AOT literature quotes; never gated on — it conflates direction and
    #: scale, which is exactly the conflation the two gated numbers separate.
    rel_l2: float = 0.0

    def __str__(self) -> str:
        return (f"{self.name} cos={self.cosine:.5f} ret={self.retention:.4f} "
                f"rel_l2={self.rel_l2:.4f} n={self.elements}")


@dataclass(frozen=True)
class Comparison:
    """One subject's whole-artifact agreement with its reference."""

    #: What was compared against (``"eager"``, the true adapter delta, ...).
    reference: str
    #: What is on trial (a cell key, an adapter ref, ...).
    subject: str
    thresholds: Thresholds
    rows: Tuple[RowStat, ...]
    #: The GATED quantity: norm-weighted over every row, i.e. the cosine of
    #: the concatenated subject against the concatenated reference.
    cosine: float
    retention: float
    rel_l2: float = 0.0

    @property
    def verdict(self) -> str:
        return self.thresholds.verdict(self.cosine, self.retention)

    @property
    def healthy(self) -> bool:
        return self.verdict == VERDICT_HEALTHY

    def worst(self, limit: int = 5) -> Tuple[RowStat, ...]:
        return tuple(sorted(self.rows, key=lambda r: r.cosine)[:limit])

    def evidence(self, limit: int = 5) -> str:
        """Everything a reader needs without the worker's logs."""
        head = (
            f"subject={self.subject or '<unnamed>'} "
            f"vs {self.reference or '<reference>'} rows={len(self.rows)} "
            f"cosine={self.cosine:.5f} retention={self.retention:.4f} "
            f"rel_l2={self.rel_l2:.4f} verdict={self.verdict} "
            f"[{self.thresholds}]")
        worst = self.worst(limit)
        if not worst:
            return head
        return head + " | worst: " + "; ".join(str(r) for r in worst)


# ---------------------------------------------------------------------------
# The O(n) output evaluator (pgw#814's "give the numerics gate its own")
# ---------------------------------------------------------------------------


def _flatten_tensors(value: Any, prefix: str, out: List[Tuple[str, Any]]) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        out.append((prefix or "out", value))
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _flatten_tensors(item, f"{prefix}[{i}]" if prefix else f"[{i}]", out)
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _flatten_tensors(
                value[key], f"{prefix}.{key}" if prefix else str(key), out)
        return
    # A dataclass-ish output container (diffusers' ``*Output``): walk its
    # public tensor fields in NAME order so two runs pair the same tensors.
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict) and fields:
        for key in sorted(fields, key=str):
            if str(key).startswith("_"):
                continue
            _flatten_tensors(
                fields[key], f"{prefix}.{key}" if prefix else str(key), out)


def flatten_outputs(value: Any) -> Tuple[Tuple[str, Any], ...]:
    """Every tensor in a forward's return, NAMED and in a deterministic order.

    Two runs of the same model must produce the same name sequence or the
    comparison is pairing different tensors — which is why ordering is by NAME
    at every dict/attribute level rather than by insertion.
    """
    out: List[Tuple[str, Any]] = []
    _flatten_tensors(value, "", out)
    return tuple(out)


def compare_outputs(
    reference: Any,
    subject: Any,
    *,
    thresholds: Thresholds,
    reference_label: str = "eager",
    subject_label: str = "",
) -> Comparison:
    """Compare two forward RETURNS, tensor by tensor, in ``O(n)``.

    Both sides are flattened to named tensors and paired by name; a name
    present on one side only is a structural mismatch and raises, because a
    silently-dropped output is the failure this gate exists to catch, not a
    row to average over.

    Everything is accumulated in float32 on the tensors' own device — the
    comparison must not itself be the source of the loss it measures, and a
    bf16 dot product over a 4096-token latent loses more precision than the
    artifact under test.
    """
    import torch

    ref_rows = flatten_outputs(reference)
    sub_rows = flatten_outputs(subject)
    ref_names = [n for n, _ in ref_rows]
    sub_names = [n for n, _ in sub_rows]
    if ref_names != sub_names:
        raise ValueError(
            f"output structure differs: reference produced {ref_names[:8]!r} "
            f"and subject produced {sub_names[:8]!r} — a comparison that "
            f"pairs different tensors proves nothing")
    if not ref_rows:
        raise ValueError(
            "neither forward returned a tensor; there is nothing to compare")

    rows: List[RowStat] = []
    cross = 0.0
    ref_sq = 0.0
    sub_sq = 0.0
    diff_sq = 0.0
    for (name, a), (_n, b) in zip(ref_rows, sub_rows):
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(
                f"output {name!r} shape differs: reference {tuple(a.shape)} "
                f"vs subject {tuple(b.shape)}")
        with torch.no_grad():
            x = a.detach().reshape(-1).to(torch.float32)
            y = b.detach().to(x.device).reshape(-1).to(torch.float32)
            c = float(torch.dot(x, y))
            xs = float(torch.dot(x, x))
            ys = float(torch.dot(y, y))
            ds = float(torch.dot(y - x, y - x))
        cross += c
        ref_sq += xs
        sub_sq += ys
        diff_sq += ds
        rows.append(RowStat(
            name=name, elements=int(x.numel()),
            cosine=_cosine(c, xs, ys), retention=_ratio(ys, xs),
            rel_l2=_ratio(ds, xs)))
    return Comparison(
        reference=reference_label, subject=subject_label,
        thresholds=thresholds, rows=tuple(rows),
        cosine=_cosine(cross, ref_sq, sub_sq),
        retention=_ratio(sub_sq, ref_sq),
        rel_l2=_ratio(diff_sq, ref_sq))


def _cosine(cross: float, ref_sq: float, sub_sq: float) -> float:
    denom = math.sqrt(max(ref_sq, 0.0)) * math.sqrt(max(sub_sq, 0.0))
    if denom <= 0.0:
        # Both sides all-zero is agreement; one side zero is not.
        return 1.0 if ref_sq <= 0.0 and sub_sq <= 0.0 else 0.0
    return max(-1.0, min(1.0, cross / denom))


def _ratio(num_sq: float, den_sq: float) -> float:
    if den_sq <= 0.0:
        return 0.0 if num_sq <= 0.0 else float("inf")
    return math.sqrt(max(num_sq, 0.0) / den_sq)


# ---------------------------------------------------------------------------
# The gate shape
# ---------------------------------------------------------------------------


def gate(
    comparison: Optional[Comparison],
    *,
    kind: str,
    refuse: Callable[[str, Comparison], BaseException],
    context: str = "",
    announce: bool = True,
) -> Optional[Comparison]:
    """Fail closed below the floor; confess in the gray band; silent above.

    ``refuse`` builds the caller's OWN typed exception from the message and
    the comparison — this module refuses to own a shared error type, because
    "this adapter cannot serve here" and "this cell must not arm" route
    through completely different classifiers.
    """
    if comparison is None:
        return None
    verdict = comparison.verdict
    if verdict == VERDICT_HEALTHY:
        return comparison
    detail = comparison.evidence()
    head = f"{context} " if context else ""
    if verdict == VERDICT_DEGRADED:
        logger.warning("numerics ladder: %sDEGRADED — %s", head, detail)
        if announce:
            activity_mod.emit_event(
                kind, f"{head}{detail}", phase=PHASE_DEGRADED)
        return comparison
    logger.error("numerics ladder: %sDESTROYED — %s", head, detail)
    activity_mod.emit_event(kind, f"{head}{detail}", phase=PHASE_REFUSED)
    raise refuse(
        f"{head}the subject does not reproduce its reference: aggregate "
        f"cosine {comparison.cosine:.5f} is below the "
        f"{comparison.thresholds.floor:g} floor "
        f"({100.0 * (1.0 - max(0.0, comparison.cosine)):.2f}% of the output's "
        f"direction is not reproduced, magnitude ratio "
        f"{comparison.retention:.4f}). {detail}",
        comparison)


# ---------------------------------------------------------------------------
# The compiled-cell (assembled-vs-eager) calibration — pgw#814's measured band
# ---------------------------------------------------------------------------
# Moved here from the retired regional module (pgw#846): the calibration is
# family-GENERAL — it reads `Compile.numerics_floor` / `numerics_warn`, which
# whole-graph families (sdxl: 0.995/0.999) declare too, and the adoption
# numerics gate is the mechanism that should have caught the regional serve
# regression.

#: Cosine floor for an ASSEMBLED-vs-EAGER comparison — below it the cell is
#: DESTROYED and refuses to arm.
#:
#: Derived from pgw#814's measured band on the production toolchain (torch
#: 2.13.0+cu130, L4/sm_89), NOT inherited from pgw#800's adapter floors —
#: pgw#814 says in as many words that those are calibrated for adapter deltas
#: and must not be assumed here:
#:
#:   worst configuration we ACCEPT   0.9890  flux2 w8a8 pertensor vs eager at
#:                                           T_img=4096 (0.9926 at 8160)
#:   best configuration we REFUSE    0.9730  flux2 w8a8 ROWWISE whole-graph vs
#:                                           eager — pgw#814's "do not adopt a
#:                                           flux2 w8a8 cell until this
#:                                           closes", i.e. the artifact the
#:                                           platform decided is not servable.
#:
#: 0.98 is that band's geometric midpoint (sqrt(0.9890 * 0.9730) = 0.98097),
#: 1.0092x of headroom below the worst accepted case and 1.0072x above the
#: best refused one. The healthy population sits far above it: bf16 control
#: 0.99979, sdxl w8a8 whole-graph 0.99984.
NUMERICS_FLOOR = 0.98

#: Gray-band ceiling — at or above this the arm is silent, below it the cell
#: arms and confesses ``cell_numerics phase=degraded``.
#:
#: Every configuration anyone has called healthy measures 0.9998+ (bf16
#: control 0.99979, sdxl w8a8 whole-graph 0.99984), so an artifact that has
#: lost more than 0.1% of the output's DIRECTION has lost it to something —
#: fp8 accumulation drift, a fused reassociation — and that is worth counting
#: fleet-wide even when it is served.
NUMERICS_WARN = 0.999

#: Magnitude band. Cosine is scale-invariant, so an artifact that reproduces
#: eager's direction exactly at 0.9x the magnitude scores a PERFECT cosine
#: while serving a systematically dimmer image; nothing in pgw#800's ladder
#: could see that, because an adapter's retention is evidence rather than a
#: bound (a destroyed one measures 15.3).
#:
#: Derived from the same measured band: worst accepted 0.997 (bf16 control),
#: best refused 0.905 (flux2 w8a8 pertensor whole-graph; rowwise 0.902).
#: sqrt(0.997 * 0.905) = 0.9500. Applied symmetrically in the log, so
#: retention outside [0.95, 1.0526] is at least DEGRADED.
NUMERICS_RETENTION_FLOOR = 0.95

DEFAULT_THRESHOLDS = Thresholds(
    floor=NUMERICS_FLOOR, warn=NUMERICS_WARN,
    retention_floor=NUMERICS_RETENTION_FLOOR,
    label="assembled-vs-eager (pgw#814 §VERDICT)")


def declared_thresholds(cfg: Any) -> Thresholds:
    """The tolerance THIS family declares, falling back to the SDK default.

    A declared per-family tolerance, because bf16 attention reassociation
    makes exact equality the wrong bar and the right bar is not the same for
    a 25-block fp8 DiT and a conv-bearing UNet. A family that declares
    nothing gets :data:`DEFAULT_THRESHOLDS`, whose derivation is the measured
    band above — a default with evidence, not a guess.
    """
    floor = getattr(cfg, "numerics_floor", None)
    warn = getattr(cfg, "numerics_warn", None)
    if floor is None and warn is None:
        return DEFAULT_THRESHOLDS
    return Thresholds(
        floor=float(NUMERICS_FLOOR if floor is None else floor),
        warn=float(NUMERICS_WARN if warn is None else warn),
        retention_floor=NUMERICS_RETENTION_FLOOR,
        label=f"assembled-vs-eager (declared by {getattr(cfg, 'family', '?')})")


__all__ = [
    "DEFAULT_THRESHOLDS",
    "NUMERICS_FLOOR", "NUMERICS_RETENTION_FLOOR", "NUMERICS_WARN",
    "PHASE_ARMED_UNDISPATCHED", "PHASE_DEGRADED", "PHASE_REFUSED",
    "VERDICT_DEGRADED", "VERDICT_DESTROYED", "VERDICT_HEALTHY",
    "Comparison", "RowStat", "Thresholds",
    "compare_outputs", "declared_thresholds", "flatten_outputs", "gate",
    "worst_of",
]
