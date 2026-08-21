"""The verdict ladder, shared."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from . import activity as activity_mod

logger = logging.getLogger(__name__)

VERDICT_HEALTHY = "healthy"
VERDICT_DEGRADED = "degraded"
VERDICT_DESTROYED = "destroyed"

_ORDER = {VERDICT_HEALTHY: 0, VERDICT_DEGRADED: 1, VERDICT_DESTROYED: 2}

PHASE_DEGRADED = "degraded"
PHASE_REFUSED = "refused"

PHASE_ARMED_UNDISPATCHED = "armed_undispatched"


def worst_of(*verdicts: str) -> str:
    """The worst rung among ``verdicts`` (unknown values read as HEALTHY)."""
    return max(
        (v for v in verdicts if v), key=lambda v: _ORDER.get(v, 0),
        default=VERDICT_HEALTHY)


@dataclass(frozen=True)
class Thresholds:
    """One population's calibration of the ladder."""

    floor: float
    warn: float
    retention_floor: float = 0.0
    label: str = ""
    source: str = ""

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
        """The rung ``(cosine, retention)`` lands on — the WORST of the two bounds, never an average of them."""
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
            by_retention = VERDICT_DESTROYED
        return worst_of(by_cosine, by_retention)

    def __str__(self) -> str:
        tail = (f" retention_floor={self.retention_floor:g}"
                if self.retention_floor else "")
        head = f"{self.label} " if self.label else ""
        return f"{head}floor={self.floor:g} warn={self.warn:g}{tail}"


@dataclass(frozen=True)
class RowStat:
    """One comparable unit's own numbers — an output tensor, a module's delta."""

    name: str
    elements: int
    cosine: float
    retention: float
    rel_l2: float = 0.0

    def __str__(self) -> str:
        return (f"{self.name} cos={self.cosine:.5f} ret={self.retention:.4f} "
                f"rel_l2={self.rel_l2:.4f} n={self.elements}")


@dataclass(frozen=True)
class Comparison:
    """One subject's whole-artifact agreement with its reference."""

    reference: str
    subject: str
    thresholds: Thresholds
    rows: Tuple[RowStat, ...]
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
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict) and fields:
        for key in sorted(fields, key=str):
            if str(key).startswith("_"):
                continue
            _flatten_tensors(
                fields[key], f"{prefix}.{key}" if prefix else str(key), out)


def flatten_outputs(value: Any) -> Tuple[Tuple[str, Any], ...]:
    """Every tensor in a forward's return, NAMED and in a deterministic order."""
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
    """Compare two forward RETURNS, tensor by tensor, in ``O(n)``."""
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
        return 1.0 if ref_sq <= 0.0 and sub_sq <= 0.0 else 0.0
    return max(-1.0, min(1.0, cross / denom))


def _ratio(num_sq: float, den_sq: float) -> float:
    if den_sq <= 0.0:
        return 0.0 if num_sq <= 0.0 else float("inf")
    return math.sqrt(max(num_sq, 0.0) / den_sq)


def gate(
    comparison: Optional[Comparison],
    *,
    kind: str,
    refuse: Callable[[str, Comparison], BaseException],
    context: str = "",
    announce: bool = True,
) -> Optional[Comparison]:
    """Fail closed below the floor; confess in the gray band; silent above."""
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


NUMERICS_FLOOR = 0.98

NUMERICS_WARN = 0.999

NUMERICS_RETENTION_FLOOR = 0.95

SOURCE_DECLARED = "declared"
SOURCE_SDK_DEFAULT = "sdk-default"

DEFAULT_THRESHOLDS = Thresholds(
    floor=NUMERICS_FLOOR, warn=NUMERICS_WARN,
    retention_floor=NUMERICS_RETENTION_FLOOR,
    label="assembled-vs-eager (pgw#814 §VERDICT)",
    source=SOURCE_SDK_DEFAULT)


def declared_thresholds(cfg: Any) -> Thresholds:
    """The tolerance THIS family declares, falling back to the SDK default."""
    floor = getattr(cfg, "numerics_floor", None)
    warn = getattr(cfg, "numerics_warn", None)
    if floor is None and warn is None:
        return DEFAULT_THRESHOLDS
    return Thresholds(
        floor=float(NUMERICS_FLOOR if floor is None else floor),
        warn=float(NUMERICS_WARN if warn is None else warn),
        retention_floor=NUMERICS_RETENTION_FLOOR,
        label=f"assembled-vs-eager (declared by {getattr(cfg, 'family', '?')})",
        source=SOURCE_DECLARED)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "NUMERICS_FLOOR", "NUMERICS_RETENTION_FLOOR", "NUMERICS_WARN",
    "PHASE_ARMED_UNDISPATCHED", "PHASE_DEGRADED", "PHASE_REFUSED",
    "SOURCE_DECLARED", "SOURCE_SDK_DEFAULT",
    "VERDICT_DEGRADED", "VERDICT_DESTROYED", "VERDICT_HEALTHY",
    "Comparison", "RowStat", "Thresholds",
    "compare_outputs", "declared_thresholds", "flatten_outputs", "gate",
    "worst_of",
]
