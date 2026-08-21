"""Fail-closed adapter-fidelity gate: does the delta SURVIVE the target grid? An adapter fused into fp8-E4M3 weights does not merely vanish, it CORRUPTS."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from .. import activity as activity_mod
from .. import numerics_ladder
from ..activity import KIND_LORA_FIDELITY
from ..api.errors import AdapterFidelityRefused

logger = logging.getLogger(__name__)

FIDELITY_FLOOR = 0.80

FIDELITY_WARN = 0.99

PHASE_REFUSED = numerics_ladder.PHASE_REFUSED
PHASE_DEGRADED = numerics_ladder.PHASE_DEGRADED

VERDICT_HEALTHY = numerics_ladder.VERDICT_HEALTHY
VERDICT_DEGRADED = numerics_ladder.VERDICT_DEGRADED
VERDICT_DESTROYED = numerics_ladder.VERDICT_DESTROYED

ADAPTER_THRESHOLDS = numerics_ladder.Thresholds(
    floor=FIDELITY_FLOOR, warn=FIDELITY_WARN, retention_floor=0.0,
    label="adapter-delta (pgw#794 §3)")

_FP8_E4M3_MAX = 448.0

_GRID_NONE = "none"
_GRID_ROW = "per-out-channel"

PATH_BRANCH = "branch"
PATH_FUSE = "fuse"


@dataclass(frozen=True)
class TargetGrid:
    """The arithmetic destination a delta is evaluated against."""

    path: str
    dtype: str
    granularity: str = _GRID_NONE

    def __str__(self) -> str:
        gran = "" if self.granularity == _GRID_NONE else f", {self.granularity}"
        return f"{self.path}:{self.dtype}{gran}"


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


class UnknownComputeDtypeError(RuntimeError):
    """A branch-capable module cannot state the dtype its arithmetic lands in."""


def branch_compute_dtype(mod: Any) -> Any:
    """The dtype a branch buffer is allocated in for this module."""
    import torch

    compute = (torch.float16, torch.bfloat16, torch.float32)
    weight = getattr(mod, "weight", None)
    bias = getattr(mod, "bias", None)
    for cand in (getattr(mod, "compute_dtype", None),
                 None if weight is None else weight.dtype,
                 None if bias is None else bias.dtype):
        if cand in compute:
            return cand
    raise UnknownComputeDtypeError(
        f"{type(mod).__name__} states no compute dtype: `compute_dtype` is "
        f"{getattr(mod, 'compute_dtype', None)!r}, weight is "
        f"{_dtype_name(getattr(weight, 'dtype', None))}, bias is "
        f"{_dtype_name(getattr(bias, 'dtype', None))} — none of them a "
        "compute dtype (float16/bfloat16/float32). A LoRA branch allocated "
        "on a guess is the pgw#1015 defect: bias-bearing and bias-free "
        "layers of one module set land in DIFFERENT dtypes and the first "
        "branch-bearing forward dies inside torch. FIX: record it — "
        f"`self.compute_dtype = compute_dtype` in {type(mod).__name__}"
        ".__init__ (every quantized leaf in gen_worker.models already takes "
        "the parameter), or set it on the instance where the lane is armed, "
        "the way `fp8_storage.restructure_fp8_storage` does.")


def branch_grid_dtype(mod: Any) -> Any:
    """The dtype this module's branch A/B ACTUALLY land in."""
    for name in ("lora_a", "lora_b"):
        t = getattr(mod, name, None)
        dtype = getattr(t, "dtype", None)
        if dtype is not None:
            return dtype
    return branch_compute_dtype(mod)


def _has_weight_scale(mod: Any) -> bool:
    scale = getattr(mod, "weight_scale", None)
    return scale is not None and hasattr(scale, "numel")


def _is_gguf_leaf(mod: Any) -> bool:
    from .gguf_torch import LEAF_MARKER

    return bool(getattr(type(mod), LEAF_MARKER, False))


def grid_of_module(mod: Any, *, path: str) -> TargetGrid:
    """The REAL grid this module's arithmetic lands on — read off the module, never supplied by the caller."""
    import torch

    if path == PATH_BRANCH:
        return TargetGrid(PATH_BRANCH, _dtype_name(branch_grid_dtype(mod)))
    weight = getattr(mod, "weight", None)
    if weight is None:
        raise ValueError(f"module {type(mod).__name__} carries no weight to fuse into")
    if _is_gguf_leaf(mod):
        raise ValueError(
            f"module {type(mod).__name__} holds GGML block bytes: there is no "
            "fuse into a quantized grid to gate — attach the adapter instead "
            "(gguf_torch.attach_lora), which applies it post-dequant")
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        gran = _GRID_ROW if _has_weight_scale(mod) else _GRID_NONE
        return TargetGrid(PATH_FUSE, _dtype_name(weight.dtype), gran)
    return TargetGrid(PATH_FUSE, _dtype_name(weight.dtype))


def quantizer_for(grid: TargetGrid) -> Callable[[Any], Any]:
    """The round-trip ``x -> dequant(quant(x))`` for one grid, in fp32."""
    import torch

    dtype = getattr(torch, grid.dtype, None)
    if dtype is None:
        raise ValueError(f"unknown target dtype {grid.dtype!r}")

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        fmax = _FP8_E4M3_MAX if dtype is torch.float8_e4m3fn else 57344.0

        def _fp8(x: Any) -> Any:
            flat = x.float().reshape(x.shape[0], -1)
            if grid.granularity == _GRID_ROW:
                scale = (flat.abs().amax(dim=1, keepdim=True) / fmax).clamp(min=1e-12)
            else:
                scale = (flat.abs().amax() / fmax).clamp(min=1e-12)
            q = (flat / scale).clamp(-fmax, fmax).to(dtype)
            return (q.float() * scale).reshape(x.shape)

        return _fp8

    def _cast(x: Any) -> Any:
        return x.float().to(dtype).float()

    return _cast


@dataclass(frozen=True)
class ModuleSurvival:
    """What one module's delta survived as."""

    module: str
    elements: int
    rel_delta: float
    retention: float
    cosine: float
    moved_fraction: float

    def __str__(self) -> str:
        return (f"{self.module} cos={self.cosine:.3f} ret={self.retention:.3f} "
                f"rel={self.rel_delta:.2e} moved={self.moved_fraction:.1%} "
                f"n={self.elements}")


@dataclass(frozen=True)
class AdapterSurvival:
    """One adapter's whole-model survival on one grid."""

    ref: str
    grid: TargetGrid
    modules: Tuple[ModuleSurvival, ...]
    cosine: float
    retention: float

    @property
    def verdict(self) -> str:
        return ADAPTER_THRESHOLDS.verdict(self.cosine, self.retention)

    def worst(self, limit: int = 5) -> Tuple[ModuleSurvival, ...]:
        return tuple(sorted(self.modules, key=lambda m: m.cosine)[:limit])

    def evidence(self, limit: int = 5) -> str:
        """Everything a reader needs without the worker's logs: identity, the grid judged, the aggregate, and the worst modules with their numbers."""
        head = (
            f"adapter={self.ref or '<unnamed>'} grid={self.grid} "
            f"modules={len(self.modules)} cosine={self.cosine:.4f} "
            f"retention={self.retention:.3f} verdict={self.verdict} "
            f"floor={FIDELITY_FLOOR:g} warn={FIDELITY_WARN:g}"
        )
        worst = self.worst(limit)
        if not worst:
            return head
        return head + " | worst: " + "; ".join(str(m) for m in worst)


def _as_2d(t: Any) -> Any:
    return t.reshape(t.shape[0], -1) if t.dim() != 2 else t


def _gram_cosine(a: Any, b: Any, a2: Any, b2: Any) -> Tuple[float, float, float]:
    cross = float(((b2.T @ b) * (a @ a2.T).T).sum())
    true_sq = float(((b.T @ b) * (a @ a.T).T).sum())
    surv_sq = float(((b2.T @ b2) * (a2 @ a2.T).T).sum())
    return cross, true_sq, surv_sq


def _finish(ref: str, grid: TargetGrid, mods: Sequence[ModuleSurvival],
            cross: float, true_sq: float, surv_sq: float) -> AdapterSurvival:
    denom = math.sqrt(max(true_sq, 0.0)) * math.sqrt(max(surv_sq, 0.0))
    cosine = (cross / denom) if denom > 0.0 else 0.0
    retention = (math.sqrt(surv_sq / true_sq) if true_sq > 0.0 else 0.0)
    return AdapterSurvival(
        ref=ref, grid=grid, modules=tuple(mods),
        cosine=max(-1.0, min(1.0, cosine)), retention=retention)


def evaluate_branch(
    mapped: Mapping[str, Tuple[Any, Any, float]],
    modules: Mapping[str, Any],
    *,
    ref: str = "",
) -> Optional[AdapterSurvival]:
    """Survival of a mapped adapter through the resident BRANCH."""
    import torch

    if not mapped:
        return None
    grid: Optional[TargetGrid] = None
    rows: List[ModuleSurvival] = []
    cross = true_sq = surv_sq = 0.0
    with torch.no_grad():
        for path in sorted(mapped):
            a_raw, b_raw, alpha = mapped[path]
            mod = modules.get(path)
            if mod is None:
                continue
            mgrid = grid_of_module(mod, path=PATH_BRANCH)
            if grid is None:
                grid = mgrid
            elif mgrid != grid:
                grid = min((grid, mgrid), key=lambda g: _dtype_rank(g.dtype))
            dtype = getattr(torch, mgrid.dtype)
            a = _as_2d(a_raw).float()
            b = _as_2d(b_raw).float()
            a2 = _as_2d(a_raw.to(dtype)).float()
            b2 = _as_2d(b_raw.to(dtype)).float()
            s2 = float(alpha) ** 2
            c, t, s = _gram_cosine(a, b, a2, b2)
            c, t, s = c * s2, t * s2, s * s2
            cross += c
            true_sq += t
            surv_sq += s
            den = math.sqrt(max(t, 0.0)) * math.sqrt(max(s, 0.0))
            rows.append(ModuleSurvival(
                module=path, elements=int(b.shape[0]) * int(a.shape[1]),
                rel_delta=0.0,
                retention=(math.sqrt(s / t) if t > 0.0 else 0.0),
                cosine=(c / den) if den > 0.0 else 0.0,
                moved_fraction=0.0,
            ))
    if grid is None:
        return None
    return _finish(ref, grid, rows, cross, true_sq, surv_sq)


_DTYPE_ORDER = ("float8_e5m2", "float8_e4m3fn", "bfloat16", "float16", "float32", "float64")


def _dtype_rank(name: str) -> int:
    try:
        return _DTYPE_ORDER.index(name)
    except ValueError:
        return len(_DTYPE_ORDER)


def evaluate_fuse(
    mapped: Mapping[str, Tuple[Any, Any, float]],
    modules: Mapping[str, Any],
    *,
    ref: str = "",
    grid: Optional[TargetGrid] = None,
) -> Optional[AdapterSurvival]:
    """Survival of a mapped adapter FUSED into the base weights, per element."""
    import torch

    if not mapped:
        return None
    rows: List[ModuleSurvival] = []
    cross = true_sq = surv_sq = 0.0
    chosen: Optional[TargetGrid] = grid
    with torch.no_grad():
        for path in sorted(mapped):
            a_raw, b_raw, alpha = mapped[path]
            mod = modules.get(path)
            if mod is None:
                continue
            mgrid = grid or grid_of_module(mod, path=PATH_FUSE)
            if chosen is None:
                chosen = mgrid
            elif mgrid != chosen:
                chosen = min((chosen, mgrid), key=lambda g: _dtype_rank(g.dtype))
            quant = quantizer_for(mgrid)
            w = _base_weight(mod).float()
            a = _as_2d(a_raw).float()
            b = _as_2d(b_raw).float()
            delta = ((b @ a) * float(alpha)).reshape(w.shape)
            surv = quant(w + delta) - quant(w)
            c = float((surv * delta).sum())
            t = float((delta * delta).sum())
            s = float((surv * surv).sum())
            cross += c
            true_sq += t
            surv_sq += s
            den = math.sqrt(max(t, 0.0)) * math.sqrt(max(s, 0.0))
            wn = float(w.norm())
            rows.append(ModuleSurvival(
                module=path, elements=int(delta.numel()),
                rel_delta=(math.sqrt(t) / wn) if wn > 0.0 else 0.0,
                retention=(math.sqrt(s / t) if t > 0.0 else 0.0),
                cosine=(c / den) if den > 0.0 else 0.0,
                moved_fraction=float((surv != 0).sum()) / max(1, delta.numel()),
            ))
    if chosen is None:
        return None
    return _finish(ref, chosen, rows, cross, true_sq, surv_sq)


def _base_weight(mod: Any) -> Any:
    w = mod.weight
    if _has_weight_scale(mod):
        scale = mod.weight_scale.float().reshape(int(w.shape[0]), 1)
        return (w.float().reshape(int(w.shape[0]), -1) * scale).reshape(w.shape)
    return w


def gate(
    survival: Optional[AdapterSurvival],
    *,
    request_id: str = "",
    announce: bool = True,
) -> Optional[AdapterSurvival]:
    """Fail closed below :data:`FIDELITY_FLOOR`; confess in the gray band."""
    if survival is None:
        return None
    verdict = survival.verdict
    if verdict == VERDICT_HEALTHY:
        return survival
    detail = survival.evidence()
    if verdict == VERDICT_DEGRADED:
        logger.warning("[request_id=%s] adapter fidelity degraded: %s",
                       request_id, detail)
        if announce:
            activity_mod.emit_event(
                KIND_LORA_FIDELITY,
                f"request={request_id} {detail}", phase=PHASE_DEGRADED)
        return survival
    activity_mod.emit_event(
        KIND_LORA_FIDELITY,
        f"request={request_id} {detail}", phase=PHASE_REFUSED)
    raise AdapterFidelityRefused(
        f"the adapter's delta does not survive the grid it would serve "
        f"through: whole-adapter cosine {survival.cosine:.4f} is below the "
        f"{FIDELITY_FLOOR:g} fidelity floor on {survival.grid}. What survives "
        f"is {survival.retention:.2f}x the true delta and "
        f"{100.0 * (1.0 - max(0.0, survival.cosine)):.1f}% orthogonal to it — "
        f"serving this would look adapted and would not be (pgw#794 §3). "
        f"{detail}",
        ref=survival.ref, survival=survival,
    )


def gate_branch(
    mapped: Mapping[str, Tuple[Any, Any, float]],
    modules: Mapping[str, Any],
    *,
    ref: str = "",
    request_id: str = "",
) -> Optional[AdapterSurvival]:
    """Gate one mapped adapter against the resident branch it will ride."""
    return gate(evaluate_branch(mapped, modules, ref=ref), request_id=request_id)


def gate_fuse(
    mapped: Mapping[str, Tuple[Any, Any, float]],
    modules: Mapping[str, Any],
    *,
    ref: str = "",
    request_id: str = "",
    grid: Optional[TargetGrid] = None,
) -> Optional[AdapterSurvival]:
    """Gate one mapped adapter against a FUSE into ``modules``' weights."""
    return gate(evaluate_fuse(mapped, modules, ref=ref, grid=grid),
                request_id=request_id)


__all__ = [
    "FIDELITY_FLOOR", "FIDELITY_WARN", "KIND_LORA_FIDELITY",
    "PHASE_DEGRADED", "PHASE_REFUSED", "PATH_BRANCH", "PATH_FUSE",
    "VERDICT_DEGRADED", "VERDICT_DESTROYED", "VERDICT_HEALTHY",
    "AdapterSurvival", "ModuleSurvival", "TargetGrid",
    "UnknownComputeDtypeError",
    "branch_compute_dtype", "branch_grid_dtype", "evaluate_branch",
    "evaluate_fuse", "gate",
    "gate_branch", "gate_fuse", "grid_of_module", "quantizer_for",
]
