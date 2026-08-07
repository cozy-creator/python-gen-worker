"""Fail-closed adapter-fidelity gate: does the delta SURVIVE the target grid?

pgw#794 §3 measured what nobody was measuring: an adapter fused into fp8-E4M3
weights does not merely vanish, it CORRUPTS. fp8-E4M3's half-ulp is 3.1-6.25%
of each weight (3 mantissa bits) against bf16's 0.20-0.39% — 16x coarser — so
``Q(W + BA) == Q(W)`` for every element whose own delta is below half an ulp,
while the few elements that happen to sit near a cell boundary jump a FULL ulp.
The qwen Lightning row is the shape of the hazard: surviving-delta cosine
**0.074** with **15x the norm** of the true delta, i.e. the served weights carry
a perturbation larger than the adapter and 99.7% orthogonal to it.

The guards that existed could not see this:

* ``utils.lora._reject_zero_delta`` (th#1036) catches an EMPTY adapter — no
  down/up pair with both halves nonzero. An INERT adapter is fully populated;
  it dies in the cast, not in the file.
* te#86's produce-time detector (``conversion/fuse_checks.py``) is advisory by
  construction, speaks only above a 50%-of-tensors threshold, and diffs in the
  SOURCE dtype — the fp8 cast is a separate later step, so the 16x-larger loss
  is measured by nothing.

So gen-worker owns the SERVE-side gate, and this module is it. The rule is one
sentence: **evaluate the delta against the grid it will actually serve
through, per element, and refuse when too little of it survives.**

Two things follow from "the grid it will ACTUALLY serve through", and both are
load-bearing:

1. The grid is read off the real destination — :func:`grid_of_module` inspects
   the module (fp8 storage + ``weight_scale`` => fp8-E4M3 per-out-channel;
   bf16/fp16 weights => that dtype, ungridded). A future fp8-branch variant is
   therefore gated the day it exists, with no edit here.
2. **An adapter is refused only on the path that destroys it.** The resident
   branch keeps A and B as separate bf16 tensors and never touches the base
   grid, so the same qwen/sdxl adapters that fuse at cosine 0.07-0.35 ride the
   branch at cosine 1.000000 (measured, this lane) — and even a hypothetical
   fp8 BRANCH measures 0.9998, because rounding the two low-rank FACTORS is
   nothing like rounding their sum onto an occupied grid. Over-refusal is its
   own failure and the evaluator's shape prevents it.

Coverage: every path in this worker where an adapter reaches weights or
arithmetic.

* resident-branch buffer copy (rank buckets) — gated in ``w8a8_lora._stage_for``,
  which is the single seam BOTH the plain buffer path and the lifted-binding
  path funnel through (``apply_branch_adapters`` / ``apply_branch_adapter_set``
  / ``LiftedLoraBinding.swap`` / ``swap_lifted_lane_set``). It is the pure pass
  that runs before any module is touched, so a refusal leaves the pipeline
  exactly as it was.
* fuse — :func:`gate_fuse` is the seam any fuse must call. gen-worker ships no
  fuse today (pgw#794 §8's platform ruling: never fuse into a denoiser that
  serves quantized), so this is the gate that makes the ruling enforceable
  rather than remembered, and the entry point for the produce-time fuse in
  ``training-endpoints`` (te#86) to adopt.
* peft — the remaining adapter surface (text-encoder halves, and the plain-lane
  whole-adapter fallback) stores A/B as separate module parameters exactly like
  the branch, and is already refused outright against a cast text encoder
  (``utils.lora._reject_te_keys_on_cast_te``, ie#374). It never lands on a
  quantized grid, so it carries no instance of this hazard class.
"""

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

# ---------------------------------------------------------------------------
# Policy (pgw#794). Named and derived, never a buried constant.
# ---------------------------------------------------------------------------

#: The gated quantity is the **norm-weighted whole-adapter cosine** between the
#: delta that SURVIVES the target grid and the delta the adapter actually
#: encodes. 1.0 = the served arithmetic is the adapter; 0.0 = what survived is
#: orthogonal noise. Cosine, not retention: pgw#794's qwen row has retention
#: 15.3 *because* it is destroyed, so a norm ratio cannot tell "intact" from
#: "replaced by something bigger and unrelated".

#: Hard floor — below this the attach/fuse is REFUSED, typed.
#:
#: Derived from the measured empty band, not chosen:
#:
#:   worst configuration we ACCEPT     0.900  sdxl lightning-4step, bf16 fuse,
#:                                            whole adapter, 788 real modules
#:                                            (this lane; pgw#794 §3 reports
#:                                            0.918 on its 53-module sample).
#:                                            This is the configuration that
#:                                            circulates publicly as a merged
#:                                            fp16 checkpoint and renders.
#:   best configuration we REFUSE      0.689  z-image fun-distill-8step, fp8
#:                                            fuse — the STRONGEST adapter we
#:                                            ship (rel |D|/|W| = 2.5%) and
#:                                            still half destroyed (pgw#794 §3).
#:
#: Nothing was measured between them. 0.80 is the geometric midpoint of that
#: band (sqrt(0.900 * 0.689) = 0.787) rounded to two figures: 1.13x of headroom
#: below the worst accepted case and 1.16x above the best refused one, so the
#: boundary is not near either edge of the evidence. Every other measured fp8
#: fuse is far below it (sdxl lightning 0.25-0.35, dmd2 0.26-0.28, qwen 0.074),
#: and every branch attach is far above (bf16 1.000000, fp8 0.9998).
FIDELITY_FLOOR = 0.80

#: Gray band ceiling — at or above this the attach is silent; in
#: ``[FIDELITY_FLOOR, FIDELITY_WARN)`` it proceeds but confesses with a typed
#: ``lora_fidelity phase=degraded`` event.
#:
#: A healthy fuse measures 0.999+ and a branch attach measures 1.000000, so
#: anything that has lost 1% of the delta's DIRECTION has lost it to the grid
#: and is worth counting hub-side. Deliberately wide: sdxl's turbo adapters
#: fusing at bf16 (0.90) land here — degraded, known, and served, which is the
#: honest verdict for them.
FIDELITY_WARN = 0.99

#: Typed hub-visible event phases for :data:`~gen_worker.activity.KIND_LORA_FIDELITY`
#: (pgw#760 doctrine: a decision that changes what this worker serves rides an
#: event, never a log line).
PHASE_REFUSED = numerics_ladder.PHASE_REFUSED
PHASE_DEGRADED = numerics_ladder.PHASE_DEGRADED

VERDICT_HEALTHY = numerics_ladder.VERDICT_HEALTHY
VERDICT_DEGRADED = numerics_ladder.VERDICT_DEGRADED
VERDICT_DESTROYED = numerics_ladder.VERDICT_DESTROYED

#: pgw#817: this population's calibration of the SHARED ladder
#: (:mod:`gen_worker.numerics_ladder`). ``retention_floor=0`` deliberately —
#: a destroyed adapter's retention is 15.3, so here the norm ratio is evidence
#: and never a bound. The output-comparison population gates it; this one
#: cannot.
ADAPTER_THRESHOLDS = numerics_ladder.Thresholds(
    floor=FIDELITY_FLOOR, warn=FIDELITY_WARN, retention_floor=0.0,
    label="adapter-delta (pgw#794 §3)")

#: torch float8_e4m3fn's finite max. The cast does NOT saturate, so the
#: producer clamps first (``convert/writer.py``); the grid here mirrors it.
_FP8_E4M3_MAX = 448.0

_GRID_NONE = "none"
_GRID_ROW = "per-out-channel"

PATH_BRANCH = "branch"
PATH_FUSE = "fuse"


# ---------------------------------------------------------------------------
# What the delta lands on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetGrid:
    """The arithmetic destination a delta is evaluated against.

    ``path`` says HOW the adapter reaches the model (``branch`` keeps A and B
    as separate tensors; ``fuse`` writes ``Q(W + BA)`` over the base weights),
    ``dtype``/``granularity`` say what the numbers land in. Both ride every
    refusal so the error names the grid it judged, not a grid inferred later.
    """

    path: str
    dtype: str
    granularity: str = _GRID_NONE

    def __str__(self) -> str:
        gran = "" if self.granularity == _GRID_NONE else f", {self.granularity}"
        return f"{self.path}:{self.dtype}{gran}"


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


class UnknownComputeDtypeError(RuntimeError):
    """A branch-capable module cannot state the dtype its arithmetic lands in.

    pgw#1019's ruling on pgw#1015. The branch-capable universe is CLOSED —
    ``w8a8_lora.branch_modules`` admits exactly ``_Fp8ScaledLinear`` and
    modules whose ``fp8_storage.structural_base`` is ``nn.Linear``/
    ``nn.Conv2d`` — and every member of it can state a compute dtype:

    * a plain ``nn.Linear``/``nn.Conv2d`` computes in its weight's dtype, by
      definition;
    * a pgw#727 fp8-storage leaf records ``compute_dtype`` at restructure
      (``fp8_storage.restructure_fp8_storage``), because its own ``forward``
      upcasts to it;
    * ``_Fp8ScaledLinear`` records it in ``__init__`` (pgw#1015).

    So there is no dtype-less caller left, and a default here can only be a
    module that FORGOT to record the fact — which is exactly what pgw#1015
    was: a bias-free fp8 layer silently taking bf16 while its bias-bearing
    siblings in the same module set took float32, and the first
    branch-bearing forward dying on ``expected mat1 and mat2 to have the same
    dtype``. "Nobody could tell" is not "bf16".
    """


def branch_compute_dtype(mod: Any) -> Any:
    """The dtype a branch buffer is allocated in for this module.

    THE definition — ``w8a8_lora.alloc_branch_buffers`` calls it, so the gate
    and the allocator can never disagree about what grid the branch is. Branch
    tensors compute in the module's COMPUTE dtype, never its storage dtype: on
    the fp8-storage lane weight AND bias rest in fp8, and a pgw#727 leaf
    declares ``compute_dtype`` for exactly this question.

    REFUSES rather than defaulting (pgw#1019) — see
    :class:`UnknownComputeDtypeError` for why every legitimate caller can
    answer.
    """
    import torch

    compute = (torch.float16, torch.bfloat16, torch.float32)
    weight = getattr(mod, "weight", None)
    bias = getattr(mod, "bias", None)
    for cand in (getattr(mod, "compute_dtype", None),
                 None if weight is None else weight.dtype,
                 None if bias is None else bias.dtype):
        # An fp8 storage dtype is never a compute dtype, so a quantized
        # weight is skipped here by the membership test itself.
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
    """The dtype this module's branch A/B ACTUALLY land in.

    A live branch buffer is the ground truth and outranks the allocation rule:
    the day someone ships an fp8-branch variant, it allocates fp8 ``lora_a``
    and this gate judges the delta on the fp8 grid with no edit here — which is
    the whole point of reading the destination instead of being told it. Falls
    back to :func:`branch_compute_dtype` before the branch is armed.
    """
    for name in ("lora_a", "lora_b"):
        t = getattr(mod, name, None)
        dtype = getattr(t, "dtype", None)
        if dtype is not None:
            return dtype
    return branch_compute_dtype(mod)


def _has_weight_scale(mod: Any) -> bool:
    scale = getattr(mod, "weight_scale", None)
    return scale is not None and hasattr(scale, "numel")


def grid_of_module(mod: Any, *, path: str) -> TargetGrid:
    """The REAL grid this module's arithmetic lands on — read off the module,
    never supplied by the caller. That is the whole correction to te#86's
    detector, which diffed in the source dtype and so could not see the cast.
    """
    import torch

    if path == PATH_BRANCH:
        return TargetGrid(PATH_BRANCH, _dtype_name(branch_grid_dtype(mod)))
    weight = getattr(mod, "weight", None)
    if weight is None:
        raise ValueError(f"module {type(mod).__name__} carries no weight to fuse into")
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        gran = _GRID_ROW if _has_weight_scale(mod) else _GRID_NONE
        return TargetGrid(PATH_FUSE, _dtype_name(weight.dtype), gran)
    return TargetGrid(PATH_FUSE, _dtype_name(weight.dtype))


def quantizer_for(grid: TargetGrid) -> Callable[[Any], Any]:
    """The round-trip ``x -> dequant(quant(x))`` for one grid, in fp32.

    The fp8 rung mirrors the producer byte for byte (``convert/writer.py``
    ``scale = amax(row)/448``; ``q = round(w/scale)`` in fp32, clamped because
    torch's cast does not saturate) and the consumer's per-``[out, 1]`` scale
    normalization (``w8a8._scale_2d``). Evaluating against anything else would
    be judging a grid we do not serve.
    """
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


# ---------------------------------------------------------------------------
# Survival records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSurvival:
    """What one module's delta survived as. The per-module evidence a refusal
    carries, so the error is actionable from hub records alone."""

    module: str
    elements: int
    #: ``|D| / |W|`` — the ratio that PREDICTS every row in pgw#794 §3, read
    #: against the grid's half-ulp. 0.0 on the branch path (no base involved).
    rel_delta: float
    #: ``|D'| / |D|``. Above 1.0 means the grid substituted something BIGGER
    #: than the adapter — see the qwen row (15.3).
    retention: float
    #: ``cos(D', D)`` — the honest fidelity number.
    cosine: float
    #: Fraction of elements the write actually moved. On a fuse this is
    #: pgw#794's ``above-half-ulp``: the elements whose own delta cleared half
    #: an ulp of their own weight. 0.0 on the branch path.
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
    #: The GATED quantity: norm-weighted over every module, i.e. the cosine of
    #: the concatenated surviving delta against the concatenated true delta.
    #: One number for the whole adapter, which is what "did this adapter
    #: survive" means — a per-module median would let a handful of destroyed
    #: high-norm blocks hide behind many intact low-norm ones.
    cosine: float
    retention: float

    @property
    def verdict(self) -> str:
        # pgw#817: ONE ladder, calibrated per population. The rungs and the
        # ordering live in `numerics_ladder`; only the numbers are ours.
        return ADAPTER_THRESHOLDS.verdict(self.cosine, self.retention)

    def worst(self, limit: int = 5) -> Tuple[ModuleSurvival, ...]:
        return tuple(sorted(self.modules, key=lambda m: m.cosine)[:limit])

    def evidence(self, limit: int = 5) -> str:
        """Everything a reader needs without the worker's logs: identity, the
        grid judged, the aggregate, and the worst modules with their numbers."""
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _as_2d(t: Any) -> Any:
    """A [r, ...] or [out, r(, 1, 1)] adapter half as a 2-D matrix. Conv pairs
    (gw#627) fold the kernel into the input axis — the delta's element set is
    unchanged, and the Gram identities below only need consistent shapes."""
    return t.reshape(t.shape[0], -1) if t.dim() != 2 else t


def _gram_cosine(a: Any, b: Any, a2: Any, b2: Any) -> Tuple[float, float, float]:
    """``(<D', D>, |D|^2, |D'|^2)`` for ``D = B A`` and ``D' = B' A'``, EXACTLY,
    without ever materializing an ``[out, in]`` delta.

    ``tr(D'^T D) = tr((B'^T B)(A A'^T))`` — two ``r x r`` Grams costing
    ``O((in + out) r^2)`` instead of the ``O(in * out * r)`` a materialized
    delta would cost. At SDXL Lightning scale (788 real modules, rank 64) the
    whole adapter evaluates in 0.86 s on 4 CPU threads, which is why this can
    sit on the cold attach path at all.
    """
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
    """Survival of a mapped adapter through the resident BRANCH.

    ``mapped`` is ``w8a8_lora.map_adapter``'s output (path -> (A, B, alpha)),
    ``modules`` is ``w8a8_lora.branch_modules``'s. The branch keeps A and B as
    separate tensors in the module's compute dtype, so what the grid can
    destroy is the FACTORS, not their sum against an occupied base grid — and
    that is why the same adapters that fuse at cosine 0.07 ride the branch at
    1.000000. Returns None when nothing evaluable was mapped.
    """
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
                # Mixed-dtype branch coverage: judge the whole adapter on the
                # COARSEST grid any of its modules lands on — the gate must
                # never be softened by the friendliest layer in the set.
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
    """Survival of a mapped adapter FUSED into the base weights, per element.

    For each module: ``W' = Q(W + D)`` against ``W0 = Q(W)`` on the module's
    real grid, exactly pgw#794 §3's protocol. ``retention``, ``cosine`` and the
    moved-element fraction are the three numbers that lane measured, so a
    refusal here is directly comparable to its table.

    Unlike the branch evaluator this materializes ``D`` — a fuse materializes
    ``W + D`` anyway, so there is nothing cheaper to be had and nothing to
    approximate. ``grid`` overrides the per-module reading for the case where
    a caller intends to fuse into a dtype the modules do not yet hold (e.g.
    proving a bf16 fuse before the fp8 cast that follows it).
    """
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
    """The module's base weights in real values — fp8 storage dequantized
    through its own ``weight_scale``, so ``W + D`` is added in the numbers the
    module actually computes with rather than in raw fp8 codes."""
    w = mod.weight
    if _has_weight_scale(mod):
        scale = mod.weight_scale.float().reshape(int(w.shape[0]), 1)
        return (w.float().reshape(int(w.shape[0]), -1) * scale).reshape(w.shape)
    return w


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def gate(
    survival: Optional[AdapterSurvival],
    *,
    request_id: str = "",
    announce: bool = True,
) -> Optional[AdapterSurvival]:
    """Fail closed below :data:`FIDELITY_FLOOR`; confess in the gray band.

    Below the floor this raises :class:`AdapterFidelityRefused` AND emits a
    typed ``lora_fidelity phase=refused`` event, because the two audiences are
    different: the error reaches the caller of this request, the event reaches
    the hub's activity records where a fleet-wide pattern is countable. Both
    carry the same evidence, on EVERY refused request — a refusal is rare by
    construction and its rate is the signal.

    ``announce=False`` suppresses only the gray-band event, for a re-check of
    an already-confessed cached verdict: a degraded adapter is a property of
    (adapter, grid), so re-emitting it on every warm swap would turn one fact
    into a per-request stream.
    """
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
    """Gate one mapped adapter against a FUSE into ``modules``' weights.

    The seam every fuse must call BEFORE writing a single weight. pgw#794 §8's
    platform ruling is "never fuse an adapter into a denoiser that serves
    quantized"; this is the enforcement, and it is per-adapter evidence rather
    than a blanket dtype ban, so a genuinely fuse-safe adapter (rel |D|/|W|
    well above the grid's half-ulp) is not refused for its neighbours' sake.
    """
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
