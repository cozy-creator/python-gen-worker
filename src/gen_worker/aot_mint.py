"""The AOT mint — export + AOTInductor-package a family's WHOLE declared
class set as ONE multi-graph cell (pgw#704 GO, #723 mint path, #758 packaging).

    compose -> per declared class: torch.export.export -> aot_compile(code-only)
            -> per-entry gates -> package_aoti({entry: files}) -> pack -> publish

Paul's ruling (pgw#758): "generate and generate_turbo are separate functions,
they have separate graphs, but they are COMBINED TOGETHER INTO ONE FILE." One
mint invocation produces one cell per (family x lane x contract) carrying every
declared graph class as a NAMED ENTRY — which removes the one-artifact-per-pod
serving ceiling the pilot runbook accepted.

``aot_serve`` owns the ENVELOPE — metadata contract, ``pack``, ``verify`` (#721
S1 / #723 S1: ONE source of truth, imported by both lanes, never re-declared)
— and consumes the result. ``aot_package`` reads facts back out of a compiled
``.pt2`` (per entry) and holds the B1 gate. ``lora_lifted`` owns the
no-baked-adapter gate. This module drives PRODUCTION and nothing else.
Deliberately NOT folded into ``compile_cache``:
``trt_engine`` already established that a compiled-lane backend is its own
module riding the compile-cache rails, and the dynamo mint stays live and
fully-forced in parallel during rollout (#722: nothing retires before sdxl AOT
is live in prod).

Why the w8a8 lane first
-----------------------
pgw#704 measured w8a8 at 276.2 vs 274.6 ms — latency PARITY with dynamo, and
numerics identical. The plain and fp8-storage lanes both carry an unexplained
systematic ~7% AOTI regression (#730 owns it), so they mint only behind
``allow_regressed_lanes``. That is not a preference: shipping a lane we
measured 7% slower, while calling the migration a win, would be a regression
sold as progress.

Where minting runs (#724 REJECTED — Paul, 2026-07-28)
-----------------------------------------------------
Serving pods background-mint their own cells under the proven pgw#677
eager-first machinery — "I'd rather keep that, rather than a whole complex
separate compilation system; our compilation system would only ever just be
running the endpoint code we have already anyway." There is no dedicated mint
fleet. ``python -m gen_worker.aot_mint`` stays CLI-invokable for ops and
testing. Mint cost is INSTRUMENTED, not assumed: every mint records a
per-phase, per-entry ``mint_phases`` table (export / lowering / codegen /
triton / host C++ compile+link) plus the graph-class count and the autotune
posture, so an AOT-vs-JIT comparison is labeled data, never folklore (#757
consumes it).

What is exported
----------------
STRICT export, by default. pgw#704 confirmed the sdxl UNet passes strict AND
non-strict, static AND dynamic, so strict is chosen for what it buys: it
refuses python side effects instead of silently baking them, which is exactly
the failure class an artifact serving live traffic must not have. Non-strict
stays available per-mint for a family that needs it (z-image's ragged list
inputs, #729) and is RECORDED in the key, because a non-strict trace is a
different guarantee and must not share identity with a strict one.

The two gates that make an artifact publishable
-----------------------------------------------
**B1 code-only.** Compiled with ``package_constants_in_so=False``, then PROVEN
so by ``aot_package.code_only_violations`` — two independent structural proofs,
failing red and naming the tensors. See that module for why weights in a cell
would destroy the CAS distribution model.

**No baked adapter.** ``lora_lifted.assert_no_baked_adapter`` runs at pack time
on the **ExportedProgram**, not on the package: packing renames a
plain-``__dict__`` adapter to ``_tensor_constant0``, so a package-side FQN scan
is a false PASS on the plain-bf16 and fp8-hooks lanes (pgw#725, measured). The
ep-level gate is also free — it needs no packing at all.

Both gates run before pack, so a violating artifact never becomes a file that a
later step could upload.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import (
    aot_compile_pool, aot_package, aot_regional, aot_serve, aot_wrapper_split,
    cell_key)
from .compile_cache import (
    _resolve_target,
    lane_bucket,
    lane_token,
    toolchain_present,
)

logger = logging.getLogger(__name__)

#: Lanes measured at latency parity under AOTI (pgw#704 Q4). Everything else
#: needs ``allow_regressed_lanes`` — see the module docstring.
PARITY_LANES = ("w8a8", "w8a8-rowwise")

#: Lanes measured SLOWER under AOTI, held on dynamo by #730 until explained.
REGRESSED_LANES = ("", "fp8-hooks", "fp8-storage")

#: The inductor config that makes the package code-only. Not a knob: B1.
CODE_ONLY_CONFIGS: Dict[str, Any] = {
    "aot_inductor.package_constants_in_so": False,
}


#: pgw#790: the SDK-owned fork coordinate of a LoRA-bucket family — one graph
#: class WITH the lifted adapter inputs and one WITHOUT.
#:
#: gw#627 gave every branch-capable leaf a canonical zeroed rank-bucket branch
#: so a curated attach is a buffer copy instead of a recompile. Measured
#: (WARM-INFERENCE-MATRIX §2b, 4090+5090, n=28 warm): those zeroed branches
#: cost **+31.8% / +44.9% of the compiled per-forward** and roughly DOUBLE the
#: kernel-launch count, and adapter-free traffic pays all of it to compute
#: zeros. On the hub's own record 95% of sdxl denoiser forwards name no
#: adapter, so the branch-bearing graph was the minority case wearing the
#: majority's clothes.
#:
#: This is a FORK, not a flag: both classes are exported, compiled, keyed and
#: shipped in the same cell (Paul: "worst case compile 2x more graphs, one with
#: LoRAs and one without"), and the serve path picks between them by the
#: DECLARED ingress contract. Nothing about a program varies with Python state
#: — the arm is a mint coordinate that lands in the class hash, exactly like
#: any other fork.
ADAPTER_FORK = "adapter"


class MintRefused(RuntimeError):
    """A named, terminal refusal to produce or publish an artifact.

    Every mint failure is one of these with a reason that names the offending
    thing — a lane, a tensor, a missing declaration field. A mint that cannot
    say what went wrong is the silent-failure path the doctrine forbids.

    ``mint_phases`` carries the PARTIAL phase table of the mint that refused
    (pgw#825): the entries that did export and compile before the refusal
    spent real minutes on a real pod, and a terminus that reports only a
    wall-clock total is a measurement lost to a pod that no longer exists.
    Populated by :func:`mint`; empty for a refusal raised before any entry.
    """

    def __init__(self, *args: Any, mint_phases: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(*args)
        self.mint_phases: Dict[str, Any] = dict(mint_phases or {})


# ---------------------------------------------------------------------------
# The declared export contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicDim:
    """One declared symbolic dimension of one input.

    ``multiple_of`` expresses a divisibility the graph genuinely requires (SDXL
    latents downsample 3x, so latent H/W must be multiples of 8) as a Dim
    factor rather than a comment. Without it export 0/1-specializes or produces
    a guard it cannot express, and the "one artifact serves every aspect ratio"
    headline quietly stops holding.

    ``dim`` is the DECLARED ``Compile.Dim`` this row came from, and it is what
    makes a multi-carrier dim expressible (pgw#812 D1). A declaration may bind
    ONE logical axis to several inputs — flux2 declares
    ``Dim("T_img", carried_by=(("hidden_states", 1), ("img_ids", 1)))``
    precisely so the edit lane cannot let ``img_ids`` specialize while
    ``hidden_states`` stays free. Every carrier of one declared dim must
    therefore share ONE torch symbol; rows that carry no declared name (the
    hand-registered builder path, where latent H and W are genuinely
    independent axes of one input) keep a symbol each.
    """

    input_name: str
    axis: int
    min: int
    max: int
    multiple_of: int = 1
    dim: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "input": self.input_name, "axis": self.axis,
            "min": self.min, "max": self.max, "multiple_of": self.multiple_of,
        }


@dataclass
class ExportSpec:
    """Everything the mint needs to produce one artifact.

    ``example_inputs`` is a zero-arg factory rather than tensors so a caller
    can build them on meta (tests, control-plane) or on cuda (a real mint)
    without this module deciding. ``lora_fqns`` names the adapter tensors that
    must stay dynamic; ``lifted_inputs`` names what was promoted to a graph
    input to keep them that way.
    """

    family: str
    target: str
    weight_lane: str = ""
    precision: str = "bf16"
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, ...], ...] = ()
    #: Traced batch. 0 = the family's input builder decides. Declared rather
    #: than inferred from ``guidance_scales`` because CFG batching is a FAMILY
    #: fact: sdxl runs CFG as one batch-2 forward, wan as two sequential batch-1
    #: forwards, so guidance changes wan's call COUNT and not its shape
    #: (ie#566 G2).
    batch: int = 0
    text_lens: Tuple[int, ...] = ()
    guidance_scales: Tuple[float, ...] = ()
    dynamic: Tuple[DynamicDim, ...] = ()
    #: pgw#739 declaration coordinate: the fork arm values and (for a
    #: static-rows family) the class row this artifact is minted at. Both
    #: KEY (a fork is a distinct graph class in #716's hash) — see
    #: :func:`cell_identity`. Sorted (name, value) pairs.
    fork: Tuple[Tuple[str, Any], ...] = ()
    class_dims: Tuple[Tuple[str, int], ...] = ()
    specialization: Dict[str, Any] = field(default_factory=dict)
    lora_fqns: Tuple[str, ...] = ()
    lifted_inputs: Tuple[str, ...] = ()
    strict: bool = True
    source_ref: str = ""
    source_digest: str = ""
    closure_roots: Tuple[str, ...] = ()
    #: pgw#817: this cell's entries are BLOCK CLASSES of the target, not shape
    #: coordinates of its whole forward. Keys the ck5 ``mode`` axis, so a
    #: regional cell can never be confused with a whole-graph cell of the same
    #: family x lane x sm.
    regional: bool = False
    #: pgw#812 S3.3, MANDATORY for a regional cell. A regional artifact
    #: describes a PART of the model, so ``combined_graph_hash`` — which is a
    #: proxy for "the graph the fleet serves" — no longer covers the assembly.
    #: Two models with identical blocks and different shells must not collide.
    #: Computed by :func:`aot_regional.shell_digest` off the resolved module.
    shell_digest: str = ""

    def lane_label(self) -> str:
        base, observed = lane_bucket(self.weight_lane)
        bucket = observed or self.lora_bucket
        token = lane_token(base)
        if bucket:
            return f"{token}-lora{bucket}" if token else f"lora{bucket}"
        return token


#: The lifted-LoRA mint's torch floor (pgw#723 residuals, pod 8): torch 2.9
#: strict export refuses ``bind_views``' in-trace ``mod.lora_a = ...`` setattr
#: ("AssertionError: Mutating module attribute lora_a during export") that
#: 2.13 traces fine. 2.9 is NOT a valid fallback for this lane.
LIFTED_LORA_TORCH_FLOOR = (2, 13)


def lifted_torch_gap(spec: ExportSpec) -> str:
    """'' when torch meets the lifted-LoRA floor (or the spec has no lifted
    fork declared), else the named refusal reason."""
    if not (spec.lora_bucket or spec.lifted_inputs or spec.lora_fqns):
        return ""
    import torch

    version = str(getattr(torch, "__version__", "") or "")
    parts = version.split("+")[0].split(".")
    try:
        found = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (
            f"cannot parse torch version {version!r} to check the "
            f"lifted-LoRA mint floor (torch >= "
            f"{'.'.join(map(str, LIFTED_LORA_TORCH_FLOOR))})")
    if found < LIFTED_LORA_TORCH_FLOOR:
        floor = ".".join(map(str, LIFTED_LORA_TORCH_FLOOR))
        return (
            f"lifted-LoRA mint requires torch >= {floor}, got {version}: "
            f"torch 2.9 strict export refuses bind_views' in-trace setattr "
            f"('Mutating module attribute lora_a during export') that 2.13 "
            f"traces fine — measured on pod 8 (pgw#723 residuals), so the "
            f"2.13 prod floor is a mint PRECONDITION for this lane, not a "
            f"preference")
    return ""


def lane_admitted(spec: ExportSpec, *, allow_regressed_lanes: bool) -> str:
    """'' when this lane may be minted, else the named refusal reason."""
    base, _bucket = lane_bucket(spec.weight_lane)
    token = lane_token(base)
    if token in PARITY_LANES:
        return ""
    if allow_regressed_lanes:
        return ""
    return (
        f"lane {token or '(plain)'!r} measured 6.9-7.0% SLOWER under AOTI than "
        f"dynamo (pgw#704 Q4) and is HELD on dynamo by #730 until explained; "
        f"mint the w8a8 lane first, or pass allow_regressed_lanes to override "
        f"deliberately"
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def dynamic_shapes_spec(
    dims: Sequence[DynamicDim], input_names: Sequence[str],
) -> Dict[str, Any]:
    """``torch.export`` ``dynamic_shapes`` for the declared dims.

    Keyed by parameter NAME (the dict form), which is the only readable way to
    mirror a combined args+kwargs structure. Inputs with no declared dynamic
    axis map to ``None`` — fully static — so an input the spec forgot cannot
    accidentally acquire a symbolic dim.

    Rows carrying the same :attr:`DynamicDim.dim` share ONE torch symbol
    (pgw#812 D1). Minting a symbol per (input, axis) instead makes a declared
    multi-carrier dim into several INDEPENDENT symbols, and strict export
    refuses the declaration outright::

        Constraints violated (img_ids_1)! The values of
        img_ids_1 = L['img_ids'].size()[1] and
        hidden_states_1 = L['hidden_states'].size()[1] must always be equal.

    So the most careful declaration in the fleet — flux2's, which binds
    ``T_img`` to both carriers so the edit lane cannot silently pin
    ``img_ids`` — was the one that could not export at all. Rows with no
    declared name keep today's per-(input, axis) symbol, which is required by
    the hand-registered builder path (``aot_inputs.latent_hw_dims``: latent H
    and W are two independent axes of ONE input and must NOT share a symbol).
    """
    from torch.export import Dim

    by_input: Dict[str, Dict[int, Any]] = {}
    shared: Dict[Tuple[str, int, int, int], Any] = {}
    for d in dims:
        if d.min < 1 or d.max < d.min:
            raise MintRefused(
                f"declared dim {d.input_name}[{d.axis}] has an empty range "
                f"[{d.min}, {d.max}]")
        # Carriers only share when the declaration says they are the SAME dim
        # AND the bounds agree; a name reused at different bounds is two
        # symbols, not one silently-widened one.
        key = (d.dim, int(d.multiple_of), int(d.min), int(d.max))
        symbol = shared.get(key) if d.dim else None
        if symbol is None:
            base_name = d.dim or f"{d.input_name}_{d.axis}"
            if d.multiple_of > 1:
                if d.min % d.multiple_of or d.max % d.multiple_of:
                    raise MintRefused(
                        f"declared dim {d.input_name}[{d.axis}] bounds "
                        f"[{d.min}, {d.max}] are not multiples of "
                        f"{d.multiple_of}; export cannot express that guard")
                symbol = d.multiple_of * Dim(
                    f"{base_name}_u",
                    min=d.min // d.multiple_of, max=d.max // d.multiple_of)
            else:
                symbol = Dim(base_name, min=d.min, max=d.max)
            if d.dim:
                shared[key] = symbol
        by_input.setdefault(d.input_name, {})[d.axis] = symbol
    unknown = sorted(set(by_input) - set(input_names))
    if unknown:
        raise MintRefused(
            f"declared dynamic dims name inputs the target does not take: "
            f"{unknown!r} (target inputs: {list(input_names)!r})")
    return {name: by_input.get(name) for name in input_names}


def export_program(
    module: Any,
    example_args: Tuple[Any, ...],
    example_kwargs: Mapping[str, Any],
    *,
    dynamic_shapes: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> Any:
    """``torch.export.export`` the target, refusing by name on failure."""
    import torch

    try:
        return torch.export.export(
            module, tuple(example_args), dict(example_kwargs),
            dynamic_shapes=dict(dynamic_shapes) if dynamic_shapes else None,
            strict=strict,
        )
    except Exception as exc:
        raise MintRefused(
            f"torch.export(strict={strict}) failed for "
            f"{type(module).__name__}: {type(exc).__name__}: {exc}"
        ) from exc


def _placeholder_shapes(program: Any) -> Dict[str, Tuple[Any, ...]]:
    """``{user input name: shape tuple}`` from the exported placeholders."""
    signature = getattr(program, "graph_signature", None)
    user_inputs = [str(n) for n in getattr(signature, "user_inputs", ()) or ()]
    by_node: Dict[str, Any] = {}
    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        if getattr(node, "op", "") == "placeholder":
            by_node[str(node.name)] = node.meta.get("val")
    out: Dict[str, Tuple[Any, ...]] = {}
    for name in user_inputs:
        val = by_node.get(name)
        if val is not None:
            out[name] = tuple(getattr(val, "shape", ()) or ())
    return out


def _free_symbols(dim: Any) -> Tuple[Any, ...]:
    node = getattr(dim, "node", None)
    expr = getattr(node, "expr", None)
    return tuple(getattr(expr, "free_symbols", ()) or ()) if expr is not None else ()


def _shape_env(program: Any) -> Any:
    for shape in _placeholder_shapes(program).values():
        for dim in shape:
            env = getattr(getattr(dim, "node", None), "shape_env", None)
            if env is not None:
                return env
    return None


def declared_range_gaps(
    program: Any, dims: Sequence[DynamicDim],
) -> List[str]:
    """Named reasons the export did not honour the declared dynamic contract.

    An export that specialized — or silently PINNED — a dim we advertise as
    dynamic produces an artifact serving one shape while its metadata claims a
    range, and pgw#704 B2 measured that nothing at runtime refuses the
    difference. So this must fail the mint, not ship.

    Three checks, because presence of a range entry is NOT proof (ie#566 G3 —
    a measured FALSE PASS: wan ti2v-5b declared symbolic H/W, exported clean,
    passed a presence-only gate, and yet a static per-token input of 27,280
    pinned H*W to exactly one shape):

    1. **specialization** — the declared axis is a plain int in the exported
       placeholder, so the dim never became symbolic at all;
    2. **solved range** — the governing symbol's range in the exported program
       must COVER the declared ``[min, max]``. A collapsed (``lower == upper``)
       or narrowed range is a pin, and the artifact admits less traffic than it
       advertises;
    3. **pinning guards** — an equality guard in the shape env mentioning a
       declared symbol. A dim that is genuinely a function of the declared
       extents forces the tracer to record ``Eq(h*w, N)``; a dim that merely
       shares a factor records nothing. This is the check the presence-only gate
       lacked, and it is evidence-based rather than arithmetic — see
       :func:`_pinning_guards` for why the arithmetic version was wrong.
    """
    gaps: List[str] = []
    shapes = _placeholder_shapes(program)
    ranges = getattr(program, "range_constraints", {}) or {}
    declared_symbols: List[Any] = []
    for d in dims:
        if d.min == d.max:
            continue
        shape = shapes.get(d.input_name)
        if shape is None:
            gaps.append(
                f"declared dynamic dim names input {d.input_name!r}, which is "
                f"not a user input of the exported program "
                f"(inputs: {sorted(shapes)!r})")
            continue
        if d.axis >= len(shape):
            gaps.append(
                f"{d.input_name}[{d.axis}] is out of range for the exported "
                f"shape {tuple(str(x) for x in shape)!r}")
            continue
        dim = shape[d.axis]
        text = str(dim)
        if text.lstrip("-").isdigit():
            gaps.append(
                f"{d.input_name}[{d.axis}] exported as the STATIC value {text} "
                f"but is declared dynamic [{d.min}, {d.max}] — export "
                f"specialized a dim the declaration advertises as dynamic")
            continue
        syms = _free_symbols(dim)
        declared_symbols.extend(syms)
        covered = False
        # The SOLVED range of the axis's full expression, when the program
        # records one. This is what makes a UNIFIED relational axis (#739 /
        # ie#566 §5) gate-able: wan ti2v's per-token dim solves to
        # ``31*s25*s56`` with its own composite range entry, while the
        # per-symbol path below would compare a governing symbol's [20, 40]
        # against the declared [12400, 49600] and refuse a sound artifact.
        # Composite entries are in the axis's OWN units, so the declared
        # bounds compare directly, with no multiple-of scaling.
        expr = getattr(getattr(dim, "node", None), "expr", None)
        interval = ranges.get(expr) if expr is not None else None
        if interval is not None:
            try:
                lo, hi = int(interval.lower), int(interval.upper)
            except (TypeError, ValueError, OverflowError):
                lo = hi = -1
            if lo >= 0:
                if lo == hi:
                    gaps.append(
                        f"{d.input_name}[{d.axis}] ({expr}) solved to the "
                        f"single value {lo} — the declared range "
                        f"[{d.min}, {d.max}] is advertised but the artifact "
                        f"admits ONE shape")
                elif lo > d.min or hi < d.max:
                    gaps.append(
                        f"{d.input_name}[{d.axis}] ({expr}) solved to "
                        f"[{lo}, {hi}] which does not cover the declared "
                        f"[{d.min}, {d.max}] — the artifact admits less "
                        f"traffic than it advertises")
                continue
        for sym in syms:
            interval = ranges.get(sym)
            if interval is None:
                continue
            try:
                lo, hi = int(interval.lower), int(interval.upper)
            except (TypeError, ValueError, OverflowError):
                continue
            if lo == hi:
                gaps.append(
                    f"{d.input_name}[{d.axis}] symbol {sym} solved to the "
                    f"single value {lo} — the declared range [{d.min}, {d.max}] "
                    f"is advertised but the artifact admits ONE shape")
                covered = True
                break
            # The symbol may carry a multiple-of factor (8*s95), so compare the
            # DECLARED bounds against the symbol's own solved bounds scaled by
            # the factor the declaration states.
            factor = max(1, int(d.multiple_of or 1))
            want_lo, want_hi = d.min // factor, d.max // factor
            if lo > want_lo or hi < want_hi:
                gaps.append(
                    f"{d.input_name}[{d.axis}] symbol {sym} solved to "
                    f"[{lo * factor}, {hi * factor}] which does not cover the "
                    f"declared [{d.min}, {d.max}] — the artifact admits less "
                    f"traffic than it advertises")
            covered = True
            break
        if not covered and not syms:
            gaps.append(
                f"{d.input_name}[{d.axis}] is symbolic ({text}) but carries no "
                f"resolvable symbol; its admissible range is unprovable")

    gaps.extend(_pinning_guards(program, declared_symbols))
    return gaps


def _is_tautology(expr: Any) -> bool:
    """``True`` only when the guard is provably true for EVERY value (pgw#812 D2).

    A gate that cannot tell "pinned" from "trivially true" refuses correct
    mints, which is the expensive direction of the error. flux2 is refused
    today on::

        Eq(Mod(3072*s50 + 1572864, 48*s50 + 24576), 0)

    and ``3072*s + 1572864 == 64 * (48*s + 24576)``, so that is
    ``Mod(64*X, X) == 0`` — identically true, pinning nothing. The algebra
    comes from attention's ``unflatten``/``flatten`` over the concatenated
    image+text stream, so it appears on the whole-graph AND the block export;
    regional does not dodge it.

    Only a PROOF admits: anything sympy cannot reduce to ``true`` keeps
    refusing, so the gate still fails closed on an unrecognised guard.
    """
    try:
        import sympy

        return sympy.simplify(expr) is sympy.true
    except Exception:  # noqa: BLE001 — an unprovable guard stays refused
        logger.debug("range gate: could not simplify guard %s", expr,
                     exc_info=True)
        return False


def _pinning_guards(program: Any, declared_symbols: Sequence[Any]) -> List[str]:
    """Equality guards that pin a declared-dynamic symbol (ie#566 G3).

    A dim that is GENUINELY a function of the declared extents forces the tracer
    to record an equality guard; a dim that merely happens to share a factor
    records nothing. Measured on this toolchain::

        genuine  x.reshape(b, c, h*w) + tokens   ->  guards ['Eq(s37*s46, 384)']
        coincidence  x.flatten(2).sum(-1) + tokens.sum()  ->  guards []

    In BOTH cases ``range_constraints`` still reports the full declared range,
    which is exactly why a presence check — and even a solved-range check —
    passes the pinned artifact. The guard is the only place the truth appears.

    This replaces an earlier divisibility heuristic that asked whether a static
    dim was a multiple of the product of the declared extents. That test could
    not distinguish the two cases above, and it FALSE-POSITIVED on sdxl, whose
    cross-attention width 2048 and pooled width 1280 are multiples of the
    declared extents' product by pure architectural coincidence — refusing every
    symbolic sdxl mint, including the one reproducing pgw#704's headline. The
    lesson is recorded here because the coincidence is easy to re-introduce:
    integer arithmetic on shapes is NOT evidence of dependence; a solved
    relation is.

    A truly free symbol carries no equality guard, so any ``Eq`` mentioning a
    declared symbol means that symbol is not free.
    """
    declared = {str(sym) for sym in declared_symbols}
    if not declared:
        return []
    env = _shape_env(program)
    if env is None:
        return []
    out: List[str] = []
    seen: set = set()
    for source in ("guards", "axioms"):
        entries = getattr(env, source, None) or ()
        if isinstance(entries, dict):
            entries = list(entries)
        for entry in entries:
            expr = getattr(entry, "expr", entry)
            if expr is None:
                continue
            if getattr(expr, "func", None) is None or \
                    type(expr).__name__ not in ("Eq", "Equality"):
                continue
            names = {str(sym) for sym in
                     (getattr(expr, "free_symbols", ()) or ())}
            hit = sorted(names & declared)
            if not hit:
                continue
            # A pin has a CONSTANT on one side (``Eq(s37*s46, 384)``). An
            # equality between two symbolic sides (``Eq(s37, 8*s95)``) is the
            # DEFINITIONAL relation our own multiple-of factor introduces —
            # ``8 * Dim(...)`` — and refusing it would block every mint that
            # declares a divisibility, which is all of the image families.
            sides = list(getattr(expr, "args", ()) or ())
            if len(sides) != 2 or not any(
                not (getattr(side, "free_symbols", None) or set())
                for side in sides
            ):
                continue
            if _is_tautology(expr):
                continue
            text = str(expr)
            if text in seen:
                continue
            seen.add(text)
            out.append(
                f"the exported program carries the equality guard {text}, "
                f"which PINS the declared dynamic symbol(s) {hit!r} — some "
                f"other input's static shape is an algebraic function of the "
                f"declared extents, so the artifact serves exactly the traced "
                f"shape even though its declared range says otherwise "
                f"(ie#566 G3). Declare that input dynamic too, or make it "
                f"independent of the latent extent")
    return out


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


#: Mint-path default worker count for inductor's parallel compile (#757,
#: MEASURED): 32 -> 4 is FREE (-2% wall clock) and is the recommended
#: default for background mints on serving pods — same speed, less CPU
#: contention with live serving. NOT seal-relevant: compile_threads is
#: outside cell identity per #757's re-key pre-verification, so this
#: default (and a caller override) never re-keys a cell. A caller value
#: wins; this is a default, not a clamp.
MINT_COMPILE_THREADS = 4


def _entry_configs(inductor_configs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The per-entry inductor config: caller options + the non-negotiable
    packaging flags. ``CODE_ONLY_CONFIGS`` is applied LAST so no caller-
    supplied config can re-enable constant baking — B1 is a fleet
    correctness requirement, not a default a caller may override. One cell's
    entries ALL compile under this one dict (a per-entry config drift would
    be an identity fact nothing records), and the resolved dict is recorded
    in the mint-phase telemetry."""
    configs: Dict[str, Any] = dict(inductor_configs or {})
    configs.setdefault("compile_threads", MINT_COMPILE_THREADS)
    overridden = sorted(set(configs) & set(CODE_ONLY_CONFIGS))
    if overridden:
        logger.warning(
            "aot-mint: ignoring caller inductor config %s — code-only is B1, "
            "not a knob", overridden)
    configs.update(CODE_ONLY_CONFIGS)
    # Emit loose files for package_aoti to combine, instead of a per-entry
    # archive: the multi-graph cell is ONE .pt2 (pgw#758).
    configs["aot_inductor.package"] = True
    return configs


def compile_entry_files(
    program: Any,
    entry: str,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    """AOTI-compile one exported program into CODE-ONLY loose files.

    This is ``aoti_compile_and_package``'s own internal compile step
    (verified on the pin: ``aot_compile(ep.module(check_guards=False),
    *ep.example_inputs, options)``), deferred before packaging so N entries
    combine into one archive. Compilation is byte-identical to the
    single-model mint; only the packaging changes.

    pgw#793: the host C++ compile is 46% of an AOTI compile and is ONE g++
    invocation on ONE translation unit whose largest function is inductor's
    constants_info_ table spelled as 26k straight-line statements.
    :func:`aot_wrapper_split.install` regroups exactly that run of
    statements before g++ sees it — same statements, same order, verified by
    reconstruction, declining unmodified on any wrapper shape it does not
    recognise. It changes no compiler, flag, inductor config or library, so
    no cell is re-keyed.
    """
    from torch._inductor import aot_compile

    aot_wrapper_split.install()
    gm = program.module(check_guards=False)
    args, kwargs = program.example_inputs
    try:
        files = aot_compile(
            gm, tuple(args), dict(kwargs or {}),
            options=_entry_configs(inductor_configs))
    except Exception as exc:
        raise MintRefused(
            f"entry {entry!r}: aot_compile failed: "
            f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(files, list):
        raise MintRefused(
            f"entry {entry!r}: aot_compile returned {type(files).__name__}, "
            f"not the loose-file list packaging needs "
            f"(aot_inductor.package was forced True)")
    return files


def package_cell(
    files_by_entry: Mapping[str, Sequence[Any]], package_path: Path,
) -> Path:
    """Combine every entry's compiled files into ONE ``.pt2`` of named
    models (``data/aotinductor/<entry>/`` each) — the pgw#758 cell."""
    from torch._inductor.package import package_aoti

    package_path = Path(package_path)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    if not files_by_entry:
        raise MintRefused("cannot package a cell with no entries")
    try:
        out = package_aoti(
            str(package_path),
            {str(name): list(files) for name, files in files_by_entry.items()})
    except Exception as exc:
        raise MintRefused(
            f"package_aoti failed: {type(exc).__name__}: {exc}") from exc
    return Path(str(out))


# ---------------------------------------------------------------------------
# Mint-phase telemetry (#757's instrument-first doctrine; recorded per cell)
# ---------------------------------------------------------------------------

#: ``compilation_time_metrics`` keys summarized into named phases. Host C++
#: compile+link (``AotCodeCompiler.compile``) is the stage the JIT path
#: skips entirely — its wrapper is Python — and the primary 3.9x suspect.
_PHASE_KEYS: Dict[str, Tuple[str, ...]] = {
    "lowering_s": ("GraphLowering.run",),
    "codegen_s": ("GraphLowering.codegen",),
    "host_compile_s": ("AotCodeCompiler.compile",),
    "graph_passes_s": (
        "_recursive_pre_grad_passes",
        "_recursive_joint_graph_passes",
        "_recursive_post_grad_passes",
    ),
}


def _phase_snapshot() -> Dict[str, float]:
    try:
        import torch._dynamo.utils as du

        return {
            str(k): float(sum(v))
            for k, v in du.compilation_time_metrics.items()}
    except Exception:
        return {}


def _phase_delta(
    before: Mapping[str, float], after: Mapping[str, float],
) -> Dict[str, float]:
    """Named phase seconds spent between two snapshots. ``triton_s`` sums
    every async-compile/triton key so GPU kernel compilation is one
    labeled number; the remainder of inductor time is NOT invented — the
    coarse wall clocks around export/compile hold the totals."""
    raw = {
        k: round(float(after.get(k, 0.0)) - float(before.get(k, 0.0)), 3)
        for k in set(after) | set(before)
    }
    out: Dict[str, float] = {}
    for label, keys in _PHASE_KEYS.items():
        value = round(sum(raw.get(k, 0.0) for k in keys), 3)
        if value:
            out[label] = value
    triton = round(sum(
        v for k, v in raw.items()
        if ("async_compile" in k or "triton" in k.lower()) and v > 0), 3)
    if triton:
        out["triton_s"] = triton
    return out


def autotune_posture(
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The benchmark-driven kernel-selection posture of THIS mint —
    recorded so an AOT-vs-JIT cost comparison can rule the asymmetry in or
    out (#757: it would dominate everything else)."""
    posture: Dict[str, Any] = {}
    try:
        import torch._inductor.config as inductor_config

        for name in ("max_autotune", "max_autotune_pointwise",
                     "max_autotune_gemm", "search_autotune_cache"):
            if hasattr(inductor_config, name):
                posture[name] = bool(getattr(inductor_config, name))
    except Exception:
        pass
    for key, value in (inductor_configs or {}).items():
        if "autotune" in str(key):
            posture[f"override.{key}"] = value
    return posture


# ---------------------------------------------------------------------------
# The mint
# ---------------------------------------------------------------------------


@dataclass
class MintResult:
    """A packed, gated, publishable artifact plus its mint telemetry."""

    artifact: Path
    metadata: Dict[str, Any]
    timings: Dict[str, float]

    @property
    def cell_key(self) -> str:
        return str(self.metadata.get("cell_key") or "")


@dataclass
class _MintedEntry:
    """One exported+compiled graph class, pre-packaging."""

    name: str
    spec: ExportSpec
    module: Any
    owner: Any
    program: Any
    input_names: Tuple[str, ...]
    flat_names: Tuple[str, ...]
    files: List[Any]
    timings: Dict[str, Any]


def _regional_dynamic(
    declared: Sequence[DynamicDim], block_inputs: Sequence[str],
) -> Tuple[DynamicDim, ...]:
    """The declared dynamic rows that SURVIVE into a block's own inputs.

    A block's inputs are internal and captured, never declared (pgw#812 S5) —
    but a declared carrier frequently reaches the block under its own name
    (flux2's ``hidden_states`` is the block's first argument). Those rows ride;
    a carrier the block never sees is dropped, because marking an axis on an
    input that does not exist is a mint error and inventing a range for an
    input the declaration never described would be worse.

    The result is that ``regional + dynamic`` is expressible without a second,
    deeper contract for the endpoint to maintain — the inner gate is derived
    from the artifact's OWN recorded ranges (S5), not from a new declaration.
    """
    names = set(str(n) for n in block_inputs)
    return tuple(d for d in declared if str(d.input_name) in names)


def adapter_arm(fork: Sequence[Tuple[str, Any]]) -> Optional[bool]:
    """The pgw#790 adapter arm a fork coordinate states, or ``None`` when the
    coordinate does not carry one (a bucket-0 family, or a target that carries
    no branch-capable module — wan's ``vae.decode`` in a bucket-128 cell)."""
    value = dict(fork).get(ADAPTER_FORK, None)
    return None if value is None else bool(value)


def declared_fork(fork: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """The coordinate MINUS the SDK-synthesized adapter arm — what the
    ENDPOINT declared, and therefore the only part its declaration can be
    asserted against."""
    return {str(n): v for n, v in fork if str(n) != ADAPTER_FORK}


def with_adapter_arm(plan: Any, arm: bool) -> Any:
    """One mint plan pinned to an adapter arm (a fork coordinate, so it names
    the entry and lands in the class hash like every other fork)."""
    from dataclasses import replace

    return replace(
        plan,
        fork=tuple(sorted(
            tuple(plan.fork) + ((ADAPTER_FORK, bool(arm)),),
            key=lambda pair: str(pair[0]))))


def _entry_spec(spec: ExportSpec, plan: Any, decl: Any) -> ExportSpec:
    """The per-entry :class:`ExportSpec` one mint plan derives from the
    cell-level request."""
    from dataclasses import replace

    from . import aot_declaration as _decl  # deferred: aot_declaration imports us

    specialization = dict(spec.specialization)
    # pgw#829: the strategy that actually governed THIS entry's population.
    # A regional cell's entries are blocks and may collapse where the
    # conv-bearing whole-graph route does not, so recording the family's
    # `shape_strategy` here would put a fact in the key that did not decide
    # anything about the artifact. Identity-inert for every existing family:
    # without a declared regional override the effective value IS
    # `shape_strategy`.
    specialization.setdefault(
        "shape_strategy",
        _decl.effective_shape_strategy(
            decl, regional=bool(getattr(spec, "regional", False))))
    specialization.setdefault("warm_changes_key", bool(decl.warm_changes_key))
    for name, value in plan.fork:
        specialization.setdefault(f"fork.{name}", value)
    fork, dims = _decl.entry_coordinates(plan)
    espec = replace(
        spec,
        target=str(plan.target),
        fork=fork,
        class_dims=dims,
        dynamic=tuple(plan.dynamic),
        specialization=specialization,
    )
    if adapter_arm(plan.fork) is False:
        # pgw#790's branchless class: no bucket, no lifted pair, nothing for
        # the adapter gates to assert. The CELL still declares its bucket —
        # the fork is what says this graph has no branch, and the specialization
        # block records both, so a reader can see which arm they are holding.
        espec = replace(espec, lora_bucket=0, lifted_inputs=(), lora_fqns=())
    return espec


def _run_declared_warm(module: Any, args: Tuple[Any, ...], entry: str) -> float:
    """Execute the declared mint-warm canon: one forward with the entry's
    own seed inputs BEFORE export (the warm-canon obligation — z-image's
    rope pre-warm measurably changes the graph, 4327 cold vs 4285 warmed
    nodes; a family declaring ``warm_changes_key=True`` that skips this
    mints the graph the fleet never serves, and the #699 double-mint
    byte-compare flaps on warm order). Returns the warm seconds."""
    import torch

    t0 = time.monotonic()
    try:
        with torch.no_grad():
            module(*args)
    except Exception as exc:
        raise MintRefused(
            f"entry {entry!r}: declared mint-warm forward failed "
            f"({type(exc).__name__}: {exc}) — warm_changes_key=True makes "
            f"the pre-warm a mint obligation, not a best effort") from exc
    return round(time.monotonic() - t0, 2)


def _export_entry(
    pipeline: Any,
    spec: ExportSpec,
    plan: Any,
    decl: Any,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    compile_now: bool = True,
) -> _MintedEntry:
    """Resolve, feed, (warm,) export, gate, and compile ONE declared graph
    class. Every refusal is prefixed with the entry name — a multi-graph
    mint that cannot say WHICH class failed is the silent-failure path in
    a new hat (pgw#758).

    ``compile_now=False`` stops after the export-side gates and returns the
    entry with no files: pgw#809's pool then compiles every entry K-wide out
    of process. Export must stay here and stay SERIAL — it runs against the
    one live pipeline, on the one card, inside the one branch-arm toggle."""
    from . import aot_declaration as _decl  # deferred: aot_declaration imports us
    from . import aot_inputs

    espec = _entry_spec(spec, plan, decl)
    entry = _decl.plan_entry_name(plan)
    timings: Dict[str, Any] = {}

    resolved = _resolve_target(pipeline, espec.target)
    if resolved is None:
        raise MintRefused(
            f"entry {entry!r}: pipeline {type(pipeline).__name__} has no "
            f"compile target {espec.target!r}")
    owner, attr, _fn = resolved
    if decl.forks:
        gaps = _decl.fork_gaps(
            decl, declared_fork(espec.fork), target=espec.target,
            pipeline=pipeline, module=owner)
        if gaps:
            raise MintRefused(
                f"entry {entry!r}: fork gate (pgw#739): " + "; ".join(gaps))
    module = owner if attr == "forward" else _CallableTarget(owner, attr)

    # The LoRA bucket is a CELL-level request but a PER-TARGET fact: adapters
    # ride the branch-capable denoisers, never the VAE (wan's vae.decode
    # entry is bucket-0 in the same cell as its bucket-128 transformer).
    # Scoped by COMPOSED truth (lora_lifted.branch_targets), not vocabulary —
    # and a branch-capable target whose lifting was not installed still fails
    # the lifted-input gate by name, never silently mints bucket-0.
    if espec.lora_bucket or espec.lifted_inputs or espec.lora_fqns:
        from dataclasses import replace

        from .models import lora_lifted

        branch_owners = {
            id(m) for m in lora_lifted.branch_targets(pipeline).values()}
        if id(owner) not in branch_owners:
            espec = replace(
                espec, lora_bucket=0, lifted_inputs=(), lora_fqns=())
        elif espec.lifted_inputs \
                and lora_lifted.lifted_binding(owner) is None:
            # pgw#822: the ARM, asserted where it is needed rather than
            # assumed. Without this the miss surfaces as `_positionalize`
            # refusing the DECLARATION ("lora_a/lora_b are not parameters of
            # forward") — a true sentence about the wrong thing, which is
            # what sent the first real mint attempt looking at the endpoint's
            # contract instead of at the pipeline's preparation.
            raise MintRefused(
                f"entry {entry!r}: this class declares lifted adapter "
                f"input(s) {list(espec.lifted_inputs)!r} but "
                f"{type(owner).__name__} carries no lifted forward — the "
                f"branch containers are armed and the lifted signature is "
                f"not, so the module about to be exported cannot take the "
                f"adapter (lora_lifted.arm_lifted_lora_lanes installs both)")

    builder = aot_inputs.builder_for(espec.family, espec.target)
    args, kwargs = builder(owner, espec)
    if kwargs:
        # All-positional example feeds are a MINT OBLIGATION (pgw#723
        # residuals, pod 9): the AOTI package's call convention mirrors the
        # traced args/kwargs split, and the serve marshal is POSITIONAL
        # (aot_serve.marshal_positional) — a kwarg-traced package arms, then
        # silently revokes to eager on its first call ("Ran into a kwarg
        # keyword mismatch: Got [] but expected ['lora_a','lora_b']",
        # measured). Refused HERE so the failure is a named mint refusal
        # instead of a vacuous eager-serving artifact.
        raise MintRefused(
            f"entry {entry!r}: example feed carries keyword argument(s) "
            f"{sorted(kwargs)!r} — all-positional feeds are a mint "
            f"obligation (pod 9, pgw#723 residuals): a kwarg-traced package "
            f"is uncallable by the positional serve marshal and fails only "
            f"at first serve, silently revoking to eager. Feed every input "
            f"positionally (signature defaults fill the gaps)")

    # The WARM CANON, EXECUTED (declared per family; previously keyed but
    # never acted on): sdxl declared False and skips; z-image's True runs.
    if bool(decl.warm_changes_key):
        timings["warm_s"] = _run_declared_warm(module, args, entry)

    input_names = _input_names(module, args, kwargs)
    flat_names = flat_input_names(module, args, kwargs)
    dynamic = dynamic_shapes_spec(espec.dynamic, input_names) \
        if espec.dynamic else None

    t0 = time.monotonic()
    try:
        program = export_program(
            module, args, kwargs, dynamic_shapes=dynamic, strict=espec.strict)
    except MintRefused as exc:
        raise MintRefused(f"entry {entry!r}: {exc}") from exc
    timings["export_s"] = round(time.monotonic() - t0, 2)

    gaps = declared_range_gaps(program, espec.dynamic)
    if gaps:
        raise MintRefused(
            f"entry {entry!r}: declared-range gate: " + "; ".join(gaps))
    lifted_gaps = lifted_input_gaps(program, espec)
    if lifted_gaps:
        raise MintRefused(
            f"entry {entry!r}: lifted-input gate: " + "; ".join(lifted_gaps))

    # pgw#725 G3, on the EXPORTEDPROGRAM and before any packing: the adapter
    # must be absent from the constant table AND present among the user inputs.
    # A missing pair is the same defect as a baked one (the branch was traced
    # away, so every request silently gets the base model), and packing renames
    # a plain-__dict__ adapter to _tensor_constant0 — which makes the
    # package-side scan a false PASS. Free here, unsound there.
    if espec.lora_bucket or espec.lifted_inputs or espec.lora_fqns:
        from .api.errors import ValidationError
        from .models import lora_lifted

        try:
            lora_lifted.assert_no_baked_adapter(
                program, label=f"{espec.family}/{espec.target}")
        except ValidationError as exc:
            raise MintRefused(
                f"entry {entry!r}: no-baked-adapter gate (#725 G3): "
                f"{exc}") from exc

    # pgw#825: BEFORE the compile, not after it. The packed bindability gate
    # runs on the artifact and is the proof of what shipped; this one asks the
    # same question of the exported program, where it costs milliseconds
    # instead of an entry's whole compile.
    unbindable = aot_package.unbindable_program_constants(
        program, _state_dict_keys(owner))
    if unbindable:
        raise MintRefused(
            f"entry {entry!r}: pre-compile bindability gate: "
            + "; ".join(unbindable))

    files: List[Any] = []
    if compile_now:
        before = _phase_snapshot()
        t0 = time.monotonic()
        files = compile_entry_files(
            program, entry, inductor_configs=inductor_configs)
        timings["compile_s"] = round(time.monotonic() - t0, 2)
        phases = _phase_delta(before, _phase_snapshot())
        if phases:
            timings["phases"] = phases

    return _MintedEntry(
        name=entry, spec=espec, module=module, owner=owner, program=program,
        input_names=input_names, flat_names=flat_names, files=files,
        timings=timings)


def _contiguous_feed(value: Any) -> Any:
    """One example-input tensor, made contiguous before it is TRACED.

    A silent wrong-answer defect, measured off-pod ($0, CPU, torch
    2.13.0+cu130, a 3-block toy whose shell does diffusers'
    ``permute(0,2,3,1).reshape(b, h*w, c)``, which yields a NON-contiguous
    view):

        traced non-contiguous, served with the pgw#791 realign   max|d| 0.1645
        traced non-contiguous, served with the realign DISABLED  max|d| 0.1690
        traced CONTIGUOUS,     served with the realign           max|d| 1.5e-08

    So this is not the realign — it is the ARTIFACT. A block's feed is
    CAPTURED from a live forward (pgw#812 S5), never constructed, so whatever
    layout the shell happens to hand it is what gets traced; AOTInductor
    generates against that layout, and the value computed for any other one
    is wrong. Nothing refuses it: the ingress contract records shapes and
    dtypes, not strides, so the call is admitted and the answer is quietly
    off by 16 %.

    The fix belongs at the MINT because that is the side with a choice.
    ``aligned_feeds`` already stages every out-of-contract input into an
    owned CONTIGUOUS buffer at serve, so tracing contiguous makes the two
    sides agree by construction — for a feed that arrives contiguous
    (diffusers' sdxl passes the block through ``proj_in``, a Linear, so it
    does) this is a no-op and the artifact is byte-identical.

    Eager is unaffected: the block still receives the shell's own tensor,
    and a matmul on a non-contiguous input computes the same values.
    """
    contiguous = getattr(value, "contiguous", None)
    return value if not callable(contiguous) else contiguous()


def _regional_derived_dynamic(
    entry: str,
    names: Sequence[str],
    shapes_by_row: Sequence[Sequence[Optional[Tuple[int, ...]]]],
) -> Tuple[DynamicDim, ...]:
    """The block's OWN varying axes, derived from what the shell fed it
    across every class row the entry serves (pgw#829).

    This is the derivation that makes 72 entries into 8. A block's inputs are
    internal and never declared (pgw#812 S5), so the declared ``Dim``
    carriers cannot describe them: sdxl declares ``H_lat``/``W_lat`` on
    ``sample``, and the block is handed a flat ``(B, H*W/f**2, C)`` hidden
    state that carries neither name. The axis that actually varies is
    therefore only observable — one eager shell forward per declared class
    row, block input shapes recorded, axes that MOVE marked dynamic over
    their observed hull.

    Two rules, both refusals rather than guesses:

    * an axis whose hull reaches 1 must FORK, never collapse — torch's 0/1
      specialization is not overridable (ie#543), so a symbol that could be 1
      would be silently specialized and the artifact would serve one shape
      while its contract advertised a range;
    * slots that vary IN LOCKSTEP across the rows share one symbol. Two
      carriers of one logical axis exported as independent symbols is exactly
      what pgw#812 D1 measured strict export refusing outright, and here the
      carriers are unnamed, so lockstep over the observed rows is the only
      evidence there is.

    ``multiple_of`` is deliberately 1: the divisibility a declared ``Dim``
    carries is a fact about a DECLARED axis, and inferring one from a
    handful of observed values would put a guard in the artifact that
    :func:`aot_serve.assert_ingress` — which checks ranges, not multiples —
    would never enforce at serve time.
    """
    rows = [tuple(row) for row in shapes_by_row]
    if not rows:
        return ()
    width = len(names)
    for row in rows:
        if len(row) != width:
            raise MintRefused(
                f"entry {entry!r}: the shell fed this block {len(row)} "
                f"argument(s) on one class row and {width} on another — the "
                f"rows are different graph classes and cannot share one "
                f"artifact")
    out: List[DynamicDim] = []
    symbol_for: Dict[Tuple[int, ...], str] = {}
    for slot in range(width):
        column = [row[slot] for row in rows]
        present = [c is not None for c in column]
        if not any(present):
            continue
        if not all(present):
            raise MintRefused(
                f"entry {entry!r}: argument {names[slot]!r} is a tensor on "
                f"some declared class rows and absent on others — that is a "
                f"fork, not a collapsible dim")
        ranks = {len(c) for c in column if c is not None}
        if len(ranks) != 1:
            raise MintRefused(
                f"entry {entry!r}: argument {names[slot]!r} changes RANK "
                f"across the declared class rows ({sorted(ranks)!r})")
        for axis in range(ranks.pop()):
            values = tuple(int(c[axis]) for c in column if c is not None)
            if len(set(values)) <= 1:
                continue
            lo, hi = min(values), max(values)
            if lo < 2:
                raise MintRefused(
                    f"entry {entry!r}: {names[slot]}[{axis}] spans "
                    f"[{lo}, {hi}] across the collapsed class rows; torch's "
                    f"0/1 specialization is not overridable (ie#543), so an "
                    f"axis that reaches 1 must FORK the class instead of "
                    f"collapsing")
            symbol = symbol_for.setdefault(values, f"{names[slot]}_{axis}")
            out.append(DynamicDim(
                input_name=str(names[slot]), axis=int(axis),
                min=int(lo), max=int(hi), multiple_of=1, dim=symbol))
    return tuple(out)


def _merge_dynamic(
    entry: str, derived: Sequence[DynamicDim], declared: Sequence[DynamicDim],
) -> Tuple[DynamicDim, ...]:
    """Union the observed rows with the declared ones that reach the block.

    A declared carrier that arrives under its own name (flux2's
    ``hidden_states``) already had a derived range from the class rows, so
    the two can only agree or the DECLARED one is wider on purpose. Widening
    to the hull is the safe direction — a contract that admits more than the
    artifact was traced on is caught by :func:`declared_range_gaps`, while
    one that admits less silently refuses live traffic to eager.

    One case is refused instead of merged: a declared row that would RENAME a
    derived symbol the observation put in a lockstep GROUP. Symbol identity
    is what :func:`dynamic_shapes_spec` shares carriers by, so renaming one
    member and not the others silently discards the evidence that they move
    together, and strict export then refuses the whole declaration with a
    ``Constraints violated`` on a symbol nobody declared (pgw#812 D1). No
    family reaches this today — sdxl's declared carriers are all on
    ``sample``, which no block sees — and inventing a merge for the first one
    that does, without its rows in hand, is the guess this module refuses to
    make anywhere else.
    """
    grouped = [str(d.dim) for d in derived]
    shared = {sym for sym in grouped if grouped.count(sym) > 1}
    by_key: Dict[Tuple[str, int], DynamicDim] = {
        (d.input_name, int(d.axis)): d for d in derived}
    for d in declared:
        key = (d.input_name, int(d.axis))
        prior = by_key.get(key)
        if prior is None:
            by_key[key] = d
            continue
        if str(prior.dim) in shared and str(d.dim) != str(prior.dim):
            peers = sorted(
                f"{p.input_name}[{p.axis}]" for p in derived
                if str(p.dim) == str(prior.dim))
            raise MintRefused(
                f"entry {entry!r}: declared dim {d.dim!r} renames "
                f"{d.input_name}[{d.axis}], which the class rows observed "
                f"moving in LOCKSTEP with {peers!r} under one derived symbol "
                f"{prior.dim!r}. Renaming one carrier of a lockstep group and "
                f"not the others is the multi-carrier split strict export "
                f"refuses (pgw#812 D1) — declare the whole group under one "
                f"Dim, or fork the class")
        multiple = max(1, int(d.multiple_of))
        lo = (min(prior.min, d.min) // multiple) * multiple
        hi = -(-max(prior.max, d.max) // multiple) * multiple
        by_key[key] = DynamicDim(
            input_name=d.input_name, axis=int(d.axis),
            min=max(lo, multiple), max=hi, multiple_of=multiple, dim=d.dim)
    return tuple(by_key[k] for k in sorted(by_key))


def _export_regional_entries(
    pipeline: Any,
    spec: ExportSpec,
    plan: Any,
    decl: Any,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    compile_now: bool = True,
) -> List[_MintedEntry]:
    """Export one mint plan's target as BLOCK entries instead of one graph.

    pgw#817, implementing pgw#812 S1: the cell is still one ``.pt2`` and the
    entry grammar is unchanged — ``<target>/block=<class>#<n>/<dims>`` — but
    the entries enumerate BLOCK CLASSES of the target instead of shape
    coordinates of its whole forward. sdxl's 70 UNet blocks collapse to 2
    entries; flux2's 25+25 to 2.

    The shell stays EAGER (S2), which is the honest bound on the win: the
    compiled fraction of the model equals the repeated-block fraction. The
    alternative — exporting the shell with the blocks elided — is not
    expressible in ``torch.export`` today, because blocks are inlined at
    trace time.

    Everything before and after this function is unchanged: the same
    per-entry gates, the same class hashes, the same packaging, the same
    pool. Only the module being exported and the feed it is exported on
    change.
    """
    from . import aot_declaration as _decl  # deferred: aot_declaration imports us
    from . import aot_inputs

    espec = _entry_spec(spec, plan, decl)
    label = _decl.plan_entry_name(plan)

    resolved = _resolve_target(pipeline, espec.target)
    if resolved is None:
        raise MintRefused(
            f"entry {label!r}: pipeline {type(pipeline).__name__} has no "
            f"compile target {espec.target!r}")
    owner, attr, _fn = resolved
    if attr != "forward":
        raise MintRefused(
            f"entry {label!r}: regional export needs a MODULE with declared "
            f"repeated blocks; target {espec.target!r} resolves to a callable "
            f"attribute ({attr!r}), which has no block structure to compile "
            f"per class")
    groups = aot_regional.repeated_block_groups(owner)
    if not groups:
        raise MintRefused(
            f"entry {label!r}: {type(owner).__name__} declares no "
            f"`_repeated_blocks`, so there is no repeated structure to "
            f"compile once and reuse — mint this target whole-graph "
            f"(regional=False) or fix the model class")

    module = owner
    builder = aot_inputs.builder_for(espec.family, espec.target)

    import torch

    def _feed_for(row: Any) -> Tuple[Any, ...]:
        """The shell's example feed for ONE declared class row."""
        row_spec = replace_spec(espec, class_dims=tuple(row.dims))
        args, kwargs = builder(owner, row_spec)
        if kwargs:
            raise MintRefused(
                f"entry {label!r}: example feed carries keyword argument(s) "
                f"{sorted(kwargs)!r} — all-positional feeds are a mint "
                f"obligation")
        return tuple(args)

    def _runner(args: Tuple[Any, ...]) -> Any:
        def _run() -> None:
            with torch.no_grad():
                module(*args)
        return _run

    # pgw#829: which block classes may COLLAPSE their class rows into one
    # dynamic entry, decided per class off the live module. #730's static-rows
    # verdict is about convs; a conv-bearing block class keeps one static
    # entry per row exactly as before.
    rows = tuple(getattr(plan, "rows", ()) or (plan.seed,))
    collapsing = len(rows) > 1
    collapse: Dict[str, bool] = {
        group.key: collapsing and not aot_regional.block_has_conv(
            group.prototype)
        for group in groups}
    conv_bearing = sorted(k for k, v in collapse.items() if collapsing and not v)
    if conv_bearing:
        logger.info(
            "aot-mint: regional block class(es) %s carry a convolution — "
            "they keep one STATIC entry per class row (pgw#829: dynamic dims "
            "cost a conv graph the channels-last layout opt)", conv_bearing)

    seed_row = plan.seed
    # Tensors are only retained for the rows an entry is actually EXPORTED
    # from: the seed for every collapsing class, and every row for a
    # conv-bearing one. The rest of the sweep costs one eager forward and a
    # tuple of shapes (pgw#829) — the activations die with the forward.
    tensor_rows = {tuple(seed_row.dims)}
    if conv_bearing:
        tensor_rows |= {tuple(row.dims) for row in rows}

    capture_s = 0.0
    feeds_by_row: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    shapes_by_row: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.dims)
        if key in shapes_by_row:
            continue  # two declared rows at one coordinate probe once
        t0 = time.monotonic()
        run = _runner(_feed_for(row))
        if key in tensor_rows:
            captured = aot_regional.capture_block_feeds(groups, run)
            feeds_by_row[key] = captured
            probed: Dict[str, Any] = {}
            for group in groups:
                values, names = aot_regional.positional_feed(
                    group.prototype, *captured[group.key])
                probed[group.key] = (tuple(names), tuple(
                    None if getattr(v, "shape", None) is None
                    else tuple(int(d) for d in v.shape) for v in values))
            shapes_by_row[key] = probed
        else:
            shapes_by_row[key] = aot_regional.capture_block_feed_shapes(
                groups, run)
        capture_s += time.monotonic() - t0
    capture_s = round(capture_s, 2)

    out: List[_MintedEntry] = []
    for group in groups:
        for row in (rows if not collapse[group.key] else (seed_row,)):
            out.append(_regional_entry(
                espec, group, row, rows,
                feeds=feeds_by_row[tuple(row.dims)][group.key],
                shapes_by_row=shapes_by_row,
                collapsed=collapse[group.key],
                capture_s=capture_s,
                inductor_configs=inductor_configs, compile_now=compile_now))
    return out


def _regional_entry(
    espec: ExportSpec,
    group: Any,
    row: Any,
    rows: Sequence[Any],
    *,
    feeds: Tuple[Tuple[Any, ...], Mapping[str, Any]],
    shapes_by_row: Mapping[Tuple[Any, ...], Mapping[str, Any]],
    collapsed: bool,
    capture_s: float,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    compile_now: bool = True,
) -> "_MintedEntry":
    """One block entry: export, gate, compile.

    ``collapsed`` decides the entry's SHAPE axis and nothing else. A
    collapsed entry is named by its fork alone and admits the hull of every
    class row's observed block shapes (pgw#829); a static one is named by
    its row's dims and admits that row, which is exactly what every regional
    entry did before.
    """
    from . import aot_declaration as _decl  # deferred: aot_declaration imports us

    block_args, block_kwargs = feeds
    block = group.prototype
    pos, names = aot_regional.positional_feed(
        block, block_args, block_kwargs)
    pos = [_contiguous_feed(v) for v in pos]
    if collapsed:
        probe_name = _decl.entry_name(espec.target, tuple(sorted(
            tuple(espec.fork) + aot_regional.block_entry_fork(group.key),
            key=lambda kv: str(kv[0]))))
        derived = _regional_derived_dynamic(
            probe_name, names,
            [shapes_by_row[tuple(r.dims)][group.key][1] for r in rows])
        block_dynamic = _merge_dynamic(
            probe_name, derived, _regional_dynamic(espec.dynamic, names))
    else:
        derived = ()
        block_dynamic = _regional_dynamic(espec.dynamic, names)
    bspec = replace_spec(
        espec,
        fork=tuple(sorted(
            tuple(espec.fork) + aot_regional.block_entry_fork(group.key),
            key=lambda kv: str(kv[0]))),
        class_dims=() if collapsed else tuple(row.dims),
        dynamic=block_dynamic,
        regional=True,
        # pgw#825: a BLOCK never carries the lifted signature. Input lifting
        # (pgw#725) is a whole-graph-cell mechanism — it wraps the DENOISER's
        # forward so the flat pair arrives in the call — and the regional
        # entry is exported one block deep, from the block's own signature.
        # The block's branch pair stays module-resident and is bound per
        # instance BY REFERENCE (`user_managed=True`), which gives regional
        # the same property lifting gives the family graph: an adapter swap
        # is an in-place buffer write, never a recompile and never a rebind.
        # Inheriting the family's `lifted_inputs` here would record a
        # contract this entry's program does not have.
        lifted_inputs=(),
        lora_fqns=(),
    )
    entry = _decl.entry_name(espec.target, bspec.fork, bspec.class_dims)
    timings: Dict[str, Any] = {
        "capture_s": capture_s, "instances": group.count,
        "collapsed_rows": len(rows) if collapsed else 1}
    if derived:
        timings["derived_dynamic"] = [d.as_row() for d in derived]
    t0 = time.monotonic()
    dynamic = dynamic_shapes_spec(bspec.dynamic, names) \
        if bspec.dynamic else None
    try:
        program = export_program(
            block, tuple(pos), {}, dynamic_shapes=dynamic,
            strict=bspec.strict)
    except MintRefused as exc:
        raise MintRefused(f"entry {entry!r}: {exc}") from exc
    timings["export_s"] = round(time.monotonic() - t0, 2)

    gaps = declared_range_gaps(program, bspec.dynamic)
    if gaps:
        raise MintRefused(
            f"entry {entry!r}: declared-range gate: " + "; ".join(gaps))

    # pgw#825: the block's own bind table, asked BEFORE its compile. A
    # regional mint pays 4-6 minutes per entry, and this mismatch is a
    # property of the exported program — the refusal must cost seconds.
    unbindable = aot_package.unbindable_program_constants(
        program, _state_dict_keys(block))
    if unbindable:
        raise MintRefused(
            f"entry {entry!r}: pre-compile bindability gate: "
            + "; ".join(unbindable))
    baked = _regional_branch_gaps(block, program, int(bspec.lora_bucket or 0))
    if baked:
        raise MintRefused(
            f"entry {entry!r}: regional no-baked-adapter gate (#725 G3): "
            + "; ".join(baked))

    files: List[Any] = []
    if compile_now:
        before = _phase_snapshot()
        t0 = time.monotonic()
        files = compile_entry_files(
            program, entry, inductor_configs=inductor_configs)
        timings["compile_s"] = round(time.monotonic() - t0, 2)
        phases = _phase_delta(before, _phase_snapshot())
        if phases:
            timings["phases"] = phases

    # `block` is the state_dict TEMPLATE the bindability gate compares the
    # artifact's constant table against — and pgw#812 S4 is explicit that
    # for a block entry that template is the BLOCK, not the target: the
    # entry's FQNs are block-relative (`attn.to_q.weight`), and at serve
    # time the values come from `transformer_blocks[i].state_dict()`, once
    # per instance. Handing the gate the whole target here would compare
    # `lin.weight` against `blocks.0.lin.weight` and refuse every correct
    # regional mint by name.
    return _MintedEntry(
        name=entry, spec=bspec, module=block, owner=block,
        program=program, input_names=tuple(names), flat_names=tuple(names),
        files=files, timings=timings)


def _regional_branch_gaps(
    block: Any, program: Any, bucket: int,
) -> List[str]:
    """pgw#725 G3, in the shape a REGIONAL entry can be asked it.

    The family gate asserts the adapter is a graph INPUT. A block cannot take
    that form — the lift wraps the denoiser, and a block entry is exported
    from the block's own signature — so regional's equivalent invariant is
    that every branch pair is a NAMED, bindable buffer of the block. Two lanes
    make that a real question rather than a formality:

    * the w8a8 lane registers the pair as non-persistent buffers, which export
      lifts under their true FQN — bindable, and the whole point of pgw#825;
    * the cast-hook/plain-Linear lanes keep it in the module ``__dict__``,
      where export can lift it as an ANONYMOUS tensor constant whose bytes
      then SHIP in the literal payload. That is a baked, permanently zeroed
      adapter: the cell arms, serves, and silently returns the base model for
      every attach — pgw#704 S9 with no error anywhere.

    Only the adapter-bearing fork class is asked (``bucket`` is zero on the
    branchless one, whose pair is deliberately absent).
    """
    if not bucket:
        return []
    from .models import w8a8_lora

    lifted = set(aot_package.program_state_dict_fqns(program))
    missing: List[str] = []
    for path, mod in w8a8_lora.branch_modules(block).items():
        for slot in ("lora_a", "lora_b"):
            if getattr(mod, slot, None) is None:
                continue
            fqn = f"{path}.{slot}" if path else slot
            if fqn not in lifted:
                missing.append(fqn)
    if not missing:
        return []
    return [
        f"{len(missing)} branch tensor(s) of this bucket-{bucket} block did "
        f"not survive export as bindable buffers, e.g. {sorted(missing)[:6]!r}"
        f" — a branch that is not a lifted buffer was either baked as a "
        f"literal (a permanently ZEROED adapter that serves the base model "
        f"silently) or traced away entirely"
    ]


def replace_spec(spec: ExportSpec, **changes: Any) -> ExportSpec:
    """``dataclasses.replace`` on an :class:`ExportSpec`, named so the
    deferred-import dance does not have to repeat at every call site."""
    from dataclasses import replace

    return replace(spec, **changes)


def adapter_arm_plans(
    plans: Sequence[Any], pipeline: Any, spec: ExportSpec,
) -> List[Tuple[Any, Optional[bool]]]:
    """``[(plan, arm)]`` — every branch-capable target's plan forked into an
    adapter-bearing and a branchless graph class (pgw#790).

    Scoped by COMPOSED truth, not vocabulary: a cell's non-branch targets
    (wan's ``vae.decode``) fork into nothing, because there is no adapter for
    them to carry and a second identical graph would be pure mint bill. A
    bucket-0 family forks into nothing either — its cell IS the branchless
    class already.

    Adapter-bearing rows come FIRST so the composed pipeline, which arrives
    lifted from :func:`aot_inputs.compose`, is disarmed exactly once.
    """
    if not int(spec.lora_bucket or 0):
        return [(plan, None) for plan in plans]
    from .models import lora_lifted

    branch_owners = {
        id(m) for m in lora_lifted.branch_targets(pipeline).values()}
    rows: List[Tuple[Any, Optional[bool]]] = []
    for plan in plans:
        resolved = _resolve_target(pipeline, str(plan.target))
        owner = resolved[0] if resolved else None
        if owner is None or id(owner) not in branch_owners:
            rows.append((plan, None))
            continue
        rows.append((with_adapter_arm(plan, True), True))
        rows.append((with_adapter_arm(plan, False), False))
    rows.sort(key=lambda row: row[1] is False)   # stable: order within a group
    return rows


def declaration_module_gaps(
    pipeline: Any, spec: ExportSpec, decl: Any,
) -> List[str]:
    """Every declared input name a graph class's target module cannot take —
    the mint's own signature refusal, asked BEFORE anything is rented
    (pgw#822).

    pgw#822 cost a real L4 pod to learn a sentence a signature comparison
    could have produced locally: the ``lora64`` bucket declares two lifted
    adapter inputs and the module handed to ``torch.export`` did not take
    them. Nothing about that needed a GPU, a child process, or a weight read.

    PER CLASS, because the fork's two halves ask different questions
    (pgw#790): the ``adapter=true`` class is exported from the LIFTED
    forward, so ``lora_a``/``lora_b`` are admissible on a target the mint
    will lift; the ``adapter=false`` class is exported from the PLAIN module
    and declares no adapter at all (``_entry_spec`` zeroes its bucket). A
    single one-size check would either refuse the branchless class or admit a
    lifted declaration on a module that can never carry one.

    ``lora_a``/``lora_b`` are admitted on the strength of the target being
    LIFT-CAPABLE (``lora_lifted.branch_targets``), not of it being lifted
    right now: the caller is a serving parent, and the lift is installed by
    :func:`_arm_branches` inside the mint. That is the same COMPOSED-truth
    scoping ``_export_entry`` applies, so this check predicts the mint rather
    than describing the parent.

    Returns gap sentences; empty means every declared class fits its module.
    Never raises — an unreadable declaration is itself a gap, and the caller
    is deciding whether to mint, not whether to serve.
    """
    from . import aot_declaration as _decl  # deferred: aot_declaration imports us
    from .models import lora_lifted

    gaps: List[str] = []
    try:
        rows = adapter_arm_plans(
            _decl.cell_plans(
                decl, regional=bool(getattr(decl, "regional", False))),
            pipeline, spec)
        branch_owners = {
            id(m) for m in lora_lifted.branch_targets(pipeline).values()}
    except MintRefused as exc:
        # A refused declaration is PROVEN unmintable — the mint would raise
        # the same sentence, so say it now.
        return [f"the declaration's class set is unmintable ({exc})"]
    except Exception as exc:  # noqa: BLE001
        # Anything else means this check could not READ the composed
        # pipeline. A gate that declines when it cannot see is a new
        # silent-decline (the pgw#813/#815 class in a new hat) — it reports
        # what it can PROVE and abstains otherwise. The mint's own gates stay
        # load-bearing either way.
        logger.debug(
            "aot-mint: declaration/module check not applicable (%s: %s)",
            type(exc).__name__, exc)
        return []

    for plan, _arm in rows:
        entry = _decl.plan_entry_name(plan)
        try:
            espec = _entry_spec(spec, plan, decl)
            resolved = _resolve_target(pipeline, espec.target)
            if resolved is None:
                # Not this gate's question. "The pipeline carries no such
                # target" is a different condition with its own owners
                # (`compile_cache.has_compile_target` on the way in,
                # `_export_entry` on the way out); with no module there is no
                # signature to compare a declaration against, and claiming
                # one would make this check refuse every pipeline shape it
                # merely does not recognise.
                continue
            owner, _attr, _fn = resolved
            positional, keyword_only = _decl.call_signature(
                owner, espec.target, decl.family)
            takes = {p.name for p in positional}
            declared = {i.top_name for i in _decl.target_inputs(
                decl, espec.target)}
            declared |= {a.name for a in _decl.target_args(decl, espec.target)}
            if espec.lifted_inputs and id(owner) in branch_owners:
                # The mint lifts this target; the pair is admissible even
                # though the resident forward does not take it yet.
                declared |= set(espec.lifted_inputs)
                takes |= set(espec.lifted_inputs)
            elif espec.lifted_inputs:
                gaps.append(
                    f"entry {entry!r}: the class declares lifted adapter "
                    f"input(s) {list(espec.lifted_inputs)!r} but "
                    f"{type(owner).__name__} carries no branch-capable "
                    f"module, so no lifted forward can be installed on it")
                continue
        except MintRefused as exc:
            gaps.append(f"entry {entry!r}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 — see the abstain note above
            logger.debug(
                "aot-mint: entry %r not checkable (%s: %s)",
                entry, type(exc).__name__, exc)
            continue
        blocked = sorted(declared & keyword_only)
        if blocked:
            gaps.append(
                f"entry {entry!r}: declared input(s) {blocked!r} are "
                f"KEYWORD-ONLY on {type(owner).__name__}."
                f"{_decl.target_attr(espec.target)} — the mint feeds every "
                f"input positionally")
        missing = sorted(declared - takes - keyword_only)
        if missing:
            gaps.append(
                f"entry {entry!r}: declared input(s) {missing!r} are not "
                f"parameters of {_decl.target_attr(espec.target)!r} on "
                f"{type(owner).__name__} (parameters: {sorted(takes)!r})")
    return gaps


def _disarm_branches(pipeline: Any) -> None:
    """Take the pipeline back to the BRANCHLESS graph family (pgw#790).

    Both halves are required: removing the lifted forward alone would leave
    the zeroed branch containers on every leaf, and the trace would still emit
    the branch — the exact arithmetic-over-zeros this fork exists to delete.
    """
    from .models import lora_lifted, w8a8_lora

    lora_lifted.remove_lifted_lora_lanes(pipeline)
    w8a8_lora.disable_branch_lanes(pipeline)
    logger.info(
        "aot-mint: branch containers dropped — exporting the adapter=false "
        "graph class(es)")


def _arm_branches(pipeline: Any, bucket: int) -> None:
    """Put the pipeline on the LIFTED branch-bearing graph family — canonical
    branch containers first, lifted call signature second (pgw#822).

    The SAME two shipped calls, in the SAME order, as the serving arm
    (``models.provision.arm_aot``): a lifted class exports the denoiser's
    lifted forward, so the module must carry it before ``builder()`` binds
    ``lora_a``/``lora_b`` to positional slots. pgw#822 is what happens when
    it does not — the child armed the DYNAMO lane (containers only) and the
    declaration was refused against the bare ``UNet2DConditionModel``.

    Owned HERE rather than left to the caller because this function already
    owns the other half of the fork (:func:`_disarm_branches`): an arm state
    machine with one end enforced and the other a convention is how the
    convention gets skipped. Idempotent, so a caller that already composed
    lifted (:func:`aot_inputs.compose`) pays nothing.

    Also the RE-arm after the branchless exports: the mint process may go on
    to serve or re-mint, and a pipeline left branchless would silently be a
    different graph family.
    """
    from .models import lora_lifted

    lora_lifted.arm_lifted_lora_lanes(pipeline, int(bucket or 0))


# pgw#824: the phase tokens the in-mint progress callback reports under.
# Wire-shared with ``activity.PHASE_*`` and framed verbatim by ``mint_child``,
# so a reader groups a delegated mint's progress on the SAME strings whether it
# came from the child's frames or from the parent's own activity.
PHASE_TRACE_GRAPH = "trace_graph"
PHASE_INDUCTOR_COMPILE = "inductor_compile"
PHASE_SEAL_PUBLISH = "seal_publish"


@dataclass
class MintProgress:
    """The ONE record of where a mint IS — live and post-mortem.

    pgw#824 and pgw#825 arrived at the same question from opposite ends and
    must not answer it twice. pgw#825 needs the partial state (`minted`,
    `timings`, `width`, the mint's own clock) so a mint that ABORTS still
    reports the seconds it spent instead of a bare wall clock. pgw#824 needs
    those same positions pushed out LIVE, because a 20-minute export that
    reports nothing is indistinguishable from a hung one.

    One object carries both, so a live beat and an abort table can never
    disagree about which entry the mint was on: :meth:`beat` records the
    position it reports, and :func:`_attach_partial_phases` stamps that same
    position onto the aborted table as ``at``. That closes pgw#825's one
    remaining blind spot — its per-entry rows name the entries that FINISHED,
    and the entry a mint dies ON is the one a reader most needs named.

    Handed down and mutated in place; ``on_progress`` is optional and
    best-effort by construction — a raising callback never costs a mint.
    """

    inductor_configs: Optional[Mapping[str, Any]] = None
    on_progress: Optional[Callable[[str, int, int, str], None]] = None
    t_mint: Optional[float] = None
    timings: Dict[str, float] = field(default_factory=dict)
    minted: List[_MintedEntry] = field(default_factory=list)
    width: Optional[aot_compile_pool.PoolWidth] = None
    #: pgw#842: the pool's own ledger (pgw#830) once it has run, carried here
    #: so it reaches the phase table — and therefore the hub — instead of
    #: dying in the mint child's log with the pod.
    pool_ledger: Dict[str, Any] = field(default_factory=dict)
    #: The last position :meth:`beat` reported, verbatim.
    at: Dict[str, Any] = field(default_factory=dict)

    def beat(self, phase: str, step: int, total: int, note: str = "") -> None:
        """Report a position, and REMEMBER it.

        Recording precedes reporting: the position must survive into the abort
        table even when there is no ``on_progress`` sink and even when the sink
        raises, or the two halves of this object would tell different stories.
        """
        phase, step, total = str(phase), int(step), int(total)
        note = str(note)[:180]
        self.at = {
            "phase": phase, "step": step, "total": total, "note": note}
        if self.on_progress is None:
            return
        try:
            self.on_progress(phase, step, total, note)
        except Exception:  # noqa: BLE001 — telemetry never fails a mint
            logger.debug("aot-mint progress callback failed", exc_info=True)


def mint(
    pipeline: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    allow_regressed_lanes: bool = False,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    entry_workers: int = 0,
    on_progress: Optional[Callable[[str, int, int, str], None]] = None,
) -> MintResult:
    """:func:`_mint_cell`, with the phase table attached to EVERY terminus.

    pgw#825: an aborted mint used to report a wall-clock total and nothing
    else — `compile_s`, `export_s` and `n_entries` all parsed to `-` — so a
    run that paid for real compiles and then refused produced no measurement
    at all, and the pod it happened on is gone by the time anyone reads the
    table. The live entry list and timings dict are handed down and mutated
    in place, so whatever completed before the terminus is reportable from
    here whether the mint returned, refused, or died.

    pgw#824 rides the SAME record (:class:`MintProgress`) rather than a second
    one: ``on_progress(phase, step, total, note)`` is where those in-flight
    positions are pushed out live, and the last one pushed is what an aborted
    table reports as ``at``.
    """
    progress = MintProgress(
        inductor_configs=inductor_configs, on_progress=on_progress)
    try:
        return _mint_cell(
            pipeline, spec, out_dir,
            allow_regressed_lanes=allow_regressed_lanes,
            inductor_configs=inductor_configs,
            entry_workers=entry_workers, progress=progress)
    except BaseException as exc:
        _attach_partial_phases(exc, progress)
        raise


def _attach_partial_phases(exc: BaseException, progress: MintProgress) -> None:
    """Hang the partial phase table off a failed mint's exception.

    Telemetry never changes an outcome, so every step is guarded: a mint that
    refuses must refuse with ITS sentence, not with a reporting error.
    """
    try:
        minted = list(progress.minted)
        timings = dict(progress.timings)
        started = progress.t_mint
        where = dict(progress.at)
        if not minted and not timings and not where:
            return
        if started is not None:
            # The mint's OWN wall clock, so an aborted total is comparable
            # with a completed one rather than being a sum of entry seconds.
            timings["total_s"] = round(time.monotonic() - float(started), 2)
        table = _mint_phase_table(
            minted, timings, progress.inductor_configs, progress.width,
            progress.pool_ledger)
        table["terminus"] = "aborted"
        if where:
            # pgw#824 x pgw#825: the entries block names what FINISHED; this
            # names what the mint was ON. Without it an 18-entry mint that
            # dies in entry 12's export reports 11 rows and no twelfth, and
            # the row that matters is the missing one.
            table["at"] = where
        setattr(exc, "mint_phases", table)
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("aot-mint: partial phase table unavailable", exc_info=True)


def _mint_cell(
    pipeline: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    allow_regressed_lanes: bool = False,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    entry_workers: int = 0,
    progress: Optional[MintProgress] = None,
) -> MintResult:
    """Export + compile EVERY declared graph class and pack them as ONE
    multi-graph cell (pgw#758).

    ``entry_workers`` CAPS pgw#809's compile pool; it never widens it. The
    width is derived from the pod (see :func:`aot_compile_pool.entry_workers`),
    and this argument exists so an operator or a test can force the serial
    path (``1``) without pretending a 4-vCPU pod is an H100 host.

    ``progress`` (:class:`MintProgress`) is the mint's own position record —
    handed down by :func:`mint`, mutated in place, and read back by
    :func:`_attach_partial_phases` on any terminus. ``progress.beat()``
    reports progress WITHIN the mint (pgw#824). This function used to be one
    opaque call spanning the family's whole declared class set — sdxl declares
    18 — so the mint child framed ``trace_graph`` once and said nothing again
    until ``seal_publish``. A real export measured ~5 minutes of complete wire
    silence, and the pod's only liveness evidence was that its CPU was warm.
    "Entry 12 of 18" is the difference between a pod that is alive and a pod
    that is working, which is exactly the distinction the no-magic-timeouts
    doctrine runs on.

    Does NOT publish — :func:`publish` is a separate step so a mint can be
    inspected, byte-compared (#699 double-mint), or produced on a box with no
    hub credentials.
    """
    from .api.export_contract import export_declaration

    refusal = lane_admitted(spec, allow_regressed_lanes=allow_regressed_lanes)
    if refusal:
        raise MintRefused(refusal)
    refusal = lifted_torch_gap(spec)
    if refusal:
        raise MintRefused(refusal)
    decl = export_declaration(spec.family)
    if decl is None:
        raise MintRefused(
            f"family {spec.family!r} has no registered export declaration — "
            f"a multi-graph cell derives its class set from the declaration "
            f"(pgw#739/#758); register one before minting")
    if decl.warm_changes_key is None:
        raise MintRefused(
            f"family {spec.family!r} declares no mint-warm canon "
            f"(warm_changes_key) — whether pre-warm changes the graph is a "
            f"measured per-family FACT (sdxl False, z-image True), not a "
            f"default")

    out_dir = Path(out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    # Handed to the caller BY REFERENCE and mutated in place: an aborted mint
    # reports the seconds it did spend (pgw#825) and the position it was on
    # (pgw#824), out of ONE record.
    progress = MintProgress() if progress is None else progress
    timings = progress.timings
    t_mint = time.monotonic()
    progress.t_mint = t_mint

    from . import aot_declaration as _decl  # deferred: aot_declaration imports us

    # pgw#817: the declaration decides the mint SHAPE. `regional=True` means
    # this family's entries are block classes of each target rather than shape
    # coordinates of its whole forward — the same plans, the same forks, the
    # same class rows, exported one block deep.
    regional = bool(getattr(decl, "regional", False))
    if regional:
        spec = replace_spec(
            spec, regional=True, shell_digest=_cell_shell_digest(pipeline, decl))
        logger.info(
            "aot-mint: REGIONAL cell for %s — entries are block classes; the "
            "shell stays eager (pgw#812 S2), shell_digest=%s",
            spec.family, spec.shell_digest)

    rows = adapter_arm_plans(
        _decl.cell_plans(decl, regional=regional), pipeline, spec)
    # pgw#809: how wide this pod may compile. Derived from the pod's REAL
    # budget (cgroup-aware vCPUs minus serving headroom, and available host
    # RAM over the measured per-entry peak) — never os.cpu_count, never a
    # constant. K=1 IS the pre-#809 serial in-process path, which is the
    # honest answer on a narrow pod.
    # pgw#817 / pgw#812 S7 — pgw#809 RE-PRICED, not assumed. Regional changes
    # BOTH of the pool's inputs, in opposite directions:
    #   * the entry COUNT goes up (one entry per block class per plan, not one
    #     per plan), which is what K parallelises over;
    #   * the per-entry DEVICE ask goes down by roughly the block fraction of
    #     the model, and VRAM is the bound that actually binds K.
    # Multiplying the two levers instead of re-pricing them would size the
    # pool for a whole-model child that regional never runs.
    # pgw#829 moves the first lever the other way: with `dynamic-collapse` the
    # plan set itself shrinks to the FORK coordinates, so a conv-free family
    # has fewer entries than the whole-graph shape had, each one wider.
    entry_count = _regional_entry_count(pipeline, decl, rows) if regional \
        else len(rows)
    width = aot_compile_pool.entry_workers(
        entry_count, limit=int(entry_workers or 0),
        device_bytes=_entry_device_bytes(
            spec, block_fraction=_block_device_fraction(pipeline, decl)
            if regional else 1.0))
    parallel = width.workers > 1
    logger.info("aot-mint: entry compile width — %s", width.reason)
    if width.underwidth:
        # pgw#842: a pool narrower than the cell could use is a COST, and it
        # is the mint's only multiplicative lever. Say so at WARNING with the
        # readings behind it — the same facts ride the `pool` event.
        logger.warning(
            "aot-mint: pgw#842 entry pool runs %d worker(s) narrower than "
            "this cell could use (K=%d of %d), held by %s — inputs %s",
            width.underwidth, width.workers,
            min(entry_count, width.ceiling), width.binding, width.facts())
    progress.width = width

    minted = progress.minted
    disarmed = False
    # pgw#822: the adapter-BEARING classes are exported from the lifted
    # forward. Arming it is this function's job, not the caller's — see
    # `_arm_branches`.
    _arm_branches(pipeline, int(spec.lora_bucket or 0))
    t_export = time.monotonic()
    progress.beat(
        PHASE_TRACE_GRAPH, 0, len(rows),
        f"{len(rows)} declared class row(s)")
    try:
        for index, (plan, arm) in enumerate(rows, start=1):
            if arm is False and not disarmed:
                # ONE toggle for the whole branchless group (the rows are
                # ordered adapter-bearing first): disable/enable reallocates
                # every leaf's branch container, and doing it per entry would
                # be N times the VRAM churn for the same graphs.
                _disarm_branches(pipeline)
                disarmed = True
            # Reported BEFORE the work, not after: a row that never returns
            # is the one a reader most needs named, and an after-the-fact tick
            # names only the rows that finished.
            progress.beat(
                PHASE_TRACE_GRAPH, index, len(rows),
                _decl.plan_entry_name(plan))
            if regional:
                minted.extend(_export_regional_entries(
                    pipeline, spec, plan, decl,
                    inductor_configs=inductor_configs,
                    compile_now=not parallel))
            else:
                minted.append(_export_entry(
                    pipeline, spec, plan, decl,
                    inductor_configs=inductor_configs,
                    compile_now=not parallel))
    finally:
        if disarmed:
            _arm_branches(pipeline, int(spec.lora_bucket or 0))

    # Asked of the EXPORTED programs: a cell whose entries cannot be told
    # apart at dispatch must cost seconds to refuse, not a full compile bill
    # (the pgw#825 discipline, one gate over). Exact on the parallel path,
    # which is every real mint — pgw#809 sizes width off the pod's own budget
    # and the pool has not built a kernel yet. A width-1 serial mint has
    # already compiled as it exported, so there it refuses late; correct
    # either way, cheap where it matters.
    _gate_dispatch_ambiguity(minted)

    if parallel:
        timings["export_all_s"] = round(time.monotonic() - t_export, 2)
        progress.beat(
            PHASE_INDUCTOR_COMPILE, 0, len(minted),
            f"{len(minted)} entries, {width.workers} wide")
        progress.pool_ledger = _compile_entries_parallel(
            minted, work, width, inductor_configs=inductor_configs,
            on_entry=lambda name, done, total: progress.beat(
                PHASE_INDUCTOR_COMPILE, done, total, name))
    timings["entry_workers"] = float(width.workers)

    t0 = time.monotonic()
    progress.beat(
        PHASE_SEAL_PUBLISH, len(minted), len(minted),
        f"packaging {len(minted)} entries")
    package = package_cell(
        {row.name: row.files for row in minted}, work / aot_serve.PACKAGE_NAME)
    timings["package_s"] = round(time.monotonic() - t0, 2)

    t0 = time.monotonic()
    entry_blocks: Dict[str, Dict[str, Any]] = {}
    for row in minted:
        entry_blocks[row.name] = _gate_and_declare_entry(row, package)
    _write_literals(minted, package, work)

    try:
        meta = aot_serve.artifact_metadata(
            family=spec.family,
            precision=spec.precision,
            cell_key="",
            entries=entry_blocks,
            strict_export=bool(spec.strict),
            lora_bucket=int(spec.lora_bucket or 0),
            source_ref=spec.source_ref,
            source_digest=spec.source_digest,
        )
    except ValueError as exc:
        # The envelope validates the contract it is handed. A malformed one must
        # fail HERE, on the mint pod, not at serve time on a paying request.
        raise MintRefused(
            f"envelope refused the declared contract: {exc}") from exc
    meta.update(shared_identity_blocks(spec))
    mode_drift = aot_package.strict_mode_drift(meta, spec.strict)
    if mode_drift:
        raise MintRefused("trace-mode drift: " + "; ".join(mode_drift))

    timings["declare_s"] = round(time.monotonic() - t0, 2)
    timings["total_s"] = round(time.monotonic() - t_mint, 2)
    phase_table = _mint_phase_table(
        minted, timings, inductor_configs, width, progress.pool_ledger)
    _emit_phase_event(spec, phase_table)

    meta["cell_key"] = key = cell_identity(meta, spec).digest
    t0 = time.monotonic()
    artifact = aot_serve.pack(work, out_dir / f"{key}.tar.gz", meta)
    timings["pack_s"] = round(time.monotonic() - t0, 2)
    # The phase table rides the RESULT (and the published checkpoint
    # metadata + the typed event), never the packed envelope: durations in
    # metadata.json would break the #699 double-mint byte-compare — the
    # artifact deliberately carries no timestamps and no wall clocks.
    meta["mint_phases"] = phase_table

    logger.info(
        "aot-mint: %s lane=%s -> %s (%d entr%s across %d target(s), %.1f MB "
        "package, combined=%s, %s)",
        spec.family, spec.lane_label() or "(plain)", key,
        len(minted), "y" if len(minted) == 1 else "ies",
        len({row.spec.target for row in minted}),
        package.stat().st_size / 1e6, meta.get("combined_graph_hash"),
        timings,
    )
    return MintResult(artifact=artifact, metadata=meta, timings=timings)


def _regional_targets(
    pipeline: Any, decl: Any,
) -> Dict[str, Tuple[Any, Tuple[aot_regional.BlockGroup, ...]]]:
    """``{target: (module, block groups)}`` for every declared target with
    repeated blocks — the regional mint's own view of the declaration.

    The resolved MODULE rides along because every caller needs it (the shell
    digest reads its config, the device re-price reads its parameter count)
    and re-resolving per caller is how the two would drift apart.
    """
    out: Dict[str, Tuple[Any, Tuple[aot_regional.BlockGroup, ...]]] = {}
    for target in tuple(getattr(decl, "targets", ()) or ()):
        resolved = _resolve_target(pipeline, str(target))
        if resolved is None:
            continue
        owner, attr, _fn = resolved
        if attr != "forward":
            continue
        groups = aot_regional.repeated_block_groups(owner)
        if groups:
            out[str(target)] = (owner, groups)
    return out


def _cell_shell_digest(pipeline: Any, decl: Any) -> str:
    """The cell-level shell digest: every regional target's shell, together.

    A cell can carry more than one regional target, so the contract fact binds
    ALL of them — binding only the first would leave a second target's shell
    free to drift under an unchanged key, which is the collision class this
    axis exists to close.
    """
    facts = {
        target: aot_regional.shell_facts(owner)
        for target, (owner, _groups) in _regional_targets(pipeline, decl).items()}
    if not facts:
        raise MintRefused(
            f"family {getattr(decl, 'family', '?')!r} declares regional=True "
            f"but no declared target resolves to a module with "
            f"`_repeated_blocks` — there is no repeated structure to compile "
            f"once and reuse")
    encoded = json.dumps(
        facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        default=str).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()[:16]


def _regional_entry_count(pipeline: Any, decl: Any, rows: Sequence[Any]) -> int:
    """How many ENTRIES a regional mint of these plans will produce.

    One per (plan, block class) — except that pgw#829 lets a CONV-FREE block
    class collapse a plan's whole row set into one dynamic entry while a
    conv-bearing one keeps a static entry per row. So the count is derived
    the same way the export loop derives it, off the same live module: a
    pool sized on the wrong number is the pgw#812 S7 re-price mistake in
    reverse.

    sdxl before pgw#829: 36 plans (18 rows x 2 adapter arms) x 2 block
    classes = 72. After: 4 plans (2 CFG forks x 2 arms) x 2 conv-free block
    classes = 8.
    """
    by_target = _regional_targets(pipeline, decl)
    total = 0
    for plan, _arm in rows:
        entry = by_target.get(str(getattr(plan, "target", "")))
        if entry is None:
            continue
        plan_rows = max(1, len(tuple(getattr(plan, "rows", ()) or ())))
        for group in entry[1]:
            total += 1 if plan_rows <= 1 or not aot_regional.block_has_conv(
                group.prototype) else plan_rows
    return max(total, len(rows))


def _block_device_fraction(pipeline: Any, decl: Any) -> float:
    """The fraction of the model's DEVICE footprint one block-entry child
    holds, read off the resolved module.

    A whole-graph entry child loads the whole target and benchmarks kernels
    against it; a regional child needs one BLOCK's weights and one block's
    activation set. Measured off the module rather than assumed: the largest
    block class's parameter count over the target's, floored at 0.1 because
    the CUDA context, torch's own allocator arenas and the activation set do
    NOT shrink with the block — a fraction that reads 0.014 (one of 70 sdxl
    blocks) would license a pool the card cannot hold.
    """
    biggest = 0.0
    for _target, (owner, groups) in _regional_targets(pipeline, decl).items():
        total = float(sum(p.numel() for p in owner.parameters()))
        if total <= 0.0:
            continue
        for group in groups:
            block = float(sum(p.numel() for p in group.prototype.parameters()))
            biggest = max(biggest, block / total)
    return max(0.1, min(1.0, biggest)) if biggest > 0.0 else 1.0


def _entry_device_bytes(spec: ExportSpec, *, block_fraction: float = 1.0) -> int:
    """One entry child's DEVICE ask, from this process rather than a constant.

    An AOTI compile benchmarks kernels on the card, so an entry child holds
    its own weight copy, one activation set and a CUDA context — which is
    exactly what ``mint_budget.co_residency`` already computes for the pgw#784
    mint child, read off the pipeline THIS process has resident. Reused rather
    than re-derived: the entry child loads the same weights at the same lane
    and runs the same declared shapes, so the mint child's own footprint is a
    proxy and not a guess. 0 means "unprobeable", and the width policy treats
    that as "do not license concurrency on a card you cannot measure".
    """
    try:
        from . import mint_budget

        budget = mint_budget.co_residency(
            family=str(spec.family or ""),
            weight_lane=str(spec.lane_label() or ""))
    except Exception:  # noqa: BLE001
        return 0
    if not budget.probed:
        return 0
    return int(int(budget.need_bytes) * max(0.0, min(1.0, block_fraction)))


def _compile_entries_parallel(
    minted: List[_MintedEntry],
    work: Path,
    width: aot_compile_pool.PoolWidth,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    on_entry: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """pgw#809: fill every entry's ``files`` K-wide, out of process.

    Returns the pool's own ledger (pgw#830) so it reaches the phase table:
    the pool emits it as a typed event too, but that emission happens in the
    mint CHILD, which holds no orchestrator session — pgw#842.

    Mutates ``minted`` in place, and every entry MUST come back with files —
    a pool that quietly returned fewer entries than it was given would pack a
    short cell. Assembly is by entry NAME: ``package_cell`` reads
    ``{row.name: row.files}`` in the order ``minted`` already holds (the
    declaration's order), so completion order is not observable in the
    artifact.
    """
    # A SIBLING of work/, never inside it. pack() only copies a fixed member
    # set so debris there would be harmless today, but a pool workdir living
    # inside the directory that becomes the artifact is one refactor away from
    # putting job files and stderr tails into a cell.
    pool = aot_compile_pool.EntryCompilePool(
        work.parent / "entry-pool", width=width,
        inductor_configs=inductor_configs)
    t0 = time.monotonic()
    try:
        by_entry = pool.compile(
            [(row.name, row.program) for row in minted], on_entry=on_entry)
    except aot_compile_pool.EntryCompileFailed as exc:
        # Named, and terminal: the siblings are already torn down group-wide
        # by the pool. A mint that says only "a compile failed" over 18
        # entries is the silent-failure path in a new hat (pgw#758).
        raise MintRefused(str(exc)) from exc
    wall = time.monotonic() - t0
    missing = [row.name for row in minted if row.name not in by_entry]
    if missing:
        raise MintRefused(
            f"entry compile pool returned {len(by_entry)} of {len(minted)} "
            f"entries — missing {missing!r}. Packing the rest would ship a "
            f"cell whose declared class set is a lie")
    for row in minted:
        row.files = list(by_entry[row.name])
        row.timings["compile_s"] = pool.entry_seconds.get(row.name, 0.0)
        # Measured in the child; folded in here so the roll-up reads the same
        # whether a cell was minted serially or K-wide.
        phases = pool.entry_phases.get(row.name) or {}
        if phases:
            row.timings["phases"] = dict(phases)
    logger.info(
        "aot-mint: pgw#809 pool compiled %d entr%s at K=%d in %.0fs "
        "(sum of entry seconds %.0fs, peak child RSS %.1f GiB)",
        len(minted), "y" if len(minted) == 1 else "ies", width.workers, wall,
        sum(pool.entry_seconds.values()), pool.peak_rss_bytes / 1024**3)
    return {
        **pool.ledger.facts(),
        # Observed, not intended: the only load-independent evidence that the
        # pool actually overlapped rather than looping K-wide on paper.
        "peak_concurrency": int(pool.peak_concurrency),
        "peak_child_rss_bytes": int(pool.peak_rss_bytes),
    }


def _entry_dispatch_signature(row: "_MintedEntry") -> str:
    """What DISPATCH can see of one entry's exported program.

    The packed ingress contract is derived from these placeholders, so two
    entries with the same signature necessarily declare the same contract —
    which makes this an exact discriminator that is available BEFORE the
    compile is paid for, rather than a second implementation of the contract
    that could drift from it.
    """
    shapes = _placeholder_shapes(row.program)
    by_node: Dict[str, Any] = {}
    graph = getattr(getattr(row.program, "graph_module", None), "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        if getattr(node, "op", "") == "placeholder":
            by_node[str(node.name)] = node.meta.get("val")
    parts = []
    for name in sorted(shapes):
        val = by_node.get(name)
        parts.append(
            f"{name}:{getattr(val, 'dtype', '?')}:"
            + ",".join(str(d) for d in shapes[name]))
    return "|".join(parts)


def _gate_dispatch_ambiguity(minted: Sequence["_MintedEntry"]) -> None:
    """Refuse a cell whose entries CANNOT be told apart at dispatch.

    :meth:`aot_serve.EntryDispatch.select` calls two entries admitting one
    call ``entry_ambiguous`` — "a declaration that cannot discriminate two
    graph classes by ingress, which is a defect to surface, never a coin to
    flip". It is a per-REQUEST refusal, so today the cell arms, reports
    armed, and serves those coordinates 100 % eager. Nothing fails.

    Found by pgw#829's own A/B, and it is not hypothetical. A REGIONAL entry
    is exported one block deep, and the shell has already flattened the
    latent extents into a token count — so two declared class rows that are
    genuinely different coordinates upstream can hand the block the identical
    shape. What actually collides is the token PRODUCT, of which a transposed
    aspect pair is only the obvious case. sdxl's nine aspect rows carry just
    FOUR distinct token counts::

        15360  1536x640, 640x1536                          (latent 80x192, 192x80)
        15808  1216x832, 832x1216                          (104x152, 152x104)
        16128  1344x768, 1152x896, 896x1152, 768x1344      (96x168, 112x144, ...)
        16384  1024x1024                                   <- the only unique row

    Three clash groups, one of them a QUADRUPLE spanning rows that are not
    each other's transpose (96*168 == 112*144). So attempt nine's 72-entry
    cell could have served exactly one of its nine aspect ratios compiled;
    the other eight were `entry_ambiguous` -> eager, per (CFG arm x adapter
    arm x block class).

    Grouped exactly the way the serve path groups — target, block class,
    adapter arm — because those are the axes dispatch resolves BEFORE
    ingress, and two entries on different arms are meant to share a contract
    (pgw#825: a block never carries the lifted pair, so both arms declare the
    same one). The remedy is the collapse: one entry over the hull admits
    both rows and is unique by construction.
    """
    groups: Dict[Tuple[str, str, Any], Dict[str, List[str]]] = {}
    for row in sorted(minted, key=lambda r: r.name):
        fork = {str(n): v for n, v in tuple(row.spec.fork)}
        key = (str(row.spec.target),
               str(fork.get(aot_regional.BLOCK_FORK) or ""),
               fork.get(ADAPTER_FORK))
        groups.setdefault(key, {}).setdefault(
            _entry_dispatch_signature(row), []).append(row.name)
    clashes = [
        names for by_digest in groups.values()
        for names in by_digest.values() if len(names) > 1]
    if not clashes:
        return
    detail = "; ".join(
        f"{names[0]!r} and {names[1]!r}" + (
            f" (+{len(names) - 2} more)" if len(names) > 2 else "")
        for names in clashes[:4])
    raise MintRefused(
        f"dispatch-ambiguity gate: {len(clashes)} group(s) of entries "
        f"declare the SAME ingress contract and route to the same dispatch, "
        f"so every call they both admit is refused 'entry_ambiguous' and "
        f"served EAGER — {detail}. Two class rows that reduce to one graph "
        f"shape are one entry: collapse them "
        f"(Compile.regional_shape_strategy='dynamic-collapse' for a "
        f"conv-free block population, pgw#829) rather than compiling and "
        f"publishing a class the cell can never select")


def _gate_and_declare_entry(
    row: _MintedEntry, package: Path,
) -> Dict[str, Any]:
    """Run every package-side gate for one entry and build its envelope
    block. Refusals name the entry AND the cause (pgw#758)."""
    entry = row.name
    violations = aot_package.code_only_violations(package, entry)
    if violations:
        raise MintRefused(
            f"entry {entry!r}: code-only gate (pgw#704 B1): "
            + "; ".join(violations))
    unbindable = aot_package.unbindable_constants(
        package, _state_dict_keys(row.owner), entry)
    if unbindable:
        raise MintRefused(
            f"entry {entry!r}: bindability gate: " + "; ".join(unbindable))
    # pgw#728: strict and non-strict traces lift DIFFERENT constant sets, so the
    # manifest must be proven to describe the package that ships beside it. Two
    # independent derivations (program vs generated wrapper) required to agree —
    # drift the env seal cannot see, because both modes run identically sealed.
    drift = aot_package.program_package_drift(row.program, package, entry)
    if drift:
        raise MintRefused(
            f"entry {entry!r}: constant-set drift: " + "; ".join(drift))
    fused = aot_package.eliminated_constants(row.program, package, entry)
    if fused:
        # Routine compiler fusion (measured on real sdxl: conv_out.bias folded
        # into the conv epilogue). Recorded, never fatal — but a surprising jump
        # in the count should be visible rather than silently discarded.
        logger.info(
            "aot-mint: %s: %d lifted constant(s) fused away by the compiler "
            "(e.g. %s)", entry, len(fused), fused[:3])
    if row.spec.regional and fused:
        # pgw#827 (found by wiring the regional arm): "recorded, never fatal"
        # is right for a WHOLE-GRAPH cell — it is minted from the very weights
        # it serves, so a folded constant is that module's own value. It is
        # FATAL for a regional cell. One artifact serves N INSTANCES with
        # DIFFERENT weights, so a constant the compiler folded away carries
        # the PROTOTYPE instance's bytes into every other instance, silently,
        # with no unbound constant and no refusal anywhere.
        #
        # MEASURED off-pod (torch 2.13.0+cu130, CPU, a 3-block toy): with
        # `ff.bias` folded, instance 0 reproduces eager exactly (0.0) and
        # instance 1 is wrong by 0.53. Nothing in the artifact, the manifest
        # or the bind gate can see it — which is why it is refused HERE.
        #
        # The remedy is proven and is NOT this refusal: compiling regional
        # entries under `aot_inductor.use_runtime_constant_folding=True` keeps
        # every folded constant a real bindable input (verified: `fused` goes
        # to [] and `ff.bias` reappears in the artifact's own table). It also
        # adds `_FOLDED_CONST_*` rows the manifest and the bind path must
        # learn, and it re-keys every cell — so it is a change with a train,
        # not a hotfix. Until then a cell that would serve wrong numbers must
        # not be published.
        state_dict_fused = [
            name for name in fused
            if name in set(aot_package.program_state_dict_fqns(row.program))]
        if state_dict_fused:
            raise MintRefused(
                f"entry {entry!r}: the compiler folded {len(state_dict_fused)} "
                f"state_dict constant(s) away "
                f"({sorted(state_dict_fused)[:6]!r}). A REGIONAL entry is "
                f"reused across every instance of its block class, so a "
                f"folded constant bakes the PROTOTYPE instance's weights into "
                f"all of them — the artifact is correct for instance 0 and "
                f"silently wrong for the rest. Recompile with "
                f"`aot_inductor.use_runtime_constant_folding=True`, which "
                f"keeps them bindable")
    try:
        inputs, symbols = aot_package.input_contract(
            row.program, row.flat_names)
        constants = aot_package.constants_manifest(package, entry)
    except aot_package.PackageIntrospectionError as exc:
        raise MintRefused(f"entry {entry!r}: declaration: {exc}") from exc
    block: Dict[str, Any] = {
        "target": row.spec.target,
        "fork": [[str(n), v] for n, v in sorted(row.spec.fork)],
        "class_dims": [
            [str(n), int(v)] for n, v in sorted(row.spec.class_dims)],
        "inputs": inputs,
        "symbols": symbols,
        "constants": constants,
        "graph": entry_graph_block(row.program, package, row.name, row.spec),
    }
    if adapter_arm(row.spec.fork) is False:
        # pgw#790: the NEGATIVE half of this class's contract. Without it the
        # branchless entry silently ADMITS an adapter-bearing call (a
        # name-keyed bind ignores inputs it does not declare), the dispatch
        # sees two admitting entries and refuses `entry_ambiguous` — and the
        # cell serves the whole attach lane eagerly. Declared, so the refusal
        # is the right one and it names the input.
        from .models import lora_lifted

        block["excluded_inputs"] = list(lora_lifted.LIFTED_INPUT_NAMES)
    return block


def _mint_phase_table(
    minted: Sequence[_MintedEntry],
    timings: Mapping[str, float],
    inductor_configs: Optional[Mapping[str, Any]],
    width: Optional[aot_compile_pool.PoolWidth] = None,
    pool_ledger: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The per-mint phase table (#757's instrument-first deliverable): one
    readable record of where the mint's seconds went, per entry and in
    total, plus the two facts an AOT-vs-JIT comparison needs to be fair —
    the graph-class COUNT this mint compiled and the autotune posture.

    pgw#809 adds the ``pool`` block: the K this mint ran at AND the budget
    that chose it. Without it a mint's wall clock is uninterpretable —
    two mints of the same cell on two pods legitimately differ by 4x."""
    entries = {
        row.name: dict(row.timings) for row in minted}
    totals: Dict[str, float] = {
        "export_s": round(sum(
            float(row.timings.get("export_s") or 0) for row in minted), 2),
        "compile_s": round(sum(
            float(row.timings.get("compile_s") or 0) for row in minted), 2),
        "warm_s": round(sum(
            float(row.timings.get("warm_s") or 0) for row in minted), 2),
    }
    phase_totals: Dict[str, float] = {}
    for row in minted:
        for label, value in (row.timings.get("phases") or {}).items():
            phase_totals[label] = round(
                phase_totals.get(label, 0.0) + float(value), 3)
    # The ONE resolved inductor config every entry compiled under, recorded
    # verbatim: #757's open seal-bypass concern is a per-call config the
    # seal cannot see, so whatever the mint passed is either identity-inert
    # (compile_threads, #757 pre-verified) or visible RIGHT HERE for the
    # audit that says otherwise.
    resolved = {
        key: value if isinstance(value, (bool, int, float, str, type(None)))
        else repr(value)
        for key, value in sorted(_entry_configs(inductor_configs).items())
    }
    return {
        "v": 1,
        "n_entries": len(minted),
        "autotune": autotune_posture(inductor_configs),
        "inductor_configs": resolved,
        "totals": {**totals, **{k: v for k, v in timings.items()}},
        "phases": phase_totals,
        "entries": entries,
        # pgw#842: the width's INPUTS and the pool's own ledger ride the same
        # block, because the two questions a slow mint raises — "why this K"
        # and "what did K buy" — are answered by different halves of it, and a
        # record that carries only the scalar K (as this one did through
        # attempts ten and eleven) can answer neither.
        "pool": {
            **(width.facts() if width is not None
               else {"entry_workers": 1, "binding": "serial"}),
            **dict(pool_ledger or {}),
        },
    }


MINT_PHASES_KIND = "aot_mint_phases"


def _entry_duration_s(timings: Mapping[str, Any]) -> float:
    """One graph-class entry's own compile cost: export + compile + warm.

    The package/declare/pack phases are per-MINT, not per-entry, so they belong
    to the roll-up only — summing the entry rows must never reproduce
    ``total_s`` or a reader would double-count.
    """
    return sum(
        float(timings.get(key) or 0.0)
        for key in ("export_s", "compile_s", "warm_s")
    )


def _emit_phase_event(spec: ExportSpec, table: Mapping[str, Any]) -> None:
    """The mint's own emission (see :func:`emit_phase_events`).

    The spec is read inside the guard: telemetry must never fail the compile
    it measures, and reading the lane label off a broken spec is still
    telemetry.
    """
    try:
        family, lane = spec.family, spec.lane_label() or "plain"
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("aot-mint: phase event emission failed", exc_info=True)
        return
    emit_phase_events(family=family, lane=lane, table=table)


#: pgw#842: the mint's WIDTH decision, as its own hub row.
POOL_PHASE = "pool"


def _emit_pool_event(
    *, family: str, lane: str, table: Mapping[str, Any],
) -> None:
    """pgw#842: one event that says what K was, what chose it, and what it
    bought — the standing "no silent decisions" rule applied to the mint's
    only multiplicative lever.

    Attempts ten and eleven compiled the same 72-entry sdxl cell for the same
    seconds (1314.94 vs 1327.23) and took 347.94 s vs 554.78 s, because K was
    5 and then 3. Nothing hub-side recorded WHY: the width block existed in
    the phase table and was never emitted, and the pgw#830 pool ledger was
    emitted from the mint CHILD, which holds no orchestrator session (see
    ``mint_delegate._emit_aot_phases``) — so both were pod-log-only and died
    with the pod. A width narrower than the pod could carry is a performance
    defect; it must be READABLE from one mint's record, not inferred by
    diffing two pods that no longer exist.
    """
    pool = dict(table.get("pool") or {})
    if not pool:
        return
    from . import activity as activity_mod

    workers = int(pool.get("entry_workers") or 1)
    binding = str(pool.get("binding") or "unknown")
    under = int(pool.get("underwidth") or 0)
    wall_s = float(pool.get("pool_wall_s") or 0.0)
    head = (
        f"family={family} lane={lane} entry_workers={workers} "
        f"binding={binding} underwidth={under}")
    if under > 0:
        # Named in the FIRST line, so a narrow pool is legible without
        # parsing the dict: this is the number that cost attempt eleven 59 %.
        head += (
            f" — the pool ran {under} worker(s) narrower than this cell could "
            f"use, held by {binding}")
    activity_mod.emit_event(
        MINT_PHASES_KIND, f"{head} pool={pool}",
        phase=POOL_PHASE,
        duration_ms=int(round(wall_s * 1000)),
    )


def emit_phase_events(
    *, family: str, lane: str, table: Mapping[str, Any],
    terminus: str = "",
) -> None:
    """Typed telemetry event — the phase table must reach observability,
    never only a pod log.

    th#1322: the mint's TOTAL now rides the numeric ``duration_ms`` field, and
    each graph-class entry gets its own event under ``phase=entry:<name>``. The
    interpolated table stays in ``detail`` as the breakdown you read per event
    (exactly as ``stage_ms`` stays in the request payload), but every number a
    measurement lane groups, percentiles or trends on is now a column.

    Paul's question — "why is AOT mint so much slower than JIT mint?" — needs
    the per-entry rows: an AOT mint compiles N graph classes where a JIT mint
    compiles the graphs one real warm plan happens to trace, and the answer is
    which entries the extra minutes are in, not just that there are more.
    """
    try:
        from . import activity as activity_mod

        totals = dict(table.get("totals") or {})
        lane = lane or "plain"
        total_s = float(totals.get("total_s") or 0.0)
        # pgw#825: the roll-up's PHASE is the mint's terminus. An aborted mint
        # measured real entries and must report them — under `aborted`, never
        # under `minted`, or a partial table would enter an AOT-vs-JIT
        # comparison as if a cell came out.
        roll_up = terminus or str(table.get("terminus") or "") \
            or activity_mod.PHASE_MINTED
        activity_mod.emit_event(
            MINT_PHASES_KIND,
            f"family={family} lane={lane} status={roll_up} "
            f"n_entries={table.get('n_entries')} totals={totals} "
            f"phases={dict(table.get('phases') or {})} "
            f"autotune={dict(table.get('autotune') or {})}",
            phase=roll_up,
            duration_ms=int(round(total_s * 1000)),
        )
        _emit_pool_event(family=family, lane=lane, table=table)
        for name, timings in sorted((table.get("entries") or {}).items()):
            if not isinstance(timings, Mapping):
                continue
            entry_s = _entry_duration_s(timings)
            if entry_s <= 0:
                continue
            activity_mod.emit_event(
                MINT_PHASES_KIND,
                f"family={family} lane={lane} entry={name} "
                f"timings={dict(timings)}",
                phase=f"entry:{name}",
                duration_ms=int(round(entry_s * 1000)),
            )
    except Exception:  # pragma: no cover — telemetry must never fail a mint
        logger.debug("aot-mint: phase event emission failed", exc_info=True)


def entry_graph_block(
    program: Any, package: Path, entry: str, spec: ExportSpec,
) -> Dict[str, Any]:
    """The per-entry graph-interface facts (fold into that entry's
    ``class_hash``): the declared constant FQN set, the lifted inputs, the
    pytree spec, and the python branches export FROZE at trace time.
    Constant BYTE SIZES are deliberately absent — they are a property of the
    resident weights, and a fine-tune of one family must keep sharing
    cells, which is the premise of family-scoped cells."""
    return {
        "v": 2,
        "constant_fqns": sorted(aot_package.constant_names(package, entry)),
        "fused_constants": sorted(
            aot_package.eliminated_constants(program, package, entry)),
        "lifted_inputs": sorted(str(n) for n in spec.lifted_inputs),
        "pytree": _pytree_facts(program),
        "specialization": _specialization_facts(spec),
    }


def shared_identity_blocks(spec: ExportSpec) -> Dict[str, Any]:
    """The cell-level ck5 identity facts an exported cell must record.

    ``aot_serve.artifact_metadata`` takes ``cell_key`` as a STRING, so the
    envelope on its own would carry a stamp WITHOUT the axes the stamp
    summarizes — and ``cell_key``'s standing discipline is that a key is always
    recomputed FROM recorded facts, so a stamp can never disagree with them.
    These blocks are what make that recomputation possible for the new kind, and
    they ride the metadata additively (the envelope's parsers read named fields
    and are unaffected). Per-entry graph facts live in the ``entries`` blocks
    (:func:`entry_graph_block`) and reach the key through the combined hash.
    """
    from . import compile_cache as cc
    from . import env_seal

    return {
        "weight_lane": str(spec.weight_lane or ""),
        # pgw#817: recorded so a CONSUMER can recompute the identity from the
        # artifact's own facts — the standing ck5 discipline — instead of
        # trusting the stamp. `mode` also tells the serve path which arm to
        # take without re-deriving it from entry names.
        "mode": aot_regional.MODE_REGIONAL if spec.regional else "",
        "shell_digest": str(spec.shell_digest or ""),
        "sm": str(cc.runtime_key().get("sm") or ""),
        env_seal.SEAL_KEY: env_seal.effective_seal(),
        "toolchain": dict(cc.toolchain_digest()),
        "code_closure": dict(cc.static_code_closure(tuple(spec.closure_roots))),
        "content_keys": dict(cc.content_keys()),
        "loaded_libs": dict(env_seal.frozen_library_digests()),
        "gen_worker": cc.gen_worker_version(),
    }


def _pytree_facts(program: Any) -> Dict[str, Any]:
    """The flattened call spec the package expects.

    An AOTI package takes FLAT tensors while the pipeline calls the module with
    diffusers' nested kwargs (``added_cond_kwargs={...}``). The consumer must
    flatten exactly as export did or it feeds the wrong tensor to the wrong
    input, so the spec is declared rather than re-derived at serve time.
    """
    signature = getattr(program, "graph_signature", None)
    call_spec = getattr(program, "call_spec", None)
    return {
        "user_inputs": [
            str(n) for n in getattr(signature, "user_inputs", ()) or ()],
        "in_spec": _treespec_text(getattr(call_spec, "in_spec", None)),
        "out_spec": _treespec_text(getattr(call_spec, "out_spec", None)),
    }


def _treespec_text(spec: Any) -> str:
    if spec is None:
        return ""
    try:
        import torch.utils._pytree as pytree

        return str(pytree.treespec_dumps(spec))
    except Exception:
        # A spec we cannot canonically serialize is recorded by repr rather than
        # dropped: an unrecordable spec must still differ from a DIFFERENT
        # unrecordable spec in the key.
        return repr(spec)


def cell_identity(meta: Mapping[str, Any], spec: ExportSpec) -> cell_key.CellKey:
    """The cell key a multi-graph artifact's OWN recorded facts describe.

    Computed from the recorded blocks, never from separate probes, so the stamp
    can never disagree with the axes it summarizes — the discipline
    ``cell_key.from_artifact_metadata`` enforces for dynamo cells, mirrored for
    the new kind. ``cell_key.from_axes`` already accepts any ``kind`` VALUE (it
    validates axis NAMES), so no KEY_SCHEME bump: the axis set is unchanged and
    ``kind`` does the discriminating. No dynamo cell is stranded.

    The ``contract`` axis is the pgw#716 formula, IMPLEMENTED AS ANTICIPATED:
    the cell keys on the ``combined_graph_hash`` — first 16 hex of the sha256
    over the newline-joined SORTED per-class hashes — while the per-class
    hashes ride ``entries[*].class_hash`` so a mismatch NAMES the class. Each
    class hash folds that entry's ``range_digest`` (the #723 S3 requirement:
    three exports differing ONLY in declared range produced identical node-only
    digests) plus its coordinate and graph-interface block.

    CONTRACT-FACTS SHAPE CHANGE (v1 -> v2, pgw#758): this re-keys every
    published ``aot-inductor`` cell; single-graph format-1 cells are RETIRED —
    correct and expected under ck5 exact identity.

    CONTRACT-FACTS SHAPE CHANGE (v2 -> v3, pgw#817): ``shell_digest`` joins the
    facts and the ``mode`` axis stops being hardcoded ``""``. This re-keys
    every published ``aot-inductor`` cell again — correct and expected: a v2
    key does not bind the assembly around the graphs, so a v2 REGIONAL cell
    could collide across shells, and there is no way to add the binding
    without moving the key.
    """
    from . import env_seal

    sm = str(meta.get("sm") or "")
    if not sm:
        raise MintRefused(
            "cannot state the compute capability (sm) of this runtime; an "
            "exported cell has no identity without it — mint on the target GPU")
    entries = dict(meta.get("entries") or {})
    combined = str(meta.get("combined_graph_hash") or "")
    if not entries or not combined:
        raise MintRefused(
            "the envelope recorded no entries/combined_graph_hash; a "
            "multi-graph cell must not be keyed without its class set "
            "(pgw#716/#758)")
    unhashed = sorted(
        name for name, block in entries.items()
        if not str((block or {}).get("class_hash") or ""))
    if unhashed:
        raise MintRefused(
            f"entries {unhashed[:4]!r} carry no class_hash; a class the key "
            f"cannot name is a class a mismatch cannot name (pgw#716)")
    # pgw#812 S3.3 / pgw#817: the shell digest is a CONTRACT FACT, not an
    # annotation. `combined_graph_hash` describes the graphs the cell carries;
    # for a regional cell those are BLOCKS, so nothing else in the key covers
    # the assembly around them — a different num_layers, a different rope
    # construction or a diffusers minor that rewrites the outer forward would
    # produce the same key and serve different math. Mandatory rather than
    # optional: an absent digest is exactly the collision the axis exists to
    # prevent, and a key that is sometimes assembly-bound and sometimes not is
    # worse than either.
    mode = aot_regional.MODE_REGIONAL if spec.regional else ""
    shell = str(spec.shell_digest or meta.get("shell_digest") or "")
    if mode and not shell:
        raise MintRefused(
            "a regional cell must carry a shell_digest: its entries are "
            "BLOCKS, so the combined graph hash covers only the parts and two "
            "models with identical blocks and different shells would key "
            "identically while serving different math (pgw#812 S3.3)")
    contract_facts: Dict[str, Any] = {
        "v": 3,
        "combined_graph_hash": combined,
        "shell_digest": shell,
        "targets": sorted({
            str((block or {}).get("target") or "") for block in entries.values()}),
        "shapes": sorted([int(v) for v in row] for row in spec.shapes),
        "text_lens": sorted({int(v) for v in spec.text_lens}),
        "guidance": sorted(float(v) for v in spec.guidance_scales),
        "lora_bucket": int(spec.lora_bucket or 0),
        "strict": bool(spec.strict),
    }
    contract = cell_key.contract_digest(contract_facts)
    return cell_key.from_axes({
        "format": str(meta.get("format") or ""),
        "kind": aot_serve.ARTIFACT_KIND,
        "family": str(meta.get("family") or ""),
        "lane": spec.lane_label(),
        # pgw#817/D4. This used to hardcode "" under the comment "an exported
        # cell is always whole-graph: 'regional' is a dynamo partitioning
        # strategy with no export counterpart". pgw#812 falsified it by
        # measurement — an exported regional cell exists, is 1.37 MB, and
        # serves. The ck5 `mode` axis already existed and already fed the
        # digest; setting it is the minimal change that makes a regional cell
        # unconfusable with a whole-graph cell of the same family x lane x sm.
        "mode": mode,
        "sm": sm,
        "contract": contract,
        "env_seal": env_seal.seal_digest(dict(meta.get(env_seal.SEAL_KEY) or {})),
        "toolchain": cell_key.facts_digest(dict(meta.get("toolchain") or {})),
        "code_closure": cell_key.facts_digest(
            dict(meta.get("code_closure") or {})),
    })


def _state_dict_keys(module: Any) -> Tuple[str, ...]:
    """The names the SERVE arm will have to bind from — pgw#825.

    ``aot_serve.resident_constants``, never ``state_dict()``: the gate's whole
    value is that it predicts the arm, and the arm binds non-persistent
    buffers (the LoRA branch pair) that ``state_dict()`` does not report.
    """
    try:
        return tuple(str(k) for k in aot_serve.resident_constants(module))
    except Exception:
        return ()


def _write_literals(
    minted: Sequence[_MintedEntry], package: Path, content_dir: Path,
) -> None:
    """Pack the bytes of every entry's declared LITERAL constants beside the
    package, keys namespaced ``<entry>::<fqn>`` (pgw#758).

    A literal has no ``state_dict`` counterpart (folded scalars, sinusoidal
    tables, shape vectors), so a consumer cannot bind it from resident weights.
    Under B1 the ``.so`` does not carry it either — which is exactly the
    unbound-constant precondition for the worker-killing segfault. Shipping the
    bytes is therefore not an optimization; it is what makes a code-only
    artifact loadable at all.
    """
    tensors: Dict[str, Any] = {}
    missing: List[str] = []
    for row in minted:
        literals = aot_package.literal_constants(package, row.name)
        if not literals:
            continue
        values = dict(getattr(row.program, "constants", {}) or {})
        for constant in literals:
            tensor = values.get(constant.fqn) or values.get(constant.name)
            if tensor is None:
                missing.append(f"{row.name}{aot_serve.LITERAL_SEP}{constant.fqn}")
                continue
            tensors[f"{row.name}{aot_serve.LITERAL_SEP}{constant.fqn}"] = \
                tensor.detach().cpu().contiguous()
    if missing:
        raise MintRefused(
            f"{len(missing)} declared literal constant(s) have no value in "
            f"their exported program, e.g. {missing[:6]!r} — the cell could "
            f"never bind them and would segfault on first call (pgw#704 B1)")
    if not tensors:
        return
    from safetensors.torch import save_file

    save_file(tensors, str(Path(content_dir) / aot_serve.LITERALS_NAME))


def _input_names(
    module: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any],
) -> Tuple[str, ...]:
    """The target's forward parameter names, in call order.

    Positional args are named from the signature so the ``dynamic_shapes`` dict
    form can key on the same names whether a caller passed a tensor
    positionally or by keyword.
    """
    import inspect

    forward = getattr(module, "forward", module)
    try:
        params = [
            p.name for p in inspect.signature(forward).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                          p.KEYWORD_ONLY)
        ]
    except (TypeError, ValueError) as exc:
        raise MintRefused(
            f"cannot read the forward signature of {type(module).__name__}: "
            f"{exc}") from exc
    positional = params[:len(args)]
    keyword = [name for name in kwargs if name not in positional]
    return tuple(positional) + tuple(keyword)


def flat_input_names(
    module: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any],
) -> Tuple[str, ...]:
    """ONE name per EXPORTED user input — containers FLATTENED the way export
    flattens them.

    MEASURED (pgw#790 lane, real sdxl UNet, torch 2.13): `aot_package.
    input_contract` zips the caller-side parameter names against the exported
    program's user inputs positionally, and a container argument occupies ONE
    parameter slot but produces N placeholders. sdxl's `added_cond_kwargs`
    ({text_embeds, time_ids}) therefore shifted every later name by one and the
    recorded contract came out as

        position 7  name 'added_cond_kwargs'                shape [2, 1280]
        position 8  name 'down_block_additional_residuals'  shape [2, 6]

    i.e. text_embeds and time_ids wearing the names of the parameters that
    follow them. At serve time `bind_call_inputs` then binds the pipeline's
    `added_cond_kwargs` DICT to a declared tensor input (`input_not_tensor`)
    and cannot find `down_block_additional_residuals` at all
    (`input_missing`) — every request refuses by name and the armed cell
    serves eager for life. That is the field symptom `bind_call_inputs`'
    own docstring records from pod ae2uc81yub0gyq; the nested-lookup patch
    treated the symptom, but a nested lookup cannot help when the NAMES it
    searches for are the wrong ones.

    Mapping leaves take their BARE KEY, which is exactly what the serve-side
    nested resolution looks for, and dicts flatten in SORTED key order because
    that is what torch's pytree does. Sequence leaves take `<param>.<index>`.
    ``_input_names`` is deliberately left alone: `dynamic_shapes_spec` keys on
    top-level PARAMETER names and mirrors containers structurally.
    """
    names = _input_names(module, args, kwargs)
    values = list(args) + [kwargs[n] for n in names[len(args):] if n in kwargs]
    out: List[str] = []

    def walk(name: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                walk(str(key), value[key])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(f"{name}.{index}", item)
        else:
            out.append(str(name))

    for name, value in zip(names, values):
        walk(str(name), value)
    return tuple(out)


def _specialization_facts(spec: ExportSpec) -> Dict[str, Any]:
    """The frozen-branch declaration, with the lane facts always present.

    The lane and bucket belong here as well as in the key axes: the
    declaration is what a human reads off a rejected cell, and "which lane was
    this traced under" is the first question.
    """
    facts: Dict[str, Any] = dict(spec.specialization)
    facts.setdefault("weight_lane", str(spec.weight_lane or ""))
    facts.setdefault("lora_bucket", int(spec.lora_bucket or 0))
    facts.setdefault("strict", bool(spec.strict))
    return facts


def lifted_input_gaps(program: Any, spec: ExportSpec) -> List[str]:
    """Named reasons the declared lifted inputs are not actually graph inputs.

    #725 option 2's guarantee is structural: the adapter cannot be baked
    because it is an INPUT. If export did not lift it, the guarantee is gone
    and the G1 constant-table check would pass on absence — the "missing FQN
    means the branch was constant-folded, the same bug in a different hat"
    case. So the presence of every declared lifted input is proven here, on
    the program, before a single second of AOTI compile is spent.
    """
    if not spec.lifted_inputs:
        return []
    signature = getattr(program, "graph_signature", None)
    user_inputs = {str(n) for n in getattr(signature, "user_inputs", ()) or ()}
    gaps: List[str] = []
    for name in spec.lifted_inputs:
        if str(name) not in user_inputs:
            gaps.append(
                f"declared lifted input {name!r} is not a user input of the "
                f"exported program (inputs: {sorted(user_inputs)!r}) — the "
                f"adapter would not be swappable (#725 option 2)"
            )
    return gaps


class _CallableTarget:
    """Adapts a non-``forward`` target (``vae.decode``) to a module for export.

    ``torch.export`` traces ``forward``; a target like ``vae.decode`` is a bound
    method on a module whose ``forward`` does something else. Wrapping keeps the
    owner's parameters reachable (so they lift as constants normally) while
    presenting the declared callable as the traced entrypoint.

    **The wrapper COPIES the bound method's signature (ie#566 G1b).** A bare
    ``forward(*args, **kwargs)`` erases the parameter names, and
    :func:`_input_names` reads exactly those names to build the
    ``dynamic_shapes`` dict — so an erased signature silently yields zero
    bindable inputs and EVERY declared dynamic dim on a dotted target fails to
    bind. Signature preservation is the whole reason this class can carry a
    dynamic contract at all.
    """

    def __init__(self, owner: Any, attr: str) -> None:
        import inspect

        import torch.nn as nn

        self._owner = owner
        self._attr = str(attr)
        bound = getattr(owner, self._attr)

        class _Wrapper(nn.Module):
            def __init__(self, owner: Any, attr: str) -> None:
                super().__init__()
                self.owner = owner
                self._attr = attr

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return getattr(self.owner, self._attr)(*args, **kwargs)

        # Per-instance class, so stamping cannot leak across targets. ``self``
        # must be prepended: the stamp lands on the UNBOUND function, and
        # ``inspect.signature`` of the BOUND ``forward`` strips its first
        # parameter — without the filler it would strip the target's real first
        # argument and shift every name by one (silently mis-binding every
        # declared dynamic dim, which is the ie#566 G1b bug in a subtler form).
        try:
            sig = inspect.signature(bound)
            _Wrapper.forward.__signature__ = sig.replace(  # type: ignore[attr-defined]
                parameters=[inspect.Parameter(
                    "self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                + list(sig.parameters.values()))
        except (TypeError, ValueError) as exc:
            raise MintRefused(
                f"cannot read the signature of target {self._attr!r} on "
                f"{type(owner).__name__} ({exc}); without it no declared "
                f"dynamic dim could bind (ie#566 G1b)") from exc

        self._module = _Wrapper(owner, self._attr)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module(*args, **kwargs)


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


def publish(result: MintResult, publisher: Any) -> str:
    """Publish a minted cell through a ``fleet_cells.CellPublisher``.

    Receipts are the HUB's business: it adds them at publish-finalize (#709),
    so the producer's whole obligation is a keyed ``metadata.json`` inside the
    tar — which :func:`mint` has already stamped and proven. Refuses before the
    wire when the artifact carries no key, since an unaddressable cell would be
    stored under a flavor nothing can request.
    """
    key = result.cell_key
    if not key:
        raise MintRefused("cannot publish an artifact with no cell_key")
    family = str(result.metadata.get("family") or "")
    if not family:
        raise MintRefused("cannot publish an artifact with no family")
    # th#1355: the mint pod already measured this (timings["total_s"]), so the
    # cell's own cell_store row records what it cost to build instead of the
    # cost living only in an activity event that carries no cell key.
    mint_duration_ms = max(0, int(round(float(result.timings.get("total_s") or 0.0) * 1000)))
    return str(publisher.publish(
        family, result.artifact, dict(result.metadata), mint_duration_ms))


# ---------------------------------------------------------------------------
# CLI — python -m gen_worker.aot_mint
# ---------------------------------------------------------------------------


def _load_spec(path: Path) -> Tuple[ExportSpec, Dict[str, Any]]:
    """A cell-level :class:`ExportSpec` from a JSON mint request.

    The request is a file rather than a pile of flags because a mint request is
    a CONTRACT (lane, precision, provenance, frozen specialization) that wants
    review and version control, not 20 argv strings. It names a FAMILY, never
    a target/fork/class coordinate: the cell covers the whole declared class
    set (pgw#758), so a coordinate-shaped request is refused by name rather
    than silently minting a subset of the contract the key advertises.
    """
    body = json.loads(Path(path).read_text())
    subset_fields = sorted(
        k for k in ("target", "fork", "class_dims", "dynamic")
        if body.get(k))
    if subset_fields:
        raise MintRefused(
            f"mint request {path} names {subset_fields!r} — a multi-graph "
            f"cell covers the family's WHOLE declared class set (pgw#758); "
            f"coordinates and dynamic rows derive from the declaration, "
            f"never from the request")
    spec = ExportSpec(
        family=str(body.get("family") or ""),
        target="",
        weight_lane=str(body.get("weight_lane") or ""),
        precision=str(body.get("precision") or "bf16"),
        lora_bucket=int(body.get("lora_bucket") or 0),
        shapes=tuple(tuple(int(v) for v in row) for row in body.get("shapes") or ()),
        batch=int(body.get("batch") or 0),
        text_lens=tuple(int(v) for v in body.get("text_lens") or ()),
        guidance_scales=tuple(float(v) for v in body.get("guidance_scales") or ()),
        specialization=dict(body.get("specialization") or {}),
        lora_fqns=tuple(str(v) for v in body.get("lora_fqns") or ()),
        lifted_inputs=tuple(str(v) for v in body.get("lifted_inputs") or ()),
        strict=bool(body.get("strict", True)),
        source_ref=str(body.get("source_ref") or ""),
        source_digest=str(body.get("source_digest") or ""),
        closure_roots=tuple(str(v) for v in body.get("closure_roots") or ()),
    )
    if not spec.family:
        raise MintRefused(f"mint request {path} must name 'family'")
    return spec, body


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m gen_worker.aot_mint <request.json> --out <dir>`` — produce
    one multi-graph cell (ops/testing entry point; production mints run in a
    serving pod's background under the pgw#677 eager-first machinery — #724
    was REJECTED, there is no dedicated mint fleet).

    Exit 0 minted (and published when asked), 2 a named mint refusal, 3 a bad
    invocation. Inspect-only by default: ``--publish`` is opt-in so a mint can
    be produced and byte-compared before anything reaches the hub.
    """
    parser = argparse.ArgumentParser(
        prog="gen_worker.aot_mint",
        description="Export + AOTI-package a family's declared class set "
                    "as one multi-graph cell.")
    parser.add_argument("request", type=Path, help="mint request JSON")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for the packed artifact")
    parser.add_argument("--model", type=str, default="",
                        help="model path/ref to compose (default: the "
                             "request's source_ref)")
    parser.add_argument("--allow-regressed-lanes", action="store_true",
                        help="mint a lane #730 holds on dynamo (plain / "
                             "fp8) — deliberate override")
    parser.add_argument("--publish", action="store_true",
                        help="publish through the fleet CellPublisher")
    parser.add_argument("--require-toolchain", action="store_true",
                        default=True,
                        help="refuse without a C toolchain (default on)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        spec, body = _load_spec(args.request)
    except Exception as exc:
        print(f"BAD REQUEST {args.request}: {exc}", file=sys.stderr)
        return 3
    try:
        # pgw#739/#758: load the endpoint's declaration module (registers the
        # family's export contract). The cell's class set, coordinates, and
        # dynamic contracts all derive from it — the request only ever names
        # the family and the lane facts.
        from . import aot_declaration

        aot_declaration.load_declaration(body, request_path=args.request)
    except MintRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.require_toolchain and not toolchain_present():
        print(
            "REFUSED: no C toolchain (cc/gcc) — an AOTI mint needs the "
            "compile-job image, not a prod worker image", file=sys.stderr)
        return 2

    model = args.model or spec.source_ref
    if not model:
        print("BAD REQUEST: no --model and the request has no source_ref",
              file=sys.stderr)
        return 3
    try:
        pipeline, _build_inputs = compose_for_mint(model, spec, body)
        result = mint(
            pipeline, spec, Path(args.out),
            allow_regressed_lanes=args.allow_regressed_lanes,
        )
    except MintRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "artifact": str(result.artifact),
        "cell_key": result.cell_key,
        "entries": sorted((result.metadata.get("entries") or {})),
        "timings": result.timings,
    }, indent=1))

    if args.publish:
        try:
            checkpoint = publish(result, _publisher_from_settings())
        except MintRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(f"published checkpoint {checkpoint}")
    return 0


def compose_for_mint(
    model: str, spec: ExportSpec, request: Mapping[str, Any],
) -> Tuple[Any, Callable[[Any], Tuple[Tuple[Any, ...], Mapping[str, Any]]]]:
    """Compose the pipeline for a mint and return its example-input builder.

    Delegates to the family's own registered input contract rather than growing
    a per-family ``if`` ladder here: each family already owns how its denoiser
    is called (SDXL's ``added_cond_kwargs``, z-image's ragged lists, #729), and
    that knowledge must live with the family, not in the mint driver.
    """
    from . import aot_inputs

    return aot_inputs.compose(model, spec, request)


def _publisher_from_settings() -> Any:
    """A ``fleet_cells.CellPublisher`` for a mint pod.

    A serving worker builds its publisher from the HelloAck ``file_base_url``
    and its rotating JWT (``executor._cell_publisher``). A mint pod has no
    orchestrator session, so the hub base comes from the injected tensorhub URL
    — the repo-commit API lives on the same combined-binary host — and the JWT
    from settings. Refuses by name when either is absent rather than attempting
    an unauthenticated publish.
    """
    from .config import get_settings
    from .fleet_cells import CellPublisher

    settings = get_settings()
    base_url = str(
        getattr(settings, "tensorhub_public_url", "")
        or getattr(settings, "tensorhub_url", "") or "").strip()
    token = str(getattr(settings, "worker_jwt", "")
                or getattr(settings, "tensorhub_token", "") or "").strip()
    if not base_url or not token:
        raise MintRefused(
            "cannot publish: TENSORHUB_PUBLIC_URL/TENSORHUB_URL and "
            "WORKER_JWT/TENSORHUB_TOKEN must both be set on a mint pod (the "
            "artifact was produced and is on disk)")
    publisher = CellPublisher(
        base_url=base_url,
        worker_jwt=lambda: token,
        image_digest=str(getattr(settings, "worker_image_digest", "") or ""),
    )
    if not publisher.enabled():
        raise MintRefused("fleet CellPublisher is not enabled")
    return publisher


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CODE_ONLY_CONFIGS",
    "DynamicDim",
    "ExportSpec",
    "MintRefused",
    "MINT_COMPILE_THREADS",
    "MintResult",
    "PARITY_LANES",
    "REGRESSED_LANES",
    "autotune_posture",
    "cell_identity",
    "compile_entry_files",
    "compose_for_mint",
    "declaration_module_gaps",
    "declared_range_gaps",
    "dynamic_shapes_spec",
    "emit_phase_events",
    "entry_graph_block",
    "export_program",
    "shared_identity_blocks",
    "LIFTED_LORA_TORCH_FLOOR",
    "lane_admitted",
    "lifted_input_gaps",
    "lifted_torch_gap",
    "main",
    "mint",
    "package_cell",
    "publish",
]
