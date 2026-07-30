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

from . import aot_package, aot_serve, cell_key
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
    """


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
    """

    input_name: str
    axis: int
    min: int
    max: int
    multiple_of: int = 1

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
    """
    from torch.export import Dim

    by_input: Dict[str, Dict[int, Any]] = {}
    for d in dims:
        if d.min < 1 or d.max < d.min:
            raise MintRefused(
                f"declared dim {d.input_name}[{d.axis}] has an empty range "
                f"[{d.min}, {d.max}]")
        if d.multiple_of > 1:
            if d.min % d.multiple_of or d.max % d.multiple_of:
                raise MintRefused(
                    f"declared dim {d.input_name}[{d.axis}] bounds "
                    f"[{d.min}, {d.max}] are not multiples of "
                    f"{d.multiple_of}; export cannot express that guard")
            base = Dim(
                f"{d.input_name}_{d.axis}_u",
                min=d.min // d.multiple_of, max=d.max // d.multiple_of)
            by_input.setdefault(d.input_name, {})[d.axis] = d.multiple_of * base
        else:
            by_input.setdefault(d.input_name, {})[d.axis] = Dim(
                f"{d.input_name}_{d.axis}", min=d.min, max=d.max)
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
    """
    from torch._inductor import aot_compile

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
    files: List[Any]
    timings: Dict[str, Any]


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
    specialization.setdefault("shape_strategy", str(decl.shape_strategy or ""))
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
) -> _MintedEntry:
    """Resolve, feed, (warm,) export, gate, and compile ONE declared graph
    class. Every refusal is prefixed with the entry name — a multi-graph
    mint that cannot say WHICH class failed is the silent-failure path in
    a new hat (pgw#758)."""
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
        input_names=input_names, files=files, timings=timings)


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


def _rearm_branches(pipeline: Any, bucket: int) -> None:
    """Restore canonical placement + the lifted signature after the
    branchless exports. The mint process may go on to serve or re-mint, and
    a pipeline left branchless would silently be a different graph family."""
    if not bucket:
        return
    from .models import lora_lifted, w8a8_lora

    w8a8_lora.enable_branch_lanes(pipeline, int(bucket))
    lora_lifted.install_lifted_lora_lanes(pipeline, int(bucket))


def mint(
    pipeline: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    allow_regressed_lanes: bool = False,
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> MintResult:
    """Export + compile EVERY declared graph class and pack them as ONE
    multi-graph cell (pgw#758).

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
    timings: Dict[str, float] = {}
    t_mint = time.monotonic()

    from . import aot_declaration as _decl  # deferred: aot_declaration imports us

    rows = adapter_arm_plans(_decl.cell_plans(decl), pipeline, spec)
    minted: List[_MintedEntry] = []
    disarmed = False
    try:
        for plan, arm in rows:
            if arm is False and not disarmed:
                # ONE toggle for the whole branchless group (the rows are
                # ordered adapter-bearing first): disable/enable reallocates
                # every leaf's branch container, and doing it per entry would
                # be N times the VRAM churn for the same graphs.
                _disarm_branches(pipeline)
                disarmed = True
            minted.append(_export_entry(
                pipeline, spec, plan, decl, inductor_configs=inductor_configs))
    finally:
        if disarmed:
            _rearm_branches(pipeline, int(spec.lora_bucket or 0))

    t0 = time.monotonic()
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
    phase_table = _mint_phase_table(minted, timings, inductor_configs)
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
    try:
        inputs, symbols = aot_package.input_contract(
            row.program, row.input_names)
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
) -> Dict[str, Any]:
    """The per-mint phase table (#757's instrument-first deliverable): one
    readable record of where the mint's seconds went, per entry and in
    total, plus the two facts an AOT-vs-JIT comparison needs to be fair —
    the graph-class COUNT this mint compiled and the autotune posture."""
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
        lane = spec.lane_label() or "plain"
        total_s = float(totals.get("total_s") or 0.0)
        activity_mod.emit_event(
            MINT_PHASES_KIND,
            f"family={spec.family} lane={lane} "
            f"n_entries={table.get('n_entries')} totals={totals} "
            f"phases={dict(table.get('phases') or {})} "
            f"autotune={dict(table.get('autotune') or {})}",
            phase=activity_mod.PHASE_MINTED,
            duration_ms=int(round(total_s * 1000)),
        )
        for name, timings in sorted((table.get("entries") or {}).items()):
            if not isinstance(timings, Mapping):
                continue
            entry_s = _entry_duration_s(timings)
            if entry_s <= 0:
                continue
            activity_mod.emit_event(
                MINT_PHASES_KIND,
                f"family={spec.family} lane={lane} entry={name} "
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
    contract_facts: Dict[str, Any] = {
        "v": 2,
        "combined_graph_hash": combined,
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
        # An exported cell is always whole-graph: "regional" is a dynamo
        # partitioning strategy with no export counterpart.
        "mode": "",
        "sm": sm,
        "contract": contract,
        "env_seal": env_seal.seal_digest(dict(meta.get(env_seal.SEAL_KEY) or {})),
        "toolchain": cell_key.facts_digest(dict(meta.get("toolchain") or {})),
        "code_closure": cell_key.facts_digest(
            dict(meta.get("code_closure") or {})),
    })


def _state_dict_keys(module: Any) -> Tuple[str, ...]:
    try:
        return tuple(str(k) for k in module.state_dict().keys())
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
    return str(publisher.publish(family, result.artifact, dict(result.metadata)))


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
    "declared_range_gaps",
    "dynamic_shapes_spec",
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
