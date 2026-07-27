"""The AOT mint — export + AOTInductor-package a compile target as a cell
(pgw#704 GO, #723; the produce half of the AOT migration).

    compose -> torch.export.export -> aoti_compile_and_package(code-only)
            -> gates -> pack -> publish

``aot_serve`` owns the ENVELOPE — metadata contract, ``pack``, ``verify`` (#721
S1 / #723 S1: ONE source of truth, imported by both lanes, never re-declared)
— and consumes the result. ``aot_package`` reads facts back out of a compiled
``.pt2`` and holds the B1 gate. ``lora_lifted`` owns the no-baked-adapter gate.
This module drives PRODUCTION and nothing else. Deliberately NOT folded into
``compile_cache``:
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

Why minting is a JOB
--------------------
An AOTI mint costs roughly double a dynamo mint (an export pass plus the AOTI
compile), and pgw#677 already proved a serving pod that spends minutes
compiling is unacceptable. So this module is invoked as
``python -m gen_worker.aot_mint`` on a pod designated for minting; serving pods
never AOT-compile (#724 owns the fleet-side invariant).

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
    3. **cross-input collapse** — a STATIC dim on ANOTHER input whose value is a
       multiple of the product of the declared dims' trace-time values. Such a
       dim is an algebraic function of the "dynamic" extents, so it silently
       fixes them even though each symbol still reports a healthy range. This is
       the check the presence-only gate lacked.
    """
    gaps: List[str] = []
    shapes = _placeholder_shapes(program)
    ranges = getattr(program, "range_constraints", {}) or {}
    env = _shape_env(program)
    hints = dict(getattr(env, "var_to_val", {}) or {}) if env is not None else {}

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

    gaps.extend(_cross_input_collapse(shapes, dims, declared_symbols, hints))
    return gaps


def _cross_input_collapse(
    shapes: Mapping[str, Tuple[Any, ...]],
    dims: Sequence[DynamicDim],
    declared_symbols: Sequence[Any],
    hints: Mapping[Any, Any],
) -> List[str]:
    """Static dims on OTHER inputs that pin the declared dynamic extents.

    ie#566 G3, measured on wan ti2v-5b: H and W were declared symbolic and each
    kept a healthy solved range, but a per-token input of static length 27,280
    is an algebraic function of H*W — so the graph serves exactly the traced
    shape regardless. Every per-symbol check passes; only comparing the static
    dims against the PRODUCT of the trace-time extents finds it.

    Fail-closed by design: the remedy is to declare that input's dim dynamic too
    (or make it independent of H*W), and B2 doctrine forbids shipping the silent
    version while we decide.
    """
    values: List[int] = []
    for sym in declared_symbols:
        try:
            values.append(int(hints[sym]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(values) < 2:
        return []
    product = 1
    for v in values:
        product *= v
    if product <= 1:
        return []
    declared_inputs = {d.input_name for d in dims}
    out: List[str] = []
    for name, shape in sorted(shapes.items()):
        if name in declared_inputs:
            continue
        for axis, dim in enumerate(shape):
            text = str(dim)
            if not text.lstrip("-").isdigit():
                continue
            n = int(text)
            if n >= product and n % product == 0:
                out.append(
                    f"{name}[{axis}] is STATIC {n}, a multiple of {product} — "
                    f"the product of the declared dynamic extents "
                    f"{values!r}. That dim is an algebraic function of the "
                    f"'dynamic' axes, so the graph is pinned to the traced "
                    f"shape even though each symbol reports a range "
                    f"(ie#566 G3). Declare {name}[{axis}] dynamic too, or make "
                    f"it independent of the latent extent")
    return out


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def compile_package(
    program: Any,
    package_path: Path,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> Path:
    """AOTI-compile an exported program into a CODE-ONLY ``.pt2``.

    ``CODE_ONLY_CONFIGS`` is applied LAST so no caller-supplied config can
    re-enable constant baking. That ordering is the point: B1 is a fleet
    correctness requirement, not a default a caller may override.
    """
    import torch

    package_path = Path(package_path)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    configs: Dict[str, Any] = dict(inductor_configs or {})
    overridden = sorted(set(configs) & set(CODE_ONLY_CONFIGS))
    if overridden:
        logger.warning(
            "aot-mint: ignoring caller inductor config %s — code-only is B1, "
            "not a knob", overridden)
    configs.update(CODE_ONLY_CONFIGS)
    try:
        out = torch._inductor.aoti_compile_and_package(
            program, package_path=str(package_path), inductor_configs=configs,
        )
    except Exception as exc:
        raise MintRefused(
            f"aoti_compile_and_package failed: {type(exc).__name__}: {exc}"
        ) from exc
    return Path(out)


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


def mint(
    module: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    example_inputs: Callable[[], Tuple[Tuple[Any, ...], Mapping[str, Any]]],
    allow_regressed_lanes: bool = False,
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> MintResult:
    """Export, compile code-only, gate, and pack one exported cell.

    Does NOT publish — :func:`publish` is a separate step so a mint can be
    inspected, byte-compared (#699 double-mint), or produced on a box with no
    hub credentials.
    """
    refusal = lane_admitted(spec, allow_regressed_lanes=allow_regressed_lanes)
    if refusal:
        raise MintRefused(refusal)

    out_dir = Path(out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    args, kwargs = example_inputs()
    input_names = _input_names(module, args, kwargs)
    dynamic = dynamic_shapes_spec(spec.dynamic, input_names) \
        if spec.dynamic else None

    t0 = time.monotonic()
    program = export_program(
        module, args, kwargs, dynamic_shapes=dynamic, strict=spec.strict)
    timings["export_s"] = round(time.monotonic() - t0, 2)

    gaps = declared_range_gaps(program, spec.dynamic)
    if gaps:
        raise MintRefused("declared-range gate: " + "; ".join(gaps))
    lifted_gaps = lifted_input_gaps(program, spec)
    if lifted_gaps:
        raise MintRefused("lifted-input gate: " + "; ".join(lifted_gaps))

    # pgw#725 G3, on the EXPORTEDPROGRAM and before any packing: the adapter
    # must be absent from the constant table AND present among the user inputs.
    # A missing pair is the same defect as a baked one (the branch was traced
    # away, so every request silently gets the base model), and packing renames
    # a plain-__dict__ adapter to _tensor_constant0 — which makes the
    # package-side scan a false PASS. Free here, unsound there.
    if spec.lora_bucket or spec.lifted_inputs or spec.lora_fqns:
        from .api.errors import ValidationError
        from .models import lora_lifted

        try:
            lora_lifted.assert_no_baked_adapter(
                program, label=f"{spec.family}/{spec.target}")
        except ValidationError as exc:
            raise MintRefused(f"no-baked-adapter gate (#725 G3): {exc}") from exc

    t0 = time.monotonic()
    package = compile_package(
        program, work / aot_serve.PACKAGE_NAME,
        inductor_configs=inductor_configs)
    timings["aoti_compile_s"] = round(time.monotonic() - t0, 2)

    violations = aot_package.code_only_violations(package)
    if violations:
        raise MintRefused("code-only gate (pgw#704 B1): " + "; ".join(violations))
    unbindable = aot_package.unbindable_constants(
        package, _state_dict_keys(module))
    if unbindable:
        raise MintRefused("bindability gate: " + "; ".join(unbindable))
    # pgw#728: strict and non-strict traces lift DIFFERENT constant sets, so the
    # manifest must be proven to describe the package that ships beside it. Two
    # independent derivations (program vs generated wrapper) required to agree —
    # drift the env seal cannot see, because both modes run identically sealed.
    drift = aot_package.program_package_drift(program, package)
    if drift:
        raise MintRefused("constant-set drift: " + "; ".join(drift))
    fused = aot_package.eliminated_constants(program, package)
    if fused:
        # Routine compiler fusion (measured on real sdxl: conv_out.bias folded
        # into the conv epilogue). Recorded, never fatal — but a surprising jump
        # in the count should be visible rather than silently discarded.
        logger.info("aot-mint: %d lifted constant(s) fused away by the compiler "
                    "(e.g. %s)", len(fused), fused[:3])

    t0 = time.monotonic()
    try:
        inputs, symbols = aot_package.input_contract(program, input_names)
        constants = aot_package.constants_manifest(package)
    except aot_package.PackageIntrospectionError as exc:
        raise MintRefused(f"declaration: {exc}") from exc
    _write_literals(program, package, work)

    identity = identity_blocks(program, package, spec)
    try:
        meta = aot_serve.artifact_metadata(
            family=spec.family,
            module=spec.target,
            precision=spec.precision,
            cell_key="",
            inputs=inputs,
            symbols=symbols,
            constants=constants,
            source_ref=spec.source_ref,
            source_digest=spec.source_digest,
        )
    except ValueError as exc:
        # The envelope validates the contract it is handed. A malformed one must
        # fail HERE, on the mint pod, not at serve time on a paying request.
        raise MintRefused(
            f"envelope refused the declared contract: {exc}") from exc
    meta.update(identity)
    mode_drift = aot_package.strict_mode_drift(meta, spec.strict)
    if mode_drift:
        raise MintRefused("trace-mode drift: " + "; ".join(mode_drift))
    meta["cell_key"] = key = cell_identity(meta, spec).digest

    artifact = aot_serve.pack(work, out_dir / f"{key}.tar.gz", meta)
    timings["pack_s"] = round(time.monotonic() - t0, 2)

    literals = sum(
        1 for row in constants if row["source"] == aot_serve.SOURCE_LITERAL)
    logger.info(
        "aot-mint: %s target=%s lane=%s -> %s (%.1f MB package, %d declared "
        "constants incl. %d literal, %d symbol(s), %s)",
        spec.family, spec.target, spec.lane_label() or "(plain)", key,
        package.stat().st_size / 1e6, len(constants), literals, len(symbols),
        timings,
    )
    return MintResult(artifact=artifact, metadata=meta, timings=timings)


def identity_blocks(
    program: Any, package: Path, spec: ExportSpec,
) -> Dict[str, Any]:
    """The ck5 identity facts an exported cell must record.

    ``aot_serve.artifact_metadata`` takes ``cell_key`` as a STRING, so the
    envelope on its own would carry a stamp WITHOUT the axes the stamp
    summarizes — and ``cell_key``'s standing discipline is that a key is always
    recomputed FROM recorded facts, so a stamp can never disagree with them.
    These blocks are what make that recomputation possible for the new kind, and
    they ride the metadata additively (the envelope's parsers read named fields
    and are unaffected).

    ``graph`` carries what IS the exported graph's interface: the declared
    constant FQN set, the lifted inputs, the pytree spec, and the python
    branches export FROZE at trace time. Constant BYTE SIZES are deliberately
    absent — they are a property of the resident weights, and a fine-tune of one
    family must keep sharing cells, which is the premise of family-scoped cells.
    """
    from . import compile_cache as cc
    from . import env_seal

    return {
        "graph": {
            "v": 1,
            "constant_fqns": sorted(aot_package.constant_names(package)),
            "fused_constants": sorted(
                aot_package.eliminated_constants(program, package)),
            "lifted_inputs": sorted(str(n) for n in spec.lifted_inputs),
            "pytree": _pytree_facts(program),
            "specialization": _specialization_facts(spec),
        },
        "weight_lane": str(spec.weight_lane or ""),
        "lora_bucket": int(spec.lora_bucket or 0),
        "strict_export": bool(spec.strict),
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
    """The cell key an exported artifact's OWN recorded facts describe.

    Computed from the recorded blocks, never from separate probes, so the stamp
    can never disagree with the axes it summarizes — the discipline
    ``cell_key.from_artifact_metadata`` enforces for dynamo cells, mirrored for
    the new kind. ``cell_key.from_axes`` already accepts any ``kind`` VALUE (it
    validates axis NAMES), so the new kind needs no KEY_SCHEME bump: the axis
    set is unchanged and ``kind`` does the discriminating, which is what it is
    for. That also means no dynamo cell is stranded by this lane existing.

    The ``contract`` axis folds THREE things: the declared shape set, the
    envelope's ``range_digest``, and the graph identity block. The range digest
    is the #716/#723 S3 requirement — pgw#704 measured that three exports
    differing ONLY in declared range produce the identical node-only digest, so
    without it two artifacts admitting different traffic collide, and B2 means
    nothing at runtime would refuse the difference.
    """
    from . import env_seal

    sm = str(meta.get("sm") or "")
    if not sm:
        raise MintRefused(
            "cannot state the compute capability (sm) of this runtime; an "
            "exported cell has no identity without it — mint on the target GPU")
    range_digest = str(meta.get("range_digest") or "")
    if not range_digest:
        raise MintRefused(
            "the envelope recorded no range_digest; an exported cell must not "
            "be keyed without its admissible-shape range (#723 S3)")
    contract = cell_key.contract_digest({
        "v": 1,
        "shapes": sorted([int(v) for v in row] for row in spec.shapes),
        "targets": [spec.target],
        "text_lens": sorted({int(v) for v in spec.text_lens}),
        "guidance": sorted(float(v) for v in spec.guidance_scales),
        "lora_bucket": int(spec.lora_bucket or 0),
        "range_digest": range_digest,
        "graph": dict(meta.get("graph") or {}),
        "strict": bool(spec.strict),
    })
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


def _write_literals(program: Any, package: Path, content_dir: Path) -> None:
    """Pack the bytes of every declared LITERAL constant beside the package.

    A literal has no ``state_dict`` counterpart (folded scalars, sinusoidal
    tables, shape vectors), so a consumer cannot bind it from resident weights.
    Under B1 the ``.so`` does not carry it either — which is exactly the
    unbound-constant precondition for the worker-killing segfault. Shipping the
    bytes is therefore not an optimization; it is what makes a code-only
    artifact loadable at all.
    """
    literals = aot_package.literal_constants(package)
    if not literals:
        return
    values = dict(getattr(program, "constants", {}) or {})
    tensors: Dict[str, Any] = {}
    missing: List[str] = []
    for constant in literals:
        tensor = values.get(constant.fqn) or values.get(constant.name)
        if tensor is None:
            missing.append(constant.fqn)
            continue
        tensors[constant.fqn] = tensor.detach().cpu().contiguous()
    if missing:
        raise MintRefused(
            f"{len(missing)} declared literal constant(s) have no value in the "
            f"exported program, e.g. {missing[:6]!r} — the cell could never "
            f"bind them and would segfault on first call (pgw#704 B1)")
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


def mint_target(
    pipeline: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    example_inputs: Callable[
        [Any], Tuple[Tuple[Any, ...], Mapping[str, Any]]],
    allow_regressed_lanes: bool = False,
) -> MintResult:
    """Resolve ``spec.target`` on a composed pipeline and mint it.

    Target resolution reuses ``compile_cache._resolve_target`` so the exported
    lane and the dynamo lane can never disagree about what ``"unet"`` or
    ``"vae.decode"`` names.
    """
    resolved = _resolve_target(pipeline, spec.target)
    if resolved is None:
        raise MintRefused(
            f"pipeline {type(pipeline).__name__} has no compile target "
            f"{spec.target!r}")
    owner, attr, _fn = resolved
    module = owner if attr == "forward" else _CallableTarget(owner, attr)
    return mint(
        module, spec, out_dir,
        example_inputs=lambda: example_inputs(owner),
        allow_regressed_lanes=allow_regressed_lanes,
    )


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
    """An :class:`ExportSpec` from a JSON mint request.

    The request is a file rather than a pile of flags because a mint request is
    a CONTRACT (shapes, dims, frozen specialization) that wants review and
    version control, not 20 argv strings.
    """
    body = json.loads(Path(path).read_text())
    dims = tuple(
        DynamicDim(
            input_name=str(row["input"]), axis=int(row["axis"]),
            min=int(row["min"]), max=int(row["max"]),
            multiple_of=int(row.get("multiple_of") or 1),
        )
        for row in body.get("dynamic") or ()
    )
    spec = ExportSpec(
        family=str(body.get("family") or ""),
        target=str(body.get("target") or ""),
        weight_lane=str(body.get("weight_lane") or ""),
        precision=str(body.get("precision") or "bf16"),
        lora_bucket=int(body.get("lora_bucket") or 0),
        shapes=tuple(tuple(int(v) for v in row) for row in body.get("shapes") or ()),
        batch=int(body.get("batch") or 0),
        text_lens=tuple(int(v) for v in body.get("text_lens") or ()),
        guidance_scales=tuple(float(v) for v in body.get("guidance_scales") or ()),
        dynamic=dims,
        specialization=dict(body.get("specialization") or {}),
        lora_fqns=tuple(str(v) for v in body.get("lora_fqns") or ()),
        lifted_inputs=tuple(str(v) for v in body.get("lifted_inputs") or ()),
        strict=bool(body.get("strict", True)),
        source_ref=str(body.get("source_ref") or ""),
        source_digest=str(body.get("source_digest") or ""),
        closure_roots=tuple(str(v) for v in body.get("closure_roots") or ()),
    )
    if not spec.family or not spec.target:
        raise MintRefused(
            f"mint request {path} must name both 'family' and 'target'")
    return spec, body


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m gen_worker.aot_mint <request.json> --out <dir>`` — produce
    one exported cell on a mint pod.

    Exit 0 minted (and published when asked), 2 a named mint refusal, 3 a bad
    invocation. Inspect-only by default: ``--publish`` is opt-in so a mint can
    be produced and byte-compared before anything reaches the hub.
    """
    parser = argparse.ArgumentParser(
        prog="gen_worker.aot_mint",
        description="Export + AOTI-package a compile target as a cell.")
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
        pipeline, build_inputs = compose_for_mint(model, spec, body)
        result = mint_target(
            pipeline, spec, Path(args.out),
            example_inputs=build_inputs,
            allow_regressed_lanes=args.allow_regressed_lanes,
        )
    except MintRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "artifact": str(result.artifact),
        "cell_key": result.cell_key,
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
    "MintResult",
    "PARITY_LANES",
    "REGRESSED_LANES",
    "cell_identity",
    "compile_package",
    "compose_for_mint",
    "declared_range_gaps",
    "dynamic_shapes_spec",
    "export_program",
    "identity_blocks",
    "lane_admitted",
    "lifted_input_gaps",
    "main",
    "mint",
    "mint_target",
    "publish",
]
