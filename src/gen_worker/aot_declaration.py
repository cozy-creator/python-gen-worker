"""Derivation over the pgw#739 export-declaration vocabulary.

:mod:`gen_worker.api.export_contract` defines the vocabulary and holds the
registry; the endpoint writes the declaration; THIS module derives everything
the mint needs from it — mint plans, dynamic-shape rows, fork assertions, and
generic example inputs. No family name appears here (the #740 pattern:
vocabulary / registration / engine, family-free all three).

The one derivation rule
-----------------------
**Bounds derive from the declared class rows.** A dim's admissible range is
the hull of its resolved values over the rows an artifact serves, rounded OUT
to its ``multiple_of`` (rounding in would exclude a declared coordinate from
the artifact's own contract, and pgw#704 B2 measured that nothing at runtime
would refuse the difference). This is uniform across plain and RELATIONAL
dims: a relational dim (``relates_to``) gets a free ranged
``torch.export.Dim`` from the same hull and the solver unifies the relation
into the shape env (ie#566 §5, measured on wan ti2v: ``31*s25*s56`` carried
with 458 asserts, no free symbol surviving). The endpoint never hand-computes
a bound — that hand-math is exactly what this module exists to prevent.

Shape strategy (#730 ratified, per-family declared):

- ``static-rows`` — one mint plan per class row, no dynamic dims (symbolic
  latent H/W costs conv families the channels-last layout opt, +7.2%);
- ``dynamic-collapse`` — one mint plan per FORK coordinate; dims that vary
  across the coordinate's rows become declared-dynamic over their hull.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .aot_contract import ADAPTER_FORK, DynamicDim, ExportSpec, MintRefused
from .api.decorators import Compile
from .api.export_contract import (
    DYNAMIC_COLLAPSE,
    STATIC_ROWS,
    Arg,
    Dim,
    Fork,
    GraphClass,
    Input,
)
import inspect
from .aot_inputs import lifted_lora_values, module_dtype_device

logger = logging.getLogger(__name__)

_MISSING = object()


@dataclass(frozen=True)
class MintPlan:
    """One artifact the declaration says to mint: a fork coordinate, the
    class rows the artifact serves, the row that seeds the trace, and the
    derived declared-dynamic rows for one target."""

    family: str
    target: str
    fork: Tuple[Tuple[str, Any], ...]
    rows: Tuple[GraphClass, ...]
    seed: GraphClass
    dynamic: Tuple[DynamicDim, ...]


def target_inputs(decl: Compile, target: str) -> Tuple[Input, ...]:
    return tuple(i for i in decl.inputs if not i.targets or target in i.targets)


def target_args(decl: Compile, target: str) -> Tuple[Arg, ...]:
    return tuple(a for a in decl.args if not a.targets or target in a.targets)


def target_forks(decl: Compile, target: str) -> Tuple[Fork, ...]:
    return tuple(f for f in decl.forks if not f.targets or target in f.targets)


def arg_carriers(decl: Compile, target: str) -> frozenset:
    """Names of TEMPLATED args a dim may be carried by (pgw#853).

    An extent that enters the traced call as a python int inside a container
    — qwen-image's ``img_shapes[b][0] = (frames, H_pat, W_pat)`` — is a real
    binding and the declaration should say so. But it is NOT a tensor axis:
    a python int SPECIALIZES the graph, so it can never become a
    ``torch.export.Dim``. Every derivation below keeps that distinction.
    """
    return frozenset(
        a.name for a in target_args(decl, target) if a.template is not None)


def _target_input_names(decl: Compile, target: str) -> Optional[frozenset]:
    """Top-level input names declared for a target; None when the declaration
    carries no input templates (bindings then pass through unfiltered and
    ``aot_mint.dynamic_shapes_spec`` re-validates against the real forward)."""
    rows = target_inputs(decl, target)
    if not rows:
        return None
    return frozenset(i.top_name for i in rows)


def _round_out(lo: int, hi: int, multiple_of: int) -> Tuple[int, int]:
    m = max(1, int(multiple_of))
    lo_r = (lo // m) * m
    hi_r = -(-hi // m) * m
    return (max(lo_r, m), hi_r)


def dim_hull(rows: Sequence[GraphClass], dim: Dim) -> Tuple[int, int]:
    """The (min, max) of one dim's resolved values over class rows, rounded
    OUT to the dim's declared multiple."""
    values = [row.dim_map[dim.name] for row in rows]
    return _round_out(min(values), max(values), dim.multiple_of)


def derived_dynamic(
    decl: Compile, target: str, rows: Sequence[GraphClass],
) -> Tuple[DynamicDim, ...]:
    """The declared-dynamic rows one collapsed artifact needs, derived from
    the class rows — the ``Compile`` -> :class:`aot_mint.DynamicDim` bridge
    for the class vocabulary."""
    names = _target_input_names(decl, target)
    by_arg = arg_carriers(decl, target)
    out: List[DynamicDim] = []
    for dim in decl.dims:
        values = {row.dim_map[dim.name] for row in rows}
        if len(values) <= 1:
            continue  # static across this artifact's rows
        # pgw#853: an extent carried ONLY by a templated Arg is a python int
        # in the traced call. It specializes, so collapsing rows that differ
        # on it would mint ONE artifact that silently serves only the seed
        # row — the same class of silent-wrongness as the 0/1 refusal below,
        # and worth refusing for the same reason.
        if all(b.split(".", 1)[0] in by_arg or b in by_arg
               for b, _ in dim.carried_by):
            raise MintRefused(
                f"dim {dim.name!r} varies across the collapsed rows "
                f"({sorted(values)!r}) but is carried only by templated "
                f"arg(s) {sorted({b for b, _ in dim.carried_by})!r} — a "
                f"python-int argument SPECIALIZES the traced graph, so these "
                f"rows cannot share one artifact. Declare this family "
                f"{STATIC_ROWS!r}, or move the extent onto a tensor axis")
        lo, hi = dim_hull(rows, dim)
        if lo < 2:
            raise MintRefused(
                f"dim {dim.name!r} spans [{lo}, {hi}] across the collapsed "
                f"rows; torch's 0/1 specialization is not overridable "
                f"(ie#543), so an axis that reaches 1 must FORK the class "
                f"instead of collapsing")
        for bound, axis in dim.carried_by:
            if bound in by_arg or bound.split(".", 1)[0] in by_arg:
                continue  # a python int is never a torch symbol
            if names is not None and bound.split(".", 1)[0] not in names \
                    and bound not in names:
                continue
            out.append(DynamicDim(
                input_name=bound, axis=int(axis), min=lo, max=hi,
                multiple_of=dim.multiple_of, dim=str(dim.name)))
    return tuple(out)


# (pgw#1030: `named_dynamic_rows` deleted — zero callers anywhere; the
# class-bearing path derives ranges via `dynamic_rows_for_plan` and the
# class-less path never grew a consumer.)


def effective_shape_strategy(decl: Compile) -> str:
    """The declared per-family shape strategy (#730 ratified).

    One population per family since pgw#846 retired regional cells: the
    whole-graph ``shape_strategy`` is the only strategy there is.
    """
    return str(decl.shape_strategy or "")


def target_classes(decl: Compile, target: str) -> Tuple[GraphClass, ...]:
    """The graph classes declared for one target (pgw#967).

    An unscoped row serves every target — the pre-scoping rule — so a
    single-target family reads exactly as it always did.
    """
    return tuple(cls for cls in decl.classes if cls.serves(target))


def mint_plans(decl: Compile, target: str) -> Tuple[MintPlan, ...]:
    """Every artifact the declaration says to mint for one target."""
    if not decl.classes:
        raise MintRefused(
            f"family {decl.family!r} declares no graph classes — there is "
            f"no coordinate to mint")
    classes = target_classes(decl, target)
    if not classes:
        raise MintRefused(
            f"family {decl.family!r} declares target {target!r} but no graph "
            f"class scoped to it — every target states its own coordinates "
            f"(pgw#967), and a target with none has nothing to mint")
    strategy = effective_shape_strategy(decl)
    if not strategy:
        raise MintRefused(
            f"family {decl.family!r} declares classes but no shape_strategy "
            f"— #730 made this a DECLARED per-family choice "
            f"({STATIC_ROWS!r} for conv-bearing families, "
            f"{DYNAMIC_COLLAPSE!r} for DiTs); declare it")
    groups: Dict[Tuple[Tuple[str, Any], ...], List[GraphClass]] = {}
    for cls in classes:
        groups.setdefault(cls.fork, []).append(cls)

    plans: List[MintPlan] = []
    if strategy == STATIC_ROWS:
        for cls in classes:
            plans.append(MintPlan(
                family=decl.family, target=target, fork=cls.fork,
                rows=(cls,), seed=cls, dynamic=()))
        return tuple(plans)
    for fork, rows in groups.items():
        # Seed with the max-volume row (deterministic; ties break on the
        # coordinate itself). Which row seeds the trace decides nothing about
        # what the artifact admits — the derived range does — but the biggest
        # row reaches the allocator peak, mirroring the warmup doctrine.
        def _volume(row: GraphClass) -> int:
            product = 1
            for _name, value in row.dims:
                product *= int(value)
            return product

        seed = max(rows, key=lambda r: (_volume(r), r.dims))
        plans.append(MintPlan(
            family=decl.family, target=target, fork=fork,
            rows=tuple(rows), seed=seed,
            dynamic=derived_dynamic(decl, target, rows)))
    return tuple(plans)


def entry_coordinates(plan: MintPlan) -> Tuple[Tuple[Tuple[str, Any], ...], Tuple[Tuple[str, int], ...]]:
    """The ``(fork, class_dims)`` coordinate one packaged entry is named by.

    A static-rows plan is one class row, so its dims ARE the coordinate; a
    dynamic-collapse plan spans its rows, so the dims segment is empty and
    the fork coordinate alone names the entry (the admitted range is the
    entry's recorded contract, not its name).
    """
    if len(plan.rows) == 1 and not plan.dynamic:
        return plan.fork, plan.rows[0].dims
    return plan.fork, ()


def entry_name(
    target: str,
    fork: Tuple[Tuple[str, Any], ...] = (),
    class_dims: Tuple[Tuple[str, int], ...] = (),
) -> str:
    """The deterministic NAMED-ENTRY label of one graph class inside a
    multi-graph cell (pgw#758, Paul's ruling: separate graphs per function,
    combined into one file).

    ``<target>/<fork k=v,...>/<dims k=v,...>`` with empty segments omitted;
    pairs are sorted, bools render lowercase. The name is the AOTI model
    name inside the ``.pt2`` AND the key of ``metadata.entries``, so it must
    be stable across mints — it derives only from declaration coordinates,
    never from trace artifacts.
    """

    def _value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    segments = [str(target)]
    if fork:
        segments.append(",".join(
            f"{n}={_value(v)}" for n, v in sorted((str(n), v) for n, v in fork)))
    if class_dims:
        segments.append(",".join(
            f"{n}={int(v)}" for n, v in sorted((str(n), int(v)) for n, v in class_dims)))
    return "/".join(segments)


def plan_entry_name(plan: MintPlan) -> str:
    fork, dims = entry_coordinates(plan)
    return entry_name(plan.target, fork, dims)


def cell_plans(decl: Compile) -> Tuple[MintPlan, ...]:
    """EVERY mint plan of one family's declaration, across ALL declared
    targets — the whole class set one cell packages (pgw#758). Refuses a
    declaration whose plans would collide on an entry name (two classes one
    label could not be told apart by a refusal)."""
    if not decl.targets:
        raise MintRefused(
            f"family {decl.family!r} declares no targets — a cell with no "
            f"functions has nothing to package")
    plans: List[MintPlan] = []
    for target in decl.targets:
        plans.extend(mint_plans(decl, target))
    seen: Dict[str, MintPlan] = {}
    for plan in plans:
        name = plan_entry_name(plan)
        if name in seen:
            raise MintRefused(
                f"family {decl.family!r}: two mint plans share entry name "
                f"{name!r} — the declaration does not discriminate them")
        seen[name] = plan
    return tuple(plans)


def select_plan(
    decl: Compile,
    target: str,
    *,
    fork: Mapping[str, Any],
    class_dims: Optional[Mapping[str, int]] = None,
) -> MintPlan:
    """The mint plan for one requested coordinate, refused by name when the
    coordinate is not declared — reading only declared facts, never family
    knowledge."""
    # pgw#790: the adapter arm is an SDK-synthesized coordinate — the endpoint
    # never declared it, so it can never select a DECLARED plan. Stripped here
    # (rather than at each call site) because every reader of a spec's fork
    # goes through this function.
    want_fork = tuple(sorted(
        (str(k), v) for k, v in dict(fork).items()
        if str(k) != ADAPTER_FORK))
    plans = [p for p in mint_plans(decl, target)
             if p.fork == want_fork]
    if not plans:
        have = sorted({str(dict(p.fork))
                       for p in mint_plans(decl, target)})
        raise MintRefused(
            f"family {decl.family!r} declares no graph class at fork "
            f"coordinate {dict(fork)!r} (declared coordinates: {have!r})")
    if class_dims is None:
        if len(plans) > 1:
            raise MintRefused(
                f"family {decl.family!r} has {len(plans)} static rows at fork "
                f"{dict(fork)!r}; the request must name the row via "
                f"'class_dims'")
        return plans[0]
    want_dims = tuple(sorted((str(k), int(v)) for k, v in dict(class_dims).items()))
    for plan in plans:
        for row in plan.rows:
            if row.dims == want_dims:
                if plan.seed.dims == want_dims:
                    return plan
                return replace(plan, seed=row)
    have_rows = sorted(str(dict(r.dims)) for p in plans for r in p.rows)
    raise MintRefused(
        f"family {decl.family!r} declares no class row {dict(class_dims)!r} "
        f"at fork {dict(fork)!r} (declared rows: {have_rows!r})")


# ---------------------------------------------------------------------------
# The fork gate — assert declared arms against the COMPOSED pipeline/module
# ---------------------------------------------------------------------------


def _read_source(obj: Any, field: str) -> Any:
    """A declared fork field off a composed object: the diffusers config
    first (``model_index.json`` / component config — where wan's
    ``expand_timesteps`` actually lives), then a plain attribute (where the
    Wan VAE's ``use_tiling`` actually lives)."""
    config = getattr(obj, "config", None)
    if config is not None:
        value = getattr(config, field, _MISSING)
        if value is _MISSING and hasattr(config, "get"):
            value = config.get(field, _MISSING)
        if value is not _MISSING:
            return value
    return getattr(obj, field, _MISSING)


def fork_gaps(
    decl: Compile,
    fork: Mapping[str, Any],
    *,
    target: str,
    pipeline: Any = None,
    module: Any = None,
) -> List[str]:
    """Named reasons the requested fork coordinate is dishonest.

    Two layers: the coordinate must state every declared fork on a SERVED
    arm, and every fork with a ``source`` binding is read off the composed
    pipeline/module and must agree — a graph exported under the wrong arm
    is the wrong class wearing the declared class's key (ie#566 G6:
    ``expand_timesteps`` is a PIPELINE field, invisible to module-config
    readers, which is why the binding names its source).
    """
    gaps: List[str] = []
    coordinate = {str(k): v for k, v in dict(fork).items()}
    declared_names = {f.name for f in target_forks(decl, target)}
    unknown = sorted(set(coordinate) - {f.name for f in decl.forks})
    if unknown:
        gaps.append(
            f"fork coordinate names undeclared fork(s) {unknown!r} "
            f"(declared: {sorted(f.name for f in decl.forks)!r})")
    for f in target_forks(decl, target):
        if f.name not in coordinate:
            gaps.append(
                f"fork coordinate omits declared fork {f.name!r} — an "
                f"unstated arm silently exports one class as another")
            continue
        declared = coordinate[f.name]
        if declared not in f.served:
            gaps.append(
                f"fork {f.name}={declared!r} is not a served arm "
                f"(served: {list(f.served)!r})")
            continue
        if f.source is None:
            continue
        scope, field = f.source
        obj = pipeline if scope == "pipeline" else module
        if obj is None:
            continue  # no composed object to assert against (planning-time)
        actual = _read_source(obj, field)
        if actual is _MISSING:
            gaps.append(
                f"fork {f.name!r} binds ({scope!r}, {field!r}) but the "
                f"composed {scope} carries neither a config field nor an "
                f"attribute of that name")
            continue
        actual_cmp = bool(actual) if isinstance(declared, bool) else actual
        if actual_cmp != declared:
            gaps.append(
                f"{scope} field {field!r} is {actual!r} but the minted class "
                f"declares {f.name}={declared!r} — the exported graph would "
                f"be the wrong class wearing this class's key")
    del declared_names
    return gaps


# ---------------------------------------------------------------------------
# Generic example inputs — the one code path for every declared family
# ---------------------------------------------------------------------------

_DTYPES: Dict[str, str] = {
    "fp16": "float16", "float16": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp32": "float32", "float32": "float32",
    "int32": "int32", "int64": "int64", "bool": "bool",
}


def _resolve_axis(
    entry: Any, seed: GraphClass, config: Any, *, input_name: str, family: str,
) -> int:
    if isinstance(entry, int):
        return entry
    if isinstance(entry, str):
        value = seed.dim_map.get(entry)
        if value is None:
            raise MintRefused(
                f"input {input_name!r} names dim {entry!r}, which the "
                f"selected class row does not carry (row dims: "
                f"{sorted(seed.dim_map)!r})")
        return int(value)
    _kind, field = entry
    raw: Any = _MISSING if config is None else getattr(config, field, _MISSING)
    if raw is _MISSING and config is not None and hasattr(config, "get"):
        raw = config.get(field, _MISSING)
    if raw is _MISSING or raw is None:
        raise MintRefused(
            f"input {input_name!r} reads config field {field!r}, which the "
            f"resolved module's config does not carry — family "
            f"{family!r}'s declaration does not fit this module")
    return int(raw)


def _resolve_template(
    node: Any, seed: GraphClass, config: Any, *, input_name: str, family: str,
) -> Any:
    """Resolve a canonical ``Arg.template`` against the selected class row.

    Nested tuples stay nested; every leaf goes through the SAME
    :func:`_resolve_axis` a tensor axis does, so "1024 here" and "1024 in a
    shape" cannot disagree about where the number came from. Emits LISTS for
    nesting and TUPLES only for the innermost group, which is the shape
    qwen-image's ``img_shapes`` takes: ``[[(1, H_pat, W_pat)]]``.
    """
    if isinstance(node, tuple) and len(node) == 2 and node[0] == "config":
        return _resolve_axis(node, seed, config,
                             input_name=input_name, family=family)
    if isinstance(node, tuple):
        resolved = [
            _resolve_template(x, seed, config,
                              input_name=input_name, family=family)
            for x in node
        ]
        # Innermost group (all leaves) is a tuple; anything holding
        # containers is a list. That is the python shape real signatures
        # take, and it is derived from the template rather than declared.
        if all(isinstance(x, int) for x in resolved):
            return tuple(resolved)
        return resolved
    return _resolve_axis(node, seed, config,
                         input_name=input_name, family=family)


def container_arities(
    decl: Compile, spec: ExportSpec, module: Any = None,
) -> Dict[str, int]:
    """``{input name: element count}`` for every declared LIST input.

    ``dynamic_shapes`` must mirror the example feed's container structure or
    ``torch.export`` refuses by name — z-image measured the exact sentence:

        Detected mismatch between the structure of `inputs` and
        `dynamic_shapes`: `inputs['x']` is a <class 'list'>, but
        `dynamic_shapes['x']` is a <class 'dict'>

    so the arity the feed used has to reach the spec builder. Derived from
    the same row the feed was built from, never re-guessed.
    """
    rows = [r for r in target_inputs(decl, spec.target) if r.repeat is not None]
    if not rows:
        return {}
    plan = select_plan(
        decl, spec.target, fork=dict(spec.fork),
        class_dims=dict(spec.class_dims) if spec.class_dims else None)
    config = getattr(module, "config", None)
    return {
        r.name: _container_arity(r, plan.seed, config, decl.family)
        for r in rows
    }


def _container_arity(row: Input, seed: GraphClass, config: Any, family: str) -> int:
    arity = _resolve_axis(row.repeat, seed, config,
                          input_name=row.name, family=family)
    if arity < 1:
        raise MintRefused(
            f"input {row.name!r} declares repeat={row.repeat!r}, which the "
            f"selected class row resolves to {arity} — a container input "
            f"with no elements cannot be traced")
    return arity


def declared_inputs(
    module: Any, spec: ExportSpec, decl: Compile,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Example export inputs for ANY declared family — the single generic
    code path that replaces per-family builder modules (#739: the worker
    knows how to compile anything; only endpoints know what they are).

    Widths come from the module's own config where the declaration says so;
    sizes come from the SELECTED CLASS ROW (which row seeds the trace
    decides nothing for a collapsed artifact — the derived range does).

    The returned kwargs dict is ALWAYS empty: all-positional example feeds
    are a mint obligation (pod 9, pgw#723 residuals — the AOTI package call
    convention mirrors the traced args/kwargs split and the serve marshal
    is positional), so every declared row — including nested containers and
    the lifted-LoRA pair — is bound to its slot in the traced signature and
    fed positionally, with signature defaults filling undeclared slots in
    between. A declared name the target only takes keyword-only is refused
    by name: such a target cannot meet the fleet call convention.
    """
    import torch

    rows = target_inputs(decl, spec.target)
    if not rows:
        raise MintRefused(
            f"family {decl.family!r} declares no Input rows for target "
            f"{spec.target!r} — the export call contract is a declaration, "
            f"and this target has none")
    plan = select_plan(
        decl, spec.target,
        fork=dict(spec.fork),
        class_dims=dict(spec.class_dims) if spec.class_dims else None)
    seed = plan.seed
    config = getattr(module, "config", None)
    mod_dtype, device = module_dtype_device(module)

    values: Dict[str, Any] = {}
    for row in rows:
        if row.dtype:
            name = _DTYPES.get(row.dtype)
            if name is None:
                raise MintRefused(
                    f"input {row.name!r} declares unknown dtype {row.dtype!r} "
                    f"(known: {sorted(set(_DTYPES))!r})")
            dtype = getattr(torch, name)
        else:
            dtype = mod_dtype
        shape = tuple(
            _resolve_axis(e, seed, config, input_name=row.name,
                          family=decl.family)
            for e in row.shape)

        def _one() -> Any:
            if not shape:
                return torch.tensor(
                    row.value if row.value is not None else 1.0,
                    dtype=dtype, device=device)
            if row.value is not None:
                return torch.full(shape, row.value, dtype=dtype, device=device)
            if dtype.is_floating_point:
                return torch.randn(shape, dtype=dtype, device=device)
            return torch.ones(shape, dtype=dtype, device=device)

        if row.repeat is None:
            _nest(values, row.name, _one())
        else:
            # pgw#853: a declared LIST parameter. z-image's `x` is a python
            # list over the CFG-doubled batch, and the #739 vocabulary could
            # not say so — `_nest` built a DICT from the dotted name, so
            # `Input('x.0', ...)` yielded {'0': tensor} where the target takes
            # [tensor]. That is blocker B1, and it is an SDK gap, not a torch
            # one: the nested form torch wants exports fine (control rc=4).
            arity = _container_arity(row, seed, config, decl.family)
            _nest(values, row.name, [_one() for _ in range(arity)])
    for arg in target_args(decl, spec.target):
        if arg.template is None:
            _nest(values, arg.name, arg.value)
            continue
        # pgw#853: a ROW-DERIVED structured argument. qwen-image's
        # `img_shapes` is the class row restated as python ints, which
        # `Arg.value` could not express on two counts — a scalar type, and
        # declaration-GLOBAL where the value is per-row (blocker B1).
        resolved = _resolve_template(
            arg.template, seed, config, input_name=arg.name,
            family=decl.family)
        if arg.repeat is not None:
            arity = _resolve_axis(
                arg.repeat, seed, config, input_name=arg.name,
                family=decl.family)
            if arity < 1:
                raise MintRefused(
                    f"arg {arg.name!r} declares repeat={arg.repeat!r}, which "
                    f"the selected class row resolves to {arity}")
            resolved = [resolved for _ in range(arity)]
        _nest(values, arg.name, resolved)
    values.update(lifted_lora_values(module, spec))
    return _positionalize(module, spec.target, decl.family, values), {}


def target_attr(target: str) -> str:
    """The callable a compile target names on its resolved module."""
    return target.rsplit(".", 1)[1] if "." in target else "forward"


def call_signature(module: Any, target: str, family: str) -> Tuple[List[Any], Set[str]]:
    """``(positional parameters, keyword-only names)`` of the callable
    ``target`` names on ``module``, refused by name when unreadable.

    ONE read for two callers (pgw#822): :func:`_positionalize`, which binds
    the declared feed to positional slots at mint time, and
    :func:`gen_worker.aot_mint.declaration_module_gaps`, which asks the same
    question on the PARENT before a child is spawned or a pod is rented.
    Two implementations of "what does this forward take" would diverge, and
    the whole value of the pre-spawn check is that it predicts the mint.
    """

    attr = target_attr(target)
    fn = getattr(module, attr, None)
    if fn is None:
        raise MintRefused(
            f"family {family!r}: resolved module {type(module).__name__} has "
            f"no callable {attr!r} to read a signature from")
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError) as exc:
        raise MintRefused(
            f"family {family!r}: cannot read the signature of {attr!r} on "
            f"{type(module).__name__} ({exc}); without it the declared "
            f"inputs cannot be bound to positional slots") from exc
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.name != "self"
    ]
    keyword_only = {p.name for p in params if p.kind == p.KEYWORD_ONLY}
    return positional, keyword_only


def _positionalize(
    module: Any, target: str, family: str, values: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Bind named values to their slots in the traced signature, all-positional.

    Undeclared slots BETWEEN declared ones are filled with their signature
    defaults; trailing undeclared slots are omitted. Refusals by name: a
    declared name that is not a parameter (declaration/module drift), a
    keyword-only declared name (cannot meet the positional serve marshal),
    and a required in-between parameter with no default and no declaration.
    """

    attr = target_attr(target)
    positional, keyword_only = call_signature(module, target, family)

    hit = sorted(keyword_only & set(values))
    if hit:
        raise MintRefused(
            f"family {family!r}: declared input(s) {hit!r} are KEYWORD-ONLY "
            f"parameters of {attr!r} — all-positional example feeds are a "
            f"mint obligation (the AOTI package call convention mirrors the "
            f"traced args/kwargs split and the serve marshal is positional; "
            f"pod 9, pgw#723 residuals), so this target cannot be exported "
            f"as declared")
    names = [p.name for p in positional]
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise MintRefused(
            f"family {family!r}: declared input(s) {unknown!r} are not "
            f"parameters of {attr!r} on {type(module).__name__} "
            f"(parameters: {names!r}) — the declaration does not fit this "
            f"module")
    last = max(names.index(n) for n in values)
    args: List[Any] = []
    for p in positional[: last + 1]:
        if p.name in values:
            args.append(values[p.name])
        elif p.default is not inspect.Parameter.empty:
            args.append(p.default)
        else:
            raise MintRefused(
                f"family {family!r}: parameter {p.name!r} of {attr!r} sits "
                f"before declared inputs, is not declared, and has no "
                f"default — the positional feed cannot skip it")
    return tuple(args)


def _nest(kwargs: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = kwargs
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise MintRefused(
                f"input {dotted!r} nests under {part!r}, which another row "
                f"already declared as a non-container")
    node[parts[-1]] = value


# ---------------------------------------------------------------------------
# Request-side resolution — python -m gen_worker.aot_mint consumes this
# ---------------------------------------------------------------------------


def load_declaration(request: Mapping[str, Any], request_path: Optional[Path] = None) -> None:
    """Import the endpoint's declaration module named by a mint request.

    ``declaration_module`` is a dotted import; ``declaration_file`` is a
    path (relative paths resolve against the request file, so a request in
    ``aot/`` can say ``"declaration.py"``). Importing runs the endpoint's
    ``register_export_declaration`` call — the same load-to-register shape
    as ``convert.registry.load_declaration_module``.
    """
    dotted = str(request.get("declaration_module") or "").strip()
    file_ref = str(request.get("declaration_file") or "").strip()
    if dotted:
        try:
            importlib.import_module(dotted)
        except ImportError as exc:
            raise MintRefused(
                f"cannot import declaration module {dotted!r}: {exc}") from exc
    if file_ref:
        path = Path(file_ref)
        if not path.is_absolute() and request_path is not None:
            path = Path(request_path).parent / path
        if not path.exists():
            raise MintRefused(f"declaration file {path} does not exist")
        module_spec = importlib.util.spec_from_file_location(
            f"_gen_worker_declaration_{path.stem}", path)
        if module_spec is None or module_spec.loader is None:
            raise MintRefused(f"cannot load declaration file {path}")
        mod = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(mod)


__all__ = [
    "MintPlan",
    "call_signature",
    "cell_plans",
    "declared_inputs",
    "derived_dynamic",
    "dim_hull",
    "effective_shape_strategy",
    "entry_coordinates",
    "entry_name",
    "fork_gaps",
    "load_declaration",
    "mint_plans",
    "plan_entry_name",
    "select_plan",
    "target_args",
    "target_attr",
    "target_forks",
    "target_inputs",
]
