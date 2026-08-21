"""Dynamic export dims: one graph where the fan was a bucket per shape (tcg#77).

The observed set is still the truth (``discovery``'s doctrine is unchanged) —
this module only decides how many EXPORTS that set costs. A denoiser observed
at seven aspect buckets x two CFG batches is fourteen calls that differ in
nothing but four integers; exported one-per-call that is fourteen graph
specializations, fourteen AOTI compiles, and fourteen artifacts to mint, adopt
and match. Exported with those four integers marked ``torch.export.Dim`` it is
ONE, and the shapes it serves are stated as ranges the dispatcher guards
(``ingress.symbols``, ``adopt._row_mismatch``) instead of as equalities.

The mechanism, in the order it runs:

1. **Group** observed calls by everything a graph cannot vary over: structure,
   non-tensor leaf values, dtypes and ranks. Two calls in different groups can
   never be one export.
2. **Classify** each (leaf, axis) inside a group by the TUPLE of values it
   takes across that group's calls. Axes with one value are static. Axes whose
   value tuples are IDENTICAL move together and must therefore be one symbol —
   a batch axis on ``sample`` and on ``encoder_hidden_states`` is one Dim, and
   exporting them as two would earn a constraint violation naming the equality
   we already knew.
3. **Filter** by the caller's policy. An axis the policy refuses stays an
   integer, so the caller can adopt dynamic dims one axis at a time — which is
   what a benchmark that must justify each axis needs (pgw#1548).
4. **Re-group** on the axes that stayed static. Whatever the policy refused is
   still a fan; it is simply a smaller one.

Nothing here promises a shape it did not observe: a symbol's range is the
observed min..max. A live call outside it misses every armed record and serves
eager, which is the same answer an unobserved bucket has always gotten.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from math import gcd
from typing import Any

logger = logging.getLogger(__name__)

#: ``(target, input name, axis) -> may this axis be exported dynamic?``
#:
#: The input name is the ingress spelling (``sample``,
#: ``added_cond_kwargs.text_embeds``), so a caller that knows its own model
#: can name an axis exactly. torchcg deliberately holds no opinion about
#: which axis is "batch" or "spatial": that is the endpoint's vocabulary.
DimPolicy = Callable[[str, str, int], bool]


def all_dims(_target: str, _name: str, _axis: int) -> bool:
    """Every axis that varies across the observed set becomes a Dim."""

    return True


def resolve_policy(policy: DimPolicy | bool | None) -> DimPolicy | None:
    """``None``/``False`` -> static exports; ``True`` -> :func:`all_dims`."""

    if policy is None or policy is False:
        return None
    if policy is True:
        return all_dims
    if not callable(policy):
        raise TypeError(
            f"dynamic_dims takes a bool or a (target, name, axis) predicate; "
            f"got {type(policy).__name__}"
        )
    return policy


def _leaves(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    if isinstance(value, Mapping):
        return [
            leaf
            for key, item in value.items()
            for leaf in _leaves(item, path + (key,))
        ]
    if isinstance(value, (list, tuple)):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _leaves(item, path + (index,))
        ]
    return [(path, value)]


def _name(param: str, path: Sequence[Any]) -> str:
    name = param
    for step in path:
        name = step if isinstance(step, str) else f"{name}.{step}"
    return name


def _flat(
    param_names: Sequence[str], args: Sequence[Any], kwargs: Mapping[str, Any]
) -> list[tuple[str, tuple[Any, ...], Any]]:
    """``[(param, path, value)]`` in the ingress's own flattening order."""

    values = list(args) + [kwargs[name] for name in param_names[len(args):]]
    return [
        (param, path, leaf)
        for param, value in zip(param_names, values, strict=True)
        for path, leaf in _leaves(value)
    ]


def _shape_free_key(flat: Sequence[tuple[str, tuple[Any, ...], Any]]) -> str:
    """Everything about a call one export cannot vary over."""

    import torch

    parts = []
    for param, path, value in flat:
        if isinstance(value, torch.Tensor):
            parts.append(f"{param}{path!r}=t({value.dtype}|rank{value.dim()})")
        else:
            parts.append(f"{param}{path!r}={value!r}")
    return "|".join(parts)


def _dims(flat: Sequence[tuple[str, tuple[Any, ...], Any]]) -> list[tuple[int, int]]:
    import torch

    return [
        (index, axis)
        for index, (_param, _path, value) in enumerate(flat)
        if isinstance(value, torch.Tensor)
        for axis in range(value.dim())
    ]


class RangeRefusal(ValueError):
    """A declared symbolic range the graph's OWN guards contradict (tcg#78).

    The refusal exists because the alternative was measured, not imagined. An
    sd15 UNet exported with a batch ``Dim`` over ``[1, 2]`` compiles for
    84-129 s of GPU and produces an artifact that, called at batch 1, RETURNS
    A ``(2, 4, 64, 64)`` TENSOR OF GARBAGE AND RAISES NOTHING -- because
    ``torch._prims.convert_element_type``'s contiguity question
    (``guard_or_false(prod(sizes) < 2)``) guards ``s53 >= 2``, ``_refine_ranges``
    narrows ``[1, 2]`` to the singleton ``[2, 2]``, and a singleton range makes
    the ShapeEnv replace the symbol with the constant 2 everywhere downstream.
    The lock still said ``[1, 2]``, so the dispatcher would have admitted the
    batch-1 call the artifact cannot serve.

    The honest answer is not to record the narrowed range -- that makes the
    declaration a claim the author never made -- but to refuse the AXIS and
    say which guard refused it. The author then declares the range the graph
    actually supports, and the shapes outside it get their own concrete
    specialization or serve eager, exactly as an unobserved bucket always has.
    """


def shape_env_of(program: object) -> Any:
    """The one ShapeEnv reachable from a program's placeholder metadata."""

    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        value = (getattr(node, "meta", None) or {}).get("val")
        for dimension in getattr(value, "shape", ()) or ():
            environment = getattr(getattr(dimension, "node", None), "shape_env", None)
            if environment is not None:
                return environment
    return None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def live_symbol_ranges(program: object) -> dict[str, tuple[int | None, int | None]]:
    """What the ShapeEnv says RIGHT NOW about every symbol it carries.

    Read before and after a stage, this states what that stage did to the
    ranges -- which is the only way to see a refinement, because
    ``ExportedProgram.range_constraints`` is a snapshot taken at export and
    never moves again. A symbol the environment has REPLACED with a constant
    reads as that constant's singleton range: a replacement is the strongest
    narrowing there is, and it must not read as "unchanged".
    """

    environment = shape_env_of(program)
    if environment is None:
        return {}
    replacements = {
        str(symbol): value
        for symbol, value in (getattr(environment, "replacements", {}) or {}).items()
    }
    ranges: dict[str, tuple[int | None, int | None]] = {}
    for symbol, interval in (getattr(environment, "var_to_range", {}) or {}).items():
        name = str(symbol)
        replaced = _integer(replacements.get(name))
        if replaced is not None:
            ranges[name] = (replaced, replaced)
            continue
        ranges[name] = (
            _integer(getattr(interval, "lower", None)),
            _integer(getattr(interval, "upper", None)),
        )
    return ranges


def declared_symbol_ranges(program: object) -> dict[str, tuple[int | None, int | None]]:
    """The ranges the exported program DECLARES, from its own constraints."""

    return {
        str(symbol): (
            _integer(getattr(interval, "lower", None)),
            _integer(getattr(interval, "upper", None)),
        )
        for symbol, interval in (getattr(program, "range_constraints", {}) or {}).items()
    }


def narrowed_symbols(
    declared: Mapping[str, tuple[int | None, int | None]],
    live: Mapping[str, tuple[int | None, int | None]],
) -> dict[str, tuple[tuple[int | None, int | None], tuple[int | None, int | None]]]:
    """``{symbol: (declared, refined)}`` for every range that got SMALLER.

    Only narrowing is reported. A range that widened -- which nothing observed
    does -- would still cover every value the declaration promised, so it is
    not a shape the dispatcher can send somewhere the artifact refuses.
    """

    moved = {}
    for name, (low, high) in declared.items():
        if name not in live:
            continue
        live_low, live_high = live[name]
        if (live_low is not None and low is not None and live_low > low) or (
            live_high is not None and high is not None and live_high < high
        ):
            moved[name] = ((low, high), (live_low, live_high))
    return moved


def range_narrowing(
    program: object,
) -> dict[str, tuple[tuple[int | None, int | None], tuple[int | None, int | None]]]:
    """Which of a program's DECLARED ranges its own ShapeEnv already denies.

    These two are not the same statement and nothing in torch reconciles them.
    ``range_constraints`` records what the AUTHOR asked for and is frozen at
    export; the ShapeEnv records what the graph's guards have established and
    keeps moving. Measured on the sd15 UNet, immediately after an export that
    raised nothing::

        declared {'s53': (1, 2)}   live {'s53': (2, 2)}

    Every dynamic dim is guarded ``>= 2`` -- torch specializes the sizes 0 and
    1 rather than reason about broadcasting and contiguity symbolically -- so
    an axis whose observed minimum is 1 is contradicted the moment it is
    exported. The lock reads ``range_constraints``, which is why the
    contradiction survived all the way to a compiled artifact.

    Costs nothing: both sides are already in memory.
    """

    return narrowed_symbols(declared_symbol_ranges(program), live_symbol_ranges(program))


def decomposition_narrowing(
    program: object,
) -> dict[str, tuple[tuple[int | None, int | None], tuple[int | None, int | None]]]:
    """Every declared range the graph's own guards contradict, decomposed.

    The free check first. If it clears, decompose and ask again: an AOTI
    compile decomposes before it generates code, and some size-1 questions are
    asked only by the decomposed graph -- ``aten.to.dtype`` carries no such
    guard, the ``torch._prims.convert_element_type`` it lowers to does. A
    narrowing found only there would otherwise land 84-129 s into a GPU
    compile. Asking here costs ~2 s of CPU and no compiler at all (tcg#78).

    The program is not copied: the probe shares its ShapeEnv, so a narrowing
    it finds has also polluted this program -- which is exactly why a caller
    that finds one must DISCARD the program rather than declare it. When there
    is no narrowing, there is nothing to have polluted.
    """

    if not declared_symbol_ranges(program):
        return {}
    moved = range_narrowing(program)
    if moved:
        return moved
    decompose = getattr(program, "run_decompositions", None)
    if not callable(decompose):
        return {}
    decompose()
    return range_narrowing(program)


def contradicted_axes(
    target: str, ingress: Any, moved: Mapping[str, Any]
) -> set[tuple[str, str, int]]:
    """``{(target, ingress name, axis)}`` carried by a contradicted symbol.

    Scoped to the target because ``sample`` axis 0 means one thing on a
    denoiser and something else on a VAE; a refusal earned by one must not be
    charged to the other.

    The unit a policy speaks in is the AXIS, not the symbol, so a refusal has
    to be translated before it can be acted on. A derived dim spells its root
    in its own name (``16*s3``), which is also the key ``range_constraints``
    uses, so one ``symbol_terms`` read connects the two.

    Refusing axes rather than the whole export is what keeps a contradicted
    batch axis from taking the aspect collapse down with it -- the case the
    dynamic-dims program is actually for (pgw#1548: SDXL 18 -> 2).
    """

    from .ingress import symbol_terms

    return {
        (target, row.name, axis)
        for row in ingress.inputs
        for axis, dimension in enumerate(row.shape)
        if isinstance(dimension, str) and symbol_terms(dimension)[1] in moved
    }


def without_axes(
    policy: DimPolicy | None, refused: set[tuple[str, str, int]]
) -> DimPolicy:
    """``policy`` with named axes refused on top of whatever it already refuses."""

    resolved = policy if policy is not None else all_dims

    def narrowed(target: str, name: str, axis: int) -> bool:
        return (target, name, axis) not in refused and resolved(target, name, axis)

    return narrowed


def format_narrowing(
    moved: Mapping[str, tuple[tuple[int | None, int | None], tuple[int | None, int | None]]],
) -> str:
    """``s53 declared [1, 2] but the graph's guards allow only [2, 2]``."""

    return "; ".join(
        f"{name} declared [{low}, {high}] but the graph's guards allow only "
        f"[{new_low}, {new_high}]"
        for name, ((low, high), (new_low, new_high)) in sorted(moved.items())
    )


Call = tuple[tuple[Any, ...], dict[str, Any], Sequence[str]]


class DynamicPlan:
    """One export: its representative call and the Dims it is exported under."""

    __slots__ = (
        "args", "kwargs", "param_names", "dynamic_shapes", "symbols", "covers",
        "classes",
    )

    def __init__(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        param_names: Sequence[str],
        dynamic_shapes: dict[str, Any] | None,
        symbols: dict[str, tuple[int, int]],
        covers: Sequence[Call] = (),
        classes: Sequence[Sequence[tuple[int, int]]] = (),
    ) -> None:
        self.args = args
        self.kwargs = kwargs
        self.param_names = tuple(param_names)
        self.dynamic_shapes = dynamic_shapes
        self.symbols = symbols
        #: Every observed call this ONE export stands for. A dynamic plan that
        #: the exporter refuses is replaced by exactly these, exported the way
        #: they always were — the fan comes back, nothing is dropped.
        self.covers = tuple(covers) or ((args, dict(kwargs), tuple(param_names)),)
        #: The ``(flat position, axis)`` groups that were exported as ONE Dim.
        #: torch may still hand each occurrence its own symbol and tie them
        #: with a runtime guard — measured: a batch-2 sample against a batch-1
        #: conditioning fails `Guard failed: …` inside the graph — so the
        #: record has to say the tie itself or the dispatcher would enter a
        #: graph that refuses the call. :func:`alias_ingress` does that.
        self.classes = tuple(tuple(group) for group in classes)

    @property
    def dynamic(self) -> bool:
        return self.dynamic_shapes is not None

    def static_fallback(self) -> list[DynamicPlan]:
        return [
            DynamicPlan(args, kwargs, names, None, {})
            for args, kwargs, names in self.covers
        ]


def _dim(name: str, low: int, high: int, step: int) -> Any:
    """The Dim for one axis, expressed in the units the model can actually take.

    An axis is rarely free over the integers: a UNet with three downsamples
    accepts a latent side only in multiples of eight, and ``torch.export``
    says so as a guard (``(size % 2) == 0``) that refuses a plain ``Dim``.
    The observed values' GCD is that stride, measured rather than assumed, so
    the exported dim is ``stride * root`` — a DERIVED dim, which is torch's
    own spelling for exactly this and the one it suggests in the refusal.
    """

    import torch

    if step <= 1:
        return torch.export.Dim(name, min=low, max=high)
    root = torch.export.Dim(f"{name}_root", min=low // step, max=high // step)
    return step * root


def _spec_for(
    value: Any, index: int, axes: Mapping[tuple[int, int], Any]
) -> Any:
    """The ``dynamic_shapes`` spec for ONE flat leaf."""

    import torch

    if not isinstance(value, torch.Tensor):
        return None
    marked = {
        axis: axes[(index, axis)]
        for axis in range(value.dim())
        if (index, axis) in axes
    }
    return marked or None


def _rebuild(value: Any, counter: list[int], axes: Mapping[tuple[int, int], Any]) -> Any:
    """Mirror one argument's container structure with per-leaf specs."""

    if isinstance(value, Mapping):
        return {key: _rebuild(item, counter, axes) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_rebuild(item, counter, axes) for item in value)
    index = counter[0]
    counter[0] += 1
    return _spec_for(value, index, axes)


def plan_exports(
    target: str,
    calls: Sequence[tuple[tuple[Any, ...], dict[str, Any], Sequence[str]]],
    policy: DimPolicy | None,
) -> list[DynamicPlan]:
    """Turn one target's observed calls into the exports they actually need.

    Every call is covered exactly once. With no policy (or nothing varying)
    this is the identity: one static plan per call, which is what discovery
    did before this module existed.
    """

    plans: list[DynamicPlan] = []
    if policy is None:
        return [
            DynamicPlan(args, kwargs, names, None, {})
            for args, kwargs, names in calls
        ]

    groups: dict[str, list[int]] = {}
    flats = [_flat(names, args, kwargs) for args, kwargs, names in calls]
    for index, flat in enumerate(flats):
        groups.setdefault(_shape_free_key(flat), []).append(index)

    for members in groups.values():
        plans.extend(_plan_group(target, calls, flats, members, policy))
    return plans


def _plan_group(
    target: str,
    calls: Sequence[tuple[tuple[Any, ...], dict[str, Any], Sequence[str]]],
    flats: Sequence[Sequence[tuple[str, tuple[Any, ...], Any]]],
    members: Sequence[int],
    policy: DimPolicy,
) -> list[DynamicPlan]:
    if len(members) == 1:
        args, kwargs, names = calls[members[0]]
        return [DynamicPlan(args, kwargs, names, None, {})]

    reference = flats[members[0]]
    positions = _dims(reference)
    # The value each axis takes across the group, in member order.
    tracks: dict[tuple[int, int], tuple[int, ...]] = {
        position: tuple(
            int(flats[member][position[0]][2].shape[position[1]]) for member in members
        )
        for position in positions
    }
    allowed = {
        position: track
        for position, track in tracks.items()
        if len(set(track)) > 1
        and policy(
            target,
            _name(reference[position[0]][0], reference[position[0]][1]),
            position[1],
        )
    }
    if not allowed:
        return [
            DynamicPlan(calls[member][0], calls[member][1], calls[member][2], None, {})
            for member in members
        ]

    # Axes whose value tracks are identical MOVE TOGETHER and are one symbol.
    classes: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for position, track in sorted(allowed.items()):
        classes.setdefault(track, []).append(position)

    # Whatever stayed static still fans out; re-group on exactly that.
    static = [
        position
        for position in positions
        if position not in allowed and len(set(tracks[position])) > 1
    ]
    buckets: dict[tuple[int, ...], list[int]] = {}
    for offset, member in enumerate(members):
        key = tuple(tracks[position][offset] for position in static)
        buckets.setdefault(key, []).append(member)

    plans: list[DynamicPlan] = []
    for bucket in buckets.values():
        plans.append(_plan_bucket(target, calls, flats, members, bucket, classes))
    return plans


def _plan_bucket(
    target: str,
    calls: Sequence[tuple[tuple[Any, ...], dict[str, Any], Sequence[str]]],
    flats: Sequence[Sequence[tuple[str, tuple[Any, ...], Any]]],
    members: Sequence[int],
    bucket: Sequence[int],
    classes: Mapping[tuple[int, ...], list[tuple[int, int]]],
) -> DynamicPlan:
    offsets = {member: offset for offset, member in enumerate(members)}
    live: dict[tuple[int, ...], tuple[int, int]] = {}
    for track in classes:
        seen = {track[offsets[member]] for member in bucket}
        if len(seen) > 1:
            live[track] = (min(seen), max(seen))

    # The representative: the LARGEST observed shape, so the compiler's
    # example is the one that has to fit, and a min-sized example cannot
    # tempt inductor into specializing the axis away. Ties break on the
    # member order, which is the observation order — deterministic.
    def extent(member: int) -> tuple[int, int]:
        product = 1
        for track in live:
            product *= track[offsets[member]]
        return (-product, member)

    chosen = sorted(bucket, key=extent)[0]
    args, kwargs, names = calls[chosen]
    if not live:
        return DynamicPlan(args, kwargs, names, None, {})

    axes: dict[tuple[int, int], Any] = {}
    symbols: dict[str, tuple[int, int]] = {}
    reference = flats[members[0]]
    for order, (track, span) in enumerate(sorted(live.items())):
        first = sorted(classes[track])[0]
        name = f"{_name(reference[first[0]][0], reference[first[0]][1])}_{first[1]}"
        name = "".join(char if char.isalnum() else "_" for char in name)
        if not name[:1].isalpha():
            name = f"d{order}_{name}"
        step = 0
        for member in bucket:
            step = gcd(step, track[offsets[member]])
        dim = _dim(name, span[0], span[1], step)
        symbols[name] = span
        for position in classes[track]:
            axes[position] = dim

    counter = [0]
    presented = list(args) + [kwargs[name] for name in names[len(args):]]
    shapes = {
        name: _rebuild(value, counter, axes)
        for name, value in zip(names, presented, strict=True)
    }
    return DynamicPlan(
        args,
        kwargs,
        names,
        shapes,
        symbols,
        [calls[member] for member in bucket],
        [classes[track] for track in sorted(live)],
    )


def alias_ingress(ingress: Any, plan: DynamicPlan) -> Any:
    """Restate one ingress so axes exported as ONE Dim carry ONE symbol name.

    ``torch.export`` gives each occurrence of a shared Dim its own symbol and
    ties them with an in-graph guard. A guard inside the graph is invisible to
    the dispatcher, which would then admit a call the compiled artifact
    refuses — so the tie is moved OUT to the ingress, where every guard we own
    can see it (``adopt`` binds one value per symbol per call).
    """

    from .ingress import CallIngress, CallInput

    if not plan.classes:
        return ingress
    rows = {row.position: row for row in ingress.inputs}
    rename: dict[str, str] = {}
    for group in plan.classes:
        names = []
        for position, axis in group:
            row = rows.get(position)
            if row is None or axis >= len(row.shape):
                continue
            dimension = row.shape[axis]
            if isinstance(dimension, str):
                names.append(dimension)
        for name in names[1:]:
            if name != names[0]:
                rename[name] = names[0]
    if not rename:
        return ingress
    inputs = tuple(
        CallInput(
            name=row.name,
            position=row.position,
            param=row.param,
            param_position=row.param_position,
            path=row.path,
            exported_name=row.exported_name,
            dtype=row.dtype,
            shape=tuple(
                rename.get(dimension, dimension) if isinstance(dimension, str)
                else dimension
                for dimension in row.shape
            ),
        )
        for row in ingress.inputs
    )
    kept = {
        dimension
        for row in inputs
        for dimension in row.shape
        if isinstance(dimension, str)
    }
    return CallIngress(
        parameters=ingress.parameters,
        flat_arity=ingress.flat_arity,
        inputs=inputs,
        symbols=tuple(
            (name, bounds) for name, bounds in ingress.symbols if name in kept
        ),
        excluded_inputs=ingress.excluded_inputs,
    )


__all__ = [
    "DimPolicy",
    "DynamicPlan",
    "RangeRefusal",
    "alias_ingress",
    "all_dims",
    "contradicted_axes",
    "declared_symbol_ranges",
    "decomposition_narrowing",
    "format_narrowing",
    "live_symbol_ranges",
    "narrowed_symbols",
    "plan_exports",
    "range_narrowing",
    "resolve_policy",
    "shape_env_of",
    "without_axes",
]
