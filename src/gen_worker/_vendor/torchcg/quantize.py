"""A quant recipe is a LANE CONTRACT's property (tcg#53), not a serve-time cast.

The second instance of tcg#52's pass mechanism, and the argument that it IS a
mechanism rather than one model's special case.

**Where the bytes come from is a separate lane.** The right place to produce
fp8 weights is INGEST: the store holds the converted bytes, ``ctx.load``
streams them straight to VRAM, and the endpoint's deferred-to-first-card-
landing dance deletes itself. That producer is the conversion lane's work
(th#2164) and is deliberately NOT here. What IS here is:

* the **recipe vocabulary** -- a closed, frozen object a lane contract names,
  shared by the ingest converter and the in-place fallback so the two cannot
  drift into two definitions of "fp8";
* the **plan with reasons** -- which modules convert and, for every module
  that does not, WHY; and
* the **in-place fallback arm** as a :class:`~torchcg.transform.TransformPass`,
  for a lane served from bf16 bytes that no ingest conversion has reached.

torchao is NOT a torchcg dependency: the caller supplies the ``convert``
callable. That is the seam -- ingest and fallback hand the same
:class:`QuantPlan` to their own converters and both are held to the same
refusals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .lane import LaneRef
from .transform import (
    TransformError,
    TransformMode,
    TransformPlan,
    TransformReport,
    dtype_label,
    module_bytes,
    register_pass,
    require_pass_ref,
    tensor_bytes,
)


class RecipeError(TransformError):
    """A quant recipe cannot be stated, planned, or honoured."""


#: Why a module the scope contains was NOT converted. A closed set, because
#: "kept, reason unknown" is the row that hides a half-quantized model.
KEPT_OUTSIDE = "outside_scope"
KEPT_SKIPPED = "skip_list"
KEPT_UNALIGNED = "unaligned"
KEPT_DTYPE = "dtype"
KEPT_KIND = "kind"


@dataclass(frozen=True, slots=True)
class ModuleSelect:
    """The scope, and the refusals that ARE the product.

    ``root`` is a submodule prefix and it is not decoration: rooting the
    conditioner scope at the DECODER STACK makes "outside the layers" the
    DEFAULT rather than a rule that must catch every future sibling -- the
    vision tower's 4304-wide MLP is not even expressible block-wise, and a
    scope that has to enumerate its exclusions will miss the next one.
    """

    root: str = ""
    skip: tuple[str, ...] = ()
    #: in/out feature alignment the kernel requires; 0 disables the check
    align: int = 16
    dtypes: tuple[str, ...] = ("bfloat16", "float16")

    def __post_init__(self) -> None:
        if self.align < 0:
            raise RecipeError("alignment must not be negative")
        if not self.dtypes:
            raise RecipeError(
                "a scope with no admitted dtypes converts nothing; state the "
                "dtypes the recipe reads"
            )


@dataclass(frozen=True, slots=True)
class QuantRecipe:
    """A named quantization contract. The author types NONE of this.

    ``granularity`` has NO DEFAULT on purpose. A bare config normalizes to
    per-tensor everything, which measured ~3.5x the distortion of per-row x
    per-row; a defaultable field would let that back in silently.

    ``pays_only_under_compile`` is a FIELD, not a comment. Eager fp8 measured
    0.82x -- a LOSS (pgw#1027) -- so a speed arm must not arm on a pod that
    cannot compile. A MEMORY arm must arm anyway: the conditioner encode is
    ~0.3 s of a 200 s request and what it buys is the all-resident pipeline,
    i.e. the 77 s/request of component land+evict pgw#1093 measured. Two arms
    of one recipe with opposite compile dependence is exactly why this is
    typed.
    """

    name: str
    scheme: str
    granularity: tuple[str, str]
    select: ModuleSelect
    min_sm: tuple[int, int]
    pays_only_under_compile: bool

    def __post_init__(self) -> None:
        require_pass_ref(self.name)
        if not self.scheme.strip():
            raise RecipeError(f"recipe {self.name}: scheme must be stated")
        if len(self.granularity) != 2 or not all(
            isinstance(part, str) and part.strip() for part in self.granularity
        ):
            raise RecipeError(
                f"recipe {self.name}: granularity is (activation, weight) and both "
                f"halves must be stated -- there is no default"
            )
        if len(self.min_sm) != 2 or not all(isinstance(n, int) for n in self.min_sm):
            raise RecipeError(f"recipe {self.name}: min_sm is a (major, minor) pair")


@dataclass(frozen=True, slots=True)
class KeptModule:
    """One module inside the scope that was NOT converted, and why."""

    fqn: str
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class QuantPlan:
    """What the recipe converts, and every reason it did not."""

    recipe: str
    convert: tuple[str, ...]
    kept: tuple[KeptModule, ...]

    @property
    def kept_by_reason(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for row in self.kept:
            grouped.setdefault(row.reason, []).append(row.fqn)
        return {reason: tuple(rows) for reason, rows in sorted(grouped.items())}


def _linear_shape(module: Any) -> tuple[int, int] | None:
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if isinstance(in_features, int) and isinstance(out_features, int):
        return in_features, out_features
    return None


def is_resident(module: Any) -> bool:
    """Whether ``module``'s weight is ALREADY quantized.

    A converted weight is a tensor subclass: it answers
    ``__tensor_flatten__`` and its real components are narrower than the
    dtype it advertises. Re-running a converter over one of those is the
    double-quantization defect, so every arm asks this first -- ingest
    conversion and the fallback alike.
    """

    weight = getattr(module, "weight", None)
    if weight is None:
        return False
    flatten = getattr(weight, "__tensor_flatten__", None)
    if not callable(flatten):
        return False
    logical = int(getattr(weight, "element_size", lambda: 0)()) * int(
        getattr(weight, "numel", lambda: 0)()
    )
    return logical == 0 or tensor_bytes(weight) < logical


def plan_quant(module: Any, recipe: QuantRecipe) -> QuantPlan:
    """Which modules the recipe converts, with a reason for every one it keeps.

    Refuses an EMPTY convert set: a recipe that matches nothing means the
    module shape moved under it, and serving a silently-bf16 model as an fp8
    lane is the defect class this exists for.
    """

    from torch import nn

    scope_root = recipe.select.root
    prefix = f"{scope_root}." if scope_root else ""
    convert: list[str] = []
    kept: list[KeptModule] = []
    for fqn, child in module.named_modules():
        if not fqn or not isinstance(child, nn.Linear):
            continue
        if prefix and not fqn.startswith(prefix):
            kept.append(KeptModule(fqn, KEPT_OUTSIDE, scope_root))
            continue
        if any(needle in fqn for needle in recipe.select.skip):
            hit = next(needle for needle in recipe.select.skip if needle in fqn)
            kept.append(KeptModule(fqn, KEPT_SKIPPED, hit))
            continue
        if is_resident(child):
            continue  # already converted upstream or by ingest: never re-run
        weight = getattr(child, "weight", None)
        label = str(getattr(weight, "dtype", "?")).removeprefix("torch.")
        if label not in recipe.select.dtypes:
            kept.append(KeptModule(fqn, KEPT_DTYPE, label))
            continue
        shape = _linear_shape(child)
        if shape is None:
            kept.append(KeptModule(fqn, KEPT_KIND, type(child).__name__))
            continue
        align = recipe.select.align
        if align and (shape[0] % align or shape[1] % align):
            kept.append(KeptModule(fqn, KEPT_UNALIGNED, f"{shape[0]}x{shape[1]}"))
            continue
        convert.append(fqn)
    if not convert:
        raise RecipeError(
            f"recipe {recipe.name}: nothing in scope {scope_root or '<root>'} is "
            f"convertible ({len(kept)} module(s) kept: "
            f"{sorted({row.reason for row in kept}) or ['<none>']}). An empty plan "
            f"means the module shape moved under the recipe -- refusing rather "
            f"than serving a silently-{'/'.join(recipe.select.dtypes)} lane as "
            f"{recipe.name}"
        )
    return QuantPlan(recipe=recipe.name, convert=tuple(convert), kept=tuple(kept))


@register_pass
class RecipeQuantize:
    """The IN-PLACE FALLBACK arm: convert at load when ingest has not.

    The preferred path is bytes at rest under the fp8 contract; this arm
    exists for a lane whose store still holds bf16. It is a pass and not a
    serve-time cast, which is the whole point: it runs in the PASS phase,
    before discovery, so the graph that gets identity is the quantized graph.

    ``convert`` is the caller's converter (``torchao.quantize_`` in
    production, the ingest job's own in the conversion lane); torchcg holds
    the vocabulary and the refusals, not the kernel.
    """

    NAME = "quantize-in-place@1"

    def __init__(
        self,
        *,
        recipe: QuantRecipe,
        convert: Callable[[Any, QuantPlan], None],
        sm: tuple[int, int] | None = None,
        compiled: bool = False,
    ) -> None:
        self.recipe = recipe
        self.convert = convert
        self.sm = sm
        self.compiled = compiled

    @property
    def name(self) -> str:
        return self.NAME

    def plan(self, target: Any, *, lane: LaneRef) -> TransformPlan:
        if self.recipe.pays_only_under_compile and not self.compiled:
            raise RecipeError(
                f"recipe {self.recipe.name} pays only under compile (eager measured "
                f"0.82x -- a LOSS, pgw#1027) and this session states compiled=False; "
                f"refusing to arm it on lane {lane.contract!r}"
            )
        if self.sm is not None and self.sm < self.recipe.min_sm:
            raise RecipeError(
                f"recipe {self.recipe.name} needs sm>={self.recipe.min_sm} for its "
                f"{self.recipe.granularity[1]} kernel; this host states sm{self.sm}"
            )
        quant = plan_quant(target, self.recipe)
        return TransformPlan(
            pass_name=self.NAME,
            rewrites=quant.convert,
            removes=(self.recipe.name,),
            # A quant recipe's domain is the recipe itself: unlike a fold,
            # the rewrite is not parameterized by a served point, so there is
            # exactly one and it is named rather than left empty.
            domain=(self.recipe.name,),
            scope_bytes=scope_census(target, quant.convert)["bytes"],
        )

    def apply(
        self,
        target: Any,
        plan: TransformPlan,
        *,
        artifacts: Any = None,
    ) -> TransformReport:
        """Convert, then PROVE it converted. A half-quantized model is refused.

        Two refusals exist because the failure is otherwise silent: a
        converter that matched nothing, and a converter that matched some.
        Serving a silently-bf16 lane under an fp8 name is the defect both are
        for.
        """

        before = scope_census(target, plan.rewrites)["bytes"]
        quant = plan_quant(target, self.recipe)
        if quant.convert != plan.rewrites:
            raise RecipeError(
                f"recipe {self.recipe.name}: the plan named {len(plan.rewrites)} "
                f"module(s) but {len(quant.convert)} are convertible now; the "
                f"module tree moved between plan and apply"
            )
        self.convert(target, quant)
        unconverted = [
            fqn for fqn in plan.rewrites if not is_resident(target.get_submodule(fqn))
        ]
        if len(unconverted) == len(plan.rewrites):
            raise RecipeError(
                f"recipe {self.recipe.name}: the converter matched NOTHING -- all "
                f"{len(unconverted)} planned module(s) are still unquantized"
            )
        if unconverted:
            raise RecipeError(
                f"recipe {self.recipe.name}: {len(unconverted)}/{len(plan.rewrites)} "
                f"planned module(s) are still unquantized (first: "
                f"{unconverted[0]!r}); refusing to serve a half-quantized model as "
                f"{self.recipe.name}"
            )
        after = scope_census(target, plan.rewrites)["bytes"]
        return TransformReport(
            pass_name=self.NAME,
            plan=plan,
            mode=TransformMode.CONVERTED,
            freed_bytes=max(before - after, 0),
            cache_bytes=after,
            loaded=(),
        )


def applied_lane_row(
    component: str, recipe: QuantRecipe, plan: QuantPlan
) -> dict[str, Any]:
    """The structured fact an emitter turns into ``report_applied_lane``.

    torchcg returns it; only gen-worker emits it (charter rule 2). Converting
    at INGEST makes this attribution a property of the BYTES, which is the
    real fix -- this row is what the fallback arm reports in the meantime,
    and what a byte census reads back.
    """

    return {
        "component": component,
        "lane": recipe.name,
        "scheme": recipe.scheme,
        "granularity": list(recipe.granularity),
        "modules": len(plan.convert),
        "kept": {reason: len(rows) for reason, rows in plan.kept_by_reason.items()},
    }


def scope_census(module: Any, fqns: Sequence[str]) -> dict[str, Any]:
    """Bytes and dtype labels for a scope, SEEN THROUGH tensor subclasses.

    The measurement that makes an fp8 claim checkable. A naive
    ``numel * element_size`` prices per-row fp8 as bf16 -- the row a reader
    reads as "no fp8 here" -- so both numbers come from ``tensor_bytes`` /
    ``dtype_label``, which flatten a subclass into its real components.
    A module whose weight is not a tensor at all falls back to its whole
    parameter+buffer census rather than reporting nothing.
    """

    rows: dict[str, dict[str, Any]] = {}
    for fqn in fqns:
        child = module.get_submodule(fqn)
        weight = getattr(child, "weight", None)
        if weight is None or not hasattr(weight, "numel"):
            rows[fqn] = {"bytes": module_bytes(child), "dtype": "?"}
            continue
        rows[fqn] = {"bytes": tensor_bytes(weight), "dtype": dtype_label(weight)}
    return {
        "modules": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows.values()),
        "dtypes": sorted({str(row["dtype"]) for row in rows.values()}),
    }


__all__ = [
    "KEPT_DTYPE",
    "KEPT_KIND",
    "KEPT_OUTSIDE",
    "KEPT_SKIPPED",
    "KEPT_UNALIGNED",
    "KeptModule",
    "ModuleSelect",
    "QuantPlan",
    "QuantRecipe",
    "RecipeError",
    "RecipeQuantize",
    "applied_lane_row",
    "is_resident",
    "plan_quant",
    "scope_census",
]
