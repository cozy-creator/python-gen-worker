"""The authoring vocabulary: what a family DECLARES, before anything is minted.

This is the one place a model family is described. A declaration states
class-level facts only — runners, bucket axes, the loop between them, the
scheduler block, and the SHAPE of the family's tuned values. It states nothing
checkpoint-level (weight sets, refs, tuned VALUES) and nothing request-level:
those are the instance's and the request's, and DESIGN-RULINGS §4.27's
checkpoint-free layer stays checkpoint-free because the field sets here cannot
express them.

The vocabulary deliberately mirrors ``torchcg.recipe``'s: a family's runners,
axes, loop, parameters and scheduler project onto the recipe document one for
one, so the mint-emitted recipe and this declaration are two readings of one
composition and ``Recipe.assert_declaration`` can compare them (torchcg G16).
What does NOT appear here is anything the recipe refuses to carry.

Two tiers, and the split is type-honest rather than cosmetic:

* :class:`ModelSpec` is eager-only — a constructor and, when the model has an
  inference vocabulary, a tuned schema. **AMENDED by Paul 2026-08-17 ("the
  eager tier is PERMANENT"): the tier is transitional only for PYTORCH-class
  models**, which can be traced and therefore owe graph classes eventually. A
  model served by an external binary or a non-PyTorch runtime — vLLM,
  llama.cpp, a bespoke engine — is an eager-PERMANENT citizen with NO graph
  obligation ever, declared through the identical ``models={...}`` surface.
  pgw#1326's "LLM/VLM families are `GraphModelSpec` citizens with sessions"
  binds only where a traceable graph exists.
* :class:`GraphModelSpec` adds the runners, the loop and the scheduler, which is
  what makes a recipe, typed bindings, and a compiled backing possible.

Both are authored identically at the call site, and an author-defined inline
family uses this exact shape — which is what makes catalog promotion a copy
rather than a rewrite (pgw#1326 "One SDK shape").
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Final

import msgspec

from .._vendor.torchcg.recipe import (
    BucketAxisName,
    FamilyName,
    LoopKind,
    ParameterName,
    RecipeError,
    RepeatKind,
    RunnerName,
    SchedulerValue,
    SessionState,
    parse_bucket_axis_name,
    parse_family_name,
    parse_layout_contract,
    parse_parameter_name,
    parse_runner_name,
    parse_scheduler_name,
)
from ..families.base import KIND_CHECKPOINT, KIND_LORA, GenerationDefaults, register_family
from ..models.tensor_layout_contract import (
    LayoutDeclarationError,
    normalize_layout_demand,
    normalize_layout_requirements,
    normalize_layout_undeclarable,
)
from .errors import ModelError, ModelRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .runtime import Model


#: The default tensor-layout contract a runner is traced against when the
#: declaration names none. It is a TOKEN this package records and never
#: interprets: the COMPATIBLE / CONVERTIBLE / PRODUCIBLE / INCOMPATIBLE verdict
#: is the hub's join of the recipe with per-checkpoint layout metadata
#: (DESIGN-RULINGS §1.32/§1.33, torchcg G15).
DEFAULT_LAYOUT: Final = "bf16"


def _identifier(kind: str, value: object, parse: Callable[[object], Any]) -> str:
    """Parse one author-facing handle through torchcg's frozen grammar.

    Codegen owns no escaping rule (torchcg G1), so a name a generator would
    have to mangle is refused HERE — at module import, on the author's own
    machine — instead of at generation time or, worse, on a pod.
    """

    try:
        return str(parse(value))
    except RecipeError as exc:
        raise ModelError(
            ModelRefusal.IDENTIFIER_INVALID,
            f"{kind} {value!r} is not a legal generated symbol: {exc}",
        ) from exc


@dataclass(frozen=True, slots=True)
class Bucket:
    """One family-wide shape axis and the closed set of values it may take.

    Buckets are the reason a generated ``Literal`` is exhaustive: every runner
    that declares this axis must have a variant at every one of these values,
    at every layout it offers (torchcg G6/G15), so a selection cannot address a
    class that does not exist.
    """

    name: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _identifier("bucket axis", self.name, parse_bucket_axis_name)
        )
        values = tuple(self.values)
        if not values or any(type(value) is not int or value <= 0 for value in values):
            raise ModelError(
                ModelRefusal.BUCKET_AXIS_INVALID,
                f"bucket axis {self.name!r} must declare positive integer values",
            )
        if values != tuple(sorted(set(values))):
            raise ModelError(
                ModelRefusal.BUCKET_AXIS_INVALID,
                f"bucket axis {self.name!r} values must be sorted and unique, got {list(values)!r}",
            )
        object.__setattr__(self, "values", values)

    @property
    def axis(self) -> BucketAxisName:
        return BucketAxisName(self.name)


@dataclass(frozen=True, slots=True)
class CallExample:
    """One concrete call a runner is EXPORTED against.

    ``params`` is the ordered parameter axis of the call — the generated
    callable's parameters are exactly these, in this order (torchcg G3), so it
    is declared rather than recovered from a signature: a ``**kwargs``-shaped
    forward would otherwise decide the binding's public shape by accident.

    ``excluded`` names inputs this class REFUSES. A class that excludes an
    input the call carries must not serve it — it would return the base model
    and look correct (``ingress_selection_v1``'s ``input_excluded``).

    ``dynamic`` is handed to ``torch.export.export(dynamic_shapes=...)``
    UNCHANGED. This package owns no shape DSL: a symbolic axis is expressed in
    Torch's own vocabulary, exported, and read back out of the program's range
    constraints — so a symbol carries the bounds the exporter actually solved
    for, never the bounds an author asserted beside it.
    """

    params: tuple[str, ...]
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] = field(default_factory=dict)
    excluded: tuple[str, ...] = ()
    dynamic: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        params = tuple(str(name) for name in self.params)
        if not params or len(set(params)) != len(params):
            raise ModelError(
                ModelRefusal.CLASS_INVALID,
                "a call example must declare a non-empty, unique parameter axis",
            )
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "excluded", tuple(sorted(set(self.excluded))))


#: Builds one runner's eager module for one tensor-layout contract. It is the
#: SAME constructor the mint traces and the eager backing serves — pgw#1326's
#: dual backing is one declaration read two ways, never two code paths.
#:
#: Author-facing callables speak plain ``str``/``int``, NOT the document's
#: ``NewType``s. The newtypes exist so a value parsed out of a document cannot
#: be confused with an arbitrary string; a declaration author is writing the
#: literals, so making them import a newtype to spell one would be ceremony
#: that buys nothing and (because ``Mapping``'s key is invariant) would refuse
#: every naturally-written signature.
BuildFn = Callable[[str], Any]

#: Builds the example call one (bucket, layout) variant is exported against.
ExampleFn = Callable[[Mapping[str, int], str], CallExample]


@dataclass(frozen=True, slots=True)
class Runner:
    """One author-facing runner handle: a named graph class family.

    ONE runner generates ONE typed binding (torchcg G2). Its variants — one per
    (bucket, layout) — may differ only in concrete dimensions and symbol
    bounds; a trace whose layout changes the CALL rather than the weights is a
    different runner, not another variant of this one.
    """

    name: str
    build: BuildFn
    example: ExampleFn
    axes: tuple[str, ...] = ()
    layouts: tuple[str, ...] = (DEFAULT_LAYOUT,)
    #: Where this runner LIVES in a loaded component tree, as a dotted
    #: attribute path (``"transformer"``, ``"vae.decode"``, ``"text_encoder_2"``).
    #:
    #: pgw#1346. ``build`` constructs a WEIGHTLESS module from config, which is
    #: what the mint traces; it cannot serve a request because it has no
    #: weights. Serving eagerly means reaching the weight-bearing module the
    #: loader already produced, and this is the only thing that says which one
    #: that is. Without it a declaration can be minted and type-checked but not
    #: served eagerly, which is exactly the gap that blocked the `Slot`
    #: deletion.
    #:
    #: Empty means "this runner has no eager equivalent" — a legitimate state
    #: for a runner that only ever exists as a compiled graph class. Such a
    #: runner simply does not appear in the eager backing, and a call to it
    #: with no armed cell refuses by name rather than running the wrong thing.
    component: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier("runner", self.name, parse_runner_name))
        if not callable(self.build) or not callable(self.example):
            raise ModelError(
                ModelRefusal.CLASS_INVALID,
                f"runner {self.name!r} needs a build() and an example() callable",
            )
        axes = tuple(
            _identifier("bucket axis", name, parse_bucket_axis_name) for name in self.axes
        )
        if axes != tuple(sorted(set(axes))):
            raise ModelError(
                ModelRefusal.BUCKET_INVALID,
                f"runner {self.name!r} axes must be sorted and unique, got {list(axes)!r}",
            )
        object.__setattr__(self, "axes", axes)
        layouts = tuple(
            _identifier("layout contract", name, parse_layout_contract) for name in self.layouts
        )
        if not layouts or layouts != tuple(sorted(set(layouts))):
            raise ModelError(
                ModelRefusal.CLASS_INVALID,
                f"runner {self.name!r} layouts must be a non-empty sorted unique set",
            )
        object.__setattr__(self, "layouts", layouts)
        component = str(self.component or "").strip()
        if component and not all(
            part.isidentifier() for part in component.split(".")
        ):
            raise ModelError(
                ModelRefusal.CLASS_INVALID,
                f"runner {self.name!r} component={self.component!r} must be a dotted "
                "attribute path in the loaded component tree, e.g. 'transformer' or "
                "'vae.decode'",
            )
        object.__setattr__(self, "component", component)

    @property
    def handle(self) -> RunnerName:
        return RunnerName(self.name)

    def buckets(
        self, axes: Mapping[str, tuple[int, ...]]
    ) -> tuple[dict[str, int], ...]:
        """Every bucket this runner must have a variant for, in canonical order.

        Total by construction: the cross product of its declared axes' values.
        A generated closed type is exhaustive exactly because this set is, and
        a gap is refused before anything is exported.
        """

        rows = tuple(axes[name] for name in self.axes)
        return tuple(
            dict(zip(self.axes, combination, strict=True))
            for combination in product(*rows)
        )


@dataclass(frozen=True, slots=True)
class Stage:
    """One ordered step of a family's loop: a runner, and how often it runs."""

    runner: str
    repeat: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "runner", _identifier("runner", self.runner, parse_runner_name))
        if self.repeat:
            object.__setattr__(
                self, "repeat", _identifier("parameter", self.repeat, parse_parameter_name)
            )

    @property
    def kind(self) -> RepeatKind:
        return RepeatKind.COUNTED if self.repeat else RepeatKind.ONCE


@dataclass(frozen=True, slots=True)
class Loop:
    """The composition's loop, or the declaration that the host owns it.

    ``staged`` is what the vocabulary fully describes. ``host`` is an
    autoregressive family: the declaration states the per-step runners in order
    and who owns the state threaded between them, and says outright that the
    iteration is host code. A repeat count under a host loop is REFUSED —
    a fabricated bound reads to a second implementation as a real one.
    """

    stages: tuple[Stage, ...]
    kind: LoopKind = LoopKind.STAGED
    session_state: SessionState = SessionState.NONE

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ModelError(ModelRefusal.LOOP_INVALID, "a loop must declare its stages")
        if self.kind is LoopKind.HOST and any(stage.repeat for stage in stages):
            raise ModelError(
                ModelRefusal.LOOP_INVALID,
                "a host-owned loop states its per-step runners, never a repeat count: the "
                "iteration is data-dependent and belongs to the host",
            )
        object.__setattr__(self, "stages", stages)


@dataclass(frozen=True, slots=True)
class Parameter:
    """One integer count a counted stage reads, with inclusive bounds."""

    name: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier("parameter", self.name, parse_parameter_name))
        if type(self.minimum) is not int or type(self.maximum) is not int:
            raise ModelError(
                ModelRefusal.PARAMETER_INVALID,
                f"parameter {self.name!r} bounds must be integers",
            )
        if self.minimum < 1 or self.maximum < self.minimum:
            raise ModelError(
                ModelRefusal.PARAMETER_INVALID,
                f"parameter {self.name!r} bounds must satisfy 1 <= minimum <= maximum",
            )

    @property
    def handle(self) -> ParameterName:
        return ParameterName(self.name)


@dataclass(frozen=True, slots=True)
class Scheduler:
    """One named scheduler KIND and its finite-scalar parameter block.

    Neither this package nor torchcg interprets it: the host implements the
    named scheduler. It rides in the declaration because two otherwise
    identical compositions under different schedulers are different pipelines,
    so it must be inside the digest a consumer pins.

    **A family declares a SET of these, keyed by the sampler name its tuned
    schema admits** — ``GraphModelSpec.schedulers`` (pgw#1346 K10). The sampler
    is a CHECKPOINT fact, stamped per release slot into ``inst.tuned.scheduler``
    and not a family constant, so a single-valued field could only ever be the
    family's trained schedule while every request asked for something else:
    SDXL's declared kind was ``euler_discrete`` while its default sampler is
    ``euler_a``. A family with one sampler declares a set of one and nothing
    about it changes.
    """

    name: str
    parameters: Mapping[str, SchedulerValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier("scheduler", self.name, parse_scheduler_name))
        rows: dict[str, SchedulerValue] = {}
        for raw_name, value in self.parameters.items():
            parsed = _identifier("scheduler parameter", raw_name, parse_parameter_name)
            if isinstance(value, bool) or isinstance(value, (int, str)):
                rows[parsed] = value
                continue
            if isinstance(value, float) and value == value and value not in (
                float("inf"),
                float("-inf"),
            ):
                rows[parsed] = value
                continue
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"scheduler parameter {parsed!r} must be a finite JSON scalar, got {value!r}",
            )
        object.__setattr__(self, "parameters", dict(sorted(rows.items())))


class TunedValues(GenerationDefaults, frozen=True, kw_only=True):
    """Base for a family's TUNED-VALUE schema — the shape, never the values.

    The schema is code and the values are catalog: a release slot stamps one
    resolved document per checkpoint and the SDK decodes it against this
    struct, exactly as th#1116 works today. What pgw#1332 changes is the
    DELIVERY ADDRESS — the decoded struct rides the instance as ``inst.tuned``
    instead of the request context, because it is a checkpoint-level fact and
    the request is not the axis it varies on.

    Declared ON the family (``Flux1Dev.Tuned``), which is how the naming
    collision with the old free-standing ``@family("...")`` decorator resolves:
    a family owns its schema, so nothing has to register a vocabulary under a
    name a second, unrelated struct could also claim.

    Inherits ``forbid_unknown_fields`` from :class:`GenerationDefaults`, so the
    exported JSON Schema stays closed (``additionalProperties: false``) — the
    contract tensorhub validates repo metadata against.
    """


def _tuned_schema(
    name: str, tuned: type[GenerationDefaults] | None, what: str
) -> type[GenerationDefaults]:
    if tuned is None:
        raise ModelError(
            ModelRefusal.TUNED_INVALID,
            f"family {name!r} declares no {what} schema; a family owns its tuned shape",
        )
    if not (isinstance(tuned, type) and issubclass(tuned, GenerationDefaults)):
        raise ModelError(
            ModelRefusal.TUNED_INVALID,
            f"family {name!r} {what} schema must be a TunedValues subclass, got {tuned!r}",
        )
    # `schema_version` comes from the base and carries no tuned value, so a
    # schema declaring only it is an empty struct nobody can stamp — an
    # authoring mistake that would otherwise surface as silently-missing
    # catalog values at serve time rather than at import.
    own = tuple(
        row.name for row in msgspec.structs.fields(tuned) if row.name != "schema_version"
    )
    if not own:
        raise ModelError(
            ModelRefusal.TUNED_INVALID,
            f"family {name!r} {what} schema {tuned.__name__} declares no tuned fields",
        )
    return tuned


@dataclass(frozen=True)
class ModelSpec:
    """An EAGER-ONLY model: a constructor, and a tuned schema when it has one.

    The day-0 escape hatch, made type-honest rather than left implicit. A model
    nobody has declared graph classes for is still authored through the one
    contract — ``@endpoint(models={...})`` — so promoting it later is adding
    runners and a loop, never rewriting the endpoint (pgw#1326 "One SDK
    shape"). It has no recipe, no bindings and no compiled backing.

    **The tier is PERMANENT, not a waiting room** (Paul, 2026-08-17). It is
    transitional only for PyTorch-class models, which owe graph classes because
    they can be traced at all. Two populations live here for good:

    * models served by an EXTERNAL BINARY or a non-PyTorch runtime — vLLM,
      llama.cpp, a bespoke engine. There is no graph to declare, so there is no
      obligation to declare one;
    * AUXILIARY models an endpoint loads beside its checkpoint — a frame
      interpolator, a latent upsampler, a tokenizer tree. These are the K5
      class of pgw#1346's W2 plan, and they belong HERE rather than in a second
      non-model spelling. They carry no inference vocabulary, which is why
      ``tuned`` is optional.

    **The layout axes are the model's, not the endpoint's** (pgw#1346 K1/K2/K4).
    ``layouts`` / ``layouts_undeclarable`` / ``layout_requirements`` move here
    from the retired ``Slot`` unchanged — same keys, same normalizers, same
    refusals — because what bytes a model's CODE can execute, and what
    executing them needs of the machine, are properties of the code. Two
    endpoints binding one model state one demand. The three endpoint-COUPLED
    axes (`selected_by`, `default_checkpoint`, `root`) went the other way, onto
    :class:`~gen_worker.model.bind.Bind`, because they name a payload field and
    a deploy rather than the model.

    ``layouts`` stays keyed by COMPONENT PATH rather than by runner (K4). A
    ``GraphModelSpec``'s runners and its component tree are not 1:1 — `SDXL`
    declares two runners over a four-component tree — so collapsing the demand
    onto ``Runner.layouts`` would make a per-text-encoder demand inexpressible.
    ``Runner.layouts`` is a different axis and keeps its meaning: which layouts
    that GRAPH CLASS has traced variants for.
    """

    name: str
    tuned: type[GenerationDefaults] | None = None
    build: Callable[[], Any] | None = None
    lora_tuned: type[GenerationDefaults] | None = None
    on_load: Callable[[Model], None] | None = None
    runners: tuple[Runner, ...] = ()
    layouts: Mapping[str, Sequence[str]] | None = None
    layouts_undeclarable: str = ""
    layout_requirements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate EVERYTHING first, register last. Registration is a
        # process-global side effect, and a declaration that raises must not
        # leave its name claimed behind it — the next import of a corrected
        # declaration would then read as a collision it cannot adjudicate.
        # Subclasses extend `_validate`, never this, so the ordering holds for
        # them too.
        self._validate()
        self._register()

    def _validate(self) -> None:
        object.__setattr__(self, "name", _identifier("family", self.name, parse_family_name))
        if self.tuned is not None:
            object.__setattr__(self, "tuned", _tuned_schema(self.name, self.tuned, "tuned"))
        elif self.lora_tuned is not None:
            raise ModelError(
                ModelRefusal.TUNED_INVALID,
                f"model {self.name!r} declares lora_tuned= without tuned=. A LoRA "
                "vocabulary refines a base one; there is none here to refine.",
            )
        self._validate_layouts()
        if self.lora_tuned is not None:
            object.__setattr__(
                self, "lora_tuned", _tuned_schema(self.name, self.lora_tuned, "lora tuned")
            )
        if self.on_load is not None and not callable(self.on_load):
            raise ModelError(
                ModelRefusal.FAMILY_INVALID, f"family {self.name!r} on_load must be callable"
            )
        if type(self) is ModelSpec and self.runners:
            raise ModelError(
                ModelRefusal.FAMILY_INVALID,
                f"family {self.name!r} declares runners but is eager-only; declare a "
                "GraphModelSpec() when the composition has graph classes",
            )

    def _validate_layouts(self) -> None:
        """The three layout axes, with ``Slot``'s own normalizers (pgw#1346).

        Deliberately the SAME functions, not a re-implementation: the ie#740
        production floors — sdxl's ``vram12g``, krea-2's ``vram72g``,
        minimax-h3's th#1754 ``vram78g``, flux.1-schnell's ``vram36g`` — migrate
        onto this declaration BY VALUE, and a second parser is how a floor
        silently stops meaning what the incident made it mean.
        """

        where = f"{type(self).__name__}({self.name!r})"
        if self.layouts is not None and str(self.layouts_undeclarable or "").strip():
            raise LayoutDeclarationError(
                f"{where}: layouts= and layouts_undeclarable= are mutually exclusive "
                "— a model either names the contracts its code executes or says why "
                "none is nameable, never both."
            )
        object.__setattr__(
            self, "layouts",
            None if self.layouts is None
            else normalize_layout_demand(self.layouts, where=where),
        )
        object.__setattr__(
            self, "layouts_undeclarable",
            "" if self.layouts_undeclarable == ""
            else normalize_layout_undeclarable(self.layouts_undeclarable, where=where),
        )
        if not self.layout_requirements:
            object.__setattr__(self, "layout_requirements", {})
        elif self.layouts is None:
            raise LayoutDeclarationError(
                f"{where}: layout_requirements= without layouts=. A requirement "
                "guards a declared contract; there is none here to guard."
            )
        else:
            object.__setattr__(
                self, "layout_requirements",
                normalize_layout_requirements(
                    self.layout_requirements, where=where,
                    accepted={h for hs in self.layouts.values() for h in hs},
                ),
            )

    def _register(self) -> None:
        """Publish this family's tuned schema(s) under its own name.

        The registration the retired ``@family("sdxl")`` decorator used to do,
        moved onto the object that owns the schema. The registry itself is
        unchanged — tensorhub still validates repo metadata against the
        exported JSON Schema and ``gen-worker families export-schemas`` still
        finds it — but a family can no longer be registered by a struct that
        does not belong to one, which is the collision Paul ruled on.
        """

        if self.tuned is None:
            # An auxiliary or external-runtime model has no inference
            # vocabulary, so there is no schema to publish and nothing for
            # tensorhub to validate repo metadata against. Registering an empty
            # one would put a name into the hub's vocabulary that answers no
            # question — pgw#1346 K8: only a model with tuned values reaches it.
            return
        try:
            register_family(self.name, self.tuned, kind=KIND_CHECKPOINT)
            if self.lora_tuned is not None:
                register_family(self.name, self.lora_tuned, kind=KIND_LORA)
        except ValueError as exc:
            raise ModelError(ModelRefusal.FAMILY_DUPLICATE, str(exc)) from exc

    @property
    def handle(self) -> FamilyName:
        return FamilyName(self.name)


@dataclass(frozen=True)
class GraphModelSpec(ModelSpec):
    """A family with declared graph classes: recipe, bindings, dual backing.

    Everything :class:`ModelSpec` is, plus the composition — which runners make
    the pipeline, the buckets they vary on, the loop between them, and the
    scheduler that loop runs under. That is exactly ``recipe_v1``'s vocabulary,
    so this declaration and the mint-emitted recipe are comparable documents
    and drift between them is an assertion rather than a hope (torchcg G16).
    """

    buckets: tuple[Bucket, ...] = ()
    loop: Loop | None = None
    parameters: tuple[Parameter, ...] = ()
    #: The scheduler SET, keyed by the SAMPLER NAME a checkpoint is stamped
    #: with — i.e. by a value of this family's ``tuned.scheduler`` field
    #: (pgw#1346 K10). Empty when the family declares no scheduler at all.
    schedulers: Mapping[str, Scheduler] = field(default_factory=dict)

    def _validate(self) -> None:
        ModelSpec._validate(self)
        self._validate_schedulers()
        if self.tuned is None:
            raise ModelError(
                ModelRefusal.TUNED_INVALID,
                f"graph model {self.name!r} declares no tuned= schema. `tuned` is "
                "optional only on the EAGER tier, where an auxiliary or "
                "external-runtime model genuinely has no inference vocabulary; a "
                "declared graph serves generation requests and its parameters are "
                "exactly what a tuned schema names.",
            )
        axis_names = tuple(bucket.name for bucket in self.buckets)
        if axis_names != tuple(sorted(set(axis_names))):
            raise ModelError(
                ModelRefusal.BUCKET_AXIS_INVALID,
                f"family {self.name!r} bucket axes must be sorted and unique",
            )
        if not self.runners:
            raise ModelError(
                ModelRefusal.FAMILY_INVALID,
                f"graph family {self.name!r} declares no runners; declare a plain ModelSpec() "
                "for an eager-only model rather than an empty graph family",
            )
        runner_names = tuple(runner.name for runner in self.runners)
        if runner_names != tuple(sorted(set(runner_names))):
            raise ModelError(
                ModelRefusal.FAMILY_INVALID,
                f"family {self.name!r} runners must be sorted by name and unique, "
                f"got {list(runner_names)!r}",
            )
        declared = {bucket.name for bucket in self.buckets}
        for runner in self.runners:
            unknown = sorted(set(runner.axes) - declared)
            if unknown:
                raise ModelError(
                    ModelRefusal.BUCKET_INVALID,
                    f"runner {runner.name!r} buckets on axis {unknown[0]!r}, which family "
                    f"{self.name!r} does not declare",
                )
        if self.loop is None:
            raise ModelError(
                ModelRefusal.LOOP_INVALID,
                f"graph family {self.name!r} declares no loop; a composition with no stated "
                "order is not a composition",
            )
        staged = {stage.runner for stage in self.loop.stages}
        unknown_runner = sorted(staged - set(runner_names))
        if unknown_runner:
            raise ModelError(
                ModelRefusal.LOOP_INVALID,
                f"loop stages runner {unknown_runner[0]!r}, which family {self.name!r} "
                "does not declare",
            )
        unused = sorted(set(runner_names) - staged)
        if unused:
            raise ModelError(
                ModelRefusal.LOOP_INVALID,
                f"runner {unused[0]!r} is declared but no loop stage runs it",
            )
        parameter_names = tuple(parameter.name for parameter in self.parameters)
        if parameter_names != tuple(sorted(set(parameter_names))):
            raise ModelError(
                ModelRefusal.PARAMETER_INVALID,
                f"family {self.name!r} parameters must be sorted by name and unique",
            )
        counted = {stage.repeat for stage in self.loop.stages if stage.repeat}
        unknown_parameter = sorted(counted - set(parameter_names))
        if unknown_parameter:
            raise ModelError(
                ModelRefusal.PARAMETER_INVALID,
                f"a counted stage reads undeclared parameter {unknown_parameter[0]!r}",
            )
        idle = sorted(set(parameter_names) - counted)
        if idle:
            raise ModelError(
                ModelRefusal.PARAMETER_INVALID,
                f"parameter {idle[0]!r} is declared but no counted stage reads it",
            )

    def _validate_schedulers(self) -> None:
        """The scheduler set: sampler-keyed, sorted, and never silently empty.

        The KEY is a sampler name — a value the family's tuned schema admits —
        and not a scheduler kind, which is why two keys may name the same kind
        with different blocks (``euler`` and ``euler_trailing`` are one class
        under two spacings) and why one key may not name two.
        """

        rows: dict[str, Scheduler] = {}
        for raw_name, scheduler in self.schedulers.items():
            sampler = _identifier("sampler", raw_name, parse_parameter_name)
            if not isinstance(scheduler, Scheduler):
                raise ModelError(
                    ModelRefusal.SCHEDULER_INVALID,
                    f"family {self.name!r} sampler {sampler!r} must name a Scheduler(...), "
                    f"got {scheduler!r}",
                )
            rows[sampler] = scheduler
        object.__setattr__(self, "schedulers", dict(sorted(rows.items())))

    @property
    def axis_values(self) -> dict[str, tuple[int, ...]]:
        return {bucket.name: bucket.values for bucket in self.buckets}

    @property
    def staged_loop(self) -> Loop:
        """The declared loop, narrowed for readers that already validated it."""

        assert self.loop is not None  # __post_init__ refuses a loopless graph family
        return self.loop

    def runner(self, name: str) -> Runner:
        """Resolve one runner handle.

        This exists for the GENERATOR and for diagnostics. A handler that
        reaches a runner through a string it typed is exactly what the
        generated bindings make unrepresentable (torchcg G7).
        """

        wanted = _identifier("runner", name, parse_runner_name)
        for runner in self.runners:
            if runner.name == wanted:
                return runner
        raise ModelError(
            ModelRefusal.FAMILY_INVALID, f"family {self.name!r} declares no runner {wanted!r}"
        )

    def variants(self) -> tuple[tuple[Runner, dict[str, int], str], ...]:
        """Every (runner, bucket, layout) this family must export, in order.

        Total per layout by construction, which is what makes a generated
        ``Literal`` exhaustive and every selection resolve (torchcg G6/G15).
        """

        axes = self.axis_values
        rows: list[tuple[Runner, dict[str, int], str]] = []
        for runner in self.runners:
            for layout in runner.layouts:
                for bucket in runner.buckets(axes):
                    rows.append((runner, bucket, layout))
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class BucketMap:
    """A typed mapping from ONE endpoint payload axis onto ONE family bucket.

    Greenfield B5: the product grid is not the family's buckets. Aspect ratios,
    megapixel tiers and pricing brackets are ENDPOINT policy; a token-count or
    resolution graph class is FAMILY truth, and conflating them is how a
    product decision quietly became a compile axis.

    Declaring the mapping makes three things true at once: a payload value that
    targets a bucket the family lacks fails the BUILD (not a pod), the handler
    reaches its bucket through a typed lookup rather than arithmetic, and the
    boot warm plan is DERIVED from the mapping instead of guessed by a
    snap-to-nearest hack.
    """

    family: GraphModelSpec
    axis: str
    mapping: Mapping[str, int]

    def __post_init__(self) -> None:
        axis = _identifier("bucket axis", self.axis, parse_bucket_axis_name)
        declared = self.family.axis_values.get(axis)
        if declared is None:
            raise ModelError(
                ModelRefusal.GRID_INVALID,
                f"family {self.family.name!r} declares no bucket axis {axis!r}; it has "
                f"{sorted(self.family.axis_values)!r}",
            )
        rows = {str(key): int(value) for key, value in self.mapping.items()}
        if not rows:
            raise ModelError(
                ModelRefusal.GRID_INVALID, f"bucket map for axis {axis!r} is empty"
            )
        missing = sorted(set(rows.values()) - set(declared))
        if missing:
            raise ModelError(
                ModelRefusal.GRID_INVALID,
                f"bucket map targets {axis}={missing[0]}, which family "
                f"{self.family.name!r} does not declare (it has {list(declared)!r})",
            )
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "mapping", dict(sorted(rows.items())))

    def bucket_for(self, value: str) -> int:
        """The family bucket one product-grid value maps onto, or a refusal."""

        try:
            return self.mapping[str(value)]
        except KeyError:
            raise ModelError(
                ModelRefusal.GRID_INVALID,
                f"no bucket declared for {self.axis}={value!r}; the map covers "
                f"{sorted(self.mapping)!r}",
            ) from None

    def warm_plan(self) -> tuple[int, ...]:
        """The distinct buckets a boot warm-up must cover, ascending.

        Derived from the mapping, so a product-grid entry nobody can serve is
        impossible by construction and the warm plan cannot drift from the grid
        it exists to warm.
        """

        return tuple(sorted(set(self.mapping.values())))


def declared_model_specs(objects: Iterable[object]) -> tuple[ModelSpec, ...]:
    """Every :class:`ModelSpec` in ``objects``, deduplicated and sorted by name."""

    rows: dict[str, ModelSpec] = {}
    for item in objects:
        if isinstance(item, ModelSpec):
            rows[item.name] = item
    return tuple(rows[name] for name in sorted(rows))


__all__ = [
    "DEFAULT_LAYOUT",
    "Bucket",
    "LoopKind",
    "SessionState",
    "BucketMap",
    "BuildFn",
    "CallExample",
    "ExampleFn",
    "ModelSpec",
    "GraphModelSpec",
    "Loop",
    "Parameter",
    "Runner",
    "Scheduler",
    "Stage",
    "TunedValues",
    "declared_model_specs",
]
