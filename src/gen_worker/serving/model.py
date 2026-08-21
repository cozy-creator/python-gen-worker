from __future__ import annotations

import ast
import inspect
import textwrap
import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from .lane_spec import (
    DYNAMIC,
    STATIC,
    DeclaredLane,
    LaneSpec,
    Structural,
    lane,
    parse_shapes,
    parse_structural,
)

if TYPE_CHECKING:
    from .context import LoadContext

MT = TypeVar("MT")

MODEL_TYPE_ATTR = "__cozy_model_type__"
LANES_ATTR = "__cozy_lanes__"
REQUIRES_ATTR = "__cozy_requires__"
DECLARED_LANES_ATTR = "__cozy_declared_lanes__"
STRUCTURAL_ATTR = "__cozy_structural__"
SHAPES_ATTR = "__cozy_shapes__"
SELF_LOADING_ATTR = "__cozy_self_loading__"


class ModelDeclarationError(TypeError):
    """A model class header does not state a valid declaration."""


def lane_handle(lane: Any) -> str:
    """The lane's stable string handle — the WIRE rendering of its stamp pair.

    pgw#1621: a lane used to be a tensorfs v1 ``Contract`` OBJECT, and this
    function was four fallbacks deep because that object spelled its handle
    four different ways (``stamp``, ``contract``, ``name``+``version``, and a
    BARE 64-hex ``digest`` that read as a handle and was not one — it once
    produced ``f1455f56…`` where ``sdxl.diffusers-bf16@1`` belonged). A lane is
    now the ``(topology, quant)`` pair and there is exactly one rendering of
    it, so the fallbacks are gone with the ambiguity that needed them.

    Accepts a :class:`DeclaredLane`, a ``LayoutId``, the rendered string, or
    the author's two-tuple — every spelling of the same identity that reaches
    this repo — and answers ``"<topology>+<quant>"``. That string is shared
    with the hub's ``tensorfs.LayoutID.String`` and with the derived-artifact
    CAS address; a drift is a fork.
    """

    from ..models.tensor_layout_contract import LayoutDeclarationError, parse_lane_stamp

    rendered = getattr(lane, "contract_id", None)
    if isinstance(rendered, str) and rendered:
        return rendered
    try:
        return parse_lane_stamp(lane, where="lane_handle").render()
    except LayoutDeclarationError as exc:
        raise ModelDeclarationError(str(exc)) from exc


def lane_dtype(lane: Any, *, where: str) -> str:
    """The lane's load dtype — the QUANT RULE's ``declared_dtype``.

    pgw#1621 deleted the refusal that used to live here, and deleted it by
    making the failure inexpressible rather than by relaxing the check.

    Under v1 the dtype was a top-level field on a per-lane document that could
    simply be absent: ``sdxl.diffusers-nvfp4-flat@1`` shipped without one, a
    dtype-less lane cleared declaration through a ``runtime_checkable``
    Protocol that never invokes its own getter, and it died at load on a rented
    pod. That cost a waiver set (``DTYPELESS_UPSTREAM_LANES``), a fence test
    over the vendored corpus, and a self-deleting-waiver test to make sure the
    waiver could not outlive its reason.

    A v2 quant rule cannot lack a dtype. ``declared_dtype`` is required by the
    rule schema, the corpus is conformance-checked against
    ``spec/v2/vectors/digests.json``, and there are EIGHT rules for the whole
    fleet rather than one document per lane. So the answer is a read with no
    arm for absence — and a handle the corpus does not carry refuses in
    :func:`~gen_worker.models.tensor_layout_contract.rule_dtype` before it ever
    gets here.
    """

    from ..models.tensor_layout_contract import LayoutDeclarationError, rule_dtype

    quant = getattr(lane, "quant", None)
    if not isinstance(quant, str) or not quant:
        from ..models.tensor_layout_contract import parse_lane_stamp

        try:
            quant = parse_lane_stamp(lane, where=where).quant
        except LayoutDeclarationError as exc:
            raise ModelDeclarationError(str(exc)) from exc
    try:
        return rule_dtype(quant)
    except LayoutDeclarationError as exc:
        raise ModelDeclarationError(f"{where}: {exc}") from exc


def _calls_ctx_compile(fn: Any) -> bool:

    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - dedent of a nested def
        return False
    definition = tree.body[0] if tree.body else None
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return load_marks_compile(definition)


def load_marks_compile(definition: Any) -> bool:
    """Whether this ``load`` DEFINITION marks a compile target — the AST half of :func:`_calls_ctx_compile`, taking the parsed node instead of a live function."""

    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    names = {argument.arg for argument in definition.args.args[1:2]}
    for node in ast.walk(definition):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
        ):
            return True
    return False


def _parse_lanes(
    cls: type, lanes: Mapping[Any, Any]
) -> tuple[DeclaredLane, ...]:
    """``lanes={(topology, quant): lane(...)}`` -> the READ ``DeclaredLane``s.

    ONE declaration, and it is the class's whole memory story. The mapping
    KEY is the ``(topology, quant)`` STAMP PAIR — never a name string, never
    implicit, never borrowed from the model type — and the VALUE is this
    lane's :func:`~gen_worker.serving.lane_spec.lane` declaration: its demand
    FORMULA and its optional additive residency override.

    What used to be here and is DELETED (pgw#1599): the machine-floor
    STRING (``"vram7g"``). Paul, 2026-08-20: *"the memory-requirement being
    bundled per lane is the wrong model entirely. Memory requirements vary
    per request … Furthermore, there is no required VRAM."* A number that
    stood for every request a lane would ever serve was wrong for all of
    them, so it is replaced by the formula rather than moved.

    What CHANGED (pgw#1621): the key. It was an imported tensorfs v1
    ``Contract`` object; it is now the pair, because tensor-layout v2 makes
    the digest pair identity and a v1 handle a display name. The refusals
    below are the same refusals — a bad key, a duplicate, a VRAM string, a
    non-``LaneSpec`` value — re-aimed at the new vocabulary.

    What survives, unchanged in substance: ``min_sm`` is DERIVED and an author
    who writes it by hand is refused. It is a per-LANE fact — an 8-bit lane
    needs 8-bit kernels because of what it IS — which is exactly why a
    hand-written floor could never be right for a class declaring bf16, fp8
    and nvfp4 at once (pgw#1606). It now falls out of the QUANT RULE, which is
    where the ratified number lives.
    """
    from ..models.tensor_layout_contract import (
        LayoutDeclarationError,
        capability_floor_for_rule,
        display_names,
        parse_lane_stamp,
        rule_dtype,
    )

    where = f"{cls.__qualname__}: lanes="
    if not isinstance(lanes, Mapping):
        raise ModelDeclarationError(
            f"{where} is a MAPPING of (topology, quant) -> lane(...), got "
            f"{type(lanes).__name__}. The tuple form is deleted: every lane "
            f"declares its own demand formula, so there is nothing for a bare "
            f"tuple to carry. Write "
            f'`lanes={{("sdxl.diffusers@1", "plain.bf16@1"): lane(request=…)}}`.'
        )
    if not lanes:
        raise ModelDeclarationError(
            f"{where} is EMPTY. `lanes=()` and `lanes={{}}` are deleted "
            f"(pgw#1597 ruling pair): a lane answers checkpoint COMPATIBILITY "
            f"and lane SELECTION, not just compilation, so every model class "
            f"names at least one real stamp pair. If you do not want this "
            f"model compiled, simply do not call `ctx.compile()` in `load()` "
            f"— that is the entire eager declaration."
        )

    display = display_names()
    declared: list[DeclaredLane] = []
    seen: set[str] = set()
    for key, spec in lanes.items():
        try:
            layout = parse_lane_stamp(key, where=f"{cls.__qualname__}: lanes=")
        except LayoutDeclarationError as exc:
            raise ModelDeclarationError(str(exc)) from exc
        handle = layout.render()
        site = f"{where}[{handle!r}]"
        if handle in seen:
            raise ModelDeclarationError(
                f"{site}: declared twice. One row per lane — two rows for one "
                f"pair can only disagree about its demand."
            )
        seen.add(handle)

        if isinstance(spec, str):
            raise ModelDeclarationError(
                f"{site}: the machine-floor STRING ({spec!r}) is DELETED. "
                f"Paul, 2026-08-20: *\"there is no required VRAM\"* — demand "
                f"varies per request (a 4 MP image is not a 1 MP image; an H3 "
                f"video is quadratic in its frame count), so a lane declares a "
                f"FORMULA. Write "
                f"`lane(request=const(GiB(1.2)) + per_mp_batch(MiB(220)))`, "
                f"with the terms this model actually scales on."
            )
        if not isinstance(spec, LaneSpec):
            raise ModelDeclarationError(
                f"{site}: the mapping value is `lane(request=…)`, got "
                f"{type(spec).__name__}. "
                f"`from gen_worker import lane` / "
                f"`from gen_worker.demand import const, per_mp_batch, GiB, MiB`."
            )

        # Both reads REFUSE an unratified quant rule rather than defaulting.
        # `parse_lane_stamp` has already proven the handle is in the corpus, so
        # neither can fail here — the conversion below exists so that if that
        # ever stops being true, the failure is a named declaration refusal and
        # not a KeyError from inside a lane ladder on a rented pod.
        try:
            dtype = rule_dtype(layout.quant)
            floor = capability_floor_for_rule(layout.quant)
        except LayoutDeclarationError as exc:  # pragma: no cover - see above
            raise ModelDeclarationError(f"{site}: {exc}") from exc

        declared.append(
            DeclaredLane(
                layout=layout,
                topology=str(layout.topology),
                quant=str(layout.quant),
                dtype=dtype,
                # The capability floor falls out of the RULE, never the header
                # and never the author. Two producers of one fact is how they
                # drift apart.
                min_sm=floor,
                display_name=display.get(handle, ""),
                spec=spec,
            )
        )
    return tuple(declared)


def lane_requirements(declared: DeclaredLane) -> Any:
    """The placement row for one lane: the DERIVED capability floor, or None.

    The VRAM half is gone with the strings (pgw#1599). What replaces it is
    not a second annotation but a COMPUTED number — pgw#1600 evaluates the
    lane's demand formula over the advertised shape envelope and serializes
    `weights + demand(envelope)` into the release document, which is the
    number the hub shops on (se#810). Until it does, this row states the one
    floor that IS a lane fact: an 8-bit lane needs 8-bit kernels.

    ``None`` for a lane whose RULE states no floor — `plain.f32@1` is the one
    that does, and it states its 0 explicitly. That is the honest answer and
    it must not
    be spelled as an empty requirement row: ``parse_layout_requirements``
    refuses one by name ("omit the entry rather than declaring an empty
    one"), and it is right to — an empty row reads to the resolver as a
    declared floor of nothing, which is th#1754's shape.
    """
    from ..models.tensor_layout_contract import (
        RequirementTerms,
        parse_layout_requirements,
    )

    if not declared.min_sm:
        return None
    return parse_layout_requirements(
        RequirementTerms(min_sm=declared.min_sm), where=declared.contract_id
    )


def _declared_model_type(cls: type) -> type | None:

    for base in cls.__dict__.get("__orig_bases__", ()):
        origin = typing.get_origin(base)
        if origin is None or not (isinstance(origin, type) and issubclass(origin, Model)):
            continue
        args = typing.get_args(base)
        if len(args) != 1:
            continue
        candidate = args[0]
        if isinstance(candidate, TypeVar):
            return None
        if not isinstance(candidate, type):
            raise ModelDeclarationError(
                f"{cls.__qualname__}: Model[...] takes a model TYPE class "
                f"(e.g. Model[SDXL]), got {candidate!r}"
            )
        return candidate
    return None


class Model(Generic[MT]):
    """Base every author model class inherits.

    THE HEADER IS THE DECLARATION, and there is exactly one form::

        class SdxlModel(
            Model[SDXL],
            lanes={("sdxl.diffusers@1", "plain.bf16@1"): lane(
                request=const(GiB(1.2)) + per_mp_batch(MiB(220)),
                resident=("vae",),
            )},
            structural={"timestep_dtype": Structural(
                field="scheduler",
                classes={"int64": "dpmpp_2m_karras", "float32": "euler"},
                measured="pgw#1572, CPU: set_timesteps(20) per served scheduler",
            )},
            shapes={"aspect": STATIC},
        ):
            def load(self, ctx: LoadContext[SDXL]) -> None: ...

    ``lanes=`` is REQUIRED on every model class, real tensorfs contracts
    only (Paul's ruling pair, 2026-08-20). It answers checkpoint
    COMPATIBILITY and lane SELECTION, not merely compilation, so there is
    nothing an implicit, derived or borrowed lane could stand in for: a model
    FAMILY has no intrinsic layout — a CHECKPOINT has one, and the serving
    declaration commits to it.

    **COMPILATION PARTICIPATION IS THE MARK, NEVER A KEYWORD.** Paul,
    verbatim: *"There is no 'eager only' even if you do not want the model
    compiled. If you do not want the model compiled, simply do not include
    any ctx.compile() invocations in your model's 'load' method."*
    ``eager_only=`` is DELETED; the presence or absence of a ``ctx.compile``
    mark in ``load()`` is the entire statement, and it is statically readable
    by AST (:func:`load_marks_compile`) with no author code executed.

    **THE AUTHOR DECLARES ONLY WHAT ONLY THE AUTHOR KNOWS.** Demand SCALING,
    fork AXES, "my VAE decode will thrash if streamed". The platform derives
    everything derivable: weight bytes from the manifest, the capability
    floor from the contract dtype, launch-residency from the compile marks,
    demand coefficients from measurement. That is why there is no VRAM
    string, no ``min_sm``, and no strategy ladder on this header.

    IT IS ALSO WHAT SERVING ADMISSION CHARGES (pgw#1590), and this is the one
    thing that goes wrong by OMITTING it. With no floor, admission has only
    the checkpoint tree to size a lane from — the WHOLE tree, at its stored
    precision, plus 25% — and that number cannot see a ``setup()``-time
    ``quantize_()`` or a component this lane offloads. minimax-h3's DiT lane
    was refused as needing 180 GB on a card that had served it, because 133 GB
    of stored bf16 becomes ~66 GB of w8a8 inside ``load``. Declaring the floor
    replaces that guess with your number. It only ever LOWERS the charge, so
    adding one can never make a lane harder to admit than leaving it off.

    ``__init__`` stays FREE (no GPU, no weights): construction and loading are
    separate moments, and derive/introspection instantiate without weights.
    """

    __cozy_model_type__: ClassVar[Any] = None
    __cozy_lanes__: ClassVar[tuple[Any, ...] | None] = None
    __cozy_declared_lanes__: ClassVar[tuple[Any, ...]] = ()
    __cozy_requires__: ClassVar[dict[str, Any]] = {}
    __cozy_structural__: ClassVar[dict[str, Any]] = {}
    __cozy_shapes__: ClassVar[dict[str, str]] = {}
    __cozy_self_loading__: ClassVar[str] = ""

    def __init_subclass__(
        cls,
        *,
        lanes: Mapping[Any, Any] | None = None,
        structural: Mapping[str, Any] | None = None,
        shapes: Mapping[str, str] | None = None,
        self_loading: str | None = None,
        **kwargs: Any,
    ) -> None:
        for dead, replacement in _DELETED_KWARGS.items():
            if dead in kwargs:
                kwargs.pop(dead)
                raise ModelDeclarationError(f"{cls.__qualname__}: {replacement}")
        if self_loading is not None:
            if not isinstance(self_loading, str):
                raise ModelDeclarationError(
                    f"{cls.__qualname__}: self_loading= must be a string "
                    f"reason, got {type(self_loading).__name__}"
                )
            reason = self_loading.strip()
            if not reason:
                raise ModelDeclarationError(
                    f"{cls.__qualname__}: self_loading= needs a REASON. Say "
                    "why `ctx.load` cannot drive this pipeline — a bespoke "
                    "loader, an external server, pipeline code that ships "
                    "inside the checkpoint. A blank escape is the silence "
                    "this declaration replaces."
                )
            cls.__cozy_self_loading__ = reason
        super().__init_subclass__(**kwargs)
        if Model in cls.__bases__ and _declared_model_type(cls) is None and not any(
            isinstance(parameter, TypeVar)
            for base in cls.__dict__.get("__orig_bases__", ())
            for parameter in typing.get_args(base)
        ):
            raise ModelDeclarationError(
                f"{cls.__qualname__}: declare the model type in the class "
                f"header — class {cls.__name__}(Model[SDXL], ...); the "
                "generic parameter is the single source of the expected "
                "model type (pgw#1377/pgw#1382)"
            )
        declared_type = _declared_model_type(cls)
        if declared_type is not None:
            cls.__cozy_model_type__ = declared_type

        concrete = getattr(cls, MODEL_TYPE_ATTR, None) is not None
        marks_compile = _calls_ctx_compile(cls.__dict__.get("load"))

        if lanes is not None:
            declared_lanes = _parse_lanes(cls, lanes)
            cls.__cozy_declared_lanes__ = declared_lanes
            cls.__cozy_lanes__ = tuple(row.layout for row in declared_lanes)
            cls.__cozy_requires__ = {
                row.contract_id: requirements
                for row, requirements in (
                    (row, lane_requirements(row)) for row in declared_lanes
                )
                if requirements is not None
            }
        elif concrete and not getattr(cls, DECLARED_LANES_ATTR, ()):
            raise ModelDeclarationError(
                f"{cls.__qualname__}: lanes= is REQUIRED and is missing. A "
                f"lane is a real layout STAMP — the (topology, quant) pair — "
                f"and it answers checkpoint compatibility and lane selection, "
                f"not just compilation, so there is nothing an implicit or "
                f"borrowed one could stand in for. A model FAMILY has no "
                f"canonical layout; a CHECKPOINT has one, and this declaration "
                f"commits to it. Write "
                f'`lanes={{("<topology>@N", "<quant>@N"): lane(request=…)}}`. '
                f"(The `canonical_contract` borrow, `lanes=()` and "
                f"`eager_only=` are all deleted — pgw#1599.)"
            )

        if structural is not None:
            cls.__cozy_structural__ = parse_structural(cls.__qualname__, structural)
        if concrete:
            cls.__cozy_shapes__ = parse_shapes(
                cls.__qualname__, shapes, marks_compile=marks_compile
            )
        elif shapes is not None:
            raise ModelDeclarationError(
                f"{cls.__qualname__}: shapes= on a class that declares no "
                f"model type; the header that declares the axes is the "
                f"concrete subclass's."
            )

    def load(self, ctx: "LoadContext[MT]") -> None:
        """Once per instance at residency-admit: build the pipeline (``ctx.load``), mark compile targets (``ctx.compile``), decode defaults (``ctx.defaults``)."""

    def unload(self, ctx: "LoadContext[MT]") -> None:
        """On LRU eviction, AFTER in-flight requests drain and BEFORE the framework drops references."""


def model_type(cls: type) -> type:
    """The declared model type of a model class — publish-time extraction."""

    if not (isinstance(cls, type) and issubclass(cls, Model)):
        raise ModelDeclarationError(
            f"{getattr(cls, '__qualname__', cls)!r} is not a gen_worker.Model "
            "subclass"
        )
    declared = getattr(cls, MODEL_TYPE_ATTR, None)
    if declared is None:
        raise ModelDeclarationError(
            f"{cls.__qualname__} declares no model type; write "
            f"class {cls.__name__}(Model[SDXL], ...)"
        )
    return typing.cast(type, declared)


def model_lanes(cls: type) -> tuple[Any, ...]:
    """The model class's lane CONTRACT objects, in declaration order."""

    model_type(cls)
    return tuple(getattr(cls, LANES_ATTR, None) or ())


def model_declared_lanes(cls: type) -> tuple[DeclaredLane, ...]:
    """The class's lanes, fully READ — the shared consumer surface."""

    model_type(cls)
    return tuple(getattr(cls, DECLARED_LANES_ATTR, ()) or ())


def model_lane_spec(cls: type, handle: str) -> LaneSpec:
    """One lane's declaration, by contract handle."""

    for row in model_declared_lanes(cls):
        if row.contract_id == handle:
            return row.spec
    raise ModelDeclarationError(
        f"{cls.__qualname__} declares no lane {handle!r}; it declares "
        f"{[row.contract_id for row in model_declared_lanes(cls)]!r}"
    )


def model_structural(cls: type) -> dict[str, Structural]:
    """The class's declared STRUCTURAL fork axes, ``{axis: Structural}``."""

    model_type(cls)
    return dict(getattr(cls, STRUCTURAL_ATTR, None) or {})


def model_shapes(cls: type) -> dict[str, str]:
    """The class's per-shape-axis choice, ``{axis: "static"|"dynamic"}``."""

    model_type(cls)
    return dict(getattr(cls, SHAPES_ATTR, None) or {})


def model_requires(cls: type) -> dict[str, Any]:
    """Per-lane placement rows, ``{contract handle: LayoutRequirements}``."""

    model_type(cls)
    return dict(getattr(cls, REQUIRES_ATTR, None) or {})


def model_marks_compile(cls: type) -> bool:
    """Whether this class's own ``load()`` marks a compile target."""

    model_type(cls)
    return _calls_ctx_compile(cls.__dict__.get("load")) or any(
        _calls_ctx_compile(base.__dict__.get("load"))
        for base in cls.__mro__[1:]
        if "load" in base.__dict__
    )


_DELETED_KWARGS: dict[str, str] = {
    "requires": (
        "`requires=` is DELETED — a lane's needs are its own declaration. "
        'Write `lanes={contract: lane(request=…)}`; the capability floor is '
        "derived from the contract dtype and is never written by hand."
    ),
    "eager_only": (
        "`eager_only=` is DELETED (Paul, 2026-08-20): *\"There is no 'eager "
        "only' even if you do not want the model compiled. If you do not want "
        "the model compiled, simply do not include any ctx.compile() "
        "invocations in your model's 'load' method.\"* It conflated two "
        "independent axes — a lane answers checkpoint compatibility and lane "
        "selection whether or not anything compiles. Delete the keyword, "
        "declare real `lanes=`, and let the absent mark be the statement. "
        "The measured no-win REASON belongs in a comment beside the absent "
        "mark, not in the API."
    ),
    "memory_forks": (
        "`memory_forks=` was never built and is not coming: tiling is an "
        "EAGER-MODE degradation (pgw#1605's catalog), never a compiled "
        "structural fork (Paul's final simplifying ruling, 2026-08-20)."
    ),
}


__all__ = [
    "DECLARED_LANES_ATTR",
    "DYNAMIC",
    "LANES_ATTR",
    "LaneSpec",
    "DeclaredLane",
    "MODEL_TYPE_ATTR",
    "Model",
    "REQUIRES_ATTR",
    "SHAPES_ATTR",
    "STATIC",
    "STRUCTURAL_ATTR",
    "Structural",
    "lane",
    "lane_dtype",
    "lane_handle",
    "lane_requirements",
    "load_marks_compile",
    "model_declared_lanes",
    "model_lane_spec",
    "model_lanes",
    "model_marks_compile",
    "model_requires",
    "model_shapes",
    "model_structural",
    "model_type",
    "ModelDeclarationError",
]
