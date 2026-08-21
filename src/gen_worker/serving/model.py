"""``Model[MT]`` — the stateful half of the pgw#1382 split.

A model class owns EVERYTHING persistent: weights, compile-marked modules,
defaults, kv-caches/sessions, and the request-scoped mutation SCOPES (context
managers the author writes on the class — "leave it as you found it": at
entrypoint return, serving configuration equals the post-load baseline;
caches may grow, configuration may not drift). Entrypoints are STATELESS
module-level functions (:mod:`.entrypoints`) that declare their model by
parameter annotation.

One instance per (checkpoint x lane), LRU-resident, SINGLE-FLIGHT — the
concurrency contract attaches to the object that has the state (the host
owns the admission; see :mod:`.host`).

The class header is the single statically-extractable declaration surface:
the model type is the generic parameter (read from ``__orig_bases__``), the
weight-format lanes are the ``lanes=`` class kwarg — both readable at
publish with no author code executed beyond import.

HARD GUARDRAIL: framework capabilities arrive via ctx ONLY (``LoadContext``
in ``load``/``unload``, ``RequestContext`` in entrypoints). This base is a
typed skeleton, not a toolbox; nothing else goes on it without a Paul ruling.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

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

if TYPE_CHECKING:  # keep the base import-weightless
    from .context import LoadContext

MT = TypeVar("MT")

#: Class attributes the header declaration lands on — the publish extractor's
#: read surface.
MODEL_TYPE_ATTR = "__cozy_model_type__"
LANES_ATTR = "__cozy_lanes__"
REQUIRES_ATTR = "__cozy_requires__"
#: pgw#1599: the fully READ lanes — ``(DeclaredLane, …)``, contract object +
#: handle + dtype + derived ``min_sm`` + the lane's demand formula. The one
#: read surface every consumer shares (pgw#1606's selection ladder, derive,
#: placement), so nothing re-parses a stamp and nothing re-derives a floor.
DECLARED_LANES_ATTR = "__cozy_declared_lanes__"
#: pgw#1599: the class-level STRUCTURAL fork axes, ``{axis: Structural}``.
STRUCTURAL_ATTR = "__cozy_structural__"
#: pgw#1599: the per-shape-axis static/dynamic choice, ``{axis: "static"}``.
SHAPES_ATTR = "__cozy_shapes__"
#: pgw#1431 fix (b). The author's REASON that this model's pipeline has no
#: class `ctx.load` can drive — the v2 successor to v1's `Slot(str)` escape
#: hatch, and the pipeline-level twin of `Slot(layouts_undeclarable=)`, which
#: says the same thing one level down about the BYTES.
SELF_LOADING_ATTR = "__cozy_self_loading__"


class ModelDeclarationError(TypeError):
    """A model class header does not state a valid declaration."""


@runtime_checkable
class LaneContract(Protocol):
    """A lane IS a tensorfs layout contract — an object carrying the layout
    the load code expects, with its load ``dtype`` readable
    (``ctx.lane.dtype``). Until tensorfs#111 ships contract objects, the
    vendored torchcg ``LaneRef`` (contract handle + dtype) satisfies this
    seam; the Protocol is what the SDK checks, never a class identity."""

    @property
    def dtype(self) -> Any: ...


def lane_handle(lane: Any) -> str:
    """The lane's stable string handle, read off the SHIPPED tensorfs surface.

    tensorfs#111 landed and its ``Contract`` spells the handle ``stamp``
    (``name@version`` for a library contract, ``sha256:<hex>`` for an
    anonymous one). ``digest`` is a BARE 64-hex string — a real attribute,
    but not a handle: reading it as one produced
    ``f1455f56…`` where ``sdxl.diffusers-bf16@1`` belonged, and torchcg
    refused the lane. So ``stamp`` leads, ``contract`` stays for objects that
    spell it that way, ``name``+``version`` is the structural fallback, and
    ``digest`` is only ever used PREFIXED.
    """

    # pgw#1391: a lane that names a contract tensorfs ships no document for
    # refuses HERE, at the one chokepoint every stamp-reading surface goes
    # through (declaration, discovery's `_lane_stamps`, derive's
    # `_lane_model_class`) — restated as a ModelDeclarationError so derive's
    # existing conversion to a loud DeriveError needs no second pattern.
    from ..models.model_types import MissingContractError

    try:
        for attribute in ("stamp", "contract"):
            value = getattr(lane, attribute, None)
            if isinstance(value, str) and value:
                return value
        name = getattr(lane, "name", None)
        version = getattr(lane, "version", None)
        if isinstance(name, str) and name and isinstance(version, int):
            return f"{name}@{version}"
        digest = getattr(lane, "digest", None)
        if isinstance(digest, str) and digest:
            return digest if digest.startswith("sha256:") else f"sha256:{digest}"
    except MissingContractError as exc:
        raise ModelDeclarationError(str(exc)) from exc
    raise ModelDeclarationError(
        f"lane {lane!r} carries no string handle (`stamp`, `contract`, "
        "`name`+`version` or `digest`); a lane is a tensorfs layout contract "
        "object"
    )


#: Lane stamps whose contract document declares no top-level dtype UPSTREAM,
#: waived at declaration time so a live model class is not refused for a gap
#: only tensorfs can close.
#:
#: EMPTY, AND IT SHOULD STAY THAT WAY — the fence already did its job once. It
#: briefly held ``minimax.h3-dit-diffusers@1`` while that document was
#: dtype-less and its endpoint was live (deploy lane ``ab56185761f1597f1``);
#: tensorfs#121 gave it ``bfloat16``, re-vendoring made
#: ``test_the_h3_dtype_waiver_deletes_itself_when_upstream_lands_the_dtype``
#: fail, and the entry was deleted. A waiver is a fact about the vendored
#: document, never a preference, so it cannot outlive its reason.
#:
#: Three vendored documents are STILL dtype-less on purpose
#: (``dit.blocks-fused-qkv``, ``sdxl.clip-g-fused-qkv``,
#: ``sdxl.clip-g-split-qkv``) and need no waiver: they are FRAGMENTS a
#: ``lanes=`` header never names on its own, so they never reach this check.
#: Adding an entry here means knowingly accepting a lane whose serve-side
#: ``ctx.lane.dtype`` will raise — do it only against a named upstream issue.
DTYPELESS_UPSTREAM_LANES: frozenset[str] = frozenset()


def lane_dtype(lane: Any, *, where: str) -> Any:
    """The lane's declared load dtype, READ rather than looked for.

    pgw#1391: ``isinstance(lane, LaneContract)`` is not this check. A
    ``runtime_checkable`` Protocol with a property member resolves through
    ``inspect.getattr_static`` on py3.12, so it never invokes the getter — a
    ``@property`` that RAISES satisfies it. That is exactly how a contract
    declaring no dtype used to clear declaration and die at load on a pod, and
    how a documentless lane used to clear it at all.

    So the dtype is read. ``MissingDtype`` (and the ``MissingContractError``
    that a documentless lane raises first) become a declaration refusal naming
    the author's own class header.
    """

    from ..models.model_types import MissingContractError

    handle = lane_handle(lane)  # refuses first if there is no document at all
    try:
        dtype = lane.dtype
    except MissingContractError as exc:  # pragma: no cover - lane_handle refuses first
        raise ModelDeclarationError(str(exc)) from exc
    except Exception as exc:
        if handle in DTYPELESS_UPSTREAM_LANES:
            return None
        raise ModelDeclarationError(
            f"{where}: lane {handle} declares no load dtype "
            f"({type(exc).__name__}: {exc}). `ctx.lane.dtype` is on the serve "
            f"path, so this refuses HERE rather than reading None at load on a "
            f"rented pod. A tensorfs document with no top-level dtype is "
            f"usually a FRAGMENT — a shared block or component spelling, not a "
            f"serve layout — and the fix is then to author the real lane "
            f"document in tensorfs spec/v1/contracts and re-vendor, NOT to add "
            f"a dtype to the fragment. Declaring the fragment in `lanes=` "
            f"explicitly does not make it a lane."
        ) from exc
    if dtype is None:
        if handle in DTYPELESS_UPSTREAM_LANES:
            return None
        raise ModelDeclarationError(
            f"{where}: lane {handle} answers None for its load dtype. A lane "
            f"IS a tensorfs layout contract and its dtype is the contract's; "
            f"None is not an answer."
        )
    return dtype


def _calls_ctx_compile(fn: Any) -> bool:
    """Whether ``load`` calls ``ctx.compile(...)`` — statically, by AST.

    pgw#1599: this IS the compilation declaration. Paul, verbatim: *"If you
    do not want the model compiled, simply do not include any ctx.compile()
    invocations in your model's 'load' method."* There is no keyword to
    contradict it and nothing to cross-check it against, so the class of
    silent contradiction pgw#1469 measured cannot be constructed any more.

    Parsed rather than grepped — the string ``ctx.compile`` appears in comments
    and docstrings that say a model deliberately does NOT compile, and a
    substring check would refuse exactly the classes that documented
    themselves best.
    """

    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):  # exec'd/builtin source is not readable
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
    """Whether this ``load`` DEFINITION marks a compile target — the AST half
    of :func:`_calls_ctx_compile`, taking the parsed node instead of a live
    function.

    PUBLIC, and split out for exactly one reason (se#809): a reader that can
    only be reached through :func:`inspect.getsource` can only be used by code
    that has already IMPORTED the endpoint, and an endpoint module imports
    torch. Every repo-side gate over the fleet is therefore torch-free and
    AST-based, so before this split the only way for one to ask "does this
    model compile?" was to write a SECOND walker — which is how
    `serverless-endpoints`' AUTHOR-CI gate ended up still counting the v1
    `@endpoint(compile=...)` spelling and reporting 2 compiling endpoints
    where the fleet had 13.

    One reader, two entry points: this takes a parsed
    ``FunctionDef``/``AsyncFunctionDef``, and ``_calls_ctx_compile`` is the
    thin wrapper that gets there from a live function object.

    Parsed rather than grepped — the string ``ctx.compile`` appears in comments
    and docstrings that say a model deliberately does NOT compile, and a
    substring check would refuse exactly the classes that documented
    themselves best.
    """

    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    # The ctx parameter: `load(self, ctx)`, the ruled signature (pgw#1382).
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
    """``lanes={contract: lane(...)}`` -> the fully READ ``DeclaredLane``s.

    ONE declaration, and it is the class's whole memory story. The mapping
    KEY is a real tensorfs contract object — never a name string, never
    implicit, never borrowed from the model type — and the VALUE is this
    lane's :func:`~gen_worker.serving.lane_spec.lane` declaration: its demand
    FORMULA and its optional additive residency override.

    What used to be here and is DELETED (pgw#1599): the machine-floor
    STRING (``"vram7g"``). Paul, 2026-08-20: *"the memory-requirement being
    bundled per lane is the wrong model entirely. Memory requirements vary
    per request … Furthermore, there is no required VRAM."* A number that
    stood for every request a lane would ever serve was wrong for all of
    them, so it is replaced by the formula rather than moved.

    What survives, unchanged in substance: ``min_sm`` is DERIVED from the
    lane contract's own load dtype (:func:`capability_floor_for_dtype`) and
    an author who writes it by hand is refused. It is a per-LANE fact — an
    8-bit lane needs 8-bit kernels because of what it IS — which is exactly
    why a hand-written floor could never be right for a class declaring
    bf16, fp8 and nvfp4 at once (pgw#1606).
    """
    from ..models.tensor_layout_contract import capability_floor_for_dtype

    where = f"{cls.__qualname__}: lanes="
    if not isinstance(lanes, Mapping):
        raise ModelDeclarationError(
            f"{where} is a MAPPING of tensorfs contract -> lane(...), got "
            f"{type(lanes).__name__}. The tuple form is deleted: every lane "
            f"declares its own demand formula, so there is nothing for a bare "
            f"tuple to carry. Write "
            f"`lanes={{contracts.SDXL_DIFFUSERS_BF16: lane(request=…)}}`."
        )
    if not lanes:
        raise ModelDeclarationError(
            f"{where} is EMPTY. `lanes=()` and `lanes={{}}` are deleted "
            f"(pgw#1597 ruling pair): a lane answers checkpoint COMPATIBILITY "
            f"and lane SELECTION, not just compilation, so every model class "
            f"names at least one real tensorfs contract. If you do not want "
            f"this model compiled, simply do not call `ctx.compile()` in "
            f"`load()` — that is the entire eager declaration."
        )

    declared: list[DeclaredLane] = []
    seen: set[str] = set()
    for contract, spec in lanes.items():
        if not isinstance(contract, LaneContract):
            raise ModelDeclarationError(
                f"{cls.__qualname__}: lane {contract!r} is not a layout "
                "contract (no `dtype`); a lane is an imported "
                "tensorfs contract object, never a name string"
            )
        # The isinstance above only proves the ATTRIBUTE exists; this
        # READS it, which is the pgw#1391 difference.
        dtype = lane_dtype(contract, where=cls.__qualname__)
        handle = lane_handle(contract)
        site = f"{where}[{handle!r}]"
        if handle in seen:
            raise ModelDeclarationError(
                f"{site}: declared twice. One row per lane — two rows for one "
                f"contract can only disagree about its demand."
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

        # A lane with no dtype cannot state a capability floor, and a floor is
        # the one place failing OPEN is invisible: an absent `min_sm` reads to
        # the resolver as "runs anywhere", which is th#1754's shape with a new
        # cause. `lane_dtype` already refuses a dtypeless contract — EXCEPT for
        # a handle in `DTYPELESS_UPSTREAM_LANES`, where it answers None.
        if not dtype:
            raise ModelDeclarationError(
                f"{site}: lane {handle} declares no load dtype, so no "
                "compute-capability floor can be derived for it. A lane that "
                "cannot state `min_sm` would publish a floor the resolver "
                "reads as 'runs anywhere' — silently, which is worse than the "
                "load crash a dtypeless contract used to cause. Declare the "
                "dtype on the tensorfs contract document (a fused-QKV or "
                "text-encoder COMPONENT layout is usually not a serve lane at "
                "all, and the fix is then to name the real lane document)."
            )
        declared.append(
            DeclaredLane(
                contract=contract,
                contract_id=handle,
                dtype=dtype,
                # The capability floor falls out of the CONTRACT, never the
                # header. Two producers of one fact is how they drift apart.
                min_sm=capability_floor_for_dtype(dtype),
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

    ``None`` for a lane whose dtype derives NO floor — fp32, and any dtype
    ``DTYPE_MIN_SM`` does not know. That is the honest answer and it must not
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
    """The ``X`` in ``class C(Model[X])`` — from ``__orig_bases__``, no
    author code executed. ``None`` for a still-generic intermediate."""

    for base in cls.__dict__.get("__orig_bases__", ()):
        origin = typing.get_origin(base)
        if origin is None or not (isinstance(origin, type) and issubclass(origin, Model)):
            continue
        args = typing.get_args(base)
        if len(args) != 1:
            continue
        candidate = args[0]
        if isinstance(candidate, TypeVar):
            return None  # generic intermediate: class Diffusion(Model[MT])
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
            lanes={contracts.SDXL_DIFFUSERS_BF16: lane(
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
            # pgw#1431 fix (b). A REASON IS MANDATORY, verbatim the rule
            # `Slot(layouts_undeclarable=)` already enforces one level down: an
            # escape hatch with no stated reason is the silence the rung exists
            # to replace, and it is the only thing standing between this marker
            # and a way to quietly silence a class discovery could have read.
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

        # A still-generic intermediate (`class Diffusion(Model[MT])`) declares
        # nothing and is refused nothing: it is not a servable model, and the
        # concrete subclass that IS one carries the header.
        concrete = getattr(cls, MODEL_TYPE_ATTR, None) is not None
        marks_compile = _calls_ctx_compile(cls.__dict__.get("load"))

        if lanes is not None:
            declared_lanes = _parse_lanes(cls, lanes)
            cls.__cozy_declared_lanes__ = declared_lanes
            cls.__cozy_lanes__ = tuple(row.contract for row in declared_lanes)
            cls.__cozy_requires__ = {
                row.contract_id: requirements
                for row, requirements in (
                    (row, lane_requirements(row)) for row in declared_lanes
                )
                if requirements is not None
            }
        elif concrete and not getattr(cls, DECLARED_LANES_ATTR, ()):
            # THE OMISSION REFUSAL (pgw#1597 ruling pair). Named, at
            # class-definition time, before any author code runs.
            raise ModelDeclarationError(
                f"{cls.__qualname__}: lanes= is REQUIRED and is missing. A "
                f"lane is a real tensorfs layout CONTRACT, and it answers "
                f"checkpoint compatibility and lane selection — not just "
                f"compilation — so there is nothing an implicit or borrowed "
                f"one could stand in for. A model FAMILY has no canonical "
                f"layout; a CHECKPOINT has one, and this declaration commits "
                f"to it. Write "
                f"`lanes={{contracts.<YOUR_CONTRACT>: lane(request=…)}}`. "
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

    # -- lifecycle hooks (the load/unload contract, pgw#1382) ---------------

    def load(self, ctx: "LoadContext[MT]") -> None:
        """Once per instance at residency-admit: build the pipeline
        (``ctx.load``), mark compile targets (``ctx.compile``), decode
        defaults (``ctx.defaults``). No-op by default."""

    def unload(self, ctx: "LoadContext[MT]") -> None:
        """On LRU eviction, AFTER in-flight requests drain and BEFORE the
        framework drops references. No-op default — the common case is
        framework-generic (drop refs + allocator reclaim); override only for
        the exceptional (external server processes, temp sockets, kv-cache
        persistence). BEST-EFFORT TIDINESS, NEVER CORRECTNESS: crash/kill -9
        calls nothing, and a failing or slow unload cannot pin VRAM —
        exceptions are logged and eviction proceeds."""


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
    """The model class's lane CONTRACT objects, in declaration order.

    Never empty: `lanes=` is required, `lanes=()` is deleted, and there is no
    implicit, canonical or derived lane to fall through to. Declaration order
    is the author's writing order and carries NO priority — choosing among
    lanes is platform machinery (pgw#1606), never endpoint code.
    """

    model_type(cls)  # validates cls
    return tuple(getattr(cls, LANES_ATTR, None) or ())


def model_declared_lanes(cls: type) -> tuple[DeclaredLane, ...]:
    """The class's lanes, fully READ — the shared consumer surface.

    One :class:`~gen_worker.serving.lane_spec.DeclaredLane` per declared lane:
    the contract object, its handle, its dtype, its DERIVED ``min_sm``, and
    its demand formula + residency override. Everything a boot-time
    selection ladder, a derive or a placement check needs, with no stamp
    re-parsed and no floor re-derived.
    """

    model_type(cls)  # validates cls
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
    """The class's declared STRUCTURAL fork axes, ``{axis: Structural}``.

    Same contract, different traced PROGRAM. Empty means the author declares
    that this model's program does not fork — a CLAIM the loud-eager leak
    detector falsifies in production if it is wrong (pgw#1597), never a
    default the platform guessed.
    """

    model_type(cls)  # validates cls
    return dict(getattr(cls, STRUCTURAL_ATTR, None) or {})


def model_shapes(cls: type) -> dict[str, str]:
    """The class's per-shape-axis choice, ``{axis: "static"|"dynamic"}``.

    Replaces derive's global ``DYNAMIC_AXES`` flag (pgw#1599): which axis is
    worth collapsing is a MEASURED, per-model question (pgw#1548), so it is
    declared on the model that measured it — never passed on a command line
    where it silently re-keys every graph in the fleet at once.
    """

    model_type(cls)  # validates cls
    return dict(getattr(cls, SHAPES_ATTR, None) or {})


def model_requires(cls: type) -> dict[str, Any]:
    """Per-lane placement rows, ``{contract handle: LayoutRequirements}``.

    The DERIVED capability floor and nothing else — the VRAM half died with
    the floor strings (pgw#1599). What the hub shops on is the demand
    formula's serialized worst case (pgw#1600), which is computed, not
    annotated.
    """

    model_type(cls)  # validates cls
    return dict(getattr(cls, REQUIRES_ATTR, None) or {})


def model_marks_compile(cls: type) -> bool:
    """Whether this class's own ``load()`` marks a compile target.

    THE eager/compiled declaration, and the only one there is (Paul's ruling
    pair): no keyword, no lane shape, no inference from an absent contract —
    the MARK, read by AST with no author code executed.
    """

    model_type(cls)  # validates cls
    return _calls_ctx_compile(cls.__dict__.get("load")) or any(
        _calls_ctx_compile(base.__dict__.get("load"))
        for base in cls.__mro__[1:]
        if "load" in base.__dict__
    )


#: Class kwargs that are DELETED, each with the message that says what
#: replaced it. A header written against the old vocabulary refuses at
#: class-definition time naming the new spelling — never silently ignored.
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
    "DTYPELESS_UPSTREAM_LANES",
    "DYNAMIC",
    "LANES_ATTR",
    "LaneContract",
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
