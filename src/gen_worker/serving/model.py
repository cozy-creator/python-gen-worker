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
import re
import textwrap
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:  # keep the base import-weightless
    from .context import LoadContext

MT = TypeVar("MT")

#: Class attributes the header declaration lands on — the publish extractor's
#: read surface.
MODEL_TYPE_ATTR = "__cozy_model_type__"
LANES_ATTR = "__cozy_lanes__"
REQUIRES_ATTR = "__cozy_requires__"
#: pgw#1431 fix (b). The author's REASON that this model's pipeline has no
#: class `ctx.load` can drive — the v2 successor to v1's `Slot(str)` escape
#: hatch, and the pipeline-level twin of `Slot(layouts_undeclarable=)`, which
#: says the same thing one level down about the BYTES.
SELF_LOADING_ATTR = "__cozy_self_loading__"
#: pgw#1488 (Paul: "A NORMAL TRACE MUST JUST WORK"). The author's REASON that
#: this model is served EAGER FOREVER. It is the ONLY way to be eager: an
#: absent lane declaration means "I state no layout contract", which now
#: TRACES under a derived identity instead of silently disabling compilation.
EAGER_ONLY_ATTR = "__cozy_eager_only__"

#: Producer namespace of a DERIVED lane handle. Reserved: a tensorfs contract
#: is named after the model that publishes it, so nothing real is ever called
#: ``derived.*`` and a reader can tell the two apart at a glance.
DERIVED_LANE_PRODUCER = "derived"


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


@dataclass(frozen=True, slots=True)
class DerivedLane:
    """The lane of a model class that states NO layout contract (pgw#1488).

    Paul's ruling: *"Traces should just trace."* A contract document is fleet
    METADATA — a name, a price, a published layout — and metadata cannot be a
    precondition for producing the artifacts it describes. So a class that
    names no contract still has exactly one lane; its handle is derived from
    the model type, deterministically, and the same derivation runs at trace
    and at serve so both ends address the same row.

    Nothing is lost by deriving it. A lane handle is a NAME, not a key: graph
    identity is ``cg-graph-v1`` (the canonical trace + ingress + passes) and
    artifact identity is ``cg-key-v1`` (graph + sm + toolchain), and the
    contract string appears in neither. That is why a contract-declaring class
    keeps every byte it had — the handle it publishes is unchanged — while a
    contract-less class stops being refused.

    ``dtype`` is None here on purpose and is NOT a gap: with no contract the
    load dtype is the CHECKPOINT's own (``serving.checkpoint_dtype``), read at
    the two places that hold a checkpoint tree.
    """

    stamp: str
    dtype: Any = None


def _slug(name: str) -> str:
    """A class name as a contract-handle path segment."""
    text = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return text or "model"


def derived_lane(cls: type) -> DerivedLane:
    """The derived lane of ``cls`` — the same answer on every machine."""

    return DerivedLane(
        stamp=f"{DERIVED_LANE_PRODUCER}.{_slug(model_type(cls).__name__)}@1"
    )


def is_derived_lane(lane: Any) -> bool:
    """Whether this lane's identity was derived rather than declared."""

    return isinstance(lane, DerivedLane)


def eager_only_reason(cls: type) -> str:
    """The class's declared eager-forever REASON, or ``""``."""

    return str(getattr(cls, EAGER_ONLY_ATTR, "") or "").strip()


def _calls_ctx_compile(fn: Any) -> bool:
    """Whether ``load`` calls ``ctx.compile(...)`` — statically, by AST.

    pgw#1469 measured the mirror of the refusal that already existed: a lane
    with no compile mark refuses, but a compile mark with no lane was SILENT —
    ``load`` was never called at all, so nothing observed the mark, and the
    author got a green lock with a byte-identical document. Under pgw#1488 the
    unmarked case traces by default, so the only way to reach that silence is
    to declare ``eager_only=`` AND mark a target, and this is what makes that
    contradiction a refusal for free: no author code runs, no model loads.

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
    cls: type, lanes: tuple[Any, ...] | Mapping[Any, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """``lanes=`` -> ``(lane contracts, {handle: LayoutRequirements})``.

    ONE declaration, two readings. The mapping form states each lane WITH the
    machine floor that lane needs::

        lanes={contracts.MINIMAX_H3_DIT_DIFFUSERS: "vram78g"}

    and the tuple form states lanes with no floor at all::

        lanes=(contracts.SDXL_DIFFUSERS_BF16,)

    A floor belongs to the lane and cannot be written anywhere else, which is
    the point of the merge: the two structures used to be keyed by the same
    contract objects and could disagree — a floor could guard a lane the model
    did not declare, and the check for that is now impossible to need.
    ``None``/``""`` is a legal mapping value for a floor-less lane, so a model
    with a floor on one lane and none on another writes ONE dict.

    The value carries **VRAM ONLY** (Paul, 2026-08-18): *"the sm_x compute
    floor should fall out of the contract itself, rather than being a separate
    annotation. Only the VRAM requirement needs a separate annotation, because
    it's not clear, based on the contract, how much VRAM is needed."* So
    ``min_sm`` is DERIVED from the lane contract's own load dtype
    (:func:`capability_floor_for_dtype`) and merged in here, and an author who
    writes it by hand is refused rather than allowed to create a second
    producer of one fact. It is parsed HERE so the refusal names the author's
    own class header, and statically extractable at publish so placement never
    has to run author code.
    """
    from ..models.tensor_layout_contract import (
        RequirementTerms,
        capability_floor_for_dtype,
        parse_layout_requirements,
    )
    import msgspec

    where = f"{cls.__qualname__}: lanes="
    if isinstance(lanes, Mapping):
        items = list(lanes.items())
    elif isinstance(lanes, tuple):
        items = [(lane, None) for lane in lanes]
    else:
        raise ModelDeclarationError(
            f"{where} must be a tuple of tensorfs contract objects, or a "
            f"mapping of contract -> machine floor, got "
            f"{type(lanes).__name__}"
        )

    contracts: list[Any] = []
    floors: dict[str, Any] = {}
    for lane, floor in items:
        if not isinstance(lane, LaneContract):
            raise ModelDeclarationError(
                f"{cls.__qualname__}: lane {lane!r} is not a layout "
                "contract (no `dtype`); a lane is an imported "
                "tensorfs contract object, never a name string"
            )
        # The isinstance above only proves the ATTRIBUTE exists; this
        # READS it, which is the pgw#1391 difference.
        dtype = lane_dtype(lane, where=cls.__qualname__)
        contracts.append(lane)
        handle = lane_handle(lane)
        site = f"{where}[{handle!r}]"

        declared = None
        if floor is not None and floor != "":
            declared = parse_layout_requirements(floor, where=site)
            _refuse_non_vram_terms(declared, where=site)

        # A lane with no dtype cannot state a capability floor, and a floor is
        # the one place failing OPEN is invisible: an absent `min_sm` reads to
        # the resolver as "runs anywhere", which is th#1754's shape with a new
        # cause. `lane_dtype` already refuses a dtypeless contract — EXCEPT for
        # a handle in `DTYPELESS_UPSTREAM_LANES`, where it answers None. That
        # escape hatch predates the derivation and would now buy silence rather
        # than the loud load crash it was traded for, so a floor closes it
        # here: fail closed, exactly as the tuple-vs-dict hardcut does.
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

        # The capability floor falls out of the CONTRACT, never the header.
        min_sm = capability_floor_for_dtype(dtype)
        if declared is None and not min_sm:
            continue  # nothing declared, nothing derived
        minimum = declared.min_terms() if declared is not None else RequirementTerms()
        if min_sm:
            minimum = msgspec.structs.replace(minimum, min_sm=min_sm)
        floors[handle] = (
            msgspec.structs.replace(declared, minimum=minimum)
            if declared is not None
            else parse_layout_requirements(minimum, where=site)
        )
    return tuple(contracts), floors


#: The lane annotation states VRAM and nothing else. `min_sm` is derived from
#: the contract; the other axes are not lane facts at all (a CUDA/torch floor
#: is a property of the IMAGE, and host RAM of the function).
_LANE_FLOOR_TERMS: frozenset[str] = frozenset({"min_vram_gb"})


def _refuse_non_vram_terms(requirements: Any, *, where: str) -> None:
    """A lane floor that states anything but VRAM is refused at declaration."""
    for level, terms in (
        ("", requirements.min_terms()),
        (" recommended", requirements.recommended_terms()),
    ):
        extra = sorted(set(terms.declared_terms()) - _LANE_FLOOR_TERMS)
        if not extra:
            continue
        if "min_sm" in extra:
            raise ModelDeclarationError(
                f"{where}{level}: min_sm is DERIVED from the lane contract's "
                "own load dtype, never written here — an 8-bit lane needs "
                "8-bit kernels because of what it IS, and two producers of "
                "one fact is how they drift apart. Drop the sm term; if the "
                "derived floor is wrong for this dtype, fix the table in "
                "`gen_worker.models.tensor_layout_contract.DTYPE_MIN_SM`."
            )
        raise ModelDeclarationError(
            f"{where}{level}: a lane floor states VRAM only, got {extra}. "
            "VRAM is the one floor the contract cannot imply, which is why it "
            "is annotated; the rest are not lane facts (a CUDA/torch floor "
            "belongs to the image, host RAM to the function's Resources)."
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
    """Base every author model class inherits::

        class SdxlModel(Model[SDXL], lanes=(contracts.SDXL_DIFFUSERS_BF16,)):
            def load(self, ctx: LoadContext[SDXL]) -> None: ...

    ``lanes=`` omitted (or ``lanes=()``) means the author states NO layout
    contract: the class gets its model type's canonical contract when one is
    published, and otherwise a DERIVED lane (:class:`DerivedLane`) — either
    way it TRACES. Eager-forever is a separate declaration and never an
    inference from an absent one::

        class RifeModel(Model[Rife], eager_only="frame interpolation runs "
                        "eager; there is no compile target"): ...

    The MAPPING form declares each lane together with what that lane needs of
    a machine, in the ie#740 grammar — one line, one place::

        class H3Model(Model[MiniMaxH3],
                      lanes={contracts.MINIMAX_H3_DIT_DIFFUSERS: "vram78g"}):
            ...

    ``None``/``""`` is a legal floor, so a mixed model still writes one dict.
    Declaring no floor leaves placement UNDECLARED, and the platform's default
    is what a deployment then gets.

    A declared floor INFORMS, it does not permit (Paul, 2026-08-18): the hub
    filters placement on it, and a worker that ends up under it warns loudly
    and serves anyway. Any model runs on any machine — a poor match is slow
    and says so, never refused, so cozy-local can run anything it has the
    patience for.

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
    __cozy_requires__: ClassVar[dict[str, Any]] = {}
    __cozy_self_loading__: ClassVar[str] = ""
    __cozy_eager_only__: ClassVar[str] = ""

    def __init_subclass__(
        cls,
        *,
        lanes: tuple[Any, ...] | Mapping[Any, Any] | None = None,
        self_loading: str | None = None,
        eager_only: str | None = None,
        **kwargs: Any,
    ) -> None:
        if "requires" in kwargs:
            raise ModelDeclarationError(
                f"{cls.__qualname__}: `requires=` is DELETED — a lane and its "
                "machine floor are ONE declaration now. Write "
                "`lanes={contract: \"vram78g\"}` instead of "
                "`lanes=(contract,), requires={contract: \"vram78g\"}`. The "
                "mapping value is the lane's VRAM floor (None/\"\" for none); "
                "the sm floor is derived from the contract, not written."
            )
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
        if eager_only is not None:
            # pgw#1488 fix (3). Same mandatory-reason pattern as
            # `self_loading=`, for the same reason: this is the ONE state in
            # which nothing is compiled, ever, and a state that costs the
            # fleet performance has to say why in the header where anyone
            # reviewing the class will read it.
            if not isinstance(eager_only, str):
                raise ModelDeclarationError(
                    f"{cls.__qualname__}: eager_only= must be a string reason, "
                    f"got {type(eager_only).__name__}"
                )
            eager_reason = eager_only.strip()
            if not eager_reason:
                raise ModelDeclarationError(
                    f"{cls.__qualname__}: eager_only= needs a REASON. Say why "
                    "this model compiles NOTHING, ever — measured no win, no "
                    "compilable module, an auxiliary model another slot "
                    "drives. Eager-forever is the one posture that costs the "
                    "fleet throughput silently, so it states its case."
                )
            if lanes:
                raise ModelDeclarationError(
                    f"{cls.__qualname__}: eager_only= and a non-empty lanes= "
                    "contradict each other — a declared lane exists to key "
                    "compiled graphs, and this class compiles none. Keep one."
                )
            if _calls_ctx_compile(cls.__dict__.get("load")):
                raise ModelDeclarationError(
                    f"{cls.__qualname__}: eager_only="
                    f"{eager_reason!r} while load() marks a compile target "
                    "via ctx.compile(). That mark can never produce a graph — "
                    "pgw#1469 measured exactly this pair going through as a "
                    "green lock with a byte-identical document. Drop the mark, "
                    "or drop eager_only= and let the lock trace."
                )
            cls.__cozy_eager_only__ = eager_reason
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
        declared = _declared_model_type(cls)
        if declared is not None:
            cls.__cozy_model_type__ = declared
        # The lanes this class actually serves, and their floors, from the ONE
        # `lanes=` declaration — or the model type's canonical contract when
        # `lanes=` is omitted (which declares no floor).
        if lanes:
            contracts, floors = _parse_lanes(cls, lanes)
            cls.__cozy_lanes__ = contracts
            cls.__cozy_requires__ = floors
        elif lanes is not None:
            # `lanes=()` — the author states no contract, which is what an
            # omitted `lanes=` says too. pgw#1488 collapses the two: neither is
            # eager-forever, and both fall through to `model_lanes`.
            cls.__cozy_lanes__ = ()
            cls.__cozy_requires__ = {}
        # pgw#1391 VALIDATED the omitted-`lanes=` fall-through HERE and refused
        # a canonical contract tensorfs publishes no readable document for.
        # pgw#1488 deletes that refusal rather than moving it: the fall-through
        # can no longer strand anyone, because `model_lanes` answers a DERIVED
        # lane when the canonical contract is missing or unreadable. The se#757
        # trap it was built for — a class silently CLAIMING a stamp that names
        # no document — is closed by the same change from the other side: an
        # unreadable contract is not borrowed at all, so no false stamp can be
        # published, and the class traces under a handle that says `derived.`.

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
    """The model class's lanes. ``()`` ONLY for ``eager_only=`` (pgw#1488).

    Three states, each a word rather than an inference:

    * ``eager_only="<reason>"`` -> ``()``. Nothing is traced, nothing compiled,
      and the reason travels to whoever asks why.
    * ``lanes=<contracts>`` -> exactly those. Unchanged, byte for byte: a
      contract-declaring class publishes the handle it always did.
    * ``lanes=()`` -> a DERIVED lane. The author stated no contract, so none is
      borrowed — and it TRACES, which is the whole of fix (3): an empty tuple
      used to disable compilation with no output whatsoever.
    * ``lanes=`` omitted -> the model type's canonical contract when tensorfs
      publishes a readable one (the convenience that spelling exists for),
      else a DERIVED lane. This is fix (1): a missing contract document stops
      being a reason to refuse to trace.

    The old refusal ("omits lanes= and its model type has no canonical contract
    yet (tensorfs#111); declare lanes= explicitly, or lanes=() for
    eager-permanent") is DELETED. It cost anima a throwaway one-tensor contract
    document, invented purely to be allowed to run torch.export, and its own
    suggested remedy (``lanes=()``) silently disabled compilation instead.
    """

    declared_type = model_type(cls)  # also validates cls
    if eager_only_reason(cls):
        return ()
    lanes = getattr(cls, LANES_ATTR, None)
    if lanes:
        return tuple(lanes)
    if lanes is not None:
        # `lanes=()` WRITTEN OUT: the author states no contract for this class,
        # so none is borrowed — not even the model type's canonical one, which
        # would publish a layout claim the author never made. It traces under
        # its own derived name.
        return (derived_lane(cls),)
    canonical = getattr(declared_type, "canonical_contract", None)
    if canonical is not None:
        try:
            lane_dtype(canonical, where=f"{cls.__qualname__} (lanes= omitted)")
        except ModelDeclarationError:
            # A canonical contract that cannot state its own load dtype is not
            # a lane; borrowing its stamp would publish a claim about a layout
            # nobody can read. Derive instead — and say `derived.` in the name.
            return (derived_lane(cls),)
        return (canonical,)
    return (derived_lane(cls),)


def model_requires(cls: type) -> dict[str, Any]:
    """The model class's per-lane machine requirements — publish-time
    extraction, ``{}`` when the header declares none (placement then falls to
    the platform default, which is what ie#740's floors exist to replace)."""

    model_type(cls)  # validates cls
    return dict(getattr(cls, REQUIRES_ATTR, None) or {})


__all__ = [
    "DERIVED_LANE_PRODUCER",
    "DTYPELESS_UPSTREAM_LANES",
    "EAGER_ONLY_ATTR",
    "LANES_ATTR",
    "LaneContract",
    "MODEL_TYPE_ATTR",
    "Model",
    "REQUIRES_ATTR",
    "DerivedLane",
    "derived_lane",
    "eager_only_reason",
    "is_derived_lane",
    "model_requires",
    "ModelDeclarationError",
    "lane_handle",
    "load_marks_compile",
    "model_lanes",
    "model_type",
]
