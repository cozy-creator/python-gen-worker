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

import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:  # keep the base import-weightless
    from .context import LoadContext

MT = TypeVar("MT")

#: Class attributes the header declaration lands on — the publish extractor's
#: read surface.
MODEL_TYPE_ATTR = "__cozy_model_type__"
LANES_ATTR = "__cozy_lanes__"
REQUIRES_ATTR = "__cozy_requires__"


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


def _parse_requires(
    cls: type, requires: Mapping[Any, Any], lanes: tuple[Any, ...] | None
) -> dict[str, Any]:
    """``requires={contract: "vram12g, sm80+"}`` -> ``{handle: LayoutRequirements}``.

    Keyed by LANE, because what a deployment needs of a machine is a property
    of the weight format it runs: the bf16 lane's VRAM floor is not the fp8
    lane's. Values speak the ie#740 requirement grammar (``"vram12g"``,
    ``"sm90+, vram80g"``, ``{"minimum": ..., "recommended": ...}``), parsed
    HERE so the refusal names the author's own class header, and statically
    extractable at publish so placement never has to run author code.
    """
    from ..models.tensor_layout_contract import parse_layout_requirements

    where = f"{cls.__qualname__} requires="
    if not isinstance(requires, Mapping):
        raise ModelDeclarationError(
            f"{where} must be a mapping of lane contract -> requirement, "
            f"got {type(requires).__name__}"
        )
    if not requires:
        raise ModelDeclarationError(
            f"{where}{{}} declares nothing. Omit it to leave placement "
            "UNDECLARED; an empty mapping is not a statement that this model "
            "runs anywhere."
        )
    # `lanes=()` IS a declaration, and every floor over it guards nothing.
    # `None` reaches here only when the model type declares no canonical
    # contract either, so there is genuinely nothing to check the keys against.
    declared = None if lanes is None else {lane_handle(lane) for lane in lanes}
    out: dict[str, Any] = {}
    for lane, value in requires.items():
        handle = lane if isinstance(lane, str) else lane_handle(lane)
        if declared is not None and handle not in declared:
            raise ModelDeclarationError(
                f"{where}[{handle!r}] guards a lane this model does not "
                f"declare. Its lanes are {sorted(declared)} — add the "
                "contract to lanes= or drop the requirement; a requirement "
                "over nothing is never checked."
            )
        out[handle] = parse_layout_requirements(value, where=f"{where}[{handle!r}]")
    return out


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

    ``lanes=`` omitted means one lane — the model type's canonical contract;
    ``lanes=()`` states eager-permanent explicitly. ``requires=`` states what
    each lane needs of the machine in the ie#740 grammar, keyed by the same
    contract objects::

        class SdxlModel(
            Model[SDXL],
            lanes=(contracts.SDXL_DIFFUSERS_BF16,),
            requires={contracts.SDXL_DIFFUSERS_BF16: "vram12g"},
        ): ...

    Omitting it leaves placement UNDECLARED, and the platform's default floor
    is what a deployment then gets. ``__init__`` stays FREE
    (no GPU, no weights): construction and loading are separate moments, and
    derive/introspection instantiate without weights.
    """

    __cozy_model_type__: ClassVar[Any] = None
    __cozy_lanes__: ClassVar[tuple[Any, ...] | None] = None
    __cozy_requires__: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(
        cls,
        *,
        lanes: tuple[Any, ...] | None = None,
        requires: Mapping[Any, Any] | None = None,
        **kwargs: Any,
    ) -> None:
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
        # The lanes this class actually serves — declared, or the model type's
        # canonical contract when `lanes=` is omitted. `requires=` is checked
        # against THESE, not against the raw kwarg (pgw#1391): with `lanes=`
        # omitted the old check had nothing to compare keys to and accepted a
        # floor keyed to any stamp at all, including one naming no document.
        effective_lanes: tuple[Any, ...] | None = lanes
        if lanes is not None:
            if not isinstance(lanes, tuple):
                raise ModelDeclarationError(
                    f"{cls.__qualname__}: lanes= must be a tuple of tensorfs "
                    f"contract objects, got {type(lanes).__name__}"
                )
            for lane in lanes:
                if not isinstance(lane, LaneContract):
                    raise ModelDeclarationError(
                        f"{cls.__qualname__}: lane {lane!r} is not a layout "
                        "contract (no `dtype`); a lane is an imported "
                        "tensorfs contract object, never a name string"
                    )
                # The isinstance above only proves the ATTRIBUTE exists; this
                # READS it, which is the pgw#1391 difference.
                lane_dtype(lane, where=cls.__qualname__)
            cls.__cozy_lanes__ = tuple(lanes)
        elif declared is not None:
            # pgw#1391: OMITTING `lanes=` IS THE se#757 TRAP, so it is checked
            # here too. `model_lanes()` falls through to the model type's
            # `canonical_contract`, and the class that declares nothing is
            # exactly the one that used to claim `sd15.diffusers-bf16@1` — a
            # document that existed nowhere — all the way into a published
            # manifest. Validating the fall-through target at declaration is
            # what makes discovery refuse it, because discovery IMPORTS the
            # author module: the guarantee rides on `__init_subclass__` rather
            # than on any particular consumer remembering to enumerate lanes.
            # (pgw#1394 deleted discovery's `_lane_stamps` for good reasons;
            # this seam is why that deletion costs the fence nothing.)
            canonical = getattr(declared, "canonical_contract", None)
            if canonical is not None:
                lane_dtype(canonical, where=f"{cls.__qualname__} (lanes= omitted)")
                effective_lanes = (canonical,)
        if requires is not None:
            cls.__cozy_requires__ = _parse_requires(cls, requires, effective_lanes)

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
    """The model class's lanes — declared, or the model type's canonical
    contract when ``lanes=`` was omitted. ``()`` is explicit eager-permanent."""

    declared_type = model_type(cls)  # also validates cls
    lanes = getattr(cls, LANES_ATTR, None)
    if lanes is not None:
        return tuple(lanes)
    canonical = getattr(declared_type, "canonical_contract", None)
    if canonical is None:
        raise ModelDeclarationError(
            f"{cls.__qualname__} omits lanes= and its model type "
            f"{declared_type.__name__} has no canonical contract yet "
            "(tensorfs#111); declare lanes= explicitly, or lanes=() for "
            "eager-permanent"
        )
    return (canonical,)


def model_requires(cls: type) -> dict[str, Any]:
    """The model class's per-lane machine requirements — publish-time
    extraction, ``{}`` when the header declares none (placement then falls to
    the platform default, which is what ie#740's floors exist to replace)."""

    model_type(cls)  # validates cls
    return dict(getattr(cls, REQUIRES_ATTR, None) or {})


__all__ = [
    "DTYPELESS_UPSTREAM_LANES",
    "LANES_ATTR",
    "LaneContract",
    "MODEL_TYPE_ATTR",
    "Model",
    "REQUIRES_ATTR",
    "model_requires",
    "ModelDeclarationError",
    "lane_handle",
    "model_lanes",
    "model_type",
]
