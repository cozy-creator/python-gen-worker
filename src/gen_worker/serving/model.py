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
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:  # keep the base import-weightless
    from .context import LoadContext

MT = TypeVar("MT")

#: Class attributes the header declaration lands on — the publish extractor's
#: read surface.
MODEL_TYPE_ATTR = "__cozy_model_type__"
LANES_ATTR = "__cozy_lanes__"


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
    """The lane's stable string handle — ``contract`` (``name@version``) now;
    tensorfs#111's digest stamp joins the fallback chain when it ships."""

    for attribute in ("contract", "digest"):
        value = getattr(lane, attribute, None)
        if isinstance(value, str) and value:
            return value
    raise ModelDeclarationError(
        f"lane {lane!r} carries no string handle (`contract` or `digest`); "
        "a lane is a tensorfs layout contract object"
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

    ``lanes=`` omitted means one lane — the model type's canonical contract;
    ``lanes=()`` states eager-permanent explicitly. ``__init__`` stays FREE
    (no GPU, no weights): construction and loading are separate moments, and
    derive/introspection instantiate without weights.
    """

    __cozy_model_type__: ClassVar[Any] = None
    __cozy_lanes__: ClassVar[tuple[Any, ...] | None] = None

    def __init_subclass__(
        cls, *, lanes: tuple[Any, ...] | None = None, **kwargs: Any
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
                lane_handle(lane)
            cls.__cozy_lanes__ = tuple(lanes)

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


__all__ = [
    "LANES_ATTR",
    "LaneContract",
    "MODEL_TYPE_ATTR",
    "Model",
    "ModelDeclarationError",
    "lane_handle",
    "model_lanes",
    "model_type",
]
