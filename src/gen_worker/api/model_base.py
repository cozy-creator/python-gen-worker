"""``gen_worker.Model`` -- the stateful half of the ship-code-as-is surface.

Paul's main_v2.py review rulings (pgw#1367 program, 2026-08-19 final form):
the author splits an endpoint into a MODEL class and free ENTRYPOINT
functions::

    class SdxlModel(Model[SDXL], lanes=(contracts.SDXL_DIFFUSERS_BF16, ...)):
        def load(self, ctx: LoadContext[SDXL]) -> None:
            self.pipe = ctx.load(StableDiffusionXLPipeline)
            self.pipe.unet = ctx.compile(self.pipe.unet)
            self.defaults = ctx.defaults()

    @entrypoint
    def generate(payload: In, model: SdxlModel, ctx: RequestContext) -> Out: ...

The class header (``Model[SDXL]``) is THE single source of the model type --
statically extractable at publish. ``lanes=`` (class kwargs) is the whole
compile/layout declaration: each lane IS a tensorfs layout-contract
reference. One instance per (checkpoint x lane), LRU-resident,
single-flight; ``__init__`` stays FREE (construction and loading are
separate moments -- derive/introspection instantiate without weights);
``load(ctx)`` runs once per instance; ``unload(ctx)`` mirrors it. Every
platform capability arrives via ctx ONLY. Serving-side residency/affinity:
pgw#1372.
"""

from __future__ import annotations

import typing
from typing import Any, Generic, Optional, TypeVar

M = TypeVar("M")

#: The lanes a Model subclass declared via class kwargs.
MODEL_LANES_ATTR = "__gen_worker_lanes__"


def lane_contract_handle(owner: str, lane: Any) -> str:
    """One lane's contract HANDLE, duck-typed.

    A lane IS a tensorfs layout-contract reference (Paul's contract-objects
    ruling): an imported registry object, an inline ``tensorfs.Contract``
    (anonymous, digest-stamped), or the bare handle string. gen-worker
    refuses to re-declare the contract vocabulary -- it reads the handle off
    whichever shape arrives (tensorfs#111-114 own the object's final form):
    ``handle``/``stamp`` attributes, ``name``+``version``, or the string
    itself (``<producer>.<format>@<major>`` or ``sha256:<hex>``).
    """
    handle: Any = None
    if isinstance(lane, str):
        handle = lane
    else:
        handle = getattr(lane, "handle", None)
        if not isinstance(handle, str):
            # tensorfs#112: an anonymous custom contract is digest-identified.
            handle = getattr(lane, "stamp", None)
        if not isinstance(handle, str):
            name = getattr(lane, "name", None)
            version = getattr(lane, "version", None)
            if isinstance(name, str) and isinstance(version, int):
                handle = f"{name}@{version}"
    if not isinstance(handle, str) or not handle.strip() or not (
        "@" in handle or handle.startswith("sha256:")
    ):
        raise ValueError(
            f"{owner}: a lane must be a tensorfs layout-contract reference "
            f"(registry object, inline Contract, or "
            f"'<producer>.<format>@<major>' handle); got {lane!r}"
        )
    return handle


def _validate_lanes(owner: str, lanes: Any) -> tuple[Any, ...]:
    out = tuple(lanes)
    if not out:
        raise ValueError(
            f"{owner}: lanes= must declare at least one contract reference "
            f"(omit lanes= entirely for the model type's canonical contract)"
        )
    handles = [lane_contract_handle(owner, lane) for lane in out]
    if len(set(handles)) != len(handles):
        raise ValueError(f"{owner}: lane contracts must be unique, got {handles!r}")
    return out


class Model(Generic[M]):
    """Base class for the stateful model half (see module docstring)."""

    def __init_subclass__(cls, *, lanes: Any = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if lanes is not None:
            setattr(cls, MODEL_LANES_ATTR, _validate_lanes(cls.__name__, lanes))

    def load(self, ctx: "LoadContext[M]") -> None:  # noqa: B027 - deliberate no-op hook
        """Acquire weights and mark compile targets; the default does nothing."""

    def unload(self, ctx: "LoadContext[M]") -> None:  # noqa: B027 - deliberate no-op hook
        """Release what load acquired -- eviction is framework-generic."""


class LoadContext(Generic[M]):
    """What ``Model.load`` receives; authors annotate against this type.

    The TRACE implementation lives in ``gen_worker.release`` (hollow,
    config-only, CPU); the SERVING implementation (chunk-store streaming,
    adopt/boot) is pgw#1372's. Both spell exactly this surface.
    """

    #: The resolved lane: contract handle + registry-derived dtype.
    lane: Any
    is_trace: bool = False

    def load(self, loader: Any) -> Any:
        """Materialize the checkpoint through ``loader`` (a pipeline class)."""
        raise NotImplementedError

    def compile(self, target: Any) -> Any:
        """torch.compile-style marking; the marked line is the swap point."""
        raise NotImplementedError

    def defaults(self) -> Any:
        """This checkpoint's defaults, typed via the ``Model[...]`` header."""
        raise NotImplementedError


class Adapter:
    """A hub-resolved adapter FACT on ctx (distillation LoRA et al.).

    Serving resolution (pgw#1372) fills these from the deployment binding;
    at trace no adapter is ever bound. ``defaults`` is the adapter's own
    metadata struct (e.g. ``SDXL.Lora.Defaults``) -- the recipe belongs to
    the ADAPTER, never to endpoint vocabulary.
    """

    def __init__(
        self,
        *,
        name: str,
        path: Any,
        defaults: Any = None,
        scale: float = 1.0,
        ref: str = "",
    ) -> None:
        self.name = name
        self.path = path
        self.defaults = defaults
        self.scale = scale
        #: The fully-pinned hub identity it was resolved from (org/repo@release).
        self.ref = ref or name


def model_lanes(cls: type) -> tuple[Any, ...]:
    return tuple(getattr(cls, MODEL_LANES_ATTR, ()))


def model_model_type(cls: type) -> Optional[type]:
    """The model type in the class header (``Model[SDXL]`` -> ``SDXL``).

    Static by construction: read off ``__orig_bases__``, never off an
    instance. ``None`` when the base is unparameterized.
    """

    for base in getattr(cls, "__orig_bases__", ()):
        if typing.get_origin(base) is Model:
            arguments = typing.get_args(base)
            if len(arguments) == 1 and isinstance(arguments[0], type):
                return arguments[0]
    return None


__all__ = [
    "Adapter",
    "LoadContext",
    "MODEL_LANES_ATTR",
    "Model",
    "lane_contract_handle",
    "model_lanes",
    "model_model_type",
]
