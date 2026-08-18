"""The ctx split (pgw#1382): ``LoadContext[MT]`` + ``RequestContext``.

Each half carries only what its moment can answer for:

* :class:`LoadContext` -> ``Model.load``/``Model.unload``: the checkpoint
  tree, the lane (a tensorfs layout contract), and the three seams —
  ``load()`` (pgw#1380's native store->VRAM engine lands here; until then a
  ``from_pretrained`` eager bridge), ``compile()`` (tcg#42's AdoptSession
  delegation), ``defaults()`` (pgw#1377's typed decode).
* :class:`RequestContext` -> entrypoints: the ``checkpoint_ref`` fact
  (RESOLVED pinned ref of what actually serves) + the salvaged base surface
  (``raise_if_cancelled``, ``save_image``, ``clamp``, ``generator``,
  ``progress``, ``log`` …) + ``step_callback`` + ``warn``.

Adapters are EXPLICIT ENTRYPOINT PARAMETERS (Paul ruling), not ctx facts:
``def generate(ctx, payload, model: SdxlModel, turbo: Adapter | None,
loras: list[Adapter])`` — the hub resolves what rides per deployment/request
into the declared slots; APPLYING them stays a model mutation through the
model-owned scope (``with model.adapters(riding): ...``).

There is deliberately NO trace flag on any ctx (Paul ruling): author code is
trace-oblivious by construction — the publish-time derive varies BINDINGS
(input enumeration: synthesized adapters, cfg-false defaults) and its
harness-private context no-ops the output surfaces; it never asks author
code to cooperate.

Per-checkpoint serving values are MUTABLE PLATFORM DEPLOY STATE (hub DB),
not release metadata: the host holds one :class:`DeployBinding` and stamps
it on each context, so a rebind swaps the binding without touching release
identity — new graphs a rebind introduces are simply holes (partial-hit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Callable, Generic, Mapping, Optional, Protocol, Type,
    TypeVar, cast,
)

import msgspec

from ..families.base import GenerationDefaults
from ..request_context import RequestContext as _BaseRequestContext

if TYPE_CHECKING:
    from ..models.defaults_decode import CarriesDefaults
    from ..models.model_types import ModelType

P = TypeVar("P")
D = TypeVar("D", bound=GenerationDefaults)
DT = TypeVar("DT", bound=msgspec.Struct)
MT_co = TypeVar("MT_co", covariant=True)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Adapter:
    """A hub-resolved adapter (LoRA). ``ref`` is its FULLY-PINNED hub
    identity (``org/repo@release`` — what evidence/output reports; hub
    resolution fills it pinned even when the pick was floating); ``name`` is
    the diffusers adapter-registry label (``load_lora_weights``'s
    ``adapter_name``); ``defaults`` is the decoded overlay (pgw#1377's
    ``adapter.defaults``, e.g. ``SDXL.Lora.Defaults``); ``scale`` is
    envelope-resolved, platform-clamped through the LoRA's strength Knob
    before the worker sees it. Entrypoints declare adapter SLOTS as explicit
    parameters — ``turbo: Adapter | None`` (single, optional) or
    ``loras: list[Adapter]`` (the request's picks; empty when none ride) —
    param name = slot name; the worker fills them from deploy/request state.
    Applying adapters is a MODEL mutation and goes through a model-owned
    scope (``model.adapters(riding)``)."""

    name: str
    path: Path
    defaults: Any = None
    scale: float = 1.0
    ref: str = ""


@dataclass(frozen=True, slots=True)
class DistillationAdapter(Adapter):
    """A distillation adapter — runtime-identical to :class:`Adapter`; its
    MEANING is the SLOT KIND (Paul's structural guard). A param annotated
    ``DistillationAdapter | None`` is a distillation-adapter slot: release
    metadata records the kind, the hub refuses envelope picks whose adapter
    row lacks the distillation marker (typed 400 naming the adapter), and
    resolution constructs THIS class for marked rows. Takeover power over
    the serving config used to be positional; now it is typed — a style
    LoRA cannot seize the config even on a misconfigured deployment."""


@dataclass(slots=True)
class DeployBinding:
    """One deployment's checkpoint binding — hub deploy state, rebindable.

    ``checkpoint_dir`` is the worker-resolved checkpoint tree, converted
    through the active lane's tensor-layout contract by tensorfs before any
    model sees it. ``defaults`` is the hub's per-checkpoint JSONB row (raw
    mapping) — typed by ``LoadContext.defaults()`` against the model type's
    ``Defaults`` schema."""

    checkpoint_ref: str
    checkpoint_dir: Path
    #: The hub row's ``model`` classification column. ``None``/empty = the
    #: checkpoint is unclassified: it serves on platform fallbacks with the
    #: named visible warning (pgw#1377's read-side matrix), never a guess.
    model: Optional[str] = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    adapter: Optional[Adapter] = None


class LoaderEngine(Protocol):
    """The pgw#1380 seam: build a pipeline with weights streamed chunk-store
    -> pinned -> CUDA (serve) or on meta (derive). gen-worker's native
    loader lands behind this; tests inject theirs."""

    def build(
        self, pipeline_cls: type, *, checkpoint_dir: Path, lane: Any
    ) -> Any: ...


class DefaultsError(msgspec.ValidationError):
    """A hub defaults row does not fit the model type's schema."""


class LoadContext(Generic[MT_co]):
    """What ``Model.load``/``Model.unload`` receive — the load moment."""

    def __init__(
        self,
        *,
        binding: DeployBinding,
        model_type: Optional[type] = None,
        lane: Any = None,
        engine: Optional[LoaderEngine] = None,
        compile_sink: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self._binding = binding
        self._model_type = model_type
        self._lane = lane
        self._engine = engine
        self._compile_sink = compile_sink

    @property
    def checkpoint_dir(self) -> Path:
        """The worker-resolved checkpoint tree, already converted to the
        active lane's tensor-layout contract."""
        return Path(self._binding.checkpoint_dir)

    @property
    def checkpoint_ref(self) -> str:
        """The hub ref of the bound checkpoint."""
        return self._binding.checkpoint_ref

    @property
    def lane(self) -> Any:
        """The active lane (the deploy's pick) — a tensorfs layout contract;
        ``ctx.lane.dtype`` is the load dtype. Raises for a model that
        declared ``lanes=()`` — eager-permanent code has no lane to read,
        and a silent default would invent one."""
        if self._lane is None:
            raise RuntimeError(
                "ctx.lane: this model declared no execution lanes "
                "(eager-permanent); there is no active lane to read"
            )
        return self._lane

    def load(self, pipeline_cls: Type[P]) -> P:
        """Build ``pipeline_cls`` with this checkpoint's weights resident —
        the ONE spelling (pgw#1380)::

            self.pipe = ctx.load(StableDiffusionXLPipeline)

        No ``torch_dtype=`` (the lane contract IS the dtype), no
        ``.to("cuda")`` (weights land on device). Behind the call is
        pgw#1380's ``gen_worker.serving.streaming`` engine: the pipeline
        built from configs on ``meta``, then its weights walked out of the
        chunk store in FILE order through pinned staging onto the device,
        with no tensor file written or read anywhere. The host binds it off
        the projected checkpoint tree (which carries its own store), so
        PLACEMENT stays a worker decision and this code names no device.

        The eager ``from_pretrained`` bridge survives for a tree with no
        chunk store behind it — a bare download, a local run, a fixture —
        where there is nothing to stream from."""
        if self._engine is not None:
            built: P = self._engine.build(
                pipeline_cls,
                checkpoint_dir=self.checkpoint_dir,
                lane=self._lane,
            )
            return built
        from_pretrained = getattr(pipeline_cls, "from_pretrained", None)
        if from_pretrained is None:
            raise RuntimeError(
                f"ctx.load({pipeline_cls.__name__}): no loader engine is "
                "bound (pgw#1380 lands the native store->VRAM engine here) "
                "and the class has no from_pretrained for the eager bridge"
            )
        logger.info(
            "ctx.load: eager from_pretrained bridge for %s (pgw#1380's "
            "native loader engine is not bound)", pipeline_cls.__name__,
        )
        if self._lane is not None and getattr(self._lane, "dtype", None) is not None:
            bridged: P = from_pretrained(self.checkpoint_dir, torch_dtype=self._lane.dtype)
            return bridged
        no_lane: P = from_pretrained(self.checkpoint_dir)
        return no_lane

    def compile(self, target: Any) -> Any:
        """Imperative module marking, the torch.compile idiom (Paul ruling)::

            self.pipe.unet = ctx.compile(self.pipe.unet)

        Everything behind the call is torchcg (tcg#42): at publish the
        hollow session records + hooks; at serve the AdoptSession arms what
        the store holds for (lane, sm) and returns the module UNCHANGED
        (eager) on a miss while the graph joins the ordered hole list the
        background mint consumes (pgw#1371). ``ctx.compile(pipe)`` walks a
        pipeline's nn.Module components. With no adoption source bound
        (local runs, the eager bridge) it returns ``target`` untouched —
        marking is always safe."""
        if self._compile_sink is None:
            return target
        return self._compile_sink(target)

    def defaults(self: "LoadContext[ModelType[DT]]") -> DT:
        """THIS checkpoint's defaults, typed via the model class's generic:
        the hub's per-checkpoint row decoded as a field-level overlay on the
        model type's ``Defaults`` struct (zero-arg = platform fallbacks —
        pgw#1377's decode matrix; ill-typed values are a typed refusal
        naming the checkpoint, never a silent coercion).

        The decode is pgw#1377's ``defaults_decode`` AUTHORITY, never a second
        msgspec pass: only that path narrows a Knob's [lo, hi] across the
        platform and checkpoint layers and re-stamps each Knob with its field
        name, which is what keeps two clamps distinguishable in the
        caller-visible adjustment ledger."""
        from ..models.defaults_decode import (
            DefaultsDecodeError,
            ModelTypeMismatch,
            decode_model_defaults,
        )

        schema = getattr(self._model_type, "Defaults", None)
        if self._model_type is None or schema is None:
            raise RuntimeError(
                "ctx.defaults(): no model type is bound to this LoadContext "
                "(the model class header's generic parameter carries it)"
            )
        # `_model_type` is erased to `type` on the instance; the self-type on
        # this method is what carries `DT` for the caller.
        carrier = cast("CarriesDefaults[DT]", self._model_type)
        try:
            decoded: DT = decode_model_defaults(
                carrier,
                model=self._binding.model,
                defaults=self._binding.defaults,
            )
        except (DefaultsDecodeError, ModelTypeMismatch) as exc:
            raise DefaultsError(
                f"per-checkpoint defaults for "
                f"{self._binding.checkpoint_ref!r}: {exc}"
            ) from None
        return decoded


class RequestContext(_BaseRequestContext[D]):
    """What entrypoints receive — request facts + the salvaged base surface.

    The base class (`gen_worker.request_context`) contributes
    ``raise_if_cancelled``, ``save_image``, ``clamp``, ``generator``,
    ``progress``, ``log`` and the rest; this subclass adds the pgw#1382
    request facts. Constructible bare (``RequestContext("req-1")``) for
    local/unit use — deployment facts then read as absent."""

    def __init__(
        self,
        request_id: str,
        *,
        binding: Optional[DeployBinding] = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(request_id, **base_kwargs)
        self._serve_binding = binding

    @property
    def checkpoint_ref(self) -> str:
        """The RESOLVED, fully-pinned hub ref of the checkpoint that
        actually served this request — honest structured output evidence
        (the ``model`` field of an output struct)."""
        if self._serve_binding is None:
            raise RuntimeError(
                "ctx.checkpoint_ref: no deploy binding rides this context "
                "(local construction without binding=)"
            )
        return self._serve_binding.checkpoint_ref

    def step_callback(self, num_inference_steps: int, **kwargs: Any) -> Any:
        """A diffusers ``callback_on_step_end`` wired to progress + cancel —
        the context spelling of :func:`gen_worker.diffusers_step_callback`."""
        from ..api.progress import diffusers_step_callback

        return diffusers_step_callback(self, num_inference_steps, **kwargs)

    def warn(self, message: str) -> None:
        """A caller-visible WARNING into the response envelope — the
        generalization of clamp visibility, same delivery path (the
        adjustment ledger rides ``JobResult.adjustments`` + the hub's
        request record/events). Warn-and-serve: for a request aspect that
        cannot apply on this serving (e.g. guidance on a cfg-free recipe),
        never a silent drop and never an aborted request."""
        self.adjusted("", None, None, str(message))

    @property
    def warnings(self) -> tuple[str, ...]:
        """The accumulated :meth:`warn` messages, in emission order (the
        field-less rows of the adjustment ledger)."""
        return tuple(
            row["reason"] for row in self.adjustments if not row["field"]
        )


__all__ = [
    "Adapter",
    "DefaultsError",
    "DistillationAdapter",
    "DeployBinding",
    "LoadContext",
    "LoaderEngine",
    "RequestContext",
]
