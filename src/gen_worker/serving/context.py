"""The serving RequestContext surface (pgw#1372) — what ``main_v2.py`` uses.

``ServeContext`` extends the base :class:`gen_worker.RequestContext` with the
ship-code-as-is serving facts: the worker-resolved checkpoint tree (already
converted through the active lane's tensor-layout contract by tensorfs), the
active execution lane, typed per-checkpoint tunable overrides, the hub-bound
adapter, and the trace flag. Everything else (``step_callback``,
``save_image``, ``clamp``, ``raise_if_cancelled``, ``progress``, ``log``…)
is the base class, salvaged rather than rebuilt.

Per-checkpoint serving values are MUTABLE PLATFORM DEPLOY STATE (hub DB), not
release metadata: the host holds one :class:`DeployBinding` and stamps it on
each context, so a rebind swaps the binding without touching release
identity — new graphs a rebind introduces are simply holes (partial-hit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Type, TypeVar

import msgspec

from ..request_context import RequestContext

T = TypeVar("T", bound=msgspec.Struct)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BoundAdapter:
    """The hub-bound LoRA riding this deployment: its name and local path."""

    name: str
    path: Path


@dataclass(slots=True)
class DeployBinding:
    """One deployment's checkpoint binding — hub deploy state, rebindable.

    ``checkpoint_dir`` is the worker-resolved checkpoint tree, converted
    through the active lane's tensor-layout contract by tensorfs before the
    endpoint ever sees it. ``defaults`` are the hub's per-checkpoint tunable
    overrides — raw mapping here, typed by ``ctx.checkpoint_defaults(T)``
    against the schema the endpoint declares.
    """

    checkpoint_ref: str
    checkpoint_dir: Path
    defaults: Mapping[str, Any] = field(default_factory=dict)
    adapter: Optional[BoundAdapter] = None


class ServeContext(RequestContext[Any]):
    """The serving context: base RequestContext + the main_v2 surface."""

    def __init__(
        self,
        request_id: str,
        *,
        binding: DeployBinding,
        lane: Any = None,
        is_trace: bool = False,
        compile_sink: Optional[Any] = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(request_id, **base_kwargs)
        self._binding = binding
        self._lane = lane
        self._is_trace = bool(is_trace)
        self._compile_sink = compile_sink

    # The base class spells `checkpoint_dir(*, key)` as JOB-SCOPED scratch;
    # the serving surface's `ctx.checkpoint_dir` is the checkpoint TREE. The
    # jobs spelling stays on job contexts (the base class); serving handlers
    # get the property. One name, two worlds, until pgw#1373 renames the
    # jobs scratch helper out of the way.
    @property
    def checkpoint_dir(self) -> Path:
        """The worker-resolved checkpoint tree for this deployment, already
        converted to the active lane's tensor-layout contract."""
        return Path(self._binding.checkpoint_dir)

    @property
    def checkpoint_ref(self) -> str:
        """The hub ref of the bound checkpoint — honest output metadata."""
        return self._binding.checkpoint_ref

    @property
    def lane(self) -> Any:
        """The active execution lane (the deploy's pick). Raises for an
        endpoint that declared no lanes — eager-permanent code has no lane
        to read, and a silent default would invent one."""
        if self._lane is None:
            raise RuntimeError(
                "ctx.lane: this endpoint declared no execution lanes "
                "(eager-permanent); there is no active lane to read"
            )
        return self._lane

    @property
    def adapter(self) -> Optional[BoundAdapter]:
        """The hub-bound LoRA (name + local path), or None."""
        return self._binding.adapter

    @property
    def is_trace(self) -> bool:
        """True under the publish-time discovery drive (sample inputs,
        fake weights): handlers may relax deploy-state checks that cannot
        hold there, and MUST NOT persist outputs."""
        return self._is_trace

    def checkpoint_defaults(self, schema: Type[T]) -> T:
        """The hub's per-checkpoint tunable overrides, typed.

        Decodes the deploy binding's raw override mapping against the
        endpoint's own schema: fields the hub says nothing about take the
        schema's defaults; an override that does not fit the declared type
        is a typed error naming the field, not a silent coercion to garbage.
        """
        try:
            return msgspec.convert(dict(self._binding.defaults), type=schema)
        except msgspec.ValidationError as exc:
            raise msgspec.ValidationError(
                f"per-checkpoint overrides for {self._binding.checkpoint_ref!r} "
                f"do not fit {schema.__name__}: {exc}"
            ) from None

    def compile(self, target: Any) -> Any:
        """Imperative module marking, the torch.compile idiom (Paul ruling
        2026-08-18)::

            self.pipe.unet = ctx.compile(self.pipe.unet)

        At publish time this records + hooks the module for the derive; at
        serve boot it consults the store per graph for the active
        (lane, sm) — hit: the module comes back with the compiled graph
        armed; miss: the module comes back UNCHANGED (eager) and the graph
        joins the ordered hole list the background mint consumes
        (pgw#1371). ``ctx.compile(pipe)`` walks the pipeline's nn.Module
        components. With no adoption source bound (local runs, the eager
        bridge) it returns ``target`` untouched — marking is always safe.
        """
        if self._compile_sink is None:
            return target
        return self._compile_sink(target)

    def step_callback(self, num_inference_steps: int, **kwargs: Any) -> Any:
        """A diffusers ``callback_on_step_end`` wired to progress + cancel —
        the context spelling of :func:`gen_worker.diffusers_step_callback`."""
        from ..api.progress import diffusers_step_callback

        return diffusers_step_callback(self, num_inference_steps, **kwargs)


__all__ = ["BoundAdapter", "DeployBinding", "ServeContext"]
