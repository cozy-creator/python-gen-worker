"""The trace-time RequestContext (pgw#1370).

The publish derive runs the author's ``setup`` and handlers AS-IS, so it must
answer the ctx surface those methods actually touch -- with trace semantics:
config-only checkpoint tree, platform-fallback defaults, no adapter, no-op
egress. ``ctx.is_trace`` is True and author code may branch on it (the
contract file does, to skip the adapter refusal).

This is NOT the serving context: real checkpoint resolution, adopt/boot and
the deploy-state defaults read are pgw#1372's surface. The two share the
SPELLING of the members below; that spelling is frozen by the Paul-reviewed
``main_v2.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional


class TraceRequestContext:
    """What ``setup``/handlers see under ``gen-worker release derive``."""

    is_trace: bool = True

    def __init__(
        self,
        *,
        lane: Any,
        checkpoint_dir: Path,
        model_type: Optional[type] = None,
        checkpoint_ref: str = "",
    ) -> None:
        self.lane = lane
        self.model_type = model_type
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_ref = checkpoint_ref or f"trace:{self.checkpoint_dir.name}"
        #: The hub-resolved distillation adapter -- never bound at trace time.
        self.adapter: Optional[Any] = None
        self.log = logging.getLogger("gen_worker.release.trace")
        #: Modules the author marked via ctx.compile() -- discovery hooks
        #: exactly these during the payload drives.
        self.marked_modules: list[Any] = []

    # -- defaults -----------------------------------------------------------
    def defaults(self) -> Any:
        """The PLATFORM-FALLBACK defaults for the class-header model type.

        ``Endpoint[SDXL]`` is the single source of the type; at serve the
        checkpoint's hub row decodes as ``SDXL.Defaults`` with missing
        fields filled from platform values, and at trace the zero-arg
        construction IS the platform row. Typed via the generic.
        """
        if self.model_type is None:
            raise TypeError(
                "ctx.defaults() reads the model type off the class header "
                "(class X(Endpoint[SDXL])); this endpoint's base is "
                "unparameterized"
            )
        defaults_type = getattr(self.model_type, "Defaults", self.model_type)
        return defaults_type()

    # -- compile marking ----------------------------------------------------
    def compile(self, target: Any) -> Any:
        """torch.compile-style marking, trace half (pgw#1370/#1372 contract).

        At DERIVE this records the marked module (discovery hooks it during
        the payload drives) and returns it unchanged. At SERVE (pgw#1372) it
        returns the adopted compiled graph for this (graph, lane, sm) when
        the store has it, else the module unchanged while the hole mints in
        the background -- the author's marked line IS the swap point.

        A non-module with ``.components`` (a diffusers pipeline) is sugar:
        every nn.Module component is marked ("compile everything
        compilable"). Typos are real AttributeErrors at the author's line --
        no strings, no self-structure assumptions.
        """
        import torch

        if isinstance(target, torch.nn.Module):
            if all(existing is not target for existing in self.marked_modules):
                self.marked_modules.append(target)
            return target
        components = getattr(target, "components", None)
        if isinstance(components, Mapping):
            for component in components.values():
                if isinstance(component, torch.nn.Module):
                    self.compile(component)
            return target
        raise TypeError(
            f"ctx.compile() marks nn.Modules (or a pipeline-like object "
            f"whose .components carries them); got {type(target).__name__}"
        )

    # -- knobs / control ----------------------------------------------------
    def clamp(
        self,
        field: str,
        requested: float,
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        reason: str = "",
    ) -> float:
        """Same arithmetic as the serving ctx; trace records nothing."""
        del reason
        applied = float(requested)
        if lo is not None and applied < lo:
            applied = float(lo)
        if hi is not None and applied > hi:
            applied = float(hi)
        return applied

    def raise_if_cancelled(self, message: str = "request cancelled") -> None:
        del message

    # -- egress -------------------------------------------------------------
    def step_callback(self, total_steps: int) -> Callable[..., dict[str, Any]]:
        """A no-op diffusers ``callback_on_step_end``."""
        del total_steps

        def callback(
            _pipe: Any, _index: Any, _timestep: Any,
            callback_kwargs: Any = None, **_: Any,
        ) -> dict[str, Any]:
            del callback_kwargs
            return {}

        return callback

    def save_image(self, image: Any, *, format: str = "webp", **_: Any) -> Any:
        """A stub asset: nothing is encoded or uploaded at trace time."""
        del image
        from ..api.types import ImageAsset

        return ImageAsset(ref=f"trace://image.{format}")


__all__ = ["TraceRequestContext"]
