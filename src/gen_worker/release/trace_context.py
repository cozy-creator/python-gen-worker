"""The trace-time contexts (pgw#1370).

The publish derive runs the author's ``Model.load`` and entrypoints AS-IS,
so it must answer exactly the ctx surface that code touches -- with trace
semantics: config-only checkpoint tree, hollow instantiation, platform-
fallback defaults, no adapter, no-op egress. ``ctx.is_trace`` is True and
author code may branch on it (the contract file does, to skip the adapter
refusal).

These are NOT the serving contexts: chunk-store streaming, adopt/boot and
the deploy-state defaults read are pgw#1372's surface. The two sides share
the SPELLING; that spelling is frozen by the Paul-reviewed ``main_v2.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional

from ..api.model_base import LoadContext


class TraceLoadContext(LoadContext[Any]):
    """What ``Model.load`` sees under ``gen-worker release derive``.

    There is NO ``is_trace`` -- Paul deleted it from the author surface
    (author code branching on it corrupts compilation coverage; author code
    is trace-oblivious by construction). Arm coverage is the DERIVE'S job,
    via input/binding enumeration (payload enums x adapter states x
    checkpoint-defaults variants).
    """

    def __init__(
        self,
        *,
        lane: Any,
        checkpoint_dir: Path,
        model_type: Optional[type] = None,
        defaults_instance: Any = None,
    ) -> None:
        self.lane = lane
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.defaults_instance = defaults_instance
        self.log = logging.getLogger("gen_worker.release.trace")
        #: Modules the author marked via ctx.compile() -- discovery hooks
        #: exactly these during the payload drives.
        self.marked_modules: list[Any] = []

    def load(self, loader: Any) -> Any:
        """Hollow-materialize the CONFIG-ONLY tree through the author's loader.

        At serve, ``ctx.load`` streams tensors from the chunk store straight
        to VRAM in the lane contract's layout (pgw#1372). At trace there are
        no tensors at all: the loader's own ``from_pretrained`` runs against
        the config-only subset snapshot inside the ambient
        ``torchcg.hollow_session`` (fake parameters, real buffers), and the
        lane's registry-derived dtype stands in for the layout's.
        """
        from_pretrained = getattr(loader, "from_pretrained", None)
        if from_pretrained is None:
            raise TypeError(
                f"ctx.load() needs a loader with from_pretrained "
                f"(a diffusers/transformers class); got {loader!r}"
            )
        loaded = from_pretrained(
            self.checkpoint_dir, torch_dtype=getattr(self.lane, "dtype", None)
        )
        # Adapter application mutates WEIGHTS (or injects adapter layers);
        # at trace every parameter is fake and no adapter bytes exist, so the
        # enumeration's fake-adapter arms must not hit real LoRA I/O. The
        # graphs observed on those arms are the base modules' -- a served
        # adapter that changes the module graph re-keys and first-encounter
        # mints (pgw#1371/#1372 own the branch-bearing lora story).
        for lora_call in ("load_lora_weights", "set_adapters", "unload_lora_weights"):
            if hasattr(loaded, lora_call):
                setattr(loaded, lora_call, _noop)
        return loaded

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

    def defaults(self) -> Any:
        """The enumerated defaults VARIANT for the class-header model type.

        ``Model[SDXL]`` is the single source of the type; at serve the
        checkpoint's hub row decodes as ``SDXL.Defaults`` with missing
        fields filled from platform values. At trace the derive enumerates
        recipe-relevant variants (platform row; cfg flipped when the schema
        carries it) because they change the executed arm and thus the
        observed graphs.
        """
        if self.defaults_instance is not None:
            return self.defaults_instance
        if self.model_type is None:
            raise TypeError(
                "ctx.defaults() reads the model type off the class header "
                "(class X(Model[SDXL], ...)); this model's base is "
                "unparameterized"
            )
        defaults_type = getattr(self.model_type, "Defaults", self.model_type)
        return defaults_type()


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class TraceRequestContext:
    """What entrypoints see under ``gen-worker release derive``.

    No ``is_trace`` here either (deleted from the author surface); warns and
    clamps are collected as log lines.
    """

    def __init__(
        self,
        *,
        lane: Any,
        checkpoint_ref: str = "",
    ) -> None:
        self.lane = lane
        self.checkpoint_ref = checkpoint_ref or "trace:config-only"
        #: The hub-resolved distillation adapter -- never bound at trace time.
        self.adapter: Optional[Any] = None
        self.log = logging.getLogger("gen_worker.release.trace")

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

    def warn(self, message: str) -> None:
        """Caller-visible advisory at serve; a log line at trace."""
        self.log.warning("trace: %s", message)

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


__all__ = ["TraceLoadContext", "TraceRequestContext"]
