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

from ..serving.context import _rejected_torch_dtype

class TraceLoadContext:
    """What ``Model.load`` sees under ``gen-worker release derive``.

    Deliberately duck-typed (the author annotates ``LoadContext[MT]``; the
    harness-private object answers the same spelling): the serving
    ``LoadContext`` carries a deploy binding and property-backed facts this
    trace half has no business constructing.

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
        # pgw#1447, SECOND SITE. This is a separate implementation of the same
        # eager bridge `serving/context.py` carries, and it had the identical
        # defect: a `ModularPipeline` does not consume `torch_dtype` in
        # `from_pretrained`, so `**kwargs` funnels it into the constructor and
        # a strict pipeline refuses it. Fixing only the serve-path copy left
        # the DERIVE — the one path that ALWAYS takes an eager bridge, because
        # a trace has no streaming engine — still broken. Two bridges, one
        # contract; the predicate is shared so they cannot drift.
        dtype = getattr(self.lane, "dtype", None)
        try:
            loaded = from_pretrained(self.checkpoint_dir, torch_dtype=dtype)
        except TypeError as exc:
            if not _rejected_torch_dtype(exc):
                raise
            loaded = from_pretrained(self.checkpoint_dir)
            to = getattr(loaded, "to", None)
            if dtype is not None and callable(to):
                to(dtype)
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


class StepBudgetReached(Exception):
    """The trace has seen every shape this denoise loop produces.

    A diffusers denoise loop runs the SAME shapes on every step -- step 3 of
    28 teaches the trace nothing step 1 did not. The derive therefore runs
    each enumerated pass under a STEP BUDGET and lets the author's own
    ``callback_on_step_end`` raise this once the budget is spent; the drive
    treats it as a completed pass. Modules that run AFTER the loop (a marked
    VAE decoder) never execute under a budget, so the derive re-drives
    unbudgeted whenever a marked module is still unobserved -- honesty is
    preserved, the redundant 27 steps are not paid.
    """


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
        step_budget: Optional[int] = None,
    ) -> None:
        self.lane = lane
        #: None = run the author's full step count.
        self.step_budget = step_budget
        self.checkpoint_ref = checkpoint_ref or "trace:config-only"
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
        """A diffusers ``callback_on_step_end`` that enforces the step budget."""
        del total_steps
        seen = 0
        budget = self.step_budget

        def callback(
            _pipe: Any, _index: Any, _timestep: Any,
            callback_kwargs: Any = None, **_: Any,
        ) -> dict[str, Any]:
            del callback_kwargs
            nonlocal seen
            seen += 1
            if budget is not None and seen >= budget:
                raise StepBudgetReached
            return {}

        return callback

    def save_image(self, image: Any, *, format: str = "webp", **_: Any) -> Any:
        """A stub asset: nothing is encoded or uploaded at trace time."""
        del image
        from ..api.types import ImageAsset

        return ImageAsset(ref=f"trace://image.{format}")


__all__ = ["StepBudgetReached", "TraceLoadContext", "TraceRequestContext"]
