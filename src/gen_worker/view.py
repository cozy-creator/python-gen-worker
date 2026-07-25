"""Per-request pipeline VIEWS (SDK v2, pgw#647 / WORKER-RESIDENCY-DESIGN
"Shared components and mutation safety").

One live instance == one binding set == one materialized graph. Everything
PER-REQUEST (sampler/scheduler state, latents, generator/seed, steps,
guidance, prompt, callbacks) lives in a lightweight VIEW over the SAME
weight tensors — microseconds to build, zero weight VRAM; it allocates only
a fresh scheduler. Everything else (dtype, memory format, attention
slicing, offload hooks, attention-processor swaps) is INSTANCE IDENTITY,
fixed at materialization: two requests needing different values there need
different instances, exactly like a different checkpoint.

The view WRAPS the module objects, never swaps them — the compiled graph is
bound to those modules, so swapping causes guard failures and recompiles.
Concretely: the diffusers pipeline OBJECT is a thin container of module
references; the view is a container copy sharing every module by reference,
with its OWN scheduler cloned from the instance scheduler's config.

This fixes a live concurrency-corruption class: diffusers holds ONE
stateful scheduler (``step_index`` and sigmas advance during the loop)
reused across calls, and endpoints that ASSIGN ``self.pipeline.scheduler``
per request (sdxl's old ``_ensure_scheduler``) let two concurrent requests
corrupt each other's trajectory — and the assignment is exactly the
swap-don't-wrap that risks a recompile. Cloning the scheduler per request
is a CORRECTNESS fix, not an optimization.

Handlers reach this through ``ctx.for_request(self.pipeline, ...)`` (which
also applies the resolved checkpoint's regime — v_prediction is a
checkpoint fact the composer applies, never payload logic) or directly::

    view = ctx.for_request(self.pipeline, sampler=p.sampler, seed=p.seed)
    image = view(prompt=p.prompt, num_inference_steps=steps).images[0]
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

# Friendly sampler name -> (diffusers scheduler class name, extra config).
# The SDK owns this table so endpoints stop shipping their own
# `_scheduler_kind` maps; a per-request sampler is a VIEW field.
SAMPLERS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "ddim": ("DDIMScheduler", {}),
    "ddpm": ("DDPMScheduler", {}),
    "deis": ("DEISMultistepScheduler", {}),
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {}),
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpmpp_2m_sde": (
        "DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    "dpmpp_2m_sde_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True}),
    "dpmpp_sde": ("DPMSolverSinglestepScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "flow_euler": ("FlowMatchEulerDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "lcm": ("LCMScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
}


class UnknownSamplerError(ValueError):
    """The requested sampler name is not in the SDK sampler table."""


def _scheduler_class(sampler: str) -> Tuple[Any, Dict[str, Any]]:
    key = str(sampler or "").strip().lower()
    if key not in SAMPLERS:
        raise UnknownSamplerError(
            f"unknown sampler {sampler!r}; known: {sorted(SAMPLERS)}"
        )
    cls_name, extra = SAMPLERS[key]
    import diffusers

    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise UnknownSamplerError(
            f"sampler {sampler!r} maps to diffusers.{cls_name}, which this "
            "diffusers build does not provide"
        )
    return cls, dict(extra)


def clone_scheduler(
    pipeline: Any,
    *,
    sampler: str = "",
    regime: str = "standard",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """A FRESH scheduler for one request, built from the instance
    scheduler's config — never the shared stateful instance.

    ``sampler`` picks a different scheduler class from the SDK table
    (``""`` keeps the instance's class). ``regime`` applies resolved-
    checkpoint facts: ``"v_prediction"`` sets ``prediction_type`` — a
    checkpoint fact applied at view construction, not payload logic.
    """
    base = getattr(pipeline, "scheduler", None)
    if base is None:
        raise ValueError(
            f"{type(pipeline).__name__} has no scheduler to clone"
        )
    base_config = getattr(base, "config", None)
    overrides = dict(config_overrides or {})
    if str(regime or "") == "v_prediction":
        overrides.setdefault("prediction_type", "v_prediction")
    if sampler:
        cls, extra = _scheduler_class(sampler)
        overrides = {**extra, **overrides}
    else:
        cls = type(base)
    if base_config is None or not hasattr(cls, "from_config"):
        # Non-diffusers scheduler shape: fall back to a deepcopy (still a
        # private per-request object; state never shared).
        fresh = copy.deepcopy(base)
        for k, v in overrides.items():
            setattr(fresh, k, v)
        return fresh
    return cls.from_config(base_config, **overrides) if overrides \
        else cls.from_config(base_config)


def for_request(
    pipeline: Any,
    *,
    sampler: str = "",
    regime: str = "standard",
    generator: Any = None,
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """A per-request VIEW of ``pipeline``: same class, same module objects
    (shared weights, compiled graph intact), OWN scheduler.

    Only separable state may live in the view — scheduler instance +
    timesteps/sigmas, latents, generator/seed, steps, guidance, prompt,
    callbacks. Instance identity (dtype, memory format, attention slicing,
    offload hooks, attention-processor swaps) is fixed at materialization;
    needing different values there means a different INSTANCE.

    ``generator`` (a seeded ``torch.Generator``) is stored as
    ``view.generator`` for convenience; pass it explicitly to the call if
    the pipeline's signature wants it per call.
    """
    cls = type(pipeline)
    view = object.__new__(cls)
    # Container copy: attributes reference the SAME module objects. The
    # config dicts are shallow-copied so per-view bookkeeping (diffusers
    # keeps registered-module names in _internal_dict) never aliases the
    # shared instance's.
    d = dict(pipeline.__dict__)
    for key in ("_internal_dict", "config", "_progress_bar_config"):
        val = d.get(key)
        if isinstance(val, dict):
            d[key] = type(val)(**val) if hasattr(val, "keys") else dict(val)
        elif val is not None and hasattr(val, "copy"):
            try:
                d[key] = val.copy()
            except Exception:
                pass
    object.__setattr__(view, "__dict__", d)
    if getattr(pipeline, "scheduler", None) is not None:
        fresh = clone_scheduler(
            pipeline, sampler=sampler, regime=regime,
            config_overrides=scheduler_config,
        )
        # Bypass DiffusionPipeline.__setattr__ (register_modules
        # bookkeeping): the view's registered names are unchanged; only the
        # object behind `scheduler` is view-private.
        object.__setattr__(view, "scheduler", fresh)
    elif sampler:
        raise ValueError(
            f"{cls.__name__} has no scheduler; sampler={sampler!r} cannot apply"
        )
    if generator is not None:
        object.__setattr__(view, "generator", generator)
    return view


__all__ = [
    "SAMPLERS",
    "UnknownSamplerError",
    "clone_scheduler",
    "for_request",
]
