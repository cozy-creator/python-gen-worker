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
also applies the resolved checkpoint's objective — v-prediction/flow are
checkpoint facts the composer applies, never payload logic) or directly::

    view = ctx.for_request(self.pipeline, sampler=p.sampler, seed=p.seed)
    image = view(prompt=p.prompt, num_inference_steps=steps).images[0]
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

# Friendly sampler name -> (diffusers scheduler class name, extra config).
# The SDK table DEFINES each named sampler COMPLETELY (pgw#654, absorbing
# pgw#647 gap #2): recipes SELECT among these names; endpoint-private
# sampler tables must not exist — two endpoints defining "euler_trailing"
# differently would make one recipe mean different math depending on which
# endpoint serves it. Per-setting rulings folded in:
# - solver_order=2 on the dpm++ multistep entries: part of the sampler's
#   DEFINITION ("2M" means second order), not a family preference.
# - final_sigmas_type="zero" on the dpm++ multistep entries: diffusers' own
#   guidance for stable final steps; definition, not family recipe.
# - "euler_trailing" (Euler + timestep_spacing="trailing"): the documented
#   SDXL-Lightning recipe, family-neutral by construction.
# Genuinely family-specific numbers (steps, guidance) are catalog recipe
# data (th#1116 family schemas), never rows here.
SAMPLERS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "ddim": ("DDIMScheduler", {}),
    "ddpm": ("DDPMScheduler", {}),
    "deis": ("DEISMultistepScheduler", {}),
    "dpmpp_2m": ("DPMSolverMultistepScheduler",
                 {"solver_order": 2, "final_sigmas_type": "zero"}),
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler",
                        {"solver_order": 2, "use_karras_sigmas": True,
                         "final_sigmas_type": "zero"}),
    "dpmpp_2m_sde": (
        "DPMSolverMultistepScheduler",
        {"solver_order": 2, "algorithm_type": "sde-dpmsolver++",
         "final_sigmas_type": "zero"}),
    "dpmpp_2m_sde_karras": (
        "DPMSolverMultistepScheduler",
        {"solver_order": 2, "algorithm_type": "sde-dpmsolver++",
         "use_karras_sigmas": True, "final_sigmas_type": "zero"}),
    "dpmpp_sde": ("DPMSolverSinglestepScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "euler_trailing": ("EulerDiscreteScheduler", {"timestep_spacing": "trailing"}),
    "flow_euler": ("FlowMatchEulerDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "lcm": ("LCMScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
}

# Scheduler classes that integrate a FLOW-MATCHING objective. A checkpoint
# stamped objective="flow" must never be driven by a diffusion (eps/v-pred)
# scheduler — the sigma schedules are different math, not a preference.
_FLOW_CLASS_PREFIX = "FlowMatch"


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
    objective: str = "",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """A FRESH scheduler for one request, built from the instance
    scheduler's config — never the shared stateful instance.

    ``sampler`` picks a different scheduler class from the SDK table
    (``""`` keeps the instance's class). ``objective`` applies the resolved
    checkpoint's stamped training-objective fact (pgw#654) — scheduler math
    at view construction, never payload logic:

    - ``"epsilon"`` / ``"v_prediction"``: sets ``prediction_type``; for
      v-prediction ALSO sets ``rescale_betas_zero_snr=True`` (th#1017's
      zero-terminal-SNR contract — folded in here so no endpoint can
      forget it and wash out).
    - ``"flow"``: requires a flow-match scheduler class — a diffusion
      sampler selection raises instead of silently integrating the wrong
      math; the sigma schedule rides the instance scheduler's config.
    - ``""`` (unstamped): applies nothing.
    """
    base = getattr(pipeline, "scheduler", None)
    if base is None:
        raise ValueError(
            f"{type(pipeline).__name__} has no scheduler to clone"
        )
    base_config = getattr(base, "config", None)
    overrides = dict(config_overrides or {})
    obj = str(objective or "")
    if obj in ("epsilon", "v_prediction"):
        overrides.setdefault("prediction_type", obj)
        if obj == "v_prediction":
            overrides.setdefault("rescale_betas_zero_snr", True)
    if sampler:
        cls, extra = _scheduler_class(sampler)
        overrides = {**extra, **overrides}
    else:
        cls = type(base)
    if obj == "flow":
        cls_name = cls.__name__ if isinstance(cls, type) else str(cls)
        if not cls_name.startswith(_FLOW_CLASS_PREFIX):
            raise ValueError(
                f"objective='flow' checkpoint cannot run under sampler "
                f"{sampler or cls_name!r} ({cls_name} is not a flow-match "
                "scheduler); flow sigma schedules are different math, not a "
                "preference"
            )
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
    objective: str = "",
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
            pipeline, sampler=sampler, objective=objective,
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
