"""Per-request pipeline VIEWS (WORKER-RESIDENCY-DESIGN "Shared components and
mutation safety").

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

This closes a concurrency-corruption class: diffusers holds ONE stateful
scheduler (``step_index`` and sigmas advance during the loop) reused across
calls, and an endpoint that ASSIGNS ``self.pipeline.scheduler`` per request
lets two concurrent requests corrupt each other's trajectory — and the
assignment is exactly the swap-don't-wrap that risks a recompile. Cloning the
scheduler per request is a CORRECTNESS fix, not an optimization.

The view clones EVERY scheduler the pipeline carries, not just the attribute
literally named ``scheduler``: a pipeline with a second stateful sampler (e.g.
``LTX2ConditionPipeline.audio_scheduler``, stepped once per denoise step by
diffusers' own ``__call__``) would otherwise keep SHARING it and silently
reintroduce the corruption ``for_request`` exists to remove. The SDK owns the
enumeration (:func:`discover_schedulers`) so no endpoint has to answer "which of
my attributes carry per-request state" by hand.

Handlers reach this through ``ctx.for_request(self.pipeline, ...)`` (which
also applies the resolved checkpoint's objective — v-prediction/flow are
checkpoint facts the composer applies, never payload logic) or directly::

    view = ctx.for_request(self.pipeline, sampler=p.sampler, seed=p.seed)
    image = view(prompt=p.prompt, num_inference_steps=steps).images[0]
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Friendly sampler name -> (diffusers scheduler class name, extra config).
# The SDK table DEFINES each named sampler COMPLETELY:
# recipes SELECT among these names; endpoint-private
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
# data, never rows here.
SAMPLERS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "ddim": ("DDIMScheduler", {}),
    # Hyper-SD's published recipe is DDIM with trailing timestep spacing, and
    # family-neutral by construction exactly like "euler_trailing".
    "ddim_trailing": ("DDIMScheduler", {"timestep_spacing": "trailing"}),
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

# A checkpoint stamped objective="flow" must never be driven by a diffusion
# (eps/v-pred) sampler — the sigma schedules are different math, not a
# preference. What makes a scheduler flow-match is CAPABILITY, not spelling:
# the `FlowMatch*` family is flow by construction, and the multistep solvers
# (UniPC, DPMSolver, DEIS) are flow when their config says so. UniPC with
# `use_flow_sigmas` IS the official Wan solver.
_FLOW_CLASS_PREFIX = "FlowMatch"
# The __init__ field that only a flow-capable non-FlowMatch class has.
_FLOW_SIGMAS_FIELD = "use_flow_sigmas"
_FLOW_PREDICTION_TYPE = "flow_prediction"
# The same flow-sigma knob under two upstream spellings: `FlowMatch*` calls it
# `shift`, the multistep solvers call it `flow_shift`.
_FLOW_SHIFT_ALIASES = ("shift", "flow_shift")


class UnknownSamplerError(ValueError):
    """The requested sampler name is not in the SDK sampler table."""


def _init_fields(cls: Any) -> frozenset:
    """The config fields ``cls`` actually accepts. ``from_config`` silently
    DROPS everything else, so this is the only honest "does it honour it"."""
    if not isinstance(cls, type):
        return frozenset()
    try:
        return frozenset(inspect.signature(cls).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callable
        return frozenset()


def _as_dict(config: Any) -> Dict[str, Any]:
    """A diffusers ``FrozenDict`` as a plain dict; anything unreadable as {}."""
    try:
        return dict(config or {})
    except (TypeError, ValueError):
        return {}


def flow_capable(cls: Any, config: Optional[Dict[str, Any]] = None) -> bool:
    """Does the scheduler this class+config BUILDS integrate flow-match sigmas?

    Two honest signals, and a class name is only one of them:

    * flow BY CONSTRUCTION — the ``FlowMatch*`` family has no other mode;
    * flow BY DECLARATION — the resolved config says
      ``use_flow_sigmas`` / ``prediction_type="flow_prediction"``, AND the
      class takes ``use_flow_sigmas`` at all. The second half is the whole
      check: every diffusers scheduler accepts ``prediction_type``, so a flow
      config carried onto a diffusion class would otherwise read as flow while
      producing a scheduler that merely looked flow-shaped in a dict.
    """
    name = cls.__name__ if isinstance(cls, type) else str(cls)
    if name.startswith(_FLOW_CLASS_PREFIX):
        return True
    if _FLOW_SIGMAS_FIELD not in _init_fields(cls):
        return False
    cfg = config or {}
    return bool(cfg.get(_FLOW_SIGMAS_FIELD)) or \
        cfg.get("prediction_type") == _FLOW_PREDICTION_TYPE


def _alias_flow_shift(cls: Any, overrides: Dict[str, Any]) -> None:
    """Rename a declared flow shift to the field ``cls`` honours:
    a published ``{"shift": …}`` on a UniPC mirror is dropped by
    ``from_config`` — ignored rather than refused. Only renames when the class
    takes exactly one of the two spellings; never invents a value."""
    fields = _init_fields(cls)
    for key, other in (_FLOW_SHIFT_ALIASES, _FLOW_SHIFT_ALIASES[::-1]):
        if key in overrides and key not in fields and other in fields:
            overrides[other] = overrides.pop(key)


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


PRIMARY_SCHEDULER = "scheduler"


def _sampler_shaped(obj: Any) -> bool:
    """The diffusers ``SchedulerMixin`` structural signature: a per-request
    sampler advances timesteps (``set_timesteps``) and integrates a step
    (``step``). Weight-bearing modules (unet/vae/text encoders) never expose
    ``set_timesteps``, so nothing else matches."""
    if obj is None or isinstance(obj, type):
        # A registered component is always an INSTANCE; a class attribute
        # (a loader reference) would match every unbound method by accident.
        return False
    return callable(getattr(obj, "step", None)) and callable(
        getattr(obj, "set_timesteps", None)
    )


def _clonable_scheduler(name: str, obj: Any) -> bool:
    """A SECONDARY attribute that carries per-request sampler state.

    Two independent admission paths, both narrow:

    * the diffusers CONVENTION — the attribute name contains ``scheduler`` and
      the object exposes ``from_config`` (what ltx's ``audio_scheduler``
      satisfies, and what a registered optional component looks like);
    * the structural sampler signature, which catches a second sampler under
      an unconventional name.

    Nothing else qualifies: a shared weight module must never be cloned.
    """
    if obj is None or isinstance(obj, type):
        return False
    if "scheduler" in name.lower() and callable(getattr(obj, "from_config", None)):
        return True
    return _sampler_shaped(obj)


def discover_schedulers(pipeline: Any) -> Tuple[str, ...]:
    """Every attribute of ``pipeline`` that carries per-request SAMPLER state
, primary first, then the rest in declaration order.

    Derived from the pipeline's own attributes — diffusers registers optional
    components (``_optional_components``) as plain attributes, so a checkpoint
    that ships an ``audio_scheduler`` has one and a checkpoint that does not
    simply has ``None`` there. The SDK enumerating this is what keeps a
    half-fixed pipeline (video scheduler per-request, audio scheduler shared)
    from reading as fully fixed.

    The attribute literally named ``scheduler`` is ALWAYS included when
    non-None, whatever its shape — ``clone_scheduler`` has a deepcopy path for
    non-diffusers samplers and that primary contract predates this discovery.
    """
    found: List[str] = []
    if getattr(pipeline, PRIMARY_SCHEDULER, None) is not None:
        found.append(PRIMARY_SCHEDULER)
    try:
        names: Iterable[str] = list(vars(pipeline).keys())
    except TypeError:  # pragma: no cover - exotic object with no __dict__
        names = ()
    for name in names:
        if name.startswith("_") or name == PRIMARY_SCHEDULER:
            continue
        if _clonable_scheduler(name, getattr(pipeline, name, None)):
            found.append(name)
    return tuple(found)


def clone_scheduler(
    pipeline: Any,
    *,
    attr: str = PRIMARY_SCHEDULER,
    sampler: str = "",
    objective: str = "",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """A FRESH scheduler for one request, built from the instance
    scheduler's config — never the shared stateful instance.

    ``sampler`` picks a different scheduler class from the SDK table
    (``""`` keeps the instance's class). ``objective`` applies the resolved
    checkpoint's stamped training-objective fact — scheduler math
    at view construction, never payload logic:

    - ``"epsilon"`` / ``"v_prediction"``: sets ``prediction_type``; for
      v-prediction ALSO sets ``rescale_betas_zero_snr=True`` (the
      zero-terminal-SNR contract — folded in here so no endpoint can
      forget it and wash out).
    - ``"flow"``: requires a scheduler that INTEGRATES flow-match sigmas
      (:func:`flow_capable` — flow by construction, or a class that takes
      ``use_flow_sigmas`` under a config that declares it); a diffusion
      sampler selection raises instead of silently integrating the wrong
      math, and the sigma schedule rides the instance scheduler's config.
    - ``""`` (unstamped): applies nothing.

    A declared flow ``shift`` is renamed to whichever of
    ``shift``/``flow_shift`` the target class honours — ``from_config`` drops
    the other spelling, so a published shift would be ignored rather than
    applied.

    ``attr`` names which scheduler attribute to clone (default the primary
    ``scheduler``); :func:`for_request` uses it for secondary samplers.
    """
    base = getattr(pipeline, attr, None)
    if base is None:
        raise ValueError(
            f"{type(pipeline).__name__} has no {attr} to clone"
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
    _alias_flow_shift(cls, overrides)
    resolved = {**_as_dict(base_config), **overrides}
    if obj == "flow" and not flow_capable(cls, resolved):
        cls_name = cls.__name__ if isinstance(cls, type) else str(cls)
        raise ValueError(
            f"objective='flow' checkpoint cannot run under sampler "
            f"{sampler or cls_name!r} ({cls_name} integrates no flow-match "
            f"sigmas: not a {_FLOW_CLASS_PREFIX}* class, and the resolved "
            f"scheduler config declares no {_FLOW_SIGMAS_FIELD} / "
            f"prediction_type={_FLOW_PREDICTION_TYPE!r} that it accepts); "
            "flow sigma schedules are different math, not a preference"
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
    schedulers: Optional[Sequence[str]] = None,
) -> Any:
    """A per-request VIEW of ``pipeline``: same class, same module objects
    (shared weights, compiled graph intact), OWN scheduler(s).

    Only separable state may live in the view — scheduler instance +
    timesteps/sigmas, latents, generator/seed, steps, guidance, prompt,
    callbacks. Instance identity (dtype, memory format, attention slicing,
    offload hooks, attention-processor swaps) is fixed at materialization;
    needing different values there means a different INSTANCE.

    ``generator`` (a seeded ``torch.Generator``) is stored as
    ``view.generator`` for convenience; pass it explicitly to the call if
    the pipeline's signature wants it per call.

    EVERY sampler-shaped attribute is cloned, not just
    ``scheduler``: ``schedulers=None`` discovers them
    (:func:`discover_schedulers`); an explicit
    ``schedulers=("scheduler", "audio_scheduler")`` pins the set. ``sampler``,
    ``objective`` and ``scheduler_config`` apply to the PRIMARY scheduler
    only — they are the denoiser's sampler decision, and forcing a video
    sampler class or a flow ``shift`` onto a second modality's sampler would
    be re-deciding its math rather than making it private. Secondary samplers
    are cloned faithfully from their own config, which is exactly the
    privacy fix they need.
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
    if schedulers is None:
        names: Tuple[str, ...] = discover_schedulers(pipeline)
    else:
        names = tuple(
            n for n in (str(x).strip() for x in schedulers)
            if n and getattr(pipeline, n, None) is not None
        )
    if PRIMARY_SCHEDULER in names:
        fresh = clone_scheduler(
            pipeline, sampler=sampler, objective=objective,
            config_overrides=scheduler_config,
        )
        # Bypass DiffusionPipeline.__setattr__ (register_modules
        # bookkeeping): the view's registered names are unchanged; only the
        # object behind `scheduler` is view-private.
        object.__setattr__(view, PRIMARY_SCHEDULER, fresh)
    elif sampler:
        raise ValueError(
            f"{cls.__name__} has no scheduler; sampler={sampler!r} cannot apply"
        )
    for name in names:
        if name == PRIMARY_SCHEDULER:
            continue
        object.__setattr__(view, name, clone_scheduler(pipeline, attr=name))
    if generator is not None:
        object.__setattr__(view, "generator", generator)
    return view


__all__ = [
    "PRIMARY_SCHEDULER",
    "SAMPLERS",
    "UnknownSamplerError",
    "clone_scheduler",
    "discover_schedulers",
    "for_request",
]
