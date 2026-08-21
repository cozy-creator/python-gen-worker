"""Per-request pipeline VIEWS (WORKER-RESIDENCY-DESIGN "Shared components and mutation safety")."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SAMPLERS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "ddim": ("DDIMScheduler", {}),
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

_FLOW_CLASS_PREFIX = "FlowMatch"
_FLOW_SIGMAS_FIELD = "use_flow_sigmas"
_FLOW_PREDICTION_TYPE = "flow_prediction"
_FLOW_SHIFT_ALIASES = ("shift", "flow_shift")


class UnknownSamplerError(ValueError):
    """The requested sampler name is not in the SDK sampler table."""


def _init_fields(cls: Any) -> frozenset:
    if not isinstance(cls, type):
        return frozenset()
    try:
        return frozenset(inspect.signature(cls).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callable
        return frozenset()


def _as_dict(config: Any) -> Dict[str, Any]:
    try:
        return dict(config or {})
    except (TypeError, ValueError):
        return {}


def flow_capable(cls: Any, config: Optional[Dict[str, Any]] = None) -> bool:
    """Does the scheduler this class+config BUILDS integrate flow-match sigmas? Two honest signals, and a class name is only one of them: * flow BY CONSTRUCTION — the ``FlowMatch*`` family has no other mo..."""
    name = cls.__name__ if isinstance(cls, type) else str(cls)
    if name.startswith(_FLOW_CLASS_PREFIX):
        return True
    if _FLOW_SIGMAS_FIELD not in _init_fields(cls):
        return False
    cfg = config or {}
    return bool(cfg.get(_FLOW_SIGMAS_FIELD)) or \
        cfg.get("prediction_type") == _FLOW_PREDICTION_TYPE


def _alias_flow_shift(cls: Any, overrides: Dict[str, Any]) -> None:
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
    if obj is None or isinstance(obj, type):
        return False
    return callable(getattr(obj, "step", None)) and callable(
        getattr(obj, "set_timesteps", None)
    )


def _clonable_scheduler(name: str, obj: Any) -> bool:
    if obj is None or isinstance(obj, type):
        return False
    if "scheduler" in name.lower() and callable(getattr(obj, "from_config", None)):
        return True
    return _sampler_shaped(obj)


def discover_schedulers(pipeline: Any) -> Tuple[str, ...]:
    """Every attribute of ``pipeline`` that carries per-request SAMPLER state , primary first, then the rest in declaration order."""
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
    """A FRESH scheduler for one request, built from the instance scheduler's config — never the shared stateful instance."""
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
    """A per-request VIEW of ``pipeline``: same class, same module objects (shared weights, compiled graph intact), OWN scheduler(s)."""
    cls = type(pipeline)
    view = object.__new__(cls)
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
