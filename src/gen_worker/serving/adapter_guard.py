"""A LIVE adapter on a compiled-armed module serves EAGER, loudly. peft wraps a denoiser's SUBMODULES, and an armed compiled graph replaces the PARENT's forward with a traced computation that never enters them — so an adapter attached after arming would otherwise serve the base model bit-identically, with no refusal and no log. install() routes a module carrying a live peft_config to its own eager forward and says so once; rearm_constants() re-installs a compiled runner's constant table after an in-place weight write (AOTI's runtime constant folding folds once on the first run() and never re-folds on a bare tensor write). Eager is a correct answer and stays one: the cost is speed, never numerics — nothing here refuses a request."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from .. import activity as activity_mod

logger = logging.getLogger(__name__)

PEFT_MARKER_ATTR = "peft_config"

_GUARD_ATTR = "_cozy_adapter_guard"


def dispatcher_of(module: Any) -> Any:
    """The torchcg dispatcher fronting ``module``, THROUGH any wrapper, or None."""
    forward = getattr(module, "forward", None)
    if forward is None:
        return None
    inner = getattr(forward, _GUARD_ATTR, None)
    candidate = inner if inner is not None else forward
    if hasattr(candidate, "eager_forward") and hasattr(candidate, "armed_graphs"):
        return candidate
    return None


def armed_graphs(module: Any) -> Tuple[str, ...]:
    """The graph identities currently armed on ``module``."""
    dispatcher = dispatcher_of(module)
    if dispatcher is None:
        return ()
    try:
        return tuple(dispatcher.armed_graphs())
    except Exception:  # noqa: BLE001 — a probe never costs a request
        return ()


def compiled_armed(module: Any) -> bool:
    """Whether a compiled artifact is currently serving ``module``'s forward."""
    return bool(armed_graphs(module))


def has_live_adapter(module: Any) -> bool:
    """Whether peft currently has adapters injected into ``module``."""
    return bool(getattr(module, PEFT_MARKER_ATTR, None))


def install(module: Any) -> bool:
    """Guard one adopted module."""
    dispatcher = dispatcher_of(module)
    if dispatcher is None:
        return False
    if getattr(getattr(module, "forward", None), _GUARD_ATTR, None) is not None:
        return True
    state: Dict[str, Any] = {"said": False, "ordered": False}

    def guarded(*args: Any, **kwargs: Any) -> Any:
        ordered = _eager_only_reason()
        if ordered:
            if not state["ordered"]:
                state["ordered"] = True
                _say_ordered(module, ordered)
            return dispatcher.eager_forward(*args, **kwargs)
        if getattr(module, PEFT_MARKER_ATTR, None):
            if not state["said"]:
                state["said"] = True
                _say(module)
            return dispatcher.eager_forward(*args, **kwargs)
        return dispatcher(*args, **kwargs)

    setattr(guarded, _GUARD_ATTR, dispatcher)
    module.forward = guarded
    return True


def sink(adopt: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """``ctx.compile``'s sink, with the guard on everything it arms."""

    def compile_sink(target: Any) -> Any:
        armed_target = adopt(target)
        for module in _adopted_modules(armed_target):
            try:
                install(module)
            except Exception:  # noqa: BLE001 — a guard never fails a load
                logger.exception(
                    "adapter guard: could not guard %s; a LoRA on this "
                    "module would serve the base weights silently",
                    type(module).__name__)
        return armed_target

    return compile_sink


def _adopted_modules(target: Any) -> List[Any]:
    components = getattr(target, "components", None)
    if isinstance(components, dict):
        return [value for value in components.values()
                if dispatcher_of(value) is not None]
    return [target] if dispatcher_of(target) is not None else []


def _eager_only_reason() -> str:
    from .. import serve_posture

    try:
        return serve_posture.block()
    except Exception:  # noqa: BLE001 — a posture probe never costs a request
        logger.debug("adapter guard: posture read failed", exc_info=True)
        return ""


def _say_ordered(module: Any, why: str) -> None:
    from .. import serve_posture

    detail = (
        f"{type(module).__name__}: {len(armed_graphs(module))} armed graph(s) "
        f"NOT dispatched to — {why}. Reversible: releasing the order resumes "
        f"compiled serving with no re-arm."
    )
    logger.warning("adapter guard: %s", detail)
    try:
        activity_mod.emit_event(
            activity_mod.KIND_LORA_HYGIENE, detail,
            phase=serve_posture.REASON,
        )
    except Exception:  # noqa: BLE001 — the request outlives its telemetry
        logger.debug("adapter guard: posture row failed to emit", exc_info=True)


def _say(module: Any) -> None:
    names = sorted(getattr(module, PEFT_MARKER_ATTR, {}) or {})
    detail = (
        f"{type(module).__name__}: live peft adapter(s) {names} on a "
        f"compiled-armed module ({len(armed_graphs(module))} graph(s) armed) — "
        f"serving EAGER for the duration, because the compiled graph was "
        f"traced without them and would silently return the BASE MODEL "
        f"(pgw#1571). Fold the adapter into the weights "
        f"(models.lora_fold.folded, rebind=adapter_guard.rearm_constants) to "
        f"keep compiled speed."
    )
    logger.warning("adapter guard: %s", detail)
    try:
        activity_mod.emit_event(
            activity_mod.KIND_LORA_HYGIENE, detail,
            phase="adapter_ops_on_compiled",
        )
    except Exception:  # noqa: BLE001 — the request outlives its telemetry
        logger.debug("adapter guard: hygiene row failed to emit", exc_info=True)


class ConstantRearmUnsupported(RuntimeError):
    """A compiled runner exposes no bound constant table to re-install."""


def rearm_constants(module: Any) -> int:
    """Re-install every armed runner's constant table after a WEIGHT WRITE."""
    dispatcher = dispatcher_of(module)
    if dispatcher is None:
        return 0
    rearmed = 0
    for _record, call in tuple(getattr(dispatcher, "_entries", ()) or ()):
        runner = getattr(call, "runner", None)
        package = getattr(runner, "_package", None)
        values = getattr(runner, "_bound_values", None)
        if package is None or not values:
            raise ConstantRearmUnsupported(
                f"the compiled runner armed on {type(module).__name__} exposes "
                f"no bound constant table to re-install after a weight "
                f"mutation; a folded constant would serve stale weights")
        package.load_constants(values, check_full_update=True, user_managed=True)
        rearmed += 1
    return rearmed


__all__ = [
    "ConstantRearmUnsupported",
    "PEFT_MARKER_ATTR",
    "armed_graphs",
    "compiled_armed",
    "dispatcher_of",
    "has_live_adapter",
    "install",
    "rearm_constants",
    "sink",
]
