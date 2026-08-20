"""A LIVE adapter on a compiled-armed module serves EAGER, loudly (pgw#1573).

pgw#1571 measured the defect and stated it exactly: peft wraps a denoiser's
SUBMODULES, and an armed compiled graph replaces the PARENT's forward with a
traced computation that never enters them. So an adapter attached after arming
does not execute — the base model is served, bit-identically, with no refusal
and no log. Measured there: eager red arm ``max|delta| = 2.2e-02``, armed
``0.0``, with 32 peft wrappers attached.

**Its fix landed in ``aot_serve``, which nothing on the serving path calls.**
Verified against ``origin/master``: ``aot_serve.wrap_module`` has zero non-test
callers — every reference in ``src/`` is a docstring — and the live arm is
``torchcg.adopt.AdoptSession`` handing ``torchcg.serve.CompiledGraphCall`` to a
``_ForwardDispatcher``. Neither ``PEFT_MARKER_ATTR``, ``_say_adapter_ops_once``
nor ``rearm_constants`` is reachable from a pod. The defect was masked only by
adoption being broken; pgw#1573 fixed adoption, so it arms itself. This module
is the same guard, on the path that runs.

TWO HALVES, both O(1) on the hot path:

* :func:`install` wraps the dispatcher torchcg installed. A module carrying a
  live ``peft_config`` routes to the module's own eager forward and says so
  ONCE. One ``getattr`` per call — a module walk here would cost more than the
  defect it prevents.
* :func:`rearm_constants` re-installs a compiled runner's bound constant table
  after an in-place weight write, which is what makes folding an adapter INTO
  the weights (:mod:`gen_worker.models.lora_fold`) work on a v2 pod:
  ``load_constants(..., user_managed=True)`` keeps raw pointers, so a fold is
  visible — except through AOTI's runtime constant folding, which folds once on
  the first ``run()`` and never re-folds on a bare tensor write.

**Eager is a correct answer and stays one.** Nothing here refuses a request:
the cost is speed, never numerics, and the alternative — a silently wrong
image — is the only outcome that is not acceptable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from .. import activity as activity_mod

logger = logging.getLogger(__name__)

#: peft writes this on the module it injects adapters into (diffusers'
#: ``load_lora_adapter`` -> ``inject_adapter_in_model``) and deletes it on
#: unload. Same attribute pgw#1571 keyed on, deliberately: one marker, and the
#: v1 spelling is the one the ecosystem writes.
PEFT_MARKER_ATTR = "peft_config"

#: Set on a guard so a second ``ctx.compile`` of the same module cannot stack
#: two of them, and so :func:`armed_graphs` can recognise one.
_GUARD_ATTR = "_cozy_adapter_guard"


def dispatcher_of(module: Any) -> Any:
    """The torchcg dispatcher fronting ``module``, THROUGH any wrapper, or None.

    PUBLIC, and it is the answer to "is this module still routing through its
    compiled dispatcher" (pgw#1591). This guard installs itself AS
    ``module.forward``, so the obvious test — ``module.forward is dispatcher``
    — reads False on every guarded module and calls a perfectly healthy arm
    DISPLACED. Measured on the live sd15 benchmark: 12/12 requests of both
    arms reported the dispatcher displaced while the compiled graphs were in
    fact executing, and a whole GPU leg was thrown away comparing eager to
    eager that was not eager.

    Anything asking that question must ask it HERE, so that adding a wrapper
    is not a change every reader has to learn about.

    Duck-typed on the two attributes the dispatcher contract actually has
    (``eager_forward`` + ``armed_graphs``) rather than on an isinstance against
    a vendored private class: the vendored snapshot is sha256-fenced, so a
    check that pins its identity here is a check that breaks on a re-vendor
    instead of on a real change.
    """
    forward = getattr(module, "forward", None)
    if forward is None:
        return None
    inner = getattr(forward, _GUARD_ATTR, None)
    candidate = inner if inner is not None else forward
    if hasattr(candidate, "eager_forward") and hasattr(candidate, "armed_graphs"):
        return candidate
    return None


def armed_graphs(module: Any) -> Tuple[str, ...]:
    """The graph identities currently armed on ``module``. Empty = eager."""
    dispatcher = dispatcher_of(module)
    if dispatcher is None:
        return ()
    try:
        return tuple(dispatcher.armed_graphs())
    except Exception:  # noqa: BLE001 — a probe never costs a request
        return ()


def compiled_armed(module: Any) -> bool:
    """Whether a compiled artifact is currently serving ``module``'s forward.

    THE v2 ANSWER. ``lora_fold._compiled_armed`` asks ``aot_serve``, whose
    marker (``_cozy_compile``) no pod has carried since pgw#1373 — so on a real
    worker that predicate answers False for every armed module and every
    compiled-aware branch behind it is dead code.
    """
    return bool(armed_graphs(module))


def has_live_adapter(module: Any) -> bool:
    """Whether peft currently has adapters injected into ``module``."""
    return bool(getattr(module, PEFT_MARKER_ATTR, None))


def install(module: Any) -> bool:
    """Guard one adopted module. Returns whether a guard was installed.

    Called after ``AdoptSession.adopt`` has installed its dispatcher, on the
    same object. Idempotent: a module already guarded is left alone, so the two
    ``ctx.compile`` hosts and a re-adopt cannot stack guards.

    The dispatcher object is NOT replaced — only ``module.forward`` is — so
    ``AdoptSession.arm``'s late-mint handoff, which reaches the dispatcher
    through its own ``_home`` map, keeps working unchanged.
    """
    dispatcher = dispatcher_of(module)
    if dispatcher is None:
        return False
    if getattr(getattr(module, "forward", None), _GUARD_ATTR, None) is not None:
        return True  # already guarded
    state: Dict[str, Any] = {"said": False, "ordered": False}

    def guarded(*args: Any, **kwargs: Any) -> Any:
        # THE OPERATOR'S ORDER FIRST — it is the cheapest read and the most
        # authoritative (pgw#1587 / pgw#1589, §4.32 item 4). A standing
        # `serve_posture{eager_only:true}` is neither a defect nor a
        # degradation: it is the answer, and it is reversible, so it is
        # re-read per call and never latched.
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
    """``ctx.compile``'s sink, with the guard on everything it arms.

    Wraps ``AdoptSession.adopt`` rather than living inside it, because the
    session is vendored and sha256-fenced. The walk is the same one ``adopt``
    does — the target itself, or every ``nn.Module`` in a pipeline-shaped
    container's ``components`` mapping — so a marked pipeline is guarded
    component by component exactly as it is armed.
    """

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
    """Why an operator has ordered this worker eager, or ``""``.

    pgw#1589: the order was applied by `worker.on_message` and read by exactly
    ONE thing — `aot_serve`'s arm, which has no production caller — so an
    operator could issue it, get an ack, and watch the pod keep serving from
    its compiled graphs. pgw#1587 added a second read in `arm_aot`, in the same
    orphaned tier. This is the read on the path that dispatches.

    ARM-time is the wrong altitude for it on its own: the order arrives over a
    live control channel long after arming, and it is RELEASABLE. Reading it
    here makes both directions work with no re-arm and no de-arm — which is
    exactly the reversibility `apply_command` promises.
    """
    from .. import serve_posture

    try:
        return serve_posture.block()
    except Exception:  # noqa: BLE001 — a posture probe never costs a request
        logger.debug("adapter guard: posture read failed", exc_info=True)
        return ""


def _say_ordered(module: Any, why: str) -> None:
    """The operator's order, stated ONCE per module, on the wire.

    ITS OWN PHASE, and that is not a detail. `serve_posture` already emits a
    TRANSITION row on `PHASE_SUPPRESSED` ("the order was applied"); this is a
    DISPATCH row ("this module stopped being called"). Sharing one phase across
    two kinds gives that phase two vocabularies and makes
    `count(*) where phase=...` mean neither — the same split pgw#1441 made for
    `boot_adopt` vs `boot_adopt_summary`, for the same reason.

    So the phase is `EagerPhase.OPERATOR_EAGER_ONLY` itself — the token
    `compiled_graph_adopt.EagerPhase` has defined for exactly this state since
    pgw#1142 and which, until now, NOTHING emitted. It must never be counted
    with the failure classes or with `hub_ordered_eager` (one PLAN's backend,
    not a standing order about this pod).
    """
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
    """Say the degradation ONCE per module, on the wire and in the log.

    Per FORWARD, not per boot: an adapter arrives at request time, so the first
    guarded call is the first instant this is true. A serve pod's stdout goes
    nowhere (pgw#760), so the log line alone would be the same silence this
    guard exists to end.
    """
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
    """A compiled runner exposes no bound constant table to re-install.

    Refused BY NAME rather than skipped: the condition this is preventing is a
    folded constant that keeps serving pre-fold weights, which produces a
    plausible wrong image and no error at all.
    """


def rearm_constants(module: Any) -> int:
    """Re-install every armed runner's constant table after a WEIGHT WRITE.

    Returns how many entries were re-armed; 0 when nothing is armed, which is
    the ordinary eager case and not an error.

    ``load_constants(..., user_managed=True)`` keeps RAW POINTERS to the
    module's own parameters, so an in-place fold is visible to the artifact
    with no bookkeeping — except through AOTI's runtime constant folding, which
    ``torchcg.compiler`` turns on. The container folds once on the first
    ``run()`` (``fold_state`` INITIALIZED -> FOLDED) and never again, and an
    in-place tensor write calls nothing. Re-installing the same pointers is
    what puts ``fold_state`` back to INITIALIZED, so the next call re-folds
    against the new weights.
    """
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
