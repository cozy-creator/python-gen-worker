"""The operator's EAGER-ONLY command (DESIGN-RULINGS §4.32 item 4).

Paul, verbatim: *"The consumer should have the ability to turn off compile
entirely, and serve-eager only, if the compile is broken or they just don't
care. I.e., they can send some sort of command to the worker to serve eager
rather than serve compiled."*

This module holds the order and nothing else: the enforcement lives at the
three places that already decide compiled-vs-eager, so there is one posture
with one token rather than a fourth mechanism.

WHY IT IS A COMMAND
-------------------
Not an env var (§1.17 / Paul's standing rule: an env may carry a VALUE, never a
DECISION), and not a second config surface. Two eager postures already exist
and neither can answer the operator's question:

* ``STEADY_BACKEND_EAGER_ONLY`` on the Plan's ``ExecutionSpec`` is chosen by
  the hub at DISPATCH — it cannot be flipped on a pod that is already serving,
  and there is nothing to un-flip. It reports ``hub_ordered_eager``.
* pgw#714's ``operator_eager_pin`` is the hub-resolved lane, pinned in config.
  Same shape: durable policy, decided before the pod serves.

The missing capability — the one Paul named — is turning compiled serving off
on a LIVE worker, and back on. It arrives as ``ServePosture`` on the existing
scheduler stream (or, on cozy-local, from the CLI), never as an environment.

THE THREE SCOPE DECISIONS, and why
----------------------------------
**Whole-worker, not per-function.** The operator's question is "is compile
broken on this pod". A per-function switch needs a second addressing vocabulary
(which function? on which execution group? under which release?) for a case
nobody has had; it stays future scope, and the ONE global order is what every
enforcement point reads, so adding a scope later narrows this rather than
forking it.

**Runtime, not boot-time.** "The compile is broken" is discovered while
serving. A boot-only switch would mean recycling the pod to use it, which is
the cost the switch exists to avoid.

**Reversible — and that decides the enforcement point.** ``suppress`` must not
unwrap an armed compiled graph: unwrapping is destructive (``aot_serve.unwrap`` restores
the captured forward and the arm is gone for the boot), so a worker that
un-suppressed would serve eager anyway until something re-armed it. Instead the
suppression is read AT THE CALL — ``aot_serve``'s wrapper answers from the
eager forward while the artifact stays armed — so releasing it resumes compiled
serving on the very next request, with no re-arm, no re-materialize and no
re-mint.

COMPOSITION WITH §4.31's STICKY DE-ARM
--------------------------------------
Same posture, two triggers, and they must never be read as each other:

===================  ==========================  ==========================
                     §4.31 sticky de-arm         §4.32 eager-only command
===================  ==========================  ==========================
trigger              a compiled graph-attributable serve   an operator, explicitly
                     failure (guard refusal,
                     artifact crash)
scope                THAT compiled graph                   the whole worker
lifetime             sticky for the boot         until released
reversible           no — it is evidence         yes — it is policy
token                ``compiled_degraded`` &c.   ``operator_eager_only``
===================  ==========================  ==========================

Releasing the command deliberately does NOT resurrect a de-armed compiled graph. A
de-arm recorded that this artifact failed on this machine; policy has no
standing to overrule evidence.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .compiled_graph_adopt import EagerPhase

logger = logging.getLogger(__name__)

#: The wire token for "an operator ordered this worker eager". Shares the
#: `EagerPhase` vocabulary — the hub groups `worker_activity_events` and joins
#: request rows on these values — and is DISTINCT from every automatic eager
#: reason so a suppressed worker can never be read as a broken adopt path.
REASON: str = EagerPhase.OPERATOR_EAGER_ONLY.value

#: Activity `phase` tokens for the two transitions. Countable hub-side.
PHASE_SUPPRESSED = "eager_only_engaged"
PHASE_RELEASED = "eager_only_released"


@dataclass(frozen=True)
class EagerOnlyOrder:
    """The standing order. ``active`` False is the default posture."""

    active: bool = False
    actor: str = ""
    reason: str = ""
    at: float = 0.0

    def describe(self) -> str:
        who = self.actor or "an operator"
        why = f": {self.reason}" if self.reason else ""
        if not self.active:
            return f"compiled serving permitted (last change by {who}{why})"
        return f"compiled serving suppressed by {who}{why}"


_LOCK = threading.Lock()
_ORDER = EagerOnlyOrder()


def order() -> EagerOnlyOrder:
    """The current standing order — a snapshot, safe to read from any thread."""
    with _LOCK:
        return _ORDER


def eager_only() -> bool:
    """True when an operator has ordered this worker to serve eager only."""
    return _ORDER.active


def block() -> str:
    """Why arming is refused right now, or ``""``.

    Shaped for :func:`compile_cache.arming_block`, which is the ONE
    precondition authority: routing through it means the command suppresses
    adoption, JIT intake, cold compile and every self-mint with one check
    rather than with a check per call site.
    """
    current = order()
    if not current.active:
        return ""
    who = current.actor or "an operator"
    why = f" ({current.reason})" if current.reason else ""
    return (f"{who} ordered this worker to serve EAGER ONLY{why} "
            f"(§4.32 item 4); release the order to arm again")


def apply_command(
    eager_only_flag: bool, *, actor: str = "", reason: str = "",
) -> bool:
    """Apply one ``ServePosture`` command. Returns True when the posture MOVED.

    Idempotent by design: the hub replays the order to a reconnecting worker
    and cozy-local may say ``on`` twice, and neither should produce a second
    transition event. A repeat with a NEW actor/reason still refreshes the
    record (who last touched it is the operator-visible fact) without claiming
    a transition.
    """
    global _ORDER
    now = time.time()
    with _LOCK:
        previous = _ORDER
        _ORDER = EagerOnlyOrder(
            active=bool(eager_only_flag),
            actor=str(actor or "").strip(),
            reason=str(reason or "").strip(),
            at=now,
        )
        moved = previous.active != _ORDER.active
        current = _ORDER
    if not moved:
        logger.info("serve-posture: unchanged — %s", current.describe())
        return False
    if current.active:
        logger.warning(
            "serve-posture: EAGER ONLY by order — %s. Nothing will arm or "
            "mint, and armed compiled graphs stay armed but are not called; releasing "
            "the order resumes compiled serving without a re-arm.",
            current.describe())
    else:
        logger.warning(
            "serve-posture: eager-only order RELEASED — %s. Armed compiled graphs serve "
            "compiled again from the next request; compiled graphs de-armed for cause "
            "(§4.31) stay de-armed.", current.describe())
    _emit_transition(current)
    return True


def _emit_transition(current: EagerOnlyOrder) -> None:
    """Confess the transition on the wire.

    Imported here rather than at module import: ``compile_cache`` reads this
    module from its precondition authority, and ``activity`` pulls the whole
    protobuf/progress stack behind it.
    """
    from . import activity as activity_mod

    phase = PHASE_SUPPRESSED if current.active else PHASE_RELEASED
    detail = (
        f"actor={current.actor or '(unnamed)'} "
        f"reason={current.reason or '(none given)'} — "
        + ("compiled serving is suppressed on this worker by operator order "
           "(§4.32 item 4); every request serves eager and the posture token "
           f"is {REASON}"
           if current.active else
           "the operator's eager-only order is released; armed compiled graphs serve "
           "compiled again and arming decisions run normally")
    )
    try:
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_POSTURE, detail, phase=phase)
    except Exception:  # noqa: BLE001 — a confession must never break a command
        logger.debug("serve-posture event emission failed", exc_info=True)


def reset(order_: Optional[EagerOnlyOrder] = None) -> None:
    """Restore the default posture (tests, and the CLI's own teardown)."""
    global _ORDER
    with _LOCK:
        _ORDER = order_ or EagerOnlyOrder()


__all__ = [
    "EagerOnlyOrder",
    "PHASE_RELEASED",
    "PHASE_SUPPRESSED",
    "REASON",
    "apply_command",
    "block",
    "eager_only",
    "order",
    "reset",
]
