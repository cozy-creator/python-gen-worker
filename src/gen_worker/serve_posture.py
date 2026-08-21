"""The operator's EAGER-ONLY command (DESIGN-RULINGS §4.32 item 4)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .compiled_graph_adopt import EagerPhase

logger = logging.getLogger(__name__)

REASON: str = EagerPhase.OPERATOR_EAGER_ONLY.value

PHASE_SUPPRESSED = "eager_only_engaged"
PHASE_RELEASED = "eager_only_released"


@dataclass(frozen=True)
class EagerOnlyOrder:
    """The standing order."""

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
    """Why arming is refused right now, or ``""``."""
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
    """Apply one ``ServePosture`` command."""
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
