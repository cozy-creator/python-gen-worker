"""Parent-side ownership of the worker's protocol-v5 sequence (#1660).

The hub sees ONE worker; the pod runs a control parent plus G compute children
that die and respawn under it. Three facts follow, and this module is all three:

1. **The emitted ``state_seq`` is the PARENT's**, never a child's. A respawned
   child restarts its registry at 1; the hub SILENTLY DROPS a snapshot whose
   ``state_seq`` regressed (`applyLifecycleSnapshotLocked` returns nil) and
   REJECTS one that reuses a seq with different bytes. Either way the projection
   freezes at the pre-respawn state and nobody is told. This was pgw's own
   2026-08-19 audit §2 — filed there, never fixed.
2. **A terminal intent stays terminal.** The hub refuses to let a previously
   terminal intent change status, and discards the WHOLE snapshot when one does.
   A respawned child re-applies the command and reports the same intent id as
   ACCEPTED again, so the parent re-emits the terminal state it already had
   accepted. What dispatch actually reads — the capability's state — still comes
   live from the child, so nothing is claimed ready that is not.
3. **All-or-nothing survives the fan-in.** One group with no projection means the
   worker has no projection, and the parent withholds the session id with it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..lifecycle import snapshot_refusal
from ..pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

_MAX_PINNED_TERMINAL = 256

_TERMINAL = {
    pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
    pb.LIFECYCLE_INTENT_STATUS_FAILED,
    pb.LIFECYCLE_INTENT_STATUS_CANCELED,
    pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED,
}


def _body(message: Any) -> bytes:
    """A message's bytes with the sequence number the parent owns removed."""
    out = type(message)()
    out.CopyFrom(message)
    out.state_seq = 0
    return bytes(out.SerializeToString(deterministic=True))


def _snapshot_body(snapshot: pb.LifecycleSnapshot) -> bytes:
    """A snapshot's CONTENT: no sequence numbers, no generation timestamp.

    A re-timestamped repeat of the same projection must not burn a sequence
    number — the hub reads a repeat as a no-op at best and a conflict at worst.
    """
    out = pb.LifecycleSnapshot()
    out.CopyFrom(snapshot)
    out.state_seq = 0
    out.generated_at_unix_ms = 0
    for intent in out.intents:
        intent.state_seq = 0
    return bytes(out.SerializeToString(deterministic=True))


class LifecycleRelay:
    """Stamps every lifecycle message the pod sends the hub."""

    def __init__(self, worker_session_id: str, release_id: str) -> None:
        self._session = str(worker_session_id or "").strip()
        self._release = str(release_id or "").strip()
        self._seq = 0
        self._last: Optional[pb.LifecycleSnapshot] = None
        self._last_body = b""
        self._intent_seq: Dict[str, Tuple[int, bytes]] = {}
        self._pinned: Dict[str, pb.IntentState] = {}
        self.announced = False

    @property
    def session_id(self) -> str:
        return self._session

    def set_release(self, release_id: str) -> None:
        self._release = str(release_id or "").strip()

    # ---- the Hello pair -----------------------------------------------------

    def hello(
        self, snapshots: Sequence[Optional[pb.LifecycleSnapshot]]
    ) -> Tuple[str, Optional[pb.LifecycleSnapshot]]:
        """``(worker_session_id, snapshot)`` for the merged Hello — both or neither."""
        from . import merge

        merged = merge.merge_lifecycle_snapshots(list(snapshots))
        if merged is None:
            logger.error(
                "protocol-v5 lifecycle WITHHELD from the merged Hello (this pod "
                "connects as legacy): %d of %d compute group(s) stated no "
                "projection. A pod that ships the pair with half a projection "
                "flips its release to ExactCapabilityRequired and starves every "
                "dispatch on it.",
                sum(1 for s in snapshots if s is None),
                len(snapshots),
            )
            self.announced = False
            return "", None
        stamped = self.stamp(merged, force=True)
        if stamped is None:
            self.announced = False
            return "", None
        self.announced = True
        return self._session, stamped

    # ---- every snapshot on the wire -----------------------------------------

    def stamp(
        self, snapshot: pb.LifecycleSnapshot, *, force: bool = False
    ) -> Optional[pb.LifecycleSnapshot]:
        """Renumber, re-stamp and pin. ``None`` = nothing to send."""
        out = pb.LifecycleSnapshot()
        out.CopyFrom(snapshot)
        out.worker_session_id = self._session
        out.full_replace = True
        for intent in out.intents:
            intent.worker_session_id = self._session
        for receipt in out.goal_receipts:
            receipt.worker_session_id = self._session
        self._apply_pins(out)

        body = _snapshot_body(out)
        if body == self._last_body and not force:
            return None
        if body == self._last_body and self._last is not None:
            return self._last

        self._seq += 1
        out.state_seq = self._seq
        for intent in out.intents:
            key = str(intent.intent_id)
            intent_body = _body(intent)
            prior = self._intent_seq.get(key)
            if prior is not None and prior[1] == intent_body:
                intent.state_seq = prior[0]
            else:
                intent.state_seq = self._seq
                self._intent_seq[key] = (self._seq, intent_body)

        refusal = snapshot_refusal(out, self._release)
        if refusal:
            logger.error(
                "lifecycle snapshot HELD BACK at the parent — the hub would "
                "discard it whole: %s", refusal,
            )
            self._seq -= 1
            return None

        self._record_pins(out)
        self._last = out
        self._last_body = body
        return out

    def receipt(self, receipt: pb.GoalReceipt) -> Optional[pb.GoalReceipt]:
        """Re-stamp a child's goal receipt with the session the hub knows."""
        if not self.announced:
            return None
        out = pb.GoalReceipt()
        out.CopyFrom(receipt)
        out.worker_session_id = self._session
        return out

    # ---- terminal stickiness -------------------------------------------------

    def _apply_pins(self, snapshot: pb.LifecycleSnapshot) -> None:
        if not self._pinned:
            return
        by_id = {str(intent.intent_id): intent for intent in snapshot.intents}
        replaced: List[pb.IntentState] = []
        for intent in snapshot.intents:
            pinned = self._pinned.get(str(intent.intent_id))
            if pinned is not None and int(intent.status) not in _TERMINAL:
                replaced.append(pinned)
            else:
                replaced.append(intent)
        for intent_id, pinned in self._pinned.items():
            if intent_id not in by_id:
                replaced.append(pinned)
        del snapshot.intents[:]
        for intent in sorted(replaced, key=lambda i: str(i.intent_id)):
            snapshot.intents.append(intent)

    def _record_pins(self, snapshot: pb.LifecycleSnapshot) -> None:
        for intent in snapshot.intents:
            if int(intent.status) not in _TERMINAL:
                continue
            key = str(intent.intent_id)
            if key in self._pinned:
                continue
            copy = pb.IntentState()
            copy.CopyFrom(intent)
            self._pinned[key] = copy
        while len(self._pinned) > _MAX_PINNED_TERMINAL:
            self._pinned.pop(next(iter(self._pinned)))


__all__ = ["LifecycleRelay"]
