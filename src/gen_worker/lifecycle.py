"""The worker's protocol-v5 lifecycle PRODUCER (#1660 / th#2300).

#1373 deleted this file along with the v1 SDK, and with it the only producer of
``Hello.lifecycle_snapshot``, of mid-stream ``WorkerMessage(lifecycle_snapshot=)``
and of the ``HelloAck.desired_state_command`` -> ``GoalReceipt`` answer. The v2
worker kept ``worker_session_id`` and inherited none of it, so every live worker
booked ``worker_protocol_rejected / hello_session_id_missing_snapshot`` and the
whole fleet fell through to the hub's LEGACY health verdicts.

**A PARTIAL PRODUCER IS WORSE THAN NO PRODUCER.** The hub's cohort flag is not
health-only: `releaseUsesExactCapabilities` -> `placement.ExactCapabilityRequired`
-> `workerHasExactReadyCapabilityLocked`. The moment every live worker on a
release carries the (session id, snapshot) pair, dispatch REQUIRES a matching
READY `FunctionCapability` at the command's `config_generation` and lane plus an
ACCEPTED `GoalReceipt` for the hub's current `DesiredStateCommand`. A worker that
ships the snapshot and nothing else converts an inert typed path into a TOTAL
DISPATCH STARVE on its release.

So the session id and the snapshot leave this module as ONE PAIR, from one call,
and only when every leg of the producer is wired and the projection validates
against the hub's own rules. Anything less answers ``("", None)`` and the worker
stays legacy — inert, and correct.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Iterable, Optional, Tuple

from .config.settings import BOOT_CONFIG_GENERATION_ABSENT
from .lifecycle_intents import CapabilityFacts, IntentRegistry
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

_TERMINAL = {
    pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
    pb.LIFECYCLE_INTENT_STATUS_FAILED,
    pb.LIFECYCLE_INTENT_STATUS_CANCELED,
    pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED,
}


def snapshot_refusal(snapshot: "pb.LifecycleSnapshot", release_id: str) -> str:
    """Why the hub would DISCARD this snapshot, or ``""``.

    A mirror of `validateLifecycleSnapshot`'s mandatory-field rules
    (tensorhub `internal/orchestrator/grpc/residency_protocol.go`). The hub
    discards a snapshot that breaks any of them WHOLE — one malformed intent
    costs the entire projection and the release's typed cohort with it. Refusing
    to send it keeps the worker legacy instead, which is inert rather than wrong.
    """
    session = str(snapshot.worker_session_id or "").strip()
    if not session:
        return "snapshot carries no worker_session_id"
    if not snapshot.full_replace:
        return "snapshot is not full_replace"
    if int(snapshot.state_seq) == 0:
        return "snapshot carries no state_seq"
    if not str(release_id or "").strip():
        return "this worker has no release_id"
    seen_intents: set[str] = set()
    for intent in snapshot.intents:
        intent_id = str(intent.intent_id or "").strip()
        if not intent_id or not str(intent.goal_id or "").strip():
            return f"intent {intent_id!r} carries no identity"
        if str(intent.release_id or "").strip() != release_id:
            return f"intent {intent_id!r} release_mismatch"
        if str(intent.worker_session_id or "").strip() != session:
            return f"intent {intent_id!r} worker_session_mismatch"
        if intent_id in seen_intents:
            return f"intent {intent_id!r} is duplicated"
        seen_intents.add(intent_id)
        if (
            int(intent.state_seq) == 0
            or int(intent.status) == pb.LIFECYCLE_INTENT_STATUS_UNSPECIFIED
            or int(intent.stage) == pb.LIFECYCLE_INTENT_STAGE_UNSPECIFIED
        ):
            return f"intent {intent_id!r} missing state_seq/status/stage"
        if int(intent.since_unix_ms) <= 0 or int(intent.updated_at_unix_ms) < int(
            intent.since_unix_ms
        ):
            return f"intent {intent_id!r} invalid since/updated timestamps"
        if intent.blocker_intent_id == intent_id:
            return f"intent {intent_id!r} blocks itself"
        if int(intent.status) == pb.LIFECYCLE_INTENT_STATUS_WAITING:
            if int(intent.reason) == pb.LIFECYCLE_WAIT_REASON_UNSPECIFIED:
                return f"waiting intent {intent_id!r} states no reason"
            blocked = bool(intent.blocker_intent_id) or bool(
                intent.blocker_request.request_id
            )
            if not blocked and not (
                int(intent.next_retry_at_unix_ms) > 0
                or int(intent.deadline_at_unix_ms) > 0
            ):
                return f"waiting intent {intent_id!r} states no retry or deadline"
    seen_functions: set[str] = set()
    for capability in snapshot.capabilities:
        name = str(capability.function_name or "").strip()
        if (
            not name
            or not str(capability.release_id or "").strip()
            or int(capability.state) == pb.FUNCTION_CAPABILITY_STATE_UNSPECIFIED
        ):
            return f"capability {name!r} missing mandatory identity/state"
        if str(capability.release_id or "").strip() != release_id:
            return f"capability {name!r} release_mismatch"
        if name in seen_functions:
            return f"capability {name!r} is duplicated"
        seen_functions.add(name)
    for receipt in snapshot.goal_receipts:
        if (
            str(receipt.worker_session_id or "").strip() != session
            or int(receipt.command_seq) == 0
            or not str(receipt.goal_id or "").strip()
            or not str(receipt.release_id or "").strip()
            or int(receipt.status) == pb.GOAL_RECEIPT_STATUS_UNSPECIFIED
            or int(receipt.received_at_unix_ms) <= 0
        ):
            return f"goal receipt {receipt.goal_id!r} missing mandatory identity/status"
        if str(receipt.release_id or "").strip() != release_id:
            return f"goal receipt {receipt.goal_id!r} release_mismatch"
    application = snapshot.config_application
    if snapshot.HasField("config_application"):
        if str(application.release_id or "").strip() != release_id:
            return "config application release_mismatch"
        if int(application.target_generation) == 0:
            return "config application carries no target_generation"
        if int(application.state) == pb.CONFIG_APPLICATION_STATE_UNSPECIFIED:
            return "config application states no state"
    return ""


class WorkerLifecycle:
    """ONE intent registry per worker process, and the complete v5 producer around it."""

    def __init__(
        self,
        *,
        release_id: str,
        function_names: Iterable[str],
        facts: Callable[[], CapabilityFacts],
        send: Callable[["pb.WorkerMessage"], Awaitable[None]],
        boot_config_generation: int = BOOT_CONFIG_GENERATION_ABSENT,
    ) -> None:
        self.registry = IntentRegistry(
            release_id,
            function_names,
            boot_config_generation=boot_config_generation,
        )
        self.release_id = self.registry.release_id
        self._facts = facts
        self._send = send
        #: Set by :meth:`hello_ack_route`. The goal-receipt answer is not
        #: optional wiring a caller may forget: without it the pair below is
        #: withheld and this process never joins the typed cohort.
        self._answers_commands = False
        #: True once a Hello went out carrying the pair. Mid-stream snapshots and
        #: receipts are only legal after that — the hub matches them against a
        #: session id it must already have seen.
        self._announced = False
        self._last_sent = b""

    @property
    def worker_session_id(self) -> str:
        """The ONE producer of this process's session id (env-seeded under procsplit)."""
        return self.registry.worker_session_id

    def refresh(self) -> None:
        self.registry.refresh_projection(self._facts())

    # ---- the all-or-nothing pair --------------------------------------------

    def hello_projection(self) -> Tuple[str, Optional["pb.LifecycleSnapshot"]]:
        """``(worker_session_id, lifecycle_snapshot)`` — both, or neither.

        The pair is the fence. A caller cannot ship the session id without the
        snapshot (the hub's `hello_session_id_missing_snapshot`, which is the
        defect this issue exists for) nor the snapshot without the session id
        (`hello_snapshot_missing_session_id`), because there is one call and it
        answers with both or with nothing.

        ``("", None)`` is a LEGACY hello: no rejection row, no typed cohort, no
        dispatch starve. It is what this process sends whenever it cannot state a
        complete projection.
        """
        refusal = self._incomplete()
        if refusal:
            logger.error(
                "protocol-v5 lifecycle WITHHELD (this worker connects as legacy): %s. "
                "A half producer would flip its release to ExactCapabilityRequired "
                "and starve every dispatch on it, so the pair is all-or-nothing.",
                refusal,
            )
            self._announced = False
            return "", None
        snapshot = self.registry.snapshot()
        self._announced = True
        self._last_sent = snapshot.SerializeToString(deterministic=True)
        return self.worker_session_id, snapshot

    def _incomplete(self) -> str:
        """Why this process is not a COMPLETE v5 producer, or ``""``."""
        if not self.release_id:
            return "no release_id, so no capability can ever name one"
        if not self._answers_commands:
            return (
                "the HelloAck -> GoalReceipt route is not installed, so the hub "
                "would never hold an ACCEPTED receipt for its own command"
            )
        if self._send is None:
            return "no transport is bound, so no mid-stream snapshot could follow"
        self.refresh()
        snapshot = self.registry.snapshot()
        projected = {
            str(capability.function_name) for capability in snapshot.capabilities
        }
        missing = sorted(self.registry.function_names - projected)
        if missing:
            return f"the capability projection names no capability for {missing}"
        refusal = snapshot_refusal(snapshot, self.release_id)
        if refusal:
            return f"the hub would discard this projection: {refusal}"
        return ""

    # ---- the goal-receipt answer --------------------------------------------

    def hello_ack_route(
        self,
        body: Callable[["pb.HelloAck"], Awaitable[None]],
        *,
        prepare: Optional[Callable[["pb.HelloAck"], None]] = None,
    ) -> Callable[["pb.HelloAck"], Awaitable[None]]:
        """Wrap the worker's HelloAck handler so the command is answered FIRST.

        The hub's accept budget is 2 s from the moment it issued the command
        (`protocolGoalAcceptBudget`), and the rest of a HelloAck handler
        configures checkpoint materialization. The receipt goes out before any of
        that. ``prepare`` absorbs the hub-stated facts the projection reads
        (desired instances, resolved lanes) so the snapshot that rides with the
        receipt already carries them. Taking this route is also what arms
        :meth:`hello_projection`.
        """
        self._answers_commands = True

        async def route(ack: "pb.HelloAck") -> None:
            if prepare is not None:
                prepare(ack)
            await self.answer_hello_ack(ack)
            await body(ack)

        return route

    async def answer_hello_ack(self, ack: "pb.HelloAck") -> None:
        if not self._announced or not ack.HasField("desired_state_command"):
            return
        command = ack.desired_state_command
        receipt = self.registry.apply_command(
            command,
            current_config_generation=int(self._facts().config_generation),
        )
        await self._send(pb.WorkerMessage(goal_receipt=receipt))
        if receipt.status == pb.GOAL_RECEIPT_STATUS_REJECTED:
            logger.error(
                "desired-state command %s/%d REJECTED: %s (%s)",
                command.goal_id, command.command_seq, receipt.detail,
                "mandatory" if self.registry.protocol_rejected else "advisory",
            )
        await self.publish(force=True)

    # ---- mid-stream snapshots ------------------------------------------------

    async def publish(self, *, force: bool = False) -> None:
        """Send the current projection if it CHANGED (the hub drops a repeat seq)."""
        if not self._announced:
            return
        self.refresh()
        snapshot = self.registry.snapshot()
        raw = snapshot.SerializeToString(deterministic=True)
        if not force and raw == self._last_sent:
            return
        refusal = snapshot_refusal(snapshot, self.release_id)
        if refusal:
            logger.error(
                "mid-stream lifecycle snapshot HELD BACK — the hub would discard "
                "it whole and the projection would freeze at its last good state: %s",
                refusal,
            )
            return
        self._last_sent = raw
        await self._send(pb.WorkerMessage(lifecycle_snapshot=snapshot))


def terminal_intents(snapshot: "pb.LifecycleSnapshot") -> dict:
    """The snapshot's intents that have reached a terminal status, by id."""
    return {
        str(intent.intent_id): intent
        for intent in snapshot.intents
        if int(intent.status) in _TERMINAL
    }


__all__ = [
    "WorkerLifecycle",
    "snapshot_refusal",
    "terminal_intents",
]
