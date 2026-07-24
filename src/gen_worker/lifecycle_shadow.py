"""th#1085 protocol-v5 shadow state.

The legacy residency state machine remains authoritative in Slice 1. This
module only validates causally identified commands and projects current state
for receipts and reconnect Hello messages.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import OrderedDict
from typing import Any, Iterable, Optional

from .pb import worker_scheduler_pb2 as pb

_MAX_INTENTS = 128
_MAX_RECEIPTS = 32
_ACTIVE_INTENT_STATES = {
    pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
    pb.LIFECYCLE_INTENT_STATUS_WAITING,
    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
}
_SUPPORTED_INTENT_KINDS = {
    pb.DESIRED_INTENT_KIND_MATERIALIZE,
    pb.DESIRED_INTENT_KIND_FUNCTION_READY,
    pb.DESIRED_INTENT_KIND_CONFIG_APPLY,
    pb.DESIRED_INTENT_KIND_COMPILE_ADOPT,
    pb.DESIRED_INTENT_KIND_DRAIN,
}


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _clone(message: Any) -> Any:
    out = type(message)()
    out.CopyFrom(message)
    return out


def _command_digest(command: "pb.DesiredStateCommand") -> bytes:
    """Stable across transport resend timestamps for the same desired goal."""
    semantic = _clone(command)
    semantic.issued_at_unix_ms = 0
    semantic.accept_by_unix_ms = 0
    semantic.first_action_by_unix_ms = 0
    return hashlib.sha256(semantic.SerializeToString(deterministic=True)).digest()


def _binding_digest(instance: Optional["pb.DesiredInstance"]) -> bytes:
    if instance is None:
        return b""
    return hashlib.sha256(instance.SerializeToString(deterministic=True)).digest()


def _intent_identity_digest(intent: "pb.DesiredIntent") -> bytes:
    identity = _clone(intent)
    identity.cause = pb.DESIRED_INTENT_CAUSE_UNSPECIFIED
    identity.ClearField("waiting_requests")
    identity.priority = 0
    identity.mandatory = False
    return hashlib.sha256(identity.SerializeToString(deterministic=True)).digest()


class LifecycleShadow:
    """Bounded full-replacement projection for one worker process."""

    def __init__(self, release_id: str, function_names: Iterable[str]) -> None:
        self.worker_session_id = uuid.uuid4().hex
        self.release_id = str(release_id or "").strip()
        self.function_names = frozenset(
            str(name).strip() for name in function_names if str(name).strip()
        )
        self.protocol_rejected = False
        self._state_seq = 1
        self._updated_at_ms = _now_ms()
        self._last_command_seq = 0
        self._last_command_digest = b""
        self._last_receipt: Optional[pb.GoalReceipt] = None
        self._command_receipts: "OrderedDict[tuple[int, bytes], pb.GoalReceipt]" = OrderedDict()
        self._target_config_generation = 0
        self._intents: "OrderedDict[str, pb.IntentState]" = OrderedDict()
        self._intent_digests: dict[str, bytes] = {}
        self._receipts: "OrderedDict[str, pb.GoalReceipt]" = OrderedDict()
        self._capabilities: list[pb.FunctionCapability] = []
        self._config_application = pb.ConfigApplication()
        self._drain = pb.DrainProjection()

    def _touch(self) -> int:
        self._state_seq += 1
        self._updated_at_ms = _now_ms()
        return self._state_seq

    def _remember_receipt(self, receipt: "pb.GoalReceipt") -> None:
        self._receipts.pop(receipt.goal_id, None)
        self._receipts[receipt.goal_id] = _clone(receipt)
        while len(self._receipts) > _MAX_RECEIPTS:
            self._receipts.popitem(last=False)
        self._last_receipt = _clone(receipt)

    def _remember_command_receipt(
        self,
        command_seq: int,
        digest: bytes,
        receipt: "pb.GoalReceipt",
    ) -> None:
        key = (command_seq, digest)
        self._command_receipts.pop(key, None)
        self._command_receipts[key] = _clone(receipt)
        while len(self._command_receipts) > _MAX_RECEIPTS:
            self._command_receipts.popitem(last=False)

    def _reject(
        self,
        command: "pb.DesiredStateCommand",
        errors: list[tuple[str, "pb.LifecycleErrorCode", str]],
        digest: bytes,
        *,
        fail_closed: bool,
    ) -> "pb.GoalReceipt":
        now = _now_ms()
        error_code = errors[0][1]
        goal_id = str(command.goal_id or "").strip()
        if not goal_id:
            goal_id = f"invalid-{digest.hex()[:16]}"
        receipt = pb.GoalReceipt(
            worker_session_id=self.worker_session_id,
            command_seq=max(1, int(command.command_seq)),
            goal_id=goal_id,
            release_id=str(command.release_id or self.release_id),
            status=pb.GOAL_RECEIPT_STATUS_REJECTED,
            error_code=error_code,
            rejections=[
                pb.IntentRejection(
                    intent_id=intent_id,
                    error_code=code,
                    detail=detail,
                )
                for intent_id, code, detail in errors
            ],
            detail=errors[0][2],
            received_at_unix_ms=now,
            command_digest=digest,
        )
        self.protocol_rejected = self.protocol_rejected or fail_closed
        self._touch()
        self._remember_receipt(receipt)
        self._remember_command_receipt(
            int(command.command_seq),
            digest,
            receipt,
        )
        return receipt

    def apply_command(
        self,
        command: "pb.DesiredStateCommand",
        *,
        current_config_generation: int = 0,
    ) -> "pb.GoalReceipt":
        """Validate and acknowledge a shadow command.

        A byte-equivalent resend at the same sequence returns the original
        receipt. Unknown or malformed mandatory work sets ``protocol_rejected``
        so Lifecycle can stop legacy fallback and advertise ERROR.
        """
        digest = _command_digest(command)
        cached = self._command_receipts.get((int(command.command_seq), digest))
        if cached is not None:
            return _clone(cached)

        command_errors: list[tuple[str, "pb.LifecycleErrorCode", str]] = []
        if not str(command.worker_session_id or "").strip():
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                    "worker_session_id is required",
                )
            )
        elif command.worker_session_id != self.worker_session_id:
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_WORKER_SESSION_MISMATCH,
                    "worker_session_id does not match this process",
                )
            )
        if int(command.command_seq) <= 0 or not str(command.goal_id or "").strip():
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                    "command_seq and goal_id are required",
                )
            )
        if not str(command.release_id or "").strip():
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                    "release_id is required",
                )
            )
        elif self.release_id and command.release_id != self.release_id:
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_RELEASE_MISMATCH,
                    "release_id does not match this worker",
                )
            )
        if int(command.command_seq) < self._last_command_seq:
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_COMMAND_SEQ_REGRESSION,
                    "command_seq regressed",
                )
            )
        elif (
            int(command.command_seq) == self._last_command_seq
            and self._last_command_digest
            and digest != self._last_command_digest
        ):
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_COMMAND_SEQ_CONFLICT,
                    "command_seq was reused for a different goal",
                )
            )
        if int(command.config_generation) < int(current_config_generation):
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_CONFIG_REGRESSION,
                    "config_generation regressed",
                )
            )
        if int(command.config_generation) > 0 and not bytes(command.config_digest):
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                    "config_digest is required when config_generation is set",
                )
            )

        seen: set[str] = set()
        for intent in command.intents:
            intent_id = str(intent.intent_id or "").strip()
            if not intent_id:
                command_errors.append(
                    (
                        "",
                        pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                        "intent_id is required",
                    )
                )
                continue
            if intent_id in seen:
                command_errors.append(
                    (
                        intent_id,
                        pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                        "intent_id is duplicated",
                    )
                )
                continue
            seen.add(intent_id)
            intent_digest = _intent_identity_digest(intent)
            if (
                intent_id in self._intent_digests
                and self._intent_digests[intent_id] != intent_digest
            ):
                command_errors.append(
                    (
                        intent_id,
                        pb.LIFECYCLE_ERROR_CODE_COMMAND_SEQ_CONFLICT,
                        "intent_id was reused for different work",
                    )
                )
            kind = int(intent.kind)
            if int(intent.cause) == pb.DESIRED_INTENT_CAUSE_UNSPECIFIED:
                command_errors.append(
                    (
                        intent_id,
                        pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                        "intent cause is required",
                    )
                )
            if kind not in _SUPPORTED_INTENT_KINDS:
                command_errors.append(
                    (
                        intent_id,
                        pb.LIFECYCLE_ERROR_CODE_UNSUPPORTED_INTENT,
                        f"unsupported intent kind {kind}",
                    )
                )
            elif (
                kind == pb.DESIRED_INTENT_KIND_FUNCTION_READY
                and str(intent.function_name or "").strip() not in self.function_names
            ):
                command_errors.append(
                    (
                        intent_id,
                        pb.LIFECYCLE_ERROR_CODE_UNKNOWN_FUNCTION,
                        f"unknown function {intent.function_name!r}",
                    )
                )
            elif kind == pb.DESIRED_INTENT_KIND_FUNCTION_READY and not bytes(intent.binding_digest):
                command_errors.append(
                    (
                        intent_id,
                        pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                        "function-ready intent requires binding_digest",
                    )
                )
            elif kind == pb.DESIRED_INTENT_KIND_MATERIALIZE and (
                not str(intent.ref or "").strip() or not bytes(intent.snapshot_digest)
            ):
                command_errors.append(
                    (
                        intent_id,
                        pb.LIFECYCLE_ERROR_CODE_SNAPSHOT_IDENTITY_MISSING,
                        "materialize intent requires ref and snapshot_digest",
                    )
                )

        if command_errors:
            mandatory_ids = {
                str(intent.intent_id or "").strip()
                for intent in command.intents
                if intent.mandatory
            }
            fail_closed = bool(command.mandatory) or any(
                not intent_id or intent_id in mandatory_ids
                for intent_id, _code, _detail in command_errors
            )
            return self._reject(
                command,
                command_errors,
                digest,
                fail_closed=fail_closed,
            )

        now = _now_ms()
        incoming_ids = {str(intent.intent_id) for intent in command.intents}
        for intent_id, state in self._intents.items():
            if intent_id not in incoming_ids and int(state.status) in _ACTIVE_INTENT_STATES:
                state.status = pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED
                state.updated_at_unix_ms = now
                state.state_seq = self._touch()
        for intent in command.intents:
            intent_digest = _intent_identity_digest(intent)
            retained = self._intents.get(intent.intent_id)
            if retained is not None and self._intent_digests.get(intent.intent_id) == intent_digest:
                retained.goal_id = command.goal_id
                retained.release_id = command.release_id
                retained.config_generation = command.config_generation
                retained.updated_at_unix_ms = now
                retained.deadline_at_unix_ms = command.first_action_by_unix_ms
                retained.state_seq = self._touch()
                self._intents.move_to_end(intent.intent_id)
                continue
            state = pb.IntentState(
                worker_session_id=self.worker_session_id,
                goal_id=command.goal_id,
                intent_id=intent.intent_id,
                release_id=command.release_id,
                config_generation=command.config_generation,
                status=pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
                stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                since_unix_ms=now,
                updated_at_unix_ms=now,
                deadline_at_unix_ms=command.first_action_by_unix_ms,
            )
            state.state_seq = self._touch()
            self._intents.pop(intent.intent_id, None)
            self._intents[intent.intent_id] = state
            self._intent_digests[intent.intent_id] = intent_digest
            if intent.kind == pb.DESIRED_INTENT_KIND_DRAIN:
                self._drain = pb.DrainProjection(
                    goal_id=command.goal_id,
                    intent_id=intent.intent_id,
                    status=pb.DRAIN_LIFECYCLE_STATUS_ACCEPTED,
                    since_unix_ms=now,
                    updated_at_unix_ms=now,
                    deadline_at_unix_ms=command.first_action_by_unix_ms,
                )
        self._trim_intents()
        self._last_command_seq = int(command.command_seq)
        self._last_command_digest = digest
        self._target_config_generation = int(command.config_generation)
        self._refresh_config_application(received=True)
        receipt = pb.GoalReceipt(
            worker_session_id=self.worker_session_id,
            command_seq=command.command_seq,
            goal_id=command.goal_id,
            release_id=command.release_id,
            status=pb.GOAL_RECEIPT_STATUS_ACCEPTED,
            received_at_unix_ms=now,
            command_digest=digest,
        )
        self._touch()
        self._remember_receipt(receipt)
        self._remember_command_receipt(
            int(command.command_seq),
            digest,
            receipt,
        )
        return receipt

    def _trim_intents(self) -> None:
        if len(self._intents) <= _MAX_INTENTS:
            return
        for intent_id, state in list(self._intents.items()):
            if int(state.status) in _ACTIVE_INTENT_STATES:
                continue
            self._intents.pop(intent_id)
            self._intent_digests.pop(intent_id, None)
            if len(self._intents) <= _MAX_INTENTS:
                return

    def _refresh_config_application(self, *, received: bool = False) -> None:
        target = self._target_config_generation
        if target <= 0:
            next_application = pb.ConfigApplication()
        else:
            next_application = pb.ConfigApplication(
                release_id=self.release_id,
                target_generation=target,
                received_generation=target
                if received
                else int(self._config_application.received_generation),
                # Slice 1 cannot prove per-class convergence yet. Keep those
                # counters at zero and report the conservative pending union.
                state=pb.CONFIG_APPLICATION_STATE_APPLYING,
                pending_classes=pb.ConfigClassMask(
                    parameters=True,
                    bindings=True,
                    boot=True,
                ),
            )
        if next_application.SerializeToString(
            deterministic=True
        ) != self._config_application.SerializeToString(deterministic=True):
            self._config_application = next_application
            self._touch()

    def refresh_projection(
        self,
        executor: Any,
        desired: Optional["pb.DesiredResidency"],
        resolutions: dict[str, tuple[str, str, str]],
    ) -> None:
        """Shadow-project exact capabilities from current legacy state."""
        if not self.release_id:
            capabilities: list[pb.FunctionCapability] = []
        else:
            hot = {
                instance.function_name: instance
                for instance in (desired.hot if desired is not None else ())
                if instance.function_name
            }
            residency = {model.ref: model for model in executor.store.residency_snapshot()}
            available = set(executor.available_functions())
            compile_targets = {
                name: target.incarnation_id
                for target in executor.compile_targets()
                for name in target.function_names
            }
            capabilities = []
            for name in sorted(self.function_names):
                instance = hot.get(name)
                model_refs = sorted(
                    {model.ref for model in (instance.models if instance else ()) if model.ref}
                )
                lanes = sorted(
                    {
                        resolutions.get(ref, ("", "", ""))[2]
                        for ref in model_refs
                        if resolutions.get(ref, ("", "", ""))[2]
                    }
                )
                if name in available:
                    state = pb.FUNCTION_CAPABILITY_STATE_READY
                elif name in executor.unavailable:
                    state = pb.FUNCTION_CAPABILITY_STATE_FAILED
                else:
                    state = pb.FUNCTION_CAPABILITY_STATE_APPLYING
                capabilities.append(
                    pb.FunctionCapability(
                        function_name=name,
                        release_id=self.release_id,
                        config_generation=int(executor.runtime_config.generation),
                        binding_digest=_binding_digest(instance),
                        lane=",".join(lanes),
                        models=[
                            pb.ModelIdentity(
                                ref=ref,
                                snapshot_digest=(
                                    residency[ref].snapshot_digest.encode()
                                    if ref in residency
                                    else b""
                                ),
                                tier=(
                                    residency[ref].tier
                                    if ref in residency
                                    else pb.RESIDENCY_TIER_UNSPECIFIED
                                ),
                                residency_generation=(
                                    residency[ref].residency_generation if ref in residency else 0
                                ),
                            )
                            for ref in model_refs
                        ],
                        compile_target_incarnation_id=compile_targets.get(name, ""),
                        state=state,
                    )
                )
        old = b"".join(item.SerializeToString(deterministic=True) for item in self._capabilities)
        new = b"".join(item.SerializeToString(deterministic=True) for item in capabilities)
        if new != old:
            self._capabilities = capabilities
            self._touch()

    def set_drain(
        self,
        status: "pb.DrainLifecycleStatus",
        *,
        deadline_at_unix_ms: int = 0,
        detail: str = "",
        error_code: "pb.LifecycleErrorCode" = pb.LIFECYCLE_ERROR_CODE_UNSPECIFIED,
    ) -> None:
        now = _now_ms()
        since = int(self._drain.since_unix_ms or now)
        goal_id = self._drain.goal_id or f"legacy-drain-{self.worker_session_id}"
        intent_id = self._drain.intent_id or goal_id
        next_drain = pb.DrainProjection(
            goal_id=goal_id,
            intent_id=intent_id,
            status=status,
            since_unix_ms=since,
            updated_at_unix_ms=now,
            deadline_at_unix_ms=(deadline_at_unix_ms or self._drain.deadline_at_unix_ms),
            error_code=error_code,
            detail=detail,
        )
        if next_drain.SerializeToString(deterministic=True) != self._drain.SerializeToString(
            deterministic=True
        ):
            self._drain = next_drain
            self._touch()

    def snapshot(self) -> "pb.LifecycleSnapshot":
        snapshot = pb.LifecycleSnapshot(
            worker_session_id=self.worker_session_id,
            state_seq=self._state_seq,
            intents=[_clone(state) for state in self._intents.values()],
            capabilities=[_clone(item) for item in self._capabilities],
            goal_receipts=[_clone(receipt) for receipt in self._receipts.values()],
            full_replace=True,
            generated_at_unix_ms=self._updated_at_ms,
        )
        if self._config_application.ByteSize():
            snapshot.config_application.CopyFrom(self._config_application)
        if self._drain.ByteSize():
            snapshot.drain.CopyFrom(self._drain)
        return snapshot


__all__ = ["LifecycleShadow"]
