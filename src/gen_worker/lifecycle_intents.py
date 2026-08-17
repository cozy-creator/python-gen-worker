"""Lifecycle intent registry and reconnect projection."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Iterable, Optional, TypeVar

from .config.settings import BOOT_CONFIG_GENERATION_ABSENT
from .pb import worker_scheduler_pb2 as pb

# Retention bounds on hub-driven maps in a long-lived worker process: the hub
# opens intents and the worker answers with receipts, and neither side ever
# says "you may forget this one". Without a bound the registry leaks at a rate
# set by hub traffic. Trimmed oldest-first, so what is dropped is what no
# reconnect projection will be asked about.
_MAX_INTENTS = 128
_MAX_RECEIPTS = 32
#: How long a caller waits for an intent's first report before proceeding
#: without one. Not a kill: nothing is cancelled, the caller just stops
#: blocking on a report that is a courtesy.
_UNREPORTED_WAIT_TIMEOUT_S = 2.0
# Fallback deadline for a WAITING state with no blocker and no retry time.
# Mirrors the hub's shadow first-action budget (60s). NOT a kill: it ends
# nothing on this side. `deadline_at_unix_ms` is a required wire field (the
# hub's shadow validator rejects a WAITING state carrying none of
# blocker/retry/deadline), so this fills a protocol hole and the hub owns what
# it decides at expiry. Changing it is a protocol change, not a tuning.
#
# pgw#1336 RE-READ IT AGAINST th#2052's `RunJob.phase_budget_s` and KEPT IT:
# they answer different questions and are not two spellings of one number.
# This one fills a REQUIRED FIELD on a WAITING report — every WAITING intent
# needs one of blocker/retry/deadline or the hub's shadow validator rejects
# the frame, whoever authored the intent. `phase_budget_s` is the operator's
# POSITION-ADVANCE budget for one job's body (`jobs.ProgressWatch`), applies
# only to a job, and kills nothing here either. Deleting this with the RunJob
# compat minter would have made every blockerless WAITING report invalid,
# including hub-authored ones.
_WAITING_DEADLINE_FALLBACK_MS = 60_000
_ACTIVE_INTENT_STATES = {
    pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
    pb.LIFECYCLE_INTENT_STATUS_WAITING,
    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
}
_TERMINAL_INTENT_STATES = {
    pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
    pb.LIFECYCLE_INTENT_STATUS_FAILED,
    pb.LIFECYCLE_INTENT_STATUS_CANCELED,
    pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED,
}
_LEGAL_TRANSITIONS = {
    pb.LIFECYCLE_INTENT_STATUS_ACCEPTED: _ACTIVE_INTENT_STATES | _TERMINAL_INTENT_STATES,
    pb.LIFECYCLE_INTENT_STATUS_WAITING: _ACTIVE_INTENT_STATES | _TERMINAL_INTENT_STATES,
    pb.LIFECYCLE_INTENT_STATUS_RUNNING: _ACTIVE_INTENT_STATES | _TERMINAL_INTENT_STATES,
}
_SUPPORTED_INTENT_KINDS = {
    pb.DESIRED_INTENT_KIND_MATERIALIZE,
    pb.DESIRED_INTENT_KIND_FUNCTION_READY,
    pb.DESIRED_INTENT_KIND_CONFIG_APPLY,
    pb.DESIRED_INTENT_KIND_COMPILE_ADOPT,
    pb.DESIRED_INTENT_KIND_DRAIN,
}
_T = TypeVar("_T")


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


def _binding_digest(
    function_name: str,
    instance: Optional["pb.DesiredInstance"],
) -> bytes:
    exact = instance or pb.DesiredInstance(function_name=function_name)
    return hashlib.sha256(exact.SerializeToString(deterministic=True)).digest()


def _intent_identity_digest(intent: "pb.DesiredIntent") -> bytes:
    identity = _clone(intent)
    identity.cause = pb.DESIRED_INTENT_CAUSE_UNSPECIFIED
    identity.ClearField("waiting_requests")
    identity.priority = 0
    identity.mandatory = False
    return hashlib.sha256(identity.SerializeToString(deterministic=True)).digest()


class UnreportedIntentWait(RuntimeError):
    """A protocol-owned await outlived the reporting grace period."""


class IntentRegistry:
    """Current desired intents, legal transitions, and bounded reconnect state."""

    def __init__(
        self,
        release_id: str,
        function_names: Iterable[str],
        *,
        boot_config_generation: int = BOOT_CONFIG_GENERATION_ABSENT,
        on_change: Optional[Callable[[], None]] = None,
        unreported_wait_timeout_s: float = _UNREPORTED_WAIT_TIMEOUT_S,
    ) -> None:
        # Under the process split the PARENT mints the session id once and
        # passes it down (GEN_WORKER_SESSION_ID) so it survives child respawns;
        # a child-minted id changes on every respawn and the hub rejects the
        # cross-session shadow state. Absent the env (no split), a fresh uuid.
        self.worker_session_id = (
            os.environ.get("GEN_WORKER_SESSION_ID", "").strip() or uuid.uuid4().hex
        )
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
        # Two distinct facts, not one number. ``_boot_config_injected`` False
        # means no boot-only environment exists for this process to be stale
        # against (a host-process/BYO worker: tensorhub injects
        # WORKER_CONFIG_GENERATION only into pod-launch env), so the boot class
        # is NOT APPLICABLE and converges. A pod-launched worker with a
        # genuinely old stamp still reports BOOT_STALE so the hub replaces it.
        self._boot_config_injected = int(boot_config_generation) >= 0
        self._boot_config_generation = max(0, int(boot_config_generation))
        self._intents: "OrderedDict[str, pb.IntentState]" = OrderedDict()
        self._intent_digests: dict[str, bytes] = {}
        self._desired_intents: dict[str, pb.DesiredIntent] = {}
        self._receipts: "OrderedDict[str, pb.GoalReceipt]" = OrderedDict()
        self._capabilities: list[pb.FunctionCapability] = []
        self._config_application = pb.ConfigApplication()
        self._drain = pb.DrainProjection()
        self._on_change = on_change or (lambda: None)
        self._unreported_wait_timeout_s = max(0.001, float(unreported_wait_timeout_s))

    def _touch(self) -> int:
        self._state_seq += 1
        self._updated_at_ms = _now_ms()
        self._on_change()
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
        """Validate, register, and acknowledge a desired-state command.

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
        if int(command.config_generation) > 0 and not bytes(command.parameter_snapshot):
            command_errors.append(
                (
                    "",
                    pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD,
                    "parameter_snapshot is required when config_generation is set",
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

        # The hub declares fail-closed scope PER INTENT. Errors on mandatory
        # work reject the command and latch; a command-level error rejects and
        # latches only when it abandons mandatory work not already registered
        # under the same identity; errors scoped to advisory intents decline
        # exactly those intents and accept the rest, so a bad preposition or
        # binding cannot brick the process.
        declined: dict[str, tuple["pb.LifecycleErrorCode", str]] = {}
        if command_errors:
            intents_by_id = {
                str(intent.intent_id or "").strip(): intent for intent in command.intents
            }
            mandatory_ids = {
                intent_id
                for intent_id, intent in intents_by_id.items()
                if intent_id and intent.mandatory
            }
            command_level = [error for error in command_errors if not error[0]]
            touches_mandatory = any(
                intent_id in mandatory_ids for intent_id, _code, _detail in command_errors
            )
            new_mandatory_work = any(
                intent_id in mandatory_ids
                and self._intent_digests.get(intent_id) != _intent_identity_digest(intent)
                for intent_id, intent in intents_by_id.items()
            )
            if command_level or touches_mandatory:
                fail_closed = (
                    bool(command.mandatory)
                    or touches_mandatory
                    or (bool(command_level) and new_mandatory_work)
                )
                return self._reject(
                    command,
                    command_errors,
                    digest,
                    fail_closed=fail_closed,
                )
            for intent_id, code, detail in command_errors:
                declined.setdefault(intent_id, (code, detail))

        now = _now_ms()
        accepted_intents = [
            intent
            for intent in command.intents
            if str(intent.intent_id or "").strip() not in declined
        ]
        incoming_ids = {str(intent.intent_id) for intent in accepted_intents}
        for intent_id, state in self._intents.items():
            # A command is a full replacement of COMMAND-OWNED work only.
            # Worker-local compat obligations (job/setup/materialize carriers,
            # never present in any hub command) must survive a generation bump:
            # superseding them terminalizes a LIVE job's intent mid-flight, so
            # its next legal transition raises. Command-born intents are exactly
            # the ids in _intent_digests.
            if intent_id not in self._intent_digests:
                continue
            if intent_id not in incoming_ids and int(state.status) in _ACTIVE_INTENT_STATES:
                state.status = pb.LIFECYCLE_INTENT_STATUS_SUPERSEDED
                state.updated_at_unix_ms = now
                state.state_seq = self._touch()
        self._desired_intents = {
            str(intent.intent_id): _clone(intent) for intent in accepted_intents
        }
        for intent in accepted_intents:
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
        for intent_id, (code, detail) in declined.items():
            prior = self._intents.get(intent_id)
            if prior is not None and int(prior.status) in _ACTIVE_INTENT_STATES:
                self.transition(
                    intent_id,
                    pb.LIFECYCLE_INTENT_STATUS_FAILED,
                    pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                    error_code=code,
                    detail=detail,
                )
            elif prior is None:
                declined_state = pb.IntentState(
                    worker_session_id=self.worker_session_id,
                    goal_id=command.goal_id,
                    intent_id=intent_id,
                    release_id=command.release_id,
                    config_generation=command.config_generation,
                    status=pb.LIFECYCLE_INTENT_STATUS_FAILED,
                    stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                    since_unix_ms=now,
                    updated_at_unix_ms=now,
                    error_code=code,
                    detail=detail,
                )
                declined_state.state_seq = self._touch()
                self._intents[intent_id] = declined_state
        self._trim_intents()
        self._last_command_seq = int(command.command_seq)
        self._last_command_digest = digest
        changed_classes = (
            command.changed_config_classes if command.HasField("changed_config_classes") else None
        )
        self.receive_config_generation(
            int(command.config_generation),
            changed_classes=changed_classes,
        )
        receipt = pb.GoalReceipt(
            worker_session_id=self.worker_session_id,
            command_seq=command.command_seq,
            goal_id=command.goal_id,
            release_id=command.release_id,
            status=pb.GOAL_RECEIPT_STATUS_ACCEPTED,
            rejections=[
                pb.IntentRejection(intent_id=intent_id, error_code=code, detail=detail)
                for intent_id, (code, detail) in declined.items()
            ],
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

    def intent_id(
        self,
        kind: "pb.DesiredIntentKind",
        *,
        function_name: str = "",
        ref: str = "",
    ) -> str:
        """Return the current command intent matching exact work identity."""
        for intent_id, intent in self._desired_intents.items():
            if int(intent.kind) != int(kind):
                continue
            if function_name and intent.function_name != function_name:
                continue
            if ref and intent.ref != ref:
                continue
            state = self._intents.get(intent_id)
            if state is not None and int(state.status) in _ACTIVE_INTENT_STATES:
                return intent_id
        return ""

    def is_active(self, intent_id: str) -> bool:
        state = self._intents.get(intent_id)
        return state is not None and int(state.status) in _ACTIVE_INTENT_STATES

    def ensure_intent(
        self,
        kind: "pb.DesiredIntentKind",
        *,
        function_name: str = "",
        ref: str = "",
    ) -> str:
        """Find command-owned work or create a compatibility intent.

        Never returns "" when a command is registered but the matching intent
        is terminal or absent: re-verifying converged command work is
        legitimate and needs a REPORTABLE carrier, so this mints the same
        worker-local compat intent the no-command path uses. Without one, the
        reconcile re-pass's first await (``wait_idle``, which blocks for the
        whole duration of any in-flight tenant job) trips the unreported-wait
        timeout and drives a healthy worker to WORKER_PHASE_ERROR.
        ``guard_await``'s fail-closed remains for waits that genuinely carry
        no intent id.
        """
        intent_id = self.intent_id(kind, function_name=function_name, ref=ref)
        if intent_id:
            return intent_id
        identity = f"{int(kind)}\0{function_name}\0{ref}".encode()
        base_intent_id = f"compat-{hashlib.sha256(identity).hexdigest()[:24]}"
        intent_id = base_intent_id
        existing = self._intents.get(intent_id)
        if existing is not None and int(existing.status) in _ACTIVE_INTENT_STATES:
            return intent_id
        if existing is not None:
            suffix = self._state_seq
            intent_id = f"{base_intent_id}-{suffix}"
            while intent_id in self._intents:
                suffix += 1
                intent_id = f"{base_intent_id}-{suffix}"
        now = _now_ms()
        state = pb.IntentState(
            worker_session_id=self.worker_session_id,
            goal_id=f"compat-{self.worker_session_id}",
            intent_id=intent_id,
            release_id=self.release_id,
            status=pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
            stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
            since_unix_ms=now,
            updated_at_unix_ms=now,
        )
        state.state_seq = self._touch()
        self._intents[intent_id] = state
        self._desired_intents[intent_id] = pb.DesiredIntent(
            intent_id=intent_id,
            kind=kind,
            cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION,
            function_name=function_name,
            ref=ref,
        )
        self._trim_intents()
        return intent_id

    def ensure_local_intent(
        self,
        scope: str,
        identity: str,
        *,
        function_name: str = "",
        detail: str = "",
    ) -> str:
        """Create a bounded worker-local compatibility obligation.

        These intents cover operations protocol v5 does not carry an intent
        for: ``setup`` / ``materialize`` and their single-flight waiters (the
        hub opens FUNCTION_READY / MATERIALIZE intents when it commands the
        work, and none at all when the worker reaches it on its own), plus a
        SERVED REQUEST, which has no intent kind on this protocol by design —
        a request is what an intent gets BLOCKED on (``IntentState
        .blocker_request``), never an intent itself.

        **A JOB is no longer one of them** (pgw#1336 / th#2052). The docstring
        that stood here said these cover "a RunJob (the wire lacks a job intent
        kind/owner field)"; the wire grew that field, so a job dispatch arrives
        owning a hub-authored carrier and goes through
        :meth:`adopt_dispatch_intent` instead.

        Their IDs are deterministic for one operation identity, but they never
        impersonate a hub-authored DesiredIntent.
        """
        raw = f"{scope}\0{identity}\0{function_name}".encode()
        base_intent_id = f"compat-{scope}-{hashlib.sha256(raw).hexdigest()[:24]}"
        intent_id = base_intent_id
        existing = self._intents.get(intent_id)
        if existing is not None and int(existing.status) in _ACTIVE_INTENT_STATES:
            return intent_id
        if existing is not None:
            suffix = self._state_seq
            intent_id = f"{base_intent_id}-{suffix}"
            while intent_id in self._intents:
                suffix += 1
                intent_id = f"{base_intent_id}-{suffix}"
        now = _now_ms()
        state = pb.IntentState(
            worker_session_id=self.worker_session_id,
            goal_id=f"compat-{scope}-{self.worker_session_id}",
            intent_id=intent_id,
            release_id=self.release_id,
            config_generation=self._target_config_generation,
            status=pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
            stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
            since_unix_ms=now,
            updated_at_unix_ms=now,
            detail=detail,
        )
        state.state_seq = self._touch()
        self._intents[intent_id] = state
        self._trim_intents()
        return intent_id

    def adopt_dispatch_intent(
        self,
        intent_id: str,
        goal_id: str,
        *,
        detail: str = "",
    ) -> str:
        """Register the HUB-AUTHORED lifecycle carrier a dispatch names.

        th#2052 put ``intent_kind`` / ``intent_id`` / ``goal_id`` on ``RunJob``,
        so a job dispatch now arrives owning its carrier and the worker reports
        against THAT id instead of minting a worker-local ``compat-*`` one.
        That minting is what pgw#1307 arm (8) could not delete, and the reason
        it could not was this exact absence.

        **The id is never renamed.** The compat minter appended a ``-N`` suffix
        when it found a terminal state under the id it wanted, because the id
        was its own to choose. This one is the hub's: a redelivery naming an id
        the registry already holds terminal is the hub saying "this obligation
        again", so the entry is REPLACED under the same id rather than reported
        under a second one the hub has never heard of. An id it holds ACTIVE is
        the same live obligation and is returned untouched.

        Deliberately NOT written into ``_desired_intents``/``_intent_digests``:
        those are the command-born set, rebuilt wholesale by every
        ``apply_command`` and superseded on a generation bump. An in-flight
        job's carrier must survive both. For the same reason
        ``DESIRED_INTENT_KIND_RUN_JOB`` is deliberately absent from
        ``_SUPPORTED_INTENT_KINDS``: the hub stamps it on the DISPATCH, never
        opens it as a DesiredIntent in a command, so accepting one there would
        be claiming support for a shape the hub does not send.
        """
        intent_id = str(intent_id or "").strip()
        if not intent_id:
            raise ValueError(
                "adopt_dispatch_intent requires the hub-authored intent id; "
                "a dispatch that carries none has no carrier to adopt"
            )
        existing = self._intents.get(intent_id)
        if existing is not None and int(existing.status) in _ACTIVE_INTENT_STATES:
            return intent_id
        now = _now_ms()
        state = pb.IntentState(
            worker_session_id=self.worker_session_id,
            goal_id=str(goal_id or ""),
            intent_id=intent_id,
            release_id=self.release_id,
            config_generation=self._target_config_generation,
            status=pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
            stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
            since_unix_ms=now,
            updated_at_unix_ms=now,
            detail=detail,
        )
        state.state_seq = self._touch()
        self._intents[intent_id] = state
        self._trim_intents()
        return intent_id

    def transition(
        self,
        intent_id: str,
        status: "pb.LifecycleIntentStatus",
        stage: "pb.LifecycleIntentStage",
        *,
        reason: "pb.LifecycleWaitReason" = pb.LIFECYCLE_WAIT_REASON_UNSPECIFIED,
        next_retry_at_unix_ms: int = 0,
        deadline_at_unix_ms: int = 0,
        blocker_intent_id: str = "",
        blocker_request: Optional["pb.RequestAttempt"] = None,
        progress: Optional["pb.LifecycleProgress"] = None,
        error_code: "pb.LifecycleErrorCode" = pb.LIFECYCLE_ERROR_CODE_UNSPECIFIED,
        detail: str = "",
        actual_digest: bytes = b"",
    ) -> None:
        """Apply one legal state transition to a current intent."""
        state = self._intents.get(intent_id)
        if state is None:
            raise KeyError(f"unknown lifecycle intent {intent_id!r}")
        current = state.status
        target = status
        if current in _TERMINAL_INTENT_STATES:
            if current == target:
                return
            raise ValueError(
                f"terminal lifecycle intent {intent_id!r} cannot transition "
                f"from {current} to {target}"
            )
        if target not in _LEGAL_TRANSITIONS.get(current, set()):
            raise ValueError(f"illegal lifecycle intent transition {current} -> {target}")
        now = _now_ms()
        if (
            current != target
            or int(state.stage) != int(stage)
            or int(state.reason) != int(reason)
            or state.blocker_intent_id != blocker_intent_id
        ):
            state.since_unix_ms = now
        state.status = status
        state.stage = stage
        state.reason = reason
        state.updated_at_unix_ms = now
        state.next_retry_at_unix_ms = next_retry_at_unix_ms
        if deadline_at_unix_ms:
            state.deadline_at_unix_ms = deadline_at_unix_ms
        state.blocker_intent_id = blocker_intent_id
        if blocker_request is None:
            state.ClearField("blocker_request")
        else:
            state.blocker_request.CopyFrom(blocker_request)
        # A WAITING state MUST carry a blocker, a retry time, or a deadline —
        # the hub's shadow validator requires one of the three, and an intent
        # minted outside a DesiredStateCommand (a compat carrier, or th#2052's
        # adopted dispatch carrier) has none of its own. Guaranteed at this
        # single choke point rather than at each call site.
        if int(status) == pb.LIFECYCLE_INTENT_STATUS_WAITING:
            blocked = bool(state.blocker_intent_id) or bool(
                state.blocker_request.request_id
            )
            if (
                not blocked
                and state.next_retry_at_unix_ms <= 0
                and state.deadline_at_unix_ms <= 0
            ):
                state.deadline_at_unix_ms = now + _WAITING_DEADLINE_FALLBACK_MS
        if progress is None:
            state.ClearField("progress")
        else:
            state.progress.CopyFrom(progress)
        state.error_code = error_code
        state.detail = detail
        state.actual_digest = actual_digest
        state.state_seq = self._touch()
        self._intents.move_to_end(intent_id)
        self._trim_intents()

    def _reported(self, intent_id: str) -> bool:
        state = self._intents.get(intent_id)
        if state is None:
            return False
        if int(state.status) == pb.LIFECYCLE_INTENT_STATUS_WAITING:
            return (
                int(state.stage) != pb.LIFECYCLE_INTENT_STAGE_UNSPECIFIED
                and int(state.reason) != pb.LIFECYCLE_WAIT_REASON_UNSPECIFIED
            )
        return (
            int(state.status) == pb.LIFECYCLE_INTENT_STATUS_RUNNING
            and int(state.stage) != pb.LIFECYCLE_INTENT_STAGE_UNSPECIFIED
        )

    def _fail_unreported_wait(self, intent_id: str, operation: str) -> None:
        detail = f"unreported protocol await: {operation}"
        if intent_id in self._intents:
            self.transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                error_code=pb.LIFECYCLE_ERROR_CODE_PROTOCOL_UNREPORTED_WAIT,
                detail=detail,
            )
        else:
            now = _now_ms()
            identity = hashlib.sha256(operation.encode()).hexdigest()[:24]
            synthetic_id = f"protocol-unreported-{identity}"
            state = pb.IntentState(
                worker_session_id=self.worker_session_id,
                goal_id=(
                    self._last_receipt.goal_id
                    if self._last_receipt is not None
                    else f"protocol-{self.worker_session_id}"
                ),
                intent_id=synthetic_id,
                release_id=self.release_id,
                config_generation=self._target_config_generation,
                status=pb.LIFECYCLE_INTENT_STATUS_FAILED,
                stage=pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                since_unix_ms=now,
                updated_at_unix_ms=now,
                error_code=pb.LIFECYCLE_ERROR_CODE_PROTOCOL_UNREPORTED_WAIT,
                detail=detail,
            )
            state.state_seq = self._touch()
            self._intents[synthetic_id] = state
            self._trim_intents()
        self.protocol_rejected = True

    async def guard_await(
        self,
        intent_id: str,
        awaitable: Awaitable[_T],
        *,
        operation: str,
    ) -> _T:
        """Assert that a long protocol-owned await already has typed state.

        An already-reported wait is awaited DIRECTLY, in the caller's own task,
        never wrapped in ``ensure_future``: a wrapper opens a window where the
        inner task completes, the caller is cancelled before it resumes, and
        asyncio discards the result. When that result is an ACQUISITION the
        resource is then held by nobody for the life of the process. Awaited
        inline, ``Lock.acquire``/``Semaphore.acquire`` handle their own
        cancel-after-grant and the leak is structurally impossible.

        An UNREPORTED wait still needs its own task, because the fail-closed
        timer must fire while the awaitable is still running. There the inner
        task is cancelled explicitly on every exit path — ``asyncio.wait`` does
        not cancel what it was given, so a cancellation arriving mid-window
        would otherwise orphan it.
        """
        if self._reported(intent_id):
            return await awaitable
        task = asyncio.ensure_future(awaitable)
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=self._unreported_wait_timeout_s,
            )
        except BaseException:
            task.cancel()
            raise
        if task in done:
            return task.result()
        self._fail_unreported_wait(intent_id, operation)
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        raise UnreportedIntentWait(f"unreported protocol await: {operation}")

    async def reported_await(
        self,
        intent_id: str,
        awaitable: Awaitable[_T],
        *,
        operation: str,
        status: "pb.LifecycleIntentStatus",
        stage: "pb.LifecycleIntentStage",
        reason: "pb.LifecycleWaitReason" = pb.LIFECYCLE_WAIT_REASON_UNSPECIFIED,
        next_retry_at_unix_ms: int = 0,
        deadline_at_unix_ms: int = 0,
        blocker_intent_id: str = "",
        detail: str = "",
    ) -> _T:
        if intent_id:
            try:
                self.transition(
                    intent_id,
                    status,
                    stage,
                    reason=reason,
                    next_retry_at_unix_ms=next_retry_at_unix_ms,
                    deadline_at_unix_ms=deadline_at_unix_ms,
                    blocker_intent_id=blocker_intent_id,
                    detail=detail,
                )
            except BaseException:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()
                raise
        return await self.guard_await(intent_id, awaitable, operation=operation)

    def _refresh_config_application(self, *, received: bool = False) -> None:
        target = self._target_config_generation
        if target <= 0:
            next_application = pb.ConfigApplication()
        else:
            current = self._config_application
            received_generation = min(
                target,
                max(
                    int(current.received_generation),
                    target if received else 0,
                ),
            )
            parameter_generation = min(target, int(current.parameter_snapshot_generation))
            binding_generation = min(target, int(current.binding_ready_generation))
            boot_generation = min(target, int(current.boot_generation))
            pending_parameters = parameter_generation < target
            pending_bindings = binding_generation < target
            pending_boot = boot_generation < target
            failed = int(current.state) == pb.CONFIG_APPLICATION_STATE_FAILED
            next_application = pb.ConfigApplication(
                release_id=self.release_id,
                target_generation=target,
                received_generation=received_generation,
                parameter_snapshot_generation=parameter_generation,
                binding_ready_generation=binding_generation,
                boot_generation=boot_generation,
                state=(
                    pb.CONFIG_APPLICATION_STATE_FAILED
                    if failed
                    else (
                        pb.CONFIG_APPLICATION_STATE_BOOT_STALE
                        if pending_boot
                        else (
                            pb.CONFIG_APPLICATION_STATE_APPLYING
                            if pending_parameters or pending_bindings
                            else pb.CONFIG_APPLICATION_STATE_CONVERGED
                        )
                    )
                ),
                pending_classes=pb.ConfigClassMask(
                    parameters=pending_parameters,
                    bindings=pending_bindings,
                    boot=pending_boot,
                ),
                error_code=(current.error_code if failed else pb.LIFECYCLE_ERROR_CODE_UNSPECIFIED),
            )
        if next_application.SerializeToString(
            deterministic=True
        ) != self._config_application.SerializeToString(deterministic=True):
            self._config_application = next_application
            self._touch()

    def _advance_config_target(
        self,
        generation: int,
        changed_classes: Optional["pb.ConfigClassMask"],
    ) -> None:
        gen = int(generation)
        old_target = self._target_config_generation
        if gen <= old_target:
            self._refresh_config_application(received=(gen == old_target))
            return

        initial = old_target <= 0
        current = _clone(self._config_application)
        self._target_config_generation = gen
        current.release_id = self.release_id
        current.target_generation = gen
        current.received_generation = gen
        if initial:
            # Only the pod-launch stamp proves which boot-only environment this
            # process received; the first command proves receipt, not boot
            # convergence. Unless there IS no boot-only environment: a process
            # never pod-launched received no WORKER_CONFIG_GENERATION, so
            # "stale boot config" is vacuous and the only advertised remedy
            # (pod replacement) does not exist. Converge the class instead of
            # pending it forever against every target >= 1.
            current.boot_generation = (
                min(gen, self._boot_config_generation)
                if self._boot_config_injected
                else gen
            )
        elif changed_classes is not None:
            if not changed_classes.parameters:
                current.parameter_snapshot_generation = gen
            if not changed_classes.bindings:
                current.binding_ready_generation = gen
            if not changed_classes.boot:
                current.boot_generation = gen
        current.state = pb.CONFIG_APPLICATION_STATE_APPLYING
        current.error_code = pb.LIFECYCLE_ERROR_CODE_UNSPECIFIED
        self._config_application = current
        self._refresh_config_application(received=True)

    def _project_config_intent(self) -> None:
        intent_id = self.intent_id(pb.DESIRED_INTENT_KIND_CONFIG_APPLY)
        if not intent_id:
            return
        application = self._config_application
        if int(application.state) == pb.CONFIG_APPLICATION_STATE_BOOT_STALE:
            self.transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_WAITING,
                pb.LIFECYCLE_INTENT_STAGE_WAIT_REPLACEMENT,
                reason=pb.LIFECYCLE_WAIT_REASON_REPLACEMENT,
            )
        elif int(application.state) == pb.CONFIG_APPLICATION_STATE_CONVERGED:
            self.transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                pb.LIFECYCLE_INTENT_STAGE_READY,
            )
        elif application.pending_classes.parameters:
            self.transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                pb.LIFECYCLE_INTENT_STAGE_CONFIG_MATERIALIZING,
            )
        else:
            self.transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                pb.LIFECYCLE_INTENT_STAGE_CONFIG_BINDINGS_APPLYING,
            )

    def config_snapshot_applied(self, generation: int) -> None:
        """Record only a generation whose atomic snapshot write succeeded."""
        target = self._target_config_generation
        if target <= 0 or int(self._config_application.state) == pb.CONFIG_APPLICATION_STATE_FAILED:
            return
        next_application = _clone(self._config_application)
        next_application.parameter_snapshot_generation = min(
            target,
            max(int(next_application.parameter_snapshot_generation), int(generation)),
        )
        next_application.error_code = pb.LIFECYCLE_ERROR_CODE_UNSPECIFIED
        next_application.state = pb.CONFIG_APPLICATION_STATE_APPLYING
        if next_application.SerializeToString(
            deterministic=True
        ) != self._config_application.SerializeToString(deterministic=True):
            self._config_application = next_application
            self._touch()
        self._refresh_config_application()
        self._project_config_intent()

    def bindings_applied(self, generation: int) -> None:
        """Record exact desired bindings only after reconcile succeeds."""
        target = self._target_config_generation
        if target <= 0 or int(self._config_application.state) == pb.CONFIG_APPLICATION_STATE_FAILED:
            return
        next_application = _clone(self._config_application)
        next_application.binding_ready_generation = min(
            target,
            max(int(next_application.binding_ready_generation), int(generation)),
        )
        self._config_application = next_application
        self._refresh_config_application()
        self._project_config_intent()

    def config_snapshot_failed(self, detail: str) -> None:
        """Withdraw config readiness without advancing an applied generation."""
        if self._target_config_generation > 0:
            next_application = _clone(self._config_application)
            next_application.state = pb.CONFIG_APPLICATION_STATE_FAILED
            next_application.error_code = pb.LIFECYCLE_ERROR_CODE_CONFIG_SNAPSHOT_WRITE_FAILED
            next_application.pending_classes.parameters = True
            if next_application.SerializeToString(
                deterministic=True
            ) != self._config_application.SerializeToString(deterministic=True):
                self._config_application = next_application
                self._touch()
        intent_id = self.intent_id(pb.DESIRED_INTENT_KIND_CONFIG_APPLY)
        if intent_id:
            self.transition(
                intent_id,
                pb.LIFECYCLE_INTENT_STATUS_FAILED,
                pb.LIFECYCLE_INTENT_STAGE_CONFIG_MATERIALIZING,
                error_code=pb.LIFECYCLE_ERROR_CODE_CONFIG_SNAPSHOT_WRITE_FAILED,
                detail=detail,
            )
        self.protocol_rejected = True

    def receive_config_generation(
        self,
        generation: int,
        *,
        changed_classes: Optional["pb.ConfigClassMask"] = None,
    ) -> None:
        """Record receipt separately from any applied config class."""
        gen = int(generation)
        if gen <= 0:
            return
        self._advance_config_target(gen, changed_classes)

    def refresh_projection(
        self,
        executor: Any,
        desired: Optional["pb.DesiredResidency"],
        resolutions: dict[str, tuple[Any, ...]],
    ) -> None:
        """Project exact capabilities and proven parameter snapshot state."""
        parameter_generation = int(
            getattr(executor.runtime_config, "parameter_snapshot_generation", 0)
        )
        if parameter_generation > int(self._config_application.parameter_snapshot_generation):
            self.config_snapshot_applied(parameter_generation)
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
            # Per-function serving tier ("eager" | "compiled"), carried only on
            # READY capabilities.
            tiers_fn = getattr(executor, "serving_tiers", None)
            tiers: dict[str, str] = tiers_fn() if callable(tiers_fn) else {}
            config_state = int(self._config_application.state)
            target_generation = int(
                self._config_application.target_generation or executor.runtime_config.generation
            )
            capabilities = []
            for name in sorted(self.function_names):
                instance = hot.get(name)
                model_refs = sorted(
                    {model.ref for model in (instance.models if instance else ()) if model.ref}
                )
                execution_lanes = sorted(
                    {
                        resolutions.get(ref, ("", "", ""))[2]
                        for ref in model_refs
                        if resolutions.get(ref, ("", "", ""))[2]
                    }
                )
                if config_state == pb.CONFIG_APPLICATION_STATE_FAILED:
                    state = pb.FUNCTION_CAPABILITY_STATE_FAILED
                elif config_state == pb.CONFIG_APPLICATION_STATE_BOOT_STALE:
                    state = pb.FUNCTION_CAPABILITY_STATE_BOOT_STALE
                elif (
                    target_generation > 0 and config_state != pb.CONFIG_APPLICATION_STATE_CONVERGED
                ):
                    state = pb.FUNCTION_CAPABILITY_STATE_APPLYING
                elif name in available:
                    state = pb.FUNCTION_CAPABILITY_STATE_READY
                elif name in executor.unavailable:
                    state = pb.FUNCTION_CAPABILITY_STATE_FAILED
                else:
                    state = pb.FUNCTION_CAPABILITY_STATE_APPLYING
                capabilities.append(
                    pb.FunctionCapability(
                        function_name=name,
                        release_id=self.release_id,
                        config_generation=target_generation,
                        binding_digest=_binding_digest(name, instance),
                        lane=",".join(execution_lanes),
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
                        serving_tier=(
                            tiers.get(name, "")
                            if state == pb.FUNCTION_CAPABILITY_STATE_READY
                            else ""
                        ),
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
        # `ensure_intent` is TOTAL: it returns hub-authored work when there is
        # any, and otherwise mints the `compat-` carrier. There is no third
        # answer, so a drain projection never needs a fabricated goal id.
        intent_id = self.ensure_intent(pb.DESIRED_INTENT_KIND_DRAIN)
        state = self._intents[intent_id]
        transition_status = {
            pb.DRAIN_LIFECYCLE_STATUS_ACCEPTED: pb.LIFECYCLE_INTENT_STATUS_ACCEPTED,
            pb.DRAIN_LIFECYCLE_STATUS_DRAINING: pb.LIFECYCLE_INTENT_STATUS_WAITING,
            pb.DRAIN_LIFECYCLE_STATUS_FINALIZING: pb.LIFECYCLE_INTENT_STATUS_RUNNING,
            pb.DRAIN_LIFECYCLE_STATUS_FLUSHING: pb.LIFECYCLE_INTENT_STATUS_RUNNING,
            pb.DRAIN_LIFECYCLE_STATUS_DRAINED: pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
            pb.DRAIN_LIFECYCLE_STATUS_FAILED: pb.LIFECYCLE_INTENT_STATUS_FAILED,
        }.get(status, pb.LIFECYCLE_INTENT_STATUS_FAILED)
        transition_stage = {
            pb.DRAIN_LIFECYCLE_STATUS_ACCEPTED: pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
            pb.DRAIN_LIFECYCLE_STATUS_DRAINING: pb.LIFECYCLE_INTENT_STAGE_DRAINING,
            pb.DRAIN_LIFECYCLE_STATUS_FINALIZING: pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
            pb.DRAIN_LIFECYCLE_STATUS_FLUSHING: pb.LIFECYCLE_INTENT_STAGE_FLUSHING,
            pb.DRAIN_LIFECYCLE_STATUS_DRAINED: pb.LIFECYCLE_INTENT_STAGE_FLUSHING,
            pb.DRAIN_LIFECYCLE_STATUS_FAILED: pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
        }.get(status, pb.LIFECYCLE_INTENT_STAGE_FINALIZING)
        self.transition(
            intent_id,
            transition_status,
            transition_stage,
            reason=(
                pb.LIFECYCLE_WAIT_REASON_TENANT_WORK
                if int(status) == pb.DRAIN_LIFECYCLE_STATUS_DRAINING
                else pb.LIFECYCLE_WAIT_REASON_UNSPECIFIED
            ),
            deadline_at_unix_ms=deadline_at_unix_ms,
            error_code=error_code,
            detail=detail,
        )
        goal_id = state.goal_id
        since = int(self._drain.since_unix_ms or now)
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


__all__ = ["IntentRegistry", "UnreportedIntentWait"]
