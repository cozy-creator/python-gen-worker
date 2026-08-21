"""gRPC transport: ONE bidi stream, bounded send queue, reconnect with jittered backoff, not_leader redirects; liveness is HTTP/2 keepalive only. Send-queue policy (CONTRACT.md §1): JobResult is NEVER dropped — results persist across reconnects until written to a live stream; under overflow the drop order is JobProgress (oldest first), and everything else blocks the producer. The evidence lane is durable FIFO with deliberately asymmetric coalescing: detail-free RUNNING beats collapse to the latest per (kind, phase), while anything carrying a detail, an error, or a non-RUNNING state is never dropped — exactly mirroring the hub's own upsert conflict key (worker_id, kind, phase, state, self_stalled, payload_digest). In-memory only: a worker restart ends the mint anyway."""

from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import random
import time
from typing import Any, List, Optional, Tuple

import grpc
import grpc.aio

from . import activity as activity_mod
from .config import Settings
from .pb import worker_scheduler_pb2 as pb
from .pb import worker_scheduler_pb2_grpc as pb_grpc
from . import worker_credential

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = pb.PROTOCOL_VERSION_CURRENT

_CONNECT_METHOD = "/{}/Connect".format(
    pb.DESCRIPTOR.services_by_name["WorkerScheduler"].full_name
)

_RESULT, _PROGRESS, _EVENT = "result", "progress", "event"
_EVIDENCE = "evidence"
_EVIDENCE_MSGS = ("activity_update", "boot_phase")

_MAX_REDIRECT_HOPS = 3

_CREDENTIAL_WARN_S = 8 * 60.0
_BACKOFF_RESET_AFTER_S = 60.0
_HELLO_ACK_TIMEOUT_S = 30.0
_PERMANENT_PRECONDITION_MARKERS = (
    "worker_id_mismatch",
    "release_id_mismatch",
    "missing worker identity",
)


async def _terminal_rpc_error(stream: Any) -> BaseException:
    try:
        code = await stream.code()
        details = await stream.details()
        return grpc.aio.AioRpcError(
            code,
            await stream.initial_metadata(),
            await stream.trailing_metadata(),
            details=details,
        )
    except Exception:
        return ConnectionError("RPC finished before Hello could be written")


class FatalTransportError(Exception):
    """Unrecoverable protocol mismatch or persistent registration rejection."""


class HandlerError(Exception):
    """A MESSAGE HANDLER raised — this is not a transport failure."""

    def __init__(self, kind: str, cause: BaseException) -> None:
        super().__init__(f"handler for {kind or 'unknown'} raised: "
                         f"{type(cause).__name__}: {cause}")
        self.kind = kind or "unknown"
        self.cause = cause


def normalize_grpc_addr(addr: str, default_tls: Optional[bool] = None) -> Tuple[str, bool]:
    """Normalize a scheduler address into (host:port, use_tls)."""
    a = (addr or "").strip()
    if not a:
        return "", False
    lower = a.lower()
    for prefix, tls in (("grpcs://", True), ("grpc://", False), ("https://", True), ("http://", False)):
        if lower.startswith(prefix):
            return a[len(prefix):].strip(), tls
    if default_tls is not None:
        return a, default_tls
    return a, a.endswith(":443")


def _backoff_ceiling(base: float, cap: float, attempt: int) -> float:
    safe_base = max(0.0, float(base))
    safe_cap = max(0.0, float(cap))
    ceiling = min(safe_base, safe_cap)
    remaining = max(0, int(attempt))
    while remaining and 0.0 < ceiling < safe_cap:
        ceiling = min(safe_cap, ceiling * 2.0)
        remaining -= 1
    return ceiling


def _confess_dropped_chunk(msg: pb.WorkerMessage) -> None:
    progress = msg.job_progress
    try:
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            f"request_id={progress.request_id} attempt={progress.attempt}: "
            f"streamed output chunk dropped — the send queue is over its bound "
            f"and progress is shed by contract; the terminal JobResult still "
            f"carries the whole output",
            phase="stream_chunk_dropped",
        )
    except Exception:  # noqa: BLE001 — a confession never breaks the queue
        logger.debug("serve_degrade row for a dropped chunk failed", exc_info=True)


def _msg_kind(msg: pb.WorkerMessage) -> str:
    which = msg.WhichOneof("msg")
    if which == "job_result":
        return _RESULT
    if which == "job_progress":
        return _PROGRESS
    if which in _EVIDENCE_MSGS:
        return _EVIDENCE
    return _EVENT


KEEPALIVE_TIMEOUT_S = 10.0

KEEPALIVE_INTERVAL_S = 20.0

_PEER_CLOSE_WAIT_S = 5.0

RECONNECT_EVENT = "worker_reconnect"


class _ReconnectEpisode:

    __slots__ = (
        "dropped_at", "cause", "uptime_s", "loop_silent_s",
        "attempts", "sched_s", "slept_s", "dialed_s", "teardown_s", "outcomes",
    )

    def __init__(
        self, *, dropped_at: float, cause: str, uptime_s: float,
        loop_silent_s: float,
    ) -> None:
        self.dropped_at = dropped_at
        self.cause = cause
        self.uptime_s = uptime_s
        self.loop_silent_s = loop_silent_s
        self.attempts = 0
        self.sched_s = 0.0
        self.slept_s = 0.0
        self.dialed_s = 0.0
        self.teardown_s = 0.0
        self.outcomes: "collections.OrderedDict[str, int]" = collections.OrderedDict()

    def note_outcome(self, token: str) -> None:
        self.outcomes[token] = self.outcomes.get(token, 0) + 1

    def histogram(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.outcomes.items()) or "none"

    def gap_s(self, now: float) -> float:
        return max(0.0, now - self.dropped_at)

    def overshoot_s(self) -> float:
        """Slept wall time minus what was actually scheduled."""
        return max(0.0, self.slept_s - self.sched_s)

    def unaccounted_s(self, now: float) -> float:
        """Gap time in no measured phase at all — neither sleeping, dialing, nor tearing down."""
        return max(
            0.0,
            self.gap_s(now)
            - self.slept_s - self.dialed_s - self.teardown_s,
        )


class SenderQuiesced(Exception):
    """The send loop was asked to stop and the queue is empty."""


EVIDENCE_MAX = 4096

_SHED_KEY = ("shed",)

RESHIP_WINDOW = 256

_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

DEFAULT_QUEUE_MAXSIZE = 1024


class SendQueue:
    """Bounded outbound queue with results-never-dropped semantics."""

    def __init__(
        self, maxsize: int = DEFAULT_QUEUE_MAXSIZE, evidence_max: int = EVIDENCE_MAX,
    ) -> None:
        if int(maxsize) <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = int(maxsize)
        self._evidence_max = evidence_max
        self._pending_evidence: dict[Any, pb.WorkerMessage] = {}
        self._evidence_key: dict[bytes, Any] = {}
        self._evidence_ordinal = 0
        self._shed_total = 0
        self._recent_shipped: collections.deque[Tuple[Any, pb.WorkerMessage]] = (
            collections.deque(maxlen=RESHIP_WINDOW)
        )
        self._items: collections.deque[Tuple[str, pb.WorkerMessage]] = collections.deque()
        self._reconnect: collections.deque[Tuple[str, pb.WorkerMessage]] = (
            collections.deque()
        )
        self._reconnect_seen: dict[Tuple[str, str], bytes] = {}
        self._in_flight: set[bytes] = set()
        self._capacity: dict[Tuple[str, str], pb.WorkerMessage] = {}
        self._capacity_in_flight: dict[Tuple[str, str], int] = {}
        self._state_attempt = 0
        self._state_attempts: dict[Tuple[str, str], int] = {}
        self._cond = asyncio.Condition()
        self._pending_results: dict[Tuple[str, int], pb.WorkerMessage] = {}
        self._quiescing = False

    def __len__(self) -> int:
        return len(self._capacity) + len(self._reconnect) + len(self._items)

    @property
    def pending_result_keys(self) -> List[Tuple[str, int]]:
        return list(self._pending_results.keys())

    def _drop_oldest_progress(self) -> bool:
        for i, (kind, msg) in enumerate(self._items):
            if kind == _PROGRESS:
                del self._items[i]
                _confess_dropped_chunk(msg)
                return True
        return False

    def _bounded_len(self) -> int:
        return sum(
            1 for kind, _msg in self._items if kind not in (_RESULT, _EVIDENCE)
        )

    @staticmethod
    def _is_coalescible_beat(msg: pb.WorkerMessage) -> bool:
        if msg.WhichOneof("msg") != "activity_update":
            return False
        u = msg.activity_update
        return (
            u.state == pb.ActivityState.ACTIVITY_STATE_RUNNING
            and not u.detail
            and not u.error
        )

    def _evidence_slot(self, msg: pb.WorkerMessage) -> Any:
        if self._is_coalescible_beat(msg):
            u = msg.activity_update
            return ("beat", u.kind, u.phase)
        self._evidence_ordinal += 1
        return ("fact", self._evidence_ordinal)

    def _remove_item(self, msg: pb.WorkerMessage) -> None:
        key = self._message_key(msg)
        self._items = collections.deque(
            (kind, queued)
            for kind, queued in self._items
            if self._message_key(queued) != key
        )

    def _put_evidence(self, msg: pb.WorkerMessage) -> None:
        slot = self._evidence_slot(msg)
        prior = self._pending_evidence.get(slot)
        if prior is not None:
            self._evidence_key.pop(self._evidence_bytes(prior), None)
            self._remove_item(prior)
        self._pending_evidence[slot] = msg
        self._evidence_key[self._evidence_bytes(msg)] = slot
        self._items.append((_EVIDENCE, msg))
        self._shed_evidence()

    def _shed_evidence(self) -> None:
        if len(self._pending_evidence) <= self._evidence_max:
            return
        shed = 0
        for pool in ("beat", "fact"):
            for slot in list(self._pending_evidence):
                if len(self._pending_evidence) <= self._evidence_max:
                    break
                if slot == _SHED_KEY or slot[0] != pool:
                    continue
                victim = self._pending_evidence.pop(slot)
                self._evidence_key.pop(self._evidence_bytes(victim), None)
                self._remove_item(victim)
                shed += 1
        if shed:
            self._record_shed(shed)

    def _record_shed(self, shed: int) -> None:
        self._shed_total += shed
        prior = self._pending_evidence.pop(_SHED_KEY, None)
        if prior is not None:
            self._evidence_key.pop(self._evidence_bytes(prior), None)
            self._remove_item(prior)
        msg = pb.WorkerMessage(activity_update=pb.ActivityUpdate(
            kind="outbox_shed",
            phase="overflow",
            state=pb.ActivityState.ACTIVITY_STATE_RUNNING,
            detail=(
                f"outbox shed {self._shed_total} undelivered fact(s) at the "
                f"{self._evidence_max}-entry bound — the hub was unreachable "
                f"long enough that this worker could not keep every "
                f"measurement it produced (pgw#869)"
            ),
            updated_at_unix_ms=int(time.time() * 1000),
        ))
        self._pending_evidence[_SHED_KEY] = msg
        self._evidence_key[self._evidence_bytes(msg)] = _SHED_KEY
        self._items.append((_EVIDENCE, msg))

    @property
    def pending_evidence_count(self) -> int:
        return len(self._pending_evidence)

    @property
    def shed_total(self) -> int:
        return self._shed_total

    @staticmethod
    def _message_key(msg: pb.WorkerMessage) -> Optional[bytes]:
        if msg.WhichOneof("msg") == "job_result":
            return None
        return msg.SerializeToString(deterministic=True)

    @staticmethod
    def _evidence_bytes(msg: pb.WorkerMessage) -> bytes:
        return msg.SerializeToString(deterministic=True)

    @classmethod
    def _reconnect_identity(
        cls, msg: pb.WorkerMessage,
    ) -> Optional[Tuple[str, str]]:
        which = msg.WhichOneof("msg")
        if which == "state_delta":
            return (which, "")
        if which == "fn_unavailable":
            return (which, msg.fn_unavailable.function_name)
        if which == "fn_degraded":
            return (which, msg.fn_degraded.function_name)
        if which == "goal_receipt":
            return (which, msg.goal_receipt.goal_id)
        if which == "lifecycle_snapshot":
            return (which, "")
        if cls._host_capacity_key(msg) is not None:
            return ("host_capacity", msg.model_event.ref)
        return None

    @staticmethod
    def _host_capacity_key(msg: pb.WorkerMessage) -> Optional[bytes]:
        if msg.WhichOneof("msg") != "model_event":
            return None
        event = msg.model_event
        if event.state == pb.MODEL_STATE_HOST_CAPACITY_PROGRESS or (
            event.state == pb.MODEL_STATE_FAILED
            and event.error == "insufficient_host_ram"
            and event.host_ram_capacity_generation > 0
        ):
            return msg.SerializeToString(deterministic=True)
        return None

    @staticmethod
    def _host_capacity_generation(msg: pb.WorkerMessage) -> int:
        return int(msg.model_event.host_ram_capacity_generation)

    def _remove_reconnect_identity(self, identity: Tuple[str, str]) -> None:
        self._items = collections.deque(
            (kind, queued)
            for kind, queued in self._items
            if self._reconnect_identity(queued) != identity
        )
        self._reconnect = collections.deque(
            (kind, queued)
            for kind, queued in self._reconnect
            if self._reconnect_identity(queued) != identity
        )

    def _begin_state_attempt(self, identity: Tuple[str, str]) -> int:
        self._state_attempt += 1
        self._state_attempts[identity] = self._state_attempt
        return self._state_attempt

    def _put_capacity(
        self, msg: pb.WorkerMessage, *, replay_order: bool = False,
    ) -> None:
        identity = self._reconnect_identity(msg)
        if identity is None or identity[0] != "host_capacity":
            raise ValueError("message is not typed host-capacity evidence")
        generation = self._host_capacity_generation(msg)
        current = self._capacity.get(identity)
        current_generation = (
            self._host_capacity_generation(current) if current is not None else -1
        )
        in_flight_generation = self._capacity_in_flight.get(identity, -1)
        if max(current_generation, in_flight_generation) >= generation:
            if (
                replay_order
                and current_generation == generation
                and in_flight_generation < generation
            ):
                self._remove_reconnect_identity(identity)
                self._reconnect_seen.pop(identity, None)
                self._capacity.pop(identity, None)
                self._capacity[identity] = msg
            return
        self._remove_reconnect_identity(identity)
        self._reconnect_seen.pop(identity, None)
        self._capacity.pop(identity, None)
        self._capacity[identity] = msg

    async def put(self, msg: pb.WorkerMessage) -> None:
        kind = _msg_kind(msg)
        async with self._cond:
            if kind == _RESULT:
                r = msg.job_result
                self._pending_results[(r.request_id, r.attempt)] = msg
                self._items.append((kind, msg))
                self._cond.notify_all()
                return
            if kind == _EVIDENCE:
                self._put_evidence(msg)
                self._cond.notify_all()
                return
            if self._host_capacity_key(msg) is not None:
                self._put_capacity(msg)
                self._cond.notify_all()
                return
            identity = self._reconnect_identity(msg)
            attempt: Optional[int] = None
            if identity is not None:
                attempt = self._begin_state_attempt(identity)
                self._remove_reconnect_identity(identity)
                self._reconnect_seen.pop(identity, None)
                self._cond.notify_all()
            while self._maxsize > 0 and self._bounded_len() >= self._maxsize:
                if (
                    identity is not None
                    and self._state_attempts.get(identity) != attempt
                ):
                    return
                if self._drop_oldest_progress():
                    continue
                if kind == _PROGRESS:
                    _confess_dropped_chunk(msg)
                    return
                await self._cond.wait()
            if (
                identity is not None
                and self._state_attempts.get(identity) != attempt
            ):
                return
            self._items.append((kind, msg))
            self._cond.notify_all()

    async def prepend_reconnect(self, messages: List[pb.WorkerMessage]) -> None:
        """Atomically prepend unseen reconnect evidence without backpressure."""
        async with self._cond:
            for msg in messages:
                key = self._message_key(msg)
                if key is None:
                    continue
                identity = self._reconnect_identity(msg)
                if identity is None:
                    raise ValueError(
                        f"{msg.WhichOneof('msg')} is not finite reconnect state"
                    )
                if identity[0] == "host_capacity":
                    self._put_capacity(msg, replay_order=True)
                    continue
                self._begin_state_attempt(identity)
                if (
                    self._reconnect_seen.get(identity) == key
                    or key in self._in_flight
                ):
                    self._items = collections.deque(
                        (kind, queued)
                        for kind, queued in self._items
                        if self._reconnect_identity(queued) != identity
                    )
                    self._reconnect = collections.deque(
                        (kind, queued)
                        for kind, queued in self._reconnect
                        if (
                            self._reconnect_identity(queued) != identity
                            or self._message_key(queued) == key
                        )
                    )
                    continue
                self._remove_reconnect_identity(identity)
                self._reconnect_seen[identity] = key
                self._reconnect.append((_msg_kind(msg), msg))
            self._cond.notify_all()

    async def quiesce(self) -> None:
        """Ask this stream's send loop to end at its next between-writes point."""
        async with self._cond:
            self._quiescing = True
            self._cond.notify_all()

    async def get(self) -> Tuple[str, pb.WorkerMessage]:
        async with self._cond:
            while not self._capacity and not self._reconnect and not self._items:
                if self._quiescing:
                    raise SenderQuiesced
                await self._cond.wait()
            if self._capacity:
                identity, message = next(iter(self._capacity.items()))
                item = (_EVENT, self._capacity.pop(identity))
                self._capacity_in_flight[identity] = (
                    self._host_capacity_generation(message)
                )
            elif self._reconnect:
                item = self._reconnect.popleft()
            else:
                item = self._items.popleft()
            key = self._message_key(item[1])
            if key is not None:
                self._in_flight.add(key)
            self._cond.notify_all()
            return item

    async def should_ship_capacity(self, msg: pb.WorkerMessage) -> bool:
        identity = self._reconnect_identity(msg)
        if identity is None or identity[0] != "host_capacity":
            return True
        generation = self._host_capacity_generation(msg)
        key = self._message_key(msg)
        async with self._cond:
            newer = self._capacity.get(identity)
            if (
                newer is not None
                and self._host_capacity_generation(newer) > generation
            ):
                if self._capacity_in_flight.get(identity) == generation:
                    self._capacity_in_flight.pop(identity, None)
                if key is not None:
                    self._in_flight.discard(key)
                self._cond.notify_all()
                return False
            return True

    async def mark_event_shipped(self, msg: pb.WorkerMessage) -> None:
        key = self._message_key(msg)
        if key is None:
            return
        async with self._cond:
            self._in_flight.discard(key)
            slot = self._evidence_key.pop(key, None)
            if slot is not None:
                shipped = self._pending_evidence.pop(slot, None)
                if shipped is not None:
                    self._recent_shipped.append((slot, shipped))
            identity = self._reconnect_identity(msg)
            if identity is not None and identity[0] == "host_capacity":
                generation = self._host_capacity_generation(msg)
                if self._capacity_in_flight.get(identity) == generation:
                    self._capacity_in_flight.pop(identity, None)
            elif identity is not None:
                self._reconnect_seen[identity] = key
            self._cond.notify_all()

    async def mark_result_shipped(self, msg: pb.WorkerMessage) -> None:
        r = msg.job_result
        async with self._cond:
            self._pending_results.pop((r.request_id, r.attempt), None)
            self._cond.notify_all()

    async def reset_for_reconnect(self) -> None:
        """Drop transient lanes; executor state replays capacity after HelloAck."""
        async with self._cond:
            self._quiescing = False
            self._reconnect.clear()
            self._reconnect_seen.clear()
            self._in_flight.clear()
            self._capacity.clear()
            self._capacity_in_flight.clear()
            self._state_attempts.clear()
            self._items.clear()
            for msg in self._pending_results.values():
                self._items.append((_RESULT, msg))
            reship = list(self._recent_shipped)
            self._recent_shipped.clear()
            revived: dict[Any, pb.WorkerMessage] = {}
            for slot, msg in reship:
                if slot in self._pending_evidence:
                    continue
                revived[slot] = msg
            for slot, msg in revived.items():
                self._evidence_key[self._evidence_bytes(msg)] = slot
            self._pending_evidence = {**revived, **self._pending_evidence}
            for msg in self._pending_evidence.values():
                self._items.append((_EVIDENCE, msg))
            self._cond.notify_all()

    async def wait_empty(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        async with self._cond:
            while (
                self._reconnect
                or self._capacity
                or self._items
                or self._in_flight
                or self._capacity_in_flight
                or self._pending_results
                or self._pending_evidence
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(self._cond.wait(), remaining)
                except asyncio.TimeoutError:
                    return False
            return True

    async def notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()


class Transport:
    """Owns the channel + bidi stream + reconnect loop."""

    def __init__(
        self,
        settings: Settings,
        handlers: Any,
        *,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
    ) -> None:
        self._settings = settings
        self._handlers = handlers
        self.queue = SendQueue(maxsize=queue_maxsize)
        self._backoff_base = backoff_base_s
        self._backoff_cap = backoff_cap_s
        self._stopping = asyncio.Event()
        self._connected = asyncio.Event()
        self._clean_close = False
        self.reconnect_delays: List[float] = []
        self._auth_verdicts: set = set()
        self._expired_rejection_reported: set = set()
        self._last_credential_left: Optional[float] = None
        self._connected_at: Optional[float] = None
        self._last_send_at: Optional[float] = None
        self._episode: Optional["_ReconnectEpisode"] = None
        self._dial_started: float = 0.0
        self._reported_handler_failures: set = set()
        self._cycle = asyncio.Event()

    async def send(self, msg: pb.WorkerMessage) -> None:
        await self.queue.put(msg)

    async def prepend_reconnect(self, messages: List[pb.WorkerMessage]) -> None:
        await self.queue.prepend_reconnect(messages)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def current_worker_jwt(self) -> str:
        """Newest worker credential: hub-rotated token, else the boot token."""
        return (worker_credential.current() or "").strip()

    async def close_after_flush(self, timeout: Optional[float] = None) -> bool:
        """Ship the queue, then stop; ``None`` waits until every result ships."""
        flushed = await self.queue.wait_empty(timeout=timeout)
        self._clean_close = flushed
        if not flushed:
            logger.warning(
                "drain flush deadline expired; closing abruptly for hub reconciliation"
            )
        self._stopping.set()
        await self.queue.notify()
        return flushed

    def stop(self) -> None:
        self._stopping.set()

    def cycle_connection(self) -> None:
        """Drop the current stream (if any) and let the reconnect loop redial with a fresh Hello: the compute child was respawned, so the hub must re-sync desired state against the new process."""
        self._cycle.set()

    def _channel_options(self) -> List[Tuple[str, int]]:
        return [
            ("grpc.keepalive_time_ms", int(KEEPALIVE_INTERVAL_S * 1000)),
            ("grpc.keepalive_timeout_ms", int(KEEPALIVE_TIMEOUT_S * 1000)),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
        ]

    def _make_channel(self, target: str, use_tls: bool) -> grpc.aio.Channel:
        if use_tls:
            return grpc.aio.secure_channel(
                target,
                grpc.ssl_channel_credentials(),
                options=self._channel_options(),
            )
        return grpc.aio.insecure_channel(target, options=self._channel_options())

    def _metadata(self) -> Optional[List[Tuple[str, str]]]:

        token = self.current_worker_jwt
        if not token:
            return None
        self._report_credential_age(token)
        return [("authorization", f"Bearer {token}")]

    def _presented_credential(self) -> str:

        return self.current_worker_jwt

    @staticmethod
    def _credential_claims(token: str) -> dict:
        try:
            from .request_context._helpers import _decode_unverified_jwt_claims

            return _decode_unverified_jwt_claims(token) or {}
        except Exception:  # noqa: BLE001 — a probe never breaks a connect
            return {}

    @classmethod
    def _credential_id(cls, token: str) -> str:
        return str(cls._credential_claims(token).get("jti") or "").strip()

    def _auth_rejection_is_fatal(self, details: str) -> bool:
        token = self._presented_credential()
        exp = float(self._credential_claims(token).get("exp") or 0.0)
        if exp > 0 and exp <= time.time():
            self._report_expired_rejection(token, details, exp)
            return False

        cred = self._credential_id(token)
        verdict = (cred, details)
        repeated = verdict in self._auth_verdicts
        self._auth_verdicts.add(verdict)
        pod = str(getattr(self._settings, "runpod_pod_id", "") or "").strip()
        logger.error(
            "stream rejected UNAUTHENTICATED while presenting a LIVE "
            "credential (jti=%s%s) — this is a verdict about this worker, not "
            "about the hub's availability: %s",
            cred or "<undecodable>", ", REPEATED" if repeated else "", details,
        )
        if pod:
            if not repeated:
                self._report_auth_verdict_deferred(cred, details, pod)
            return False
        return repeated

    def _report_auth_verdict_deferred(
        self, cred: str, details: str, pod: str,
    ) -> None:
        try:
            activity_mod.emit_event(
                "worker_credential",
                f"live credential refused by the hub (jti={cred or '<undecodable>'}"
                f"): {details} — NOT self-terminating: pod {pod} is wedge-reaped "
                f"by the hub, which can revoke the token and close the pod-hour "
                f"ledger. Retrying (pgw#873).",
                phase="auth_verdict_deferred_to_hub",
            )
        except Exception:  # noqa: BLE001 — telemetry never breaks a connect
            logger.debug("auth-verdict event failed", exc_info=True)

    def _report_expired_rejection(
        self, token: str, details: str, exp: float,
    ) -> None:
        cred = self._credential_id(token) or "<undecodable>"
        if cred in self._expired_rejection_reported:
            return
        self._expired_rejection_reported.add(cred)
        ago = time.time() - exp
        logger.error(
            "stream rejected UNAUTHENTICATED with an EXPIRED credential "
            "(jti=%s, expired %.0fs ago): %s — NOT treating this as a verdict "
            "against this worker. The hub admits an expired boot token inside "
            "its grace window and rotates on admission, so this worker keeps "
            "retrying rather than dying with its queue full (pgw#869).",
            cred, ago, details,
        )
        try:
            from . import activity as activity_mod

            activity_mod.emit_event(
                "worker_credential",
                f"expired credential refused by the hub (jti={cred}, "
                f"expired {ago:.0f}s ago): {details} — retrying patiently; "
                f"the queued evidence is held, not dropped",
                phase="expired_rejected_retrying",
            )
        except Exception:  # noqa: BLE001 — telemetry never breaks a connect
            logger.debug("expired-rejection event failed", exc_info=True)

    def _report_credential_age(self, token: str) -> None:
        try:
            from .request_context._helpers import _decode_unverified_jwt_claims

            claims = _decode_unverified_jwt_claims(token) or {}
            exp = float(claims.get("exp") or 0.0)
        except Exception:  # noqa: BLE001 — a probe never breaks a connect
            return
        if exp <= 0:
            return
        left = exp - time.time()
        if left > _CREDENTIAL_WARN_S or left == self._last_credential_left:
            return
        self._last_credential_left = left
        if left > 0:
            logger.warning(
                "worker JWT expires in %.0fs and no rotation has arrived — the "
                "hub pushes one at ~80%% of TTL over this stream, and if it is "
                "missed there is NO other channel to receive one (pgw#848)",
                left)
            detail = f"worker_jwt_expiring in={left:.0f}s exp={int(exp)}"
        else:
            logger.error(
                "worker JWT EXPIRED %.0fs ago and is being presented anyway — "
                "if this connect is rejected this worker cannot recover: the "
                "only refresh channel is the stream it cannot open (pgw#848)",
                -left)
            detail = f"worker_jwt_expired ago={-left:.0f}s exp={int(exp)}"
        try:
            from . import activity as activity_mod

            activity_mod.emit_event(
                "worker_credential", detail,
                phase="expiring" if left > 0 else "expired")
        except Exception:  # noqa: BLE001 — telemetry never breaks a connect
            logger.debug("credential-age event failed", exc_info=True)

    def _report_handler_failure(self, err: HandlerError) -> None:
        logger.error(
            "HANDLER FAILURE while handling %s (this is NOT a connection "
            "failure; the process is alive and will reconnect): %s: %s",
            err.kind, type(err.cause).__name__, err.cause,
            exc_info=err.cause,
        )
        key = (err.kind, type(err.cause).__name__)
        if key in self._reported_handler_failures:
            return
        self._reported_handler_failures.add(key)
        try:
            from .worker_fatal import report_worker_fatal

            delivered = report_worker_fatal(
                self._settings, f"message_handler:{err.kind}", err.cause, exit_code=0,
            )
            logger.info(
                "handler-failure report for %s delivered=%s", err.kind, delivered,
            )
        except Exception:
            logger.warning("handler-failure report raised unexpectedly", exc_info=True)

    def _open_episode(
        self, *, dropped_at: float, cause: str, uptime_s: float,
    ) -> "_ReconnectEpisode":
        last_send = self._last_send_at
        loop_silent_s = (
            max(0.0, dropped_at - last_send) if last_send is not None else 0.0)
        ep = _ReconnectEpisode(
            dropped_at=dropped_at, cause=cause, uptime_s=uptime_s,
            loop_silent_s=loop_silent_s)
        detail = (
            f"stream dropped after {uptime_s:.1f}s connected — cause={cause} "
            f"loop_silent={loop_silent_s:.1f}s; reconnecting with full-jitter "
            f"backoff (ceiling {self._backoff_cap:.3g}s)")
        logger.warning("[reconnect] %s", detail)
        activity_mod.emit_event(RECONNECT_EVENT, detail, phase="dropped")
        return ep

    def _close_episode(
        self, ep: "_ReconnectEpisode", *, at: float, dial_s: float,
    ) -> None:
        ep.attempts += 1
        ep.dialed_s += max(0.0, dial_s)
        gap = ep.gap_s(at)
        unaccounted = ep.unaccounted_s(at)
        overshoot = ep.overshoot_s()
        detail = (
            f"reconnected after {gap:.1f}s — cause={ep.cause} "
            f"attempts={ep.attempts} "
            f"sched={ep.sched_s:.1f}s slept={ep.slept_s:.1f}s "
            f"dialed={ep.dialed_s:.1f}s teardown={ep.teardown_s:.1f}s "
            f"overshoot={overshoot:.1f}s unaccounted={unaccounted:.1f}s "
            f"loop_silent_at_drop={ep.loop_silent_s:.1f}s; "
            f"attempt outcomes: {ep.histogram()}"
        )
        starved = overshoot + unaccounted
        if starved > 0:
            detail += (
                f" — {starved:.1f}s of the gap is NOT the reconnect loop's "
                f"pacing (it was not scheduled: starved event loop, frozen "
                f"process, or a drop the socket never surfaced)")
        logger.warning("[reconnect] %s", detail)
        activity_mod.emit_event(
            RECONNECT_EVENT, detail, phase="reconnected",
            duration_ms=int(gap * 1000))

    def _note_connected(self, connected_at: float) -> None:
        ep, self._episode = self._episode, None
        if ep is not None:
            self._close_episode(
                ep, at=connected_at,
                dial_s=connected_at - self._dial_started)

    def _account_connection(
        self, *, outcome: str, ended_at: float, teardown_s: float,
    ) -> None:
        connected_at = self._connected_at
        if connected_at is not None:
            self._episode = self._open_episode(
                dropped_at=ended_at, cause=outcome,
                uptime_s=max(0.0, ended_at - connected_at))
            self._episode.teardown_s += teardown_s
            return
        ep = self._episode
        if ep is None:
            return
        ep.attempts += 1
        ep.dialed_s += max(0.0, ended_at - self._dial_started)
        ep.teardown_s += teardown_s
        ep.note_outcome(outcome)

    async def run(self) -> None:
        """Reconnect until stopped; fatal protocol/auth failures still exit."""
        attempt = 0
        redirect_addr: Optional[str] = None
        redirect_tls: Optional[bool] = None
        redirect_hops = 0
        while not self._stopping.is_set():
            if redirect_addr is not None:
                target, use_tls = normalize_grpc_addr(redirect_addr, default_tls=redirect_tls)
                redirect_addr = None
            else:
                target, use_tls = normalize_grpc_addr(self._settings.orchestrator_public_addr)
            self._connected_at = None
            self._dial_started = time.monotonic()
            outcome = "stream_ended"
            try:
                await self._connect_once(target, use_tls)
            except grpc.aio.AioRpcError as e:
                code, details = e.code(), str(e.details() or "")
                outcome = f"grpc_{(code.name if code is not None else 'unknown').lower()}"
                if code == grpc.StatusCode.UNAUTHENTICATED:
                    if self._auth_rejection_is_fatal(details):
                        raise FatalTransportError(
                            f"the same live credential was handed the same "
                            f"authentication verdict twice, and this worker "
                            f"runs on no pod, so no hub-side wedge can act for "
                            f"it (pgw#873): {details}"
                        ) from e
                elif code == grpc.StatusCode.FAILED_PRECONDITION:
                    if details.startswith("not_leader:"):
                        if redirect_hops < _MAX_REDIRECT_HOPS:
                            redirect_hops += 1
                            redirect_addr = details.split(":", 1)[1].strip()
                            redirect_tls = use_tls
                            logger.info("not_leader redirect -> %s (hop %d)", redirect_addr, redirect_hops)
                            continue
                        logger.warning("redirect hop limit reached; falling back with backoff")
                    elif "protocol_version_mismatch" in details:
                        raise FatalTransportError(f"protocol version mismatch: {details}") from e
                    elif any(m in details for m in _PERMANENT_PRECONDITION_MARKERS):
                        raise FatalTransportError(f"permanent registration rejection: {details}") from e
                    else:
                        logger.error("protocol violation: %s", details)
                elif code == grpc.StatusCode.UNIMPLEMENTED:
                    raise FatalTransportError(
                        f"hub does not serve this wire-protocol major "
                        f"(UNIMPLEMENTED on {_CONNECT_METHOD}): {details}"
                    ) from e
                else:
                    logger.warning("stream error %s: %s", code, details)
            except FatalTransportError:
                raise
            except HandlerError as e:
                outcome = f"handler_{e.kind}_{type(e.cause).__name__}"
                self._report_handler_failure(e)
            except Exception as e:
                outcome = f"exc_{type(e).__name__}"
                logger.warning("connection to %s failed: %s: %s", target, type(e).__name__, e)
            finally:
                teardown_started = time.monotonic()
                self._connected.clear()
                await self.queue.reset_for_reconnect()
                try:
                    await self._handlers.on_disconnect()
                except Exception:
                    logger.exception("on_disconnect handler failed")
                self._account_connection(
                    outcome=outcome,
                    ended_at=teardown_started,
                    teardown_s=time.monotonic() - teardown_started,
                )

            if self._stopping.is_set():
                return

            redirect_hops = 0

            now = time.monotonic()
            if self._connected_at is not None:
                if now - self._connected_at >= _BACKOFF_RESET_AFTER_S:
                    attempt = 0

            delay = random.uniform(
                0, _backoff_ceiling(self._backoff_base, self._backoff_cap, attempt)
            )
            attempt += 1
            self.reconnect_delays.append(delay)
            logger.info("reconnecting in %.2fs (attempt %d)", delay, attempt)
            slept_from = time.monotonic()
            try:
                await asyncio.wait_for(self._stopping.wait(), delay)
                return
            except asyncio.TimeoutError:
                pass
            finally:
                if self._episode is not None:
                    self._episode.sched_s += delay
                    self._episode.slept_s += max(
                        0.0, time.monotonic() - slept_from)

    async def _connect_once(self, target: str, use_tls: bool) -> None:
        self._cycle.clear()
        channel = self._make_channel(target, use_tls)
        try:
            stub = pb_grpc.WorkerSchedulerStub(channel)

            async def _handshake() -> Any:
                hello = self._handlers.build_hello()
                if inspect.isawaitable(hello):
                    hello = await hello
                stream = stub.Connect(metadata=self._metadata())
                try:
                    await stream.write(pb.WorkerMessage(hello=hello))
                    return stream, await stream.read()
                except asyncio.InvalidStateError:
                    raise await _terminal_rpc_error(stream) from None

            try:
                stream, first = await asyncio.wait_for(
                    _handshake(), _HELLO_ACK_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                raise ConnectionError(
                    f"no HelloAck within {_HELLO_ACK_TIMEOUT_S:.0f}s"
                ) from None
            if first is grpc.aio.EOF:
                raise ConnectionError("stream closed before HelloAck")
            if first.WhichOneof("msg") != "hello_ack":
                raise ConnectionError(f"first scheduler message was {first.WhichOneof('msg')!r}, not hello_ack")
            ack = first.hello_ack
            if ack.protocol_version != PROTOCOL_VERSION:
                raise FatalTransportError(
                    f"HelloAck protocol_version={ack.protocol_version} != {PROTOCOL_VERSION} "
                    "(stale orchestrator build)"
                )
            self._connected_at = time.monotonic()
            self._last_send_at = self._connected_at
            self._note_connected(self._connected_at)
            self._auth_verdicts.clear()
            self._expired_rejection_reported.clear()
            self._connected.set()
            logger.info("connected to %s (HelloAck ok)", target)
            await self._handlers.on_hello_ack(ack)

            send_task = asyncio.create_task(self._send_loop(stream), name="transport-send")
            recv_task = asyncio.create_task(self._recv_loop(stream), name="transport-recv")
            stop_task = asyncio.create_task(self._stopping.wait(), name="transport-stop")
            cycle_task = asyncio.create_task(self._cycle.wait(), name="transport-cycle")
            try:
                done, pending = await asyncio.wait(
                    (send_task, recv_task, stop_task, cycle_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cycle_task in done and stop_task not in done:
                    await self._close_sender(send_task)
                    await self._await_peer_close(stream, recv_task)
                    raise ConnectionError(
                        "stream cycled by supervisor (compute process restart)"
                    )
                if stop_task in done and self._clean_close:
                    await self._close_sender(send_task)
                    await self._await_peer_close(stream, recv_task)
                    return
                for t in pending:
                    t.cancel()
                for t in done:
                    if t is not stop_task:
                        try:
                            t.result()
                        except asyncio.CancelledError:
                            raise ConnectionError(
                                "stream cancelled by the transport layer"
                            ) from None
            finally:
                for t in (send_task, recv_task, stop_task, cycle_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(
                    send_task, recv_task, stop_task, cycle_task, return_exceptions=True
                )
        finally:
            self._connected.clear()
            await channel.close()

    async def _close_sender(self, send_task: "asyncio.Task[None]") -> bool:
        await self.queue.quiesce()
        done, _pending = await asyncio.wait({send_task}, timeout=KEEPALIVE_TIMEOUT_S)
        if send_task in done:
            return True
        logger.warning(
            "send loop did not stop within %.0fs of the close; cancelling it "
            "mid-write, which RSTs the stream — unretired results: %s",
            KEEPALIVE_TIMEOUT_S, self.queue.pending_result_keys,
        )
        send_task.cancel()
        self._clean_close = False
        return False

    async def _await_peer_close(self, stream: Any, recv_task: "asyncio.Task[None]") -> None:
        try:
            await stream.done_writing()
        except Exception:
            return
        await asyncio.wait({recv_task}, timeout=_PEER_CLOSE_WAIT_S)

    async def _send_loop(self, stream: Any) -> None:
        while True:
            try:
                kind, msg = await self.queue.get()
            except SenderQuiesced:
                return
            if not await self.queue.should_ship_capacity(msg):
                continue
            await stream.write(msg)
            self._last_send_at = time.monotonic()
            if kind == _RESULT:
                await self.queue.mark_result_shipped(msg)
            else:
                await self.queue.mark_event_shipped(msg)
            shipped = getattr(self._handlers, "on_message_shipped", None)
            if shipped is not None:
                await shipped(msg)

    async def _recv_loop(self, stream: Any) -> None:
        while True:
            msg = await stream.read()
            if msg is grpc.aio.EOF:
                raise ConnectionError("scheduler closed the stream")
            which = msg.WhichOneof("msg")
            if which is None:
                raise FatalTransportError("scheduler sent an unknown mandatory command")
            if which == "hello_ack":
                try:
                    await self._handlers.on_hello_ack(msg.hello_ack)
                except (FatalTransportError, asyncio.CancelledError):
                    raise
                except Exception as e:
                    raise HandlerError(which, e) from e
                continue
            if which == "token_refresh":
                token = (msg.token_refresh.token or "").strip()
                if token:
                    worker_credential.install(
                        token, float(msg.token_refresh.expires_at_unix or 0))
                    self._auth_verdicts.clear()
                    self._expired_rejection_reported.clear()
                    logger.info(
                        "worker JWT rotated by hub (exp=%d)",
                        msg.token_refresh.expires_at_unix,
                    )
                    refreshed = getattr(self._handlers, "on_token_refresh", None)
                    if refreshed is not None:
                        try:
                            await refreshed(
                                token, int(msg.token_refresh.expires_at_unix)
                            )
                        except Exception:
                            logger.warning(
                                "on_token_refresh handler failed", exc_info=True
                            )
                continue
            try:
                await self._handlers.on_message(msg)
            except (FatalTransportError, asyncio.CancelledError):
                raise
            except Exception as e:
                raise HandlerError(which, e) from e
