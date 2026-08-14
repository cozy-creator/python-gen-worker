"""gRPC transport: ONE bidi stream, bounded send queue, reconnect with jittered
backoff, `not_leader` redirects. Liveness is HTTP/2 keepalive only.

Send-queue policy (CONTRACT.md §1): JobResult is NEVER dropped — results
persist across reconnects until written to a live stream. Under overflow the
drop order is JobProgress (oldest first); everything else blocks the producer.

THE EVIDENCE LANE. Evidence rides its own durable lane on the same principle as
results: enqueue-and-return (never blocks a producer), survives reconnect in FIFO
order, and retires only when a live stream has taken it. Coalescing is
deliberately asymmetric — detail-free RUNNING beats (`Activity.heartbeat` /
`progress_beat`) collapse to the latest per (kind, phase); anything carrying a
`detail`, an `error`, or a non-RUNNING state is evidence and is never dropped.
That split mirrors the hub's own upsert conflict key,
`(worker_id, kind, phase, state, self_stalled, payload_digest)` with
`payload_digest = sha256(error || 0x00 || detail)`
(tensorhub `internal/db/gen/worker_activity_events.sql.go`,
`repository/worker_activity_event_store.go`): exactly what this coalesces is what
the hub would itself have folded into one row via `occurrences + 1`, while
everything it preserves is a distinct row hub-side.

In-memory only, deliberately: a mint is a child of the worker, so a worker
restart ends the mint anyway. Disk persistence is out of scope and must be
argued on its own merits.

THE RECONNECT EPISODE IS A HUB ROW. Every duration in this module is retry
PACING or an RPC deadline, never a kill decision: the backoff is
`uniform(0, min(30 s, 2**attempt))` with `attempt` reset after
`_BACKOFF_RESET_AFTER_S` of connectedness, a dead peer is called by h2 keepalive
within `KEEPALIVE_INTERVAL_S` + `KEEPALIVE_TIMEOUT_S`, and a hung dial is cut at
`_HELLO_ACK_TIMEOUT_S`. So a gap larger than those bounds is time the reconnect
loop was NOT RUNNING. Each episode emits two evidence rows (`dropped`, then
`reconnected`) partitioning the gap into scheduled backoff, slept wall time,
dial+Hello, teardown, and the remainder. No new bound is introduced anywhere:
this measures, it never decides.
"""

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

#: The full gRPC method path this client dials. Derived from the generated
#: descriptor rather than written out, so it can never drift from the proto
#: package that defines the wire-protocol major.
_CONNECT_METHOD = "/{}/Connect".format(
    pb.DESCRIPTOR.services_by_name["WorkerScheduler"].full_name
)

_RESULT, _PROGRESS, _EVENT = "result", "progress", "event"
#: A hub-bound FACT with no live-state replay behind it. Results have
#: `_pending_results`; state has `LifecycleSnapshot`/`StateDelta`/the capacity
#: lane. These two carry measurements and terminals and have neither.
_EVIDENCE = "evidence"
_EVIDENCE_MSGS = ("activity_update", "boot_phase")

_MAX_REDIRECT_HOPS = 3
# WHAT THIS LADDER MAY AND MAY NOT JUDGE. A rejected credential is a verdict
# about US; an ABSENT hub is a verdict about nothing. An expired presented
# credential never reaches this ladder — see `_auth_rejection_is_fatal`.
#
# The rule is EVIDENCE-KEYED, NOT COUNT-KEYED — deliberately no
# "N consecutive over T seconds" threshold:
#
#   * the hub returns `codes.Unavailable` for every transient store error and
#     reserves `Unauthenticated` for genuine credential verdicts
#     (`connect_worker.go:341-349`), so waiting out a window here only delays a
#     verdict that cannot change.
#   * on a POD the worker is not the actuator at all. `worker_wedge.go`
#     terminates the pod on repeated auth rejects from the side that can see
#     it — it revokes the token and closes the pod-hour ledger, neither of
#     which a worker can do for itself. A worker-side exit is a duplicate that
#     can only fire EARLIER than the authority, and a MINTING pod that exits
#     takes its mint with it.
#
# So: a pod NEVER self-terminates on auth. A pod-less worker (`worker_wedge.go`
# returns immediately on an empty `RunpodPodID`, so it has no hub-side actuator
# at all) escalates on EVIDENCE — the identical verdict on the identical
# credential, seen twice — decidable from the `jti` and the rejection details,
# with nothing counted and no window waited.

#: How long before its own expiry a worker starts SAYING SO. Wider
#: than the hub's ~80%-of-TTL rotation point (24 min of a 30 min TTL, i.e.
#: 6 min of remaining life), so a missed rotation is visible while the stream
#: is still usable rather than after it is not.
_CREDENTIAL_WARN_S = 8 * 60.0
_BACKOFF_RESET_AFTER_S = 60.0
_HELLO_ACK_TIMEOUT_S = 30.0
# FAILED_PRECONDITION details that can never heal by retrying: identity is
# wrong for this deployment, so retrying cannot repair it.
_PERMANENT_PRECONDITION_MARKERS = (
    "worker_id_mismatch",
    "release_id_mismatch",
    "missing worker identity",
)


async def _terminal_rpc_error(stream: Any) -> BaseException:
    """Recover the REAL status of a call grpc.aio reported as InvalidStateError.

    A finished call still knows its code/details, so rebuild the AioRpcError the
    caller would have seen had the write raced differently. Keeps the reconnect
    classifier (auth ladder, not_leader, permanent refusals) working on a status
    instead of a nameless `InvalidStateError`.
    """
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
    """A MESSAGE HANDLER raised — this is not a transport failure.

    `_recv_loop` awaits the handlers inline, so without this type a handler bug
    reaches `run()`'s catch-all and is logged as a dropped connection. Carries
    which message was being handled so the report names it.
    """

    def __init__(self, kind: str, cause: BaseException) -> None:
        super().__init__(f"handler for {kind or 'unknown'} raised: "
                         f"{type(cause).__name__}: {cause}")
        self.kind = kind or "unknown"
        self.cause = cause


def normalize_grpc_addr(addr: str, default_tls: Optional[bool] = None) -> Tuple[str, bool]:
    """Normalize a scheduler address into (host:port, use_tls).

    ``default_tls`` is the TLS mode for schemeless addresses (e.g. a
    ``not_leader`` redirect target inherits it from the connection that issued
    the redirect); when None, fall back to the bare ``:443`` heuristic.
    """
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
    """Return ``min(cap, base * 2**attempt)`` without building ``2**attempt``.

    A cap does not make the original expression safe: Python evaluates the
    float multiplication before ``min``, and converting a sufficiently large
    integer exponent to float raises ``OverflowError``. Double only until the
    ceiling saturates instead. The loop is bounded by the number of doublings
    between ``base`` and ``cap`` (five for the production 1 s -> 30 s policy),
    never by ``attempt`` once the cap has been reached.
    """
    safe_base = max(0.0, float(base))
    safe_cap = max(0.0, float(cap))
    ceiling = min(safe_base, safe_cap)
    remaining = max(0, int(attempt))
    while remaining and 0.0 < ceiling < safe_cap:
        ceiling = min(safe_cap, ceiling * 2.0)
        remaining -= 1
    return ceiling


def _msg_kind(msg: pb.WorkerMessage) -> str:
    which = msg.WhichOneof("msg")
    if which == "job_result":
        return _RESULT
    if which == "job_progress":
        return _PROGRESS
    if which in _EVIDENCE_MSGS:
        return _EVIDENCE
    return _EVENT


#: How long the channel itself waits for a keepalive answer before calling the
#: peer dead. Anything that needs "we tried, and the peer is not taking bytes"
#: derives its bound from this rather than inventing one.
KEEPALIVE_TIMEOUT_S = 10.0

#: How often the channel pings an idle stream. With KEEPALIVE_TIMEOUT_S this is
#: the worker's detection budget for a peer that has stopped reading:
#: interval + timeout = 30 s, and everything downstream (the reconnect
#: accounting at `RECONNECT_EVENT`, the send-loop wait) derives from that sum
#: rather than restating either half.
KEEPALIVE_INTERVAL_S = 20.0

#: How long to wait for the peer to end a half-closed call before giving up on
#: a graceful close. Unchanged value, named so both close paths share it.
_PEER_CLOSE_WAIT_S = 5.0

#: The typed disconnect/reconnect record. The loop's own constants bound the
#: gap it can produce: the delay is `uniform(0, min(30s, 2**attempt))` (full
#: jitter, ceiling :attr:`Transport._backoff_cap` = 30 s), `attempt` resets to 0
#: after :data:`_BACKOFF_RESET_AFTER_S` of connectedness, a dead peer is called
#: by h2 keepalive within :data:`KEEPALIVE_INTERVAL_S` +
#: :data:`KEEPALIVE_TIMEOUT_S`, and a dial that hangs is cut at
#: :data:`_HELLO_ACK_TIMEOUT_S`. Anything beyond that is time the reconnect loop
#: WAS NOT RUNNING — a starved event loop, a frozen process, a drop the socket
#: never surfaced.
#:
#: So an episode ACCOUNTS for itself: the gap from drop to reconnect is
#: partitioned into what the loop can prove it spent (backoff sleep, dial+Hello
#: attempts, teardown) and the remainder. `unaccounted_s` is the diagnostic —
#: ~0 on a healthy reconnect — and it introduces no bound of any kind.
RECONNECT_EVENT = "worker_reconnect"


class _ReconnectEpisode:
    """One disconnect -> reconnect episode, measured.

    Coalescing is by COUNT, never by dropping: an hour-long hub outage is two
    hub rows (`dropped`, then `reconnected` carrying the attempt histogram),
    not one row per attempt — the queue is bounded and a replay storm would
    cost the very evidence it is made of.
    """

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
        #: How stale this process's last PROVEN scheduling instant already was
        #: when the drop was detected. The heartbeat forces a StateDelta every
        #: beat in every state, so on a live worker this is ~one beat; a large
        #: value says the drop was detected late because the process was not
        #: running, not because the loop was slow to react.
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
        """Slept wall time minus what was actually scheduled.

        An asyncio sleep on a healthy loop overshoots by milliseconds. Minutes
        of overshoot is a starved loop, and it is the one term that cannot be
        confused with a slow network or a slow hub.
        """
        return max(0.0, self.slept_s - self.sched_s)

    def unaccounted_s(self, now: float) -> float:
        """Gap time in no measured phase at all — neither sleeping, dialing,
        nor tearing down. Should be ~0; anything else is a phase this ledger
        cannot see, and is worth more than a guess about the backoff."""
        return max(
            0.0,
            self.gap_s(now)
            - self.slept_s - self.dialed_s - self.teardown_s,
        )


class SenderQuiesced(Exception):
    """The send loop was asked to stop and the queue is empty.

    Ending the sender BETWEEN writes is not cosmetic: in grpc.aio a cancelled
    ``write()`` cancels the whole RPC (``_call._write`` calls ``self.cancel()``
    on CancelledError), which RSTs the stream and discards every byte buffered
    but not yet flushed — including a JobResult that ``mark_result_shipped``
    has already retired from the durable queue.
    """


#: How many undelivered evidence facts the outbox holds before it shortens
#: itself. Derived: a whole-graph sdxl mint emits O(10^2) phase/measurement
#: facts per hour, so this is hours of a real mint's evidence. Growth is bounded
#: by SHEDDING, and a shed is itself a reported fact — a loss is never silent.
EVIDENCE_MAX = 4096

#: The shed report's own slot. Exempt from shedding: the one fact that must
#: never be lost is the fact that facts were lost.
_SHED_KEY = ("shed",)

#: How many just-shipped facts are re-offered after a DIRTY reconnect.
#:
#: `stream.write()` returning means BUFFERED, not delivered (cancelling a write
#: RSTs the call and throws away everything buffered behind it — see
#: `_close_sender`). So retiring a fact the instant its write returns loses the
#: tail of facts produced around a stream death. A trailing ring makes that
#: window recoverable at bounded cost, and re-offering is safe by construction:
#: the hub folds an identical redelivery into `occurrences + 1` on one row (see
#: the module docstring), so the choice is between a duplicate and a hole.
RESHIP_WINDOW = 256

#: Largest single gRPC message, either direction. See `_channel_options`.
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

#: Outbound queue depth. Bounds the pod's send-side memory when the hub stops
#: reading: progress is dropped, events block their producer, results are never
#: shed. One number, stated once, so `Transport` and `worker` cannot drift.
DEFAULT_QUEUE_MAXSIZE = 1024


class SendQueue:
    """Bounded outbound queue with results-never-dropped semantics."""

    def __init__(
        self, maxsize: int = DEFAULT_QUEUE_MAXSIZE, evidence_max: int = EVIDENCE_MAX,
    ) -> None:
        # A non-positive `maxsize` would delete the bound outright — an
        # unbounded outbound queue in a pod whose producer (progress, events)
        # never blocks, i.e. the pod OOMs instead of shedding. Absence is the
        # stated default; a stated non-positive is refused.
        if int(maxsize) <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = int(maxsize)
        # The durable evidence lane. Ordered (dict preserves insertion
        # order), so replay is FIFO. A coalescible beat REPLACES its slot in
        # place, which keeps a phase's beat where the phase began rather than
        # letting liveness chatter reorder the record.
        self._evidence_max = evidence_max
        self._pending_evidence: dict[Any, pb.WorkerMessage] = {}
        # serialized bytes -> evidence key, so retiring a shipped message is a
        # dict pop rather than a scan, and a superseded copy cannot retire the
        # newer one that replaced it.
        self._evidence_key: dict[bytes, Any] = {}
        self._evidence_ordinal = 0
        self._shed_total = 0
        # Facts whose write returned on the stream that has since died. See
        # RESHIP_WINDOW: a returned write is a buffered write.
        self._recent_shipped: collections.deque[Tuple[Any, pb.WorkerMessage]] = (
            collections.deque(maxlen=RESHIP_WINDOW)
        )
        self._items: collections.deque[Tuple[str, pb.WorkerMessage]] = collections.deque()
        # Reconnect evidence is inserted atomically ahead of preserved results.
        # It is a small state replay, not ordinary producer traffic, so it is
        # exempt from the progress/event bound and cannot deadlock HelloAck
        # before the send loop starts.
        self._reconnect: collections.deque[Tuple[str, pb.WorkerMessage]] = (
            collections.deque()
        )
        # Per-stream fences. ``_reconnect_seen`` holds only the latest finite
        # HelloAck baseline identity (state or function status) prepended or
        # written in this epoch. Host-capacity evidence has its own generation-
        # fenced delivery lane below. ``_in_flight`` closes the get/write race
        # while the single sender owns one message.
        self._reconnect_seen: dict[Tuple[str, str], bytes] = {}
        self._in_flight: set[bytes] = set()
        # Typed host-capacity evidence is finite state, not ordinary traffic:
        # retain only the newest undelivered generation per ref. It bypasses
        # the ordinary bound so an older blocked put can never wake after a
        # newer satisfying generation and reinsert stale FAILED evidence.
        self._capacity: dict[Tuple[str, str], pb.WorkerMessage] = {}
        self._capacity_in_flight: dict[Tuple[str, str], int] = {}
        # Finite full-replace identities only (state delta and per-function
        # status). A newer enqueue/prepend attempt fences an older producer
        # that is still blocked on the ordinary lane. Capacity refs do not
        # enter this map; their generation-fenced lane above is authoritative.
        self._state_attempt = 0
        self._state_attempts: dict[Tuple[str, str], int] = {}
        self._cond = asyncio.Condition()
        # (request_id, attempt) -> JobResult WorkerMessage, until written to a
        # live stream. Survives reconnects; drives Hello.in_flight.
        self._pending_results: dict[Tuple[str, int], pb.WorkerMessage] = {}
        # Set when this stream's sender must end. It ends only once
        # nothing is left to write, so quiescing never drops a queued message.
        self._quiescing = False

    def __len__(self) -> int:
        return len(self._capacity) + len(self._reconnect) + len(self._items)

    @property
    def pending_result_keys(self) -> List[Tuple[str, int]]:
        return list(self._pending_results.keys())

    def _drop_oldest_progress(self) -> bool:
        for i, (kind, _m) in enumerate(self._items):
            if kind == _PROGRESS:
                del self._items[i]
                return True
        return False

    def _bounded_len(self) -> int:
        # Durable JobResults are explicitly exempt from the queue bound. They
        # must not consume event/progress capacity merely because they share
        # the same deque (especially after reconnect requeues several results).
        # Evidence is exempt for the same reason and one stronger — it has its
        # own bound (`_evidence_max`) with its own shed policy, and a producer
        # that just MEASURED something must never block on a dead connection to
        # report it.
        return sum(
            1 for kind, _msg in self._items if kind not in (_RESULT, _EVIDENCE)
        )

    # ---- the evidence lane -------------------------------------------------

    @staticmethod
    def _is_coalescible_beat(msg: pb.WorkerMessage) -> bool:
        """A pure liveness re-report: RUNNING, carrying no payload.

        `Activity.heartbeat` and `Activity.progress_beat` re-state the current
        phase with a fresh seq/counter and nothing else. Only the latest can
        matter — and the hub agrees by construction: its conflict key covers
        (kind, phase, state, payload_digest), and `seq`/`step`/`counter_*` are
        taken from EXCLUDED on conflict, so N beats of one phase were always
        going to become ONE row holding the last one's numbers.
        """
        if msg.WhichOneof("msg") != "activity_update":
            return False
        u = msg.activity_update
        return (
            u.state == pb.ActivityState.ACTIVITY_STATE_RUNNING
            and not u.detail
            and not u.error
        )

    def _evidence_slot(self, msg: pb.WorkerMessage) -> Any:
        """The coalescing slot: shared for beats, unique for everything else."""
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
        """Record a fact durably and queue it. NEVER blocks, never raises."""
        slot = self._evidence_slot(msg)
        prior = self._pending_evidence.get(slot)
        if prior is not None:
            # A superseded beat leaves no copy behind to be written after the
            # newer one; its slot position is retained by the dict.
            self._evidence_key.pop(self._evidence_bytes(prior), None)
            self._remove_item(prior)
        self._pending_evidence[slot] = msg
        self._evidence_key[self._evidence_bytes(msg)] = slot
        self._items.append((_EVIDENCE, msg))
        self._shed_evidence()

    def _shed_evidence(self) -> None:
        """Bound the lane. Coalescible first, evidence last."""
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
        """Shedding evidence must itself emit a fact.

        Built here rather than through `activity.emit_event` so it cannot
        recurse into the queue it is reporting on. It occupies the reserved
        slot, so the count is always the running total and the hub's row is the
        final one.
        """
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
        """The wire identity of a message, or None for a JobResult.

        **The None is load-bearing, not a gap.** A result is not a fact: it is
        durable under `(request_id, attempt)` in `_pending_results`, it is
        exempt from the queue bound, and it must never be folded with anything.
        So it deliberately has no content key, and **every keyed structure here
        is therefore a structure results do not belong in** — `_in_flight`, the
        reconnect fences, and the evidence map all key on this and all exclude
        results by construction. A consumer that treats this map as covering
        every message has an unhandled case, not a typing nuisance.
        """
        if msg.WhichOneof("msg") == "job_result":
            return None
        return msg.SerializeToString(deterministic=True)

    @staticmethod
    def _evidence_bytes(msg: pb.WorkerMessage) -> bytes:
        """`_message_key` for a message already classified as `_EVIDENCE`.

        Total by precondition: `_msg_kind` admits exactly `activity_update` and
        `boot_phase` to this lane, and the only keyless message is `job_result`.
        Both alternatives are wrong — skipping a keyless message would make the
        map silently lossy, and giving results their own evidence key would put
        them in two durable lanes at once (`_pending_results` and
        `_pending_evidence`), double-shipping a result on reconnect and
        coalescing something the queue contract says is never coalesced.
        """
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
                # HelloAck's active-failure/undelivered-progress snapshot is
                # authoritative even when the exact entries were already
                # pending in the opposite insertion order.
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
                self._items.append((kind, msg))       # results exempt from the bound
                self._cond.notify_all()
                return
            if kind == _EVIDENCE:
                self._put_evidence(msg)               # never blocks
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
                    return                            # drop this progress chunk
                await self._cond.wait()               # backpressure: block the producer
            if (
                identity is not None
                and self._state_attempts.get(identity) != attempt
            ):
                return
            self._items.append((kind, msg))
            self._cond.notify_all()

    async def prepend_reconnect(self, messages: List[pb.WorkerMessage]) -> None:
        """Atomically prepend unseen reconnect evidence without backpressure.

        The logical-identity map lasts for one connection epoch. Thus a midstream
        duplicate HelloAck is idempotent, while reset_for_reconnect clears the
        fence so the next stream can replay the same process generation.
        Older copies of the same logical identity are replaced, not copied, so
        stale state/capacity evidence cannot remain behind a durable result.
        """
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
                    # Drop stale ordinary/reconnect copies of this logical
                    # state while retaining the exact current prepend copy.
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
                # Dict insertion order is causal: the live executor outbox
                # inserts global commit order, while reconnect replay inserts
                # every active FAILED before undelivered PROGRESS.
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
            # A fact retires from the durable lane only once a LIVE stream has
            # taken it. `_evidence_key` holds only the currently stored copy, so
            # a superseded beat can never retire the newer one.
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
            self._cond.notify_all()  # wake wait_empty (drain flush)

    async def reset_for_reconnect(self) -> None:
        """Drop transient lanes; executor state replays capacity after HelloAck.

        DURABLE lanes are requeued, not dropped: results, and evidence in the
        order it was produced. Anything a live stream took but never confirmed
        (`mark_event_shipped` did not run because the write failed) is still in
        `_pending_evidence` and comes back with it.
        """
        async with self._cond:
            self._quiescing = False   # the quiesce belonged to the dead stream
            self._reconnect.clear()
            self._reconnect_seen.clear()
            self._in_flight.clear()
            self._capacity.clear()
            self._capacity_in_flight.clear()
            self._state_attempts.clear()
            self._items.clear()
            for msg in self._pending_results.values():
                self._items.append((_RESULT, msg))
            # Re-offer the trailing window of facts whose write RETURNED on the
            # stream that just died — buffered is not delivered (RESHIP_WINDOW).
            # Oldest first, and never over a slot a newer fact has since taken.
            reship = list(self._recent_shipped)
            self._recent_shipped.clear()
            revived: dict[Any, pb.WorkerMessage] = {}
            for slot, msg in reship:
                if slot in self._pending_evidence:
                    continue          # superseded by a newer fact; that one wins
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
                or self._pending_evidence   # a drain flushes facts too
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
    """Owns the channel + bidi stream + reconnect loop.

    handlers must provide:
      build_hello() -> pb.Hello | Awaitable[pb.Hello]  (fresh full snapshot;
        awaitable form serves the split parent, which fetches the Hello from
        its compute child)
      on_hello_ack(ack: pb.HelloAck) -> Awaitable   (also mid-stream re-sends)
      on_message(msg: pb.SchedulerMessage) -> Awaitable  (MUST NOT block)
      on_message_shipped(msg: pb.WorkerMessage) -> Awaitable (optional)
      on_token_refresh(token, expires_at_unix) -> Awaitable (optional)
      on_disconnect() -> Awaitable
    """

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
        self.reconnect_delays: List[float] = []  # observability + tests
        #: The (jti, details) verdicts this worker has already been handed. A
        #: pod-less worker escalates when one REPEATS — same credential, same
        #: answer — because that is the evidence that the verdict cannot change.
        #: A rotation retires the old key with the old credential.
        self._auth_verdicts: set = set()
        #: Credentials whose expired-rejection has already been confessed.
        self._expired_rejection_reported: set = set()
        #: The last remaining-lifetime this worker reported, so a reconnect
        #: storm does not emit one event per attempt.
        self._last_credential_left: Optional[float] = None
        self._connected_at: Optional[float] = None  # set on each HelloAck
        #: Last proven scheduling instant (see `_send_loop`).
        self._last_send_at: Optional[float] = None
        #: The open disconnect->reconnect episode, if any.
        self._episode: Optional["_ReconnectEpisode"] = None
        self._dial_started: float = 0.0
        # (message kind, exception class) already dialed to the hub.
        self._reported_handler_failures: set = set()
        # A supervisor-requested stream cycle (compute child respawn needs a
        # fresh Hello). Cleared at the start of each connection, so a request
        # set BEFORE a connection is satisfied by that connection's own fresh
        # Hello.
        self._cycle = asyncio.Event()

    # ---- send API --------------------------------------------------------

    async def send(self, msg: pb.WorkerMessage) -> None:
        await self.queue.put(msg)

    async def prepend_reconnect(self, messages: List[pb.WorkerMessage]) -> None:
        await self.queue.prepend_reconnect(messages)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def current_worker_jwt(self) -> str:
        """Newest worker credential: hub-rotated token, else the boot token.

        ONE source, and it is the process-wide one: never a stream-local cache,
        or a failed publish leaves this stream authenticating with a token no
        other dial in the process can see.
        """
        return (worker_credential.current() or "").strip()

    # ---- drain / shutdown --------------------------------------------------

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
        """Drop the current stream (if any) and let the reconnect loop redial
        with a fresh Hello: the compute child was respawned, so the hub must
        re-sync desired state against the new process."""
        self._cycle.set()

    # ---- connection loop ---------------------------------------------------

    def _channel_options(self) -> List[Tuple[str, int]]:
        return [
            ("grpc.keepalive_time_ms", int(KEEPALIVE_INTERVAL_S * 1000)),
            ("grpc.keepalive_timeout_ms", int(KEEPALIVE_TIMEOUT_S * 1000)),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            # The frame cap on ONE WorkerMessage, both directions. Nothing else
            # bounds a single message: the queue caps the COUNT in flight
            # (`DEFAULT_QUEUE_MAXSIZE`), never the size of one. Generous
            # relative to real traffic (control messages and metadata; artifact
            # bytes travel over HTTP to the CAS, never on this stream) so it
            # never binds on legitimate traffic and only catches a runaway.
            ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
        ]

    def _make_channel(self, target: str, use_tls: bool) -> grpc.aio.Channel:
        if use_tls:
            # System trust roots; no custom-CA-bundle knob by design.
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

    # ---- which auth rejections are a verdict about US ----------------------

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
        """`jti` — the same key the hub's admission and wedge streaks use."""
        return str(cls._credential_claims(token).get("jti") or "").strip()

    def _auth_rejection_is_fatal(self, details: str) -> bool:
        """Whether THIS UNAUTHENTICATED is evidence against this worker.

        **An expired presented credential is never such evidence**, and the hub
        agrees — measured in tensorhub, not inferred:

        * `AdmitExpiredBootToken` (`repository/expected_worker_store.go:311`)
          admits an expired-but-valid token while
          `now < ew.IssuedAtUnix + BootGraceWindow`, keyed on the `jti` the hub
          itself last minted;
        * `DefaultWorkerJWTTTL = 30 min` vs `DefaultWorkerJWTBootGrace = 4 h`
          (`config/config.go:619,675`), and `RotateExpectedWorkerJTI` moves
          `IssuedAtUnix` forward on every rotation — so a worker is admissible
          for ~3.5 h PAST its expiry, counted from its last rotation;
        * on admission the hub pushes a `TokenRefresh` immediately
          (`connect_worker.go:392`, "credential rotated on connect"), which
          fully heals the worker.

        So retrying with an expired token is not a doomed loop — it is the
        documented recovery path. The billing risk is owned by the party that
        can actually see the pod: `grpc/worker_wedge.go` terminates a pod after
        `wedgeAuthRejectThreshold` consecutive auth rejects through the tracked
        provider path.

        A LIVE (or absent) credential refused by an answering hub IS about us —
        revoked, superseded, misconfigured. **What happens then depends on
        whether anyone ELSE can act.** On a pod, `worker_wedge.go` is the
        actuator and this worker is not, so the worker keeps retrying and lets
        the authority decide (a minting pod that exits early takes its mint with
        it). A pod-less worker has no such authority — `noteWedgeStreak` returns
        immediately on an empty `RunpodPodID` — so it is the only actuator there
        is, and it escalates on EVIDENCE rather than on a count: the same `jti`
        handed the same verdict twice cannot be waited out. No threshold, no
        window; a rotation retires the evidence with the credential.
        """
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
            # The hub owns the actuator for a pod. Say so once per verdict so
            # the silence is attributable off-pod.
            if not repeated:
                self._report_auth_verdict_deferred(cred, details, pod)
            return False
        return repeated

    def _report_auth_verdict_deferred(
        self, cred: str, details: str, pod: str,
    ) -> None:
        """A refusal this worker deliberately does NOT act on is still a fact
        the hub should hold."""
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
        """Patient is not the same as silent. Once per credential."""
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
        """Say something BEFORE the credential dies, not after the pod does.

        The refresh arrives ONLY as a ``token_refresh`` down the stream (see the
        handler below), so a credential deliverable solely over the connection
        it authenticates cannot be delivered once that connection stops
        authenticating. ``DefaultWorkerJWTTTL`` is 30 minutes from POD CREATE,
        so this fires on any pod that lives past half an hour.

        DIAGNOSIS ONLY, deliberately: it changes no behaviour and refuses
        nothing. An expired token is not always fatal — the hub has a
        boot-grace admission for exactly that case — so shortcutting the
        reconnect here would break a path that legitimately heals.
        """
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
        """Log a handler failure as ITSELF and dial it to the hub.

        Reuses the `worker_fatal` carrier, so this lands as a durable
        `pod_events` row with no proto change. Deduped per (message kind,
        exception class): the reconnect loop would otherwise re-dial the same
        fault every cycle. Liveness policy is unchanged — this only unmasks the
        fault.
        """
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

    # ---- the disconnect/reconnect episode ledger ---------------------------

    def _open_episode(
        self, *, dropped_at: float, cause: str, uptime_s: float,
    ) -> "_ReconnectEpisode":
        """A stream that HAD reached HelloAck ended: the episode starts here.

        Emitted immediately rather than only at reconnect: the evidence lane
        survives the disconnect, so the drop is a durable fact even for a pod
        that dies before it ever gets back.
        """
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
        """Reconnected. Report the gap AND the partition that explains it."""
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
        # `overshoot` + `unaccounted` is the part of the gap the reconnect loop
        # cannot account for, i.e. time it was not running at all. Named on the
        # row so a long gap is read as "the process was not scheduling" rather
        # than re-litigated as "the backoff must be wrong".
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
        """HelloAck landed. Close any open episode HERE — not when this new
        stream later ends: a reconnect is news the instant it happens, and a
        row that waits for the next drop to be emitted arrives after the pod it
        describes may already be dead."""
        ep, self._episode = self._episode, None
        if ep is not None:
            self._close_episode(
                ep, at=connected_at,
                dial_s=connected_at - self._dial_started)

    def _account_connection(
        self, *, outcome: str, ended_at: float, teardown_s: float,
    ) -> None:
        """Fold one ended connection attempt into the ledger.

        ``ended_at`` is when the connection ENDED, i.e. before this iteration's
        teardown — not "now". The distinction is the accounting: an episode's
        clock starts at the drop, so charging it a teardown that ran BEFORE its
        own start would leave that time in `unaccounted` forever, and reading a
        constant offset there as loop starvation is exactly the wrong answer.

        The episode stays None while this worker has never been connected.
        """
        connected_at = self._connected_at
        if connected_at is not None:
            # This attempt reached HelloAck (and `_note_connected` already
            # closed the previous episode), so the stream that was UP has now
            # ended: a new episode opens here, AT the end of the stream. Its
            # teardown is the first thing inside it.
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
                # Schemeless redirect targets inherit the TLS mode of the
                # connection that issued the redirect — never downgrade.
                target, use_tls = normalize_grpc_addr(redirect_addr, default_tls=redirect_tls)
                redirect_addr = None
            else:
                target, use_tls = normalize_grpc_addr(self._settings.orchestrator_public_addr)
            self._connected_at = None
            self._dial_started = time.monotonic()
            # The classified outcome of THIS attempt. Default names the clean
            # case — `_connect_once` returning is a stream that ended without
            # raising.
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
                            continue  # immediate, no backoff
                        logger.warning("redirect hop limit reached; falling back with backoff")
                    elif "protocol_version_mismatch" in details:
                        raise FatalTransportError(f"protocol version mismatch: {details}") from e
                    elif any(m in details for m in _PERMANENT_PRECONDITION_MARKERS):
                        raise FatalTransportError(f"permanent registration rejection: {details}") from e
                    else:
                        logger.error("protocol violation: %s", details)
                elif code == grpc.StatusCode.UNIMPLEMENTED:
                    # The wire-protocol MAJOR is the proto package, so it is in
                    # the service path (/cozy.scheduler.v1.WorkerScheduler/
                    # Connect). A hub that does not serve our major never
                    # registers that path and gRPC answers UNIMPLEMENTED before
                    # any hub code runs.
                    #
                    # FATAL and unretryable, and that is load-bearing HUB-side:
                    # the worker's exit is what makes the pod die before Hello,
                    # leaving `everConnected` false, which is how the hub's death
                    # taxonomy marks the release `boot_crashing` and fails its
                    # queued requests. Retrying would reconnect-loop forever
                    # against a hub that can never answer, producing NO durable
                    # mark and no operator signal.
                    raise FatalTransportError(
                        f"hub does not serve this wire-protocol major "
                        f"(UNIMPLEMENTED on {_CONNECT_METHOD}): {details}"
                    ) from e
                else:
                    logger.warning("stream error %s: %s", code, details)
            except FatalTransportError:
                raise
            except HandlerError as e:
                # NEVER let this look like a dropped socket again.
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
                # After the reset: the evidence lane this emits into is the one
                # `reset_for_reconnect` just rebuilt, so the drop row rides the
                # NEXT stream instead of being cleared by this one's teardown.
                self._account_connection(
                    outcome=outcome,
                    ended_at=teardown_started,
                    teardown_s=time.monotonic() - teardown_started,
                )

            if self._stopping.is_set():
                return

            # The immediate-redirect chain is over; the hop budget refreshes
            # for the next leadership-churn episode.
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
                # BOTH the scheduled delay and the slept wall time: on a healthy
                # loop they agree to milliseconds, and their difference is the
                # loop starvation this issue exists to name.
                if self._episode is not None:
                    self._episode.sched_s += delay
                    self._episode.slept_s += max(
                        0.0, time.monotonic() - slept_from)

    async def _connect_once(self, target: str, use_tls: bool) -> None:
        """One connection lifetime; sets self._connected_at once HelloAck lands."""
        self._cycle.clear()
        channel = self._make_channel(target, use_tls)
        try:
            stub = pb_grpc.WorkerSchedulerStub(channel)

            async def _handshake() -> Any:
                # The Hello is built BEFORE the RPC opens: grpc.aio turns the
                # first write into `InvalidStateError: RPC already finished`
                # whenever the call terminated during any await between
                # `Connect()` and that write, which erases the status run()
                # classifies on (UNAUTHENTICATED's fatal-exit ladder,
                # not_leader redirects, permanent registration refusals).
                # `build_hello` is awaitable (in split mode a seam round-trip to
                # the child), so such an await is always present.
                hello = self._handlers.build_hello()
                if inspect.isawaitable(hello):
                    hello = await hello
                stream = stub.Connect(metadata=self._metadata())
                try:
                    await stream.write(pb.WorkerMessage(hello=hello))
                    return stream, await stream.read()
                except asyncio.InvalidStateError:
                    raise await _terminal_rpc_error(stream) from None

            # Deadline on the whole dial+Hello+HelloAck handshake: a hub that
            # accepts the stream but never answers must not hang the worker
            # forever (h2 keepalive is answered below the app layer).
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
            self._auth_verdicts.clear()               # healed
            self._expired_rejection_reported.clear()   # healed
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
                    # Same discipline as the drain close below: a write()
                    # return only means "buffered", so an abrupt channel close
                    # here would RST the call and discard the final buffered
                    # writes — including the supervisor's typed death
                    # JobResult, which mark_result_shipped has already retired
                    # from the durable queue. Half-close and wait briefly for
                    # the peer to end the call, then reconnect.
                    await self._close_sender(send_task)
                    await self._await_peer_close(stream, recv_task)
                    raise ConnectionError(
                        "stream cycled by supervisor (compute process restart)"
                    )
                if stop_task in done and self._clean_close:
                    # Drain close: half-close the stream, then WAIT for the
                    # peer to end the call. Closing the channel immediately
                    # after done_writing() RSTs the call and can discard the
                    # final buffered writes (e.g. the last JobResult).
                    await self._close_sender(send_task)
                    await self._await_peer_close(stream, recv_task)
                    return
                for t in pending:
                    t.cancel()
                for t in done:
                    if t is not stop_task:
                        try:
                            t.result()  # re-raise stream errors
                        except asyncio.CancelledError:
                            # NOT this worker being cancelled: grpc.aio raises
                            # CancelledError out of read()/write() when the CALL
                            # is locally cancelled (`_call._raise_for_status`).
                            # Nothing here was cancelled by us — `pending` was,
                            # and `pending` is not `done` — so this is a dead
                            # stream, and it reconnects like any other. Left to
                            # propagate it would ride out of run() and kill the
                            # process with no exit code.
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
        """End this stream's sender BETWEEN writes, never inside one.

        `send_task.cancel()` here drops a COMPLETED job's result: the sender may
        already have written that result (so `mark_result_shipped` retired the
        only durable copy) and be inside `stream.write()` of the next event when
        the cancel lands — and grpc.aio answers a cancelled write by cancelling
        the whole RPC, which RSTs the call and throws away everything buffered
        behind it, the result included.

        So: ask the queue to end the sender once it has nothing left to write,
        and give it the peer-alive window to get there. Cancelling remains the
        last resort, and it makes the close abrupt — which the hub reconciles
        — rather than a clean close that silently lost bytes.
        """
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
        """Half-close, then wait for the peer to end the call — the only
        evidence that the writes we buffered actually landed.

        `asyncio.wait` rather than `wait_for(shield(...))`: it neither cancels
        the receiver nor re-raises what it ended with, so an RPC-level
        cancellation cannot escape a graceful close. The `finally`
        in `_connect_once` retrieves the outcome.
        """
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
            # The last instant this process is PROVEN to have been scheduling.
            # A write into a dead socket still succeeds, so this is not peer
            # liveness — it is OUR liveness. The heartbeat loop forces a
            # StateDelta every beat in every state, so on a healthy worker this
            # is never more than one beat stale.
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
                # Kubelet-style rotation (contract §1): swap the stored
                # credential in place — no reconnect, no re-Hello. A fresh
                # rotation also clears any stale-token auth strikes.
                token = (msg.token_refresh.token or "").strip()
                if token:
                    # Publish where EVERY hub dial reads — the attestation
                    # carrier opens its own Connect and must not authenticate
                    # with the frozen boot token. This is the ONLY place the
                    # rotation lands, and it is deliberately NOT wrapped in a
                    # swallowing `except`: `install` is an assignment under a
                    # lock, so a raise means the rotation genuinely did not
                    # happen and must reach the transport's error path.
                    worker_credential.install(
                        token, float(msg.token_refresh.expires_at_unix or 0))
                    self._auth_verdicts.clear()
                    self._expired_rejection_reported.clear()
                    logger.info(
                        "worker JWT rotated by hub (exp=%d)",
                        msg.token_refresh.expires_at_unix,
                    )
                    # The rotation is OFFERED to the handler, which is NOT the
                    # same as forwarded to the compute child —
                    # `ParentControl.on_token_refresh` deliberately does nothing
                    # with it, because the child holds no credential and renews
                    # through `procsplit.broker` instead.
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
