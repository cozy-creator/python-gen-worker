"""gRPC transport: ONE bidi stream, bounded send queue, reconnect with jittered
backoff, `not_leader` redirects. Liveness is HTTP/2 keepalive only.

Send-queue policy (CONTRACT.md §1): JobResult is NEVER dropped — results
persist across reconnects until written to a live stream. Under overflow the
drop order is JobProgress (oldest first); everything else blocks the producer.

pgw#869 — THE EVIDENCE LANE. Results had a durable lane; measurements did not.
`reset_for_reconnect` cleared `_items` wholesale and restored only
`_pending_results`, so every queued ActivityUpdate/BootPhase — every
`self_mint_compile` phase, every `aot_mint phase=pool` ledger, every terminal —
was destroyed on each disconnect, and any evidence already popped by the sender
when the write failed had no durable copy at all. That is wall #7's mechanism:
the pod stayed alive with its numbers, the route to them did not.

Evidence now rides its own durable lane on the same principle as results:
enqueue-and-return (never blocks a producer), survives reconnect in FIFO order,
and retires only when a live stream has taken it. Coalescing is deliberately
asymmetric — detail-free RUNNING beats (`Activity.heartbeat` /
`progress_beat`) collapse to the latest per (kind, phase); anything carrying a
`detail`, an `error`, or a non-RUNNING state is evidence and is never dropped.
That split is not a guess: the hub's own upsert conflict key is
`(worker_id, kind, phase, state, self_stalled, payload_digest)` with
`payload_digest = sha256(error || 0x00 || detail)`
(tensorhub `internal/db/gen/worker_activity_events.sql.go`,
`repository/worker_activity_event_store.go`), so exactly what this coalesces is
what the hub would itself have folded into one row via `occurrences + 1`, while
everything it preserves is a distinct row hub-side.

In-memory only, deliberately (pgw#869 §1): the worker process did not restart in
the motivating incident, and a mint is a child of the worker, so a worker restart
ends the mint anyway. Disk persistence is out of scope and must be argued on its
own merits, not borrowed from this one.
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

from .config import Settings
from .pb import worker_scheduler_pb2 as pb
from .pb import worker_scheduler_pb2_grpc as pb_grpc
from . import worker_credential

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = pb.PROTOCOL_VERSION_CURRENT

#: The full gRPC method path this client dials. Derived from the generated
#: descriptor rather than written out, so it can never drift from the proto
#: package that defines the wire-protocol major (th#1597, §1.27(b)).
_CONNECT_METHOD = "/{}/Connect".format(
    pb.DESCRIPTOR.services_by_name["WorkerScheduler"].full_name
)

_RESULT, _PROGRESS, _EVENT = "result", "progress", "event"
#: pgw#869: a hub-bound FACT with no live-state replay behind it. Results have
#: `_pending_results`; state has `LifecycleSnapshot`/`StateDelta`/the capacity
#: lane. These two carry measurements and terminals and had neither.
_EVIDENCE = "evidence"
_EVIDENCE_MSGS = ("activity_update", "boot_phase")

_MAX_REDIRECT_HOPS = 3
# UNAUTHENTICATED can be transient hub-side (duplicate stream teardown, pg
# blip): exit only when failures persist across a real time window.
#
# pgw#869 — WHAT THIS LADDER MAY AND MAY NOT JUDGE. A rejected credential is a
# verdict about US; an ABSENT hub is a verdict about nothing. Before this
# change the ladder could not tell them apart, and the difference is the whole
# outbox: a worker that outlives a hub outage ages past its 30 min JWT, redials,
# is refused because its token expired, and DIES WITH A FULL QUEUE — every
# property the outbox proves, right up to the moment it matters. See
# `_auth_rejection_is_fatal`: an expired presented credential never reaches this
# ladder.
#
# The 3/60 s pair is a magic number and is left ONLY because it now governs the
# genuinely-about-us case that #372's tests pin. Replacing it with an
# evidence-keyed rule is filed separately (pgw#873), not smuggled in here.
_AUTH_FAILURE_EXIT_THRESHOLD = 3
_AUTH_FAILURE_EXIT_WINDOW_S = 60.0

#: pgw#848: how long before its own expiry a worker starts SAYING SO. Wider
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
    """A MESSAGE HANDLER raised — this is not a transport failure (gw#640).

    `_recv_loop` awaits the handlers inline, so a raise while handling (say) a
    RunJob used to propagate into `run()`'s catch-all and be logged as
    "connection to <addr> failed" — a handler bug wearing a dropped socket's
    clothes. The worker then reconnected forever while the hub, whose only
    death signal is a closed stream, reported `young worker death` and
    `workers kept dying mid-job`. Ten live th#1085 runs and two prior
    instrument releases (0.56.1 fatals, 0.56.2 post-mortem supervisor) found
    nothing because the process never died and nothing ever escaped to them.

    Carries which message was being handled so the report names it.
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

#: How long to wait for the peer to end a half-closed call before giving up on
#: a graceful close. Unchanged value, named so both close paths share it.
_PEER_CLOSE_WAIT_S = 5.0


class SenderQuiesced(Exception):
    """The send loop was asked to stop and the queue is empty (pgw#845).

    Ending the sender BETWEEN writes is not cosmetic: in grpc.aio a cancelled
    ``write()`` cancels the whole RPC (``_call._write`` calls ``self.cancel()``
    on CancelledError), which RSTs the stream and discards every byte buffered
    but not yet flushed — including a JobResult that ``mark_result_shipped``
    has already retired from the durable queue.
    """


#: pgw#869 §5: how many undelivered evidence facts the outbox holds before it
#: shortens itself. Sized against the motivating outage rather than guessed: a
#: whole-graph sdxl mint emits O(10^2) phase/measurement facts per hour, so this
#: is hours of a real mint's evidence. Growth is bounded by SHEDDING, and a shed
#: is itself a reported fact — silence about a loss is the thing this issue
#: exists to delete.
EVIDENCE_MAX = 4096

#: The shed report's own slot. Exempt from shedding: the one fact that must
#: never be lost is the fact that facts were lost.
_SHED_KEY = ("shed",)

#: pgw#869: how many just-shipped facts are re-offered after a DIRTY reconnect.
#:
#: `stream.write()` returning means BUFFERED, not delivered — the codebase
#: already knows this and says so in `_close_sender`: cancelling a write "RSTs
#: the call and throws away everything buffered behind it". So retiring a fact
#: the instant its write returns loses exactly the tail of facts produced in the
#: moments around a stream death, which is the window this whole issue is about.
#: MEASURED: the acceptance test below fails without this — three measurements
#: emitted immediately after the hub died were written into a socket that was
#: already gone, retired, and never replayed.
#:
#: A trailing ring makes that window recoverable at bounded cost. Re-offering is
#: safe by construction: the hub folds an identical redelivery into
#: `occurrences + 1` on one row (see the module docstring), so the choice is
#: between a duplicate and a hole, and a hole is not a choice.
RESHIP_WINDOW = 256


class SendQueue:
    """Bounded outbound queue with results-never-dropped semantics."""

    def __init__(self, maxsize: int = 1024, evidence_max: int = EVIDENCE_MAX) -> None:
        self._maxsize = maxsize
        # pgw#869: the durable evidence lane. Ordered (dict preserves insertion
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
        # pgw#845: set when this stream's sender must end. It ends only once
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
        # pgw#869: evidence is exempt for the same reason and one stronger —
        # it has its own bound (`_evidence_max`) with its own shed policy, and
        # a producer that just MEASURED something must never block on a dead
        # connection to report it.
        return sum(
            1 for kind, _msg in self._items if kind not in (_RESULT, _EVIDENCE)
        )

    # ---- pgw#869: the evidence lane ---------------------------------------

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
        """Bound the lane. Coalescible first, evidence last (pgw#869 §5)."""
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
        """Shedding evidence must itself emit a fact (pgw#869 §5).

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
        reconnect fences, and the pgw#869 evidence map all key on this and all
        exclude results by construction. A consumer that treats this map as
        covering every message has an unhandled case, not a typing nuisance.
        """
        if msg.WhichOneof("msg") == "job_result":
            return None
        return msg.SerializeToString(deterministic=True)

    @staticmethod
    def _evidence_bytes(msg: pb.WorkerMessage) -> bytes:
        """`_message_key` for a message already classified as `_EVIDENCE`.

        pgw#869 semantics, decided rather than defaulted. The two candidate
        answers for "what does the evidence map do with a keyless message" are
        both wrong, because **the question cannot arise**: `_msg_kind` admits
        exactly `activity_update` and `boot_phase` to this lane, and the only
        keyless message is `job_result`.

        * *Skip it* would make the map silently lossy — in the one module whose
          entire purpose is that facts are not silently lost.
        * *Give results their own evidence key* would put them in two durable
          lanes at once (`_pending_results` and `_pending_evidence`), which
          double-ships a result on reconnect and coalesces something the queue
          contract says is never coalesced.

        So this is total by precondition, and it says so here — at the boundary
        where the invariant could be violated — instead of narrowing an
        Optional at five call sites and leaving the reason nowhere.
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
                self._put_evidence(msg)               # pgw#869: never blocks
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
            # pgw#869: a fact retires from the durable lane only once a LIVE
            # stream has taken it. `_evidence_key` holds only the currently
            # stored copy, so a superseded beat can never retire the newer one.
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

        pgw#869: DURABLE lanes are requeued, not dropped. Results were already;
        evidence now is, in the order it was produced. Anything a live stream
        took but never confirmed (`mark_event_shipped` did not run because the
        write failed) is still in `_pending_evidence` and comes back with it —
        which is the second of the two loss sites this closes, and the one no
        amount of queueing would have covered on its own.
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
                or self._pending_evidence   # pgw#869: a drain flushes facts too
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
        awaitable form serves the pgw#763 split parent, which fetches the
        Hello from its compute child)
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
        queue_maxsize: int = 1024,
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
        self._consecutive_auth_failures = 0
        self._first_auth_failure_at: Optional[float] = None
        #: pgw#869: the `jti` the streak above is about. A rotation makes the
        #: accumulated evidence stale, so the streak restarts.
        self._auth_credential_id = ""
        #: Credentials whose expired-rejection has already been confessed.
        self._expired_rejection_reported: set = set()
        #: pgw#848: the last remaining-lifetime this worker reported, so a
        #: reconnect storm does not emit one event per attempt.
        self._last_credential_left: Optional[float] = None
        self._connected_at: Optional[float] = None  # set on each HelloAck
        # gw#640: (message kind, exception class) already dialed to the hub.
        self._reported_handler_failures: set = set()
        # Latest hub-pushed worker JWT (TokenRefresh, contract §1 rotation).
        # Used by the live connection's successor dials: reconnects always
        # present the newest credential; the boot-time settings token is only
        # the pre-rotation fallback.
        self._worker_jwt: Optional[str] = None
        # pgw#763: a supervisor-requested stream cycle (compute child respawn
        # needs a fresh Hello). Cleared at the start of each connection, so a
        # request set BEFORE a connection is satisfied by that connection's
        # own fresh Hello.
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
        """Newest worker credential: hub-rotated token, else the boot token."""
        # pgw#848: ONE source. `_worker_jwt` is this stream's rotation cache;
        # `worker_credential` is the process-wide truth every hub dial reads.

        return (self._worker_jwt or worker_credential.current() or "").strip()

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
        with a fresh Hello (pgw#763: the compute child was respawned, so the
        hub must re-sync desired state against the new process)."""
        self._cycle.set()

    # ---- connection loop ---------------------------------------------------

    def _channel_options(self) -> List[Tuple[str, int]]:
        return [
            ("grpc.keepalive_time_ms", 20000),
            ("grpc.keepalive_timeout_ms", int(KEEPALIVE_TIMEOUT_S * 1000)),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]

    def _make_channel(self, target: str, use_tls: bool) -> grpc.aio.Channel:
        if use_tls:
            # System trust roots. The custom-CA-bundle knob (GRPC_CA_BUNDLE)
            # was deleted in pgw#514 — no deployment ever set it.
            return grpc.aio.secure_channel(
                target,
                grpc.ssl_channel_credentials(),
                options=self._channel_options(),
            )
        return grpc.aio.insecure_channel(target, options=self._channel_options())

    def _metadata(self) -> Optional[List[Tuple[str, str]]]:

        token = (self._worker_jwt or worker_credential.current() or "").strip()
        if not token:
            return None
        self._report_credential_age(token)
        return [("authorization", f"Bearer {token}")]

    # ---- pgw#869: which auth rejections are a verdict about US ------------

    def _presented_credential(self) -> str:

        return (self._worker_jwt or worker_credential.current() or "").strip()

    @staticmethod
    def _credential_claims(token: str) -> dict:
        try:
            from .request_context import _decode_unverified_jwt_claims

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
        documented recovery path, and the ONLY thing that was preventing the
        heal was this worker killing itself first. `_report_credential_age`
        already said as much ("the hub has a boot-grace admission for exactly
        that case — so shortcutting the reconnect here would break a path that
        legitimately heals"); the ladder shortcut it anyway.

        And the billing risk that motivated the ladder is owned by the party
        that can actually see the pod: `grpc/worker_wedge.go` terminates a pod
        after `wedgeAuthRejectThreshold` consecutive auth rejects through the
        tracked provider path. A worker cannot reap itself when it cannot reach
        the hub, and it does not have to.

        A LIVE (or absent) credential refused by an answering hub IS about us —
        revoked, superseded, misconfigured — and keeps the existing ladder. The
        streak resets whenever the presented credential CHANGES, because a
        rotation means the next attempt genuinely differs.
        """
        token = self._presented_credential()
        exp = float(self._credential_claims(token).get("exp") or 0.0)
        if exp > 0 and exp <= time.time():
            self._consecutive_auth_failures = 0
            self._first_auth_failure_at = None
            self._report_expired_rejection(token, details, exp)
            return False

        cred = self._credential_id(token)
        if cred != self._auth_credential_id:
            # A different credential is a different attempt: the evidence the
            # streak had accumulated was about the old one.
            self._auth_credential_id = cred
            self._consecutive_auth_failures = 0
            self._first_auth_failure_at = None

        now = time.monotonic()
        if self._first_auth_failure_at is None:
            self._first_auth_failure_at = now
        self._consecutive_auth_failures += 1
        logger.error(
            "stream rejected UNAUTHENTICATED (%d consecutive over %.0fs) while "
            "presenting a LIVE credential — this is a verdict about this "
            "worker, not about the hub's availability: %s",
            self._consecutive_auth_failures,
            now - self._first_auth_failure_at, details,
        )
        return (
            self._consecutive_auth_failures >= _AUTH_FAILURE_EXIT_THRESHOLD
            and now - self._first_auth_failure_at >= _AUTH_FAILURE_EXIT_WINDOW_S
        )

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

        pgw#848. The worker has never looked at its own token's ``exp``. It
        learns of expiry only by being rejected — and by then it cannot
        recover, because the refresh arrives ONLY as a ``token_refresh`` down
        the stream (see the handler below) and the stream is the thing it can
        no longer open. A credential deliverable solely over the connection it
        authenticates cannot be delivered once that connection stops
        authenticating.

        MEASURED, hub `pod_events`, two consecutive whole-graph mints:

            attempt 16  T+32.4 min  worker_token_expired
                        T+42.9 min  "stream dropped involuntarily and never
                                     reconnected (10m13s of silence) — silent
                                     death mid-activity"
            attempt 17  T+31.2 min  worker_token_expired -> auth wedge

        ``DefaultWorkerJWTTTL`` is 30 minutes from POD CREATE, so this fires on
        any pod that lives past half an hour — which, until the pgw#848 cap
        fix, no mint ever did. Both runs destroyed a self-mint the hub's own
        record describes as "reporting fresh progress".

        DIAGNOSIS ONLY, deliberately: it changes no behaviour and refuses
        nothing. An expired token is not always fatal — the hub has a
        boot-grace admission for exactly that case — so shortcutting the
        reconnect here would break a path that legitimately heals. What was
        missing is not a decision, it is that ten minutes of silence carried
        no name.
        """
        try:
            from .request_context import _decode_unverified_jwt_claims

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
        """Log a handler failure as ITSELF and dial it to the hub (gw#640).

        Reuses the `worker_fatal` carrier, so this lands as a durable
        `pod_events` row on every hub pin already deployed — no proto change,
        no hub redeploy. Deduped per (message kind, exception class): the
        reconnect loop would otherwise re-dial the same fault every cycle.
        The reconnect itself is unchanged — this release unmasks the fault, it
        does not change liveness policy.
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
            try:
                await self._connect_once(target, use_tls)
            except grpc.aio.AioRpcError as e:
                code, details = e.code(), str(e.details() or "")
                if code == grpc.StatusCode.UNAUTHENTICATED:
                    if self._auth_rejection_is_fatal(details):
                        raise FatalTransportError(
                            f"authentication rejected {self._consecutive_auth_failures} times "
                            f"over {time.monotonic() - (self._first_auth_failure_at or 0):.0f}s "
                            f"while presenting a live credential: {details}"
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
                    # th#1597 / DESIGN-RULINGS §1.27(b): the wire-protocol MAJOR
                    # is the proto package, so it is in the service path
                    # (/cozy.scheduler.v1.WorkerScheduler/Connect). A hub that
                    # does not serve our major never registers that path and
                    # gRPC answers UNIMPLEMENTED before any hub code runs.
                    #
                    # This is FATAL and unretryable, and that is load-bearing
                    # HUB-side, not merely tidy here: the worker's exit is what
                    # makes the pod die before Hello, which leaves
                    # `everConnected` false, which is how th#874's death
                    # taxonomy marks the release `boot_crashing` and fails its
                    # queued requests. Retrying would reconnect-loop forever
                    # against a hub that can never answer, producing NO durable
                    # mark and no operator signal — strictly worse than dying.
                    raise FatalTransportError(
                        f"hub does not serve this wire-protocol major "
                        f"(UNIMPLEMENTED on {_CONNECT_METHOD}): {details}"
                    ) from e
                else:
                    logger.warning("stream error %s: %s", code, details)
            except FatalTransportError:
                raise
            except HandlerError as e:
                # gw#640: NEVER let this look like a dropped socket again.
                self._report_handler_failure(e)
            except Exception as e:
                logger.warning("connection to %s failed: %s: %s", target, type(e).__name__, e)
            finally:
                self._connected.clear()
                await self.queue.reset_for_reconnect()
                try:
                    await self._handlers.on_disconnect()
                except Exception:
                    logger.exception("on_disconnect handler failed")

            if self._stopping.is_set():
                return

            # The immediate-redirect chain is over; the hop budget refreshes
            # for the next leadership-churn episode.
            redirect_hops = 0

            now = time.monotonic()
            if self._connected_at is not None:
                if now - self._connected_at >= _BACKOFF_RESET_AFTER_S:
                    attempt = 0

            delay = random.uniform(0, min(self._backoff_cap, self._backoff_base * (2 ** attempt)))
            attempt += 1
            self.reconnect_delays.append(delay)
            logger.info("reconnecting in %.2fs (attempt %d)", delay, attempt)
            try:
                await asyncio.wait_for(self._stopping.wait(), delay)
                return
            except asyncio.TimeoutError:
                pass

    async def _connect_once(self, target: str, use_tls: bool) -> None:
        """One connection lifetime; sets self._connected_at once HelloAck lands."""
        self._cycle.clear()
        channel = self._make_channel(target, use_tls)
        try:
            stub = pb_grpc.WorkerSchedulerStub(channel)

            async def _handshake() -> Any:
                # The Hello is built BEFORE the RPC opens. grpc.aio turns the
                # first write into `InvalidStateError: RPC already finished`
                # whenever the call terminated during any await between
                # `Connect()` and that write — which erases the status run()
                # classifies on (UNAUTHENTICATED's fatal-exit ladder,
                # not_leader redirects, permanent registration refusals).
                # pgw#763 made build_hello awaitable (in split mode it is a
                # seam round-trip to the child), so that await was always
                # present and every refusal degraded to a nameless retry loop.
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
            self._consecutive_auth_failures = 0
            self._first_auth_failure_at = None
            self._auth_credential_id = ""
            self._expired_rejection_reported.clear()   # pgw#869: healed
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
                            # propagate it rode out of run() -> arun() ->
                            # Worker.run() and killed the process with no exit
                            # code (pgw#845).
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
        """End this stream's sender BETWEEN writes, never inside one (pgw#845).

        Measured: `send_task.cancel()` here dropped a COMPLETED job's result
        about one drain in six. The sender had already written that result
        (so `mark_result_shipped` had retired the only durable copy) and was
        inside `stream.write()` of the next event when the cancel landed —
        and grpc.aio answers a cancelled write by cancelling the whole RPC,
        which RSTs the call and throws away everything buffered behind it,
        the result included. The half-close that follows then had nothing
        left to flush, and `read()` raised CancelledError past every handler.

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
        cancellation cannot escape a graceful close (pgw#845). The `finally`
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
                    self._worker_jwt = token
                    # pgw#848: publish it where EVERY hub dial reads, not just
                    # this stream. The attestation carrier opens its own
                    # Connect and used to authenticate with the frozen boot
                    # token — which is what wedged attempts 16 and 17.
                    try:

                        worker_credential.install(
                            token, float(msg.token_refresh.expires_at_unix or 0))
                    except Exception:  # noqa: BLE001 — never break a rotation
                        logger.debug("credential publish failed", exc_info=True)
                    self._consecutive_auth_failures = 0
                    self._first_auth_failure_at = None
                    self._auth_credential_id = ""
                    self._expired_rejection_reported.clear()
                    logger.info(
                        "worker JWT rotated by hub (exp=%d)",
                        msg.token_refresh.expires_at_unix,
                    )
                    # pgw#763 delta 1: the rotation is OFFERED to the handler,
                    # which is NOT the same as forwarded to the compute child
                    # — `ParentControl.on_token_refresh` deliberately does
                    # nothing with it, because the child holds no credential
                    # and renews through `procsplit.broker` instead. The old
                    # comment here said the parent "forwards rotations to the
                    # compute child"; it never has (pgw#876 §2).
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
