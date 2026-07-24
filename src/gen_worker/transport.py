"""gRPC transport: ONE bidi stream, bounded send queue, reconnect with jittered
backoff, `not_leader` redirects. Liveness is HTTP/2 keepalive only.

Send-queue policy (CONTRACT.md §1): JobResult is NEVER dropped — results
persist across reconnects until written to a live stream. Under overflow the
drop order is JobProgress (oldest first); everything else blocks the producer.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import random
import time
from typing import Any, List, Optional, Tuple

import grpc
import grpc.aio

from .config import Settings
from .pb import worker_scheduler_pb2 as pb
from .pb import worker_scheduler_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = pb.PROTOCOL_VERSION_CURRENT

_RESULT, _PROGRESS, _EVENT = "result", "progress", "event"

_MAX_REDIRECT_HOPS = 3
_AUTH_FAILURE_EXIT_THRESHOLD = 3
# UNAUTHENTICATED can be transient hub-side (duplicate stream teardown, pg
# blip): exit only when failures persist across a real time window.
_AUTH_FAILURE_EXIT_WINDOW_S = 60.0
_BACKOFF_RESET_AFTER_S = 60.0
_HELLO_ACK_TIMEOUT_S = 30.0
# FAILED_PRECONDITION details that can never heal by retrying: identity is
# wrong for this deployment, so retrying cannot repair it.
_PERMANENT_PRECONDITION_MARKERS = (
    "worker_id_mismatch",
    "release_id_mismatch",
    "missing worker identity",
)


class FatalTransportError(Exception):
    """Unrecoverable protocol mismatch or persistent registration rejection."""


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
    return _EVENT


class SendQueue:
    """Bounded outbound queue with results-never-dropped semantics."""

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
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
        return sum(1 for kind, _msg in self._items if kind != _RESULT)

    @staticmethod
    def _message_key(msg: pb.WorkerMessage) -> Optional[bytes]:
        if msg.WhichOneof("msg") == "job_result":
            return None
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

    async def get(self) -> Tuple[str, pb.WorkerMessage]:
        async with self._cond:
            while not self._capacity and not self._reconnect and not self._items:
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
        """Drop transient lanes; executor state replays capacity after HelloAck."""
        async with self._cond:
            self._reconnect.clear()
            self._reconnect_seen.clear()
            self._in_flight.clear()
            self._capacity.clear()
            self._capacity_in_flight.clear()
            self._state_attempts.clear()
            self._items.clear()
            for msg in self._pending_results.values():
                self._items.append((_RESULT, msg))
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
      build_hello() -> pb.Hello                     (fresh full snapshot)
      on_hello_ack(ack: pb.HelloAck) -> Awaitable   (also mid-stream re-sends)
      on_message(msg: pb.SchedulerMessage) -> Awaitable  (MUST NOT block)
      on_message_shipped(msg: pb.WorkerMessage) -> Awaitable (optional)
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
        self._connected_at: Optional[float] = None  # set on each HelloAck
        # Latest hub-pushed worker JWT (TokenRefresh, contract §1 rotation).
        # Used by the live connection's successor dials: reconnects always
        # present the newest credential; the boot-time settings token is only
        # the pre-rotation fallback.
        self._worker_jwt: Optional[str] = None

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
        return (self._worker_jwt or self._settings.worker_jwt or "").strip()

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

    # ---- connection loop ---------------------------------------------------

    def _channel_options(self) -> List[Tuple[str, int]]:
        return [
            ("grpc.keepalive_time_ms", 20000),
            ("grpc.keepalive_timeout_ms", 10000),
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
        token = (self._worker_jwt or self._settings.worker_jwt or "").strip()
        if not token:
            return None
        return [("authorization", f"Bearer {token}")]

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
                    now = time.monotonic()
                    if self._first_auth_failure_at is None:
                        self._first_auth_failure_at = now
                    self._consecutive_auth_failures += 1
                    logger.error("stream rejected UNAUTHENTICATED (%d consecutive over %.0fs): %s",
                                 self._consecutive_auth_failures,
                                 now - self._first_auth_failure_at, details)
                    if (
                        self._consecutive_auth_failures >= _AUTH_FAILURE_EXIT_THRESHOLD
                        and now - self._first_auth_failure_at >= _AUTH_FAILURE_EXIT_WINDOW_S
                    ):
                        raise FatalTransportError(
                            f"authentication rejected {self._consecutive_auth_failures} times "
                            f"over {now - self._first_auth_failure_at:.0f}s: {details}"
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
                else:
                    logger.warning("stream error %s: %s", code, details)
            except FatalTransportError:
                raise
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
        channel = self._make_channel(target, use_tls)
        try:
            stub = pb_grpc.WorkerSchedulerStub(channel)
            stream = stub.Connect(metadata=self._metadata())

            async def _handshake() -> Any:
                await stream.write(pb.WorkerMessage(hello=self._handlers.build_hello()))
                return await stream.read()

            # Deadline on the whole dial+Hello+HelloAck handshake: a hub that
            # accepts the stream but never answers must not hang the worker
            # forever (h2 keepalive is answered below the app layer).
            try:
                first = await asyncio.wait_for(_handshake(), _HELLO_ACK_TIMEOUT_S)
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
            self._connected.set()
            logger.info("connected to %s (HelloAck ok)", target)
            await self._handlers.on_hello_ack(ack)

            send_task = asyncio.create_task(self._send_loop(stream), name="transport-send")
            recv_task = asyncio.create_task(self._recv_loop(stream), name="transport-recv")
            stop_task = asyncio.create_task(self._stopping.wait(), name="transport-stop")
            try:
                done, pending = await asyncio.wait(
                    (send_task, recv_task, stop_task), return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task in done and self._clean_close:
                    # Drain close: half-close the stream, then WAIT for the
                    # peer to end the call. Closing the channel immediately
                    # after done_writing() RSTs the call and can discard the
                    # final buffered writes (e.g. the last JobResult).
                    send_task.cancel()
                    try:
                        await stream.done_writing()
                        await asyncio.wait_for(asyncio.shield(recv_task), 5.0)
                    except (asyncio.TimeoutError, ConnectionError):
                        pass
                    except Exception:
                        pass
                    return
                for t in pending:
                    t.cancel()
                for t in done:
                    if t is not stop_task:
                        t.result()  # re-raise stream errors
            finally:
                for t in (send_task, recv_task, stop_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(send_task, recv_task, stop_task, return_exceptions=True)
        finally:
            self._connected.clear()
            await channel.close()

    async def _send_loop(self, stream: Any) -> None:
        while True:
            kind, msg = await self.queue.get()
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
            if which == "hello_ack":
                await self._handlers.on_hello_ack(msg.hello_ack)
                continue
            if which == "token_refresh":
                # Kubelet-style rotation (contract §1): swap the stored
                # credential in place — no reconnect, no re-Hello. A fresh
                # rotation also clears any stale-token auth strikes.
                token = (msg.token_refresh.token or "").strip()
                if token:
                    self._worker_jwt = token
                    self._consecutive_auth_failures = 0
                    self._first_auth_failure_at = None
                    logger.info(
                        "worker JWT rotated by hub (exp=%d)",
                        msg.token_refresh.expires_at_unix,
                    )
                continue
            await self._handlers.on_message(msg)
