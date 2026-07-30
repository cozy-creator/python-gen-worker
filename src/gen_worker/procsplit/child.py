"""Child (compute-plane) side of the pgw#763 split: a Transport-shaped object
that speaks frames to the control parent instead of gRPC to the hub.

Lifecycle/Executor are wired to this exactly as they are to the real
Transport — the residency protocol (CONFIG_APPLY / MATERIALIZE /
FUNCTION_READY, th#1283 receipts), ctx, cancellation, and job execution all
run in-process here, unchanged. Durable result queueing lives in the PARENT's
real SendQueue; this side writes through.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional, Tuple

from ..config import Settings
from ..pb import worker_scheduler_pb2 as pb
from ..transport import FatalTransportError
from . import ENV_SOCKET, ENV_WATCHDOG_PING_S, frames

logger = logging.getLogger(__name__)

_DEFAULT_WATCHDOG_PING_S = 5.0


class _QueueShim:
    """Lifecycle.build_hello reads ``transport.queue.pending_result_keys``.
    The child holds no durable results (the parent's SendQueue does, and the
    parent merges its pending keys into every Hello it relays)."""

    @property
    def pending_result_keys(self) -> List[Tuple[str, int]]:
        return []


class ChildTransport:
    """The compute child's stand-in for ``Transport``.

    handlers is the Lifecycle: build_hello / on_hello_ack / on_message /
    on_message_shipped / on_disconnect, same contract as Transport.
    """

    def __init__(self, settings: Settings, handlers: Any) -> None:
        self._settings = settings
        self._handlers = handlers
        self.queue = _QueueShim()
        self._connected = False
        self._worker_jwt: Optional[str] = None
        self._stopping = asyncio.Event()
        self._writer: Optional[frames.FrameWriter] = None
        self._flush_waiter: Optional[asyncio.Future] = None
        self._reported_handler_failures: set = set()

    # ---- Transport surface used by Lifecycle / Executor / Worker ---------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def current_worker_jwt(self) -> str:
        return (self._worker_jwt or self._settings.worker_jwt or "").strip()

    async def send(self, msg: pb.WorkerMessage) -> None:
        writer = self._writer
        if writer is None:
            return
        await writer.frame(frames.T_WORKER_MSG, msg.SerializeToString())

    async def prepend_reconnect(self, messages: List[pb.WorkerMessage]) -> None:
        writer = self._writer
        if writer is None:
            return
        payload = frames.pack_meta([m.SerializeToString() for m in messages])
        await writer.frame(frames.T_PREPEND, payload)

    async def close_after_flush(self, timeout: Optional[float] = None) -> bool:
        """Drain: ask the parent to flush its durable queue, then stop."""
        writer = self._writer
        flushed = False
        if writer is not None:
            loop = asyncio.get_running_loop()
            self._flush_waiter = loop.create_future()
            await writer.frame(
                frames.T_FLUSH_REQ, frames.pack_meta({"timeout": timeout})
            )
            wait = None if timeout is None else timeout + 10.0
            try:
                flushed = bool(await asyncio.wait_for(self._flush_waiter, wait))
            except (asyncio.TimeoutError, asyncio.CancelledError):
                flushed = False
            finally:
                self._flush_waiter = None
        self._stopping.set()
        return flushed

    def stop(self) -> None:
        self._stopping.set()

    # ---- run loop --------------------------------------------------------

    async def run(self) -> None:
        path = os.environ.get(ENV_SOCKET, "").strip()
        if not path:
            raise FatalTransportError(f"{ENV_SOCKET} is not set in the compute child")
        try:
            reader, writer = await asyncio.open_unix_connection(path)
        except OSError as exc:
            raise FatalTransportError(f"cannot reach control parent at {path}: {exc}") from exc
        self._writer = frames.FrameWriter(writer)
        ping = asyncio.create_task(self._watchdog_ping(), name="child-watchdog-ping")
        stop_task = asyncio.create_task(self._stopping.wait(), name="child-stop")
        recv = asyncio.create_task(self._recv_loop(reader), name="child-recv")
        try:
            done, _pending = await asyncio.wait(
                (recv, stop_task), return_when=asyncio.FIRST_COMPLETED
            )
            if recv in done:
                recv.result()
        finally:
            for t in (ping, recv, stop_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(ping, recv, stop_task, return_exceptions=True)
            self._connected = False
            self._writer.close()
            self._writer = None

    async def _watchdog_ping(self) -> None:
        try:
            interval = float(os.environ.get(ENV_WATCHDOG_PING_S, "") or _DEFAULT_WATCHDOG_PING_S)
        except ValueError:
            interval = _DEFAULT_WATCHDOG_PING_S
        payload = frames.pack_meta({})
        while not self._stopping.is_set():
            writer = self._writer
            if writer is not None:
                try:
                    await writer.frame(frames.T_WATCHDOG, payload)
                except Exception:
                    return
            await asyncio.sleep(interval)

    async def _recv_loop(self, reader: asyncio.StreamReader) -> None:
        while True:
            try:
                ftype, payload = await frames.read_frame(reader)
            except (asyncio.IncompleteReadError, ConnectionError, OSError):
                if self._stopping.is_set():
                    return
                # The control parent is the container: without it there is no
                # stream, no identity, and no reason to compute.
                raise FatalTransportError("control parent link lost")
            await self._dispatch(ftype, payload)

    async def _dispatch(self, ftype: int, payload: bytes) -> None:
        try:
            if ftype == frames.T_CONNECTED:
                self._connected = True
            elif ftype == frames.T_DISCONNECTED:
                self._connected = False
                await self._handlers.on_disconnect()
            elif ftype == frames.T_HELLO_ACK:
                await self._handlers.on_hello_ack(pb.HelloAck.FromString(payload))
            elif ftype == frames.T_SCHED:
                await self._handlers.on_message(pb.SchedulerMessage.FromString(payload))
            elif ftype == frames.T_SHIPPED:
                shipped = getattr(self._handlers, "on_message_shipped", None)
                if shipped is not None:
                    await shipped(pb.WorkerMessage.FromString(payload))
            elif ftype == frames.T_HELLO_REQ:
                hello = self._handlers.build_hello()
                writer = self._writer
                if writer is not None:
                    await writer.frame(frames.T_HELLO, hello.SerializeToString())
            elif ftype == frames.T_TOKEN:
                meta = frames.unpack_meta(payload)
                token = str(meta.get("token") or "").strip()
                if token:
                    self._worker_jwt = token
            elif ftype == frames.T_FLUSH_ACK:
                meta = frames.unpack_meta(payload)
                waiter = self._flush_waiter
                if waiter is not None and not waiter.done():
                    waiter.set_result(bool(meta.get("flushed")))
            else:
                logger.warning("unknown control frame type %d ignored", ftype)
        except (FatalTransportError, asyncio.CancelledError):
            raise
        except Exception:
            # Same doctrine as Transport.HandlerError (gw#640): a handler bug
            # must never masquerade as a dropped link. Log as itself; the
            # parent's gRPC transport already dials handler failures when they
            # occur there — here the process stays alive and keeps serving.
            key = (ftype, "handler")
            if key not in self._reported_handler_failures:
                self._reported_handler_failures.add(key)
                logger.exception(
                    "HANDLER FAILURE while handling control frame %d "
                    "(process alive; link kept)", ftype,
                )
            else:
                logger.error("handler failure for control frame %d (repeat)", ftype)
