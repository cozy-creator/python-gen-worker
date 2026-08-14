"""Child (compute-plane) side of the process split: a Transport-shaped object
that speaks frames to the control parent instead of gRPC to the hub.

Lifecycle/Executor are wired to this exactly as they are to the real
Transport — the residency protocol (CONFIG_APPLY / MATERIALIZE /
FUNCTION_READY, receipts), ctx, cancellation and job execution all run
in-process here, unchanged. Durable result queueing lives in the PARENT's real
SendQueue; this side writes through.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import Settings
from ..pb import worker_scheduler_pb2 as pb
from ..transport import FatalTransportError
from . import ENV_LIVENESS_FD, ENV_SOCKET, ENV_WATCHDOG_PING_S, frames

logger = logging.getLogger(__name__)

_DEFAULT_WATCHDOG_PING_S = 5.0
_BOOT_FATAL_SEND_TIMEOUT_S = 5.0
# How long the dying child waits for the parent to CONFIRM it has recorded the
# verdict. Bounded — a wedged parent must not hold a doomed child alive — and
# losing the ack only degrades to fire-and-forget.
_BOOT_FATAL_ACK_TIMEOUT_S = 10.0


def send_boot_fatal(report: Dict[str, Any], *, kind: str = "hardware_unsuitable") -> bool:
    """Hand the parent a TERMINAL typed boot verdict before exiting.

    Runs pre-transport (the CUDA probe fails before ChildTransport exists), so
    it opens its own short-lived socket. The parent propagates the report on
    its credential and exits 1 instead of respawning — a hardware verdict is
    not a transient fault.

    After sending it WAITS (bounded) for the parent's T_BOOT_FATAL_ACK: a child
    that exits immediately can be reaped before the parent reads the frame off
    the socket buffer, downgrading the typed verdict to a crash-to-retry. The
    ack is written only after the parent has recorded the verdict, so surviving
    this call means the respawn decision will see it.
    """
    path = os.environ.get(ENV_SOCKET, "").strip()
    if not path:
        return False
    payload = frames.pack_meta({"kind": kind, "terminal": True, "report": report})
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_BOOT_FATAL_SEND_TIMEOUT_S)
            sock.connect(path)
            sock.sendall(frames.frame_bytes(frames.T_BOOT_FATAL, payload))
            sock.settimeout(_BOOT_FATAL_ACK_TIMEOUT_S)
            try:
                _wait_boot_fatal_ack(sock)
            except (socket.timeout, OSError, ValueError):
                logger.warning("no boot-fatal ack from the control parent "
                               "within %.0fs; exiting anyway (the verdict may "
                               "race the reap)", _BOOT_FATAL_ACK_TIMEOUT_S,
                               exc_info=True)
    except OSError:
        logger.warning("could not hand the boot fatal to the control parent",
                       exc_info=True)
        return False
    return True


def _wait_boot_fatal_ack(sock: socket.socket) -> None:
    """Blocking read until T_BOOT_FATAL_ACK arrives.

    The transient boot-fatal connection becomes the slot's link parent-side,
    so unrelated parent->child frames (e.g. a T_HELLO_REQ from a concurrently
    connecting hub stream) may precede the ack — skip them. Bounded by the
    socket timeout the caller set.
    """
    while True:
        header = _recv_exact(sock, 5)
        ftype, length = header[0], int.from_bytes(header[1:5], "big")
        # `length` is 4 attacker-supplied bytes
        # off the control socket, so it may declare up to 4 GiB. The bound
        # already exists — frames.MAX_FRAME_BYTES, enforced by BOTH ends of the
        # normal path (frames.read_frame and FrameWriter.frame). This reader is
        # hand-rolled for the boot-fatal ack and skipped it, so one route into
        # the child had no ceiling while its siblings did.
        #
        # The docstring's claim that this is "bounded by the socket timeout"
        # was wrong in a way worth naming: settimeout bounds each recv, not the
        # accumulation, so a peer that dribbles bytes resets the clock forever.
        if length > frames.MAX_FRAME_BYTES:
            raise ValueError(
                f"frame of {length} bytes exceeds {frames.MAX_FRAME_BYTES}")
        if length:
            _recv_exact(sock, length)
        if ftype == frames.T_BOOT_FATAL_ACK:
            return


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    # accumulate into a bytearray, not `buf += chunk`. The old form
    # reallocated and copied the whole buffer per chunk, so a large frame cost
    # O(n^2) — the length bound above caps n, but quadratic copying at 128 MiB
    # is still a stall, and there is no reason to pay it.
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise OSError("socket closed before the boot-fatal ack arrived")
        buf.extend(chunk)
    return bytes(buf)


def _ping_interval() -> float:
    try:
        return float(os.environ.get(ENV_WATCHDOG_PING_S, "") or _DEFAULT_WATCHDOG_PING_S)
    except ValueError:
        return _DEFAULT_WATCHDOG_PING_S


def start_liveness_thread() -> Optional[threading.Thread]:
    """Report WHAT IS OPEN on a thread, not on the event loop.

    The frame ping in ``ChildTransport`` is an asyncio task, so an inductor
    compile that starves the loop silences it — and a parent that kills on that
    silence SIGKILLs a live compile and labels it ``watchdog_hang``. Loop
    silence may only ARM the verdict; the open activity DECIDES it.

    This thread carries only that one fact — which activity is open — over a
    dedicated pipe with one atomic ``os.write``. Deliberately NOT the evidence:
    a CPU-bound Python phase (dynamo tracing) can starve a thread of the GIL
    for seconds, so nothing the child says about its own liveness can be the
    decider. The parent measures the evidence from /proc instead.
    """
    raw = os.environ.get(ENV_LIVENESS_FD, "").strip()
    if not raw:
        return None
    try:
        fd = int(raw)
    except ValueError:
        return None
    interval = max(0.1, _ping_interval())
    from .. import activity as activity_mod

    def _run() -> None:
        while True:
            try:
                act = activity_mod.current()
                os.write(fd, frames.frame_bytes(frames.T_LIVENESS, frames.pack_meta({
                    "act": act is not None,
                    "kind": getattr(act, "kind", "") or "",
                })))
            except OSError:
                return          # parent gone; the parent's waitpid is the truth
            except Exception:
                logger.debug("liveness ping sample failed", exc_info=True)
            time.sleep(interval)

    t = threading.Thread(target=_run, name="compute-liveness", daemon=True)
    t.start()
    return t


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
        self._stopping = asyncio.Event()
        self._broker: Optional[Any] = None
        self._writer: Optional[frames.FrameWriter] = None
        self._flush_waiter: Optional[asyncio.Future] = None
        self._reported_handler_failures: set = set()

    # ---- Transport surface used by Lifecycle / Executor / Worker ---------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def current_worker_jwt(self) -> str:
        """Always empty in the compute child (delta 1).

        The worker JWT is the pod's signing identity — the stream credential,
        the capability minter, the authority behind the platform C2PA oracle.
        This process imports tenant endpoint code, so anything it holds, that
        code holds. The parent strips ``WORKER_JWT`` from this process's
        environment and no frame carries it, so there is nothing here to
        return; identity-bearing calls go through ``procsplit.broker`` and the
        parent decides.

        Kept as a property (rather than deleted) because it is the Transport
        surface Lifecycle/Executor bind at boot, and an empty string is the
        honest answer: this process has no credential.
        """
        return ""

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
        # delta 1: the child's only route to an identity-bearing hub call.
        from .broker import ChildBroker, install as install_broker

        self._broker = ChildBroker(asyncio.get_running_loop(), self._frame)
        install_broker(self._broker)
        start_liveness_thread()
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
            from .broker import install as install_broker

            if self._broker is not None:
                self._broker.fail_all("control parent link lost")
            install_broker(None)
            self._broker = None
            self._writer.close()
            self._writer = None

    async def _frame(self, ftype: int, payload: bytes) -> None:
        writer = self._writer
        if writer is None:
            raise ConnectionError("control parent link is down")
        await writer.frame(ftype, payload)

    async def _watchdog_ping(self) -> None:
        """EVENT-LOOP liveness: this proves the loop is turning, and
        nothing more. Its silence ARMS the parent's hang verdict; the
        thread-sourced liveness pipe is what decides it."""
        interval = _ping_interval()
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
            elif ftype == frames.T_ACTION_RESP:
                if self._broker is not None:
                    self._broker.resolve(frames.unpack_meta(payload))
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
            # Same doctrine as Transport.HandlerError: a handler bug
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
