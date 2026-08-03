"""Length-prefixed frames over the parent<->child unix socket.

The seam carries the SAME serialized protobuf messages that cross the
worker<->hub gRPC stream (WorkerMessage / SchedulerMessage bytes), plus a
handful of msgpack control frames. No second serialization of payloads or
results exists: what was already wire-bytes stays wire-bytes.

Header: 1-byte frame type + 4-byte big-endian payload length.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any, Tuple

import msgspec

_HEADER = struct.Struct(">BI")
# One frame comfortably holds the largest gRPC message the stream itself
# allows (64 MiB) with headroom.
MAX_FRAME_BYTES = 128 * 1024 * 1024

# parent -> child
T_HELLO_ACK = 1      # pb.HelloAck bytes
T_SCHED = 2          # pb.SchedulerMessage bytes
T_SHIPPED = 3        # pb.WorkerMessage bytes (model_event delivery receipts)
T_HELLO_REQ = 4      # msgpack {}
T_CONNECTED = 5      # empty
T_DISCONNECTED = 6   # empty
# 7 was T_TOKEN — the worker JWT, handed to the child on every rotation.
# DELETED (delta 1): the child must never hold the signing identity, so there is
# nothing to send. What replaced it is T_ACTION_REQ/RESP below: the child ASKS
# the parent for a narrow, allowlisted, audited action and the parent — which
# holds the JWT — performs it.
T_FLUSH_ACK = 8      # msgpack {"flushed": bool}
T_ACTION_RESP = 9    # msgpack {"id": int, "ok": bool, "status": int,
                     #          "body": str, "error": str}
# pgw#833 (pgw#826 follow-on): written by the parent as soon as it has
# RECORDED a T_BOOT_FATAL verdict. The dying child waits (bounded) for this
# before exiting, so the parent's respawn decision can never race the frame
# still sitting in the socket buffer.
T_BOOT_FATAL_ACK = 10  # msgpack {}

# child -> parent
T_HELLO = 20         # pb.Hello bytes
T_WORKER_MSG = 21    # pb.WorkerMessage bytes
T_PREPEND = 22       # msgpack [pb.WorkerMessage bytes, ...]
T_FLUSH_REQ = 23     # msgpack {"timeout": float | None}
T_WATCHDOG = 24      # msgpack {}  (event-loop liveness: the loop is turning)
# pgw#771: which activity is open, written by a THREAD over a dedicated pipe so
# a starved event loop cannot silence it. Evidence is NOT carried here — the
# parent measures the child's CPU/IO itself, because a GIL-starved thread
# cannot be the decider of its own process's liveness.
T_LIVENESS = 25      # msgpack {"act": bool, "kind": str}
# delta 1: the child's request for a parent-mediated action. The IPC surface is
# an AUTHORIZATION surface — the child names an action and its arguments; the
# parent decides, supplies the credential and the base URL, and returns only the
# result. msgpack {"id": int, "method": str, "path": str, "query": {..},
#                  "json": {..}, "timeout": float}
T_ACTION_REQ = 26
# pgw#826: a TERMINAL typed boot verdict, sent pre-transport by a child that is
# about to exit (e.g. the CUDA probe failed). The parent propagates the report
# on its credential and exits 1 instead of respawning.
# msgpack {"kind": str, "terminal": bool, "report": {..HardwareReport fields..}}
T_BOOT_FATAL = 27


def pack_meta(obj: Any) -> bytes:
    return msgspec.msgpack.encode(obj)


def unpack_meta(raw: bytes) -> Any:
    return msgspec.msgpack.decode(raw)


class FrameWriter:
    """Serializes concurrent frame writers onto one stream."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._lock = asyncio.Lock()

    async def frame(self, ftype: int, payload: bytes = b"") -> None:
        if len(payload) > MAX_FRAME_BYTES:
            raise ValueError(f"frame of {len(payload)} bytes exceeds {MAX_FRAME_BYTES}")
        async with self._lock:
            self._writer.write(_HEADER.pack(ftype, len(payload)))
            if payload:
                self._writer.write(payload)
            await self._writer.drain()

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass


def frame_bytes(ftype: int, payload: bytes = b"") -> bytes:
    """One frame as a single buffer — for callers that must write it with one
    atomic ``os.write`` from a thread (pgw#771's liveness pipe)."""
    return _HEADER.pack(ftype, len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    header = await reader.readexactly(_HEADER.size)
    ftype, length = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {length} bytes exceeds {MAX_FRAME_BYTES}")
    payload = await reader.readexactly(length) if length else b""
    return ftype, payload
