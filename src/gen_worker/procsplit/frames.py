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
T_TOKEN = 7          # msgpack {"token": str, "exp": int}
T_FLUSH_ACK = 8      # msgpack {"flushed": bool}

# child -> parent
T_HELLO = 20         # pb.Hello bytes
T_WORKER_MSG = 21    # pb.WorkerMessage bytes
T_PREPEND = 22       # msgpack [pb.WorkerMessage bytes, ...]
T_FLUSH_REQ = 23     # msgpack {"timeout": float | None}
T_WATCHDOG = 24      # msgpack {}  (sd_notify-style liveness ping)


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


async def read_frame(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    header = await reader.readexactly(_HEADER.size)
    ftype, length = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {length} bytes exceeds {MAX_FRAME_BYTES}")
    payload = await reader.readexactly(length) if length else b""
    return ftype, payload
