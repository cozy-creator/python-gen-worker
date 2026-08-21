"""Length-prefixed frames over the parent<->child unix socket."""

from __future__ import annotations

import asyncio
import struct
from typing import Any, Tuple

import msgspec

_HEADER = struct.Struct(">BI")
MAX_FRAME_BYTES = 128 * 1024 * 1024

T_HELLO_ACK = 1
T_SCHED = 2
T_SHIPPED = 3
T_HELLO_REQ = 4
T_CONNECTED = 5
T_DISCONNECTED = 6
T_FLUSH_ACK = 8
T_ACTION_RESP = 9
T_BOOT_FATAL_ACK = 10

T_HELLO = 20
T_WORKER_MSG = 21
T_PREPEND = 22
T_FLUSH_REQ = 23
T_WATCHDOG = 24
T_LIVENESS = 25
T_ACTION_REQ = 26
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
    """One frame as a single buffer — for callers that must write it with one atomic ``os.write`` from a thread (the liveness pipe)."""
    return _HEADER.pack(ftype, len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    header = await reader.readexactly(_HEADER.size)
    ftype, length = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {length} bytes exceeds {MAX_FRAME_BYTES}")
    payload = await reader.readexactly(length) if length else b""
    return ftype, payload
