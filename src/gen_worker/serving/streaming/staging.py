"""Double-buffered staging: store bytes -> pinned host memory -> device."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_BYTES = 64 * 1024 * 1024
DEFAULT_BUFFERS = 4


class StagingError(RuntimeError):
    """The staging pool could not be built or used as declared."""


def _writable_view(tensor: "torch.Tensor") -> memoryview:
    import ctypes

    count = tensor.numel()
    block = (ctypes.c_ubyte * count).from_address(tensor.data_ptr())
    return memoryview(block).cast("B")


@dataclass(slots=True)
class _Slot:

    index: int
    tensor: "torch.Tensor"
    view: memoryview
    event: Optional[Any] = None


class StagingPool:
    """A ring of host buffers and the stream the copies ride."""

    def __init__(
        self,
        device: "torch.device",
        *,
        buffer_bytes: int = DEFAULT_BUFFER_BYTES,
        buffers: int = DEFAULT_BUFFERS,
    ) -> None:
        import torch

        if buffers < 2:
            raise StagingError(
                f"a staging pool of {buffers} buffer(s) cannot double-buffer; "
                "the read and the copy would serialize"
            )
        if buffer_bytes <= 0:
            raise StagingError("staging buffers must hold bytes")

        self.device = device
        self.buffer_bytes = int(buffer_bytes)
        self.pinned = device.type == "cuda"
        self._torch = torch
        self._stream: Optional[Any] = (
            torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            if self.pinned
            else None
        )
        self._slots: List[_Slot] = []
        for index in range(buffers):
            host = torch.empty(
                self.buffer_bytes, dtype=torch.uint8, pin_memory=self.pinned
            )
            self._slots.append(
                _Slot(index=index, tensor=host, view=_writable_view(host))
            )
        self._next = 0

    def acquire(self) -> _Slot:
        """The next buffer, once its previous copy has actually landed."""
        slot = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        if slot.event is not None:
            slot.event.synchronize()
            slot.event = None
        return slot

    def copy_out(
        self, slot: _Slot, src_offset: int, dst: "torch.Tensor", dst_offset: int, count: int
    ) -> None:
        """Enqueue ``count`` bytes of ``slot`` into a flat uint8 destination."""
        source = slot.tensor[src_offset : src_offset + count]
        target = dst[dst_offset : dst_offset + count]
        if self._stream is None:
            target.copy_(source)
            return
        with self._torch.cuda.stream(self._stream):
            target.copy_(source, non_blocking=True)

    def release(self, slot: _Slot) -> None:
        """Mark every copy enqueued out of this buffer, so reuse can wait."""
        if self._stream is None:
            return
        event = self._torch.cuda.Event()  # type: ignore[no-untyped-call]
        event.record(self._stream)
        slot.event = event

    def track(self, tensor: "torch.Tensor") -> None:
        """Tell the caching allocator this tensor is written on the copy stream, not the stream it was allocated on."""
        if self._stream is not None:
            tensor.record_stream(self._stream)

    def finish(self) -> None:
        """Block until every enqueued copy has landed, then make the compute stream see them."""
        if self._stream is None:
            return
        self._stream.synchronize()
        self._torch.cuda.current_stream(self.device).wait_stream(self._stream)

    def close(self) -> None:
        self._slots.clear()

    @property
    def staging(self) -> str:
        """The telemetry token: ``pinned`` or ``pageable``."""
        return "pinned" if self.pinned else "pageable"

    def __enter__(self) -> "StagingPool":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.finish()
        self.close()


__all__ = [
    "DEFAULT_BUFFERS",
    "DEFAULT_BUFFER_BYTES",
    "StagingError",
    "StagingPool",
]
