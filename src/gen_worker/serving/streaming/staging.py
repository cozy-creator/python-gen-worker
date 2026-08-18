"""Double-buffered staging: store bytes -> pinned host memory -> device.

The pool is a small ring of large buffers. One buffer is being FILLED by the
store's ``readinto`` (a blocking read that releases the GIL in Rust) while the
others are being DRAINED by ``cudaMemcpyAsync`` on a dedicated copy stream.
A CUDA event per buffer gates reuse, so a buffer is never refilled while its
H2D is still in flight — that, and nothing else, is what makes the read and
the copy overlap.

Sizes: the default is 4 x 64 MiB. 64 MiB is the CAS object grid
(tensorfs#81's contract-directed chunking splits a tensor every 64 MiB), so a
window lines up with whole objects instead of straddling them, and 4 buffers
is enough to keep a read in flight while three copies drain without pinning
a gigabyte of host memory per load.

The CPU arm is not a simulation of this — it is the SAME driver loop with a
pageable pool and synchronous copies. Every ordering decision the engine makes
(file-order windows, scatter, event gating) is therefore exercised by the
CPU-side integration test, and only the ``cudaMemcpyAsync`` half is left for
the pod benchmark to prove.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

#: One staging buffer. A multiple of the 64 MiB CAS object grid.
DEFAULT_BUFFER_BYTES = 64 * 1024 * 1024
#: How many buffers ride the ring. Three is the minimum that keeps a read in
#: flight behind two draining copies; four is the default headroom.
DEFAULT_BUFFERS = 4


class StagingError(RuntimeError):
    """The staging pool could not be built or used as declared."""


def _writable_view(tensor: "torch.Tensor") -> memoryview:
    """A writable buffer-protocol view over a contiguous CPU uint8 tensor.

    ``torch.Tensor`` exports no buffer protocol, and ``.numpy()`` would put
    numpy on the base serving path, where this package deliberately carries
    none. The address is stable for a pinned allocation and the tensor is
    held by the slot that owns this view, so the view cannot outlive it.
    """
    import ctypes

    count = tensor.numel()
    block = (ctypes.c_ubyte * count).from_address(tensor.data_ptr())
    return memoryview(block).cast("B")


@dataclass(slots=True)
class _Slot:
    """One buffer plus the event that says when its last copy finished."""

    index: int
    tensor: "torch.Tensor"
    view: memoryview
    event: Optional[Any] = None


class StagingPool:
    """A ring of host buffers and the stream the copies ride.

    ``pinned`` is a FACT about the pool, not a request: pinned memory needs a
    CUDA context, so a CPU-destination load reports ``staging=pageable`` and
    says so in telemetry rather than claiming a property it does not have.
    """

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

    # -- the ring ----------------------------------------------------------

    def acquire(self) -> _Slot:
        """The next buffer, once its previous copy has actually landed.

        This wait is the whole gate. Without it the store would overwrite
        bytes the device is still reading, and the corruption would be
        nondeterministic and load-order dependent — the worst possible shape.
        """
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
        """Tell the caching allocator this tensor is written on the copy
        stream, not the stream it was allocated on. Skipping this is a
        use-after-free the allocator hands out as a silently wrong tensor."""
        if self._stream is not None:
            tensor.record_stream(self._stream)

    def finish(self) -> None:
        """Block until every enqueued copy has landed, then make the compute
        stream see them. Both halves are needed: the sync makes the bytes
        real, the ``wait_stream`` makes the ordering visible to later work."""
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
