"""GPU->host frame staging for the video encode path (gw#549).

The finalize wall and the D2H stall inside the compute wall are CPU/PCIe
bound (ie#484: identical GPU step-ms, 4-10x spread in decode/encode tails).
Three levers live here, all behind the stable ``write_video`` surface:

- **uint8 on-GPU before D2H**: a float32 frame chunk crossing PCIe costs 4x
  the bytes of the uint8 the encoder actually consumes (bf16: 2x). Convert
  on-device, transfer 1 byte/px/channel.
- **pinned double-buffer + dedicated copy stream**: chunk N's D2H runs on
  the copy engine while the producer decodes chunk N+1 on the SMs and the
  CPU encodes chunk N-1. Pinned staging keeps the copy DMA-speed; pageable
  and CPU-only inputs degrade to the exact previous behavior.
- **zero-copy handoff**: the pinned buffer is exposed to the encoder as a
  numpy view (no intermediate copy); the encoder wraps it with PyAV's
  ``from_numpy_buffer`` when available (see ``video_encode``).

Frame buffers are transient media staging, NOT model placement — the
GEN_WORKER_FORBID_CPU_OFFLOAD veto (weights) does not apply here.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, Optional, Tuple

from .api.errors import ValidationError

logger = logging.getLogger(__name__)

#: Staging buffers above this stay pageable — pinned host memory is a shared,
#: non-swappable resource (same cap rationale as w8a8_lora's adapter staging).
_PIN_MAX_BYTES = 512 << 20


def _as_torch_cuda(chunk: Any) -> Optional[Any]:
    """The chunk as a CUDA torch tensor, or None for every other input."""
    if not hasattr(chunk, "detach") or not getattr(chunk, "is_cuda", False):
        return None
    return chunk


def gpu_frames_to_uint8(t: Any) -> Any:
    """On-device mirror of ``video_encode.frames_to_uint8`` for torch tensors.

    Returns a contiguous uint8 ``[F, H, W, 3]`` tensor on the same device,
    cropped to even H/W (the encoder's yuv420p rule) so the host array never
    needs a non-contiguous crop. Floats in [0, 1] are scaled; anything else
    is clipped to [0, 255].
    """
    import torch

    t = t.detach()
    if t.ndim == 3 and t.shape[-1] == 3:
        t = t[None]
    if t.ndim != 4 or t.shape[-1] != 3:
        raise ValidationError(
            f"write_video: frames must be [F, H, W, 3], got shape {tuple(t.shape)}"
        )
    h = int(t.shape[1]) - int(t.shape[1]) % 2
    w = int(t.shape[2]) - int(t.shape[2]) % 2
    t = t[:, :h, :w]
    if t.dtype != torch.uint8:
        t = t.float()
        if t.numel() and float(t.amax()) <= 1.0:
            t = t * 255.0
        t = t.round().clamp_(0, 255).to(torch.uint8)
    return t.contiguous()


class _PinnedSlot:
    """One reusable host staging buffer (pinned when the size allows)."""

    def __init__(self) -> None:
        self.buf: Any = None  # flat uint8 CPU tensor
        self.pinned = False

    def ensure(self, nbytes: int, torch: Any) -> None:
        if self.buf is not None and self.buf.numel() >= nbytes:
            return
        pin = nbytes <= _PIN_MAX_BYTES
        try:
            self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
            self.pinned = pin
        except Exception:
            # Pinned alloc can fail on RAM-starved hosts; pageable still works
            # (the copy silently degrades to staged cudaMemcpy speed).
            self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=False)
            self.pinned = False
            logger.warning(
                "frame staging: pinned host alloc failed for %d bytes; "
                "using pageable memory", nbytes)


class _Staged:
    """An in-flight D2H copy: the event to wait on + the host view."""

    __slots__ = ("event", "view", "_device_ref")

    def __init__(self, event: Any, view: Any, device_ref: Any) -> None:
        self.event = event
        self.view = view
        # Keep the device-side uint8 tensor alive until the copy completes.
        self._device_ref = device_ref

    def take(self) -> Any:
        if self.event is not None:
            self.event.synchronize()
        self._device_ref = None
        return self.view


class FrameStager:
    """Double-buffered pinned staging over a dedicated CUDA copy stream.

    ``stage`` enqueues (convert-on-GPU already done by the caller) an async
    D2H copy and returns immediately; ``take`` on the returned handle blocks
    only until THAT copy's event fires. With two slots, the buffer a consumer
    is encoding from is never the one being DMA'd into.
    """

    def __init__(self) -> None:
        self._slots = (_PinnedSlot(), _PinnedSlot())
        self._next = 0
        self._stream: Any = None

    def stage(self, dev_uint8: Any) -> _Staged:
        import torch

        slot = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        nbytes = dev_uint8.numel()
        slot.ensure(nbytes, torch)
        host_flat = slot.buf[:nbytes]
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=dev_uint8.device)
        current = torch.cuda.current_stream(dev_uint8.device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            host_flat.copy_(dev_uint8.view(-1), non_blocking=slot.pinned)
            event = torch.cuda.Event()
            event.record(self._stream)
        view = host_flat.view(dev_uint8.shape).numpy()
        return _Staged(event, view, dev_uint8)


def staged_uint8_chunks(chunks: Iterable[Any]) -> Iterator[Any]:
    """Pipeline an iterator of frame chunks into host uint8 arrays.

    CUDA torch tensor chunks are converted to uint8 on-device (2-4x PCIe byte
    cut) and staged through :class:`FrameStager`, one chunk behind the
    producer — chunk N's D2H overlaps chunk N+1's production and chunk N-1's
    CPU encode. Every other chunk type (numpy, PIL list, CPU tensor) passes
    through unchanged, in order, so the encoder's own coercion applies and
    CPU-only hosts keep the exact previous behavior.

    Each yielded array borrows a reused staging buffer: it is valid until the
    next iteration step (the encoder consumes it synchronously in ``add``).
    """
    stager: Optional[FrameStager] = None
    pending: Optional[_Staged] = None
    for chunk in chunks:
        dev = _as_torch_cuda(chunk)
        if dev is None:
            if pending is not None:
                yield pending.take()
                pending = None
            yield chunk
            continue
        if stager is None:
            stager = FrameStager()
        staged = stager.stage(gpu_frames_to_uint8(dev))
        if pending is not None:
            yield pending.take()
        pending = staged
    if pending is not None:
        yield pending.take()


def cuda_tensor_to_uint8_host(t: Any) -> Tuple[Any, bool]:
    """Buffered-input path: convert a CUDA tensor on-device, one D2H of uint8.

    Returns ``(numpy_array, True)`` when handled, ``(original, False)`` when
    the input is not a CUDA tensor. No pinned staging here: the win for a
    single buffered tensor is the byte cut; a clip-sized pinned buffer is not
    worth the non-swappable host RAM.
    """
    dev = _as_torch_cuda(t)
    if dev is None:
        return t, False
    return gpu_frames_to_uint8(dev).cpu().numpy(), True


__all__ = [
    "FrameStager",
    "cuda_tensor_to_uint8_host",
    "gpu_frames_to_uint8",
    "staged_uint8_chunks",
]
