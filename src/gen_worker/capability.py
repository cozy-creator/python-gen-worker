"""Hardware capability gates — worker self-disables a function on unmet hardware.

When a tenant function needs a hardware resource that varies worker-to-worker
(compute capability / SM level, VRAM, disk, CUDA libraries, ...), the first
attempt on an unsuitable worker raises a ``HardwareUnmetError`` subclass.
``gen_worker.worker`` classifies that as a function-level self-disable
terminal — the job fails fast (no retry on the same worker), the worker
emits a ``WorkerFunctionUnavailableSignal`` upstream, and the orchestrator
narrows subsequent dispatches for that function to workers where the gate
has not triggered.

This is the general mechanism behind e2e issue #40. Initial consumers:
- ``conversion-gpu.modelopt_quantization`` — ``require_compute_capability((10, 0))``
  for ``scheme=nvfp4`` (Blackwell only), ``(9, 0)`` for ``scheme=fp8`` (Hopper+).
- Inference endpoints with large-model functions — ``require_vram(N * 1024**3)``
  at function entry so a function declaring ``required_vram_gb=28`` refuses to
  run on a 24 GB GPU while sibling 20 GB functions on the same endpoint stay
  dispatchable.

Error → signal → orchestrator flow is driven by ``gen_worker.worker``; tenants
just call the ``require_*`` helpers at the top of their function body.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class HardwareUnmetError(RuntimeError):
    """Base class for hardware-axis gates the worker cannot satisfy.

    Recognised by ``gen_worker.worker`` as a terminal 'disable this function
    on this worker' signal. Subclasses carry axis-specific fields; the
    ``axes()`` method returns the key/value map the ``WorkerFunctionUnavailableSignal``
    proto embeds as free-form hardware specifics.
    """

    #: One of: compute_capability_unmet, insufficient_vram, missing_cuda_library,
    #: insufficient_disk. Subclasses override.
    reason: str = "hardware_unmet"

    def __init__(self, message: str) -> None:
        super().__init__(message)

    def axes(self) -> dict[str, str]:
        """Axis-specific detail for the orchestrator signal. Override per-subclass."""
        return {}


class ComputeCapabilityUnmetError(HardwareUnmetError):
    """GPU compute capability (SM level) is below the function's requirement."""

    reason = "compute_capability_unmet"

    def __init__(
        self,
        message: str,
        *,
        detected_sm: str = "",
        required_sm: str = "",
    ) -> None:
        super().__init__(message)
        self.detected_sm = detected_sm
        self.required_sm = required_sm

    def axes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.detected_sm:
            out["detected_sm"] = self.detected_sm
        if self.required_sm:
            out["required_sm"] = self.required_sm
        return out


class InsufficientVRAMError(HardwareUnmetError):
    """GPU VRAM is smaller than the function declares it needs."""

    reason = "insufficient_vram"

    def __init__(
        self,
        message: str,
        *,
        available_bytes: int = 0,
        required_bytes: int = 0,
    ) -> None:
        super().__init__(message)
        self.available_bytes = int(available_bytes)
        self.required_bytes = int(required_bytes)

    def axes(self) -> dict[str, str]:
        return {
            "available_bytes": str(self.available_bytes),
            "required_bytes": str(self.required_bytes),
        }


class MissingCudaLibraryError(HardwareUnmetError):
    """A required CUDA library / kernel module is not available on this host.

    Typical reasons: ``flash_attn`` wheel not built for this GPU family, or a
    CUDA minor version mismatch. Not resolvable at runtime — the only fix is a
    hardware / image swap.
    """

    reason = "missing_cuda_library"

    def __init__(
        self,
        message: str,
        *,
        library: str = "",
        min_version: str = "",
    ) -> None:
        super().__init__(message)
        self.library = library
        self.min_version = min_version

    def axes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.library:
            out["library"] = self.library
        if self.min_version:
            out["min_version"] = self.min_version
        return out


class InsufficientDiskError(HardwareUnmetError):
    """Host disk free space is below the function's declared requirement."""

    reason = "insufficient_disk"

    def __init__(
        self,
        message: str,
        *,
        available_bytes: int = 0,
        required_bytes: int = 0,
        path: str = "",
    ) -> None:
        super().__init__(message)
        self.available_bytes = int(available_bytes)
        self.required_bytes = int(required_bytes)
        self.path = path

    def axes(self) -> dict[str, str]:
        out = {
            "available_bytes": str(self.available_bytes),
            "required_bytes": str(self.required_bytes),
        }
        if self.path:
            out["path"] = self.path
        return out


# ---------------------------------------------------------------------------
# require_* helpers — call at the top of a @training_function / @inference_function
# body. On unmet they raise the matching error; the worker classifies it as a
# self-disable terminal and emits the upstream signal.


def require_compute_capability(minimum: tuple[int, int]) -> None:
    """Raise ``ComputeCapabilityUnmetError`` if host GPU SM < ``minimum``.

    ``minimum`` is ``(major, minor)``, e.g. ``(10, 0)`` for Blackwell SM 10.0,
    ``(9, 0)`` for Hopper SM 9.0, ``(8, 9)`` for Ada SM 8.9. No-op if the
    minimum is satisfied.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dep in practice
        raise MissingCudaLibraryError(
            "torch is not importable; cannot check compute capability",
            library="torch",
        ) from exc

    if not torch.cuda.is_available():
        raise ComputeCapabilityUnmetError(
            f"cuda_unavailable (function requires SM {minimum[0]}.{minimum[1]}+)",
            detected_sm="none",
            required_sm=f"{minimum[0]}.{minimum[1]}",
        )

    cap = torch.cuda.get_device_capability(0)
    if cap < minimum:
        raise ComputeCapabilityUnmetError(
            f"host SM {cap[0]}.{cap[1]} < required SM {minimum[0]}.{minimum[1]}",
            detected_sm=f"{cap[0]}.{cap[1]}",
            required_sm=f"{minimum[0]}.{minimum[1]}",
        )


def require_vram(required_bytes: int) -> None:
    """Raise ``InsufficientVRAMError`` if GPU 0's total VRAM is less than ``required_bytes``.

    ``required_bytes`` is the function's declared footprint at load time —
    e.g. 22 GB for flux.2-klein-4b bf16. Worker uses the primary device's
    total memory (``torch.cuda.get_device_properties(0).total_memory``); this
    ignores other running processes and accelerate-style sharding, which is
    intentional: the check is about the function's hardware gate, not
    moment-to-moment memory pressure.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise MissingCudaLibraryError(
            "torch is not importable; cannot check vram",
            library="torch",
        ) from exc

    if not torch.cuda.is_available():
        raise InsufficientVRAMError(
            f"cuda_unavailable (function requires {required_bytes} bytes VRAM)",
            available_bytes=0,
            required_bytes=int(required_bytes),
        )

    available = int(torch.cuda.get_device_properties(0).total_memory)
    if available < int(required_bytes):
        raise InsufficientVRAMError(
            f"host VRAM {available} < required {required_bytes}",
            available_bytes=available,
            required_bytes=int(required_bytes),
        )


def require_cuda_library(name: str, *, min_version: str = "") -> None:
    """Raise ``MissingCudaLibraryError`` if the named Python package is not importable.

    Narrow check: attempts ``importlib.import_module(name)``. Version comparison
    is a best-effort string compare against ``module.__version__`` if
    ``min_version`` is given — callers needing strict semver comparison should
    layer their own check on top.
    """
    import importlib

    try:
        mod: Any = importlib.import_module(name)
    except ImportError as exc:
        raise MissingCudaLibraryError(
            f"cuda library {name!r} is not importable on this host",
            library=name,
            min_version=min_version,
        ) from exc

    if not min_version:
        return
    got = str(getattr(mod, "__version__", "") or "")
    if got and got < min_version:
        raise MissingCudaLibraryError(
            f"cuda library {name!r} version {got!r} < required {min_version!r}",
            library=name,
            min_version=min_version,
        )


def require_disk(required_bytes: int, *, path: str | Path = "/") -> None:
    """Raise ``InsufficientDiskError`` if free disk bytes at ``path`` < ``required_bytes``."""
    import shutil

    p = str(path)
    try:
        usage = shutil.disk_usage(p)
    except OSError as exc:
        raise InsufficientDiskError(
            f"cannot stat disk at {p!r}: {exc}",
            available_bytes=0,
            required_bytes=int(required_bytes),
            path=p,
        ) from exc

    if usage.free < int(required_bytes):
        raise InsufficientDiskError(
            f"disk free {usage.free} < required {required_bytes} at {p!r}",
            available_bytes=int(usage.free),
            required_bytes=int(required_bytes),
            path=p,
        )


__all__ = [
    "HardwareUnmetError",
    "ComputeCapabilityUnmetError",
    "InsufficientVRAMError",
    "MissingCudaLibraryError",
    "InsufficientDiskError",
    "require_compute_capability",
    "require_vram",
    "require_cuda_library",
    "require_disk",
]
