"""Hardware capability error types for the worker's self-disable signaling.

The public ``require_vram`` / ``require_compute_capability`` / ``require_cuda_library``
helpers were removed in gen-worker 0.7.0 (decorator-table-model-bindings) —
the per-function :class:`gen_worker.Resources` envelope is now the single
declarative source of truth, checked by the worker at boot.

The error types below remain because :mod:`gen_worker.worker` still raises
them internally when its boot-time self-advertise gate runs (and during
runtime fallbacks for resources that can't be known statically). They flow
upstream as ``WorkerFunctionUnavailableSignal`` payloads so the orchestrator
narrows subsequent dispatches to suitable workers.

Migration::

    # Before (0.6.x):
    from gen_worker.capability import require_vram, require_compute_capability

    @inference(resources=ResourceRequirements(...))
    def generate_bf16(ctx, payload):
        require_vram(22 * 1024**3)
        require_compute_capability((8, 0))
        ...

    # After (0.7.0):
    from gen_worker import Resources, inference_function

    _flux_bf16 = Resources(requires_gpu=True, min_vram_gb=22.0, min_compute_capability=8.0)

    @inference(resources=_flux_bf16)
    def generate_bf16(ctx, payload):
        # Worker boot-time self-advertise checks _flux_bf16 against host
        # hardware and marks this function unavailable on hosts that can't
        # satisfy it. No runtime check needed.
        ...
"""

from __future__ import annotations


class HardwareUnmetError(RuntimeError):
    """Base class for hardware-axis gates the worker cannot satisfy.

    Recognised by :mod:`gen_worker.worker` as a terminal 'disable this
    function on this worker' signal. Subclasses carry axis-specific fields;
    :meth:`axes` returns the key/value map the
    ``WorkerFunctionUnavailableSignal`` proto embeds as free-form hardware
    specifics.
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
    CUDA minor version mismatch. Not resolvable at runtime — the only fix is
    a hardware / image swap.
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


_REMOVED_HELPERS = {
    "require_vram": (
        "gen_worker.capability.require_vram was removed in gen-worker 0.7.0. "
        "Declare min_vram_gb on the per-function Resources struct instead: "
        "Resources(requires_gpu=True, min_vram_gb=22.0). The worker checks "
        "host VRAM against the declared envelope at boot and self-disables "
        "functions whose hosts can't satisfy them."
    ),
    "require_compute_capability": (
        "gen_worker.capability.require_compute_capability was removed in "
        "gen-worker 0.7.0. Declare min_compute_capability on the per-function "
        "Resources struct instead: Resources(requires_gpu=True, "
        "min_compute_capability=10.0). The worker checks host SM at boot."
    ),
    "require_cuda_library": (
        "gen_worker.capability.require_cuda_library was removed in "
        "gen-worker 0.7.0. Declare required_libraries on the per-function "
        "Resources struct instead: Resources(required_libraries=(\"flash_attn\",))."
    ),
}


def __getattr__(name: str):
    if name in _REMOVED_HELPERS:
        raise ImportError(_REMOVED_HELPERS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HardwareUnmetError",
    "ComputeCapabilityUnmetError",
    "InsufficientVRAMError",
    "MissingCudaLibraryError",
    "InsufficientDiskError",
]
