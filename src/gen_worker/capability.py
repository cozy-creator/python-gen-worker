"""Hardware capability error types for the worker's self-disable signaling.

Raised on the worker's boot-time self-advertise gate and runtime resource
checks; they flow upstream as ``WorkerFunctionUnavailableSignal`` payloads so
the orchestrator narrows subsequent dispatches to suitable workers.
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

    #: Subclasses override with their signal reason vocabulary token.
    reason: str = "hardware_unmet"

    def __init__(self, message: str) -> None:
        super().__init__(message)

    def axes(self) -> dict[str, str]:
        """Axis-specific detail for the orchestrator signal. Override per-subclass."""
        return {}


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


__all__ = [
    "HardwareUnmetError",
    "InsufficientDiskError",
]
