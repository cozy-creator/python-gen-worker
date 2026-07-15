"""Typed worker capacity errors.

Static hardware gates flow upstream as ``WorkerFunctionUnavailableSignal``
payloads so the orchestrator narrows subsequent dispatches to suitable
workers. Transient capacity pressure remains retryable and does not disable a
function on the worker.
"""

from __future__ import annotations

from .api.errors import RetryableError


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


class InsufficientHostRamError(RetryableError):
    """A worker cannot stage a model without crossing its host-RAM floor.

    Unlike a static hardware gate, this is transient: another worker may have
    room and this worker may recover after an idle record is evicted.  Keep the
    measured capacity facts typed so the scheduler can distinguish local
    pressure from a corrupt model load.
    """

    reason = "insufficient_host_ram"

    def __init__(
        self,
        function_name: str,
        *,
        incoming_bytes: int,
        floor_bytes: int,
        required_bytes: int,
        available_before_bytes: int,
        available_after_bytes: int,
        evicted_refs: tuple[str, ...] = (),
    ) -> None:
        self.function_name = str(function_name)
        self.incoming_bytes = int(incoming_bytes)
        self.floor_bytes = int(floor_bytes)
        self.required_bytes = int(required_bytes)
        self.available_before_bytes = int(available_before_bytes)
        self.available_after_bytes = int(available_after_bytes)
        self.evicted_refs = tuple(evicted_refs)

        gib = float(1024**3)
        evicted = ",".join(self.evicted_refs) or "none"
        super().__init__(
            f"insufficient host RAM to load {self.function_name}: "
            f"~{self.incoming_bytes / gib:.1f}GiB incoming + "
            f"{self.floor_bytes / gib:.1f}GiB safety floor = "
            f"{self.required_bytes / gib:.1f}GiB required; "
            f"{self.available_before_bytes / gib:.1f}GiB available before eviction, "
            f"{self.available_after_bytes / gib:.1f}GiB after; "
            f"evicted_refs={evicted}"
        )

    def axes(self) -> dict[str, str]:
        return {
            "incoming_bytes": str(self.incoming_bytes),
            "floor_bytes": str(self.floor_bytes),
            "required_bytes": str(self.required_bytes),
            "available_before_bytes": str(self.available_before_bytes),
            "available_after_bytes": str(self.available_after_bytes),
            "evicted_refs": ",".join(self.evicted_refs),
        }


__all__ = [
    "HardwareUnmetError",
    "InsufficientDiskError",
    "InsufficientHostRamError",
]
