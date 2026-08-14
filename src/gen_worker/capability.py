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


class _HostRamFacts:
    """The measured facts behind one host-RAM admission refusal.

    Shared by the transient and the structural verdict so both speak one
    vocabulary on the wire; each carries its own error base because the two
    demand opposite dispatch behaviour.
    """

    #: Message stem; the terminal form names the structural verdict.
    _headline = "insufficient host RAM"

    def _describe(
        self,
        function_name: str,
        *,
        incoming_bytes: int,
        floor_bytes: int,
        required_bytes: int,
        available_before_bytes: int,
        available_after_bytes: int,
        evicted_refs: tuple[str, ...],
        total_bytes: int,
    ) -> str:
        self.function_name = str(function_name)
        self.incoming_bytes = int(incoming_bytes)
        self.floor_bytes = int(floor_bytes)
        self.required_bytes = int(required_bytes)
        self.available_before_bytes = int(available_before_bytes)
        self.available_after_bytes = int(available_after_bytes)
        self.evicted_refs = tuple(evicted_refs)
        self.total_bytes = int(total_bytes)

        gib = float(1024**3)
        evicted = ",".join(self.evicted_refs) or "none"
        total = (
            f"; host total {self.total_bytes / gib:.1f}GiB"
            if self.total_bytes else ""
        )
        return (
            f"{self._headline} to load {self.function_name}: "
            f"~{self.incoming_bytes / gib:.1f}GiB incoming + "
            f"{self.floor_bytes / gib:.1f}GiB safety floor = "
            f"{self.required_bytes / gib:.1f}GiB required; "
            f"{self.available_before_bytes / gib:.1f}GiB available before eviction, "
            f"{self.available_after_bytes / gib:.1f}GiB after; "
            f"evicted_refs={evicted}{total}"
        )

    def axes(self) -> dict[str, str]:
        return {
            "incoming_bytes": str(self.incoming_bytes),
            "floor_bytes": str(self.floor_bytes),
            "required_bytes": str(self.required_bytes),
            "available_before_bytes": str(self.available_before_bytes),
            "available_after_bytes": str(self.available_after_bytes),
            "evicted_refs": ",".join(self.evicted_refs),
            "total_bytes": str(self.total_bytes),
        }


class InsufficientHostRamError(_HostRamFacts, RetryableError):
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
        total_bytes: int = 0,
    ) -> None:
        super().__init__(self._describe(
            function_name,
            incoming_bytes=incoming_bytes,
            floor_bytes=floor_bytes,
            required_bytes=required_bytes,
            available_before_bytes=available_before_bytes,
            available_after_bytes=available_after_bytes,
            evicted_refs=tuple(evicted_refs),
            total_bytes=total_bytes,
        ))


class HostRamCapacityError(_HostRamFacts, HardwareUnmetError):
    """This pod SIZE can never stage the model — an empty host is too small.

    The retryable form says "another worker may have room", which is true only
    while the shortfall is pressure. When the requirement exceeds the host's
    TOTAL RAM, every retry re-buys an identically-sized pod and refuses
    identically. Reporting a hardware axis instead disables the function on
    this worker and hands the
    orchestrator the exact placement fact (required vs total) it needs to pick
    a bigger host.
    """

    reason = "host_ram_capacity"
    _headline = "host RAM capacity too small"

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
        total_bytes: int = 0,
    ) -> None:
        super().__init__(self._describe(
            function_name,
            incoming_bytes=incoming_bytes,
            floor_bytes=floor_bytes,
            required_bytes=required_bytes,
            available_before_bytes=available_before_bytes,
            available_after_bytes=available_after_bytes,
            evicted_refs=tuple(evicted_refs),
            total_bytes=total_bytes,
        ))


#: Both host-RAM admission verdicts — one transient, one structural.
HOST_RAM_REFUSALS = (InsufficientHostRamError, HostRamCapacityError)


__all__ = [
    "HOST_RAM_REFUSALS",
    "HardwareUnmetError",
    "HostRamCapacityError",
    "InsufficientDiskError",
    "InsufficientHostRamError",
]
