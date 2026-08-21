"""The torch-free data seam into tensorfs's one fill implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple


@dataclass(frozen=True, slots=True)
class Destination:
    """Where one named tensor lands; every field is plain data."""

    name: str
    pointer: int
    capacity: int
    source_offset: int
    shape: Tuple[int, ...]
    element_bytes: int
    layout: str


class FillClient(Protocol):
    """Fill destinations without learning which allocator produced them."""

    staging: str

    def fill(self, reader: Any, destination: Destination) -> Any: ...


class HostFillClient:
    """The host destination is its own staging allocation."""

    staging = "destination"

    def fill(self, reader: Any, destination: Destination) -> Any:
        return reader.fill_host_address(
            destination.name,
            destination.pointer,
            destination.capacity,
            layout=destination.layout,
        )


class CudaFillClient:
    """One reusable tensorfs-owned pinned slab for all CUDA destinations."""

    staging = "tensorfs-pinned"

    def __init__(self, staging_bytes: int, device: int) -> None:
        from tensorfs.native import CudaFillClient as NativeCudaFillClient

        self._client = NativeCudaFillClient(staging_bytes, device)

    def fill(self, reader: Any, destination: Destination) -> Any:
        native_reader = getattr(reader, "native_reader", reader)
        return self._client.fill(
            native_reader,
            destination.name,
            destination.pointer,
            destination.capacity,
            layout=destination.layout,
        )


def client_for(device_type: str, *, device_index: int, staging_bytes: int) -> FillClient:
    """Bind the one backend implied by the granted destination device."""

    if device_type == "cpu":
        return HostFillClient()
    if device_type == "cuda":
        return CudaFillClient(staging_bytes, device_index)
    raise ValueError(f"tensorfs fill has no destination backend for {device_type!r}")


__all__ = ["Destination", "FillClient", "client_for"]
