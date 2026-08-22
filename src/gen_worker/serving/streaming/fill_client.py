"""The torch-free data seam into tensorfs's one fill implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class AddressSource:
    """One contiguous source allocation, described without its owner type."""

    pointer: int
    capacity: int


@dataclass(frozen=True, slots=True)
class FileSource:
    """One immutable file range, or a zero-filled hole when ``path`` is None."""

    path: Optional[str]
    offset: int
    length: int


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

    def fill_address(
        self, source: AddressSource, destination: Destination
    ) -> Any: ...

    def fill_files(
        self, sources: Sequence[FileSource], destination: Destination
    ) -> Any: ...


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

    @staticmethod
    def _native() -> Any:
        native = importlib.import_module("tensorfs.native")
        NativeHostFillClient = getattr(native, "HostFillClient")
        return NativeHostFillClient()

    def fill_address(
        self, source: AddressSource, destination: Destination
    ) -> Any:
        return self._native().fill_address(
            source.pointer,
            source.capacity,
            destination.pointer,
            destination.capacity,
            destination.shape,
            destination.element_bytes,
            layout=destination.layout,
        )

    def fill_files(
        self, sources: Sequence[FileSource], destination: Destination
    ) -> Any:
        records = [(source.path, source.offset, source.length) for source in sources]
        return self._native().fill_files(
            records,
            destination.pointer,
            destination.capacity,
            destination.shape,
            destination.element_bytes,
            layout=destination.layout,
        )


class CudaFillClient:
    """One reusable tensorfs-owned pinned slab for all CUDA destinations."""

    staging = "tensorfs-pinned"

    def __init__(self, staging_bytes: int, device: int) -> None:
        native = importlib.import_module("tensorfs.native")
        NativeCudaFillClient = getattr(native, "CudaFillClient")
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

    def fill_address(
        self, source: AddressSource, destination: Destination
    ) -> Any:
        return self._client.fill_address(
            source.pointer,
            source.capacity,
            destination.pointer,
            destination.capacity,
            destination.shape,
            destination.element_bytes,
            layout=destination.layout,
        )

    def fill_files(
        self, sources: Sequence[FileSource], destination: Destination
    ) -> Any:
        records = [(source.path, source.offset, source.length) for source in sources]
        return self._client.fill_files(
            records,
            destination.pointer,
            destination.capacity,
            destination.shape,
            destination.element_bytes,
            layout=destination.layout,
        )


def client_for(device_type: str, *, device_index: int, staging_bytes: int) -> FillClient:
    """Bind the one backend implied by the granted destination device."""

    if device_type == "cpu":
        return HostFillClient()
    if device_type == "cuda":
        return CudaFillClient(staging_bytes, device_index)
    raise ValueError(f"tensorfs fill has no destination backend for {device_type!r}")


__all__ = [
    "AddressSource",
    "Destination",
    "FileSource",
    "FillClient",
    "client_for",
]
