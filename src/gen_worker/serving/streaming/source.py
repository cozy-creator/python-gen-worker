"""The byte-source seam: what the loader engine needs from tensorfs."""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

TENSOR_PLANNERS = frozenset({"safetensors-v1", "gguf-v1"})

TENSOR_SUFFIXES = (".safetensors", ".gguf")


class StreamedTensor(Protocol):
    """One tensor's geometry: the container's own dtype spelling, its shape, and its byte span **absolute within the container file**."""

    @property
    def name(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> Sequence[int]: ...
    @property
    def offset(self) -> int: ...
    @property
    def nbytes(self) -> int: ...


@runtime_checkable
class TensorStream(Protocol):
    """One tensor container consumable by tensorfs's fill path."""

    @property
    def tensors(self) -> Sequence[StreamedTensor]: ...
    @property
    def length(self) -> int: ...

    def fill_host_address(
        self,
        name: str,
        destination_ptr: int,
        destination_bytes: int,
        destination_offset: int = 0,
        layout: str = "torch.contiguous@1",
    ) -> Any: ...


class WeightStore(Protocol):
    """The checkpoint's tensor containers, addressable without a file."""

    def containers(self) -> Sequence[str]:
        """Checkpoint-relative paths of every tensor container, in a stable order."""
        ...

    def open(self, container: str, *, direct: bool = False) -> TensorStream: ...


class WeightStoreUnavailable(RuntimeError):
    """No streamed byte source could be built for this checkpoint."""


def _native_module() -> Any:
    import importlib

    try:
        return importlib.import_module("tensorfs.native")
    except ImportError as exc:
        raise WeightStoreUnavailable(
            "the tensorfs wheel (tensorfs#57) is not installed, so the "
            "native store->VRAM stream surface is unavailable in this image"
        ) from exc


def _native_stream_reader() -> Any:
    return getattr(_native_module(), "TensorStreamReader")


class NativeWeightStore:
    """The production byte source: the Rust extension's stream reader."""

    KIND = "native"

    def __init__(self, store: Any, records: Mapping[str, Sequence[Any]]) -> None:
        self._store = store
        self._records = dict(records)

    @classmethod
    def from_manifest(cls, root: Path | str, manifest: Any) -> "NativeWeightStore":
        """The native reader over the store the worker ALREADY wrote."""
        native = _native_module()
        records: dict[str, Sequence[Any]] = {}
        for entry in manifest.files:
            if not entry.path.endswith(TENSOR_SUFFIXES):
                continue
            records[entry.path] = [
                native.FileRecord.data(ref.digest, length)
                for ref, length in entry.objects()
            ]
        return cls(native.ObjectStore(Path(root)), records)

    @classmethod
    def from_snapshot(cls, store: Any, snapshot: Any) -> "NativeWeightStore":
        """Every tensor container of a committed snapshot, in snapshot order."""
        records: dict[str, Sequence[Any]] = {}
        for entry in snapshot.entries:
            if getattr(entry, "planner", None) not in TENSOR_PLANNERS:
                continue
            run = snapshot.file_records(entry.path)
            if run is None:
                raise WeightStoreUnavailable(
                    f"{entry.path!r} is planned {entry.planner!r} but the "
                    f"snapshot carries no record run for it"
                )
            records[entry.path] = run
        return cls(store, records)

    def containers(self) -> Sequence[str]:
        return tuple(self._records)

    def open(self, container: str, *, direct: bool = False) -> TensorStream:
        run = self._records.get(container)
        if run is None:
            raise WeightStoreUnavailable(
                f"{container!r} is not a tensor container of this snapshot"
            )
        reader: TensorStream = _native_stream_reader()(self._store, run, direct)
        return reader


def store_for(checkpoint_dir: Path | str) -> WeightStore | None:
    """The byte source backing a projected checkpoint tree, or ``None``."""
    from ...models import projection

    projected = projection.resolve_projection(checkpoint_dir)
    if projected is None:
        return None
    return NativeWeightStore.from_manifest(projected.cas.root, projected.manifest)


def component_of(container: str) -> str:
    """The pipeline component a container belongs to: its parent directory, or ``""`` for a container at the checkpoint root (single-file pipelines)."""
    parent = Path(container).parent
    return "" if str(parent) == "." else str(parent)


__all__ = [
    "TENSOR_PLANNERS",
    "TENSOR_SUFFIXES",
    "NativeWeightStore",
    "StreamedTensor",
    "TensorStream",
    "WeightStore",
    "WeightStoreUnavailable",
    "component_of",
    "store_for",
]
