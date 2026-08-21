"""The byte-source seam: what the loader engine needs from tensorfs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

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
    """One tensor container, readable at arbitrary offsets into caller memory."""

    @property
    def tensors(self) -> Sequence[StreamedTensor]: ...
    @property
    def length(self) -> int: ...

    def readinto(self, offset: int, length: int, buffer: Any) -> int:
        """Copy ``[offset, offset+length)`` into a writable C-contiguous buffer — typically CUDA-pinned host memory, which the store neither knows nor cares about."""
        ...


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


class _BridgeStream:

    def __init__(self, reader: Any, container: str) -> None:
        self._reader = reader
        self._container = container
        views = sorted(reader.values(), key=lambda view: view.offset)
        self._tensors: list[StreamedTensor] = list(views)
        self._length = max((v.offset + v.nbytes for v in views), default=0)

    @property
    def tensors(self) -> Sequence[StreamedTensor]:
        return tuple(self._tensors)

    @property
    def length(self) -> int:
        return self._length

    def readinto(self, offset: int, length: int, buffer: Any) -> int:
        target = memoryview(buffer).cast("B")
        if len(target) < length:
            raise ValueError(
                f"{self._container}: buffer holds {len(target)} bytes, "
                f"the read needs {length}"
            )
        at = 0
        for piece in self._reader._pieces(self._container, offset, length):
            target[at : at + len(piece)] = piece
            at += len(piece)
        return at


class BridgeWeightStore:
    """The interim byte source over ``_vendor.tensorfs``'s reader."""

    KIND = "bridge"

    SUFFIXES = TENSOR_SUFFIXES

    def __init__(self, cas: Any, manifest: Any, *, verify: bool = False) -> None:
        self._cas = cas
        self._manifest = manifest
        self._verify = verify
        self._open: list[Any] = []
        self._entries = {
            entry.path: entry
            for entry in manifest.files
            if entry.path.endswith(self.SUFFIXES)
        }

    def containers(self) -> Sequence[str]:
        return tuple(self._entries)

    def open(self, container: str, *, direct: bool = False) -> TensorStream:
        from gen_worker._vendor.tensorfs.manifest import RepositoryManifest
        from gen_worker._vendor.tensorfs.tensors import TensorReader

        if direct:
            raise WeightStoreUnavailable(
                "io=direct needs the native reader's O_DIRECT open "
                "(tensorfs#115); the interim bridge is buffered only"
            )
        entry = self._entries.get(container)
        if entry is None:
            raise WeightStoreUnavailable(
                f"{container!r} is not a tensor container of this manifest"
            )
        reader = TensorReader(
            self._cas,
            RepositoryManifest(files=(entry,)),
            verify=self._verify,
        )
        self._open.append(reader)
        return _BridgeStream(reader, container)

    def close(self) -> None:
        for reader in self._open:
            reader.close()
        self._open.clear()


def store_for(
    checkpoint_dir: Path | str, *, native: Optional[bool] = None
) -> Optional[WeightStore]:
    """The byte source backing a projected checkpoint tree, or ``None``."""
    from ...models import projection

    projected = projection.resolve_projection(checkpoint_dir)
    if projected is None:
        return None
    if native is None:
        native = native_available()
    if native:
        try:
            return NativeWeightStore.from_manifest(
                projected.cas.root, projected.manifest
            )
        except Exception:
            logger.warning(
                "ctx.load: the native tensorfs store would not open over %s "
                "— falling back to the GIL-bound bridge, which is ~10x "
                "slower (tensorfs#115)",
                projected.cas.root,
                exc_info=True,
            )
    return BridgeWeightStore(projected.cas, projected.manifest)


def native_available() -> bool:
    """True when a tensorfs carrying the #115 stream surface is importable."""
    try:
        _native_stream_reader()
    except (WeightStoreUnavailable, AttributeError):
        return False
    return True


def component_of(container: str) -> str:
    """The pipeline component a container belongs to: its parent directory, or ``""`` for a container at the checkpoint root (single-file pipelines)."""
    parent = Path(container).parent
    return "" if str(parent) == "." else str(parent)


__all__ = [
    "TENSOR_PLANNERS",
    "TENSOR_SUFFIXES",
    "BridgeWeightStore",
    "NativeWeightStore",
    "StreamedTensor",
    "TensorStream",
    "WeightStore",
    "WeightStoreUnavailable",
    "component_of",
    "native_available",
    "store_for",
]
