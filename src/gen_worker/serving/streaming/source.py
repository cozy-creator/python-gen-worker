"""The byte-source seam: what the loader engine needs from tensorfs.

The torch boundary is a ruling (tensorfs#115 item 4): **tensorfs yields bytes
and geometry; gen-worker owns torch, CUDA and the pinned pool.** These
Protocols are that boundary written down on this side of it, and they are the
shape tensorfs#115 landed — ``TensorStreamReader`` satisfies
:class:`TensorStream` structurally, with no adapter code at all.

Two stores implement :class:`WeightStore`:

* :class:`NativeWeightStore` over the maturin wheel's
  ``tensorfs.native.{ObjectStore, Snapshot, TensorStreamReader}``. This is the
  production path: ordered iteration is an API contract, ``readinto`` copies
  in Rust with the GIL released, and ``direct=True`` opens ``O_DIRECT``.
* :class:`BridgeWeightStore` over the storage plane pgw actually carries today
  (``_vendor.tensorfs``'s ``LocalCAS`` + ``TensorReader``). Same real store,
  same real chunked CAS objects, same file-order contract — it just copies
  under the GIL, which is precisely the ~10x tensorfs#115 measured away.

**The bridge dies with tensorfs#57**, the wheel-publish lane, and nothing else
in the engine moves when it does: the engine only ever speaks
:class:`WeightStore`.
"""

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

#: Container formats whose tensors this engine can stream. A snapshot entry
#: planned any other way (``blob-v1``) is not a tensor container and is left
#: to the projected tree, where it is a symlink costing nothing.
TENSOR_PLANNERS = frozenset({"safetensors-v1", "gguf-v1"})

#: The same fact spelled for a TFM1 manifest, which carries no planner: a
#: vendored ``FileEntry`` says only where the bytes are, so the suffix is all
#: there is to ask. Used on both sides so one store never sees a container
#: the other would skip.
TENSOR_SUFFIXES = (".safetensors", ".gguf")


class StreamedTensor(Protocol):
    """One tensor's geometry: the container's own dtype spelling, its shape,
    and its byte span **absolute within the container file**."""

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
    """One tensor container, readable at arbitrary offsets into caller memory.

    ``tensors`` is **ascending file offset by contract** — not by accident,
    and not safetensors header order, which is a different order in general.
    Walking ``tensors`` and reading each in turn is the sequential access
    pattern the load-order ruling measured at ~10x, and it is the ONLY order
    this engine ever reads in.
    """

    @property
    def tensors(self) -> Sequence[StreamedTensor]: ...
    @property
    def length(self) -> int: ...

    def readinto(self, offset: int, length: int, buffer: Any) -> int:
        """Copy ``[offset, offset+length)`` into a writable C-contiguous
        buffer — typically CUDA-pinned host memory, which the store neither
        knows nor cares about. Holes zero-fill, never skip."""
        ...


class WeightStore(Protocol):
    """The checkpoint's tensor containers, addressable without a file."""

    def containers(self) -> Sequence[str]:
        """Checkpoint-relative paths of every tensor container, in a stable
        order. Non-tensor files are not listed: they stay symlinks in the
        projected tree and are read from there."""
        ...

    def open(self, container: str, *, direct: bool = False) -> TensorStream: ...


class WeightStoreUnavailable(RuntimeError):
    """No streamed byte source could be built for this checkpoint."""


# -- the native store (tensorfs#115, via the tensorfs#57 wheel) -------------


def _native_module() -> Any:
    """:mod:`tensorfs.native`, resolved at call time.

    Resolved through :mod:`importlib` rather than a module-level import
    because ``tensorfs`` is not yet a declared dependency of this package
    (tensorfs#57 is the wheel-publish lane). A static import would make every
    torchless boot pay an ImportError for a module only the serving path uses.
    """
    import importlib

    try:
        return importlib.import_module("tensorfs.native")
    except ImportError as exc:
        raise WeightStoreUnavailable(
            "the tensorfs wheel (tensorfs#57) is not installed, so the "
            "native store->VRAM stream surface is unavailable in this image"
        ) from exc


def _native_stream_reader() -> Any:
    """``tensorfs.native.TensorStreamReader``, or a loud refusal."""
    return getattr(_native_module(), "TensorStreamReader")


class NativeWeightStore:
    """The production byte source: the Rust extension's stream reader.

    Constructed by the WORKER from the store and snapshot it already resolved
    — the engine never opens a store of its own, because which store and
    which snapshot serve is a worker decision, exactly as placement is.
    """

    #: What a `LoadReport` calls this plane, so a GB/s number is never read
    #: as the other one's.
    KIND = "native"

    def __init__(self, store: Any, records: Mapping[str, Sequence[Any]]) -> None:
        self._store = store
        self._records = dict(records)

    @classmethod
    def from_manifest(cls, root: Path | str, manifest: Any) -> "NativeWeightStore":
        """The native reader over the store the worker ALREADY wrote.

        The two planes share their on-disk layout exactly —
        ``objects/sha256/aa/bb/<hex>`` — so nothing is copied, converted or
        re-ingested here: the same chunk objects are simply mapped by Rust
        instead of by Python. Only the manifest spelling differs, and a
        record run is the whole of what the reader wanted from it.

        This is the path production actually takes. ``from_snapshot`` is for a
        store whose refs are native snapshots; the worker's projection plane
        still writes TFM1 manifests, and it is the manifest that resolves.
        """
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
        """Every tensor container of a committed snapshot, in snapshot order.

        A snapshot entry's ``planner`` is what says a file is a tensor
        container: the contract that planned it. Nothing here sniffs a
        suffix, because the planner is the fact and the suffix is a habit.
        """
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


# -- the bridge over the plane pgw carries today (dies with tensorfs#57) ----


class _BridgeStream:
    """One container of a :class:`BridgeWeightStore`.

    Geometry comes from the vendored ``TensorReader``'s own header parse, so
    the names, dtypes, shapes and offsets are the SAME facts the native
    reader reports; only the copy is Python's. Ordering is imposed here for
    the same reason the Rust reader imposes it: header key order is not
    offset order.

    ONE container per reader, deliberately. The vendored reader indexes a
    whole manifest into a flat name->view map and refuses a name that appears
    twice — and in a pipeline checkpoint ``text_encoder`` and
    ``text_encoder_2`` share every name they have. Names are only unique
    WITHIN a container, which is also the only scope this engine ever
    resolves them in.
    """

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
        # `_pieces` rather than `read_range` on purpose: `read_range` returns
        # `bytes`, so every window would allocate and copy a second time. The
        # pieces are zero-copy slices of the mapped CAS objects, which is the
        # closest this plane gets to what the Rust reader does natively.
        for piece in self._reader._pieces(self._container, offset, length):
            target[at : at + len(piece)] = piece
            at += len(piece)
        return at


class BridgeWeightStore:
    """The interim byte source over ``_vendor.tensorfs``'s reader.

    NOT a stand-in for missing code and NOT a mock: it reads the same chunked
    CAS objects the fleet stores today, through the same header parse. What
    it cannot carry is the SPEED — the copy happens in Python under the GIL,
    which is the path tensorfs#115 measured at 0.59 GiB/s against the native
    reader's 6.4 GiB/s. Delete this class the day tensorfs#57 publishes a
    wheel; :class:`NativeWeightStore` already does everything it does.
    """

    KIND = "bridge"

    #: What the vendored reader knows how to parse a header out of.
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
        # Manifest order, which `RepositoryManifest` sorts by path — the same
        # stable packaging order the native snapshot reports.
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
    """The byte source backing a projected checkpoint tree, or ``None``.

    ``None`` means the tree is not a snapshot this worker projected — a bare
    HF download, a materialized directory, a fixture. There is no chunk store
    behind it to stream from, so ``ctx.load`` has nothing to do here and the
    caller keeps whatever path it had.

    The resolution is the same one :mod:`gen_worker.models.projection` already
    performs for every other consumer, so the engine needs no store handle
    plumbed to it: the DIRECTORY the worker resolved carries its own store.

    **Which store is a CAPABILITY question, not a configuration one.** The
    native reader is strictly better at the one job both stores have, so the
    only thing worth asking is whether this image carries it — and se#756
    ships a real ``tensorfs`` to the endpoint image, so in production it does.
    ``native=`` forces the answer for a caller that wants to measure one plane
    against the other; nothing reads an environment variable to decide.
    """
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
            # The bridge reads the SAME objects, so a native store that will
            # not build costs speed and nothing else. Serving a request slowly
            # beats refusing it — but say so loudly, because a silent fallback
            # is how a ~10x regression hides.
            logger.warning(
                "ctx.load: the native tensorfs store would not open over %s "
                "— falling back to the GIL-bound bridge, which is ~10x "
                "slower (tensorfs#115)",
                projected.cas.root,
                exc_info=True,
            )
    return BridgeWeightStore(projected.cas, projected.manifest)


def native_available() -> bool:
    """True when a tensorfs carrying the #115 stream surface is importable.

    Presence of the PACKAGE is not the question — pgw already resolves one
    transitively through torchcg, pinned below #115. The question is whether
    THIS build carries ``TensorStreamReader``, so the check asks for the
    class rather than for the import.
    """
    try:
        _native_stream_reader()
    except (WeightStoreUnavailable, AttributeError):
        return False
    return True


def component_of(container: str) -> str:
    """The pipeline component a container belongs to: its parent directory,
    or ``""`` for a container at the checkpoint root (single-file pipelines)."""
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
