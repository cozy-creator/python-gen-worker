"""Content-addressed local storage, chunk manifests, and direct tensor reads."""

from .local import DigestMismatch, LocalCAS, RefConflict
from .manifest import MAX_CHUNK_SIZE, Chunk, FileEntry, RepositoryManifest
from .project import (
    STUB_MAGIC,
    TENSOR_SUFFIXES,
    ProjectionError,
    Stub,
    is_tensor_container,
    parse_stub,
    project_snapshot,
    read_stub,
    stub_bytes,
    tree_bytes,
)
from .refs import CASRef
from .tensors import (
    DTYPE_BITS,
    BlockLayout,
    FileTooLarge,
    TensorError,
    TensorReader,
    TensorView,
    dtype_itemsize,
    open_tensors,
    read_entry,
)

__all__ = [
    "CASRef",
    "Chunk",
    "DTYPE_BITS",
    "DigestMismatch",
    "BlockLayout",
    "FileEntry",
    "FileTooLarge",
    "LocalCAS",
    "MAX_CHUNK_SIZE",
    "ProjectionError",
    "RefConflict",
    "RepositoryManifest",
    "STUB_MAGIC",
    "Stub",
    "TENSOR_SUFFIXES",
    "TensorError",
    "TensorReader",
    "TensorView",
    "dtype_itemsize",
    "is_tensor_container",
    "open_tensors",
    "parse_stub",
    "project_snapshot",
    "read_entry",
    "read_stub",
    "stub_bytes",
    "tree_bytes",
]
