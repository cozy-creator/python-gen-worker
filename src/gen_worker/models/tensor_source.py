"""One way to read tensors, whether or not their bytes are at a file path.

``safetensors.safe_open(path)`` is the shape every weight-loading lane in this
repo was written against, and it is exactly the shape a projected tree cannot
serve: the bytes are CAS objects, and the path holds a ~128 B pointer stub.
:func:`open_tensor_source` keeps the shape and moves the source — ``keys()``,
``get_tensor()`` and ``metadata()`` behave the same either way, so a lane is
cut over by changing one ``with`` line rather than by being rewritten.

The native arm is not a fallback; on a projected tree it is the ONLY arm, and
a lane that reaches this module with a stub and no recoverable manifest gets a
:class:`~gen_worker.models.projection.UnresolvedProjection` refusal instead of
a wrong model. What it must never do is what master did — read the stub's
first eight bytes as a header length, fail, and carry on with a default.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Protocol

from gen_worker._vendor.tensorfs import RepositoryManifest, TensorReader, TensorView

from . import projection
from .safetensors_header import read_metadata

_log = logging.getLogger(__name__)

#: safetensors dtype spelling -> torch dtype attribute name. Only the dtypes
#: this fleet's checkpoints actually carry; an unknown one is a refusal, never
#: a guess, because guessing an element width silently reshapes the tensor.
#:
#: PUBLIC because a second consumer arrived (pgw#1329's constant store, which
#: must compare a shard header's dtype against the artifact manifest's torch
#: dtype name before it allocates). A private copy per lane is the fail-open
#: duplication this module was written to end, one level down.
TORCH_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I16": "int16",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "F32": "float32",
    "I64": "int64",
    "F64": "float64",
}


class TensorSource(Protocol):
    """The ``safe_open`` shape, satisfied by a file and by a manifest alike."""

    def keys(self) -> List[str]: ...
    def get_tensor(self, name: str) -> Any: ...
    def metadata(self) -> Dict[str, Any]: ...


class _NativeSource:
    """Tensors read straight out of CAS objects. No file, no materialization."""

    def __init__(self, reader: TensorReader, files: set[str], device: str, meta: Dict[str, Any]):
        self._reader = reader
        self._files = files
        self._device = device
        self._meta = meta

    def keys(self) -> List[str]:
        return [name for name, view in self._reader.items() if view.file in self._files]

    def metadata(self) -> Dict[str, Any]:
        return self._meta

    def get_tensor(self, name: str) -> Any:
        import torch

        view: TensorView = self._reader[name]
        try:
            torch_dtype = getattr(torch, TORCH_DTYPES[view.dtype])
        except (KeyError, AttributeError) as exc:
            raise projection.UnresolvedProjection(
                f"{name}: safetensors dtype {view.dtype!r} has no torch dtype "
                f"here, so its element width is unknown. Refusing to guess."
            ) from exc
        raw = torch.empty(view.nbytes, dtype=torch.uint8)
        view.readinto(memoryview(raw.numpy()))
        tensor = raw.view(torch_dtype).reshape(view.shape)
        return tensor.to(self._device) if self._device != "cpu" else tensor


@contextlib.contextmanager
def open_tensor_source(
    path: Path | str, *, device: str = "cpu", why: str
) -> Iterator[TensorSource]:
    """Read the tensors of ``path``, from the file or from the manifest.

    ``why`` is the caller's sentence about what it would otherwise get wrong;
    it appears verbatim in the refusal when a stub's manifest is unrecoverable.
    """

    file = Path(path)
    if projection.stub_at(file) is None:
        from safetensors import safe_open

        with safe_open(str(file), framework="pt", device=device) as handle:
            yield handle
        return

    snapshot, entry = projection.require_projection_for(file, why=why)
    meta = read_metadata(file, why=why)
    reader = TensorReader(snapshot.cas, RepositoryManifest((entry,)), verify=False)
    try:
        _log.info("native tensor read: %s (%d bytes) served from the CAS", entry.path, entry.size_bytes)
        yield _NativeSource(reader, {entry.path}, device, meta)
    finally:
        reader.close()


def load_state_dict(
    path: Path | str, *, device: str = "cpu", why: str
) -> Dict[str, Any]:
    """One file's whole state dict, from the file or from the manifest.

    The ``safetensors.torch.load_file`` replacement. It is NOT a
    materialization: each tensor is copied once out of mmapped CAS objects
    into the tensor that ends up in the model, which is the same number of
    copies ``load_file`` makes and one fewer than materialize-then-load.
    """

    with open_tensor_source(path, device=device, why=why) as source:
        return {name: source.get_tensor(name) for name in source.keys()}


__all__ = [
    "TORCH_DTYPES",
    "TensorSource",
    "load_state_dict",
    "open_tensor_source",
]
