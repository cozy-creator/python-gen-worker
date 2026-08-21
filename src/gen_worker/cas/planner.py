"""The canonical object planner, in pure Python."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import BinaryIO

from .._vendor.tensorfs import gguf
from .._vendor.tensorfs.manifest import MAX_CHUNK_SIZE

from .._vendor.tensorfs.tensors import _MAX_HEADER_BYTES as _MAX_SAFETENSORS_HEADER_BYTES

MAX_OBJECT_SIZE = MAX_CHUNK_SIZE
MAX_OBJECT_COUNT = 1_000_000

SAFETENSORS_V1 = "safetensors-v1"
GGUF_V1 = "gguf-v1"
BLOB_V1 = "blob-v1"

HEADER = "header"
TENSOR = "tensor"
BLOB = "blob"

_MAX_USIZE = (1 << 64) - 1

_DTYPE_BITS = {
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E4M3FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
    "C64": 64,
}


class ObjectLimit(Exception):
    """A plan would exceed the bounded object cardinality."""


@dataclass(frozen=True, slots=True)
class Region:
    offset: int
    length: int
    kind: str


@dataclass(frozen=True, slots=True)
class Plan:
    planner: str
    file_size: int
    regions: tuple[Region, ...]

    def lengths(self) -> tuple[int, ...]:
        return tuple(region.length for region in self.regions)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key {key!r}")
        result[key] = value
    return result


def _append_split(regions: list[Region], offset: int, length: int, kind: str) -> None:

    if length == 0:
        return
    if len(regions) + -(-length // MAX_OBJECT_SIZE) > MAX_OBJECT_COUNT:
        raise ObjectLimit
    cursor = offset
    remaining = length
    while remaining > 0:
        part = min(remaining, MAX_OBJECT_SIZE)
        regions.append(Region(cursor, part, kind))
        cursor += part
        remaining -= part


def _blob_plan(file_size: int) -> Plan:

    if file_size == 0:
        return Plan(BLOB_V1, 0, ())
    return Plan(BLOB_V1, file_size, (Region(0, file_size, BLOB),))


def _tensor_spans(source: BinaryIO, size: int) -> tuple[int, tuple[tuple[int, int], ...]] | None:

    if size < 10:
        return None
    source.seek(0)
    prefix = source.read(8)
    if len(prefix) != 8:
        return None
    header_length = int.from_bytes(prefix, "little")
    header_end = 8 + header_length
    if (
        header_length < 2
        or header_length > _MAX_SAFETENSORS_HEADER_BYTES
        or header_end > size
    ):
        return None
    # bound-justified: the four-clause refusal directly above caps
    # `header_length` at `_MAX_SAFETENSORS_HEADER_BYTES` (100 MiB, the
    # safetensors reference limit) AND at the file's own declared size before
    # this line can run, so the allocation is bounded by a constant, not by
    # the header the file claims. Upstream's `safetensors.rs::read_layout`
    # applies the same two bounds in the same order.
    header_bytes = source.read(header_length)
    if len(header_bytes) != header_length or not header_bytes.startswith(b"{"):
        return None
    try:
        raw = json.loads(header_bytes, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(raw, dict):
        return None
    metadata = raw.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        return None
    if len(raw) > MAX_OBJECT_COUNT:
        return None

    spans: list[tuple[int, int]] = []
    for name, value in raw.items():
        if (
            not name
            or not isinstance(value, dict)
            or set(value) != {"dtype", "shape", "data_offsets"}
        ):
            return None
        dtype = value["dtype"]
        shape = value["shape"]
        offsets = value["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
            return None
        if not isinstance(shape, list) or any(
            type(dimension) is not int or not 0 <= dimension <= _MAX_USIZE
            for dimension in shape
        ):
            return None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            return None
        start, end = offsets
        if start < 0 or end < start or end > _MAX_USIZE:
            return None
        dtype_bits = _DTYPE_BITS[dtype]
        body_bits = (end - start) * 8
        elements = 1
        for dimension in shape:
            if dimension and elements > _MAX_USIZE // dimension:
                return None
            elements *= dimension
        if elements and dtype_bits > _MAX_USIZE // elements:
            return None
        tensor_bits = elements * dtype_bits
        if tensor_bits != body_bits:
            return None
        spans.append((start, end))

    spans.sort()
    data_size = size - header_end
    cursor = 0
    for start, end in spans:
        if start != cursor or end > data_size:
            return None
        cursor = end
    if cursor != data_size:
        return None
    return header_end, tuple(spans)


def _safetensors_plan(source: BinaryIO, size: int) -> Plan | None:

    parsed = _tensor_spans(source, size)
    if parsed is None:
        return None
    header_end, spans = parsed
    regions: list[Region] = []
    try:
        _append_split(regions, 0, header_end, HEADER)
        for start, end in spans:
            _append_split(regions, header_end + start, end - start, TENSOR)
    except ObjectLimit:
        return None
    return Plan(SAFETENSORS_V1, size, tuple(regions))


def _gguf_plan(source: BinaryIO, size: int) -> Plan | None:

    if size < 24:
        return None

    def read(offset: int, length: int) -> bytes:
        source.seek(offset)
        data = source.read(length)
        if len(data) != length:
            raise gguf.GGUFError("GGUF read ran past the end of the file")
        return data

    try:
        header = gguf.read_header(read, size)
    except (gguf.GGUFError, OSError, UnicodeDecodeError):
        return None

    if header.alignment < gguf.MIN_ALIGNMENT:
        return None
    names = {tensor.name for tensor in header.tensors}
    if len(names) != len(header.tensors):
        return None

    data_start = header.data_start if header.tensors else header.directory_end
    expected = 0
    for tensor in header.tensors:
        if tensor.offset != data_start + expected:
            return None
        expected += gguf.align_up(tensor.nbytes, header.alignment)
    if data_start + expected != size:
        return None

    regions: list[Region] = []
    try:
        _append_split(regions, 0, header.directory_start, HEADER)
        _append_split(
            regions,
            header.directory_start,
            header.directory_end - header.directory_start,
            HEADER,
        )
        _append_split(regions, header.directory_end, data_start - header.directory_end, HEADER)
        for tensor in header.tensors:
            _append_split(regions, tensor.offset, tensor.nbytes, TENSOR)
            padding = gguf.align_up(tensor.nbytes, header.alignment) - tensor.nbytes
            _append_split(regions, tensor.offset + tensor.nbytes, padding, HEADER)
    except ObjectLimit:
        return None
    return Plan(GGUF_V1, size, tuple(regions))


def plan(source: BinaryIO, size: int) -> Plan:
    """`planner/mod.rs::plan_once` — the closed automatic planner registry."""

    if size < 10:
        return _blob_plan(size)
    if size >= 24:
        planned = _gguf_plan(source, size)
        if planned is not None:
            return planned
    planned = _safetensors_plan(source, size)
    if planned is not None:
        return planned
    return _blob_plan(size)


def plan_chunks(source: BinaryIO, size: int) -> tuple[int, ...]:
    """The v1-manifest adapter: ordered chunk lengths, or `()` for a whole blob."""

    planned = plan(source, size)
    if planned.planner != BLOB_V1:
        return planned.lengths()
    if size <= MAX_OBJECT_SIZE:
        return ()
    full, remainder = divmod(size, MAX_OBJECT_SIZE)
    lengths = (MAX_OBJECT_SIZE,) * full
    return lengths + ((remainder,) if remainder else ())
