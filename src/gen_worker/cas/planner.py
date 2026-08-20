"""The canonical object planner, in pure Python. FIRST-PARTY, not vendored.

pgw#1575 moved this module out of `_vendor/tensorfs/`, where it had been
masquerading as a vendored file it never was: it exists at NO upstream rev.
Upstream deleted the Python chunker (`chunking.py`, the greedy small-tensor
packer) at `00c57c1` — "pgw#1259: one chunker, the legacy Python data plane is
gone" — because its grid disagreed with the Rust planner's: packed tensors own
no digest of their own and cannot be inherited by `TensorWriter`. Upstream's
successor is `crates/tensorfs-core/src/planner/{mod,safetensors,gguf}.rs`,
reachable from Python only through the compiled extension, and pgw#1310 rules a
compiled extension out of a source-vendored wheel. So the grid is restored here
in pure Python — the pgw#1344 precedent — and pinned to upstream's own released
conformance corpus (`spec/v1/planner-vectors`), vendored at
`tests/testdata/planner-vectors/` and asserted object-for-object by
`tests/test_planner_grid.py`.

The grid, in one paragraph. A safetensors file is its header region, then one
region per declared tensor, each split every 64 MiB from ITS OWN start. A GGUF
file is three header domains kept apart — metadata block, tensor directory,
pre-data padding — then every tensor's unpadded extent with its trailing
alignment padding as a region of its own. Everything else is `blob-v1`: ONE
unchunked region of any size, never a grid. Isolating padding and starting the
grid at each tensor's own start is the whole mechanism: it is what lets the
same weights share objects across a safetensors/GGUF pair, across a fused and
a split packaging, and across two publishes of the same checkpoint.

The safetensors structural proof below (`_tensor_spans`) is unchanged from the
retired module — it was already a faithful port of `safetensors.rs::read_layout`
and it survives; only the greedy packing it fed died.

ONE structural refusal this port cannot see: `gguf.rs::parse` refuses a
duplicate metadata KEY, and `gguf.read_header` skips metadata values without
recording their keys. A GGUF carrying one would be planned on the tensor grid
here and as `blob-v1` upstream. Coverage is still exact and the manifest is
still valid; the cost is a dedup miss on a file that is already malformed.
`test_the_duplicate_metadata_key_gap_is_named` keeps it from becoming folklore.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import BinaryIO

from .._vendor.tensorfs import gguf
from .._vendor.tensorfs.manifest import MAX_CHUNK_SIZE

# THE SAFETENSORS HEADER BOUND IS UPSTREAM'S, AND IT IS IMPORTED RATHER THAN
# RESTATED. pgw#973 says one threat, one number, and it is right — but there
# are two OWNERS here, not two copies of one number.
# `models/safetensors_header.MAX_HEADER_BYTES` is 100 MiB (104_857_600) and is
# a REFUSAL bound for readers. This one is `planner/safetensors.rs`'s
# `MAX_HEADER_SIZE` (100_000_000, safetensors' own `tensor.rs` limit); it
# decides a PLAN rather than a refusal — above it a file is `blob-v1`, not an
# error — and the released `spec/v1/planner-vectors` corpus is keyed on it.
# Swapping in the reader's number would make this port disagree with the Rust
# planner for any header between the two, a dedup miss the conformance suite
# cannot see. So it comes from the vendored snapshot, which states it once for
# this lineage.
from .._vendor.tensorfs.tensors import _MAX_HEADER_BYTES as _MAX_SAFETENSORS_HEADER_BYTES

# `planner/mod.rs`: the tensor chunk grid constant. It is NOT a store admission
# cap — a blob is one object of any size.
MAX_OBJECT_SIZE = MAX_CHUNK_SIZE
MAX_OBJECT_COUNT = 1_000_000

SAFETENSORS_V1 = "safetensors-v1"
GGUF_V1 = "gguf-v1"
BLOB_V1 = "blob-v1"

HEADER = "header"
TENSOR = "tensor"
BLOB = "blob"

# TensorFS v1 targets 64-bit Linux/POSIX. Safetensors decodes every shape
# dimension into Rust usize before it checks tensor byte lengths.
_MAX_USIZE = (1 << 64) - 1

# This mirrors safetensors::tensor::Dtype::bitsize at upstream 6eb4dc9, and
# `safetensors.rs::dtype_bits`. Unknown future dtypes make the file a blob
# rather than being guessed here.
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
    """A plan would exceed the bounded object cardinality.

    `try_plan` in both Rust planners turns this into "not a tensor container",
    so the file falls back to `blob-v1` rather than refusing.
    """


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
    """`planner/mod.rs::append_split_region` — the 64 MiB grid inside one region.

    The grid starts at the region's OWN start, which is what makes a fused
    file's objects the split file's objects.
    """

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
    """`planner/mod.rs::blob_plan` — one unchunked region of any size."""

    if file_size == 0:
        return Plan(BLOB_V1, 0, ())
    return Plan(BLOB_V1, file_size, (Region(0, file_size, BLOB),))


# -- safetensors -------------------------------------------------------------


def _tensor_spans(source: BinaryIO, size: int) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """`safetensors.rs::read_layout` — the whole structural proof, no body read."""

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
    """`safetensors.rs::try_plan` — the header, then one grid per tensor."""

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


# -- GGUF --------------------------------------------------------------------


def _gguf_plan(source: BinaryIO, size: int) -> Plan | None:
    """`gguf.rs::try_plan` — three header domains, then tensors with padding split off."""

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

    # `gguf.rs::parse` refuses these; `read_header` is a superset that does not.
    # Each one makes the file a blob upstream, so it has to make it a blob here.
    if header.alignment < gguf.MIN_ALIGNMENT:
        return None
    names = {tensor.name for tensor in header.tensors}
    if len(names) != len(header.tensors):
        return None

    # An empty GGUF has no data section to align to, which is the one case
    # where the planner does not round the directory up.
    data_start = header.data_start if header.tensors else header.directory_end
    expected = 0
    for tensor in header.tensors:
        # Offsets are declared relative to `data_start` and must be sequential
        # over the padded extents, with no gap and no reordering.
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


# -- the automatic registry --------------------------------------------------


def plan(source: BinaryIO, size: int) -> Plan:
    """`planner/mod.rs::plan_once` — the closed automatic planner registry.

    Format parse failure is not a refusal: every stable byte stream has an
    automatic plan, and a malformed or unsupported container falls back
    atomically to the whole-blob plan. The choice cannot be selected by callers.
    """

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
    """The v1-manifest adapter: ordered chunk lengths, or `()` for a whole blob.

    A tensor container becomes its planned regions verbatim, which is the point
    of this module. `blob-v1` becomes a chunkless entry, which is the same
    single object the plan names.

    THE ONE DEVIATION, and it is upstream's counterpart's, not a preference.
    A blob above 64 MiB is still cut on the fixed grid here, because
    tensorhub's publish-v2 lane promotes an object with a SINGLE PUT
    and would refuse the whole-blob entry terminally. Its refusal, quoted from
    `internal/s3/sha256_cas.go` so a reader who hits it can grep for it, reads
    "verified promote: size %d is outside the HashRepo single-PUT range
    0..67108864". Files at or below 64 MiB — every `config.json`, every
    small non-tensor file — are already exactly blob-v1. The residual is
    tracked as pgw#1366: it closes when publish-v2 grants ride th#2064's
    multipart blob lane, and `test_the_oversized_blob_deviation_is_the_only_one`
    fails the moment anyone widens it.
    """

    planned = plan(source, size)
    if planned.planner != BLOB_V1:
        return planned.lengths()
    if size <= MAX_OBJECT_SIZE:
        return ()
    full, remainder = divmod(size, MAX_OBJECT_SIZE)
    lengths = (MAX_OBJECT_SIZE,) * full
    return lengths + ((remainder,) if remainder else ())
