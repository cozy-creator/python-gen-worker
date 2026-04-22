"""Incremental safetensors writer: write header first, then each tensor sequentially.

This avoids holding all output tensors in memory at once. The standard
`save_file()` requires all tensors in a dict before writing. This writer
pre-computes the header from tensor metadata, writes it, then streams each
tensor's bytes directly to the output file.

Usage:
    with IncrementalSafetensorsWriter(output_path) as writer:
        writer.add_tensor_metadata("layer.weight", dtype="F8_E4M3", shape=[4096, 4096])
        writer.add_tensor_metadata("layer.bias", dtype="F8_E4M3", shape=[4096])
        writer.write_header()
        writer.write_tensor("layer.weight", tensor_bytes)
        writer.write_tensor("layer.bias", tensor_bytes)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

# Safetensors dtype names → bytes per element
_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1, "I8": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

# PyTorch dtype → safetensors dtype name
_TORCH_TO_ST_DTYPE = {
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
    "torch.float8_e4m3fn": "F8_E4M3",
    "torch.float8_e5m2": "F8_E5M2",
}


def torch_dtype_to_st(dtype) -> str:
    """Convert a torch dtype to safetensors dtype string."""
    key = str(dtype)
    if key in _TORCH_TO_ST_DTYPE:
        return _TORCH_TO_ST_DTYPE[key]
    raise ValueError(f"Unsupported torch dtype for safetensors: {dtype}")


def _compute_tensor_bytes(dtype_str: str, shape: list[int]) -> int:
    """Compute total byte size for a tensor given its safetensors dtype and shape."""
    elem_size = _DTYPE_SIZES.get(dtype_str)
    if elem_size is None:
        raise ValueError(f"Unknown safetensors dtype: {dtype_str}")
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * elem_size


class IncrementalSafetensorsWriter:
    """Write a safetensors file incrementally: header first, then tensors one at a time."""

    def __init__(self, output_path: Path):
        self._output_path = Path(output_path)
        self._tensors_meta: list[dict[str, Any]] = []  # [{name, dtype, shape}]
        self._tensor_order: list[str] = []
        self._header_written = False
        self._data_offset = 0
        self._fh = None
        self._written_tensors: set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def add_tensor_metadata(self, name: str, *, dtype: str, shape: list[int]) -> None:
        """Register a tensor's metadata. Must be called before write_header()."""
        if self._header_written:
            raise RuntimeError("Cannot add metadata after header is written")
        self._tensors_meta.append({"name": name, "dtype": dtype, "shape": shape})
        self._tensor_order.append(name)

    def write_header(self) -> None:
        """Compute and write the safetensors header. Must be called once before any write_tensor()."""
        if self._header_written:
            raise RuntimeError("Header already written")

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._output_path, "wb")

        # Build header JSON: {tensor_name: {dtype, shape, data_offsets: [start, end]}}
        header: dict[str, Any] = {}
        offset = 0
        for meta in self._tensors_meta:
            size = _compute_tensor_bytes(meta["dtype"], meta["shape"])
            header[meta["name"]] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [offset, offset + size],
            }
            offset += size

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Pad header to 8-byte alignment.
        padding = (8 - (len(header_json) % 8)) % 8
        header_json += b" " * padding

        # Write: [8 bytes header_size] [header_json] [tensor data...]
        self._fh.write(struct.pack("<Q", len(header_json)))
        self._fh.write(header_json)
        self._header_written = True
        self._data_offset = 0

    def write_tensor(self, name: str, data: bytes) -> None:
        """Write one tensor's raw bytes. Must match the order of add_tensor_metadata() calls."""
        if not self._header_written:
            raise RuntimeError("Must call write_header() before write_tensor()")
        if self._fh is None:
            raise RuntimeError("Writer is closed")
        if name in self._written_tensors:
            raise RuntimeError(f"Tensor {name!r} already written")

        expected_name = self._tensor_order[len(self._written_tensors)]
        if name != expected_name:
            raise RuntimeError(f"Expected tensor {expected_name!r}, got {name!r} (must write in order)")

        self._fh.write(data)
        self._written_tensors.add(name)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
