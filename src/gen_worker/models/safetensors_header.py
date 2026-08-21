"""The one way to read a safetensors header, and the one bound on its length. The 8-byte little-endian length prefix is attacker- or corruption-controlled and sizes a read+parse before anything validates it (a crafted 2**63-1 is an OOM in whichever process opened the file), so this bound is load-bearing — and stated ONCE: a second copy that disagrees means the writer accepts headers the loader refuses. 100 MiB is ~20x anything observed — a plausibility floor, not a tuned capacity."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict

MAX_HEADER_BYTES: int = 100 << 20


def header_len_ok(n: int) -> bool:
    """Whether a declared safetensors header length is plausible."""
    return 0 < n <= MAX_HEADER_BYTES


def read_header(path: Path | str, *, why: str) -> Dict[str, Any]:
    """The parsed safetensors header at ``path``, served from the MANIFEST when that path is a projection stub."""

    from . import projection

    file = Path(path)
    if projection.stub_at(file) is not None:
        snapshot, entry = projection.require_projection_for(file, why=why)
        with snapshot.open_tensors(verify=False) as reader:
            (n,) = struct.unpack("<Q", reader.read_range(entry.path, 0, 8))
            if not header_len_ok(n) or 8 + n > entry.size_bytes:
                return {}
            header = json.loads(reader.read_range(entry.path, 8, n))
        return header if isinstance(header, dict) else {}
    try:
        with open(file, "rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return {}
            (n,) = struct.unpack("<Q", raw)
            if not header_len_ok(n):
                return {}
            header = json.loads(handle.read(n))
    except (OSError, ValueError):
        return {}
    return header if isinstance(header, dict) else {}


def read_metadata(path: Path | str, *, why: str) -> Dict[str, Any]:
    """A safetensors file's ``__metadata__`` block, stub-aware."""

    meta = read_header(path, why=why).get("__metadata__")
    return meta if isinstance(meta, dict) else {}


__all__ = ["MAX_HEADER_BYTES", "header_len_ok", "read_header", "read_metadata"]
