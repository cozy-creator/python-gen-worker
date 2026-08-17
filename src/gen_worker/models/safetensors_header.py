"""The one way to read a safetensors header, and the one bound on its length.

A safetensors file opens with an 8-byte
little-endian header length taken straight from the file. It is attacker- or
corruption-controlled, and every reader turns it directly into an allocation
(``json.loads(f.read(n))``). Unbounded, one crafted file declaring 2**63-1
is an OOM in whichever process opened it — a serving worker, a conversion
pod, or the shard writer.

THE THREAT, stated once: a declared header length that the file cannot back
sizes a read and a parse before anything has validated it.

WHY NOTHING ELSE PREVENTS IT: the length is read before any other structure
exists, so there is nothing earlier to lean on. This bound is load-bearing.

Stated ONCE, here. A second copy that disagrees means the writer accepts headers
the loader refuses, so the re-shard path emits a shard the serving path cannot
open — same bytes, two verdicts.

WHY 100 MiB AND NOT A MEASUREMENT: real safetensors headers are tens of KB;
the largest sharded checkpoints in the fleet are a few MB of JSON. 100 MiB is
~20x above anything observed and exists only to make the number finite — it
is a plausibility floor, not a tuned capacity. What would change it: a
legitimate model whose header exceeds ~10 MiB, which would mean the tensor
count per shard grew by an order of magnitude.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict

MAX_HEADER_BYTES: int = 100 << 20


def header_len_ok(n: int) -> bool:
    """Whether a declared safetensors header length is plausible.

    Zero and negative are refusals, not "no header": a file that declares
    nothing is malformed, and treating it as an empty header would let a
    truncated blob parse as a valid one (§4.24 item 4).
    """
    return 0 < n <= MAX_HEADER_BYTES


def read_header(path: Path | str, *, why: str) -> Dict[str, Any]:
    """The parsed safetensors header at ``path``, served from the MANIFEST when
    that path is a projection stub.

    ``{}`` still means "there is no readable header here", and it is still the
    honest answer for a real file that is truncated, empty, or not
    safetensors. What it must NEVER mean again is "this file is a pointer
    stub". pgw#1308 counted 39 header readers in this repo that treated a
    parse failure as no-information and fell back to a default; a stub makes
    every one of them wrong at once, silently, on the serving path
    (``detect_on_disk_dtype`` -> fp32 at 2x VRAM; ``_quantized_layers`` -> an
    fp8 artifact routed to the plain bf16 lane; ``_svdq_from_file`` -> an svdq
    checkpoint that stops being an svdq checkpoint).

    ``why`` is the caller's own sentence about what it would otherwise get
    wrong, and it appears in the refusal. Three copies of this function were
    deleted to write this one: a fail-open that is duplicated per lane cannot
    be fixed per lane.
    """

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
