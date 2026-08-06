"""The one bound on a safetensors declared header length.

pgw#973 (DESIGN-RULINGS §4.24). A safetensors file opens with an 8-byte
little-endian header length taken straight from the file. It is attacker- or
corruption-controlled, and every reader turns it directly into an allocation
(``json.loads(f.read(n))``). Unbounded, one crafted file declaring 2**63-1
is an OOM in whichever process opened it — a serving worker, a conversion
pod, or the shard writer.

THE THREAT, stated once: a declared header length that the file cannot back
sizes a read and a parse before anything has validated it.

WHY NOTHING ELSE PREVENTS IT: the length is read before any other structure
exists, so there is nothing earlier to lean on. This bound is load-bearing.

It was previously stated SIX times — ``models/w4a4.py``, ``models/w8a8.py``,
``models/svdq.py``, ``models/loading.py``, ``convert/ingest.py`` all at
100 MiB, and ``convert/writer.py`` at 512 MiB. The odd one out was not
harmless: writer accepted headers loading would refuse, so the re-shard path
could emit a shard the serving path could not open. Same bytes, two verdicts.

WHY 100 MiB AND NOT A MEASUREMENT: real safetensors headers are tens of KB;
the largest sharded checkpoints in the fleet are a few MB of JSON. 100 MiB is
~20x above anything observed and exists only to make the number finite — it
is a plausibility floor, not a tuned capacity. What would change it: a
legitimate model whose header exceeds ~10 MiB, which would mean the tensor
count per shard grew by an order of magnitude.
"""

from __future__ import annotations

MAX_HEADER_BYTES: int = 100 << 20


def header_len_ok(n: int) -> bool:
    """Whether a declared safetensors header length is plausible.

    Zero and negative are refusals, not "no header": a file that declares
    nothing is malformed, and treating it as an empty header would let a
    truncated blob parse as a valid one (§4.24 item 4).
    """
    return 0 < n <= MAX_HEADER_BYTES


__all__ = ["MAX_HEADER_BYTES", "header_len_ok"]
