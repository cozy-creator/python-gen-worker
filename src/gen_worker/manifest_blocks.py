"""The declaration rows of a baked ``endpoint.lock``.

ONE walk over BOTH publishable shapes. A release may declare ``@endpoint``
functions, ``@job`` functions, or both, and a release with jobs and ZERO
functions is legal (th#2049) — so every reader that asks "what did this image
declare" must ask it of both blocks, or it goes blind on the shape it was not
written for.

pgw#1354 is what blindness costs: ``entrypoint.get_modules_from_manifest`` and
``cuda_probe.should_probe_cuda`` each walked ``functions[]`` alone, both
written when everything was a function. The first pod ever booted off a
jobs-only image (27 jobs, 0 functions) therefore found zero user modules and
exited 1 before hello, twice, on two card families — and was never CUDA-probed
on the way. The fix is this module: not a parallel jobs walk beside the
function walk (two walks drift; the next block makes it three), but one
derivation both feeders read.

The two row shapes are deliberately identical where they overlap — ``name``,
``module``, ``resources``, ``env``, ``publishes`` — because a job promoted to a
serverless endpoint must not change identity on the way. That overlap is what
makes one walk correct rather than merely convenient. A reader that needs a
FUNCTION-ONLY concept (slots, bindings, compile cells, execution lanes — none
of which a job declares) must still walk ``functions[]`` directly and say why.

Stdlib only, and no ``gen_worker`` imports: the control parent reads this and
must stay a bare interpreter with no path to torch (pgw#763).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

#: Every manifest block carrying declaration rows, in wire order.
DECLARATION_BLOCKS = ("functions", "jobs")


def declaration_rows(manifest: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every declaration row in ``manifest``: functions then jobs.

    Tolerant of a malformed or absent block — a lock this process cannot read
    must not raise here, because the callers turn "nothing declared" into a
    typed refusal that names the gap.
    """
    rows: List[Dict[str, Any]] = []
    if not isinstance(manifest, Mapping):
        return rows
    for block in DECLARATION_BLOCKS:
        for row in manifest.get(block) or ():
            if isinstance(row, dict):
                rows.append(row)
    return rows


def block_counts(manifest: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    """``{block: row count}`` for every declaration block — the shape a boot
    log or a refusal quotes when it has to say what the image declared."""
    counts: Dict[str, int] = {}
    for block in DECLARATION_BLOCKS:
        rows = (manifest or {}).get(block) if isinstance(manifest, Mapping) else None
        counts[block] = len(rows) if isinstance(rows, list) else 0
    return counts


def declared_row_count(manifest: Optional[Mapping[str, Any]]) -> int:
    """How many declarations this image carries, across both blocks."""
    return sum(block_counts(manifest).values())
