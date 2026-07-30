"""The exported lane proves adoption its OWN way (pgw#735).

`executor.py`'s adoption gates required an FX cache hit. That is the right proof
for a dynamo cell — a hit means the delivered graph was reused rather than
silently re-traced — but an EXPORTED artifact performs no FX lookup at all, so
the gate could never pass and every `.pt2` adoption scored as a failure.

The fix must not be a synthesized hit counter: that would be a lie inside the one
path whose whole job is to detect lies about serving compiled. These tests pin
the honest predicate and the fail-closed cases.
"""

from __future__ import annotations

import pytest

from gen_worker import aot_serve, cell_key


class _Pipe:
    """A pipeline carrying only what the proof reads."""


def _arm(pipe, *, calls: int, failed: bool = False) -> None:
    setattr(pipe, aot_serve._MARKER_ATTR, {
        "meta": {"sku": "sm_89", "torch": "2.13", "precision": "bf16"},
        "state": {"successful_calls": calls, "failed": failed},
    })


def test_proven_since_requires_new_successful_calls_and_no_revocation():
    pipe = _Pipe()
    assert not aot_serve.proven_since(pipe, 0)   # unarmed is never proven
    _arm(pipe, calls=0)
    assert not aot_serve.proven_since(pipe, 0)
    _arm(pipe, calls=3)
    assert aot_serve.proven_since(pipe, 0)
    assert aot_serve.proven_since(pipe, 2)
    # The delta matters, not the absolute count: a previous boot's calls
    # cannot prove THIS warmup exercised the artifact.
    _arm(pipe, calls=7)
    assert not aot_serve.proven_since(pipe, 7)
    assert not aot_serve.proven_since(pipe, 9)
    # An artifact that ran and then revoked (a B1/B2 refusal) has proven
    # nothing — fail closed, exactly like a dynamo cell with zero hits.
    _arm(pipe, calls=5, failed=True)
    assert aot_serve.execution_count(pipe) == 5
    assert not aot_serve.proven_since(pipe, 0)


def test_exported_kind_cell_key_refusals_are_named():
    """cell_key.from_artifact_metadata recomputes INDUCTOR-cache axes. An
    exported cell rides the same ck5 key space but its key is STAMPED at mint,
    so the refusal must say so by name instead of failing opaquely."""
    with pytest.raises(cell_key.CellKeyError) as exc:
        cell_key.from_artifact_metadata(
            {"kind": "aot-inductor", "cell_key": "ck5-" + "a" * 56})
    message = str(exc.value)
    assert "aot-inductor" in message
    assert "cell_key" in message and "STAMPED" in message
    assert "ck5-" + "a" * 56 in message

    with pytest.raises(cell_key.CellKeyError) as missing:
        cell_key.from_artifact_metadata({"kind": "aot-inductor"})
    assert "MISSING" in str(missing.value)

    with pytest.raises(cell_key.CellKeyError) as unknown:
        cell_key.from_artifact_metadata({"kind": "trt-engine"})
    assert "no cell-key identity" in str(unknown.value)
