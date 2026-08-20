"""pgw#1596: the headroom gate must charge for what is still MISSING.

`ModelStore._ensure_disk_headroom` compared free space against the ref's ENTIRE
manifest, with nothing subtracting the part of that same manifest already in the
CAS. First pass over an empty disk: correct. RE-ENTRY: self-defeating — the
bytes already fetched are counted as consumed AND demanded again as free, so a
materialization restarted by an ordinary residency-generation supersede can
never satisfy its own check. It needs 2x the tree and dies at the END of its own
pull.

MEASURED, pod `6uneiwhdl7fz8u`, H200, 159 GB container disk, 2026-08-20:

    InsufficientDiskError: need 104956706657 bytes for
    tensorhub/minimax-h3@serve-narrowed-fp8te; 65659441152 free after disk GC

93.3 GB consumed = 7.14 GB image + ~86.2 GB of THAT VERY TREE. It demanded the
whole 105 GB free while 82% of it was already resident, and the last position
row was 157 MB short of a complete pull. Meanwhile a pod with a WORSE
disk-to-tree ratio (208 GB / 144 GB = 1.44 vs 159 / 105 = 1.51) succeeded,
because it checked ONCE against an empty disk — which is what rules out "the
disk was simply too small" and points at re-entry.

The scenario numbers below are those bytes, scaled by 1/1000 so the test is
cheap; the RATIO that decides pass/fail is preserved exactly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

import pytest

from gen_worker.capability import InsufficientDiskError
from gen_worker.models.refs import WireRef
from gen_worker.models.store import _DISK_GC_MARGIN_BYTES, ModelStore

# The incident, /1000. Ratio-preserving, so the arithmetic under test is identical.
TREE_BYTES = 104_956_706
RESIDENT_BYTES = 86_200_000
REMAINING_BYTES = TREE_BYTES - RESIDENT_BYTES  # ~18.76 MB


class _File:
    """A SnapshotFile as the gate reads it: a digest and a size."""

    def __init__(self, digest: str, size_bytes: int) -> None:
        self.digest = digest
        self.size_bytes = size_bytes


async def _emit(_msg: Any) -> None:
    return None


def _store(tmp_path: Path, free: int) -> ModelStore:
    return ModelStore(
        _emit,
        cache_dir=tmp_path / "cas",
        disk_free_bytes_fn=lambda: free,
    )


def _files_with_resident(store: ModelStore, monkeypatch: pytest.MonkeyPatch) -> List[_File]:
    """One resident object and one missing one, summing to the whole tree."""
    resident = _File("sha256:" + "a" * 64, RESIDENT_BYTES)
    missing = _File("sha256:" + "b" * 64, REMAINING_BYTES)
    held = {resident.digest}
    monkeypatch.setattr(
        ModelStore,
        "_already_resident_bytes",
        lambda self, files: sum(
            int(f.size_bytes) for f in (files or []) if f.digest in held
        ),
    )
    return [resident, missing]


def test_a_partially_resident_ref_passes_on_its_REMAINING_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE REGRESSION. Free space covers what is missing but not the whole tree.

    This is exactly the incident's shape: most of the tree is already down, the
    disk cannot fit a second copy of it, and the fetch has only a little left to
    write. Before pgw#1596 this raised; it must now pass.
    """
    free = REMAINING_BYTES + _DISK_GC_MARGIN_BYTES  # enough for the remainder
    assert free < TREE_BYTES + _DISK_GC_MARGIN_BYTES, (
        "the scenario is only meaningful when the WHOLE tree would NOT fit"
    )
    store = _store(tmp_path, free)
    files = _files_with_resident(store, monkeypatch)

    # No raise == the gate charged the remainder, not the tree.
    asyncio.run(
        store._ensure_disk_headroom(WireRef("acme/model@prod"), TREE_BYTES, files=files)
    )


def test_the_gate_still_refuses_when_even_the_REMAINDER_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The twin one input away. Subtracting resident bytes must not disarm it."""
    store = _store(tmp_path, REMAINING_BYTES // 2)
    files = _files_with_resident(store, monkeypatch)

    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(
            store._ensure_disk_headroom(
                WireRef("acme/model@prod"), TREE_BYTES, files=files
            )
        )
    # required_bytes is the REMAINDER, and the message shows its working — the
    # old message named only the whole tree, which is why an 86 GB-resident
    # refusal read as a 105 GB shortfall.
    assert caught.value.required_bytes == REMAINING_BYTES
    text = str(caught.value)
    assert str(REMAINING_BYTES) in text
    assert str(RESIDENT_BYTES) in text and "already resident" in text


def test_a_cold_disk_is_unchanged_and_still_charges_the_whole_tree(
    tmp_path: Path,
) -> None:
    """Nothing resident means remaining == total. The first-pass case must not move."""
    store = _store(tmp_path, TREE_BYTES // 2)
    files = [_File("sha256:" + "c" * 64, TREE_BYTES)]

    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(
            store._ensure_disk_headroom(
                WireRef("acme/model@prod"), TREE_BYTES, files=files
            )
        )
    assert caught.value.required_bytes == TREE_BYTES


def test_no_files_argument_keeps_the_old_whole_tree_behaviour(tmp_path: Path) -> None:
    """Callers that cannot name their files are charged the full amount.

    The subtraction is an OPTIMISATION AGAINST A KNOWN LIST, never an
    assumption. A caller with no manifest must not get a cheaper gate.
    """
    store = _store(tmp_path, TREE_BYTES // 2)
    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(
            store._ensure_disk_headroom(WireRef("acme/model@prod"), TREE_BYTES)
        )
    assert caught.value.required_bytes == TREE_BYTES


def test_an_overstated_residency_cannot_talk_the_gate_out_of_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale reading claiming MORE resident than the tree holds is clamped.

    Otherwise a bad residency answer becomes a negative requirement and the
    gate passes anything — the failure mode that turns a guard into a no-op.
    """
    monkeypatch.setattr(
        ModelStore, "_already_resident_bytes", lambda self, files: TREE_BYTES * 10
    )
    store = _store(tmp_path, 0)
    files = [_File("sha256:" + "d" * 64, TREE_BYTES)]

    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(
            store._ensure_disk_headroom(
                WireRef("acme/model@prod"), TREE_BYTES, files=files
            )
        )
    assert caught.value.required_bytes == 0, "clamped to zero, never negative"


def test_the_real_predicate_reads_the_cas(tmp_path: Path) -> None:
    """`_already_resident_bytes` must answer from the CAS, not from a promise.

    The other tests stub the predicate to control the scenario; this one runs
    the real thing so the stub can never be the only thing that works.
    """
    from gen_worker._vendor.tensorfs import LocalCAS

    cas_dir = tmp_path / "cas"
    cas = LocalCAS(cas_dir)
    payload = b"resident-object-bytes"
    ref = cas.put_bytes(payload)

    store = _store(tmp_path, 0)
    present = _File(str(ref), len(payload))
    absent = _File("sha256:" + "e" * 64, 999_999)
    nodigest = _File("", 12_345)

    got = store._already_resident_bytes([present, absent, nodigest])
    assert got == len(payload), (
        "only the object actually in the CAS counts; a missing object and a "
        "file with no digest are both absent"
    )
