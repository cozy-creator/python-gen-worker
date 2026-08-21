"""pgw#1596: the headroom gate must charge for what is still MISSING.

`ModelStore._ensure_disk_headroom` compared free space against the ref's ENTIRE
manifest, with nothing subtracting the part of that same manifest already in the
CAS. First pass over an empty disk: correct. RE-ENTRY: self-defeating — the
bytes already fetched are counted as consumed AND demanded again as free, so a
materialization restarted by an ordinary residency-generation supersede can
never satisfy its own check. It needs 2x the tree and it dies at the END of its
own pull.

MEASURED, pod `6uneiwhdl7fz8u`, H200, 159 GB container disk, 2026-08-20:

    InsufficientDiskError: need 104956706657 bytes for
    tensorhub/minimax-h3@serve-narrowed-fp8te; 65659441152 free after disk GC

93.3 GB consumed = 7.14 GB image + ~86.2 GB of THAT VERY TREE. It demanded the
whole 105 GB free while 82% of it was already resident, and the last position
row was 157 MB short of a complete pull. Meanwhile a pod with a WORSE
disk-to-tree ratio (208 GB / 144 GB = 1.44 vs 159 / 105 = 1.51) succeeded,
because it checked ONCE against an empty disk — which is what rules out "the
disk was simply too small" and points at re-entry.

pgw#1631 promoted the fix from patch to construction: the gate now takes the
fill's PLAN and has no manifest to re-price. These cases are therefore stated in
plans rather than in (total, files) pairs — the arithmetic under test is
identical, and there is one less way to express it wrong.

The scenario numbers below are those bytes, scaled by 1/1000 so the test is
cheap; the RATIO that decides pass/fail is preserved exactly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from gen_worker.capability import InsufficientDiskError
from gen_worker.models.fill_plan import FillPlan, PlannedObject, plan_fill
from gen_worker.models.refs import WireRef
from gen_worker.models.store import _DISK_GC_MARGIN_BYTES, ModelStore

# The incident, /1000. Ratio-preserving, so the arithmetic under test is identical.
TREE_BYTES = 104_956_706
RESIDENT_BYTES = 86_200_000
REMAINING_BYTES = TREE_BYTES - RESIDENT_BYTES  # ~18.76 MB


async def _emit(_msg: Any) -> None:
    return None


def _store(tmp_path: Path, free: int) -> ModelStore:
    return ModelStore(
        _emit,
        cache_dir=tmp_path / "cas",
        disk_free_bytes_fn=lambda: free,
    )


def _partial_plan() -> FillPlan:
    """One resident object and one missing one, summing to the whole tree."""
    return FillPlan(
        present=(PlannedObject("sha256:" + "a" * 64, RESIDENT_BYTES),),
        missing=(PlannedObject("sha256:" + "b" * 64, REMAINING_BYTES),),
    )


def _cold_plan() -> FillPlan:
    return FillPlan(missing=(PlannedObject("sha256:" + "c" * 64, TREE_BYTES),))


def test_a_partially_resident_ref_passes_on_its_REMAINING_bytes(
    tmp_path: Path,
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

    # No raise == the gate charged the remainder, not the tree.
    asyncio.run(
        store._ensure_disk_headroom(WireRef("acme/model@prod"), _partial_plan())
    )


def test_the_gate_still_refuses_when_even_the_REMAINDER_does_not_fit(
    tmp_path: Path,
) -> None:
    """The twin one input away. Subtracting resident bytes must not disarm it."""
    store = _store(tmp_path, REMAINING_BYTES // 2)

    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(
            store._ensure_disk_headroom(WireRef("acme/model@prod"), _partial_plan())
        )
    # required_bytes is the REMAINDER, and the message shows its working — the
    # old message named only the whole tree, which is why an 86 GB-resident
    # refusal read as a 105 GB shortfall.
    assert caught.value.required_bytes == REMAINING_BYTES
    text = str(caught.value)
    assert str(REMAINING_BYTES) in text
    assert str(RESIDENT_BYTES) in text and "already banked" in text
    # pgw#1612: and it names the mount it ran out on.
    assert "mount=" in text


def test_a_cold_disk_is_unchanged_and_still_charges_the_whole_tree(
    tmp_path: Path,
) -> None:
    """Nothing resident means remaining == total. The first-pass case must not move."""
    store = _store(tmp_path, TREE_BYTES // 2)

    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(
            store._ensure_disk_headroom(WireRef("acme/model@prod"), _cold_plan())
        )
    assert caught.value.required_bytes == TREE_BYTES


def test_a_file_with_no_digest_is_charged_for(tmp_path: Path) -> None:
    """A caller whose manifest cannot name an object is charged the full amount.

    The subtraction is an OPTIMISATION AGAINST A KNOWN OBJECT, never an
    assumption. An entry with no digest is unplannable, so the plan counts it
    as missing — the honest direction, since it will be fetched.
    """
    from gen_worker._vendor.tensorfs import LocalCAS

    class _File:
        def __init__(self, digest: str, size_bytes: int) -> None:
            self.digest = digest
            self.size_bytes = size_bytes

    plan = plan_fill(LocalCAS(tmp_path / "cas"), [_File("", TREE_BYTES)])
    assert plan.missing_bytes == TREE_BYTES
    assert plan.undigested and plan.present_bytes == 0

    store = _store(tmp_path, TREE_BYTES // 2)
    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(store._ensure_disk_headroom(WireRef("acme/model@prod"), plan))
    assert caught.value.required_bytes == TREE_BYTES


def test_the_gate_cannot_be_talked_out_of_existence_by_a_bad_reading(
    tmp_path: Path,
) -> None:
    """pgw#1631 deletes the failure mode rather than clamping it.

    The pre-plan gate did `remaining = total - resident` and had to clamp,
    because an overstated residency produced a NEGATIVE requirement and the gate
    passed anything. A plan cannot overstate: `missing_bytes` is a sum over a
    disjoint list, so it is non-negative and bounded by the total by
    construction. There is no subtraction left to go wrong.
    """
    plan = _partial_plan()
    assert plan.missing_bytes >= 0
    assert plan.missing_bytes + plan.present_bytes == plan.total_bytes == TREE_BYTES
    assert plan.missing_bytes <= plan.total_bytes


def test_the_real_predicate_reads_the_cas(tmp_path: Path) -> None:
    """`plan_fill` must answer from the CAS, not from a promise.

    The other tests state plans directly to control the scenario; this one runs
    the real predicate so a stated plan can never be the only thing that works.
    """
    from gen_worker._vendor.tensorfs import LocalCAS

    class _File:
        def __init__(self, digest: str, size_bytes: int) -> None:
            self.digest = digest
            self.size_bytes = size_bytes

    cas_dir = tmp_path / "cas"
    cas = LocalCAS(cas_dir)
    payload = b"resident-object-bytes"
    ref = cas.put_bytes(payload)

    plan = plan_fill(cas, [
        _File(str(ref), len(payload)),
        _File("sha256:" + "e" * 64, 999_999),
        _File("", 12_345),
    ])

    assert plan.present_bytes == len(payload), (
        "only the object actually in the CAS counts; a missing object and a "
        "file with no digest are both absent"
    )
    assert plan.missing_bytes == 999_999 + 12_345
