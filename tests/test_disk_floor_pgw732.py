"""pgw#732 — `Resources` gets the disk axis the hub has been waiting for.

th#1233 sizes a conversion pod's container disk from the bytes the job will
materialize, so the COMMON case needs no declaration. The residue is jobs
whose need is not derivable from the source: multi-source marries, large
intermediate scratch, and the live example — `mirror_svdq`, which fetches a
13.08 GB nunchaku checkpoint straight from HuggingFace. No catalog read will
ever see those bytes; only the author knows they exist.

The hub half was already wired: `mergeRequestDiskIntoSupply` honours a
per-function `min_disk_gb` as an additional floor, exactly like
the VRAM floor th#1867 has since deleted. Nothing emitted it, because
`Resources` had no disk axis at
all. This closes the emitter.

Shape is copied from `ram_gb_hint` / `compute_capability`
deliberately — a fourth spelling of "an allocation-time ask" is how an author
gets one of them wrong.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from gen_worker.api.decorators import Resources


def test_the_axis_exists_and_projects_under_the_hubs_own_key() -> None:
    """No remap, unlike `ram_gb_hint` -> `ram_gb`: the hub already reads
    `min_disk_gb`, so the declaration and the wire name are the same word."""
    assert Resources(min_disk_gb=64).manifest_dict() == {"min_disk_gb": 64.0}


def test_it_is_a_FLOOR_not_a_hint_and_the_name_says_so() -> None:
    """Naming is the contract. `ram_gb_hint` is an ask the
    platform may miss; this one the hub raises the pod's disk to meet, and may
    exceed — so `min_` reads truer than a `_hint` suffix, and there must be no
    `disk_gb_hint` spelling for an author to reach for by analogy."""
    assert not hasattr(Resources(), "disk_gb_hint")
    assert not hasattr(Resources(), "disk_gb")
    assert Resources().min_disk_gb is None


def test_it_does_NOT_imply_gpu() -> None:
    """The case that needed it first is a CPU-only conversion. `gpu_count>1`,
    `compute_capability` forces `gpu=True`; disk must
    not, or declaring a scratch floor would rent a card."""
    assert Resources(min_disk_gb=200).gpu is False
    # ...and the projection carries no `gpu` key at all when it is false,
    # which is the existing contract (`Resources(vcpus=4)` projects `{vcpus}`).
    assert "gpu" not in Resources(min_disk_gb=200).manifest_dict()


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
def test_a_nonpositive_floor_is_a_DECLARATION_time_error(bad: float) -> None:
    """Same posture as every other axis: a contradiction costs a
    declaration-time ValueError instead of a build, and 0 is a contradiction —
    "no floor" is spelled by not declaring one."""
    with pytest.raises(ValueError, match="min_disk_gb must be positive"):
        Resources(min_disk_gb=bad)


def test_an_int_declaration_is_normalized_to_float() -> None:
    """`64` and `64.0` must not reach the hub as two different JSON types."""
    res = Resources(min_disk_gb=64)
    assert isinstance(res.min_disk_gb, float)
    assert isinstance(res.manifest_dict()["min_disk_gb"], float)


def test_an_undeclared_floor_emits_NOTHING() -> None:
    """The common case is th#1233's derivation, and a key that is always
    present would turn "the author said nothing" into "the author said zero"
    at the one gate that cannot tell them apart."""
    projected: Dict[str, Any] = Resources(vcpus=4).manifest_dict()
    assert "min_disk_gb" not in projected


def test_it_composes_with_the_other_axes() -> None:
    res = Resources(gpu=True, ram_gb_hint=64,
                    min_disk_gb=200, vcpus=8)
    assert res.manifest_dict() == {
        "gpu": True,
        "ram_gb": 64.0,          # the ONE remapped key
        "min_disk_gb": 200.0,    # not remapped
        "vcpus": 8,
    }
