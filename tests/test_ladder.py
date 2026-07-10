"""Precision-ladder classification + placement schema (th#697 P1)."""

from __future__ import annotations

import pytest

from gen_worker.models.ladder import (
    CLASS_BASE,
    CLASS_FP8,
    CLASS_NVFP4,
    CLASS_SVDQ_FP4,
    CLASS_SVDQ_INT4,
    Placement,
    classify_flavor_token,
    default_placement,
    placement_for_flavor,
    placement_from_metadata,
    placement_to_metadata,
)
from gen_worker.models.svdq import SVDQ_FP4_SMS, SVDQ_INT4_SMS


@pytest.mark.parametrize(
    "token,cls",
    [
        ("", CLASS_BASE),
        ("bf16", CLASS_BASE),
        ("fp16", CLASS_BASE),
        ("fp8", CLASS_FP8),
        ("FP8", CLASS_FP8),
        ("svdq-fp4-r128", CLASS_SVDQ_FP4),
        ("svdq-fp4-r32", CLASS_SVDQ_FP4),
        ("svdq-int4-r128", CLASS_SVDQ_INT4),
        ("nvfp4", CLASS_NVFP4),
        ("gguf-q4km", ""),
        ("trt-rtx-4090-trt10.16-fp16", ""),
        ("vae-fix", ""),
    ],
)
def test_classify_flavor_token(token: str, cls: str) -> None:
    assert classify_flavor_token(token) == cls


def test_default_placements_match_svdq_windows() -> None:
    fp4 = default_placement(CLASS_SVDQ_FP4)
    int4 = default_placement(CLASS_SVDQ_INT4)
    assert fp4.sm_allowed == tuple(SVDQ_FP4_SMS) and fp4.engines == ("nunchaku",)
    assert int4.sm_allowed == tuple(SVDQ_INT4_SMS) and int4.engines == ("nunchaku",)
    # fp8-storage serves anywhere: no constraints
    fp8 = default_placement(CLASS_FP8)
    assert fp8.sm_allowed == () and fp8.sm_min == 0 and fp8.engines == ()
    assert default_placement(CLASS_NVFP4).sm_min == 100
    assert default_placement("bogus") is None


@pytest.mark.parametrize(
    "gpu_sm,fp4_ok,int4_ok",
    [(120, True, False), (121, True, False), (100, False, False),
     (90, False, False), (89, False, True), (86, False, True), (70, False, False)],
)
def test_admits_sm(gpu_sm: int, fp4_ok: bool, int4_ok: bool) -> None:
    assert placement_for_flavor("svdq-fp4-r128").admits_sm(gpu_sm) is fp4_ok
    assert placement_for_flavor("svdq-int4-r128").admits_sm(gpu_sm) is int4_ok
    assert placement_for_flavor("fp8").admits_sm(gpu_sm) is True


def test_metadata_roundtrip() -> None:
    p = Placement(CLASS_SVDQ_INT4, sm_allowed=(75, 80, 86, 89), engines=("nunchaku",))
    block = placement_to_metadata(p)
    assert block == {
        "precision_class": "svdq-int4",
        "sm_allowed": [75, 80, 86, 89],
        "engines": ["nunchaku"],
    }
    assert placement_from_metadata(block) == p
    # whole checkpoint metadata bag with the block nested under "placement"
    assert placement_from_metadata({"placement": block, "kind": "model"}) == p


def test_metadata_parse_fails_soft() -> None:
    assert placement_from_metadata(None) is None
    assert placement_from_metadata({}) is None
    assert placement_from_metadata({"placement": "fp8"}) is None
    assert placement_from_metadata({"precision_class": ""}) is None
    assert placement_from_metadata({"precision_class": "fp8", "sm_allowed": ["x"]}) is None
