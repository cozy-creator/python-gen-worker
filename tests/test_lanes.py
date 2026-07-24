"""th#913/gw#596 lane vocabulary — twin of tensorhub's
internal/orchestrator/precision/lane_test.go vectors (shared spec: ids and
semantics must match byte-for-byte across repos)."""

from __future__ import annotations

import pytest

from gen_worker.models import lanes


def test_known_lanes_stable() -> None:
    assert lanes.known_lanes() == [
        "fp8-w8a8-dynamic+compiled",
        "fp8-w8a8-dynamic+eager",
        "nvfp4-w4a4-static+compiled",
        "nvfp4-w4a4-static+eager",
        "svdq-fp4-w4a4+eager",
        "svdq-int4-w4a4+eager",
        "bf16-w16a16+compiled",
        "bf16-w16a16+eager",
        "fp8-w8a16+compiled",
        "fp8-w8a16+eager",
    ]


def test_parse_lane_round_trip() -> None:
    for lane_id in lanes.known_lanes():
        assert lanes.lane_id(lanes.parse_lane(lane_id)) == lane_id


@pytest.mark.parametrize("bad", [
    "", "bf16", "fp8", "4bit",
    "fp8-w8a8-dynamic",
    "fp8-w8a8+turbo",
    "fp8-w4a4-dynamic+compiled",
    "svdq-fp4-w4a4+compiled",
    "int8-w8a8+eager",
])
def test_parse_lane_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        lanes.parse_lane(bad)


def test_parse_lane_spec_dual_form() -> None:
    spec = lanes.parse_lane_spec("bf16")
    assert spec.family == lanes.FAMILY_BF16 and spec.lane is None

    spec = lanes.parse_lane_spec("FP8-W8A8-Dynamic+Compiled")
    assert spec.family == lanes.FAMILY_FP8
    assert spec.lane is not None
    assert lanes.lane_id(spec.lane) == "fp8-w8a8-dynamic+compiled"

    assert lanes.parse_lane_spec("").is_zero
    with pytest.raises(ValueError):
        lanes.parse_lane_spec("int8")


@pytest.mark.parametrize("flavor,storage,compiled,want", [
    ("", "", False, "bf16-w16a16+eager"),
    ("", "", True, "bf16-w16a16+compiled"),
    ("", "fp8", False, "fp8-w8a16+eager"),
    ("fp8", "", True, "fp8-w8a16+compiled"),
    ("fp8-w8a8", "", True, "fp8-w8a8-dynamic+compiled"),
    ("fp8-w8a8-cal1", "", False, "fp8-w8a8-dynamic+eager"),
    ("svdq-fp4-r128", "", True, "svdq-fp4-w4a4+eager"),
    ("svdq-int4-r128", "", False, "svdq-int4-w4a4+eager"),
    ("nvfp4-w4a4", "", True, "nvfp4-w4a4-static+compiled"),
])
def test_lane_of_binding(flavor: str, storage: str, compiled: bool, want: str) -> None:
    assert lanes.lane_id(lanes.lane_of_binding(flavor, storage, compiled)) == want
