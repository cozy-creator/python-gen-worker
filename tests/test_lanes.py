"""th#913/gw#596 lane vocabulary — twin of tensorhub's
internal/orchestrator/precision/lane_test.go vectors (shared spec: ids and
semantics must match byte-for-byte across repos)."""

from __future__ import annotations

import pytest

from gen_worker.models import execution_lanes


def test_known_execution_lanes_stable() -> None:
    assert execution_lanes.known_execution_lanes() == [
        "fp8-w8a8-dynamic+compiled",
        "nvfp4-w4a4-static+compiled",
        "svdq-fp4-w4a4+eager",
        "svdq-int4-w4a4+eager",
        "bf16-w16a16+compiled",
        "bf16-w16a16+eager",
        "fp8-w8a16+compiled",
        "fp8-w8a16+eager",
    ]


def test_parse_execution_lane_round_trip() -> None:
    parsed = [execution_lanes.parse_execution_lane(execution_lane_id) for execution_lane_id in execution_lanes.known_execution_lanes()]
    assert [execution_lanes.execution_lane_id(execution_lane) for execution_lane in parsed] == execution_lanes.known_execution_lanes()
    assert {execution_lanes.execution_lane_body_id(execution_lane) for execution_lane in parsed} == set(execution_lanes.known_execution_lane_bodies())


@pytest.mark.parametrize("bad", [
    "", "bf16", "fp8", "4bit",
    "fp8-w8a8-dynamic",
    "fp8-w8a8-dynamic+eager",
    "fp8-w8a8+turbo",
    "fp8-w4a4-dynamic+compiled",
    "svdq-fp4-w4a4+compiled",
    "nvfp4-w4a4-static+eager",
    "int8-w8a8+eager",
])
def test_parse_execution_lane_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        execution_lanes.parse_execution_lane(bad)


def test_parse_execution_lane_spec_dual_form() -> None:
    spec = execution_lanes.parse_execution_lane_spec("bf16")
    assert spec.family == execution_lanes.FAMILY_BF16 and spec.execution_lane is None

    spec = execution_lanes.parse_execution_lane_spec("FP8-W8A8-Dynamic+Compiled")
    assert spec.family == execution_lanes.FAMILY_FP8
    assert spec.execution_lane is not None
    assert execution_lanes.execution_lane_id(spec.execution_lane) == "fp8-w8a8-dynamic+compiled"

    assert execution_lanes.parse_execution_lane_spec("").is_zero
    with pytest.raises(ValueError):
        execution_lanes.parse_execution_lane_spec("int8")


# pgw#1148: the `flavor` argument is GONE (§1.32(d)). A BINDING names a tag or
# a digest and declares at most a CAST; it no longer asserts a stored quant.
@pytest.mark.parametrize("storage,compiled,want", [
    ("", False, "bf16-w16a16+eager"),
    ("", True, "bf16-w16a16+compiled"),
    ("fp8", False, "fp8-w8a16+eager"),
    ("fp8", True, "fp8-w8a16+compiled"),
    ("fp8+te", False, "fp8-w8a16+eager"),
])
def test_execution_lane_of_binding(storage: str, compiled: bool, want: str) -> None:
    assert execution_lanes.execution_lane_id(execution_lanes.execution_lane_of_binding(storage, compiled)) == want


def test_execution_lane_of_binding_is_always_a_valid_lane() -> None:
    """The stored-quant BODIES are still in the table — they are just no
    longer reachable from a binding. They arrive from the hub-resolved lane
    and from setup()'s applied report (pgw#1104), both of which carry
    evidence rather than a token in a ref."""
    bodies = set()
    for storage in ("", "fp8", "fp8+te"):
        for compiled in (False, True):
            execution_lane = execution_lanes.execution_lane_of_binding(storage, compiled)
            assert execution_lanes.valid_execution_lane(execution_lane)
            bodies.add(execution_lanes.execution_lane_body_id(execution_lane))
    assert bodies == {"bf16-w16a16", "fp8-w8a16"}
    assert bodies < set(execution_lanes.known_execution_lane_bodies())
