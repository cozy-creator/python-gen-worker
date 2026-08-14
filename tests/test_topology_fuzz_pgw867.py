"""pgw#867 / th#1382 — property tests over the WORKER_EXECUTION_TOPOLOGY decoder.

Why this file exists, stated as the bug it would have caught. The contract's
docstring said an unrecognised payload was a typed refusal; the decoder in fact
IGNORED unknown keys, and 5 of 5 version-skew payloads decoded clean against the
shipped worker. Ignoring a key reads the field it names as
ABSENT, absent is legal, and absent means one slot — so a hub that bought degree
2 was served degree 1, silently, with nothing logged anywhere. The sibling defect
was ``_opt("gpu_count") or 1``, which laundered a ``gpu_count=0`` REFUSAL into
the legal single-slot default.

Neither is a crash. A fuzzer that checks for crashes finds neither. So the
properties here are:

  P1  only ``TopologyError`` escapes ``decode``. A bare ``ValueError`` or
      ``OverflowError`` is an UNTYPED refusal that every caller catching
      ``TopologyError`` will miss.
  P2  ``decode(as_dict(t)) == t`` — round trip, including the th#1375 legacy
      spellings ``as_dict`` still emits.
  P3  acceptance implies the PARTITION INVARIANT holds: gpu_count =
      execution_groups × gpus_per_execution_group, with the parallel/degree
      coherence rules. This is what makes "silently served the wrong packing"
      unreachable rather than merely unobserved.
  P4  the Python side answers every vector in ``testdata/topology_wire_vectors
      .json`` as recorded (the merge-path half; the random cross-language search
      lives in ``scripts/topology_differential_pgw867.py``, not here).

hypothesis over atheris: these decoders are shallow, so libFuzzer's coverage
feedback buys little, and ``@example`` puts each historical defect in the source
as an always-run regression case rather than an opaque corpus blob.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from gen_worker.topology import (
    KEY_EXECUTION_GROUPS,
    KEY_GPU_COUNT,
    KEY_GPUS_PER_GROUP,
    KEY_PARALLEL,
    PARALLEL_CFG,
    PARALLEL_INTERNAL,
    PARALLEL_NONE,
    PARALLEL_SEQUENCE,
    MAX_GPU_COUNT,
    ExecutionTopology,
    TopologyError,
)

VECTORS = pathlib.Path(__file__).parent / "testdata" / "topology_wire_vectors.json"

_LEGAL_PARALLEL = (PARALLEL_NONE, PARALLEL_INTERNAL, PARALLEL_SEQUENCE, PARALLEL_CFG)

def _decode(raw: str) -> ExecutionTopology:
    """decode, asserting P1: nothing but TopologyError may escape.

    pgw#870 was the live exception — ``json.loads`` accepts the non-standard
    ``NaN``/``Infinity`` literals and ``int()`` of them raised untyped. Fixed:
    the decoder refuses a non-integer before anything can coerce it, so there is
    no named exception here any more and ANY untyped escape fails.
    """
    try:
        return ExecutionTopology.decode(raw)
    except TopologyError:
        raise
    except Exception as exc:  # noqa: BLE001 - an untyped escape IS the defect
        raise AssertionError(
            f"decode({raw!r}) raised an UNTYPED {type(exc).__name__}: {exc} — every "
            "refusal must be a TopologyError, or callers that catch TopologyError "
            "crash instead of refusing"
        ) from exc


def _assert_partition(topo: ExecutionTopology, source: object) -> None:
    """P3: an accepted topology describes a real packing."""
    assert topo.gpu_count >= 1, f"{source!r} accepted gpu_count={topo.gpu_count}"
    assert topo.gpus_per_execution_group >= 1, f"{source!r} accepted degree < 1"
    assert topo.gpu_count % topo.gpus_per_execution_group == 0, (
        f"{source!r} accepted a packing where a group cannot exclusively own its devices"
    )
    assert topo.execution_groups * topo.gpus_per_execution_group == topo.gpu_count
    assert topo.parallel in _LEGAL_PARALLEL, f"{source!r} accepted parallel={topo.parallel!r}"
    if topo.gpus_per_execution_group > 1:
        assert topo.parallel != PARALLEL_NONE, (
            f"{source!r} accepted a {topo.gpus_per_execution_group}-card group with no "
            "mechanism: it would hold those cards and serve one card's worth"
        )
    else:
        assert topo.parallel == PARALLEL_NONE, (
            f"{source!r} accepted parallel={topo.parallel!r} at degree 1"
        )
    # The decoder now has a CEILING as well as a floor
    # (``MAX_GPU_COUNT``), so enumerating every group of an accepted topology is
    # bounded work and this property no longer needs an escape hatch.
    assert topo.gpu_count <= MAX_GPU_COUNT, (
        f"{source!r} accepted gpu_count={topo.gpu_count} above the ceiling"
    )
    # Every group's devices are disjoint, in range, and cover the pod exactly.
    seen: set[int] = set()
    for g in range(topo.execution_groups):
        devices = topo.group(g).devices
        assert len(devices) == topo.gpus_per_execution_group
        assert not (seen & set(devices)), f"{source!r}: group {g} shares a device"
        seen |= set(devices)
    assert seen == set(range(topo.gpu_count)), (
        f"{source!r}: groups cover {sorted(seen)}, not the pod's {topo.gpu_count} cards"
    )


# --------------------------------------------------------------------------
# P4 — the shared, byte-identical cross-language fixture. This is the merge-path
# half of the differential and is milliseconds.
# --------------------------------------------------------------------------

def _fixture() -> dict[str, Any]:
    return json.loads(VECTORS.read_text())


@pytest.mark.parametrize("vector", _fixture()["agreed"], ids=lambda v: v["wire"] or "<empty>")
def test_agreed_wire_vectors(vector: dict[str, Any]) -> None:
    """Every vector both decoders must agree on, asserted against this one."""
    wire = vector["wire"]
    note = vector.get("note", "")
    if not vector["accept"]:
        with pytest.raises(TopologyError) as excinfo:
            ExecutionTopology.decode(wire)
        assert excinfo.value.code == vector["code"], f"{wire!r} ({note})"
        return
    if vector.get("code") == "absent":
        # "No decision delivered". The Python side spells that as the single
        # slot from `from_env`; `decode` is never called on a blank value.
        assert ExecutionTopology.from_env({}) == ExecutionTopology.single()
        return
    topo = ExecutionTopology.decode(wire)
    assert topo.gpu_count == vector["gpu_count"], note
    assert topo.gpus_per_execution_group == vector["gpus_per_execution_group"], note
    assert topo.execution_groups == vector["execution_groups"], note
    assert topo.parallel == vector.get("parallel", ""), note
    _assert_partition(topo, wire)


# --------------------------------------------------------------------------
# P1/P2/P3 — the search half.
# --------------------------------------------------------------------------

_VALUES = st.one_of(
    st.integers(min_value=-4, max_value=64),
    st.integers(min_value=-(2 ** 70), max_value=2 ** 70),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.none(),
    st.text(max_size=8),
    st.lists(st.integers(), max_size=2),
    st.dictionaries(st.text(max_size=3), st.integers(), max_size=2),
)

_KEYS = st.sampled_from([
    KEY_GPU_COUNT, KEY_GPUS_PER_GROUP, KEY_EXECUTION_GROUPS, KEY_PARALLEL,
    # Keys that are NOT in the closed set: the retired pre-rename spellings,
    # the near-misses a rename produces, and the case variants the Go side is
    # known to mis-handle.
    "group_degree", "groups",
    "gpus_per_group", "group_size", "GPU_COUNT", "tensor_parallel_size", "", "0",
])


@settings(max_examples=400, deadline=None)
@given(st.dictionaries(_KEYS, _VALUES, max_size=7))
# Seeds from the shipped defects — always run, whatever hypothesis draws.
@example({KEY_GPU_COUNT: 0, KEY_GPUS_PER_GROUP: 1})                       # the `or 1` launder
@example({KEY_GPU_COUNT: 0, "group_degree": 1})
@example({KEY_GPU_COUNT: 4, KEY_GPUS_PER_GROUP: 2, "gpus_per_group": 2})  # unknown field
@example({KEY_GPU_COUNT: 4, "tensor_parallel_size": 2})
@example({KEY_GPU_COUNT: 4, KEY_GPUS_PER_GROUP: 2, "group_degree": 1,
          KEY_PARALLEL: PARALLEL_SEQUENCE})                               # retired spelling
@example({KEY_GPU_COUNT: 4, KEY_GPUS_PER_GROUP: 2, KEY_EXECUTION_GROUPS: 4,
          KEY_PARALLEL: PARALLEL_SEQUENCE})                               # groups disagree
@example({KEY_GPU_COUNT: 4, KEY_GPUS_PER_GROUP: 3, KEY_PARALLEL: PARALLEL_SEQUENCE})
@example({KEY_GPU_COUNT: 4, KEY_GPUS_PER_GROUP: 2})                       # parallel required
@example({KEY_GPU_COUNT: 4, KEY_PARALLEL: PARALLEL_SEQUENCE})             # parallel w/o degree
@example({KEY_GPU_COUNT: 1, KEY_PARALLEL: False})                         # the `or ""` launder
@example({KEY_GPU_COUNT: 1, KEY_PARALLEL: 0})
@example({KEY_GPU_COUNT: 2.0})
@example({KEY_GPU_COUNT: 10 ** 30})
def test_decode_is_typed_and_sound(payload: dict[str, Any]) -> None:
    """P1 + P3 over structured payloads."""
    try:
        raw = json.dumps(payload)
    except (TypeError, ValueError):
        assume(False)
        return
    try:
        topo = _decode(raw)
    except TopologyError:
        return  # a typed refusal is a correct outcome; P1 is what was under test
    _assert_partition(topo, payload)


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=120))
@example("")
@example("{}")
@example("null")
@example('{"gpu_count":1}garbage')
@example('{"gpu_count":NaN}')
@example('{"gpu_count":Infinity}')
def test_decode_of_arbitrary_text_is_typed(raw: str) -> None:
    """P1 over arbitrary bytes — the decoder is total over strings."""
    try:
        topo = _decode(raw)
    except TopologyError:
        return
    _assert_partition(topo, raw)


@settings(max_examples=300, deadline=None)
@given(
    st.integers(min_value=1, max_value=64),
    st.integers(min_value=1, max_value=64),
    st.sampled_from(_LEGAL_PARALLEL),
)
def test_round_trip(gpu_count: int, degree: int, parallel: str) -> None:
    """P2: ``decode(as_dict(t)) == t`` for every constructible topology.

    ``as_dict`` emits the canonical spellings ONLY, so this is also
    the assertion that the dual write is gone and the round trip never
    depended on it.
    """
    try:
        topo = ExecutionTopology(
            gpu_count=gpu_count, gpus_per_execution_group=degree, parallel=parallel
        )
    except TopologyError:
        return
    payload = topo.as_dict()
    # the canonical spelling ONLY -- the retired names are neither written
    # nor read.
    assert "group_degree" not in payload
    assert "groups" not in payload
    back = _decode(json.dumps(payload))
    assert back == topo, f"round trip changed the value: {topo} -> {payload} -> {back}"
    assert _decode(json.dumps(back.as_dict())) == back, "as_dict is not a fixed point"
    _assert_partition(back, payload)


@settings(max_examples=200, deadline=None)
@given(st.integers(min_value=1, max_value=32), st.integers(min_value=1, max_value=32))
def test_group_ordinal_is_exact_for_rank0_devices(gpu_count: int, degree: int) -> None:
    """Dispatch translation: the hub always names a group's rank-0 device, and
    ``group_ordinal_exact`` must agree with ``group`` for exactly those.

    The floored variant exists so a single-group pod cannot index off the end;
    on a wide pod flooring is the silent bug — every dispatch the hub
    got wrong lands on group 0, which is also the busiest group.
    """
    assume(gpu_count % degree == 0)
    parallel = PARALLEL_NONE if degree == 1 else PARALLEL_SEQUENCE
    topo = ExecutionTopology(
        gpu_count=gpu_count, gpus_per_execution_group=degree, parallel=parallel
    )
    for g in range(topo.execution_groups):
        rank0 = g * degree
        assert topo.group_ordinal_exact(rank0) == g
        assert topo.group(g).devices[0] == rank0
    for idx in range(gpu_count):
        if idx % degree == 0:
            continue
        with pytest.raises(TopologyError) as excinfo:
            topo.group_ordinal_exact(idx)
        assert excinfo.value.code == "topology_dispatch_gpu_index_invalid"


def test_nonfinite_numbers_are_typed_refusals_pgw870() -> None:
    """pgw#870 FIXED, untyped-escape half — revert-turns-red guard.

    ``json.loads`` accepts the non-standard ``NaN``/``Infinity`` literals, and
    ``int()`` of either raises ``OverflowError``/``ValueError`` — neither is a
    ``TopologyError``, so a caller that catches ``TopologyError`` around
    ``from_env`` crashed at boot instead of refusing. The decoder now refuses a
    non-integer before anything coerces it, so both are ordinary typed
    refusals.
    """
    for raw in ('{"gpu_count":1e400}', '{"gpu_count":-1e400}', '{"gpu_count":NaN}',
                '{"gpu_count":Infinity}'):
        with pytest.raises(TopologyError) as excinfo:
            ExecutionTopology.decode(raw)
        assert excinfo.value.code == "topology_decode_failed", raw
