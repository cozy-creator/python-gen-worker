"""pgw#876 §4 — the recurrence guard for "two builders for the same fact, only
one of which is ever on the wire".

`WorkerResources` is built in two places: `lifecycle.build_resources()` in the
COMPUTE CHILD, and `procsplit.parent.ParentControl._parent_resources()` in the
CONTROL PARENT. The process split is UNCONDITIONAL, and
`_apply_identity_and_resources` overwrites the child's copy wholesale (or
clears the field outright when the parent has no measurement), so **only the
parent's copy ever reaches the hub.**

th#1359 Part 2 taught `worker_mode` to the child's builder only. Every forge
pod bought after that shipped the protobuf default `""` and was idle-reaped as
`cold_idle_never_dispatched` — two pods at 391 s each, measured 2026-08-02,
with `WORKER_MODE=forge` present in the container env the whole time.

`worker_mode` itself is now `reserved 12` (§4.28 / th#1751 W4 + pgw#1092) and
the field is gone from both builders, but the GUARD it produced is the point
and is what this file keeps: any field only one builder assigns is dead on the
wire, and the value-level row below is how that is caught before a pod pays.

These two rows make that fingerprint a red test instead of a paid pod:

* the parent's builder must assign every wire field (except the documented
  exemption), proved at the VALUE level — feed it all-distinct inputs and no
  field may come back at its protobuf default;
* the two builders must assign the SAME field set, proved at the SOURCE level —
  so teaching one of them a new field and not the other goes red here rather
  than on a pod.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from gen_worker import lifecycle as lifecycle_mod
from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit.parent import ParentControl
from gen_worker.topology import ExecutionTopology

# `git_commit` is deliberately unpopulated by BOTH builders: no launcher ever
# set WORKER_GIT_COMMIT and Go never read WorkerResources.git_commit, so it is
# dead on both ends (pgw#514/P4). The field stays on the wire because deleting
# it needs a coordinated tensorhub proto update.
_INTENTIONALLY_UNSET = frozenset({"git_commit"})

# Every field the parent's measurement subprocess can fill, with a value that
# is distinct from the protobuf default for its type.
_MEASUREMENT = {
    "hardware": {
        "gpu_count": 4,
        "gpu_total_mem": 85899345920,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_sm": "90",
        "torch_version": "2.13.0+cu130",
        "installed_libs": ["diffusers==0.36.0"],
        # the HOST driver. 580.159.04 is a real RunPod draw
        # and the tuple-vs-float trap (as floats 580.159 < the 580.65 floor).
        "driver_version": "580.159.04",
    },
    "canary": {
        "memcpy_gbps": 1.5,
        "d2h_gbps": 2.5,
        "pinned_alloc_ok": True,
        "cpu_single_mbps": 3.5,
        "cpu_multi_mbps": 4.5,
        "vcpus": 32,
        "ram_total_gb": 251.0,
        "duration_ms": 1234,
        "interconnect": "nvlink",
        "peer_gbps": 5.5,
        "peer_access": True,
        "topo_link": "NV18",
    },
    "gen_worker_version": "0.90.6",
}


def _parent_control(**settings_kw: object) -> ParentControl:
    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1",
        worker_id="w-pgw876",
        **settings_kw,
    )
    return ParentControl(
        settings,
        socket_path="/tmp/gen-worker-pgw876.sock",
        topology=ExecutionTopology.single(),
    )


def _default_valued_fields(msg: pb.WorkerResources) -> set:
    """Field names still carrying the protobuf default — i.e. never assigned.

    `ListFields()` is exactly the "what would actually go on the wire" test:
    proto3 serializes a singular scalar only when it differs from the default,
    a message field only when present, a repeated field only when non-empty.
    """
    on_the_wire = {field.name for field, _ in msg.ListFields()}
    return {f.name for f in pb.WorkerResources.DESCRIPTOR.fields} - on_the_wire


def test_the_parent_builder_assigns_every_wire_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE `worker_mode=""` GUARD, at the value level.

    Every input is distinct from its type's protobuf default, so any field
    that comes back at the default was NEVER ASSIGNED by the builder the hub
    actually receives. That is the exact fingerprint th#1359 Part 2 left on two
    forge pods.
    """
    pc = _parent_control(
        worker_image_digest="sha256:deadbeef",
        runpod_pod_id="pod-pgw876",
    )
    pc._measurement = dict(_MEASUREMENT)

    res = pc._parent_resources()
    assert res is not None

    unset = _default_valued_fields(res) - _INTENTIONALLY_UNSET
    assert not unset, (
        f"the ON-THE-WIRE WorkerResources builder never assigns {sorted(unset)}. "
        "An empty value here is not a default the hub can read as a choice — it "
        "is the signature of a field taught to `lifecycle.build_resources()` "
        "(the compute child's builder, which the parent overwrites wholesale) "
        "instead of to `_parent_resources()`. th#1359 Part 2 did exactly that "
        "with `worker_mode` and every forge pod bought afterwards was "
        "idle-reaped as a serving pod."
    )


def test_the_retired_wire_words_are_gone_from_the_contract() -> None:
    """§4.28 / th#1751 W4 + pgw#1092 — the vocabulary cut, at the DESCRIPTOR
    and in the contract text.

    RED before this change: both names resolved to live fields (12 and 11) and
    a stale `WORKER_MODE=forge` could still be echoed to the hub.
    """
    assert "worker_mode" not in {
        f.name for f in pb.WorkerResources.DESCRIPTOR.fields}
    assert "requested_cell_axes" not in {
        f.name for f in pb.CompileTarget.DESCRIPTOR.fields}
    # ...and no lane may reclaim the numbers or the names (§1.27(f): a
    # within-major retirement reserves BOTH). The python descriptor does not
    # expose reserved ranges, so the vendored contract itself is the assertion.
    contract = (
        Path(__file__).resolve().parents[1] / "proto" / "worker_scheduler.proto"
    ).read_text()
    assert "reserved 12;" in contract and 'reserved "worker_mode";' in contract
    assert "reserved 11;" in contract
    assert 'reserved "requested_cell_axes";' in contract


def _append_unknown_string(wire: bytearray, field_no: int, value: bytes) -> None:
    """Append `field_no` as a length-delimited (wire type 2) string."""
    from google.protobuf.internal import encoder as _encoder

    _encoder._VarintEncoder()(wire.extend, (field_no << 3) | 2, False)
    _encoder._VarintEncoder()(wire.extend, len(value), False)
    wire.extend(value)


def test_a_wheel_that_still_sends_the_retired_fields_is_not_refused() -> None:
    """THE FLEET-SAFETY CLAIM, proved rather than asserted in a PR body.

    Wheels <= 0.112.0 SEND `worker_mode` (field 12), and a 0.94-vintage one
    sends `requested_cell_axes` (field 11). proto3 has no strict mode: an
    unknown field is skipped at decode and every KNOWN field parses normally,
    so a peer on the new contract reads an old worker's message exactly as
    before. This is why the hub half of the cut is safe to land while the fleet
    still runs old wheels.

    RED before this change is not available by construction (the fields WERE
    known), which is the point: the row exists to fail if anyone ever makes the
    decode strict.
    """
    old = pb.WorkerResources(gpu_count=4, gpu_sm="90")
    wire = bytearray(old.SerializeToString())
    _append_unknown_string(wire, 12, b"forge")

    fresh = pb.WorkerResources()
    assert fresh.ParseFromString(bytes(wire)) == len(wire)
    assert fresh.gpu_count == 4 and fresh.gpu_sm == "90"
    assert not hasattr(fresh, "worker_mode")

    # Same for the CompileTarget half: field 11 was a map, whose entries are
    # also length-delimited, so a single entry is the honest shape to feed it.
    target = pb.CompileTarget(family="sdxl", requested_cell_key="ck1-abc")
    twire = bytearray(target.SerializeToString())
    _append_unknown_string(twire, 11, b"\n\x03sku\x12\x04L40S")

    got = pb.CompileTarget()
    assert got.ParseFromString(bytes(twire)) == len(twire)
    assert got.family == "sdxl" and got.requested_cell_key == "ck1-abc"
    assert not hasattr(got, "requested_cell_axes")


def _assigned_field_names(func: object) -> set:
    """The `pb.WorkerResources(...)` keyword names one builder assigns."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == "WorkerResources":
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError(f"{func!r} no longer constructs a pb.WorkerResources")


def test_the_two_resources_builders_assign_the_same_fields() -> None:
    """Both builders exist; only the parent's is on the wire. Until there is
    ONE builder, they must at least stay field-for-field identical, so a lane
    adding a fact does not have to know which of the two is live."""
    child = _assigned_field_names(lifecycle_mod.Lifecycle.build_resources)
    parent = _assigned_field_names(ParentControl._parent_resources)
    assert child == parent, (
        "the two WorkerResources builders have drifted: "
        f"child-only={sorted(child - parent)} parent-only={sorted(parent - child)}. "
        "`procsplit/parent.py::_parent_resources` is the one the hub receives — "
        "the process split is unconditional and the parent overwrites the "
        "child's copy wholesale. A field only the child sets is dead on the wire."
    )
