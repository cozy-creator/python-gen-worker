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
with `WORKER_MODE=forge` present in the container env the whole time. The
fingerprint that named it was `worker_mode=""` rather than `"serve"`: the
field's own default is `"serve"`, so an EMPTY string is the signature of a
field that was never assigned, distinguishable from one assigned a default.

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

import pytest

from gen_worker import lifecycle as lifecycle_mod
from gen_worker import config as gw_config
from gen_worker import worker_goals
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
        # pgw#1129/th#1798: the HOST driver. 580.159.04 is a real RunPod draw
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
    monkeypatch.setenv("WORKER_MODE", "forge")
    # pgw#930: the wire value is projected from the PUBLISHED goal set, not
    # re-derived from env at the builder. Seeding the declaration therefore
    # means installing the goals it seeds — which is what a process entry does.
    settings = gw_config.reload_for_test()
    worker_goals.install(worker_goals.from_settings(settings))
    pc = _parent_control(
        worker_mode="forge",
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
    assert res.worker_mode == "forge"


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
