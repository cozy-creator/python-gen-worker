"""pgw#1314 — `cuda_version` reaches the hub from a LIVE worker.

`min_cuda` is a term in the one compatibility vocabulary (pgw#1313), and a
requirement term is only real if the machine FACT it is compared against
arrives. This one was measured (`HostFacts.cuda_version`) and LOCAL ONLY: the
hub's single carrier for it was `HardwareUnsuitable.torch_cuda_version`, so it
could learn a worker's CUDA version from exactly one kind of worker — one that
had already died.

Everything here drives the ONE on-the-wire builder
(`procsplit.parent.ParentControl._parent_resources`, pgw#898) and the Hello
path that uses it. A second builder is a second answer.
"""

from __future__ import annotations

from typing import Any

import pytest

from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit.parent import ParentControl
from gen_worker.topology import ExecutionTopology

_MEASURED: dict[str, Any] = {
    "hardware": {
        "gpu_count": 1,
        "vram_total_bytes": 85899345920,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_sm": "90",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "driver_version": "580.159.04",
        "installed_libs": ["torchao"],
    },
    "gen_worker_version": "0.118.0",
}


def _parent() -> ParentControl:
    return ParentControl(
        load_settings(orchestrator_public_addr="127.0.0.1:1",
                      worker_id="w-pgw1314"),
        socket_path="/tmp/gen-worker-pgw1314.sock",
        topology=ExecutionTopology.single(),
    )


def _wire_fields(msg: pb.WorkerResources) -> set[str]:
    """What actually goes on the wire — proto3 serializes a singular scalar
    only when it differs from the default."""
    return {field.name for field, _ in msg.ListFields()}


def test_a_live_workers_resources_carry_the_measured_cuda_version() -> None:
    pc = _parent()
    pc._measurement = dict(_MEASURED)
    res = pc._parent_resources()
    assert res is not None
    assert res.cuda_version == "13.0"
    assert "cuda_version" in _wire_fields(res)


def test_it_rides_the_HELLO_the_hub_receives_not_just_the_builder() -> None:
    """The assertion that survives a refactor of the Hello path: the field is
    on the message the transport serializes, through the real
    `_apply_identity_and_resources` delta."""
    pc = _parent()
    pc._measurement = dict(_MEASURED)
    hello = pb.Hello(worker_id="stale-child-claim")
    pc._apply_identity_and_resources(hello)
    assert hello.resources.cuda_version == "13.0"
    round_tripped = pb.Hello()
    round_tripped.ParseFromString(hello.SerializeToString())
    assert round_tripped.resources.cuda_version == "13.0"


def test_an_unreadable_cuda_runtime_still_HELLOS_and_the_field_is_ABSENT(
) -> None:
    """The always-runs posture, on this axis: a host whose CUDA runtime cannot
    be read is not a `HardwareUnsuitable` — it Hellos, with the CUDA axis
    UNDECLARED. Absent and zero must stay distinguishable, which is the same
    rule the requirement side keeps: `""`-as-a-value would read as a measured
    answer, and no reader can tell it from a measurement of nothing."""
    hardware: dict[str, Any] = {**_MEASURED["hardware"], "cuda_version": ""}
    measurement: dict[str, Any] = {**_MEASURED, "hardware": hardware}
    pc = _parent()
    pc._measurement = measurement

    res = pc._parent_resources()
    assert res is not None, "an unreadable CUDA runtime is not a dead parent"
    assert "cuda_version" not in _wire_fields(res)
    # ...and the rest of the measurement is untouched: the axis is undeclared,
    # not the machine unmeasured.
    assert res.gpu_sm == "90" and res.driver_version == "580.159.04"

    hello = pb.Hello()
    pc._apply_identity_and_resources(hello)
    assert hello.HasField("resources")
    assert "cuda_version" not in _wire_fields(hello.resources)


def test_an_unmeasured_HOST_still_ships_no_resources_at_all() -> None:
    """pgw#898's rule, unweakened: absent measurement means NO `resources`,
    loudly — never a partially-filled struct with this new field in it."""
    pc = _parent()
    pc._measurement = None
    assert pc._parent_resources() is None
    hello = pb.Hello()
    pc._apply_identity_and_resources(hello)
    assert not hello.HasField("resources")


def test_the_fact_is_spelled_the_same_on_both_carriers() -> None:
    """ONE vocabulary. The failure carrier already had the fact under
    `torch_cuda_version`; the live carrier is the addition, and the value is
    the same measurement from the same producer — not a second answer."""
    live = {f.name for f in pb.WorkerResources.DESCRIPTOR.fields}
    corpse = {f.name for f in pb.HardwareUnsuitable.DESCRIPTOR.fields}
    assert "cuda_version" in live
    assert "torch_cuda_version" in corpse
    # The SM axis settles the one real spelling trap: `HostFacts.gpu_sm` and
    # `WorkerResources.gpu_sm` are BARE ("90"), which is also the spelling
    # `min_sm` uses. Anything dotted is a normalization at its own boundary.
    pc = _parent()
    pc._measurement = dict(_MEASURED)
    res = pc._parent_resources()
    assert res is not None and res.gpu_sm == "90"


@pytest.mark.parametrize("reserved", ["worker_mode"])
def test_the_new_field_did_not_reclaim_a_retired_number(reserved: str) -> None:
    names = {f.name: f.number for f in pb.WorkerResources.DESCRIPTOR.fields}
    assert reserved not in names
    assert names["cuda_version"] == 14
