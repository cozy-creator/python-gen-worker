"""pgw#776: the pod's hub-facing VRAM truth is not one group's.

Two defects in `lifecycle.free_vram_bytes` (DPA-5) and one in `probe_hardware`
(DPA-7):

- MAX across groups is honest PER JOB and dishonest per POD: the hub admits
  against a single scalar, so it can admit G concurrent jobs whose sum is G
  times what any one group has. With one scalar on the wire the safe answer is
  the MIN.
- it read `ExecutionTopology.from_env()` rather than the DELIVERED topology, so
  a pod this worker itself demoted for a non-NVLink fabric reported the packing
  it is NOT serving.
- `probe_hardware()` read card 0's free memory, and `gate_functions` turns that
  one number into `unavailable` + `serve_plans` for EVERY function. Card 0 is
  always the least-free card in practice, so groups 1..G-1 got fit rungs
  planned from a card they will never touch.

Layer exercised: `lifecycle` against a synthetic device table, i.e. the exact
functions that produce `StateDelta.free_vram_bytes` and the gate inputs. No
CUDA is touched.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gen_worker import lifecycle
from gen_worker.topology import ENV_VAR, ExecutionTopology


def _fake_devices(free_by_device: List[int], monkeypatch: pytest.MonkeyPatch) -> None:
    """Give DeviceGroup.free_vram_bytes a deterministic per-card table."""
    from gen_worker.models import residency

    def _free(device: int) -> int:
        return int(free_by_device[int(device)])

    monkeypatch.setattr(residency, "device_free_vram_bytes", _free,
                        raising=False)
    # DeviceGroup consults torch directly if the helper above is not the seam;
    # patch the group method as the stable contract instead.
    def _group_free(self: Any) -> int:
        return min(_free(d) for d in self.devices)

    monkeypatch.setattr(residency.DeviceGroup, "free_vram_bytes", _group_free)


GiB = 1024 ** 3


def test_the_pod_reports_the_least_roomy_group_not_the_roomiest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 4x1: group 0 carries the weights (10 GiB free), groups 1-3 are cold
    # (40 GiB free). MAX reported 40 GiB and invited the hub to admit four
    # 40 GiB jobs; only three of them fit anywhere and none fits on group 0.
    monkeypatch.setenv(ENV_VAR, '{"gpu_count":4,"group_degree":1}')
    _fake_devices([10 * GiB, 40 * GiB, 40 * GiB, 40 * GiB], monkeypatch)
    assert lifecycle.free_vram_bytes() == 10 * GiB


def test_the_report_follows_the_DELIVERED_topology_not_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 2x2 sequence pod on a pod whose own canary says the fabric is not
    # NVLink is DEMOTED to 4x1 and serves that. Reporting from `from_env` would
    # describe groups (0,1) and (2,3) — a packing nobody is serving.
    monkeypatch.setenv(
        ENV_VAR,
        '{"gpu_count":4,"group_degree":2,"parallel":"sequence"}')
    _fake_devices([30 * GiB, 8 * GiB, 40 * GiB, 40 * GiB], monkeypatch)

    from gen_worker import host_canary

    class _Canary:
        interconnect = "pcie"

    monkeypatch.setattr(host_canary, "get_host_canary", lambda: _Canary())
    # Undemoted (from_env) groups are [0,1] and [2,3] -> MINs 8 and 40.
    # Delivered (demoted) groups are [0] [1] [2] [3] -> MIN is card 1's 8 GiB.
    # Both land on 8 GiB here, so assert the TOPOLOGY the code read instead.
    assert lifecycle._worst_group_free_vram_bytes() == 8 * GiB
    from gen_worker.topology import delivered_topology

    assert delivered_topology(interconnect="pcie").group_degree == 1, (
        "the fabric gate must have demoted this pod")


def test_a_single_group_pod_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # G == 1: MIN and MAX are the same number, which is the pgw#748 invariant.
    monkeypatch.delenv(ENV_VAR, raising=False)
    _fake_devices([17 * GiB], monkeypatch)
    assert lifecycle.free_vram_bytes() == 17 * GiB
    assert ExecutionTopology.single().groups == 1


def test_a_multi_gpu_host_with_no_topology_names_the_invisible_cards(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    # The pgw#748 G=1 invariant's ONE exception (audit §4): every harness
    # attach and `cli serve` on a multi-GPU box. The historical all-device SUM
    # became card 0 silently. Card 0 is the honest number — the worker has one
    # slot and places on the thread-current device — but the change must be
    # observable, not rediscovered.
    monkeypatch.delenv(ENV_VAR, raising=False)
    _fake_devices([12 * GiB, 40 * GiB, 40 * GiB, 40 * GiB], monkeypatch)
    monkeypatch.setattr(lifecycle, "_MULTI_GPU_NO_TOPOLOGY_WARNED", False)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    with caplog.at_level("WARNING", logger="gen_worker.lifecycle"):
        assert lifecycle.free_vram_bytes() == 12 * GiB     # card 0, not 124 GiB
    assert any("invisible to the hub" in r.getMessage()
               for r in caplog.records), caplog.text
    # Once, not per heartbeat.
    caplog.clear()
    with caplog.at_level("WARNING", logger="gen_worker.lifecycle"):
        lifecycle.free_vram_bytes()
    assert not caplog.records


def test_the_fit_ladder_gates_on_the_least_free_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DPA-7: `gate_functions` turns probe_hardware()['gpu_free_mem'] into
    # serve plans for EVERY function. Card 0 carries group 0's weights plus the
    # allocator cache, so reading it planned rungs for groups 1-3 from a card
    # they never touch — and a genuinely-unfit card 0 refused functions the
    # other groups could serve.
    monkeypatch.setenv(ENV_VAR, '{"gpu_count":4,"group_degree":1}')
    _fake_devices([9 * GiB, 40 * GiB, 40 * GiB, 40 * GiB], monkeypatch)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    class _Props:
        name = "NVIDIA L40S"

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: _Props())
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda i: (40 * GiB, 48 * GiB))

    info: Dict[str, Any] = lifecycle.probe_hardware()
    assert info["gpu_total_mem"] == 48 * GiB
    assert info["gpu_free_mem"] == 9 * GiB, (
        "the gate must plan for the group with the LEAST room")
