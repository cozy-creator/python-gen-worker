"""Boot host canary — the designed home for its rows."""

from __future__ import annotations


import pytest

from gen_worker import host_canary as hc
from gen_worker.config import Settings
from gen_worker.procsplit import measure
from gen_worker.procsplit.parent import ParentControl
from gen_worker.topology import ExecutionTopology


_TOPO_NVLINK = """\t\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV18\tSYS\t0-51,104-155\t0\t\tN/A
GPU1\tNV18\t X \tSYS\t0-51,104-155\t0\t\tN/A
NIC0\tSYS\tSYS\t X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect
  NV#  = Connection traversing a bonded set of # NVLinks
"""

_TOPO_PCIE = """\t\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity
GPU0\t X \tPHB\t0-31\t0
GPU1\tPHB\t X \t0-31\t0
"""


def test_topo_matrix_parse_reads_the_pair_not_the_row_end() -> None:
    assert hc.parse_nvidia_smi_topo(_TOPO_NVLINK, 0, 1) == "NV18"
    assert hc.parse_nvidia_smi_topo(_TOPO_NVLINK, 1, 0) == "NV18"
    assert hc.parse_nvidia_smi_topo(_TOPO_PCIE, 0, 1) == "PHB"
    assert hc.parse_nvidia_smi_topo(_TOPO_PCIE, 0, 3) == ""
    assert hc.parse_nvidia_smi_topo("", 0, 1) == ""


def test_peer_access_overrules_the_wiring() -> None:
    assert hc.classify_interconnect("NV18", True) == "nvlink"
    assert hc.classify_interconnect("PHB", True) == "pcie-p2p"
    assert hc.classify_interconnect("PIX", True) == "pcie-p2p"
    assert hc.classify_interconnect("", True) == "pcie-p2p"
    assert hc.classify_interconnect("NODE", False) == "host-staged"
    assert hc.classify_interconnect("PIX", False) == "host-staged"
    assert hc.classify_interconnect("NV18", False) == "host-staged"


def test_a_cross_socket_pair_is_host_staged_however_the_flag_reads() -> None:
    assert hc.classify_interconnect("SYS", True) == "host-staged"
    assert hc.classify_interconnect("SYS", False) == "host-staged"


@pytest.fixture()
def _fake_cuda(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")

    def install(count: int) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        monkeypatch.setattr(hc, "_measure_pcie", lambda: (0.0, 0.0, False))
        monkeypatch.setattr(hc, "_measure_memcpy_gbps", lambda: 1.0)
        monkeypatch.setattr(hc, "_measure_cpu_mbps", lambda workers: 1.0)
        monkeypatch.setattr(hc, "_cached", None)

    return install


def _hello_canary(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hc, "_cached", None)
    pc = ParentControl(
        Settings(bootstrap_worker_jwt="", worker_id="w-748",
                 runpod_pod_id="", worker_image_digest="",
                 orchestrator_public_addr="127.0.0.1:1"),
        socket_path="/tmp/gen-worker-canary.sock",
        topology=ExecutionTopology.single(),
    )
    pc._measurement = measure.measure()
    resources = pc._parent_resources()
    assert resources is not None
    return resources.host_canary


def test_measured_fabric_reaches_the_hub(_fake_cuda, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(2)
    monkeypatch.setattr(hc, "_measure_peer", lambda *a, **k: ("nvlink", 348.5, True, "NV18"))
    canary = _hello_canary(monkeypatch)
    assert canary.interconnect == "nvlink"
    assert canary.peer_gbps == pytest.approx(348.5)
    assert canary.peer_access is True
    assert canary.topo_link == "NV18"


def test_a_pod_whose_cards_have_no_peer_access_says_so(
    _fake_cuda, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(2)
    monkeypatch.setattr(hc, "_measure_peer", lambda *a, **k: ("host-staged", 8.1, False, "PHB"))
    canary = _hello_canary(monkeypatch)
    assert canary.interconnect == "host-staged"
    assert canary.peer_gbps == pytest.approx(8.1)
    assert canary.peer_access is False


def test_single_gpu_pods_never_measure_and_never_claim(
    _fake_cuda, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(1)

    def _boom(*a, **k):
        raise AssertionError("peer leg ran on a single-GPU pod")

    monkeypatch.setattr(hc, "_measure_peer", _boom)
    report = hc.measure_host_canary()
    assert report.gpu_count == 1
    assert report.interconnect == ""
    assert report.peer_gbps == 0.0
    assert report.peer_access is False
    canary = _hello_canary(monkeypatch)
    assert canary.interconnect == ""
    assert canary.peer_gbps == 0.0


def test_the_probe_targets_the_production_activation_shape() -> None:
    assert hc.PRODUCTION_ACTIVATION_SHAPE == (1, 40, 37800, 128)
    numel = 1
    for d in hc.PRODUCTION_ACTIVATION_SHAPE:
        numel *= d
    assert numel * 2 == 387_072_000
    assert numel % 2 == 0
