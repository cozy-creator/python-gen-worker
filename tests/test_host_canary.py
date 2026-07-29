"""Boot host canary (gw#550) — the designed home for its rows.

Covers the pgw#748 phase-0 2-GPU leg: a delivered pod's GPU fabric is a
MEASUREMENT, not an inference from the SKU. The hub can tell SXM from PCIe
by SKU identity, but only the pod can say whether ITS two cards have peer
access — and that is what decides whether a sequence-parallel release meets
its latency SLO. The leg must reach the hub through Hello.resources, and it
must stay inert on the 1-GPU pods that are the entire fleet today.

Fakes only at the torch/CUDA boundary (th#1105); the parse and the
classification are pure and tested against verbatim ``nvidia-smi`` output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gen_worker import host_canary as hc
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle


# Verbatim `nvidia-smi topo -m` from a 2x H100 80GB HBM3 (SXM) host.
_TOPO_NVLINK = """\t\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV18\tSYS\t0-51,104-155\t0\t\tN/A
GPU1\tNV18\t X \tSYS\t0-51,104-155\t0\t\tN/A
NIC0\tSYS\tSYS\t X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect
  NV#  = Connection traversing a bonded set of # NVLinks
"""

# Verbatim shape from a 2x consumer-GPU host wired through the root complex.
_TOPO_PCIE = """\t\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity
GPU0\t X \tPHB\t0-31\t0
GPU1\tPHB\t X \t0-31\t0
"""


def test_topo_matrix_parse_reads_the_pair_not_the_row_end() -> None:
    # The GPU columns are followed by affinity columns; indexing off the
    # header is what keeps "SYS" (a NIC row) from being read as the link.
    assert hc.parse_nvidia_smi_topo(_TOPO_NVLINK, 0, 1) == "NV18"
    assert hc.parse_nvidia_smi_topo(_TOPO_NVLINK, 1, 0) == "NV18"
    assert hc.parse_nvidia_smi_topo(_TOPO_PCIE, 0, 1) == "PHB"
    # A pair the matrix does not contain is "unknown", never a crash.
    assert hc.parse_nvidia_smi_topo(_TOPO_PCIE, 0, 3) == ""
    assert hc.parse_nvidia_smi_topo("", 0, 1) == ""


def test_peer_access_overrules_the_wiring() -> None:
    assert hc.classify_interconnect("NV18", True) == "nvlink"
    assert hc.classify_interconnect("PHB", True) == "pcie-p2p"
    assert hc.classify_interconnect("PIX", True) == "pcie-p2p"
    assert hc.classify_interconnect("", True) == "pcie-p2p"
    # The GeForce trap: two cards can look adjacent and still have no P2P at
    # all, in which case every byte is staged through host RAM. Wiring never
    # overrules the capability query.
    assert hc.classify_interconnect("PIX", False) == "host-staged"
    assert hc.classify_interconnect("NV18", False) == "host-staged"


@pytest.fixture()
def _fake_cuda(monkeypatch: pytest.MonkeyPatch):
    """Pin the visible device count at the torch boundary."""
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
    lc = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="w-748",
                        runpod_pod_id="", worker_image_digest=""),
        Executor([], lambda *a, **k: None),
    )
    return lc.build_hello().resources.host_canary


def test_measured_fabric_reaches_the_hub(_fake_cuda, monkeypatch: pytest.MonkeyPatch) -> None:
    # pgw#748: the whole point of the leg is that the number leaves the pod.
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
    # The outcome that matters for placement: the SKU still says "2 GPUs",
    # and the pod still reports the fabric it actually got.
    _fake_cuda(2)
    monkeypatch.setattr(hc, "_measure_peer", lambda *a, **k: ("host-staged", 8.1, False, "PHB"))
    canary = _hello_canary(monkeypatch)
    assert canary.interconnect == "host-staged"
    assert canary.peer_gbps == pytest.approx(8.1)
    assert canary.peer_access is False


def test_single_gpu_pods_never_measure_and_never_claim(
    _fake_cuda, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The entire fleet today. The leg must cost nothing and assert nothing.
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
    # The collective leg is only ground truth if it moves the bytes the
    # model moves: [batch, heads, tokens, head_dim] bf16 = 387 MB per call.
    assert hc.PRODUCTION_ACTIVATION_SHAPE == (1, 40, 37800, 128)
    numel = 1
    for d in hc.PRODUCTION_ACTIVATION_SHAPE:
        numel *= d
    assert numel * 2 == 387_072_000
    assert numel % 2 == 0  # splittable across a degree-2 mesh
