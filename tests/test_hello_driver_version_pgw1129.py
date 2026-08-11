"""pgw#1129 / th#1798: Hello carries the HOST driver version.

The hub only ever saw a driver version on the FAILURE carrier
(``HardwareUnsuitable``), so a fleet that is fine and a fleet whose placement
filter silently stopped working produced identical hub data. RunPod's driver is
per-HOST — a secure A40 create pinned to ``allowedCudaVersions=["12.8"]`` drew
570.211.01 twice on 2026-08-11, where torch 2.13.0+cu130 imports fine and the
first allocation dies "driver too old (found version 12080)".
"""

from __future__ import annotations

import gen_worker.lifecycle as lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb


def test_probe_hardware_reports_the_host_driver(monkeypatch):
    monkeypatch.setattr(
        "gen_worker.hardware_report._nvidia_smi_driver_and_gpu",
        lambda: ("570.211.01", "NVIDIA A40"),
    )
    info = lifecycle.probe_hardware()
    assert info["driver_version"] == "570.211.01"


def test_probe_hardware_degrades_to_empty_when_nvidia_smi_is_unreadable(monkeypatch):
    def boom():
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(
        "gen_worker.hardware_report._nvidia_smi_driver_and_gpu", boom
    )
    info = lifecycle.probe_hardware()
    # An unread driver is "" — the hub records that as `unknown`, never as fine.
    assert info["driver_version"] == ""


def test_worker_resources_carries_the_driver_on_the_wire():
    res = pb.WorkerResources(driver_version="580.159.04")
    assert res.driver_version == "580.159.04"
    # Additive: an old worker leaves it at the protobuf default.
    assert pb.WorkerResources().driver_version == ""
