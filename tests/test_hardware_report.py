"""gw#619/th#988: the worker's boot-time CUDA probe failure must dial the hub
with a typed HardwareUnsuitable report BEFORE the pre-existing silent exit
(cuda_probe.py, gw#529) — closing the th#986 blindness where ~20 pod deaths
carried zero orchestrator-visible evidence. Real gRPC sockets throughout
(tests/harness/hardware_report_hub.py); no mocking of the transport layer.
"""

from __future__ import annotations

import time
from concurrent import futures

import grpc
import pytest

from gen_worker import hardware_report
from gen_worker.config import Settings
from gen_worker.cuda_probe import CudaProbeResult, classify_probe_failure
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc

from harness.hardware_report_hub import closed_port_addr, old_hub, recording_hub
from harness.hub_double import FakeScheduler

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# classify_probe_failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("", "unknown"),
        ("torch unavailable: no module named torch", "torch_unavailable"),
        ("torch.cuda.is_available() is False", "cuda_unavailable"),
        # th#591/th#979's exact real-world signature, reproduced verbatim.
        ("RuntimeError: CUDA initialization: driver too old (found version 12080)", "driver_too_old"),
        ("RuntimeError: CUDA-capable device(s) is/are busy or unavailable", "cuda_error"),
    ],
)
def test_classify_probe_failure_vocabulary(reason: str, expected: str) -> None:
    assert classify_probe_failure(reason) == expected


# ---------------------------------------------------------------------------
# build_hardware_report
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    base = dict(
        orchestrator_public_addr="127.0.0.1:1",
        worker_id="worker-1",
        bootstrap_worker_jwt="",
        worker_image_digest="sha256:deadbeef",
        runpod_pod_id="pod-1",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_build_hardware_report_degrades_safely_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise FileNotFoundError("no nvidia-smi on this box")

    monkeypatch.setattr(hardware_report, "_nvidia_smi_driver_and_gpu", lambda: ("", ""))
    probe = CudaProbeResult(ok=False, reason="torch.cuda.is_available() is False")
    report = hardware_report.build_hardware_report(probe, _settings())
    assert report.reason_class == "cuda_unavailable"
    assert report.detail == probe.reason
    assert report.image_digest == "sha256:deadbeef"
    assert report.instance_id == "pod-1"
    # torch IS importable in this env (it's a gen-worker dependency); driver
    # detection degrading to "" must not take torch_version down with it.
    assert report.torch_version


def test_build_hardware_report_uses_nvidia_smi_when_torch_cuda_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware_report, "_nvidia_smi_driver_and_gpu", lambda: ("570.211.01", "NVIDIA GeForce RTX 4070")
    )
    probe = CudaProbeResult(
        ok=False, reason="RuntimeError: CUDA initialization: driver too old (found version 12080)"
    )
    report = hardware_report.build_hardware_report(probe, _settings())
    assert report.reason_class == "driver_too_old"
    assert report.driver_version == "570.211.01"
    assert report.gpu_name == "NVIDIA GeForce RTX 4070"


# ---------------------------------------------------------------------------
# report_hardware_unsuitable — real gRPC socket round trips
# ---------------------------------------------------------------------------


def test_report_hardware_unsuitable_delivers_to_a_new_hub() -> None:
    with recording_hub() as (servicer, addr):
        settings = _settings(orchestrator_public_addr=addr, bootstrap_worker_jwt="")
        probe = CudaProbeResult(ok=False, reason="torch.cuda.is_available() is False")
        delivered = hardware_report.report_hardware_unsuitable(settings, probe)
        assert delivered is True

        msg = servicer.wait_for_message(timeout=5.0)
        assert msg.WhichOneof("msg") == "hardware_unsuitable"
        hw = msg.hardware_unsuitable
        assert hw.worker_id == "worker-1"
        assert hw.reason_class == "cuda_unavailable"
        assert hw.detail == probe.reason
        assert hw.image_digest == "sha256:deadbeef"
        assert hw.instance_id == "pod-1"
        assert hw.reported_at_unix_ms > 0


def test_general_hub_double_preserves_a_diagnostic_first_stream() -> None:
    """The ordinary hub double mirrors both valid Connect first-message shapes.

    Worker fatal/error reporting deliberately opens a separate Connect whose
    first and only message is ``HardwareUnsuitable``.  Rejecting that valid
    shape with a bare "first message must be Hello" assertion erased the
    exception detail that caused the worker's primary stream to end, which is
    exactly what the merge-queue failure needed to reveal.
    """
    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        settings = _settings(orchestrator_public_addr=f"127.0.0.1:{port}")
        probe = CudaProbeResult(ok=False, reason="the root failure survives")
        assert hardware_report.report_hardware_unsuitable(settings, probe) is True
        assert len(scheduler.diagnostic_reports) == 1
        report = scheduler.diagnostic_reports[0]
        assert report.reason_class == "cuda_error"
        assert report.detail == probe.reason
    finally:
        server.stop(grace=0)


def test_report_hardware_unsuitable_delivers_with_worker_jwt_identity() -> None:
    """worker_id/release_id fall back to the JWT claims when Settings.worker_id
    is unset — exactly Lifecycle's own identity resolution (lifecycle.py).

    pgw#848/`e8b3109`: the token comes from `worker_credential.current()` and
    ONLY from there. That commit deleted the `or settings.worker_jwt` fallback
    on purpose — *"the fallbacks are GONE, not deprioritised: hardware_report is
    bare"* — so seeding this through `Settings` no longer reaches the code under
    test, and the assertion below was passing for the wrong reason before and
    failing for the right one after. The PROPERTY is unchanged; only the source
    of the credential moved, so the test moves with it.
    """
    import base64
    import json

    from gen_worker import worker_credential

    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "jwt-worker", "release_id": "release-77"}).encode()
    ).rstrip(b"=")
    fake_jwt = b"header." + payload + b".sig"

    worker_credential.reset()
    try:
        worker_credential.install(fake_jwt.decode())
        with recording_hub() as (servicer, addr):
            settings = _settings(orchestrator_public_addr=addr, worker_id="")
            probe = CudaProbeResult(ok=False, reason="")
            assert hardware_report.report_hardware_unsuitable(settings, probe) is True
            hw = servicer.wait_for_message(timeout=5.0).hardware_unsuitable
            assert hw.worker_id == "jwt-worker"
            assert hw.release_id == "release-77"
    finally:
        worker_credential.reset()


def test_report_hardware_unsuitable_old_hub_rejects_gracefully() -> None:
    """A pre-gw#619 hub rejects the missing-Hello stream with
    FAILED_PRECONDITION — the worker must treat that as not-delivered and
    return promptly, never raise, never hang."""
    with old_hub() as addr:
        settings = _settings(orchestrator_public_addr=addr)
        probe = CudaProbeResult(ok=False, reason="cuda oops")
        start = time.monotonic()
        delivered = hardware_report.report_hardware_unsuitable(settings, probe)
        elapsed = time.monotonic() - start
        assert delivered is False
        assert elapsed < 10.0


def test_report_hardware_unsuitable_unreachable_hub_is_bounded() -> None:
    """Connection-refused (nothing listening) must fall through to
    not-delivered within the bounded retry budget — never hang, matching the
    entrypoint's silent-exit fallback contract."""
    settings = _settings(orchestrator_public_addr=closed_port_addr())
    probe = CudaProbeResult(ok=False, reason="cuda oops")
    start = time.monotonic()
    delivered = hardware_report.report_hardware_unsuitable(settings, probe)
    elapsed = time.monotonic() - start
    assert delivered is False
    assert elapsed < 10.0


def test_report_hardware_unsuitable_no_orchestrator_addr_is_a_noop() -> None:
    settings = _settings(orchestrator_public_addr="")
    probe = CudaProbeResult(ok=False, reason="cuda oops")
    assert hardware_report.report_hardware_unsuitable(settings, probe) is False
