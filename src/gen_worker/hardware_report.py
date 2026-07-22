"""gw#619/th#988: worker -> hub hardware-unsuitable report, dialed BEFORE the
boot-time CUDA probe's silent exit.

Previously a probe failure (cuda_probe.py, gw#529) logged
``GEN_WORKER_CUDA_PROBE_FAILED`` and exit(1)'d with no orchestrator contact —
the pod died invisibly to every layer of telemetry (th#986). This module
sends ONE ``HardwareUnsuitable`` ``WorkerMessage`` on the Connect stream, in
place of Hello, so the hub can attribute the death and reschedule/blacklist
the host. Bounded best-effort: a couple of short retries, then give up — the
silent exit remains the fallback (unreachable hub, or an old hub that
predates this field and rejects the connection for not sending Hello first).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import Any, Tuple

import grpc
import grpc.aio
import msgspec

from .config import Settings
from .cuda_probe import CudaProbeResult, classify_probe_failure
from .pb import worker_scheduler_pb2 as pb
from .pb import worker_scheduler_pb2_grpc as pb_grpc
from .transport import normalize_grpc_addr

logger = logging.getLogger(__name__)

# Bounded budget: this is a best-effort diagnostic dial ahead of an exit that
# has already been decided, never a reason to delay it materially — an
# unroutable/blackholed address must not turn a clean pre-hello exit into a
# multi-minute pod-billing hang. Total worst case ~=
# _MAX_ATTEMPTS * _REPORT_RPC_TIMEOUT_S + sum(_RETRY_BACKOFF_S) ~= 7s.
_REPORT_RPC_TIMEOUT_S = 3.0
_MAX_ATTEMPTS = 2
_RETRY_BACKOFF_S = (1.0,)
_NVIDIA_SMI_TIMEOUT_S = 5.0


class HardwareReport(msgspec.Struct, frozen=True):
    reason_class: str
    detail: str
    driver_version: str = ""
    gpu_name: str = ""
    torch_version: str = ""
    torch_cuda_version: str = ""
    gen_worker_version: str = ""
    image_digest: str = ""
    instance_id: str = ""


def _nvidia_smi_driver_and_gpu() -> Tuple[str, str]:
    """driver_version + gpu_name via nvidia-smi/NVML — the point of asking
    here rather than torch is that this must work even when the CUDA
    *runtime* is unusable (that mismatch is exactly the failure we report;
    NVML/nvidia-smi talks to the driver directly, no CUDA context needed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            driver = parts[0] if len(parts) > 0 else ""
            gpu = parts[1] if len(parts) > 1 else ""
            return driver, gpu
    except Exception:
        pass
    return "", ""


def build_hardware_report(probe: CudaProbeResult, settings: Settings) -> HardwareReport:
    """Assemble the typed report from the failed probe + whatever hardware/
    build identity can still be read. Every field degrades to "" rather than
    raising — a probe failure is exactly the moment nothing can be assumed
    to work."""
    driver_version, gpu_name = _nvidia_smi_driver_and_gpu()
    torch_version = ""
    torch_cuda_version = ""
    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "") or "")
        torch_cuda_version = str(getattr(torch.version, "cuda", "") or "")
        if not gpu_name:
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        pass
    gen_worker_version = ""
    try:
        from .compile_cache import gen_worker_version as _gwv

        gen_worker_version = _gwv()
    except Exception:
        pass
    return HardwareReport(
        reason_class=classify_probe_failure(probe.reason),
        detail=probe.reason,
        driver_version=driver_version,
        gpu_name=gpu_name,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        gen_worker_version=gen_worker_version,
        image_digest=settings.worker_image_digest or "",
        instance_id=settings.runpod_pod_id or "",
    )


def _identity_from_settings(settings: Settings) -> Tuple[str, str]:
    """(worker_id, release_id): Settings.worker_id when set, else the JWT
    claims, mirroring Lifecycle's own identity resolution (lifecycle.py)."""
    worker_id = (settings.worker_id or "").strip()
    release_id = ""
    token = (settings.worker_jwt or "").strip()
    if token:
        try:
            from .request_context import _decode_unverified_jwt_claims

            claims = _decode_unverified_jwt_claims(token)
            if not worker_id:
                worker_id = str(claims.get("sub") or "").strip()
            release_id = str(claims.get("release_id") or "").strip()
        except Exception:
            pass
    return worker_id, release_id


def _report_to_wire(report: HardwareReport, worker_id: str, release_id: str) -> pb.WorkerMessage:
    return pb.WorkerMessage(
        hardware_unsuitable=pb.HardwareUnsuitable(
            worker_id=worker_id,
            release_id=release_id,
            reason_class=report.reason_class,
            detail=report.detail,
            driver_version=report.driver_version,
            gpu_name=report.gpu_name,
            torch_version=report.torch_version,
            torch_cuda_version=report.torch_cuda_version,
            gen_worker_version=report.gen_worker_version,
            image_digest=report.image_digest,
            instance_id=report.instance_id,
            reported_at_unix_ms=int(time.time() * 1000),
        )
    )


async def _send_once(
    target: str, use_tls: bool, token: str, msg: pb.WorkerMessage,
) -> bool:
    """One dial attempt: open Connect, write the report, half-close, and wait
    for the hub to end the call. Returns True only on a clean, unrejected
    round trip (delivered); any exception (unreachable, UNAUTHENTICATED, a
    pre-th#988 hub rejecting the missing Hello with FAILED_PRECONDITION) is
    "not delivered" — the caller retries or falls through to the silent
    exit."""
    channel = (
        grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
        if use_tls
        else grpc.aio.insecure_channel(target)
    )
    try:
        stub = pb_grpc.WorkerSchedulerStub(channel)
        metadata = [("authorization", f"Bearer {token}")] if token else None
        stream = stub.Connect(metadata=metadata)

        async def _once() -> Any:
            await stream.write(msg)
            await stream.done_writing()
            return await stream.read()

        await asyncio.wait_for(_once(), _REPORT_RPC_TIMEOUT_S)
        return True
    except Exception as e:
        logger.warning(
            "hardware-unsuitable report attempt to %s failed: %s: %s",
            target, type(e).__name__, e,
        )
        return False
    finally:
        await channel.close()


async def _report_async(settings: Settings, report: HardwareReport) -> bool:
    target, use_tls = normalize_grpc_addr(settings.orchestrator_public_addr)
    if not target:
        return False
    worker_id, release_id = _identity_from_settings(settings)
    token = (settings.worker_jwt or "").strip()
    msg = _report_to_wire(report, worker_id, release_id)
    for attempt in range(_MAX_ATTEMPTS):
        if await _send_once(target, use_tls, token, msg):
            return True
        if attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
    return False


def report_hardware_unsuitable(settings: Settings, probe: CudaProbeResult) -> bool:
    """Bounded best-effort: build the typed report and dial the hub with it
    in place of Hello. Never raises. Returns whether the hub is believed to
    have received it — the entrypoint logs+exits either way; this is purely
    the diagnostic channel, not a gate on shutting down."""
    if not (settings.orchestrator_public_addr or "").strip():
        return False
    report = build_hardware_report(probe, settings)
    try:
        return asyncio.run(_report_async(settings, report))
    except Exception:
        logger.warning("hardware-unsuitable report failed entirely", exc_info=True)
        return False
