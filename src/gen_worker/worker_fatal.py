"""Worker -> hub fatal report, dialed BEFORE the process dies.

Writing the exception + traceback to the pod's stdout is not enough: RunPod
exposes no container-logs API, so a cloud-only worker death is unobservable by
construction — the traceback exists and is unreachable.

This module reuses the ``HardwareUnsuitable`` carrier with
``reason_class="worker_fatal"``: the hub already persists that message as a
durable ``pod_events`` row (class ``hardware_unsuitable``, reason = the class,
full JSON payload in ``provider_message``) and logs it, so a fatal becomes
queryable per pod with NO proto change and NO hub redeploy — it works against
every hub pin already deployed.

Bounded best-effort, exactly like the hardware report: the process is already
dying, so this is a diagnostic dial, never a reason to delay the exit.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Optional

from .config import Settings
from .hardware_report import (
    HardwareReport,
    _identity_from_settings,
    _report_async,
)

logger = logging.getLogger(__name__)

REASON_CLASS = "worker_fatal"


def _broker_active() -> bool:
    """True in a compute child with a live control seam."""
    try:
        from .procsplit import broker

        return broker.active()
    except Exception:
        return False

# The hub stores `detail` in a jsonb payload; a full traceback of a deep
# framework stack can be very long. Keep the head (the raise site chain) and
# the tail (the actual exception) — the middle is the least diagnostic part.
_DETAIL_MAX = 8000
_TAIL_KEEP = 3000


def _clip(text: str) -> str:
    if len(text) <= _DETAIL_MAX:
        return text
    head = _DETAIL_MAX - _TAIL_KEEP - len("\n...[clipped]...\n")
    return text[:head] + "\n...[clipped]...\n" + text[-_TAIL_KEEP:]


def build_fatal_detail(
    phase: str, exc: Optional[BaseException], *, exit_code: int
) -> str:
    """phase + exception identity + traceback, as one human-readable blob."""
    lines = [f"phase={phase or 'unknown'} exit_code={int(exit_code)}"]
    if exc is not None:
        lines.append(f"{type(exc).__name__}: {exc}")
        try:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            tb = traceback.format_exc()
        lines.append(tb.rstrip())
    return _clip("\n".join(lines))


def _build_report(settings: Settings, detail: str) -> HardwareReport:
    gen_worker_version = ""
    try:
        from .toolchain import gen_worker_version as _gwv

        gen_worker_version = _gwv()
    except Exception:
        pass
    torch_version = ""
    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "") or "")
    except Exception:
        pass
    return HardwareReport(
        reason_class=REASON_CLASS,
        detail=detail,
        torch_version=torch_version,
        gen_worker_version=gen_worker_version,
        image_digest=settings.worker_image_digest or "",
        instance_id=settings.runpod_pod_id or "",
    )


def report_worker_fatal(
    settings: Optional[Settings],
    phase: str,
    exc: Optional[BaseException],
    *,
    exit_code: int,
) -> bool:
    """Dial the hub with this process's cause of death. Never raises; returns
    whether the hub is believed to have received it. Safe to call from a
    non-async context only (it owns its own event loop) — the caller is on
    its way out."""
    if settings is None or not (settings.orchestrator_public_addr or "").strip():
        return False
    detail = build_fatal_detail(phase, exc, exit_code=exit_code)
    # In the compute child there is no worker JWT to open a
    # Connect with, and there should not be — a HardwareUnsuitable-carrier
    # report is a fleet-wide verdict key, so it is worth more dialed
    # by the process that runs no tenant code. The parent dials it.
    if _broker_active():
        from .procsplit import broker

        return broker.report_detail(detail)
    try:
        report = _build_report(settings, detail)
        return asyncio.run(_report_async(settings, report))
    except Exception:
        logger.warning("worker-fatal report failed entirely", exc_info=True)
        return False


async def report_worker_error_async(settings: Optional[Settings], detail: str) -> bool:
    """Dial the hub from a STILL-LIVE worker that just entered
    WORKER_PHASE_ERROR — same carrier and durable ``pod_events`` row as the
    process-death fatal. The hub persists only the bare phase flip ("worker
    phase reported error"), so without this the cause of a phase error is
    unreachable (RunPod exposes no logs API, and a malformed lifecycle
    snapshot is dropped by hub shadow validation). Best-effort on the
    caller's running loop; opens its own short Connect like every report."""
    if settings is None or not (settings.orchestrator_public_addr or "").strip():
        return False
    if _broker_active():
        from .procsplit import broker

        return await asyncio.to_thread(broker.report_detail, _clip(detail))
    try:
        return await _report_async(settings, _build_report(settings, _clip(detail)))
    except Exception:
        logger.warning("worker-error report failed entirely", exc_info=True)
        return False


def report_worker_detail(settings: Optional[Settings], detail: str) -> bool:
    """Dial the hub with an already-formatted fatal detail (post-mortem).

    Same carrier and same durable `pod_events` row as `report_worker_fatal`;
    used by the supervisor parent, which has a `waitpid` verdict rather than a
    Python exception to report.
    """
    if settings is None or not (settings.orchestrator_public_addr or "").strip():
        return False
    if _broker_active():
        from .procsplit import broker

        return broker.report_detail(_clip(detail))
    try:
        return asyncio.run(_report_async(settings, _build_report(settings, _clip(detail))))
    except Exception:
        logger.warning("worker-postmortem report failed entirely", exc_info=True)
        return False


def fatal_identity(settings: Settings) -> str:
    worker_id, release_id = _identity_from_settings(settings)
    return f"worker={worker_id or '?'} release={release_id or '?'}"
