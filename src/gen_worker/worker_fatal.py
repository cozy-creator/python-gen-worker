"""gw#640/th#1077: worker -> hub fatal report, dialed BEFORE the process dies.

``entrypoint._log_worker_fatal`` wrote the exception + traceback to the pod's
stdout ONLY. RunPod exposes no container-logs API, so every cloud-only worker
death was unobservable by construction — the th#1085 cold-boot investigation
burned six live runs on a crash whose traceback existed and was unreachable.

This module reuses the ``HardwareUnsuitable`` carrier (gw#619/th#988) with
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
        from .compile_cache import gen_worker_version as _gwv

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
    try:
        report = _build_report(settings, detail)
        return asyncio.run(_report_async(settings, report))
    except Exception:
        logger.warning("worker-fatal report failed entirely", exc_info=True)
        return False


def fatal_identity(settings: Settings) -> str:
    worker_id, release_id = _identity_from_settings(settings)
    return f"worker={worker_id or '?'} release={release_id or '?'}"
