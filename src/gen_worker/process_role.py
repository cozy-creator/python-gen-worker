from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

ROLE_SERVING = "serving"
ROLE_UNKNOWN = "unknown"

_role = ROLE_UNKNOWN


def declare(role: str) -> None:
    global _role
    _role = str(role or ROLE_UNKNOWN)


def role() -> str:
    return _role


def serving_pid() -> int:
    """This process's pid IF it is the serving one, else 0 (== not known here)."""
    return os.getpid() if _role == ROLE_SERVING else 0


def facts() -> str:
    return f"serving_pid={serving_pid()} role={_role}"


def emit_boot_role() -> None:
    """One event stating the serving pid and this process's role."""
    try:
        from . import activity as activity_mod

        activity_mod.emit_event(
            activity_mod.KIND_PROCESS_ROLE,
            f"{facts()} pid={os.getpid()} ppid={os.getppid()} "
            f"boot_unix={time.time():.0f}",
            phase=_role,
        )
    except Exception:  # pragma: no cover — telemetry never fails a boot
        logger.debug("process role event failed", exc_info=True)
