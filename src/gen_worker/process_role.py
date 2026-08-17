"""pgw#1309: WHICH process this interpreter is, and its pid, as a wire fact.

The pgw#1232 acceptance legs have to answer one question off-pod: did the
inductor compile run in a CHILD, or on the process that serves tenants? Every
half of the answer was pod-local — the compile child's pid is one
``aot_compile_pool`` log line and the serving pid lives in
``entrypoint._startup_payload`` — and a serve pod exposes no logs (pgw#760).

The role is DECLARED by the process that knows it (the serving loop, at sink
bind), never inferred. An interpreter that never declared answers ``unknown``
and a serving pid of ``0``: a compile child volunteering its own pid as the
serving one is exactly the confusion the legs exist to rule out.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

#: The process that holds the orchestrator session, serves tenant requests and
#: supervises compile children (`mint_supervisor`, two tiers since pgw#1215).
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
    """The pid half of every pgw#1309 event, in one spelling."""
    return f"serving_pid={serving_pid()} role={_role}"


def emit_boot_role() -> None:
    """One event stating the serving pid and this process's role.

    Emitted at every sink bind rather than once per interpreter: the fact is
    boot-scoped but the transport is not, and a reconnect must not cost the
    fleet the one row legs B/C assert "no compile ran under this pid" against.
    Never raises — telemetry never breaks the process it describes.
    """
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
