"""gw#640: a supervisor parent that outlives the worker and names its death.

The worker process is the container's PID 1. When it is killed by a signal —
cgroup OOM SIGKILL, SIGSEGV in a C extension, an external kill — there is no
Python left to report anything, which is why six live runs of the th#1085
cold-boot gate produced restarts and zero diagnostics.

So we fork FIRST, before the heavy imports: the parent stays a few MiB of
interpreter, the child is the worker. The parent forwards signals, waits, and
on an abnormal exit reports ``WTERMSIG``/``WCOREDUMP`` plus the container's
memory facts through the existing ``worker_fatal`` carrier — a durable
``pod_events`` row on any already-deployed hub. It then exits with the child's
status, so container-restart semantics are unchanged.

Set ``GEN_WORKER_SUPERVISOR=0`` to run the worker in-process (local dev, or an
escape hatch if a fork-hostile environment ever turns up).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from . import postmortem

logger = logging.getLogger("WorkerSupervisor")

_CHILD_ENV = "GEN_WORKER_SUPERVISED"
_DISABLE_ENV = "GEN_WORKER_SUPERVISOR"
# Where the parent mirrors its report when asked (tests, and a pod-side
# forensic file that survives the report failing to reach the hub).
_SINK_ENV = "GEN_WORKER_POSTMORTEM_FILE"

_FORWARDED = (
    signal.SIGTERM,
    signal.SIGINT,
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGUSR1,
    signal.SIGUSR2,
)


def supervision_enabled() -> bool:
    if os.environ.get(_DISABLE_ENV, "").strip() in ("0", "false", "no"):
        return False
    if os.environ.get(_CHILD_ENV):
        return False
    return hasattr(os, "fork")


def _forward(child_pid: int) -> None:
    def handler(signum: int, _frame: object) -> None:
        try:
            os.kill(child_pid, signum)
        except ProcessLookupError:
            pass

    for sig in _FORWARDED:
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):
            pass


def _wait_for(child_pid: int) -> int:
    """Reap until OUR child exits; PID 1 also reaps reparented orphans."""
    while True:
        try:
            pid, status = os.waitpid(-1, 0)
        except InterruptedError:
            continue
        except ChildProcessError:
            return 0
        if pid == child_pid:
            return status


def _emit(detail: str, *, dial: bool = True) -> None:
    """Record a post-mortem. ``dial`` reaches the hub (bounded ~7s).

    Only the deaths Python could not report itself are worth the dial: a
    signal, or a previous container that vanished whole. An ordinary non-zero
    exit already had 0.56.1's in-process `worker_fatal`, so re-dialing it would
    only duplicate the row and add its budget to every failing boot's exit.
    """
    logger.error("worker.postmortem\n%s", detail)
    sink = os.environ.get(_SINK_ENV, "").strip()
    if sink:
        try:
            Path(sink).write_text(detail)
        except OSError:
            logger.warning("post-mortem sink write failed", exc_info=True)
    if not dial:
        return
    try:
        from .config import get_settings
        from .worker_fatal import report_worker_detail

        delivered = report_worker_detail(get_settings(), detail)
        logger.info("worker.postmortem wire report delivered=%s", delivered)
    except Exception:
        logger.warning("post-mortem wire report failed", exc_info=True)


def report_previous_container_death(
    record_path: Path = postmortem.BOOT_RECORD_PATH,
) -> Optional[str]:
    """Report a previous process whose supervisor did not survive either."""
    detail = postmortem.previous_boot_detail(record_path)
    if detail:
        _emit(detail)
    return detail


def supervise(record_path: Path = postmortem.BOOT_RECORD_PATH) -> None:
    """Parent: fork, wait, report, exit. Child: return and be the worker.

    Called before the worker's heavy imports so the parent stays tiny — the
    kernel's OOM killer picks the fat child, leaving the reporter alive.
    """
    if not supervision_enabled():
        return

    report_previous_container_death(record_path)
    postmortem.write_boot_record(record_path)

    started = time.time()
    oom_before = postmortem.oom_kill_count()
    try:
        child_pid = os.fork()
    except OSError:
        logger.warning("fork failed; running the worker in-process", exc_info=True)
        return
    if child_pid == 0:
        os.environ[_CHILD_ENV] = "1"
        return  # the child continues into the worker

    # The parent forked before the worker configured logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _forward(child_pid)
    status = _wait_for(child_pid)
    verdict = postmortem.describe_exit(status)
    clean = not verdict.get("signaled") and verdict.get("exit_code") == 0
    if clean:
        postmortem.clear_boot_record(record_path)
    else:
        oom_after = postmortem.oom_kill_count()
        _emit(
            postmortem.format_detail(
                phase="worker_process_exit",
                verdict=verdict,
                limits=postmortem.container_limits(),
                oom_kill_delta=max(0, oom_after - oom_before),
                lifetime_s=time.time() - started,
                extra={"child_pid": child_pid},
            ),
            dial=bool(verdict.get("signaled")),
        )
        postmortem.clear_boot_record(record_path)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(verdict.get("exit_code") or 0))


__all__ = ["report_previous_container_death", "supervise", "supervision_enabled"]
