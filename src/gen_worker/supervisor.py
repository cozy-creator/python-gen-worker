"""A supervisor parent that outlives the worker and names its death."""

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
_SINK_ENV = "GEN_WORKER_POSTMORTEM_FILE"

_FORWARDED = (
    signal.SIGTERM,
    signal.SIGINT,
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGUSR1,
    signal.SIGUSR2,
)
_TERMINATING = (
    signal.SIGTERM,
    signal.SIGINT,
    signal.SIGHUP,
    signal.SIGQUIT,
)
_CONTRACT_SIGNALS = frozenset(_FORWARDED) | {signal.SIGCHLD}

_STOP_TIMEOUT_S = 180.0


def supervision_enabled() -> bool:
    if os.environ.get(_DISABLE_ENV, "").strip() in ("0", "false", "no"):
        return False
    if os.environ.get(_CHILD_ENV):
        return False
    return hasattr(os, "fork")


def _mask(how: int, signals: frozenset) -> None:
    try:
        signal.pthread_sigmask(how, set(signals))
    except (OSError, ValueError, AttributeError):
        pass


def _forward(child_pid: int, stop_timeout_s: float = _STOP_TIMEOUT_S) -> None:
    armed = False

    def escalate(_signum: int, _frame: object) -> None:
        logger.error(
            "worker %d did not exit within %.0fs of the shutdown signal — "
            "SIGKILL (the pod must not outlive its drain)",
            child_pid, stop_timeout_s,
        )
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def handler(signum: int, _frame: object) -> None:
        nonlocal armed
        try:
            os.kill(child_pid, signum)
        except ProcessLookupError:
            pass
        if signum in _TERMINATING and stop_timeout_s > 0 and not armed:
            armed = True
            try:
                signal.setitimer(signal.ITIMER_REAL, stop_timeout_s)
            except (OSError, ValueError):
                pass

    try:
        signal.signal(signal.SIGALRM, escalate)
    except (OSError, ValueError):
        pass
    for sig in _FORWARDED:
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):
            pass


def _wait_for(child_pid: int) -> int:
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
        from . import config
        from .worker_fatal import report_worker_detail

        delivered = report_worker_detail(
            config.current() if config.installed() else config.load_settings(),
            detail,
        )
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


def supervise(
    record_path: Path = postmortem.BOOT_RECORD_PATH,
    *,
    stop_timeout_s: float = _STOP_TIMEOUT_S,
) -> None:
    """Parent: fork, wait, report, exit."""
    _mask(signal.SIG_UNBLOCK, _CONTRACT_SIGNALS)
    if not supervision_enabled():
        return

    report_previous_container_death(record_path)
    postmortem.clear_all_inflight(record_path.parent)
    postmortem.write_boot_record(record_path)

    started = time.time()
    oom_before = postmortem.oom_kill_count()
    _mask(signal.SIG_BLOCK, frozenset(_FORWARDED))
    try:
        child_pid = os.fork()
    except OSError:
        logger.warning("fork failed; running the worker in-process", exc_info=True)
        _mask(signal.SIG_UNBLOCK, _CONTRACT_SIGNALS)
        return
    if child_pid == 0:
        os.environ[_CHILD_ENV] = "1"
        _mask(signal.SIG_UNBLOCK, _CONTRACT_SIGNALS)
        return

    _forward(child_pid, stop_timeout_s)
    _mask(signal.SIG_UNBLOCK, _CONTRACT_SIGNALS)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    status = _wait_for(child_pid)
    verdict = postmortem.describe_exit(status)
    clean = not verdict.get("signaled") and verdict.get("exit_code") == 0
    if clean:
        postmortem.clear_boot_record(record_path)
        postmortem.clear_all_inflight(record_path.parent)
        postmortem.clear_all_fault_dumps(record_path.parent)
    else:
        oom_after = postmortem.oom_kill_count()
        extra: dict = {"child_pid": child_pid}
        if verdict.get("signaled"):
            extra.update(postmortem.attribute_all_signal_deaths(
                signal_name=str(verdict.get("signal_name") or ""),
                marker_dir=record_path.parent,
            ))
        _emit(
            postmortem.format_detail(
                phase="worker_process_exit",
                verdict=verdict,
                limits=postmortem.container_limits(),
                oom_kill_delta=max(0, oom_after - oom_before),
                lifetime_s=time.time() - started,
                extra=extra,
            ),
            dial=bool(verdict.get("signaled")),
        )
        postmortem.clear_boot_record(record_path)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(verdict.get("exit_code") or 0))


__all__ = ["report_previous_container_death", "supervise", "supervision_enabled"]
