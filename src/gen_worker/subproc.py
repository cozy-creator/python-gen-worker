"""Run a delegated subprocess (ai-toolkit run.py, external trainers) with
cancellation and log tailing.

The primitive is generic: run a command, stream its merged stdout/stderr
lines to a callback, honor ``ctx.cancelled`` by SIGTERM-ing the process
group (escalating to SIGKILL after a grace period). Endpoints own all
line parsing — e.g. mapping trainer output to ``ctx.progress(...)``.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .api.errors import CanceledError

logger = logging.getLogger(__name__)

_DEFAULT_TERM_GRACE_S = 10.0
_POLL_INTERVAL_S = 0.2


def run_process(
    cmd: Sequence[str],
    *,
    ctx: Any = None,
    on_line: Optional[Callable[[str], None]] = None,
    cwd: "str | os.PathLike[str] | None" = None,
    env: Optional[Mapping[str, str]] = None,
    term_grace_s: float = _DEFAULT_TERM_GRACE_S,
) -> int:
    """Run ``cmd``, streaming merged stdout+stderr lines to ``on_line``.

    - ``ctx``: anything with a ``cancelled`` bool (a RequestContext). When it
      flips true, the process GROUP gets SIGTERM; after ``term_grace_s``
      seconds without exit, SIGKILL. Raises ``CanceledError`` afterwards.
    - ``on_line``: called from a reader thread with each output line
      (trailing newline stripped). Exceptions in the callback are logged
      and swallowed — a bad parse must not kill the trainer.
    - Returns the process exit code on natural exit (callers decide whether
      nonzero is fatal).
    """
    # th#1087: an explicit env mapping still carries the config-snapshot
    # path so the child can read the worker's current runtime config
    # (env=None inherits it from os.environ, exported by ConfigStore).
    child_env = dict(env) if env is not None else None
    if child_env is not None:
        from .runtime_config import SNAPSHOT_PATH_ENV

        child_env.setdefault(
            SNAPSHOT_PATH_ENV, os.environ.get(SNAPSHOT_PATH_ENV, "")
        )
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(cwd)) if cwd is not None else None,
        env=child_env,
        text=True,
        bufsize=1,
        start_new_session=True,  # own process group → group-wide signals
    )

    def _tail() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if on_line is None:
                continue
            try:
                on_line(line)
            except Exception:
                logger.exception("run_process on_line callback failed")
        proc.stdout.close()

    reader = threading.Thread(target=_tail, name="subproc-tail", daemon=True)
    reader.start()

    try:
        while True:
            code = proc.poll()
            if code is not None:
                reader.join(timeout=5.0)
                return int(code)
            if ctx is not None and getattr(ctx, "cancelled", False):
                _terminate_group(proc, term_grace_s=term_grace_s)
                reader.join(timeout=5.0)
                raise CanceledError("subprocess cancelled")
            time.sleep(_POLL_INTERVAL_S)
    finally:
        if proc.poll() is None:  # unexpected exit path (exception in caller)
            _terminate_group(proc, term_grace_s=term_grace_s)


def _terminate_group(proc: "subprocess.Popen[str]", *, term_grace_s: float) -> None:
    """SIGTERM the process group, escalate to SIGKILL after the grace."""
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=max(0.1, term_grace_s))
        return
    except subprocess.TimeoutExpired:
        pass
    logger.warning("subprocess ignored SIGTERM for %.1fs; sending SIGKILL", term_grace_s)
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait(timeout=10.0)
