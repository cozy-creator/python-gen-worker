"""Run a delegated subprocess (ai-toolkit run.py, external trainers) with
cancellation and log tailing.

The primitive is generic: run a command, stream its merged stdout/stderr
lines to a callback, honor ``ctx.cancelled`` by SIGTERM-ing the process
group (escalating to SIGKILL after a grace period). Endpoints own all
line parsing — e.g. mapping trainer output to ``ctx.progress(...)``.
"""
from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .api.errors import CanceledError
from .runtime_config import SNAPSHOT_PATH_ENV
from .stall import SilenceWindow

logger = logging.getLogger(__name__)

_DEFAULT_TERM_GRACE_S = 10.0
_POLL_INTERVAL_S = 0.2


class LineTail:
    """Reader thread over a child's merged stdout+stderr.

    Streams every line to ``on_line`` and stamps a :class:`SilenceWindow`, so
    "the child has said nothing for N seconds" is answerable at any moment.
    Draining is not optional: a child whose pipe fills BLOCKS, so whoever
    captures output must keep reading for the process's whole life.

    ``run_process`` uses it to bound a run-to-completion tool (gw#665);
    ``runtimes.server`` uses it to bound an engine BOOT while keeping the
    child alive afterwards (gw#666).
    """

    __slots__ = ("_proc", "_on_line", "_window", "_thread")

    def __init__(
        self,
        proc: "subprocess.Popen[str]",
        *,
        window_s: float,
        on_line: Optional[Callable[[str], None]] = None,
        name: str = "subproc-tail",
    ) -> None:
        self._proc = proc
        self._on_line = on_line
        self._window = SilenceWindow(window_s)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> "LineTail":
        self._thread.start()
        return self

    def _run(self) -> None:
        stdout = self._proc.stdout
        assert stdout is not None
        for raw in stdout:
            self._window.touch()
            line = raw.rstrip("\n")
            if self._on_line is None:
                continue
            try:
                self._on_line(line)
            except Exception:
                logger.exception("subprocess output callback failed")
        stdout.close()

    def silent_for(self) -> float:
        return self._window.silent_for()

    def stalled(self) -> bool:
        return self._window.stalled()

    @property
    def window_s(self) -> float:
        return self._window.window_s

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)


class ProcessStalledError(RuntimeError):
    """The child produced no output for its stall window — presumed wedged.

    Raised only by ``run_process(stall_window_s=...)``. It is the
    progress-based replacement for a wall-clock ``timeout=``: a long job that
    keeps talking is never killed, a silent one is killed quickly.
    """

    def __init__(self, cmd: Sequence[str], silent_for_s: float, window_s: float) -> None:
        super().__init__(
            f"subprocess produced no output for {silent_for_s:.0f}s "
            f"(stall window {window_s:.0f}s): {' '.join(cmd)}"
        )
        self.silent_for_s = silent_for_s
        self.window_s = window_s


def run_process(
    cmd: Sequence[str],
    *,
    ctx: Any = None,
    on_line: Optional[Callable[[str], None]] = None,
    cwd: "str | os.PathLike[str] | None" = None,
    env: Optional[Mapping[str, str]] = None,
    term_grace_s: float = _DEFAULT_TERM_GRACE_S,
    stall_window_s: Optional[float] = None,
) -> int:
    """Run ``cmd``, streaming merged stdout+stderr lines to ``on_line``.

    - ``ctx``: anything with a ``cancelled`` bool (a RequestContext). When it
      flips true, the process GROUP gets SIGTERM; after ``term_grace_s``
      seconds without exit, SIGKILL. Raises ``CanceledError`` afterwards.
    - ``on_line``: called from a reader thread with each output line
      (trailing newline stripped). Exceptions in the callback are logged
      and swallowed — a bad parse must not kill the trainer.
    - ``stall_window_s``: optional PROGRESS watchdog. Every output line is an
      advance; the group is terminated and ``ProcessStalledError`` raised only
      once the child has been silent this long. ``None`` (default) = no
      watchdog. There is deliberately no total-runtime bound: a wall clock
      cannot tell a healthy 3-hour quantize from a wedge, so it is either
      useless or it kills real work (gw#655's residency-design principle).
    - Returns the process exit code on natural exit (callers decide whether
      nonzero is fatal).
    """

    invocation_snapshot_path = _write_invocation_snapshot(ctx)
    child_env = dict(env) if env is not None else None
    if invocation_snapshot_path:
        child_env = dict(os.environ) if child_env is None else child_env
        child_env[SNAPSHOT_PATH_ENV] = invocation_snapshot_path
    elif child_env is not None:
        child_env.setdefault(
            SNAPSHOT_PATH_ENV, os.environ.get(SNAPSHOT_PATH_ENV, "")
        )
    try:
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

        window = (
            float(stall_window_s)
            if stall_window_s is not None and stall_window_s > 0
            else math.inf
        )
        reader = LineTail(proc, window_s=window, on_line=on_line).start()

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
                if reader.stalled():
                    silent_for = reader.silent_for()
                    _terminate_group(proc, term_grace_s=term_grace_s)
                    reader.join(timeout=5.0)
                    raise ProcessStalledError(cmd, silent_for, window)
                time.sleep(_POLL_INTERVAL_S)
        finally:
            if proc.poll() is None:  # unexpected exit path (exception in caller)
                _terminate_group(proc, term_grace_s=term_grace_s)
    finally:
        if invocation_snapshot_path:
            try:
                os.unlink(invocation_snapshot_path)
            except FileNotFoundError:
                pass


def _write_invocation_snapshot(ctx: Any) -> str:
    raw = getattr(ctx, "_config_snapshot", None) if ctx is not None else None
    if not isinstance(raw, bytes):
        return ""
    fd, path = tempfile.mkstemp(prefix=".runtime_config-invoke-", suffix=".msgpack")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


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
