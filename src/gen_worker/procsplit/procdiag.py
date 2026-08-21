"""pgw#1630: the artifact that turns the NEXT kill into a filed bug.

The old watchdog was a cliff: silent past a budget, SIGKILL, exit 137, and
nothing anywhere said WHAT the child had been doing. Two H3 pods died that way
and the diagnosis had to be reconstructed days later from a rental invoice and a
docstring.

So the ladder has a rung between "report" and "signal" whose only product is
evidence. It runs while the child is still ALIVE, which is the only moment any
of this is readable, and it is deliberately cheap and total-failure-tolerant: a
diagnosis that can raise is a diagnosis that turns a stall into a crash.

What it reads, in increasing order of how much it costs:

* ``/proc/<pid>/status`` — the SCHEDULER STATE. ``State: D`` is the single most
  valuable line here: it names an uninterruptible wait, which is a stall the
  child could not have avoided and the parent must not blame it for. This fleet
  has seen exactly that shape on a wedged mount.
* ``/proc/<pid>/wchan`` and ``/proc/<pid>/stack`` — WHERE in the kernel. `stack`
  needs privileges the container usually does not have, so its absence is
  normal and is recorded as such rather than as a failure.
* ``/proc/<pid>/syscall`` — the syscall it is parked in.
* ``py-spy dump`` — the PYTHON stack, when py-spy is installed. Bounded by a
  short subprocess timeout because the tool itself can block on a wedged
  process, and a diagnostic that hangs is worse than no diagnostic.

Every reader is independently guarded: one unreadable source must never cost
the others.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["capture", "read_proc_state"]

#: py-spy attaches to a possibly-wedged process. Bounded so the diagnostic
#: cannot become the hang. Not a liveness judgement — nothing is decided from
#: whether this completes.
_PY_SPY_TIMEOUT_S = 15.0
#: Truncation for any single source, so one enormous stack cannot crowd out the
#: cheap lines that usually carry the answer.
_SOURCE_CAP = 8000


def _read(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return handle.read(_SOURCE_CAP).decode("utf-8", "replace").strip()
    except PermissionError:
        return "<not permitted: the container lacks the capability>"
    except FileNotFoundError:
        return "<absent>"
    except OSError as exc:
        return f"<unreadable: {exc}>"


def read_proc_state(pid: int) -> str:
    """The scheduler state letter (``R``/``S``/``D``/``Z``/``T``), or ``""``.

    Split out because it is the one line worth naming on its own: ``D`` means
    an uninterruptible wait, which is a stall nobody can be blamed for and a
    SIGTERM cannot end.
    """
    for line in _read(f"/proc/{int(pid)}/status").splitlines():
        if line.startswith("State:"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else ""
    return ""


def _py_spy(pid: int) -> str:
    binary = shutil.which("py-spy")
    if binary is None:
        return "<py-spy not installed>"
    try:
        done = subprocess.run(
            [binary, "dump", "--pid", str(int(pid))],
            capture_output=True, timeout=_PY_SPY_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"<py-spy timed out after {_PY_SPY_TIMEOUT_S:.0f}s>"
    except OSError as exc:
        return f"<py-spy failed: {exc}>"
    text = (done.stdout or b"").decode("utf-8", "replace").strip()
    if not text:
        text = (done.stderr or b"").decode("utf-8", "replace").strip()
    return text[:_SOURCE_CAP] or "<py-spy produced nothing>"


def capture(pid: int, out_dir: Optional[Path] = None) -> str:
    """Diagnose a flat child, while it is still alive. NEVER raises.

    The return value is the report; when ``out_dir`` is given the same text is
    also written beside the other post-mortem artifacts, so a diagnosis
    survives a pod whose logs nobody can reach.
    """

    try:
        pid = int(pid)
        state = read_proc_state(pid)
        sections = [
            f"pid={pid} state={state or 'unknown'}"
            + ("  <- UNINTERRUPTIBLE WAIT: the child cannot respond to SIGTERM "
               "and did not choose this" if state == "D" else ""),
            f"[wchan] {_read(f'/proc/{pid}/wchan')}",
            f"[syscall] {_read(f'/proc/{pid}/syscall')}",
            f"[kernel stack]\n{_read(f'/proc/{pid}/stack')}",
            f"[python stack]\n{_py_spy(pid)}",
        ]
        report = "\n".join(sections)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never be the failure
        logger.debug("liveness diagnosis failed", exc_info=True)
        report = f"pid={pid} <diagnosis failed: {type(exc).__name__}: {exc}>"

    if out_dir is not None:
        try:
            target = Path(out_dir)
            target.mkdir(parents=True, exist_ok=True)
            path = target / f"liveness-diagnosis-{pid}-{int(time.time())}.txt"
            path.write_text(report)
            report += f"\n[written] {path}"
        except OSError:
            logger.debug("liveness diagnosis could not be persisted", exc_info=True)
    return report
