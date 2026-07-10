"""gw#425: run_process — delegated-trainer subprocess primitive.

Real subprocesses: line streaming to the callback, natural exit codes,
ctx-cancellation → SIGTERM to the process group, SIGKILL escalation for
TERM-ignoring children.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import List

import pytest

from gen_worker.api.errors import CanceledError
from gen_worker.request_context import RequestContext
from gen_worker.subproc import run_process

PY = sys.executable


def test_lines_stream_to_callback_and_exit_code() -> None:
    lines: List[str] = []
    code = run_process(
        [PY, "-u", "-c", "print('step=1 loss=0.5'); print('step=2 loss=0.4')"],
        on_line=lines.append,
    )
    assert code == 0
    assert lines == ["step=1 loss=0.5", "step=2 loss=0.4"]


def test_stderr_merged_and_nonzero_exit() -> None:
    lines: List[str] = []
    code = run_process(
        [PY, "-u", "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"],
        on_line=lines.append,
    )
    assert code == 3
    assert lines == ["boom"]


def test_bad_on_line_callback_does_not_kill_the_run() -> None:
    seen: List[str] = []

    def _bad(line: str) -> None:
        seen.append(line)
        raise ValueError("parse error")

    assert run_process([PY, "-u", "-c", "print('a'); print('b')"], on_line=_bad) == 0
    assert seen == ["a", "b"]


def test_cancellation_sigterms_process_group() -> None:
    ctx = RequestContext(request_id="r1")
    threading.Timer(0.5, ctx._cancel).start()
    t0 = time.monotonic()
    with pytest.raises(CanceledError):
        run_process(
            [PY, "-u", "-c", "import time; print('up', flush=True); time.sleep(60)"],
            ctx=ctx,
        )
    assert time.monotonic() - t0 < 15.0


def test_sigkill_escalation_for_term_ignoring_child() -> None:
    ctx = RequestContext(request_id="r1")
    lines: List[str] = []
    started = threading.Event()

    def _on_line(line: str) -> None:
        lines.append(line)
        started.set()

    def _cancel_when_up() -> None:
        started.wait(10.0)
        ctx._cancel()

    threading.Thread(target=_cancel_when_up, daemon=True).start()
    t0 = time.monotonic()
    with pytest.raises(CanceledError):
        run_process(
            [PY, "-u", "-c",
             "import signal, time\n"
             "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
             "print('armed', flush=True)\n"
             "time.sleep(60)\n"],
            ctx=ctx,
            on_line=_on_line,
            term_grace_s=0.5,
        )
    assert "armed" in lines
    assert time.monotonic() - t0 < 20.0
