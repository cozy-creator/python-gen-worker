"""gw#665: the conversion toolchain is bounded by SILENCE, not by wall clock.

Real ``run_process`` over real child processes — no mocks. The children stand
in for ``convert_hf_to_gguf.py`` / ``llama-quantize``: both print a line per
tensor, so "has said nothing for N seconds" is the wedge signal and "has been
running for N seconds" is not.
"""

from __future__ import annotations

import sys
import time

import pytest

from gen_worker.subproc import ProcessStalledError, run_process


def _python(script: str) -> list[str]:
    return [sys.executable, "-u", "-c", script]


# A chatty long-runner: 40 lines at 0.05s each. Under the old ``timeout=``
# regime a job like this died purely for outliving the constant.
_CHATTY = """
import sys, time
for i in range(40):
    print("quantizing blk.%d" % i, flush=True)
    time.sleep(0.05)
"""

# Speaks once, then wedges forever.
_WEDGED = """
import sys, time
print("quantizing blk.0", flush=True)
time.sleep(600)
"""


def test_a_talking_process_is_never_killed_however_long_it_runs() -> None:
    """The window is far SHORTER than the runtime; only silence may kill."""
    lines: list[str] = []
    start = time.monotonic()
    rc = run_process(_python(_CHATTY), on_line=lines.append, stall_window_s=0.6)
    elapsed = time.monotonic() - start

    assert rc == 0
    assert len(lines) == 40
    # It genuinely outlived several stall windows while advancing.
    assert elapsed > 0.6 * 2


def test_a_silent_process_is_killed_at_the_stall_window() -> None:
    start = time.monotonic()
    with pytest.raises(ProcessStalledError) as excinfo:
        run_process(_python(_WEDGED), stall_window_s=0.5)
    elapsed = time.monotonic() - start

    assert excinfo.value.window_s == 0.5
    assert excinfo.value.silent_for_s >= 0.5
    # Killed on the window, not on the child's own 600s sleep.
    assert elapsed < 30


def test_no_stall_window_means_no_watchdog_at_all() -> None:
    """Default is unbounded: there is no wall clock hiding in the primitive."""
    lines: list[str] = []
    rc = run_process(_python(_CHATTY), on_line=lines.append)
    assert rc == 0
    assert len(lines) == 40


def test_the_gguf_toolchain_window_is_a_stall_window_not_a_runtime_cap() -> None:
    """Guard against a wall-clock constant creeping back into the call sites."""
    import inspect

    from gen_worker.convert import convert as convert_mod
    from gen_worker.convert import gguf_tools

    assert gguf_tools.GGUF_TOOLCHAIN_STALL_WINDOW_S > 0
    for mod in (gguf_tools, convert_mod):
        code = "\n".join(
            line for line in inspect.getsource(mod).splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "timeout=7200" not in code
        assert "subprocess.run(" not in code
