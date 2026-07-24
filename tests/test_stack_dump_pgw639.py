"""pgw#639: the SIGUSR2 stack-dump channel must actually be armed."""
from __future__ import annotations

import os
import subprocess
import sys

PROG = r'''
import faulthandler, os, signal, sys, threading, time
sys.path.insert(0, %r)
from gen_worker.entrypoint import _install_stack_dump_handler

def spin():
    time.sleep(30)

t = threading.Thread(target=spin, name="wedged-model-thread", daemon=True)
t.start()
_install_stack_dump_handler()
print("ARMED", flush=True)
os.kill(os.getpid(), signal.SIGUSR2)
time.sleep(0.5)
'''


def test_sigusr2_dumps_all_thread_stacks() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    out = subprocess.run(
        [sys.executable, "-c", PROG % src],
        capture_output=True, text=True, timeout=60,
    )
    assert "ARMED" in out.stdout, out.stderr
    # faulthandler writes to stderr, all threads, without allocating.
    assert "Current thread" in out.stderr or "Thread 0x" in out.stderr, out.stderr
    assert "wedged-model-thread" in out.stderr or "spin" in out.stderr, out.stderr
