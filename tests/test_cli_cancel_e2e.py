"""End-to-end cancellation contract with REAL SIGINT (#346 / #352 / #353).

Proves, with actual ``Ctrl-C`` (SIGINT) delivered to real subprocesses, that:

  1. SIGINT to a standalone ``gen-worker run`` cancels the in-flight handler
     (cooperative ctx.cancel()) and exits 130.
  2. SIGINT to a ``gen-worker invoke`` client dispatching against a warm
     ``serve`` cancels THAT request (server trips ctx.cancel()) — and the serve
     process stays alive and answers a subsequent request.
  3. SIGINT to the ``serve`` process itself tears it down cleanly (exit 0,
     socket removed).

These are timing-based integration tests over subprocesses; each waits on real
readiness signals rather than fixed sleeps where possible.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


import gen_worker.cli as cli

_ENDPOINT_SRC = '''\
import time
import msgspec
from gen_worker import RequestContext, endpoint


class In(msgspec.Struct):
    text: str = ""


class Out(msgspec.Struct):
    response: str


@endpoint
class EP:
    def setup(self) -> None:
        pass

    def slow(self, ctx: RequestContext, data: In) -> Out:
        # Block cooperatively until canceled — the canonical idiom.
        for _ in range(100000):
            ctx.raise_if_cancelled()
            time.sleep(0.02)
        return Out(response="done")

    def ping(self, ctx: RequestContext, data: In) -> Out:
        return Out(response="pong")
'''


def _make_endpoint(tmp_path: Path) -> Path:
    pkg = tmp_path / "proj"
    (pkg / "cancel_ep").mkdir(parents=True)
    (pkg / "cancel_ep" / "__init__.py").write_text("")
    (pkg / "cancel_ep" / "main.py").write_text(_ENDPOINT_SRC)
    (pkg / "pyproject.toml").write_text('[tool.gen_worker]\nmain = "cancel_ep.main"\n')
    return pkg


def _env(pkg: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(pkg)}


def test_sigint_cancels_standalone_run(tmp_path) -> None:
    """Ctrl-C on a one-shot `run` cancels the in-flight handler -> exit 130."""
    pkg = _make_endpoint(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.cli", "run",
         "--module", "cancel_ep.main", "--method", "slow", "--payload", "{}"],
        cwd=str(pkg), env=_env(pkg),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        time.sleep(2.0)  # let it enter the handler loop
        assert proc.poll() is None, "run should still be executing the slow handler"
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 130  # EXIT_SIGINT (cooperative cancel)


def _wait_socket(sock: Path, proc: subprocess.Popen, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"serve exited early rc={proc.returncode}: {proc.stderr.read()}")
        if sock.exists():
            return
        time.sleep(0.05)
    raise AssertionError("serve never created the socket")


def test_sigint_cancels_request_then_server_survives_then_stops(tmp_path, capsys) -> None:
    """Ctrl-C on an `invoke` client cancels its request; serve stays warm and
    answers another; Ctrl-C on serve stops it cleanly."""
    pkg = _make_endpoint(tmp_path)
    sock = tmp_path / "rt.sock"
    serve = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.cli", "serve",
         "--module", "cancel_ep.main", "--socket", str(sock), "--no-stdin"],
        cwd=str(pkg), env=_env(pkg),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        _wait_socket(sock, serve)

        # An invoke that blocks in the slow handler.
        inv = subprocess.Popen(
            [sys.executable, "-m", "gen_worker.cli", "invoke", "slow", "{}",
             "--socket", str(sock)],
            cwd=str(pkg), env=_env(pkg),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(2.0)  # ensure the request is in-flight + the canceler installed
        assert inv.poll() is None, "invoke should be waiting on the slow request"

        inv.send_signal(signal.SIGINT)  # 1st Ctrl-C -> cancel THIS request
        inv.wait(timeout=10)
        assert inv.returncode == 130            # canceled -> EXIT_SIGINT
        assert "cancel" in inv.stderr.read().lower()

        # Server SURVIVED the request cancel -> a fresh request still works.
        assert serve.poll() is None
        rc = cli.main(["invoke", "ping", "{}", "--socket", str(sock)])
        assert rc == 0
        assert json.loads(capsys.readouterr().out.strip())["response"] == "pong"

        # Now Ctrl-C the SERVER itself -> clean teardown.
        serve.send_signal(signal.SIGINT)
        serve.wait(timeout=10)
        assert serve.returncode == 0
        assert not sock.exists()
    finally:
        for p in (serve,):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)
