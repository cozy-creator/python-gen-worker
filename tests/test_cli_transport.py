"""serve/invoke transport: Unix socket (default) + TCP (issue #347).

Unit-tests address parsing and an end-to-end TCP round-trip: a serve subprocess
listening on ``tcp://127.0.0.1:PORT`` answers an ``invoke --socket tcp://...``
client — the Docker / cross-process submission path.
"""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import gen_worker.cli as cli
from gen_worker.cli import transport

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "marco-polo"


def test_parse_addr_forms() -> None:
    assert transport.parse_addr("tcp://0.0.0.0:9000") == ("tcp", "0.0.0.0", 9000)
    assert transport.parse_addr("tcp://host:1") == ("tcp", "host", 1)
    assert transport.parse_addr("unix:///tmp/x.sock") == ("unix", "/tmp/x.sock")
    assert transport.parse_addr("./.gen-worker.sock") == ("unix", "./.gen-worker.sock")
    assert transport.is_unix("./x.sock") and not transport.is_unix("tcp://h:1")
    assert transport.display("tcp://h:2") == "tcp://h:2"


def test_parse_addr_bad_tcp() -> None:
    with pytest.raises(ValueError):
        transport.parse_addr("tcp://nohostport")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_tcp(port: int, proc: subprocess.Popen, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"serve exited early rc={proc.returncode}: {proc.stderr.read()}")
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            c.close()
            return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("serve never accepted on the TCP port")


@pytest.mark.skipif(
    not (_EXAMPLE_DIR / "pyproject.toml").exists(),
    reason="marco-polo example not present",
)
def test_serve_sidecar_written_and_removed(tmp_path) -> None:
    """serve writes a machine-readable .gen-worker.serve.json on ready (pid,
    listen, functions, protocol_version) and removes it on teardown (#349)."""
    sock = tmp_path / "sc.sock"
    sidecar = tmp_path / "sc.sock.json"
    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.cli", "serve", "--socket", str(sock), "--no-stdin"],
        cwd=str(_EXAMPLE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + 12
        while not sidecar.exists() and time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(f"serve exited rc={proc.returncode}: {proc.stderr.read()}")
            time.sleep(0.05)
        assert sidecar.exists()
        doc = json.loads(sidecar.read_text())
        assert doc["pid"] == proc.pid
        assert "protocol_version" in doc and "functions" in doc
        assert doc["listen"] == str(sock)
        assert "marco_polo" in doc["functions"]
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    assert not sidecar.exists()  # removed on teardown


@pytest.mark.skipif(
    not (_EXAMPLE_DIR / "pyproject.toml").exists(),
    reason="marco-polo example not present",
)
def test_tcp_roundtrip(capsys) -> None:
    port = _free_port()
    addr = f"tcp://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.cli", "serve", "--listen", addr, "--no-stdin"],
        cwd=str(_EXAMPLE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        _wait_tcp(port, proc)
        rc = cli.main(["invoke", "marco_polo", json.dumps({"text": "marco"}), "--socket", addr])
        assert rc == 0
        assert json.loads(capsys.readouterr().out.strip())["response"] == "polo"
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 0
