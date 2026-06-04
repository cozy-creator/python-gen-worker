"""Streamed responses over the serve socket (issue #344).

A request with ``stream:true`` gets each event as its own NDJSON frame as
produced, terminated by a ``{"ok":true,"done":true}`` frame — instead of one
buffered ``{"ok":true,"events":[...]}`` envelope. Covered at the _Endpoint
level (on_event callback) and end-to-end via a real serve subprocess + the
``invoke --stream`` client.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Iterator

import msgspec
import pytest

import gen_worker.cli as cli
import gen_worker.cli.run as run_mod
import gen_worker.cli.serve as serve_mod
from gen_worker import RequestContext, inference


class _In(msgspec.Struct):
    n: int = 3


class _Delta(msgspec.Struct):
    i: int


def _counter_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    @inference()
    class Counter:
        def setup(self) -> None:
            pass

        @inference.function(name="count")
        def count(self, ctx: RequestContext, data: _In) -> Iterator[_Delta]:
            for i in range(data.n):
                yield _Delta(i=i)

    Counter.__module__ = name
    mod.Counter = Counter
    sys.modules[name] = mod
    return mod


def test_dispatch_on_event_streams_each_frame() -> None:
    mod = _counter_module("_stream_counter")
    ep = serve_mod._Endpoint(offline=False, allow_publish=False)
    ep.boot(run_mod.discover_candidates(mod))

    frames = []
    terminal = ep.dispatch("count", {"n": 3}, request_id="r1", on_event=frames.append)

    kinds = [f["event"] for f in frames]
    assert kinds == ["yield", "yield", "yield", "result"]
    assert [f["value"]["i"] for f in frames if f["event"] == "yield"] == [0, 1, 2]
    assert all(f["request_id"] == "r1" for f in frames)
    assert terminal == {"ok": True, "done": True}  # events streamed, not buffered
    ep.shutdown()


def test_dispatch_buffers_when_not_streaming() -> None:
    mod = _counter_module("_stream_counter_buf")
    ep = serve_mod._Endpoint(offline=False, allow_publish=False)
    ep.boot(run_mod.discover_candidates(mod))
    env = ep.dispatch("count", {"n": 2})  # no on_event
    assert env["ok"] is True
    assert [e["event"] for e in env["events"]] == ["yield", "yield", "result"]
    ep.shutdown()


_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "marco-polo"


@pytest.mark.skipif(
    not (_EXAMPLE_DIR / "endpoint.toml").exists(),
    reason="marco-polo example not present",
)
def test_invoke_stream_roundtrip(tmp_path, capsys) -> None:
    """End-to-end: a serve subprocess streams a generator endpoint's frames to
    the `invoke --stream` client, one JSON line per event."""
    name = "_stream_e2e_counter"
    pkg = tmp_path / "ep"
    (pkg / name).mkdir(parents=True)
    (pkg / name / "__init__.py").write_text("")
    (pkg / name / "main.py").write_text(
        "import msgspec\n"
        "from typing import Iterator\n"
        "from gen_worker import RequestContext, inference\n"
        "class In(msgspec.Struct):\n"
        "    n: int = 3\n"
        "class Delta(msgspec.Struct):\n"
        "    i: int\n"
        "@inference()\n"
        "class Counter:\n"
        "    def setup(self): pass\n"
        "    @inference.function(name='count')\n"
        "    def count(self, ctx: RequestContext, data: In) -> Iterator[Delta]:\n"
        "        for i in range(data.n):\n"
        "            yield Delta(i=i)\n"
    )
    (pkg / "endpoint.toml").write_text(
        f'schema_version = 1\nmain = "{name}.main"\n'
    )
    sock = tmp_path / "s.sock"
    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.cli", "serve", "--socket", str(sock), "--no-stdin"],
        cwd=str(pkg), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(pkg)},
    )
    try:
        deadline = time.time() + 15
        while not sock.exists() and time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(f"serve exited early rc={proc.returncode}: {proc.stderr.read()}")
            time.sleep(0.05)
        assert sock.exists()
        rc = cli.main(["invoke", "count", "n:=3", "--stream", "--socket", str(sock)])
        assert rc == 0
        out_lines = [l for l in capsys.readouterr().out.strip().splitlines() if l]
        # 3 streamed yields, then the generator result summary.
        vals = [json.loads(l) for l in out_lines]
        assert {"i": 0} in vals and {"i": 1} in vals and {"i": 2} in vals
    finally:
        proc.send_signal(__import__("signal").SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
