"""``gen-worker repl`` — interactive single-endpoint session (issue #348).

Drives ``cli.main(["repl", ...])`` with a scripted stdin (one request per line,
then :quit), asserting the model loads ONCE (eager) and each request is
dispatched against the resident endpoint with the ergonomic field=value grammar.
"""

from __future__ import annotations

import io
import sys
import types

import msgspec
import pytest

import gen_worker.cli as cli
from gen_worker import RequestContext, inference


class _In(msgspec.Struct):
    prompt: str
    steps: int = 1


class _Out(msgspec.Struct):
    prompt: str
    steps: int
    setups: int


def _echo_module(name: str, counter: dict) -> None:
    mod = types.ModuleType(name)

    @inference()
    class Echo:
        def setup(self) -> None:
            counter["setups"] = counter.get("setups", 0) + 1

        @inference.function(name="echo")
        def echo(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(prompt=data.prompt, steps=data.steps, setups=counter["setups"])

    Echo.__module__ = name
    mod.Echo = Echo
    sys.modules[name] = mod


def _feed(monkeypatch, lines: str) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(lines))


def test_repl_loads_once_and_dispatches(monkeypatch, capsys) -> None:
    counter: dict = {}
    _echo_module("_repl_echo", counter)
    # Two requests (ergonomic args; multi-word primary quoted like a shell), then quit.
    _feed(monkeypatch, '"a cat" steps=5\n"a dog" steps=2\n:quit\n')

    rc = cli.main(["repl", "--module", "_repl_echo"])
    assert rc == 0

    out_lines = [l for l in capsys.readouterr().out.strip().splitlines() if l]
    results = [__import__("json").loads(l) for l in out_lines]
    assert results[0] == {"prompt": "a cat", "steps": 5, "setups": 1}
    assert results[1] == {"prompt": "a dog", "steps": 2, "setups": 1}  # setup ran ONCE


def test_repl_raw_json_and_unknown_field(monkeypatch, capsys) -> None:
    counter: dict = {}
    _echo_module("_repl_echo2", counter)
    _feed(monkeypatch, '{"prompt":"hi","steps":9}\nbogus=1 prompt=x\n:quit\n')

    rc = cli.main(["repl", "--module", "_repl_echo2"])
    assert rc == 0
    captured = capsys.readouterr()
    res = __import__("json").loads(captured.out.strip().splitlines()[0])
    assert res == {"prompt": "hi", "steps": 9, "setups": 1}
    # the unknown-field line is reported on stderr, not crashing the session
    assert "bogus" in captured.err


def test_repl_eof_exits_cleanly(monkeypatch, capsys) -> None:
    counter: dict = {}
    _echo_module("_repl_echo3", counter)
    _feed(monkeypatch, "")  # immediate EOF
    assert cli.main(["repl", "--module", "_repl_echo3"]) == 0
