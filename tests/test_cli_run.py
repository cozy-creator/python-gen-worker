"""``gen-worker run`` — collapsed integration suite (floor 14 + 18).

Drives the real ``cli.main(["run", ...])`` against in-memory endpoint modules:

  14a. real one-shot dispatch against a payload (cold load),
  14b. auto-attach to a live ``serve`` socket when one exists,
  18.  payload forms (inline / @file) + the exit-code matrix
       (success / usage / model-resolution / user exception / no-class).

cli.main() returns the exit code so we assert it directly; stdout/stderr via
capsys. No mocks of the unit under test — only ``monkeypatch`` to point the
warm-serve probe + the model-resolution leaf at test doubles.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Iterator

import msgspec
import pytest

import gen_worker.cli as cli
import gen_worker.cli.run as run_mod
from gen_worker import HFRepo, Repo, RequestContext, inference


class _In(msgspec.Struct):
    text: str = ""


class _Out(msgspec.Struct):
    response: str


class _Delta(msgspec.Struct):
    chunk: str


def _marco_module(name: str = "_test_marco") -> types.ModuleType:
    mod = types.ModuleType(name)

    @inference()
    class MarcoPolo:
        def setup(self) -> None:
            pass

        @inference.function(name="marco_polo")
        def marco_polo(self, ctx: RequestContext, data: _In) -> _Out:
            ctx.raise_if_canceled()
            return _Out(response="polo" if (data.text or "").strip().lower() == "marco" else "bro")

        def shutdown(self) -> None:
            pass

    MarcoPolo.__module__ = name
    mod.MarcoPolo = MarcoPolo
    sys.modules[name] = mod
    return mod


def _last_event(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# 14a. Real one-shot dispatch (cold load), inline + @file payloads             #
# --------------------------------------------------------------------------- #


def test_run_cold_dispatch_inline_and_file_payload(tmp_path, capsys, monkeypatch) -> None:
    _marco_module()
    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: None)  # force cold path

    # inline
    assert cli.main(["run", "--module", "_test_marco", "--payload", json.dumps({"text": "marco"})]) == 0
    last = _last_event(capsys)
    assert last["event"] == "result" and last["value"]["response"] == "polo"

    # @file
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"text": "nope"}))
    assert cli.main(["run", "--module", "_test_marco", "--payload-file", str(p)]) == 0
    assert _last_event(capsys)["value"]["response"] == "bro"


def test_run_streams_generator_yields(capsys, monkeypatch) -> None:
    name = "_test_stream"
    mod = types.ModuleType(name)

    @inference()
    class Streamer:
        def setup(self) -> None:
            pass

        @inference.function
        def stream(self, ctx: RequestContext, data: _In) -> Iterator[_Delta]:
            for word in (data.text or "").split():
                yield _Delta(chunk=word)

        def shutdown(self) -> None:
            pass

    Streamer.__module__ = name
    mod.Streamer = Streamer
    sys.modules[name] = mod
    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: None)

    assert cli.main(["run", "--module", name, "--payload", json.dumps({"text": "a b c"})]) == 0
    events = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    yields = [e for e in events if e["event"] == "yield"]
    assert [y["value"]["chunk"] for y in yields] == ["a", "b", "c"]
    assert [e for e in events if e["event"] == "result"][0]["value"]["yielded"] == 3


# --------------------------------------------------------------------------- #
# 14b. Auto-attach to a live serve socket                                      #
# --------------------------------------------------------------------------- #


def test_run_auto_attaches_to_warm_serve(monkeypatch, capsys, tmp_path) -> None:
    """When a warm serve socket exists, `run` dispatches through it (reusing the
    invoke client) instead of cold-loading."""
    _marco_module()
    import gen_worker.cli.invoke as invoke_mod

    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: tmp_path / ".gen-worker.sock")
    captured: dict = {}

    def _fake_send(sock_path, request):
        captured["request"] = request
        return {"ok": True, "events": [{"event": "result", "value": {"response": "polo"}}]}

    monkeypatch.setattr(invoke_mod, "_send_request", _fake_send)

    assert cli.main(["run", "--module", "_test_marco", "--payload", json.dumps({"text": "marco"})]) == 0
    # Routed through the warm server with the resolved function NAME + payload.
    assert captured["request"]["function"] == "marco_polo"
    assert captured["request"]["payload"] == {"text": "marco"}
    assert _last_event(capsys)["value"]["response"] == "polo"


# --------------------------------------------------------------------------- #
# 18. Exit-code matrix                                                         #
# --------------------------------------------------------------------------- #


def test_run_exit_code_matrix(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: None)

    # No subcommand -> usage (2).
    assert cli.main([]) == 2
    assert "gen-worker" in capsys.readouterr().err

    # Ambiguous class selection -> usage (2), lists candidates.
    name = "_test_two"
    mod = types.ModuleType(name)
    for cls_name in ("Alpha", "Beta"):
        @inference()
        class _C:
            def setup(self) -> None:
                pass

            @inference.function
            def run(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(response="x")

            def shutdown(self) -> None:
                pass

        _C.__name__ = cls_name
        _C.__qualname__ = cls_name
        _C.__module__ = name
        setattr(mod, cls_name, _C)
    sys.modules[name] = mod
    assert cli.main(["run", "--module", name, "--payload", json.dumps({"text": "x"})]) == 2
    assert "ambiguous" in capsys.readouterr().err

    # Payload validation failure -> usage (2).
    _marco_module()
    assert cli.main(["run", "--module", "_test_marco", "--payload", json.dumps({"text": 42})]) == 2
    assert "payload validation failed" in capsys.readouterr().err

    # Offline model-resolution miss -> exit 3.
    cozy_name = "_test_cozy"
    cmod = types.ModuleType(cozy_name)

    @inference(models={"pipe": Repo("test-org/test-repo").flavor("bf16").allow_override(_Out)})
    class CozyEndpoint:
        def setup(self, pipe=None) -> None:
            self.pipe = pipe

        @inference.function
        def run(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="cozy")

        def shutdown(self) -> None:
            pass

    CozyEndpoint.__module__ = cozy_name
    cmod.CozyEndpoint = CozyEndpoint
    sys.modules[cozy_name] = cmod
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path / "empty-cas"))
    assert cli.main(["run", "--module", cozy_name, "--offline", "--payload", json.dumps({"text": "x"})]) == 3
    assert "model resolution failed" in capsys.readouterr().err

    # User exception in the handler -> exit 1 with traceback.
    exc_name = "_test_exc"
    emod = types.ModuleType(exc_name)

    @inference()
    class Broken:
        def setup(self) -> None:
            pass

        @inference.function
        def run(self, ctx: RequestContext, data: _In) -> _Out:
            raise RuntimeError("boom")

        def shutdown(self) -> None:
            pass

    Broken.__module__ = exc_name
    emod.Broken = Broken
    sys.modules[exc_name] = emod
    assert cli.main(["run", "--module", exc_name, "--payload", json.dumps({"text": "x"})]) == 1
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "boom" in err
