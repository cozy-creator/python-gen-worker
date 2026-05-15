"""Tests for ``gen-worker run`` (cli/run.py).

Covers:
  * class/method selection (1 / N / ambiguous)
  * payload validation errors
  * ``payload._models`` override (allow_override OK and reject path)
  * ``--offline`` cache-miss path (cozy + huggingface)
  * SIGINT cooperative cancellation
  * generator streaming
  * exit-code matrix (success, usage, model resolution, user exception)

The tests build fresh endpoint classes per case (mirroring the
``_make_serial_class`` factory in tests/test_serial_worker_dispatch.py) and
invoke ``cli.main()`` directly so the exit code can be asserted without
spawning a subprocess. Stdout / stderr are captured via the standard
pytest ``capsys`` fixture.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import types
from pathlib import Path
from typing import Iterator, List

import msgspec
import pytest

import gen_worker.cli as cli
import gen_worker.cli.run as run_mod
from gen_worker import HFRepo, Repo, RequestContext, inference


# --------------------------------------------------------------------------
# Fixture endpoints
# --------------------------------------------------------------------------

class _MarcoInput(msgspec.Struct):
    text: str = ""


class _MarcoOutput(msgspec.Struct):
    response: str


def _make_marco_module(module_name: str = "_test_marco_polo") -> types.ModuleType:
    """Create an in-memory module hosting a no-model @inference class."""
    mod = types.ModuleType(module_name)

    @inference()
    class MarcoPolo:
        def setup(self) -> None:
            pass

        @inference.function(name="marco_polo")
        def marco_polo(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            ctx.raise_if_canceled()
            if (data.text or "").strip().lower() == "marco":
                return _MarcoOutput(response="polo")
            return _MarcoOutput(response="bro")

        def shutdown(self) -> None:
            pass

    MarcoPolo.__module__ = module_name
    mod.MarcoPolo = MarcoPolo
    sys.modules[module_name] = mod
    return mod


def _make_two_class_module(module_name: str = "_test_two_class") -> types.ModuleType:
    """Module with two endpoint classes — exercises selection ambiguity."""
    mod = types.ModuleType(module_name)

    @inference()
    class Alpha:
        def setup(self) -> None: pass

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            return _MarcoOutput(response=f"alpha:{data.text}")

        def shutdown(self) -> None: pass

    @inference()
    class Beta:
        def setup(self) -> None: pass

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            return _MarcoOutput(response=f"beta:{data.text}")

        def shutdown(self) -> None: pass

    Alpha.__module__ = module_name
    Beta.__module__ = module_name
    mod.Alpha = Alpha
    mod.Beta = Beta
    sys.modules[module_name] = mod
    return mod


class _StreamDelta(msgspec.Struct):
    chunk: str


def _make_streaming_module(module_name: str = "_test_streaming") -> types.ModuleType:
    mod = types.ModuleType(module_name)

    @inference()
    class Streamer:
        def setup(self) -> None: pass

        @inference.function
        def stream(self, ctx: RequestContext, data: _MarcoInput) -> Iterator[_StreamDelta]:
            for word in (data.text or "").split():
                ctx.raise_if_canceled()
                yield _StreamDelta(chunk=word)

        def shutdown(self) -> None: pass

    Streamer.__module__ = module_name
    mod.Streamer = Streamer
    sys.modules[module_name] = mod
    return mod


def _make_hf_binding_module(module_name: str = "_test_hf_binding") -> types.ModuleType:
    """Endpoint with an HFRepo binding — exercises model resolution paths."""
    mod = types.ModuleType(module_name)
    qwen = HFRepo("Qwen/Qwen2.5-1.5B-Instruct")

    @inference(models={"pipe": qwen.allow_override(_MarcoOutput)})
    class HFEndpoint:
        def setup(self, pipe=None) -> None:
            # `pipe` is the resolved local path / loader-ready string.
            self.pipe = pipe

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            return _MarcoOutput(response=f"hf:{self.pipe}:{data.text}")

        def shutdown(self) -> None: pass

    HFEndpoint.__module__ = module_name
    mod.HFEndpoint = HFEndpoint
    sys.modules[module_name] = mod
    return mod


def _make_cozy_binding_module(module_name: str = "_test_cozy_binding") -> types.ModuleType:
    """Endpoint with a tensorhub Repo binding — for cozy-CAS-miss tests."""
    mod = types.ModuleType(module_name)
    repo = Repo("test-org/test-repo").flavor("bf16").allow_override(_MarcoOutput)

    @inference(models={"pipe": repo})
    class CozyEndpoint:
        def setup(self, pipe=None) -> None:
            self.pipe = pipe

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            return _MarcoOutput(response=f"cozy:{self.pipe}")

        def shutdown(self) -> None: pass

    CozyEndpoint.__module__ = module_name
    mod.CozyEndpoint = CozyEndpoint
    sys.modules[module_name] = mod
    return mod


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_single_class_single_method_inferred(capsys) -> None:
    _make_marco_module()
    rc = cli.main([
        "run",
        "--module", "_test_marco_polo",
        "--payload", json.dumps({"text": "marco"}),
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out, "expected at least one stdout line"
    last = json.loads(out[-1])
    assert last["event"] == "result"
    assert last["value"]["response"] == "polo"


def test_two_classes_requires_class_filter(capsys) -> None:
    _make_two_class_module()
    rc = cli.main([
        "run",
        "--module", "_test_two_class",
        "--payload", json.dumps({"text": "x"}),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "Alpha.run" in err
    assert "Beta.run" in err


def test_two_classes_with_class_filter(capsys) -> None:
    _make_two_class_module()
    rc = cli.main([
        "run",
        "--module", "_test_two_class",
        "--class", "Beta",
        "--payload", json.dumps({"text": "z"}),
    ])
    assert rc == 0
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["value"]["response"] == "beta:z"


def test_unknown_class_filter_lists_available(capsys) -> None:
    _make_two_class_module()
    rc = cli.main([
        "run",
        "--module", "_test_two_class",
        "--class", "Gamma",
        "--payload", json.dumps({"text": "x"}),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no @inference.function method matches" in err
    assert "Alpha.run" in err
    assert "Beta.run" in err


# --------------------------------------------------------------------------
# Payload validation
# --------------------------------------------------------------------------

def test_payload_validation_error_exits_2(capsys) -> None:
    _make_marco_module()
    rc = cli.main([
        "run",
        "--module", "_test_marco_polo",
        "--payload", json.dumps({"text": 42}),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "payload validation failed" in err


def test_payload_invalid_json_exits_2(capsys) -> None:
    _make_marco_module()
    rc = cli.main([
        "run",
        "--module", "_test_marco_polo",
        "--payload", "{not json",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not valid JSON" in err


def test_payload_file_mutex_with_inline(tmp_path, capsys) -> None:
    _make_marco_module()
    p = tmp_path / "p.json"
    p.write_text("{}")
    rc = cli.main([
        "run",
        "--module", "_test_marco_polo",
        "--payload", "{}",
        "--payload-file", str(p),
    ])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_payload_file_read(tmp_path, capsys) -> None:
    _make_marco_module()
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"text": "marco"}))
    rc = cli.main([
        "run",
        "--module", "_test_marco_polo",
        "--payload-file", str(p),
    ])
    assert rc == 0
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["value"]["response"] == "polo"


# --------------------------------------------------------------------------
# _models override
# --------------------------------------------------------------------------

def test_models_override_string_shorthand_parsed(monkeypatch) -> None:
    """The CLI normalizes the short-form ``owner/repo:tag#flavor`` string."""
    _make_hf_binding_module()

    # Intercept the resolution helper so we observe what override the CLI
    # constructed without actually fetching from HF.
    captured: dict = {}

    def _fake_resolve(*, ref, provider, offline, emit):
        captured["ref"] = ref
        captured["provider"] = provider
        return "/fake/local/path"

    monkeypatch.setattr(run_mod, "_resolve_local_path", _fake_resolve)

    rc = cli.main([
        "run",
        "--module", "_test_hf_binding",
        "--payload", json.dumps({
            "text": "x",
            "_models": {"pipe": "other/repo:canary#bf16"},
        }),
    ])
    assert rc == 0
    assert captured["ref"] == "other/repo:canary#bf16"
    assert captured["provider"] == "tensorhub"


def test_models_override_structured_parsed(monkeypatch) -> None:
    _make_hf_binding_module()
    captured: dict = {}

    def _fake_resolve(*, ref, provider, offline, emit):
        captured["ref"] = ref
        return "/fake/local/path"

    monkeypatch.setattr(run_mod, "_resolve_local_path", _fake_resolve)

    rc = cli.main([
        "run",
        "--module", "_test_hf_binding",
        "--payload", json.dumps({
            "text": "x",
            "_models": {"pipe": {"ref": "other/repo", "tag": "prod", "flavor": "fp8"}},
        }),
    ])
    assert rc == 0
    # tag==prod is omitted from the canonical string; flavor appended.
    assert captured["ref"] == "other/repo#fp8"


def test_models_override_rejected_when_binding_not_overridable(capsys) -> None:
    """Cozy endpoint without allow_override declared on the binding."""
    mod = types.ModuleType("_test_no_override")
    repo = Repo("test-org/no-override").flavor("bf16")  # no allow_override

    @inference(models={"pipe": repo})
    class NoOverride:
        def setup(self, pipe=None) -> None: pass

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            return _MarcoOutput(response="x")

        def shutdown(self) -> None: pass

    NoOverride.__module__ = "_test_no_override"
    mod.NoOverride = NoOverride
    sys.modules["_test_no_override"] = mod

    rc = cli.main([
        "run",
        "--module", "_test_no_override",
        "--payload", json.dumps({
            "text": "x",
            "_models": {"pipe": "other/repo"},
        }),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no .allow_override()" in err


# --------------------------------------------------------------------------
# --offline cache miss
# --------------------------------------------------------------------------

def test_offline_cozy_cache_miss_exits_3(tmp_path, capsys, monkeypatch) -> None:
    """Cozy ref without a local snapshot under TENSORHUB_CAS_DIR → exit 3."""
    _make_cozy_binding_module()
    # Point the CAS at an empty temp dir so the lookup definitely misses.
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path / "empty-cas"))

    rc = cli.main([
        "run",
        "--module", "_test_cozy_binding",
        "--offline",
        "--payload", json.dumps({"text": "x"}),
    ])
    assert rc == 3
    err = capsys.readouterr().err
    assert "model resolution failed" in err
    assert "--offline" in err


def test_offline_hf_cache_miss_exits_3(tmp_path, capsys, monkeypatch) -> None:
    """HF ref without a local snapshot under HF_HOME → exit 3 (offline)."""
    _make_hf_binding_module()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))

    rc = cli.main([
        "run",
        "--module", "_test_hf_binding",
        "--offline",
        "--payload", json.dumps({"text": "x"}),
    ])
    assert rc == 3
    err = capsys.readouterr().err
    assert "model resolution failed" in err


# --------------------------------------------------------------------------
# Generator streaming
# --------------------------------------------------------------------------

def test_generator_streams_yields_on_stdout(capsys) -> None:
    _make_streaming_module()
    rc = cli.main([
        "run",
        "--module", "_test_streaming",
        "--payload", json.dumps({"text": "alpha beta gamma"}),
    ])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in lines]
    yields = [e for e in events if e["event"] == "yield"]
    results = [e for e in events if e["event"] == "result"]
    assert len(yields) == 3
    assert [y["value"]["chunk"] for y in yields] == ["alpha", "beta", "gamma"]
    assert len(results) == 1
    assert results[0]["value"]["yielded"] == 3


# --------------------------------------------------------------------------
# SIGINT cooperative cancellation
# --------------------------------------------------------------------------

def test_sigint_first_press_trips_canceled(capsys) -> None:
    """First Ctrl-C trips ``ctx._canceled``; user code observes via
    ``ctx.is_canceled()`` and can raise CanceledError to exit 130.
    """
    mod = types.ModuleType("_test_sigint")

    @inference()
    class Slow:
        def setup(self) -> None: pass

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            # Fire SIGINT in another thread so the CLI's handler trips
            # ctx._canceled mid-call; then this loop notices and raises.
            def _kick() -> None:
                time.sleep(0.05)
                os.kill(os.getpid(), signal.SIGINT)
            t = threading.Thread(target=_kick)
            t.start()
            for _ in range(50):
                time.sleep(0.05)
                if ctx.is_canceled():
                    ctx.raise_if_canceled()
            t.join(timeout=1.0)
            return _MarcoOutput(response="should not get here")

        def shutdown(self) -> None: pass

    Slow.__module__ = "_test_sigint"
    mod.Slow = Slow
    sys.modules["_test_sigint"] = mod

    rc = cli.main([
        "run",
        "--module", "_test_sigint",
        "--payload", json.dumps({"text": "x"}),
    ])
    assert rc == 130
    err = capsys.readouterr().err
    assert "canceled" in err.lower() or "cancel" in err.lower()


# --------------------------------------------------------------------------
# User exception handling
# --------------------------------------------------------------------------

def test_user_exception_exits_1_with_traceback(capsys) -> None:
    mod = types.ModuleType("_test_user_exc")

    @inference()
    class Broken:
        def setup(self) -> None: pass

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            raise RuntimeError("boom")

        def shutdown(self) -> None: pass

    Broken.__module__ = "_test_user_exc"
    mod.Broken = Broken
    sys.modules["_test_user_exc"] = mod

    rc = cli.main([
        "run",
        "--module", "_test_user_exc",
        "--payload", json.dumps({"text": "x"}),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "RuntimeError" in err
    assert "boom" in err


# --------------------------------------------------------------------------
# ctx.emit lands on stderr as JSON line
# --------------------------------------------------------------------------

def test_ctx_emit_goes_to_stderr_as_json_line(capsys) -> None:
    mod = types.ModuleType("_test_emit")

    @inference()
    class Emitter:
        def setup(self) -> None: pass

        @inference.function
        def run(self, ctx: RequestContext, data: _MarcoInput) -> _MarcoOutput:
            ctx.emit("test.event", {"step": 5})
            return _MarcoOutput(response="done")

        def shutdown(self) -> None: pass

    Emitter.__module__ = "_test_emit"
    mod.Emitter = Emitter
    sys.modules["_test_emit"] = mod

    rc = cli.main([
        "run",
        "--module", "_test_emit",
        "--payload", json.dumps({"text": "x"}),
    ])
    assert rc == 0
    cap = capsys.readouterr()
    # Result on stdout.
    result = json.loads(cap.out.strip().splitlines()[-1])
    assert result["event"] == "result"
    assert result["value"]["response"] == "done"
    # Event on stderr (one of the lines is the emitted event).
    stderr_lines = [
        line for line in cap.err.strip().splitlines() if line.strip()
    ]
    parsed_events = []
    for line in stderr_lines:
        try:
            parsed_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert any(
        e.get("type") == "test.event" and e.get("payload", {}).get("step") == 5
        for e in parsed_events
    ), f"expected test.event on stderr, got: {parsed_events}"


# --------------------------------------------------------------------------
# --help and bare invocation
# --------------------------------------------------------------------------

def test_no_subcommand_prints_help_exits_2(capsys) -> None:
    rc = cli.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "gen-worker" in err
