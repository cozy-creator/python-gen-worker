"""Ergonomic `field=value` payload args (issue #350).

Unit-tests the schema-aware coercion in ``cli.args.build_payload`` and an
end-to-end ``cli.main(["run", ...])`` integration proving tokens reach the
handler coerced to the declared types.
"""

from __future__ import annotations

import json
import sys
import types
from typing import List

import msgspec
import pytest

import gen_worker.cli as cli
import gen_worker.cli.run as run_mod
from gen_worker import RequestContext, endpoint
from gen_worker.cli.args import ArgError, build_payload


class _P(msgspec.Struct):
    prompt: str
    steps: int = 28
    scale: float = 1.0
    hires: bool = False
    seed: int | None = None
    tags: List[str] = []


def test_build_payload_coerces_by_schema() -> None:
    out = build_payload(["a cat", "steps=5", "scale=2.5", "hires=true"], _P)
    assert out == {"prompt": "a cat", "steps": 5, "scale": 2.5, "hires": True}
    assert isinstance(out["steps"], int) and isinstance(out["scale"], float)


def test_build_payload_optional_and_rawjson() -> None:
    out = build_payload(["seed=7", 'tags:=["a","b"]', "prompt=x"], _P)
    assert out["seed"] == 7
    assert out["tags"] == ["a", "b"]


def test_build_payload_merges_over_base() -> None:
    out = build_payload(["steps=10"], _P, base={"prompt": "base", "steps": 1})
    assert out == {"prompt": "base", "steps": 10}


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("nope=1", id="unknown-field"),
        pytest.param("hires=maybe", id="bad-bool"),
        pytest.param("tags=a", id="list-needs-rawjson"),  # must use :=
    ],
)
def test_build_payload_errors(token: str) -> None:
    with pytest.raises(ArgError):
        build_payload([token], _P)


def test_build_payload_file(tmp_path) -> None:
    f = tmp_path / "p.txt"
    f.write_text("a long prompt from a file")
    out = build_payload([f"prompt@{f}"], _P)
    assert out["prompt"] == "a long prompt from a file"


# --------------------------------------------------------------------------
# run integration
# --------------------------------------------------------------------------

class _Echo(msgspec.Struct):
    prompt: str
    steps: int = 28
    hires: bool = False


class _EchoOut(msgspec.Struct):
    prompt: str
    steps: int
    hires: bool


def _echo_module(name: str = "_argmod") -> None:
    mod = types.ModuleType(name)

    @endpoint
    class Echo:
        def echo(self, ctx: RequestContext, data: _Echo) -> _EchoOut:
            return _EchoOut(prompt=data.prompt, steps=data.steps, hires=data.hires)

    Echo.__module__ = name
    mod.Echo = Echo
    sys.modules[name] = mod


def test_run_with_ergonomic_args(capsys, monkeypatch) -> None:
    _echo_module()
    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: None)  # cold path
    rc = cli.main(["run", "--module", "_argmod", "a cat", "steps=5", "hires=true"])
    assert rc == 0
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["event"] == "result"
    assert last["value"] == {"prompt": "a cat", "steps": 5, "hires": True}


def test_run_fields_merge_over_payload(capsys, monkeypatch) -> None:
    _echo_module()
    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: None)
    rc = cli.main([
        "run", "--module", "_argmod",
        "--payload", json.dumps({"prompt": "base", "steps": 1}),
        "steps=9",
    ])
    assert rc == 0
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["value"] == {"prompt": "base", "steps": 9, "hires": False}


def test_run_unknown_field_is_usage_error(capsys, monkeypatch) -> None:
    _echo_module()
    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: None)
    rc = cli.main(["run", "--module", "_argmod", "x", "bogus=1"])
    assert rc == run_mod.EXIT_USAGE


# --------------------------------------------------------------------------
# invoke payload resolution (blob vs ergonomic), no socket needed
# --------------------------------------------------------------------------

def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


def test_invoke_resolve_json_blob() -> None:
    from gen_worker.cli.invoke import _resolve_payload
    ns = _ns(args=['{"text":"hi"}'], function_name="f", config_path=None, module=None)
    assert _resolve_payload(ns) == {"text": "hi"}


def test_invoke_resolve_schemaless_ergonomic() -> None:
    # No importable module -> schema-less guessing: ':=' for typed, '=' string.
    from gen_worker.cli.invoke import _resolve_payload
    ns = _ns(args=["seed:=42", "name=bob"], function_name="f", config_path=None, module="_nope_missing")
    assert _resolve_payload(ns) == {"seed": 42, "name": "bob"}


def test_invoke_resolve_with_schema(monkeypatch) -> None:
    from gen_worker.cli import invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "_schema_for", lambda *a, **k: _Echo)
    ns = _ns(args=["a cat", "steps=5"], function_name="echo", config_path=None, module=None)
    assert invoke_mod._resolve_payload(ns) == {"prompt": "a cat", "steps": 5}
