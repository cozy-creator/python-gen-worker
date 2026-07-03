"""``gen-worker describe`` — machine-readable introspection (issue #349).

Drives the real ``cli.main(["describe", ...])`` against an in-memory endpoint
module and asserts the stable document shape: protocol_version, capabilities,
endpoint metadata, and per-function input schema + model bindings. No model is
loaded. Also checks ``serve --list-functions --json`` shares the same shape.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Iterator

import msgspec

import gen_worker.cli as cli
from gen_worker import HFRepo, Repo, RequestContext, inference
from gen_worker.cli.protocol import CAPABILITIES, PROTOCOL_VERSION


class _In(msgspec.Struct):
    prompt: str
    steps: int = 28


class _Out(msgspec.Struct):
    response: str


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    @inference(models={"pipe": HFRepo("black-forest-labs/FLUX.1-dev")})
    class Endpoint:
        def setup(self, pipe) -> None:  # never called by describe
            self.pipe = pipe

        @inference.function(name="generate")
        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response=data.prompt)

    Endpoint.__module__ = name
    mod.Endpoint = Endpoint
    sys.modules[name] = mod
    return mod


def _run(args) -> dict:
    rc = cli.main(args)
    assert rc == 0
    return rc


def test_describe_json_document_shape(capsys) -> None:
    _module("_desc_basic")
    assert cli.main(["describe", "--module", "_desc_basic"]) == 0
    doc = json.loads(capsys.readouterr().out.strip())

    assert doc["protocol_version"] == PROTOCOL_VERSION
    assert doc["capabilities"] == list(CAPABILITIES)
    assert "describe" in doc["capabilities"]
    assert doc["gen_worker_version"]  # non-empty string

    assert doc["endpoint"]["main_module"] == "_desc_basic"
    assert "Endpoint" in doc["endpoint"]["classes"]

    fns = {f["name"]: f for f in doc["functions"]}
    assert "generate" in fns
    g = fns["generate"]
    assert g["class"] == "Endpoint"
    assert g["method"] == "generate"
    assert g["is_generator"] is False
    # input schema carries the msgspec fields + the default
    props = g["input_schema"]["properties"]
    assert set(props) >= {"prompt", "steps"}
    # model binding introspected without loading
    assert g["models"]["pipe"]["type"] == "HFRepo"
    assert g["models"]["pipe"]["provider"] == "hf"
    assert g["models"]["pipe"]["ref"] == "black-forest-labs/FLUX.1-dev"


def test_describe_reports_generator_and_dispatch_binding(capsys) -> None:
    name = "_desc_gen"
    mod = types.ModuleType(name)

    @inference(models={"m": Repo("acme/flux")})
    class Streamer:
        def setup(self, m) -> None:
            self.m = m

        @inference.function(name="stream")
        def stream(self, ctx: RequestContext, data: _In) -> Iterator[_Out]:
            yield _Out(response="x")

    Streamer.__module__ = name
    mod.Streamer = Streamer
    sys.modules[name] = mod

    assert cli.main(["describe", "--module", name]) == 0
    doc = json.loads(capsys.readouterr().out.strip())
    fn = doc["functions"][0]
    assert fn["is_generator"] is True
    assert fn["models"]["m"]["type"] == "Repo"
    assert fn["models"]["m"]["provider"] == "tensorhub"


def test_serve_list_functions_json_matches_describe(capsys) -> None:
    _module("_desc_alias")
    # serve --list-functions --json is a thin alias of describe's functions array
    assert cli.main(["serve", "--module", "_desc_alias", "--list-functions", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out.strip())["functions"]

    assert cli.main(["describe", "--module", "_desc_alias"]) == 0
    described = json.loads(capsys.readouterr().out.strip())["functions"]

    assert listed == described
