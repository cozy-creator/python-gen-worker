"""Authoring surface + discovery — collapsed integration suite.

One test per distinct discovery/decorator behavior in the HARD floor:

  1. @invocable parity: discovered AND dispatched identically to
     @inference.function.
  2. parametrize=[Case(...)] stamps N routable functions with per-row
     Resources / model / input override.
  3. Optional shutdown: a class with NO shutdown registers, serves a real
     request, and tears down cleanly.
  4. msgspec.Meta bounds compile into the discovered endpoint.lock schema.
  5. Duplicate routable function name within an endpoint → discovery raises.
  6. Discovery walks submodules / class-shape endpoints across package layouts.

Every test exercises the REAL discovery walker / registration path on a real
Worker (built bare to skip the gRPC init the dispatch tables don't need).
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path
from typing import Annotated, Iterator

import msgspec
import pytest

from gen_worker import (
    Case,
    HFRepo,
    Repo,
    RequestContext,
    Resources,
    inference,
    invocable,
)
from gen_worker.discovery.discover import (
    _assert_unique_function_names,
    _extract_class_function_methods,
)
from gen_worker.discovery.walk import find_endpoint_classes


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    result: str


class BoundedInput(msgspec.Struct):
    steps: Annotated[int, msgspec.Meta(ge=1, le=50)] = 4
    name: Annotated[str, msgspec.Meta(min_length=1, max_length=64)] = "x"


# --------------------------------------------------------------------------- #
# 1. @invocable parity: discovered + dispatched like @inference.function       #
# --------------------------------------------------------------------------- #


def test_invocable_discovered_and_dispatched_like_inference_function() -> None:
    @inference()
    class ViaInvocable:
        def setup(self) -> None:
            pass

        @invocable(name="run")
        def run(self, ctx: RequestContext, payload: _In) -> _Out:
            return _Out(result="ok")

    @inference()
    class ViaFunction:
        def setup(self) -> None:
            pass

        @inference.function(name="run")
        def run(self, ctx: RequestContext, payload: _In) -> _Out:
            return _Out(result="ok")

    a = _extract_class_function_methods(ViaInvocable, "m")[0]
    b = _extract_class_function_methods(ViaFunction, "m")[0]
    # Same wire name + payload/output types, decorator-agnostic.
    assert a["name"] == b["name"] == "run"
    assert a["payload_type"] == b["payload_type"]
    assert a["output_type"] == b["output_type"]
    # @invocable attaches the SAME marker @inference.function uses.
    inv_spec = getattr(ViaInvocable.run, "__gen_worker_function_spec__")
    fn_spec = getattr(ViaFunction.run, "__gen_worker_function_spec__")
    assert type(inv_spec) is type(fn_spec) and inv_spec.name == fn_spec.name == "run"

    # Both produce a routable EndpointSpec via the one registry walker.
    from gen_worker.registry import extract_specs

    specs = extract_specs(ViaInvocable)
    assert [s.name for s in specs] == ["run"]
    assert specs[0].output_mode == "single" and specs[0].payload_type is _In


# --------------------------------------------------------------------------- #
# 2. parametrize= stamps N routable functions w/ per-row Resources/model/input #
# --------------------------------------------------------------------------- #


def test_parametrize_stamps_n_functions_with_per_row_overrides() -> None:
    class _AltIn(msgspec.Struct):
        prompt: str = ""
        guidance: float = 3.5

    @inference(
        models={"pipe": Repo("org/base#bf16")},
        resources=Resources(accelerator="cuda", min_vram_gb=24.0),
        parametrize=[
            Case(name="gen_bf16", resources=Resources(accelerator="cuda", min_vram_gb=24.0), model=Repo("org/base#bf16")),
            Case(name="gen_fp8", resources=Resources(accelerator="cuda", min_vram_gb=16.0), model=HFRepo("org/base-fp8"), input=_AltIn),
        ],
    )
    class FluxGrid:
        def setup(self, pipe) -> None:
            self.pipe = pipe

        @invocable
        def generate(self, ctx: RequestContext, payload: _In) -> _Out:
            return _Out(result="ok")

    fns = {f["name"]: f for f in _extract_class_function_methods(FluxGrid, "m")}
    assert set(fns) == {"gen-bf16", "gen-fp8"}
    # Per-row Resources.
    assert fns["gen-bf16"]["resources"]["min_vram_gb"] == 24.0
    assert fns["gen-fp8"]["resources"]["min_vram_gb"] == 16.0
    # All backed by the one shared method body.
    assert {f["python_name"] for f in fns.values()} == {"generate"}
    # Per-row model binding + per-row input struct override.
    assert "pipe" in fns["gen-bf16"]["bindings"] and "pipe" in fns["gen-fp8"]["bindings"]
    assert "guidance" in fns["gen-fp8"]["input_schema"]["$defs"]["_AltIn"]["properties"]

    # Build-time guards: exactly one body required; duplicate case names rejected.
    with pytest.raises(ValueError, match="exactly ONE"):
        @inference(parametrize=[Case(name="a")])
        class _TwoBodies:
            def setup(self) -> None:
                pass

            @invocable
            def one(self, ctx: RequestContext, payload: _In) -> _Out:
                return _Out(result="ok")

            @invocable
            def two(self, ctx: RequestContext, payload: _In) -> _Out:
                return _Out(result="ok")

    with pytest.raises(ValueError, match="duplicate parametrize"):
        @inference(parametrize=[Case(name="dup"), Case(name="dup")])
        class _Dup:
            def setup(self) -> None:
                pass

            @invocable
            def gen(self, ctx: RequestContext, payload: _In) -> _Out:
                return _Out(result="ok")


# --------------------------------------------------------------------------- #
# 3. Optional shutdown: register, serve a real request, drain cleanly          #
# --------------------------------------------------------------------------- #


def test_class_with_no_shutdown_registers_serves_and_tears_down() -> None:
    served: list = []

    @inference()
    class NoShutdown:
        def setup(self) -> None:
            self.ready = True

        @invocable
        def generate(self, ctx: RequestContext, payload: _In) -> _Out:
            served.append(payload.prompt)
            return _Out(result=f"echo:{payload.prompt}")

    assert not hasattr(NoShutdown, "shutdown")

    import asyncio

    from gen_worker.executor import Executor
    from gen_worker.registry import extract_specs

    async def _drive() -> None:
        sent: list = []

        async def _send(msg) -> None:
            sent.append(msg)

        ex = Executor(extract_specs(NoShutdown), _send)
        spec = ex.specs["generate"]
        inst = await ex.ensure_setup(spec)
        assert inst.ready is True
        out = inst.generate(RequestContext.__new__(RequestContext), _In(prompt="hi"))
        assert out.result == "echo:hi" and served == ["hi"]
        # Drain: missing shutdown() is a no-op, must not raise.
        await ex.shutdown_instances()

    asyncio.run(_drive())


# --------------------------------------------------------------------------- #
# 4. msgspec.Meta bounds compile into the discovered input schema              #
# --------------------------------------------------------------------------- #


def test_meta_bounds_compiled_into_discovered_input_schema() -> None:
    @inference()
    class Bounded:
        def setup(self) -> None:
            pass

        @invocable
        def generate(self, ctx: RequestContext, payload: BoundedInput) -> _Out:
            return _Out(result="ok")

    props = _extract_class_function_methods(Bounded, "m")[0]["input_schema"]["$defs"]["BoundedInput"]["properties"]
    assert props["steps"]["minimum"] == 1 and props["steps"]["maximum"] == 50
    assert props["name"]["minLength"] == 1 and props["name"]["maxLength"] == 64


# --------------------------------------------------------------------------- #
# 5. Duplicate routable function name within an endpoint → discovery raises     #
# --------------------------------------------------------------------------- #


def test_duplicate_function_names_raise_at_discovery() -> None:
    # Distinct names across two classes: fine.
    _assert_unique_function_names([
        {"name": "generate_sd15", "class_name": "SD15Base", "module": "m"},
        {"name": "generate_sdxl", "class_name": "SDXLBase", "module": "m"},
    ])
    # Two classes both exposing "generate" -> build-time error naming both sites.
    with pytest.raises(ValueError) as exc:
        _assert_unique_function_names([
            {"name": "generate", "class_name": "SD15Base", "module": "sd.main"},
            {"name": "generate", "class_name": "SDXLBase", "module": "sd.main"},
        ])
    msg = str(exc.value)
    assert "duplicate function name" in msg.lower()
    assert "generate" in msg and "SD15Base" in msg and "SDXLBase" in msg


# --------------------------------------------------------------------------- #
# 6. Discovery walks submodules / class-shape endpoints across package layouts  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_pkg(tmp_path, monkeypatch) -> Iterator[Path]:
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("ep_") or mod_name.startswith("other_pkg"):
            sys.modules.pop(mod_name, None)


def _endpoint_src(name: str, fn: str) -> str:
    return textwrap.dedent(f"""
        from gen_worker.api.decorators import inference

        @inference()
        class {name}:
            def setup(self) -> None:
                pass

            @inference.function(name="{fn}")
            def {fn}(self, ctx, payload) -> dict:
                return {{}}
    """)


def test_discovery_walks_layouts_dedups_and_skips_third_party(tmp_pkg: Path, caplog) -> None:
    # (a) re-exported submodule class is returned exactly once (the conversion
    # endpoint outage of 2026-05-16).
    reexport = tmp_pkg / "ep_reexport"
    reexport.mkdir()
    (reexport / "clone.py").write_text(_endpoint_src("CloneHF", "clone"))
    (reexport / "main.py").write_text("from .clone import CloneHF\n")
    (reexport / "__init__.py").write_text("from .main import *  # noqa: F403\n")

    # (b) un-reexported submodule class found via the pkgutil walk.
    walk = tmp_pkg / "ep_walk"
    walk.mkdir()
    (walk / "synth.py").write_text(_endpoint_src("SynthEp", "synth"))
    (walk / "__init__.py").write_text("")

    # (c) third-party class re-exported in is SKIPPED with a diagnostic.
    other = tmp_pkg / "other_pkg"
    other.mkdir()
    (other / "__init__.py").write_text(_endpoint_src("StrayClass", "stray"))
    stray = tmp_pkg / "ep_stray"
    stray.mkdir()
    (stray / "__init__.py").write_text("from other_pkg import StrayClass\n")

    with caplog.at_level(logging.INFO, logger="gen_worker.discovery.walk"):
        found = find_endpoint_classes(["ep_reexport", "ep_walk", "ep_stray"])

    qualnames = [f.qualname for f in found]
    assert "CloneHF" in qualnames
    assert qualnames.count("CloneHF") == 1  # re-export dedup
    assert "SynthEp" in qualnames           # submodule walk fallback
    assert "StrayClass" not in qualnames     # third-party leak rejected
    assert any(
        "Skipping endpoint class" in r.message and "outside the walked package" in r.message
        for r in caplog.records
    )
