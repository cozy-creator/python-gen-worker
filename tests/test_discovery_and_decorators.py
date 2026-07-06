"""@endpoint authoring surface + discovery — integration suite.

One test per distinct behavior:

  1. Plain-function endpoint: discovered + dispatched with no class/setup.
  2. Class endpoint: public methods are routable; helpers must be private.
  3. variants={name: (binding, Resources)} stamps N routable functions with
     per-variant binding/resources (+ payload 'variant' Literal validation).
  4. Optional shutdown/setup; async-generator = streaming.
  5. Resources(vram_gb=..) implies gpu.
  6. msgspec.Meta bounds compile into the discovered endpoint.lock schema.
  7. Duplicate routable names raise at collection.
  8. Discovery walks submodules across package layouts, dedups re-exports,
     and skips third-party leaks.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

import msgspec
import pytest

from gen_worker import HF, Hub, RequestContext, Resources, endpoint
from gen_worker.registry import collect_from_namespace, extract_specs


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    result: str


class _VarIn(msgspec.Struct):
    variant: str = "a"


class _VarIn2(msgspec.Struct):
    variant: Literal["a", "zz"] = "a"


# --------------------------------------------------------------------------- #
# 1. plain-function endpoint                                                    #
# --------------------------------------------------------------------------- #


def test_function_endpoint_discovered_and_shaped() -> None:
    @endpoint
    def hello(ctx: RequestContext, data: _In) -> _Out:
        return _Out(result=data.prompt)

    specs = extract_specs(hello)
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "hello"
    assert s.cls is None
    assert s.output_mode == "single"
    assert s.kind == "inference"


def test_function_endpoint_kwargs() -> None:
    @endpoint(kind="dataset", name="make-rows", resources=Resources(gpu=True))
    def build(ctx: RequestContext, data: _In) -> _Out:
        return _Out(result="x")

    (s,) = extract_specs(build)
    assert s.name == "make-rows"
    assert s.kind == "dataset"
    assert s.resources.gpu is True


def test_function_endpoint_rejects_bad_shapes() -> None:
    with pytest.raises(TypeError, match=r"\(ctx, payload\)"):
        @endpoint
        def nope(ctx: RequestContext) -> _Out:  # missing payload
            return _Out(result="")

    with pytest.raises(ValueError, match="runtime="):
        @endpoint(runtime="vllm")
        def nope2(ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")


# --------------------------------------------------------------------------- #
# 2. class endpoint: public methods route; setup optional                       #
# --------------------------------------------------------------------------- #


def test_class_methods_route_and_helpers_stay_private() -> None:
    @endpoint
    class Multi:
        def alpha(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result=self._suffix(data.prompt))

        def beta(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="b")

        def _suffix(self, s: str) -> str:  # private helper: not routable
            return s + "!"

    specs = extract_specs(Multi)
    assert sorted(s.name for s in specs) == ["alpha", "beta"]


def test_public_non_handler_method_is_rejected() -> None:
    with pytest.raises(TypeError, match="underscore"):
        @endpoint
        class Bad:
            def handler(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(result="")

            def helper(self):  # public non-handler
                pass


def test_model_slot_must_match_setup_param() -> None:
    with pytest.raises(ValueError, match="pipe"):
        @endpoint(models={"pipe": HF("o/r")})
        class Bad:
            def setup(self, other_name: str) -> None:
                pass

            def gen(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(result="")

    # model= shorthand: slot name comes from the setup parameter.
    @endpoint(model=HF("o/r", dtype="bf16"))
    class Good:
        def setup(self, pipe: str) -> None:
            self.pipe = pipe

        def gen(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (s,) = extract_specs(Good)
    assert list(s.models) == ["pipe"]
    assert s.models["pipe"].dtype == "bf16"


# --------------------------------------------------------------------------- #
# 3. variants fan-out                                                           #
# --------------------------------------------------------------------------- #


def test_variants_stamp_routable_functions_with_per_variant_overrides() -> None:
    @endpoint(
        model=HF("o/base", dtype="bf16"),
        resources=Resources(vram_gb=24),
        variants={
            "gen-fp8": (HF("o/base-fp8"), Resources(vram_gb=14)),
            "gen-nvfp4": HF("o/base-nvfp4"),  # bare binding: class resources
        },
    )
    class Gen:
        def setup(self, model: str) -> None:
            self.model = model

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    specs = {s.name: s for s in extract_specs(Gen)}
    assert sorted(specs) == ["gen-fp8", "gen-nvfp4", "generate"]
    assert specs["generate"].models["model"].ref == "o/base"
    assert specs["generate"].resources.vram_gb == 24
    assert specs["gen-fp8"].models["model"].ref == "o/base-fp8"
    assert specs["gen-fp8"].resources.vram_gb == 14
    assert specs["gen-nvfp4"].resources.vram_gb == 24
    # Variant specs must not share one instance (different weights).
    assert len({s.instance_key for s in specs.values()}) == 3


def test_variants_inherit_shared_aux_slots() -> None:
    @endpoint(
        models={"pipeline": HF("o/base", dtype="fp16"), "vae": HF("o/vae-fix", dtype="fp16")},
        variants={"gen-alt": HF("o/alt")},
    )
    class Gen:
        def setup(self, pipeline: str, vae: str) -> None: ...

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    specs = {s.name: s for s in extract_specs(Gen)}
    # The variant swaps only the primary slot; the aux vae slot is inherited.
    assert specs["gen-alt"].models["pipeline"].ref == "o/alt"
    assert specs["gen-alt"].models["vae"].ref == "o/vae-fix"
    assert specs["generate"].models["pipeline"].ref == "o/base"


def test_variants_without_class_model_are_the_full_routable_set() -> None:
    @endpoint(variants={
        "small": Hub("o/small"),
        "large": Hub("o/large"),
    })
    class Gen:
        def setup(self, model: str) -> None:
            self.model = model

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    assert sorted(s.name for s in extract_specs(Gen)) == ["large", "small"]


def test_variants_validation() -> None:
    with pytest.raises(ValueError, match="exactly\\s+ONE handler"):
        @endpoint(variants={"a": HF("o/r")})
        class TwoHandlers:
            def setup(self, model: str) -> None: ...
            def f(self, ctx: RequestContext, data: _In) -> _Out: ...
            def g(self, ctx: RequestContext, data: _In) -> _Out: ...

    with pytest.raises(ValueError, match="duplicate"):
        @endpoint(variants={" a": HF("o/r"), "a": HF("o/r2")})
        class DupNames:
            def setup(self, model: str) -> None: ...
            def gen(self, ctx: RequestContext, data: _In) -> _Out: ...

    # payload 'variant' field must be Literal-typed with declared members.
    with pytest.raises(ValueError, match="Literal"):
        @endpoint(variants={"a": HF("o/r")})
        class BadLiteral:
            def setup(self, model: str) -> None: ...
            def gen(self, ctx: RequestContext, data: _VarIn) -> _Out: ...

    with pytest.raises(ValueError, match="zz"):
        @endpoint(variants={"a": HF("o/r")})
        class BadMember:
            def setup(self, model: str) -> None: ...
            def gen(self, ctx: RequestContext, data: _VarIn2) -> _Out: ...


# --------------------------------------------------------------------------- #
# 4. optional setup/shutdown; async-generator = streaming                       #
# --------------------------------------------------------------------------- #


def test_streaming_detected_from_async_generator() -> None:
    @endpoint
    class Streamer:
        async def stream(self, ctx: RequestContext, data: _In) -> AsyncIterator[_Out]:
            yield _Out(result="x")

    (s,) = extract_specs(Streamer)
    assert s.output_mode == "stream"
    assert s.is_async_gen


def test_no_setup_no_shutdown_class_is_valid() -> None:
    @endpoint
    class Bare:
        def run(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="ok")

    (s,) = extract_specs(Bare)
    assert s.name == "run"
    assert not hasattr(Bare, "setup")


# --------------------------------------------------------------------------- #
# 5. Resources                                                                  #
# --------------------------------------------------------------------------- #


def test_resources_vram_implies_gpu() -> None:
    assert Resources().gpu is False
    assert Resources(vram_gb=12).gpu is True
    assert Resources(compute_capability=8.0).gpu is True
    assert Resources(gpu=True).vram_gb is None
    with pytest.raises(ValueError):
        Resources(vram_gb=0)
    with pytest.raises(ValueError):
        Resources(compute_capability=-1)


# --------------------------------------------------------------------------- #
# 6. msgspec.Meta bounds compile into the discovered schema                     #
# --------------------------------------------------------------------------- #


class BoundedInput(msgspec.Struct):
    steps: Annotated[int, msgspec.Meta(ge=1, le=50)] = 4
    name: Annotated[str, msgspec.Meta(min_length=1, max_length=64)] = "x"


def test_meta_bounds_compiled_into_discovered_input_schema() -> None:
    from gen_worker.discovery.discover import _extract_entries

    @endpoint
    class Bounded:
        def gen(self, ctx: RequestContext, data: BoundedInput) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Bounded, "testmod")
    schema = entry["input_schema"]
    defs = schema.get("$defs", {})
    props = (defs.get("BoundedInput") or {}).get("properties", {})
    assert props["steps"]["minimum"] == 1
    assert props["steps"]["maximum"] == 50
    assert props["name"]["minLength"] == 1
    assert props["name"]["maxLength"] == 64


# --------------------------------------------------------------------------- #
# 7. duplicate routable names                                                   #
# --------------------------------------------------------------------------- #


def test_duplicate_function_names_raise_at_collection() -> None:
    import types

    @endpoint
    class A:
        def gen(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="a")

    @endpoint
    class B:
        def gen(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="b")

    mod = types.ModuleType("dupes")
    mod.A, mod.B = A, B
    with pytest.raises(ValueError, match="duplicate"):
        collect_from_namespace(mod)


# --------------------------------------------------------------------------- #
# 8. package walking                                                            #
# --------------------------------------------------------------------------- #


def _endpoint_src(cls_name: str, fn_name: str) -> str:
    return textwrap.dedent(f"""
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            x: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class {cls_name}:
            def {fn_name}(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """)


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_discovery_walks_layouts_dedups_and_skips_third_party(tmp_pkg: Path, caplog) -> None:
    from gen_worker.discovery.walk import find_endpoints

    # (a) re-exported submodule class is returned exactly once.
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
        found = find_endpoints(["ep_reexport", "ep_walk", "ep_stray"])

    qualnames = [f.qualname for f in found]
    assert "CloneHF" in qualnames
    assert qualnames.count("CloneHF") == 1  # re-export dedup
    assert "SynthEp" in qualnames           # submodule walk fallback
    assert "StrayClass" not in qualnames     # third-party leak rejected
    assert any(
        "outside the walked package" in r.message for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# 9. producer kinds publish explicitly — generator handlers are rejected       #
# --------------------------------------------------------------------------- #


def test_producer_kind_rejects_sync_generator_class_handler() -> None:
    with pytest.raises(TypeError, match="publish_flavors"):
        @endpoint(kind="conversion")
        class BadConversion:
            def run(self, ctx: RequestContext, data: _In):
                yield _Out(result="x")


def test_producer_kind_rejects_async_generator_class_handler() -> None:
    with pytest.raises(TypeError, match="inference-only"):
        @endpoint(kind="training")
        class BadTraining:
            async def run(self, ctx: RequestContext, data: _In):
                yield _Out(result="x")


def test_producer_kind_rejects_generator_function() -> None:
    with pytest.raises(TypeError, match="must not be a generator"):
        @endpoint(kind="dataset")
        def bad_dataset(ctx: RequestContext, data: _In):
            yield _Out(result="x")


def test_from_scratch_example_uses_publish_contract() -> None:
    """Discovery smoke for examples/from-scratch: imports, one conversion-kind
    non-generator handler, result struct output (the explicit-publish shape)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "from-scratch" / "from_scratch.py"
    spec = importlib.util.spec_from_file_location("from_scratch_example", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    (s,) = extract_specs(mod.FromScratch)
    assert s.kind == "conversion"
    assert s.output_mode == "single"
    assert not s.is_async_gen
    assert s.output_type is mod.FromScratchResult


def test_manifest_keeps_variant_entries(tmp_pkg: Path) -> None:
    """variants= rows share (module, class, python_name) with their base
    function; the manifest dedup must key on the stamped name too."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_variants"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import HF, RequestContext, Resources, endpoint

        class In_(msgspec.Struct):
            x: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(
            model=HF("o/base", dtype="bf16"),
            resources=Resources(vram_gb=24),
            variants={"generate_fp8": (HF("o/base-fp8"), Resources(vram_gb=14))},
        )
        class Gen:
            def setup(self, model: str) -> None:
                self.model = model

            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_variants.main")
    assert sorted(f["name"] for f in fns) == ["generate", "generate-fp8"]


def test_model_shorthand_skips_server_handle_setup_param() -> None:
    """runtime= endpoints inject a ServerHandle into setup(); the model=
    shorthand must resolve the slot from the remaining parameter."""
    from gen_worker.runtimes.server import ServerHandle

    @endpoint(model=HF("o/llm"), resources=Resources(vram_gb=40), runtime="vllm")
    class Chat:
        def setup(self, model: str, server: ServerHandle) -> None:
            self.base_url = server.base_url

        def complete(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (s,) = extract_specs(Chat)
    assert list(s.models) == ["model"]
    assert s.models["model"].ref == "o/llm"
